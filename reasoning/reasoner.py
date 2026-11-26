"""
reasoner.py - The reasoning component of the agent (Block 2 facade).

This is the only object the other blocks need:

    from reasoning import Reasoner
    r = Reasoner(env)                 # loads the rules, infers, follows the environment
    r.requirements()                  # -> input of Search (Block 4)
    r.why("RequiresICU(P1)")          # -> proof tree for the demo / report
    print(r.ask("RequiresICU(P1)"))   # -> resolution refutation trace
    r.derived("P1")                   # -> everything concluded about P1

Truth maintenance
-----------------
Derived facts depend on facts that can change (vitals, ML/DL evidence,
resource availability...). Forward chaining only ADDS facts, so on every
change we:
    1. retract every fact whose source is "derived"
    2. run forward chaining again from the base facts (env / ML / DL)
This "recompute from scratch" strategy is simple and always correct; the KB
holds a few hundred facts so it costs about a millisecond.

The Reasoner subscribes to Environment events (not to KB events): an
Environment event is emitted once per change, after the KB has been updated.
Some operations emit several events (update_vitals emits
'evidence_invalidated' before the new vitals are written, then
'vitals_updated'); we recompute on each, and the last one sees the final state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from kb import Fact, as_fact

from .forward import DERIVED, Justification, explain, forward_chain
from .resolution import ProofResult, prove
from .rules import Rule, describe, load_rules, stratify


@dataclass
class InferenceResult:
    """What changed among the derived facts during one inference run."""
    cause: str
    added: List[Fact] = field(default_factory=list)
    removed: List[Fact] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)

    def __str__(self) -> str:
        lines = [f"Inference after {self.cause}: +{len(self.added)} / -{len(self.removed)} derived facts"]
        lines += [f"  + {f}" for f in self.added]
        lines += [f"  - {f}" for f in self.removed]
        return "\n".join(lines)


class Reasoner:
    def __init__(self, env, rules_path: Optional[str] = None, auto: bool = True) -> None:
        from . import RULES_FILE  # local import: avoids a circular import at package load

        self.env = env
        self.kb = env.kb
        self.rules: List[Rule] = load_rules(rules_path or RULES_FILE)
        self.strata: List[List[Rule]] = stratify(self.rules)
        self.justifications: Dict[Fact, Justification] = {}
        self.last_trace: List[str] = []
        self.history: List[InferenceResult] = []   # only runs that changed something
        self._inferring = False
        self.auto = auto
        if auto:
            env.subscribe(self._on_event)
        self.run("initialisation")

    # ---- inference --------------------------------------------------------
    def derived_facts(self) -> List[Fact]:
        return [f for f in self.kb.facts() if self.kb.source_of(f) == DERIVED]

    def retract_derived(self) -> List[Fact]:
        removed = self.derived_facts()
        for f in removed:
            self.kb.retract(f)
        return removed

    def run(self, cause: str = "manual run") -> Optional[InferenceResult]:
        """Recompute all derived facts. Returns the diff with the previous run."""
        if self._inferring:          # re-entrant call (e.g. a listener reacting to our own changes)
            return None
        self._inferring = True
        try:
            before = set(self.retract_derived())
            trace: List[str] = []
            self.justifications = forward_chain(self.kb, self.strata, trace)
            self.last_trace = trace
            after = set(self.justifications)
            result = InferenceResult(cause, sorted(after - before, key=str),
                                     sorted(before - after, key=str))
            if result.changed:
                self.history.append(result)
            return result
        finally:
            self._inferring = False

    def _on_event(self, event) -> None:
        self.run(f"{event.kind}({event.subject})")

    # ---- queries ----------------------------------------------------------
    def holds(self, fact: Union[Fact, str]) -> bool:
        return self.kb.holds(fact)

    def derived(self, subject: Optional[str] = None) -> List[Fact]:
        """Derived facts, optionally only those about one patient/resource."""
        facts = self.derived_facts()
        return facts if subject is None else [f for f in facts if f.args and f.args[0] == subject]

    def why(self, fact: Union[Fact, str]) -> str:
        """Proof tree of a fact, down to env / ML / DL facts."""
        return explain(as_fact(fact), self.kb, self.justifications)

    def ask(self, query: Union[Fact, str], **kwargs) -> ProofResult:
        """Formal proof by resolution refutation (base facts + Horn rules).
        print(r.ask("RequiresICU(P1)")) shows the trace; bool(result) is the verdict;
        r.ask("RequiresICU(?x)").answers lists the bindings."""
        return prove(query, self.kb, self.rules, **kwargs)

    def justification(self, fact: Union[Fact, str]) -> Optional[Justification]:
        return self.justifications.get(as_fact(fact))

    def describe_rules(self) -> str:
        return describe(self.rules)

    # ---- contract with Search (Block 4) / Planning (Block 5) ----------------
    def requirements(self) -> Dict[str, dict]:
        """Per active patient, ordered by priority (1 = most urgent) then id:

        {"P1": {"priority": 1,
                "needs": [("Bed","ICU"), ("Doctor","Cardiology"), ("Equipment","ECG")],
                "urgent_evaluation": True},
         "P2": {"priority": 3, "needs": [("Room","Examination")], "urgent_evaluation": False}}

        (kind, subtype) pairs match env.free_resources(kind, subtype).
        Call again after every environment change: the content is recomputed.
        """
        out: Dict[str, dict] = {}
        for f in self.kb.facts("Priority"):
            pid, prio = f.args
            out[pid] = {"priority": prio, "needs": [], "urgent_evaluation": False}
        for f in self.kb.facts("RequiresResource"):
            pid, kind, subtype = f.args
            out.setdefault(pid, {"priority": None, "needs": [], "urgent_evaluation": False})
            out[pid]["needs"].append((kind, subtype))
        for f in self.kb.facts("RequiresUrgentEvaluation"):
            if f.args[0] in out:
                out[f.args[0]]["urgent_evaluation"] = True
        for req in out.values():
            req["needs"].sort()
        return dict(sorted(out.items(), key=lambda kv: (kv[1]["priority"] or 99, kv[0])))

    def describe_requirements(self) -> str:
        lines = ["Requirements (input of Search):"]
        for pid, req in self.requirements().items():
            needs = ", ".join(f"{k}:{s}" for k, s in req["needs"]) or "-"
            urgent = "  [urgent evaluation]" if req["urgent_evaluation"] else ""
            lines.append(f"  {pid}  priority {req['priority']}  needs {needs}{urgent}")
        return "\n".join(lines)
