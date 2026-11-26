"""
forward.py - Deduction by forward chaining (Block 2).

Generalised Modus Ponens applied to a fixpoint, one stratum at a time:

    for each stratum (rules.stratify):
        repeat
            for each rule  L1 & ... & Ln => H
                for each substitution theta that satisfies the body in the KB:
                    add  SUBST(theta, H)  to the KB with source="derived"
        until a full pass adds nothing

Body literals are satisfied left to right (rules.py already put them in the
right order):
    KB literal      -> UNIFY(literal, fact) with every stored fact of that predicate
    builtin         -> ground by now: evaluated (Less(87,90) -> True)
    not literal     -> ground by now: true iff the atom is NOT in the KB
                       (negation as failure; correct because all lower strata are complete)

Standardising apart is not needed here: KB facts are ground, so rule
variables can never clash with fact variables. (Resolution does need it.)

Every derived fact keeps a Justification (rule, premises, theta), from which
`explain` rebuilds the proof tree down to env / ML / DL facts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from kb import Fact, KnowledgeBase
from .rules import Literal, Rule, eval_builtin
from .unify import Theta, fmt_theta, substitute, unify

DERIVED = "derived"

# A premise is (kind, atom) with kind in {"fact", "builtin", "not"}
Premise = Tuple[str, Fact]


@dataclass(frozen=True)
class Justification:
    rule: Rule
    premises: Tuple[Premise, ...]
    theta: Tuple[Tuple[str, object], ...]   # frozen theta, for hashing / display

    @property
    def substitution(self) -> Theta:
        return dict(self.theta)

    def __str__(self) -> str:
        prem = ", ".join(_fmt_premise(p) for p in self.premises)
        return f"[{self.rule.id}] {prem}  {fmt_theta(self.substitution)}"


def _fmt_premise(p: Premise) -> str:
    kind, atom = p
    return {"fact": str(atom), "builtin": f"{atom}=true", "not": f"not {atom}"}[kind]


# --------------------------------------------------------------------------
# Body matching
# --------------------------------------------------------------------------
def match_body(body: List[Literal], kb: KnowledgeBase, theta: Optional[Theta] = None
               ) -> Iterator[Tuple[Theta, Tuple[Premise, ...]]]:
    """All (theta, premises) under which the conjunction `body` holds in kb."""
    yield from _match(body, 0, kb, dict(theta or {}), ())


def _match(body, i, kb, theta, used):
    if i == len(body):
        yield theta, used
        return
    lit = body[i]
    atom = substitute(lit.atom, theta)
    if lit.builtin:
        if eval_builtin(atom):
            yield from _match(body, i + 1, kb, theta, used + (("builtin", atom),))
    elif lit.negated:
        if not atom.is_ground:
            raise ValueError(f"'not {atom}' reached with unbound variables")
        if not kb.holds(atom):
            yield from _match(body, i + 1, kb, theta, used + (("not", atom),))
    else:
        for fact in kb.facts(atom.predicate):
            theta2 = unify(atom, fact, theta)
            if theta2 is not None:
                yield from _match(body, i + 1, kb, theta2, used + (("fact", fact),))


# --------------------------------------------------------------------------
# Forward chaining
# --------------------------------------------------------------------------
def forward_chain(kb: KnowledgeBase, strata: List[List[Rule]],
                  trace: Optional[List[str]] = None,
                  max_passes: int = 1000) -> Dict[Fact, Justification]:
    """Saturate kb with the consequences of the stratified rules.
    Returns {derived_fact: Justification} for the facts added by this call."""
    justifications: Dict[Fact, Justification] = {}
    for s, rules in enumerate(strata):
        if trace is not None:
            trace.append(f"--- stratum {s} ---")
        for n in range(1, max_passes + 1):
            added = 0
            for rule in rules:
                # materialise matches first: we add to the KB while iterating
                for theta, premises in list(match_body(rule.body, kb)):
                    fact = substitute(rule.head, theta)
                    if kb.add(fact, source=DERIVED):
                        added += 1
                        just = Justification(rule, premises, tuple(sorted(
                            (k, v) for k, v in theta.items())))
                        justifications[fact] = just
                        if trace is not None:
                            trace.append(f"  pass {n}: {just}  =>  {fact}")
            if not added:
                if trace is not None:
                    trace.append(f"  pass {n}: nothing new -> fixpoint")
                break
        else:
            raise RuntimeError(f"No fixpoint after {max_passes} passes in stratum {s}")
    return justifications


# --------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------
def explain(fact: Fact, kb: KnowledgeBase, justifications: Dict[Fact, Justification]) -> str:
    """Proof tree of `fact` down to its base facts (env / ML / DL)."""
    if not kb.holds(fact):
        return f"{fact} is not in the KB (not derivable from the current knowledge)"
    lines: List[str] = []
    _explain(fact, kb, justifications, "", "", lines, set())
    return "\n".join(lines)


def _explain(fact, kb, justs, prefix, child_prefix, lines, path):
    src = kb.source_of(fact)
    j = justs.get(fact)
    if j is None or fact in path:
        lines.append(f"{prefix}{fact}  [{src}]")
        return
    lines.append(f"{prefix}{fact}  [{j.rule.id}] {fmt_theta(j.substitution)}")
    for i, (kind, atom) in enumerate(j.premises):
        last = i == len(j.premises) - 1
        head = child_prefix + ("`- " if last else "|- ")
        nxt = child_prefix + ("   " if last else "|  ")
        if kind == "fact":
            _explain(atom, kb, justs, head, nxt, lines, path | {fact})
        elif kind == "builtin":
            lines.append(f"{head}{atom}  [builtin: true]")
        else:
            lines.append(f"{head}not {atom}  [negation as failure]")
