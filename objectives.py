"""
objectives.py - Objectives of the intelligent agent (Block 1).

Purely declarative: it states WHAT the agent wants. It does not evaluate
states or compute costs; Block 4 (Search) turns these objectives into a
cost function / heuristic and a goal test, and Block 5 (Planning) uses them
to judge plans.

kind:
  "hard" -> a constraint: a solution violating it is invalid (goal test)
  "soft" -> a preference: violations add a penalty (cost function)
weight: relative importance of a SOFT objective (ignored for hard ones).
        Placeholder values - Block 4 should tune/justify them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from kb import F, KnowledgeBase


@dataclass(frozen=True)
class Objective:
    id: str
    description: str
    kind: str            # "hard" or "soft"
    weight: float = 1.0
    used_by: str = ""    # which later block consumes it

    def __post_init__(self) -> None:
        if self.kind not in ("hard", "soft"):
            raise ValueError("kind must be 'hard' or 'soft'")


OBJECTIVES: Dict[str, Objective] = {o.id: o for o in [
    Objective("SatisfyRequirements",
              "Every patient receives the resources their derived requirements demand "
              "(e.g. a critical patient gets an ICU bed and a cardiologist when needed).",
              "hard", used_by="Block 4 goal test"),
    Objective("AvoidConflicts",
              "No resource is assigned to several patients at the same time.",
              "hard", used_by="Block 4 goal test / Block 5 preconditions"),
    Objective("PrioritizeCritical",
              "Critical and high-risk patients are served before less urgent ones; "
              "giving a scarce resource to a lower-priority patient is penalised.",
              "soft", weight=10.0, used_by="Block 4 cost / heuristic"),
    Objective("MinimizeWaitingTime",
              "Reduce the time patients spend waiting, weighted by urgency.",
              "soft", weight=3.0, used_by="Block 4 cost"),
    Objective("EfficientResourceUse",
              "Avoid wasting scarce resources, e.g. do not occupy an ICU bed "
              "with a patient who does not need it.",
              "soft", weight=2.0, used_by="Block 4 cost"),
]}


def get(objective_id: str) -> Objective:
    return OBJECTIVES[objective_id]


def hard_objectives() -> List[Objective]:
    return [o for o in OBJECTIVES.values() if o.kind == "hard"]


def soft_objectives() -> List[Objective]:
    return [o for o in OBJECTIVES.values() if o.kind == "soft"]


def register_in_kb(kb: KnowledgeBase) -> None:
    """Optionally expose the objectives as facts:
    Objective(Id), ObjectiveKind(Id,hard|soft), ObjectiveWeight(Id,w)."""
    for o in OBJECTIVES.values():
        kb.add(F("Objective", o.id), source="env")
        kb.add(F("ObjectiveKind", o.id, o.kind), source="env")
        if o.kind == "soft":
            kb.add(F("ObjectiveWeight", o.id, o.weight), source="env")


def describe() -> str:
    lines = ["Agent objectives"]
    for o in OBJECTIVES.values():
        w = f" (weight {o.weight})" if o.kind == "soft" else ""
        lines.append(f"  [{o.kind.upper():4}] {o.id}{w}: {o.description}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
