"""
demo_block2.py - Block 2 demonstration (run from the repository root):

    python -m reasoning.demo_block2

1. Knowledge Base + rules       what the agent knows and the rules it uses
2. Unification                  matching expressions that contain variables
3. Deduction                    forward chaining trace + proof tree (why)
4. ML / DL evidence             learned predictions change the conclusions
5. Resolution                   refutation proofs: success, failure, answer extraction
6. Dynamic environment          new events -> conclusions and requirements recomputed
"""
import os

from environment import Environment
from interface import apply_event, load_events, load_scenario
from kb import Fact

from reasoning import Reasoner
from reasoning.unify import Compound, unify_with_trace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def title(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def show_changes(r: Reasoner, since: int, no_change_note: str) -> None:
    """Print the inference diffs recorded after position `since` of r.history."""
    new = r.history[since:]
    for res in new:
        print(res)
    if not new:
        print(f"  no derived fact changed: {no_change_note}")


def event_label(ev: dict) -> str:
    who = ev.get("patient")
    who = who.get("id") if isinstance(who, dict) else who
    who = who or ev.get("resource") or ev.get("time", "")
    extra = {k: v for k, v in ev.items() if k not in ("type", "patient", "resource", "vitals", "time")}
    return f"{ev['type']}({who})" + (f" {extra}" if extra else "")


def main() -> None:
    env = Environment()
    load_scenario(env, os.path.join(ROOT, "scenario_example.json"))
    r = Reasoner(env)          # loads + stratifies the rules, infers, then follows env events

    # ------------------------------------------------------------------
    title("1. KNOWLEDGE BASE AND RULES")
    print("Base facts about P1 (from the environment, Block 1):")
    for f in env.kb.about("P1"):
        if env.kb.source_of(f) != "derived":
            print(f"  {f}    [{env.kb.source_of(f)}]")
    print("\nRule base (rules_hospital.txt), grouped by stratum:")
    print(r.describe_rules())

    # ------------------------------------------------------------------
    title("2. UNIFICATION")
    examples = [
        (Fact.parse("RequiresSpecialist(?x,Cardiology)"), Fact.parse("RequiresSpecialist(P1,Cardiology)")),
        (Fact.parse("Specialty(?d,?s)"), Fact.parse("Specialty(D1,Cardiology)")),
        (Fact.parse("Assigned(?p,?p)"), Fact.parse("Assigned(P1,D1)")),
        (Fact.parse("DL_Risk(?x,Critical,?p)"), Fact.parse("DL_Risk(P1,High,0.87)")),
        ("?x", Compound("f", ("?x",))),
    ]
    for a, b in examples:
        print("\n".join(unify_with_trace(a, b)[1]))
        print()

    # ------------------------------------------------------------------
    title("3. DEDUCTION (forward chaining, before any ML/DL evidence)")
    print("\n".join(r.last_trace))
    print("\nWhy does P1 need a cardiologist?")
    print(r.why("RequiresResource(P1,Doctor,Cardiology)"))
    print("\nWhy does P3 only need the examination room? (negation as failure)")
    print(r.why("Priority(P3,3)"))
    print()
    print(r.describe_requirements())
    print(f"\nCriticalRisk(P1)? {r.holds('CriticalRisk(P1)')}  -> rules alone say High, not Critical")

    # ------------------------------------------------------------------
    title("4. ML / DL EVIDENCE ENTERS THE KNOWLEDGE BASE (Block 3 -> Block 2)")
    notes = {"ML": "P1 was already HighRisk by rule R1; the ML evidence corroborates it (R5)",
             "DL": ""}
    for src, level, p in [("ML", "High", 0.87), ("DL", "Critical", 0.91)]:
        n = len(r.history)
        fact = env.record_risk("P1", src, level, p)
        print(f"Inserted {fact}  [{src}]")
        show_changes(r, n, notes[src])
        print()
    print("Why is P1 now in the ICU list?")
    print(r.why("RequiresICU(P1)"))
    print()
    print(r.describe_requirements())

    # ------------------------------------------------------------------
    title("5. RESOLUTION (refutation: KB + ~Query  |-  empty clause)")
    print(r.ask("RequiresICU(P1)").format())
    print("\n" + "-" * 78)
    print(r.ask("RequiresICU(P3)").format())
    print("\n" + "-" * 78)
    print(r.ask("RequiresICU(?x)").format(show_search=False))
    print("\n" + "-" * 78)
    print(r.ask("CandidateDoctor(P1,?d)").format(show_search=False))

    # ------------------------------------------------------------------
    title("6. DYNAMIC ENVIRONMENT: the knowledge is revised after each change")
    for ev in load_events(os.path.join(ROOT, "scenario_dynamic_events.json")):
        n = len(r.history)
        apply_event(env, ev)
        print(f"\n>>> event: {event_label(ev)}")
        if ev["type"] == "set_available":
            note = ("patient needs do not depend on bed availability; "
                    "Search (Block 4) will see that ICU1 is no longer free")
        else:
            note = "nothing to conclude from this event"
        show_changes(r, n, note)
    print()
    print(r.describe_requirements())
    print("\nP1 and P4 both need ICU1 (which is now down), D1 and ECG1:")
    print("this conflict is the input of the Search component (Block 4).")
    print("\nWhy is P4 critical without any DL prediction?")
    print(r.why("CriticalRisk(P4)"))


if __name__ == "__main__":
    main()
