"""demo.py - Block 1 demonstration: static scenario + dynamic changes."""
from environment import Environment
from interface import apply_event, load_events, load_scenario

env = Environment()
load_scenario(env, "scenario_example.json")

print("=" * 70, "\n1. ENVIRONMENT AFTER LOADING THE INPUT\n" + "=" * 70)
print(env.describe())

print("\n" + "=" * 70, "\n2. SYMBOLIC FACTS ABOUT P1 (what the agent KNOWS)\n" + "=" * 70)
for f in env.kb.about("P1"):
    print(" ", f)

print("\n3. QUERIES (pattern matching on the KB)")
for q in ["LowBloodPressure(?p)", "HasSymptom(?p,ChestPain)", "Available(?r)", "Type(?b,ICU)"]:
    print(f"  {q:28} ->", env.kb.query(q))

print("\n4. IMMUTABLE STATE for Search / Planning")
s0 = env.snapshot()
print(" ", s0)
s1 = s0.assign("P1", "ICU1").assign("P1", "D1")
print("  after assign(P1,ICU1), assign(P1,D1):\n ", s1)

print("\n5. ML / DL EVIDENCE BECOMES FACTS (Block 3 -> Block 2 bridge)")
print(" ", env.record_risk("P1", "ML", "High", 0.87))
print(" ", env.record_risk("P1", "DL", "Critical", 0.91))

print("\n6. DYNAMIC ENVIRONMENT: agent gets a plan, then the world changes")
env.assign("P1", "ICU1"); env.assign("P1", "D1")
print("  P1 holds:", sorted(env.patient("P1").assigned))
start = len(env.events)
env.subscribe(lambda e: print("   EVENT ->", e))
for ev in load_events("scenario_dynamic_events.json"):
    print(f"  apply {ev['type']}")
    apply_event(env, ev)

print("\n7. FINAL STATE")
print(env.describe())
print("  P1 facts:", [str(f) for f in env.kb.about("P1")])
print("  ICU1 facts:", [str(f) for f in env.kb.about("ICU1")])
print("  ", env.snapshot())
