# Block 2 – Logical Reasoning, Deduction, Unification and Resolution

Pure Python, no dependencies. **Run everything from the repository root** (Block 1 modules live there).

```
python -m reasoning.demo_block2           # full demonstration (6 parts)
python -m unittest reasoning.test_block2  # Block 2 tests
python -m unittest discover               # all blocks
python cli.py scenario_example.json       # interactive: 'help' lists the reasoning commands
```

## Files
| File | Role |
|---|---|
| `unify.py` | Unification (Robinson / Russell & Norvig) with occurs check, substitution, composition, renaming, traces |
| `rules.py` | Rule language `ID: A(?x) & not B(?x) => C(?x)`, builtins (`Less`, `GreaterEq`...), safety check, stratification |
| `rules_hospital.txt` | The rule base (33 rules, editable without touching the code) |
| `forward.py` | Deduction: forward chaining to a fixpoint, stratum by stratum, with a justification for every derived fact; `explain()` proof trees |
| `resolution.py` | Resolution refutation prover (linear input / set of support), traces, answer extraction |
| `reasoner.py` | `Reasoner`: the facade used by the other blocks (truth maintenance, `why`, `ask`, `requirements`) |
| `cli_commands.py` | Reasoning commands for `cli.py` |
| `demo_block2.py` | Demonstration script |
| `test_block2.py` | Unit tests |

## Interface for the other blocks
```python
from reasoning import Reasoner

r = Reasoner(env)                 # loads + stratifies the rules, infers, follows every env event
r.requirements()                  # input of Search (Block 4), recomputed after every change
# {"P1": {"priority": 1, "urgent_evaluation": True,
#         "needs": [("Bed","ICU"), ("Doctor","Cardiology"), ("Equipment","ECG")]},
#  "P3": {"priority": 3, "urgent_evaluation": False, "needs": [("Room","Examination")]}}
r.holds("RequiresICU(P1)")        # is a conclusion currently true?
r.derived("P1")                   # everything concluded about P1
r.why("RequiresICU(P1)")          # proof tree (string)
print(r.ask("RequiresICU(P1)"))   # resolution proof trace; bool(...) = verdict; .answers for ?vars
r.history                         # what each environment change added / removed
```
- `(kind, subtype)` pairs are exactly the arguments of `env.free_resources(kind, subtype)`.
- Priority: 1 = critical, 2 = high risk, 3 = other active patients. Priority-3 needs may stay unmet (the patient waits).
- Discharged patients get no requirements (rules only apply to `Active` patients).
- Block 3 inserts predictions with `env.record_risk(pid, "ML"|"DL", level, prob)`; the reasoner reacts automatically.
- `Reasoner(env, auto=False)` disables automatic inference; then call `r.run()` yourself.

## Rule base (4 layers)
Thresholds are educational (simulation only, not medical advice).

1. **Clinical signs** from numeric vitals with builtins: `Hypoxemia` (SpO2 < 90), `SevereHypoxemia` (≤ 85),
   `Tachycardia` (HR > 100), `Fever` (T ≥ 38), `Tachypnea` (RR ≥ 22), `Elderly` (age ≥ 65), `Active`.
2. **Risk** from explicit knowledge *and* learned evidence:
   R1 `HasSymptom(?x,ChestPain) & LowBloodPressure(?x) => HighRisk(?x)` (PDF), R2 hypoxemia + dyspnea,
   R3 sepsis-like pattern, R4 shock pattern `=> CriticalRisk`, R5–R8 ML/DL evidence with probability thresholds,
   e.g. R7 `DL_Risk(?x,Critical,?p) & GreaterEq(?p,0.8) => CriticalRisk(?x)` (PDF), R9 `CriticalRisk => HighRisk`.
3. **Requirements**: urgent evaluation, ICU (PDF: `CriticalRisk => RequiresICU`), specialist, ECG,
   emergency bed (`not CriticalRisk`), exam room (`not HighRisk`), `CandidateDoctor` (2-variable unification).
4. **Bridge to Search**: generic `RequiresResource(P,Kind,Subtype)` and `Priority(P,n)`.

Design note: the PDF writes `ChestPain(x)`; Block 1 produces `HasSymptom(x,ChestPain)`. We keep Block 1's
convention, so the constant sits inside an argument and is matched by unification.

## Techniques and design decisions (for the report)

**Unification** – two-way (variables on both sides), chained bindings, occurs check
(`?x` vs `f(?x)` fails; compound terms exist only to demonstrate it, hospital facts are flat).

**Deduction (forward chaining)** – Generalised Modus Ponens until a fixpoint. Body literals are matched left to
right with `unify`; the parser reorders bodies so builtins and `not` literals come after the literals that bind
their variables. Safety (range restriction) is checked when a rule is loaded. Every derived fact stores its
justification (rule, premises, substitution) → `why()`.

**Negation as failure and stratification** – `not P` means "P cannot be derived". This is only sound if every
`P` fact has been derived *before* the rule using `not P` runs. `stratify()` builds the predicate dependency
graph and assigns strata (a negative edge raises the stratum); a cycle through `not` is rejected. Note that
stratification is per *predicate*: Q3 has no `not` but shares the head `RequiresSpecialist` with Q4, so both are
in stratum 1. Example of a rule we had to rewrite: `HighRisk & not RequiresSpecialist(?x,Cardiology) =>
RequiresSpecialist(?x,General)` is not stratifiable (the predicate depends negatively on itself), so Q4 negates
the base fact `HasSymptom(?x,ChestPain)` instead.

**Truth maintenance** – forward chaining only adds facts, but the world changes (new vitals delete old ML/DL
predictions, resources become busy...). On every environment event the reasoner retracts all `derived` facts
and recomputes from the base facts (env / ML / DL). Simple, always correct, about a millisecond here.

**Resolution** – refutation: add `{~Q}` and derive the empty clause.
- Input clauses: base facts only (never derived facts, otherwise the proof would be trivial) and the 28 Horn
  rules. The 5 rules with `not` are excluded: negation as failure is not classical negation, so they have no
  sound clausal form; their conclusions come from stratified forward chaining.
- Strategy: linear input resolution with set of support (SLD) – always resolve the newest clause (a
  descendant of `~Q`) with a fact or rule, selecting the leftmost negative literal. Complete for Horn clauses.
- Rules are standardized apart at each use (`?x_7`); duplicate literals are merged (factoring).
- Ground builtins are evaluated: true → literal removed, false → branch closed.
- Answer extraction: `RequiresICU(?x)` uses the goal `{~RequiresICU(?x) | Answer(?x)}`.
- A test checks that resolution and forward chaining agree on every Horn predicate.

**ML/DL integration** – predictions are facts (`ML_Risk(P1,High,0.87)`, `DL_Risk(P1,Critical,0.91)`) used by
rules R5–R8 with a confidence threshold. In the scenario the rules alone give `HighRisk(P1)`; the DL prediction
is what makes P1 critical and leads to `RequiresICU(P1)`.

**Limits** – fixed thresholds; probabilities are only used as cut-offs (no probabilistic reasoning, no
combination of contradictory ML and DL evidence); full recomputation does not scale to very large KBs
(incremental truth maintenance would); resolution covers only the Horn part.
