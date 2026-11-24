# Block 1 – Agent, Environment and Knowledge Representation

Pure Python 3.9+, no dependencies.

```
python3 demo.py                 # full demonstration (static + dynamic)
python3 -m unittest test_block1 # 13 tests
python3 cli.py scenario_example.json   # interactive interface ('help')
```

## Files
| File | Role |
|---|---|
| `kb.py` | Knowledge Base: ground facts `Pred(a,b)`, `?var` patterns, sources, change listeners |
| `model.py` | Entities: `Patient`, `VitalSigns` (time series), `Resource` (Doctor/Nurse/Bed/Equipment/Room) |
| `environment.py` | World model, auto-sync to KB, immutable `State`, `Event` stream, ML/DL evidence bridge |
| `interface.py` | Validated JSON input (scenario + events), export, event application |
| `cli.py` | Interactive command line / guided patient form |
| `scenario_example.json` | The 3-patient example of the project statement |
| `scenario_dynamic_events.json` | New critical patient, ICU failure, deterioration |

## Predicates produced (the contract for Blocks 2–6)
Patient: `Patient(P)`, `Age(P,n)`, `HeartRate`, `SystolicBP`, `DiastolicBP`, `OxygenSaturation`,
`Temperature`, `RespiratoryRate`, `LowBloodPressure(P)` / `NormalBloodPressure(P)` / `HighBloodPressure(P)`,
`HasSymptom(P,Symptom)`, `PatientWaiting(P)` / `PatientUnderTreatment(P)` / `PatientDischarged(P)`,
`DoctorAssigned(P,R)`, `BedAssigned(P,R)`, `NurseAssigned`, `RoomAssigned`, `EquipmentAssigned`.

Resource: `Doctor(R)`, `Nurse(R)`, `Bed(R)`, `Room(R)`, `Equipment(R)`, `Specialty(R,x)` (doctor/nurse),
`Type(R,x)` (bed/room/equipment), `Available(R)` / `NotAvailable(R)`.

Learned evidence: `ML_Risk(P,Level,Prob)`, `DL_Risk(P,Level,Prob)`.
Facts carry a source tag: `env`, `ML`, `DL` (Block 2 should use `derived`).

## How each block plugs in
- **Block 2 (reasoning): read/write `env.kb` (`add(fact, source="derived")`, `query`, `holds`, `retract_where`);
  `kb.subscribe(cb)` for incremental inference. `kb.py::match` is one-way matching; implement full unification there.
- **Block 3 (ML/DL)**: `patient.feature_dict()` = current structured features; `patient.history` = list of
  `VitalSigns` (`.as_row()` -> `[HR,SBP,DBP,SpO2,Temp,RR]`) for LSTM/GRU. Output goes back with
  `env.record_risk(pid, "ML"|"DL", level, prob)`.
- **Blocks 4–5 (search/planning)**: `env.snapshot()` returns a hashable `State`
  (`assign`, `release`, `start_treatment` return new states). Use `env.free_resources(kind, subtype)` for actions.
  Apply the final plan to the real world with `env.assign / start_treatment / release`.
- **Block 6 (agent loop)**: `env.subscribe(cb)` receives `Event`s. Replan on `assignment_broken`,
  `resource_unavailable`, `patient_arrived`, `vitals_updated`, `evidence_invalidated`.
  New vitals/symptoms automatically delete stale ML/DL facts, so the agent must re-run the models.

## Design notes (for the report)
- State = patients (waiting / under treatment), resource availability, patient-resource assignments.
- A resource is `Available` only if usable AND not assigned (closed-world: otherwise `NotAvailable`).
- Free-text symptoms are normalised to symbols (`"chest pain"` -> `ChestPain`).
- Numeric systolic BP is categorised (<90 Low, >=140 High) unless an explicit category is given.
- Simulation only: no real medical diagnosis.



## File-by-file explanation

The files form layers: model.py and kb.py are the foundation, environment.py builds on both, and interface.py and cli.py feed input into it.

```
JSON / CLI  ->  interface.py  ->  model.py objects  ->  environment.py  ->  kb.py facts
                                                              |
                                                     State snapshot + Events (for Blocks 4-6)
```


**1. kb.py – the Knowledge Base**

This is the agent's symbolic memory. It stores facts like HeartRate(P1,125) and lets you query them.


- is_var(term):	True if a term is a variable (a string starting with ?, such as ?x).
- _coerce(token):	Turns "125" into the int 125, "0.91" into a float, and leaves anything else as text.
- Fact (class):	An immutable, hashable atom: a predicate plus a tuple of arguments.
- Fact.parse("Age(P1,72)"):	Converts text into a Fact.
- Fact.arity, Fact.is_ground:	The number of arguments, and whether the fact has no variables.
- F("Age","P1",72):	Shorthand constructor for a Fact.
- as_fact(x):	Accepts either a Fact or a string and always returns a Fact.
- match(pattern, fact):	One-way matching. HasSymptom(?p,ChestPain) against HasSymptom(P1,ChestPain) returns {"?p":"P1"}, or None if they don't match. Block 2 will extend this to full unification.

KnowledgeBase methods:

- add(fact, source): stores a ground fact together with its source (env, ML, DL or derived). It returns False if the - fact already existed and rejects facts containing variables.
- retract(fact): removes one fact.
- retract_where(pattern, source): removes every fact matching a pattern, optionally only those from one source.
- retract_subject(subject, source): removes all facts whose first argument is that subject. The environment uses - - - this to refresh a patient or resource.
- holds(fact): checks whether a fact is true, and in works the same way.
- query(pattern): returns all variable bindings that match.
- facts(predicate): lists facts, either all of them or one predicate.
- about(subject): lists everything known about one patient or resource.
- source_of(fact): says where a fact came from, which is useful for Block 2's reasoning traces.
- subscribe(callback): and _notify call your function on every add or retract, so Block 2 can do incremental inference.
- dump(): returns a printable listing with sources.
- version: is a counter that goes up on every real change.
**2. model.py – the entities**

Plain data classes with validation and no reasoning logic.

- Constants: RESOURCE_KINDS, PATIENT_STATUSES and BP_LEVELS list the allowed values. LOW_SYSTOLIC=90 and HIGH_SYSTOLIC=140 are the thresholds used to categorise blood pressure.
- camel(text): turns "chest pain" into ChestPain, so free text becomes a valid symbol.
- _check(name, value, lo, hi): raises an error if a value is implausible, such as SpO2 = 150.

- VitalSigns: holds one measurement: time, heart rate, systolic and diastolic pressure, SpO2, temperature and respiratory rate. It can also carry an explicit bp_category.

- __post_init__: validates ranges.
- bp_level: returns Low, Normal or High. An explicit category wins, otherwise it is derived from systolic pressure.
- as_dict(): is used when exporting.
- as_row(): returns [HR, SBP, DBP, SpO2, Temp, RR], ready to build feature matrices for the Block 3 LSTM/GRU.

- Patient: holds id, age, symptoms, a chronological history of VitalSigns, a status (Waiting, UnderTreatment or Discharged), arrival time and the set of resources the patient holds.

- current: returns the latest measurement.
- feature_dict(): returns the flat structured features for the Block 3 ML models.

- Resource: is any limited resource: Doctor, Nurse, Bed, Equipment or Room. subtype is the specialty, bed type or equipment type. available means it is physically usable, and assigned_to records which patient currently holds it. - is_free: is true only if the resource is usable and unassigned.

**3. environment.py – the world model**

This is the central file. It keeps the entities, the KB, the state snapshot and the change events consistent with each other.

- Event: is a record of one change. Its kinds are patient_arrived, vitals_updated, assignment_broken, evidence_invalidated and others.

- State: is an immutable, hashable snapshot for Search and Planning.

- is_free: and assigned_to are lookups.
- assign(pid, rid), release(pid, rid) and start_treatment(pid): each return a new state and raise an error if the move is illegal. They are the transition model for Blocks 4 and 5.

Environment methods:

- Events and time: subscribe(cb) registers a listener, _emit creates and broadcasts an event, and advance_time moves the simulation clock.
- Lookups: patient, resource, patients_with_status, and free_resources(kind, subtype) (for example, free ICU beds).
- Patients: add_patient, update_vitals, add_symptom, remove_symptom, start_treatment and discharge.
- Resources: add_resource, and set_resource_available(rid, bool). If you disable a resource that is assigned, the assignment is broken and assignment_broken is emitted, which is how Block 6 detects that a plan is obsolete.
- Assignments: assign(pid, rid) and release(pid, rid). The internal _release does the actual work.
- Learned evidence: record_risk(pid, "ML"|"DL", level, prob) stores ML_Risk(...) or DL_Risk(...) in the KB and replaces the previous prediction from the same source. _invalidate_evidence deletes stale predictions when vitals or symptoms change.
- State: snapshot() returns the current State.
KB synchronisation:
- _patient_facts and _resource_facts: generate the facts for one entity.
- _sync_patient and _sync_resource: delete an entity's old env facts and re-add the new ones.
- resync_all: does this for everything.
- describe(): returns a readable summary.

- The helper _num prints 125.0 as 125.

**4. interface.py – structured input**

It converts JSON or dicts into validated entities, so no LLM is needed.

- _require(d, key, ctx): raises a clear error for a missing field.
- build_vitals, build_patient and build_resource: turn dicts into model objects. They reject unknown vitals fields and accept either "vitals" (a single measurement) or "vitals_history" (a time series).
- load_scenario(env, file_or_dict): loads the initial resources and patients.
- load_events(source): reads a list of events.
- apply_event(env, event): applies one external change. The supported event types are add_patient, update_vitals, - add_symptom, remove_symptom, add_resource, set_available, discharge, start_treatment and advance_time.
- dump_scenario(env) and save_scenario export the current environment back to JSON.
**5. cli.py – interactive command line****

Run it with python3 cli.py scenario_example.json.

HELP is the command list text.
- _ask(prompt, cast, default) is a small input helper.
- _form_patient(env) is the guided form that asks for id, age, symptoms and vitals, then creates the patient and its facts.
- _opt parses - as unknown.
- run(env, line) executes one command. The commands are load, events, show, kb, facts, query, about, patient, vitals, symptom, down, up, assign, release, state, log, save and quit.
- main() is the REPL loop. It catches errors and prints them instead of crashing.
**6. demo.py**

A script that demonstrates the whole block in seven steps: load the scenario, print P1's facts, run KB queries, create state snapshots, inject ML/DL predictions, simulate a plan being broken by an event stream (new patient, ICU failure, deterioration), and print the final state. You can use it as the basis for your demonstration.

**7. test_block1.py**

13 unittest tests.

- TestKB: parsing and querying, duplicate and version handling, and rejection of variables.
- TestEnvironment: facts created from the scenario, assignment effects on the KB and state, state immutability, evidence invalidation, resource failure breaking assignments, the new-patient event, replacement of old predictions, validation errors, a JSON round trip, and the dynamic events file.

Run them with python3 -m unittest test_block1.

**8. scenario_example.json**

The example from the project statement. It has 8 resources: 2 doctors, 1 nurse, 2 emergency beds, 1 ICU bed, 1 examination room and 1 ECG. It has 3 patients: P1 with the 5-step vitals series from the PDF, plus P2 and P3 who are stable.

**9. scenario_dynamic_events.json**

The Block 6 test events: a new critical patient P4 arrives, ICU1 fails, and P2's vitals deteriorate.

