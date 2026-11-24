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
- **Block 2 (reasoning)**: read/write `env.kb` (`add(fact, source="derived")`, `query`, `holds`, `retract_where`);
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
