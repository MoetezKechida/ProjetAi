# Block 3 – Machine Learning and Deep Learning for Risk Prediction

```
pip install -r requirements.txt
python -m learning.demo_block3            # demonstration (needs only learning/models + reports)
python -m unittest learning.test_block3   # 15 tests
```

Retrain from scratch (the dataset is not versioned): download `training_setA` and `training_setB` from
https://physionet.org/content/challenge-2019/1.0.0/ and zip them as `data/raw/training_setA.zip` and
`data/raw/training_setB.zip`, then:
```
python -m learning.dataset     # ~3 min: load both hospitals, label, split, build samples -> data/cache/
python -m learning.train_ml    # ~30 s  -> learning/models/ml_model.joblib
python -m learning.train_dl    # ~3 min on CPU -> learning/models/dl_gru.pt
```

## Role in the agent
| Component | Sees | Writes into the KB | Used by (Block 2) |
|---|---|---|---|
| ML (Random Forest) | age + vitals **now** | `ML_Risk(P,Level,Prob)` | R5 (High ≥ 0.7), R6 (Critical ≥ 0.8) |
| DL (GRU) | vitals of the **last 6 hours** | `DL_Risk(P,Level,Prob)` | R7 (Critical ≥ 0.8), R8 (High ≥ 0.7) |

```python
from learning import RiskPredictor
RiskPredictor.load().attach(env)   # predicts all patients, then re-runs on
                                   # patient_arrived / vitals_updated / symptom_changed
```
Block 1 deletes stale predictions when a patient's data changes; the predictor writes fresh ones, which
emits `risk_recorded`, so the Reasoner (Block 2) re-reasons automatically.

**Patient P1 of the project statement**: `ML_Risk(P1,Critical,0.69)` and `DL_Risk(P1,Critical,0.84)`. With the
Block 2 rules, the DL evidence passes the 0.8 threshold (R7) → `CriticalRisk(P1)` → `RequiresICU(P1)`. The rules
alone only conclude `HighRisk(P1)`: the learned evidence changes the decision, as in the statement.

## Files
| File | Role |
|---|---|
| `data_loader.py` | Reads the PhysioNet `.psv` files (from the zips) of both hospitals. Keeps age + the 6 vitals of Block 1, drops lab values (the agent does not observe them). Rebuilds a missing DBP from MAP. |
| `labeling.py` | NEWS2 score and the 4 risk classes. |
| `dataset.py` | Stratified split by patient, imputation, 6-hour windows, samples. |
| `evaluate.py` | Metrics, AUROC, persistence baseline, "changing cases", per-hospital scores, confusion matrices. |
| `train_ml.py` | Trains Logistic Regression, Decision Tree, Random Forest, k-NN; keeps the best on validation. |
| `train_dl.py` | GRU model, normaliser, training loop with early stopping. |
| `predictor.py` | `RiskPredictor`: Block 1 `Patient` → model inputs → `env.record_risk(...)`. |
| `demo_block3.py` | End-to-end demonstration. |
| `models/` | `ml_model.joblib` (9 MB), `dl_gru.pt` (170 KB). |
| `reports/` | Metrics (JSON) and confusion matrices (PNG) for the report. |

## Dataset
PhysioNet/Computing in Cardiology Challenge 2019 (Reyna et al., *Crit Care Med* 2020), open access, one row per
ICU hour, **40,336 patients** from two hospital systems:
- hospital A – Beth Israel Deaconess (the hospital of MIMIC): 20,336 patients, 1,790 septic (8.8 %);
- hospital B – Emory University: 20,000 patients, 1,142 septic (5.7 %).

Missing values: temperature 66 %, respiratory rate 15 %, other vitals 10–15 %.

Why this dataset: real **time series** (needed for the DL), same variables as Block 1, reliable labels
(Sepsis-3), open access, two hospitals. MIMIC-IV, eICU and HiRID need credentialed access (1–2 weeks); their
open demos are too small (100 patients) or have no ready-made label. KTAS (Kaggle) has no time series, VitalDB
is perioperative (anaesthesia), not emergency care.

## The 4 classes (labeling.py)
PhysioNet only gives a binary label (`SepsisLabel = 1` from 6 h before the onset of sepsis). For each hour t:
- **Critical**: sepsis within 6 h, **or** NEWS2 ≥ 9 within the next 6 h (severe decompensation, so that a
  patient in shock without sepsis is also Critical);
- **High / Medium / Low**: worst NEWS2 level within the next 6 h (7–8 / 5–6 or one red parameter / otherwise).

NEWS2 is partial (no consciousness, no oxygen therapy in the data: max 15 instead of 20).

**Not circular**: the label uses hours **> t**, the models only see hours ≤ t, and the NEWS2 score is never an
input. In 39 % of the test windows the future level differs from the current one, so copying the present is
not enough. The remaining risk ("the future looks like the present") is measured by a **persistence baseline**.

## Preprocessing (dataset.py)
- **Split by patient, never by hour** (70/15/15): the hours of one patient are correlated; mixing them between
  train and test would leak information and inflate the scores. Checked: 0 patient shared between train,
  validation and test.
- **Stratified split**: the classes are per hour but the split is per patient, so patients are stratified on
  their profile (hospital × septic yes/no × worst NEWS2 level reached). Same proportions in the three sets:

  | | patients | windows | Low | Medium | High | Critical | septic patients |
  |---|---|---|---|---|---|---|---|
  | train | 27,889 | 272,288 | 49.6 % | 39.1 % | 7.4 % | 3.9 % | 7.4 % |
  | validation | 5,978 | 58,623 | 50.8 % | 38.3 % | 7.0 % | 3.9 % | 7.4 % |
  | test | 5,967 | 58,042 | 50.3 % | 38.4 % | 7.4 % | 4.0 % | 7.4 % |
- **No leakage through preprocessing**: imputation medians, GRU normalisation and class weights are computed
  on the training patients only; model choice and early stopping use the validation set only; the test set
  is used once, at the end.
- **Imputation**: last observation carried forward, then the train median. The DL also gets a "measured this
  hour" flag per vital, so it knows what was imputed.
- One window every 3 h (fewer near-duplicates); natural class proportions in every set.
- **Class imbalance** handled by class weights: `balanced` for the ML, square root of the balanced weights for
  the GRU. No oversampling: it does not improve the ranking (AUROC) and it inflates P(Critical), whereas Block 2
  applies probability thresholds.

## Models
- **ML**: 4 candidates, chosen on the validation macro-F1: Random Forest 0.426, Decision Tree 0.403,
  k-NN 0.369, Logistic Regression 0.365. Logistic Regression = linear reference; Decision Tree = readable but
  unstable; **Random Forest** (kept) = non-linear thresholds (vitals are dangerous at both extremes), robust to
  noise, gives feature importances; k-NN = slow at prediction, weak on rare classes.
- **DL**: 2-layer **GRU** (hidden 64, 42k parameters) on (6 hours × 13 features). GRU rather than LSTM: same
  principle with 2 gates instead of 3, fewer parameters, as good on short sequences. Early stopping on the
  validation macro-F1 (stopped at epoch 14).

## Results (test: 58,042 windows of 5,967 patients never seen during training)
| | Accuracy | Macro-F1 | Precision / Recall Critical | Macro-F1 on changing cases | AUROC Critical | AUROC ≥ High |
|---|---|---|---|---|---|---|
| Persistence (copy current level) | 0.611 | 0.356 | 0 / 0 | 0.000 | – | – |
| ML – Random Forest | 0.560 | 0.424 | 0.21 / 0.24 | 0.282 | 0.762 | 0.831 |
| **DL – GRU** | **0.646** | **0.490** | **0.44** / 0.18 | **0.304** | **0.785** | **0.850** |

Per hospital (macro-F1 / AUROC Critical):

| | ML | DL |
|---|---|---|
| hospital A | 0.416 / 0.736 | 0.483 / 0.756 |
| hospital B | 0.432 / 0.787 | 0.495 / 0.815 |

- Both models beat the persistence rule: they learned more than "the future looks like the present".
- The DL beats the ML on every global measure: the history of the vitals carries information. The ML recalls a
  few more Critical windows at the most probable class, but with half the precision of the DL.
- The scores are close in both hospitals: the models do not depend on one hospital's habits (only the 6 basic
  vitals are used, not lab tests whose units and frequencies differ between hospitals).
- Random Forest importances: respiratory rate 0.25, heart rate 0.24, systolic BP 0.19, SpO2 0.12.
- Example (demo §4): same vitals at the last hour, different histories → identical ML outputs, different DL
  outputs. The GRU learned that a patient who was very unstable in the last hours stays at high risk.
- Accuracy is not the right metric here: the persistence rule has a good accuracy because Low/Medium dominate,
  but never detects a Critical patient. Macro-F1, recall per class and AUROC are reported instead.

## Limitations
- **Population**: ICU patients, not emergency department; no symptoms (handled by the Block 2 rules).
- **Constructed labels**: the NEWS2 part encodes clinical thresholds; Critical recall stays low (~20 % at the
  most probable class) – predicting sepsis from vitals alone, without lab tests, is hard.
- **Class imbalance**: Critical is 4 % of the windows; weights help recall but lower precision.
- **Missing data**: temperature is missing 2 hours out of 3; imputation hides real uncertainty.
- **Uncertainty**: probabilities are spread over 4 classes and not formally calibrated, while Block 2 uses fixed
  thresholds (0.7 / 0.8): for P1 the ML says Critical at 0.69, which R6 ignores; the threshold matters.
- **Overfitting**: limited by the split by patient, early stopping, minimum leaf size and dropout; validation
  and test scores are close (Random Forest 0.426 / 0.424, GRU 0.492 / 0.490).
- Simulation only – not a medical tool.
