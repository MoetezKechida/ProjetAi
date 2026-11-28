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
python -m learning.train_ml    # ~3 min: 5 ML candidates -> learning/models/ml_model.joblib
python -m learning.train_dl    # ~3 min on CPU -> learning/models/dl_gru.pt
python -m learning.evaluate    # comparison table + confusion matrices of all models -> learning/reports/
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
| `evaluate.py` | Metrics, AUROC, persistence baseline, "changing cases", per-hospital scores, confusion matrices, model comparison. |
| `train_ml.py` | Trains Logistic Regression, Decision Tree, Random Forest, SVM, k-NN; keeps the best on validation. |
| `train_dl.py` | GRU model, normaliser, training loop with early stopping. |
| `predictor.py` | `RiskPredictor`: Block 1 `Patient` → model inputs → `env.record_risk(...)`. |
| `demo_block3.py` | End-to-end demonstration. |
| `models/` | `ml_model.joblib` (9 MB), `dl_gru.pt` (170 KB). |
| `reports/` | Metrics (JSON), `model_comparison.md` (table of all models), `cm_all_models.png` (all confusion matrices), confusion matrices of the kept models. |

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
**ML – 5 candidates**, all with class weights, trained on the same windows (current vitals + age):

| Model | Why it is a candidate | Limit for this problem |
|---|---|---|
| Logistic Regression | linear reference, fast, interpretable coefficients | a vital is dangerous at **both** extremes (low and high heart rate): not linear |
| Decision Tree | readable thresholds, close to clinical rules | unstable, overfits |
| **Random Forest** (kept) | many trees on random subsets: non-linear thresholds, robust to noise, feature importances | larger model (9 MB) |
| SVM (RBF kernel) | non-linear frontier with a maximal margin | training time grows ~quadratically: trained on 20,000 windows |
| k-NN (k = 31) | "similar patients have similar risk" | stores the data, slow to predict, sensitive to scaling: trained on 40,000 windows |

**DL**: 2-layer **GRU** (hidden 64, 42k parameters) on (6 hours × 13 features). GRU rather than LSTM: same
principle with 2 gates instead of 3, fewer parameters, as good on short sequences. Early stopping on the
validation macro-F1 (stopped at epoch 14).

The model kept for each component is the best **on the validation set**; the test set is only used to report
the scores below.

## Results (test: 58,042 windows of 5,967 patients never seen during training)
| Model | Type | Val macro-F1 | Accuracy | Macro precision | Macro recall | Macro-F1 | Critical P / R / F1 | AUROC Critical | AUROC ≥ High |
|---|---|---|---|---|---|---|---|---|---|
| Persistence (copy current level) | rule | – | 0.611 | 0.417 | 0.362 | 0.356 | 0.00 / 0.00 / 0.00 | – | – |
| Logistic Regression | ML | 0.365 | 0.495 | 0.370 | 0.416 | 0.368 | 0.07 / 0.23 / 0.11 | 0.729 | 0.806 |
| Decision Tree | ML | 0.403 | 0.541 | 0.401 | 0.454 | 0.402 | 0.13 / 0.26 / 0.18 | 0.712 | 0.813 |
| **Random Forest** | ML | **0.426** | 0.559 | 0.426 | 0.474 | 0.424 | 0.21 / 0.24 / 0.22 | 0.762 | 0.831 |
| SVM (RBF) | ML | 0.360 | 0.625 | 0.528 | 0.365 | 0.364 | 0.42 / 0.02 / 0.03 | 0.744 | 0.819 |
| k-NN | ML | 0.368 | 0.617 | 0.541 | 0.360 | 0.362 | 0.54 / 0.03 / 0.06 | 0.695 | 0.803 |
| **GRU** | DL | **0.494** | **0.646** | 0.533 | **0.475** | **0.490** | 0.44 / 0.18 / 0.25 | **0.785** | **0.850** |

Confusion matrices of all models: `reports/cm_all_models.png`.

Per hospital (macro-F1 / AUROC Critical):

| | Random Forest | GRU |
|---|---|---|
| hospital A | 0.416 / 0.736 | 0.483 / 0.756 |
| hospital B | 0.432 / 0.787 | 0.495 / 0.815 |

**ML comparison**
- **Random Forest** has the best validation macro-F1 and the best AUROC of the ML models: kept.
- **Logistic Regression** is the weakest on macro-F1: the risk is not a linear function of the vitals.
- **Decision Tree** is close to the Random Forest on recall but less precise and ranks patients less well
  (AUROC 0.71): one tree is less stable than a forest.
- **SVM and k-NN** have a high accuracy but almost never predict Critical (recall 0.02–0.03). The SVM ranks
  patients reasonably well (AUROC 0.74), but its probabilities are recalibrated on the training data, which
  cancels the class weights; k-NN votes among neighbours, where Critical is always a minority. Both are also
  trained on a subsample only.

**ML vs DL**
- Both beat the persistence rule: they learned more than "the future looks like the present".
- The GRU beats every ML model on macro-F1, AUROC and accuracy: the history of the vitals carries information.
  The Random Forest recalls a few more Critical windows at the most probable class, but with half the precision.
- The scores are close in both hospitals: the models do not depend on one hospital's habits (only the 6 basic
  vitals are used, not lab tests whose units and frequencies differ between hospitals).
- Random Forest importances: respiratory rate 0.25, heart rate 0.24, systolic BP 0.19, SpO2 0.12.
- Example (demo §4): same vitals at the last hour, different histories → identical ML outputs, different DL
  outputs. The GRU learned that a patient who was very unstable in the last hours stays at high risk.
- **Accuracy is misleading** here: the persistence rule, the SVM and k-NN have a good accuracy because
  Low/Medium dominate, but (almost) never detect a Critical patient. Macro-F1, recall per class and AUROC are
  the measures used to compare.

## Limitations
- **Population**: ICU patients, not emergency department; no symptoms (handled by the Block 2 rules).
- **Constructed labels**: the NEWS2 part encodes clinical thresholds; Critical recall stays low (~20 % at the
  most probable class) – predicting sepsis from vitals alone, without lab tests, is hard.
- **Class imbalance**: Critical is 4 % of the windows; weights help recall but lower precision.
- **Missing data**: temperature is missing 2 hours out of 3; imputation hides real uncertainty.
- **Uncertainty**: probabilities are spread over 4 classes and not formally calibrated, while Block 2 uses fixed
  thresholds (0.7 / 0.8): for P1 the ML says Critical at 0.69, which R6 ignores; the threshold matters.
- **Overfitting**: limited by the split by patient, early stopping, minimum leaf size and dropout; validation
  and test scores are close (Random Forest 0.426 / 0.424, GRU 0.494 / 0.490).
- Simulation only – not a medical tool.
