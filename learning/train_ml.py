"""
train_ml.py - Machine Learning component: risk from the CURRENT state.

Input : age + the 6 vitals at hour t (what Patient.feature_dict() gives).
Output: one of Low / Medium / High / Critical + probability.

Candidates (all with class weights against the imbalance):
  * Logistic Regression - linear reference, fast, probabilities well calibrated
  * Decision Tree       - readable rules, but unstable / overfits
  * Random Forest       - many trees on random subsets: non-linear thresholds
                          (vitals are dangerous at BOTH extremes), robust to
                          noise, gives feature importances
  * SVM (RBF kernel)    - non-linear frontier with a margin; training time grows
                          ~quadratically with the data -> trained on 20,000 windows
  * k-NN                - "similar patients have similar risk"; stores the data,
                          slow at prediction -> trained on 40,000 windows
Every candidate is scored on the validation set (choice) AND on the test set
(reported for the comparison only). The best on the VALIDATION macro-F1 is kept.

Run:  python -m learning.train_ml
"""
from __future__ import annotations

import os
import pickle
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .dataset import SAMPLES, TAB_FEATURES
from .evaluate import (baseline_report, plot_confusion, save_json, scores_proba, short, test_report)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
# Training-set size limit for the models that do not scale to 272k windows.
SUBSAMPLE = {"SVM": 20000, "kNN": 40000}


def candidates(seed=42):
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced")),
        "DecisionTree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=50,
                                               class_weight="balanced", random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=50,
                                               class_weight="balanced_subsample", n_jobs=-1,
                                               random_state=seed),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, class_weight="balanced",
                                                   probability=True, random_state=seed)),
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=31, weights="distance")),
    }


def fit_candidates(tr, va, te=None, verbose=True):
    """Train every candidate on `tr`; score on `va` (and on `te` if given).
    Returns (best_name, fitted models, {name: {"validation": ..., "test": ...}})."""
    results, fitted = {}, {}
    rng = np.random.default_rng(0)
    for name, model in candidates().items():
        X, y = tr.X_tab, tr.y
        if name in SUBSAMPLE:
            idx = rng.choice(len(y), SUBSAMPLE[name], replace=False); X, y = X[idx], y[idx]
        t = time.time(); model.fit(X, y); fit_s = time.time() - t
        t = time.time(); val = scores_proba(va.y, model.predict_proba(va.X_tab), va.cur)
        pred_s = time.time() - t
        results[name] = {"train_windows": len(y), "fit_seconds": round(fit_s, 1),
                         "predict_seconds_val": round(pred_s, 1), "validation": val}
        if te is not None:
            results[name]["test"] = test_report(te, model.predict_proba(te.X_tab))
        fitted[name] = model
        if verbose:
            print(short(name + " (val)", val), f"[fit {fit_s:.0f}s]")
    best = max(results, key=lambda n: results[n]["validation"]["macro_f1"])
    return best, fitted, results


def main():
    tr, va, te, med = pickle.load(open(SAMPLES, "rb"))
    print(f"train={len(tr)} val={len(va)} test={len(te)}  features={TAB_FEATURES}")
    best, fitted, results = fit_candidates(tr, va, te)
    print(f"\nbest on validation: {best}")
    model = fitted[best]
    base = baseline_report(te)
    print(short("persistence (test)", base["all"]))
    for name, r in results.items():
        print(short(f"{name} (test)", r["test"]["all"]))

    importance = None
    if hasattr(model, "feature_importances_"):
        importance = dict(sorted(zip(TAB_FEATURES, model.feature_importances_.round(3).tolist()),
                                 key=lambda kv: -kv[1]))
        print("feature importance:", importance)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": model, "features": TAB_FEATURES, "medians": med, "name": best},
                os.path.join(MODEL_DIR, "ml_model.joblib"), compress=3)
    save_json({"best": best, "candidates": results, "test": results[best]["test"],
               "persistence_test": base, "feature_importance": importance},
              os.path.join(REPORT_DIR, "ml_results.json"))
    plot_confusion(results[best]["test"]["all"]["confusion"], f"ML ({best}) - test",
                   os.path.join(REPORT_DIR, "cm_ml.png"))
    plot_confusion(base["all"]["confusion"], "Persistence baseline - test",
                   os.path.join(REPORT_DIR, "cm_persistence.png"))


if __name__ == "__main__":
    main()
