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
  * k-NN                - "similar patients have similar risk"; slow at
                          prediction time, sensitive to scaling
The best one on the VALIDATION macro-F1 is kept; the test set is used once.

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
from sklearn.tree import DecisionTreeClassifier

from .dataset import SAMPLES, TAB_FEATURES
from .evaluate import (baseline_report, plot_confusion, save_json, scores, short, test_report)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")


def candidates(seed=42):
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced")),
        "DecisionTree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=50,
                                               class_weight="balanced", random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=50,
                                               class_weight="balanced_subsample", n_jobs=-1,
                                               random_state=seed),
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=31, weights="distance")),
    }


def fit_candidates(tr, va, verbose=True):
    """Train every candidate on `tr`, score on `va`. Returns (best_name, models, val_scores)."""
    results, fitted = {}, {}
    rng = np.random.default_rng(0)
    for name, model in candidates().items():
        X, y = tr.X_tab, tr.y
        if name == "kNN":                       # k-NN stores the data: subsample for speed
            idx = rng.choice(len(y), 40000, replace=False); X, y = X[idx], y[idx]
        t = time.time(); model.fit(X, y)
        sc = scores(va.y, model.predict(va.X_tab), va.cur)
        sc["train_seconds"] = round(time.time() - t, 1)
        results[name] = sc; fitted[name] = model
        if verbose:
            print(short(name + " (val)", sc), f"[{sc['train_seconds']}s]")
    best = max(results, key=lambda n: results[n]["macro_f1"])
    return best, fitted, results


def main():
    tr, va, te, med = pickle.load(open(SAMPLES, "rb"))
    print(f"train={len(tr)} val={len(va)} test={len(te)}  features={TAB_FEATURES}")
    best, fitted, results = fit_candidates(tr, va)
    print(f"\nbest on validation: {best}")
    model = fitted[best]
    test = test_report(te, model.predict_proba(te.X_tab))
    base = baseline_report(te)
    for part in test:
        print(short(f"persistence ({part})", base[part]))
        print(short(f"{best} ({part})", test[part]))

    importance = None
    if hasattr(model, "feature_importances_"):
        importance = dict(sorted(zip(TAB_FEATURES, model.feature_importances_.round(3).tolist()),
                                 key=lambda kv: -kv[1]))
        print("feature importance:", importance)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": model, "features": TAB_FEATURES, "medians": med, "name": best},
                os.path.join(MODEL_DIR, "ml_model.joblib"), compress=3)
    save_json({"validation": results, "best": best, "test": test, "persistence_test": base,
               "feature_importance": importance}, os.path.join(REPORT_DIR, "ml_results.json"))
    plot_confusion(test["all"]["confusion"], f"ML ({best}) - test", os.path.join(REPORT_DIR, "cm_ml.png"))
    plot_confusion(base["all"]["confusion"], "Persistence baseline - test",
                   os.path.join(REPORT_DIR, "cm_persistence.png"))


if __name__ == "__main__":
    main()
