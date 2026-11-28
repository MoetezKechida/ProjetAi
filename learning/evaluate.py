"""
evaluate.py - Metrics shared by the ML and DL components.

Besides the usual metrics (accuracy, precision, recall, F1, confusion matrix)
we report two things that answer "did the model learn more than a rule?":
  * the PERSISTENCE baseline: predict the patient's current NEWS2 level
    (= "the future looks like the present", never predicts Critical);
  * the scores on the CHANGING cases only (future level != current level),
    where copying the present is by definition wrong.
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from .labeling import LEVELS, CRITICAL


def scores(y_true, y_pred, cur=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(4), zero_division=0)
    out = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", labels=range(4), zero_division=0), 4),
        "per_class": {LEVELS[i]: {"precision": round(p[i], 3), "recall": round(r[i], 3),
                                  "f1": round(f[i], 3), "support": int(s[i])} for i in range(4)},
        "confusion": confusion_matrix(y_true, y_pred, labels=range(4)).tolist(),
    }
    if cur is not None:
        cur = np.asarray(cur)
        ch = cur != y_true                       # level changes in the next hours
        worse = y_true > cur                     # deterioration (incl. Critical)
        out["changing_cases"] = {
            "n": int(ch.sum()),
            "accuracy": round(accuracy_score(y_true[ch], y_pred[ch]), 4),
            "macro_f1": round(f1_score(y_true[ch], y_pred[ch], average="macro",
                                       labels=range(4), zero_division=0), 4),
        }
        # did the model anticipate the deterioration (predict at least the future level)?
        out["deterioration_detected"] = round(float((y_pred[worse] >= y_true[worse]).mean()), 4)
    return out


def scores_proba(y_true, proba, cur=None) -> dict:
    """scores() on the most probable class + ranking quality of the probabilities:
    AUROC Critical-vs-rest and AUROC (High or Critical)-vs-rest."""
    y_true = np.asarray(y_true)
    out = scores(y_true, proba.argmax(1), cur)
    out["auroc_critical"] = round(float(roc_auc_score(y_true == CRITICAL, proba[:, CRITICAL])), 4)
    out["auroc_high_or_more"] = round(float(roc_auc_score(y_true >= 2, proba[:, 2:].sum(1))), 4)
    return out


def test_report(split, proba) -> dict:
    """Scores on the whole test set and per hospital."""
    rep = {"all": scores_proba(split.y, proba, split.cur)}
    for h in sorted(set(split.hospital)):
        m = split.hospital == h
        rep[f"hospital_{h}"] = scores_proba(split.y[m], proba[m], split.cur[m])
    return rep


def baseline_report(split) -> dict:
    rep = {"all": scores(split.y, persistence_baseline(split.cur), split.cur)}
    for h in sorted(set(split.hospital)):
        m = split.hospital == h
        rep[f"hospital_{h}"] = scores(split.y[m], persistence_baseline(split.cur[m]), split.cur[m])
    return rep


def persistence_baseline(cur):
    """Predict the current NEWS2 level."""
    return np.asarray(cur)


def short(name: str, sc: dict) -> str:
    line = (f"{name:<22} acc={sc['accuracy']:.3f}  macroF1={sc['macro_f1']:.3f}  "
            f"recall(Critical)={sc['per_class']['Critical']['recall']:.3f}")
    if "auroc_critical" in sc:
        line += f"  AUROC(Critical)={sc['auroc_critical']:.3f}"
    if "changing_cases" in sc:
        line += (f"  | changing: acc={sc['changing_cases']['accuracy']:.3f} "
                 f"F1={sc['changing_cases']['macro_f1']:.3f}  deterioration seen={sc['deterioration_detected']:.3f}")
    return line


def plot_confusion(cm, title: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{norm[i, j]:.0%}\n({int(cm[i, j])})", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xticks(range(4), LEVELS); ax.set_yticks(range(4), LEVELS)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
