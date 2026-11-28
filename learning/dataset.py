"""
dataset.py - Turn the patient-hour table into ML / DL samples.

One sample = one patient at one hour t:
  X_seq  (WINDOW, 12)  the last WINDOW hours: 6 vitals (forward-filled) +
                       6 "was measured this hour" flags            -> DL (GRU)
  X_tab  (7,)          age + the 6 vitals at hour t (last known)    -> ML
  y                    risk class 0..3 (see labeling.py)
  cur                  NEWS2 level at hour t (persistence baseline)
  pid, hospital        patient id and hospital ('A' / 'B')

Key choices (to justify in the report):
  * Split BY PATIENT, never by hour (70/15/15): the hours of one patient are
    strongly correlated; putting some in train and others in test would leak
    information and inflate the scores.
  * STRATIFIED split: patients are stratified on (hospital, septic yes/no,
    worst NEWS2 level reached), so train, validation and test have the same
    proportions of hospitals, septic patients and risk classes.
  * Imputation: last observation carried forward (what a nurse would read on
    the monitor), then the TRAIN median for values never measured yet.
    The DL also gets the "measured" flags, so it knows what was imputed.
  * One sample every `stride` hours to limit near-duplicate samples; natural
    class proportions everywhere (imbalance is handled by class weights).

Run:  python -m learning.dataset      (needs data/raw/training_setA.zip and training_setB.zip)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_loader import VITALS
from .labeling import LEVELS, future_labels, news2_level

WINDOW = 6
STRIDE = 3
TAB_FEATURES = ["age"] + VITALS
SAMPLES = "data/cache/samples.pkl"


@dataclass
class Split:
    X_seq: np.ndarray
    X_tab: np.ndarray
    y: np.ndarray
    cur: np.ndarray
    pid: np.ndarray
    hospital: np.ndarray

    def __len__(self):
        return len(self.y)

    def subset(self, mask) -> "Split":
        return Split(*(getattr(self, f)[mask] for f in
                       ("X_seq", "X_tab", "y", "cur", "pid", "hospital")))


def patient_strata(df: pd.DataFrame) -> pd.Series:
    """One profile per patient, used to STRATIFY the split:
    hospital x septic (0/1) x worst NEWS2 level reached during the stay.
    Classes are given per hour but the split is per patient, so we stratify on
    this patient profile; the hourly class proportions then follow."""
    def profile(g):
        v = g[VITALS].ffill().to_numpy(dtype=float)
        _, level = news2_level(v[:, 0], v[:, 1], v[:, 3], v[:, 4], v[:, 5])
        return f"{g['hospital'].iloc[0]}-{int(g['sepsis'].max())}-{int(level.max())}"
    return df.groupby("pid", sort=True).apply(profile)


def split_patients(strata: pd.Series, seed: int = 42, frac=(0.70, 0.15)):
    """Stratified split of PATIENTS (never hours) into train / val / test."""
    from sklearn.model_selection import train_test_split
    p = strata.index.to_numpy()
    rest = 1 - frac[0]
    tr, tmp = train_test_split(p, test_size=rest, random_state=seed, stratify=strata.loc[p])
    va, te = train_test_split(tmp, test_size=(rest - frac[1]) / rest, random_state=seed,
                              stratify=strata.loc[tmp])
    return set(tr), set(va), set(te)


def train_medians(df: pd.DataFrame, train_ids) -> dict:
    return df[df.pid.isin(train_ids)][VITALS].median().to_dict()


def patient_matrix(vitals_raw: np.ndarray, medians: dict) -> tuple:
    """(T,6) raw vitals with NaN -> (filled (T,6), mask (T,6))."""
    mask = ~np.isnan(vitals_raw)
    filled = pd.DataFrame(vitals_raw).ffill().to_numpy()
    med = np.array([medians[v] for v in VITALS])
    filled = np.where(np.isnan(filled), med, filled)
    return filled.astype("float32"), mask.astype("float32")


def window_at(filled, mask, t, window=WINDOW) -> np.ndarray:
    """Window ending at hour t; left-padded by repeating the first hour."""
    lo = t - window + 1
    idx = np.clip(np.arange(lo, t + 1), 0, None)
    w = np.concatenate([filled[idx], mask[idx]], axis=1)
    if lo < 0:                       # padded hours were not measured
        w[: -lo, len(VITALS):] = 0
    return w


def build_samples(df: pd.DataFrame, ids, medians: dict, stride: int = STRIDE) -> Split:
    seqs, tabs, ys, curs, pids, hosp = [], [], [], [], [], []
    sub = df[df.pid.isin(ids)]
    for pid, g in sub.groupby("pid", sort=True):
        filled, mask = patient_matrix(g[VITALS].to_numpy(dtype=float), medians)
        hr, sbp, _, spo2, temp, rr = filled.T
        score, level = news2_level(hr, sbp, spo2, temp, rr)
        y = future_labels(score, level, g["sepsis"].to_numpy())
        age, h = float(g["age"].iloc[0]), g["hospital"].iloc[0]
        for t in range(WINDOW - 1, len(g), stride):
            if y[t] < 0:
                continue
            seqs.append(window_at(filled, mask, t))
            tabs.append(np.concatenate([[age], filled[t]]))
            ys.append(y[t]); curs.append(level[t]); pids.append(pid); hosp.append(h)
    return Split(np.stack(seqs).astype("float32"), np.stack(tabs).astype("float32"),
                 np.array(ys), np.array(curs), np.array(pids), np.array(hosp))


def build_all(df: pd.DataFrame, seed: int = 42):
    """(train, val, test, train medians)."""
    tr, va, te = split_patients(patient_strata(df), seed)
    med = train_medians(df, tr)
    return (build_samples(df, tr, med), build_samples(df, va, med),
            build_samples(df, te, med), med)


def class_share(s: Split) -> dict:
    return {LEVELS[i]: round(float(c), 3) for i, c in enumerate(np.bincount(s.y, minlength=4) / len(s))}


def main():
    import pickle
    from learning.data_loader import load_all, summary   # absolute: pickle Split as learning.dataset.Split
    from learning.dataset import build_all as _build
    from learning.evaluate import save_json
    data = load_all()
    print(summary(data))
    parts = _build(data)
    report = {"patients": int(data.pid.nunique()), "rows": int(len(data)),
              "hospitals": {h: {"patients": int(g.pid.nunique()),
                                "septic_patients": int(g.groupby("pid").sepsis.max().sum())}
                            for h, g in data.groupby("hospital")},
              "missing_pct": data[VITALS].isna().mean().mul(100).round(1).to_dict(), "splits": {}}
    for name, s in zip(("train", "val", "test"), parts[:3]):
        pat = data[data.pid.isin(set(s.pid))].groupby("pid").sepsis.max()
        report["splits"][name] = {"patients": len(set(s.pid)), "windows": len(s),
                                  "septic_patients": round(float(pat.mean()), 3),
                                  "class_share": class_share(s)}
        print(f"{name:5} {len(set(s.pid)):6} patients {len(s):7} windows  "
              + "  ".join(f"{k}={v:.1%}" for k, v in class_share(s).items())
              + f"  septic={pat.mean():.1%}")
    with open(SAMPLES, "wb") as f:
        pickle.dump(parts, f)
    save_json(report, "learning/reports/dataset_summary.json")
    print("saved", SAMPLES)


if __name__ == "__main__":
    main()
