"""
labeling.py - Build the 4 risk classes (Low / Medium / High / Critical).

PhysioNet 2019 only gives a binary label (SepsisLabel = 1 from 6 h before the
Sepsis-3 onset). We derive 4 classes for every patient-hour t:

  Critical  SepsisLabel(t) = 1              (sepsis within 6 h, or just started)
            or NEWS2 >= 9 within the next `horizon` hours (severe decompensation)
  otherwise the WORST NEWS2 level over the NEXT `horizon` hours (t+1 .. t+h):
  High      NEWS2 7-8
  Medium    NEWS2 5-6, or one single parameter scoring 3 ("red score")
  Low       otherwise

Critical covers both sepsis AND severe physiological decompensation, so that a
patient in shock without sepsis is also Critical (what matters for ICU beds).

Why this is not circular:
  * the model sees hours <= t, the label is computed on hours > t: it must
    FORECAST, not re-compute a formula on its inputs;
  * sepsis comes from Sepsis-3 (antibiotics, cultures, SOFA from lab tests),
    none of which is a model input;
  * the NEWS2 score itself is never given to the models (raw vitals only).
The remaining risk ("the future looks like the present") is measured by the
persistence baseline in evaluate.py.

NEWS2 (Royal College of Physicians, 2017) - partial version: consciousness
(AVPU) and supplemental oxygen are not in the data, so the maximum is 15
instead of 20. SIMULATION ONLY - educational thresholds, not medical advice.
"""
from __future__ import annotations

import numpy as np

LEVELS = ["Low", "Medium", "High", "Critical"]
LOW, MEDIUM, HIGH, CRITICAL = range(4)
HORIZON = 6            # hours looked ahead
CRITICAL_NEWS2 = 9     # future NEWS2 at or above this value -> Critical


def _band(x: np.ndarray, edges, points) -> np.ndarray:
    """Score each value: points[i] for edges[i-1] < x <= edges[i]. NaN -> 0
    (unknown parameter counts as normal - documented limitation)."""
    idx = np.searchsorted(np.asarray(edges, dtype=float), x, side="left")
    out = np.asarray(points)[idx]
    return np.where(np.isnan(x), 0, out)


def news2_components(hr, sbp, spo2, temp, rr) -> np.ndarray:
    """Return an (n, 5) array of NEWS2 sub-scores (HR, SBP, SpO2, Temp, RR)."""
    hr, sbp, spo2, temp, rr = (np.asarray(v, dtype=float) for v in (hr, sbp, spo2, temp, rr))
    return np.stack([
        _band(hr,   [40, 50, 90, 110, 130],      [3, 1, 0, 1, 2, 3]),
        _band(sbp,  [90, 100, 110, 219],         [3, 2, 1, 0, 3]),
        _band(spo2, [91, 93, 95],                [3, 2, 1, 0]),
        _band(temp, [35.0, 36.0, 38.0, 39.0],    [3, 1, 0, 1, 2]),
        _band(rr,   [8, 11, 20, 24],             [3, 1, 0, 2, 3]),
    ], axis=1)


def news2_level(hr, sbp, spo2, temp, rr) -> tuple:
    """(score, level) per row, level in {LOW, MEDIUM, HIGH}."""
    comp = news2_components(hr, sbp, spo2, temp, rr)
    score = comp.sum(axis=1)
    level = np.where(score >= 7, HIGH,
             np.where((score >= 5) | (comp.max(axis=1) == 3), MEDIUM, LOW))
    return score, level


def future_labels(score: np.ndarray, level: np.ndarray, sepsis: np.ndarray,
                  horizon: int = HORIZON) -> np.ndarray:
    """Label of each hour of ONE patient (arrays in time order).
    -1 = no label (not enough future hours and not septic)."""
    n = len(level)
    y = np.full(n, -1, dtype=int)
    for t in range(n):
        if sepsis[t] == 1:
            y[t] = CRITICAL
        elif t + horizon < n:
            future = slice(t + 1, t + 1 + horizon)
            y[t] = CRITICAL if score[future].max() >= CRITICAL_NEWS2 else int(level[future].max())
    return y
