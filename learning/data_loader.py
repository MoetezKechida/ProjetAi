"""
data_loader.py - Read the PhysioNet/CinC Challenge 2019 data (Block 3).

Each patient is one pipe-separated file pXXXXXX.psv with one row per ICU hour
(41 columns: 8 vital signs, 26 lab values, 6 demographics, SepsisLabel).
The data come from two hospital systems, distributed as two folders:
  training_setA  Beth Israel Deaconess Medical Center   (ids p000001...)
  training_setB  Emory University Hospital              (ids p100001...)

We keep ONLY what the agent can observe in Block 1 (model.VitalSigns):
age + heart rate, systolic/diastolic BP, SpO2, temperature, respiratory rate.
Lab values are dropped: the environment does not represent them, so a model
trained on them could not be used by the agent.

MAP is kept temporarily, only to rebuild a missing diastolic pressure
(MAP ~ (SBP + 2*DBP) / 3  =>  DBP ~ (3*MAP - SBP) / 2).

Data source: Reyna et al., "Early Prediction of Sepsis From Clinical Data:
the PhysioNet/Computing in Cardiology Challenge 2019", Crit Care Med 2020.
"""
from __future__ import annotations

import io
import os
import zipfile
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# PhysioNet column -> name used in Block 1 (model.VitalSigns)
VITAL_MAP = {
    "HR": "heart_rate",
    "SBP": "systolic",
    "DBP": "diastolic",
    "O2Sat": "spo2",
    "Temp": "temperature",
    "Resp": "resp_rate",
}
# Same order as VitalSigns.as_row(): [HR, SBP, DBP, SpO2, Temp, RR]
VITALS = ["heart_rate", "systolic", "diastolic", "spo2", "temperature", "resp_rate"]

_KEEP = list(VITAL_MAP) + ["MAP", "Age", "ICULOS", "SepsisLabel"]


def _clean(df: pd.DataFrame, pid: str) -> pd.DataFrame:
    df = df[_KEEP].rename(columns=VITAL_MAP)
    # Rebuild diastolic from MAP when only DBP is missing (frequent).
    rebuilt = (3 * df["MAP"] - df["systolic"]) / 2
    rebuilt = rebuilt.where((rebuilt > 20) & (rebuilt < 200))
    df["diastolic"] = df["diastolic"].fillna(rebuilt)
    df = df.drop(columns="MAP").rename(
        columns={"Age": "age", "ICULOS": "hour", "SepsisLabel": "sepsis"})
    df.insert(0, "pid", pid)
    return df


def read_psv(source, pid: Optional[str] = None) -> pd.DataFrame:
    """Read one .psv file (path or file-like) into a cleaned DataFrame."""
    if pid is None:
        pid = os.path.splitext(os.path.basename(str(source)))[0]
    return _clean(pd.read_csv(source, sep="|"), pid)


def _iter_zip(path: str, limit: Optional[int]) -> Iterable[pd.DataFrame]:
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".psv"))
        for name in names[:limit]:
            with z.open(name) as f:
                yield read_psv(io.BytesIO(f.read()),
                               os.path.splitext(os.path.basename(name))[0])


def _iter_dir(path: str, limit: Optional[int]) -> Iterable[pd.DataFrame]:
    names = sorted(n for n in os.listdir(path) if n.endswith(".psv"))
    for name in names[:limit]:
        yield read_psv(os.path.join(path, name))


def load_physionet(source: str, limit: Optional[int] = None,
                   cache: Optional[str] = None) -> pd.DataFrame:
    """Load every patient of a folder or .zip into one long DataFrame
    (one row per patient-hour). With `cache`, the result is pickled once and
    reloaded instantly afterwards."""
    if cache and os.path.exists(cache):
        return pd.read_pickle(cache)
    it = _iter_zip(source, limit) if source.endswith(".zip") else _iter_dir(source, limit)
    df = pd.concat(list(it), ignore_index=True)
    for c in VITALS + ["age"]:
        df[c] = df[c].astype("float32")
    df["sepsis"] = df["sepsis"].astype("int8")
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        df.to_pickle(cache)
    return df


HOSPITALS = {"A": "training_setA.zip", "B": "training_setB.zip"}


def load_all(raw_dir: str = "data/raw", cache_dir: str = "data/cache") -> pd.DataFrame:
    """Both hospitals in one table, with a `hospital` column ('A' or 'B')."""
    parts = []
    for h, name in HOSPITALS.items():
        df = load_physionet(os.path.join(raw_dir, name), cache=os.path.join(cache_dir, f"set{h}.pkl"))
        parts.append(df.assign(hospital=h))
    return pd.concat(parts, ignore_index=True)


def summary(df: pd.DataFrame) -> str:
    """Short description of the loaded data (for the report)."""
    per_patient = df.groupby("pid")["sepsis"].max()
    missing = df[VITALS].isna().mean().mul(100).round(1)
    lines = [
        f"patients: {df.pid.nunique()}  rows (patient-hours): {len(df)}",
        *([f"  hospital {h}: {g.pid.nunique()} patients, {g.groupby('pid').sepsis.max().sum()} septic"
           for h, g in df.groupby("hospital")] if "hospital" in df else []),
        f"septic patients: {per_patient.sum()} ({per_patient.mean():.1%})",
        f"hours per patient: median {df.groupby('pid').size().median():.0f}",
        "missing values per vital (%): " + ", ".join(f"{k}={v}" for k, v in missing.items()),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary(load_all()))
