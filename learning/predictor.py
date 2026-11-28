"""
predictor.py - Plugs the trained ML and DL models into the agent (Block 3 -> 1/2).

    from learning import RiskPredictor
    pred = RiskPredictor.load()
    pred.attach(env)          # predicts every patient now, then after each change

For each active patient it writes two facts into the Knowledge Base through
the Block 1 API env.record_risk():
    ML_Risk(P, Level, Prob)   Random Forest on the CURRENT vitals + age
    DL_Risk(P, Level, Prob)   GRU on the LAST 6 HOURS of vitals
Level = most probable class, Prob = its probability. The rules R5-R8 of
Block 2 then turn this evidence into HighRisk / CriticalRisk.

Dynamic behaviour: when vitals or symptoms change, Block 1 deletes the stale
predictions (event 'evidence_invalidated'); attach() listens to
patient_arrived / vitals_updated / symptom_changed and re-runs both models,
so the KB always holds fresh evidence and Block 2 re-reasons automatically.

Conversion Block 1 -> model input (same as training, see dataset.py):
  * VitalSigns.as_row() gives [HR, SBP, DBP, SpO2, Temp, RR] (None -> missing);
  * a blood-pressure category without a number is mapped to a typical
    systolic value (Low 85, Normal 120, High 150);
  * missing values: last observation carried forward, then train medians;
  * fewer than 6 measurements: the window is left-padded (flags = 0).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import joblib
import numpy as np

from .dataset import WINDOW, patient_matrix, window_at
from .labeling import LEVELS

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
BP_CATEGORY_SYSTOLIC = {"Low": 85.0, "Normal": 120.0, "High": 150.0}
_TRIGGERS = ("patient_arrived", "vitals_updated", "symptom_changed")


@dataclass
class Prediction:
    source: str            # "ML" or "DL"
    level: str             # Low / Medium / High / Critical
    probability: float     # probability of that level
    probs: Dict[str, float]

    def __str__(self) -> str:
        dist = ", ".join(f"{k}={v:.2f}" for k, v in self.probs.items())
        return f"{self.source}_Risk = {self.level} ({self.probability:.2f})   [{dist}]"


def _to_prediction(source: str, p: np.ndarray) -> Prediction:
    k = int(p.argmax())
    return Prediction(source, LEVELS[k], round(float(p[k]), 4),
                      {LEVELS[i]: round(float(p[i]), 3) for i in range(4)})


class RiskPredictor:
    def __init__(self, ml_bundle: dict, gru, normalizer) -> None:
        self.ml = ml_bundle["model"]
        self.ml_name = ml_bundle.get("name", "ML")
        self.medians = ml_bundle["medians"]
        self.gru = gru
        self.norm = normalizer
        self.env = None
        self.log = []                       # (pid, Prediction) history, for demos

    @classmethod
    def load(cls, model_dir: str = MODEL_DIR) -> "RiskPredictor":
        import torch                        # imported lazily: only needed for the DL part
        from .train_dl import Normalizer, RiskGRU
        bundle = joblib.load(os.path.join(model_dir, "ml_model.joblib"))
        ck = torch.load(os.path.join(model_dir, "dl_gru.pt"), weights_only=False)
        gru = RiskGRU(hidden=ck["config"]["hidden"])
        gru.load_state_dict(ck["state_dict"]); gru.eval()
        return cls(bundle, gru, Normalizer(**ck["normalizer"]))

    # ---- Block 1 Patient -> model inputs ----------------------------------
    def _matrices(self, patient):
        rows = []
        for v in patient.history:
            row = [np.nan if x is None else float(x) for x in v.as_row()]
            if np.isnan(row[1]) and v.bp_level in BP_CATEGORY_SYSTOLIC:
                row[1] = BP_CATEGORY_SYSTOLIC[v.bp_level]
            rows.append(row)
        return patient_matrix(np.array(rows, dtype=float), self.medians)

    def predict(self, patient) -> Optional[Dict[str, Prediction]]:
        """ML and DL predictions for one Block 1 Patient (None if no vitals)."""
        if not patient.history:
            return None
        import torch
        filled, mask = self._matrices(patient)
        x_tab = np.concatenate([[patient.age], filled[-1]])[None].astype("float32")
        p_ml = self.ml.predict_proba(x_tab)[0]
        x_seq = window_at(filled, mask, len(filled) - 1, WINDOW)[None].astype("float32")
        with torch.no_grad():
            logits = self.gru(torch.from_numpy(self.norm(x_seq, np.array([patient.age], "float32"))))
            p_dl = torch.softmax(logits, 1)[0].numpy()
        return {"ML": _to_prediction("ML", p_ml), "DL": _to_prediction("DL", p_dl)}

    # ---- writing the evidence into the Knowledge Base ---------------------
    def record(self, env, pid: str) -> Optional[Dict[str, Prediction]]:
        p = env.patient(pid)
        if p.status == "Discharged":
            return None
        preds = self.predict(p)
        if preds:
            for src in ("ML", "DL"):
                env.record_risk(pid, src, preds[src].level, preds[src].probability)
                self.log.append((pid, preds[src]))
        return preds

    def record_all(self, env) -> Dict[str, Dict[str, Prediction]]:
        return {pid: r for pid in list(env.patients) if (r := self.record(env, pid))}

    def attach(self, env) -> Dict[str, Dict[str, Prediction]]:
        """Predict every patient now and keep predictions fresh afterwards."""
        self.env = env
        env.subscribe(self._on_event)
        return self.record_all(env)

    def _on_event(self, event) -> None:
        if event.kind in _TRIGGERS and event.subject in self.env.patients:
            self.record(self.env, event.subject)
