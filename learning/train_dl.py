"""
train_dl.py - Deep Learning component: risk from the EVOLUTION of the vitals.

Input : the last 6 hours, each hour = 6 vitals + 6 "measured" flags + age
        (tensor of shape (batch, 6, 13)).
Model : GRU (Gated Recurrent Unit). It reads the hours in order and keeps a
        memory (hidden state) of what it has seen, so it can react to a TREND
        (HR going up while SpO2 and BP go down) and not only to the last values.
        GRU vs LSTM: same idea, GRU has 2 gates instead of 3 -> fewer
        parameters, trains faster, performs similarly on short sequences.
Loss  : cross-entropy with class weights (Critical is rare).
Stop  : early stopping on the validation macro-F1.

Run:  python -m learning.train_dl
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np
import torch
from torch import nn

from .data_loader import VITALS
from .dataset import SAMPLES
from .evaluate import plot_confusion, save_json, scores, scores_proba, short, test_report

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
NV = len(VITALS)


class RiskGRU(nn.Module):
    def __init__(self, n_in: int = 2 * NV + 1, hidden: int = 64, n_classes: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Linear(32, n_classes))

    def forward(self, x):                 # x: (batch, time, features)
        _, h = self.gru(x)                # h: (layers, batch, hidden)
        return self.head(h[-1])           # last layer's memory after the last hour


class Normalizer:
    """Standardise the vitals (not the 0/1 flags) and append normalised age."""

    def __init__(self, mean, std, age_mean, age_std):
        self.mean, self.std = np.asarray(mean, "float32"), np.asarray(std, "float32")
        self.age_mean, self.age_std = float(age_mean), float(age_std)

    @classmethod
    def fit(cls, X_seq, ages):
        v = X_seq[:, :, :NV].reshape(-1, NV)
        return cls(v.mean(0), v.std(0) + 1e-6, ages.mean(), ages.std() + 1e-6)

    def __call__(self, X_seq, ages) -> np.ndarray:
        x = X_seq.copy()
        x[:, :, :NV] = (x[:, :, :NV] - self.mean) / self.std
        a = ((np.asarray(ages, "float32") - self.age_mean) / self.age_std)[:, None, None]
        return np.concatenate([x, np.repeat(a, x.shape[1], axis=1)], axis=2).astype("float32")

    def state(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "age_mean": self.age_mean, "age_std": self.age_std}


def predict_proba(model, X, batch=4096) -> np.ndarray:
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(torch.softmax(model(torch.from_numpy(X[i:i + batch])), 1).numpy())
    return np.concatenate(out)


def fit_gru(tr, va, epochs: int = 25, patience: int = 4, seed: int = 42, verbose=True):
    """Train a GRU on `tr` with early stopping on `va`. Returns (model, normalizer, history)."""
    torch.manual_seed(seed); np.random.seed(seed)
    norm = Normalizer.fit(tr.X_seq, tr.X_tab[:, 0])
    Xtr, Xva = norm(tr.X_seq, tr.X_tab[:, 0]), norm(va.X_seq, va.X_tab[:, 0])
    ytr = torch.from_numpy(tr.y).long()

    counts = np.bincount(tr.y, minlength=4)
    w = torch.tensor((counts.sum() / (4 * counts)) ** 0.5, dtype=torch.float32)  # softened balance
    model = RiskGRU()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    if verbose:
        print(f"train={len(tr)} val={len(va)}  input={Xtr.shape[1:]}  "
              f"params={sum(p.numel() for p in model.parameters())}  class weights={w.numpy().round(2)}")

    best, best_state, bad, history = -1, None, 0, []
    Xtr_t = torch.from_numpy(Xtr)
    for ep in range(1, epochs + 1):
        t = time.time(); model.train(); perm = torch.randperm(len(ytr)); tot = 0.0
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad(); loss = loss_fn(model(Xtr_t[idx]), ytr[idx]); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(idx)
        sc = scores(va.y, predict_proba(model, Xva).argmax(1), va.cur)
        history.append({"epoch": ep, "train_loss": round(tot / len(ytr), 4), "val_macro_f1": sc["macro_f1"]})
        if verbose:
            print(f"epoch {ep:2d} loss={tot / len(ytr):.4f} val macroF1={sc['macro_f1']:.3f} "
                  f"recall(Critical)={sc['per_class']['Critical']['recall']:.3f} [{time.time() - t:.0f}s]")
        if sc["macro_f1"] > best:
            best, bad = sc["macro_f1"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print("early stopping")
                break
    model.load_state_dict(best_state); model.eval()
    return model, norm, history


def main():
    tr, va, te, _ = pickle.load(open(SAMPLES, "rb"))
    model, norm, history = fit_gru(tr, va)
    val = scores_proba(va.y, predict_proba(model, norm(va.X_seq, va.X_tab[:, 0])), va.cur)
    test = test_report(te, predict_proba(model, norm(te.X_seq, te.X_tab[:, 0])))
    for part, sc in test.items():
        print(short(f"GRU ({part})", sc))
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "normalizer": norm.state(),
                "config": {"hidden": 64}}, os.path.join(MODEL_DIR, "dl_gru.pt"))
    save_json({"validation": val, "test": test, "history": history}, os.path.join(REPORT_DIR, "dl_results.json"))
    plot_confusion(test["all"]["confusion"], "DL (GRU) - test", os.path.join(REPORT_DIR, "cm_dl.png"))


if __name__ == "__main__":
    main()
