"""
demo_block3.py - Block 3 demonstration (run from the repository root):

    python -m learning.demo_block3

1. Data            PhysioNet 2019 (2 hospitals) and the 4 risk classes
2. Evaluation      persistence baseline vs ML vs DL on unseen patients
3. Patient P1      what each model says about the project's example patient
4. ML vs DL        same last vitals, different histories
5. Integration     predictions become KB facts, Block 2 reasons with them
6. Dynamic         new events -> models re-run -> new facts -> new conclusions

Only the trained models (learning/models/) and the saved metrics
(learning/reports/) are needed: the raw dataset is not required.
"""
import json
import os
import warnings

warnings.filterwarnings("ignore")

from environment import Environment
from interface import apply_event, load_events, load_scenario
from model import Patient, VitalSigns
from reasoning import Reasoner

from learning import RiskPredictor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "learning", "reports")


def title(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def load(name: str) -> dict:
    with open(os.path.join(REPORTS, name)) as f:
        return json.load(f)


def risk_facts(env, pid=None):
    facts = env.kb.facts("ML_Risk") + env.kb.facts("DL_Risk")
    return [str(f) for f in facts if pid is None or f.args[0] == pid]


def main() -> None:
    # ------------------------------------------------------------------
    title("1. DATA: PhysioNet/CinC Challenge 2019 (two hospitals)")
    d = load("dataset_summary.json")
    print(f"{d['patients']} ICU patients, {d['rows']} patient-hours")
    for h, v in d["hospitals"].items():
        print(f"  hospital {h}: {v['patients']} patients, {v['septic_patients']} septic")
    print("Missing values (%):", ", ".join(f"{k}={v}" for k, v in d["missing_pct"].items()))
    print("\nLabel of hour t (the model only sees hours <= t):")
    print("  Critical = sepsis within 6 h  OR  NEWS2 >= 9 within the next 6 h")
    print("  High / Medium / Low = worst NEWS2 level within the next 6 h")
    print("\nStratified split BY PATIENT (70/15/15), one window every 3 h:")
    for split, s in d["splits"].items():
        print(f"  {split:5} {s['patients']:6} patients {s['windows']:7} windows  "
              + "  ".join(f"{k}={v:.1%}" for k, v in s["class_share"].items())
              + f"  septic={s['septic_patients']:.1%}")

    # ------------------------------------------------------------------
    title("2. EVALUATION on the test patients (never seen during training)")
    ml, dl = load("ml_results.json"), load("dl_results.json")
    print("ML candidates (validation macro-F1):",
          ", ".join(f"{k}={v['macro_f1']:.3f}" for k, v in ml["validation"].items()),
          f"-> kept: {ml['best']}")
    rows = [("Persistence (copy current NEWS2 level)", ml["persistence_test"]),
            (f"ML  {ml['best']} (current vitals)", ml["test"]),
            ("DL  GRU (last 6 hours)", dl["test"])]
    print(f"\n{'':40} {'acc':>6} {'macroF1':>8} {'recall':>7} {'AUROC':>7} {'changing':>9}")
    print(f"{'':40} {'':>6} {'':>8} {'Critic.':>7} {'Critic.':>7} {'macroF1':>9}")
    for name, r in rows:
        s = r["all"]
        auc = f"{s['auroc_critical']:7.3f}" if "auroc_critical" in s else f"{'-':>7}"
        print(f"{name:40} {s['accuracy']:6.3f} {s['macro_f1']:8.3f} "
              f"{s['per_class']['Critical']['recall']:7.3f} {auc} {s['changing_cases']['macro_f1']:9.3f}")
    print("\n'changing' = hours where the future level differs from the current one;")
    print("copying the present is always wrong there, so it measures what was LEARNED.")
    print("\nPer hospital (macro-F1 / AUROC Critical):")
    for h in ("hospital_A", "hospital_B"):
        print(f"  {h}:  ML {ml['test'][h]['macro_f1']:.3f} / {ml['test'][h]['auroc_critical']:.3f}"
              f"    DL {dl['test'][h]['macro_f1']:.3f} / {dl['test'][h]['auroc_critical']:.3f}")
    if ml.get("feature_importance"):
        print("\nRandom Forest feature importance:", ml["feature_importance"])

    # ------------------------------------------------------------------
    title("3. PATIENT P1 OF THE PROJECT STATEMENT")
    env = Environment()
    load_scenario(env, os.path.join(ROOT, "scenario_example.json"))
    pred = RiskPredictor.load()
    p1 = env.patient("P1")
    print("time   HR   SBP  SpO2  Temp")
    for v in p1.history:
        print(f"t{int(v.time):<4} {v.heart_rate:4.0f} {v.systolic:5.0f} {v.spo2:5.0f} {v.temperature:5.1f}")
    out = pred.predict(p1)
    print("\n" + str(out["ML"]) + "\n" + str(out["DL"]))

    # ------------------------------------------------------------------
    title("4. ML vs DL: SAME LAST VITALS, DIFFERENT HISTORIES")
    last = dict(heart_rate=112, systolic=100, diastolic=60, spo2=93, temperature=38.2, resp_rate=22)
    starts = {"deteriorating": dict(heart_rate=80, systolic=125, diastolic=75, spo2=98,
                                    temperature=37.0, resp_rate=15),
              "stable":        last,
              "recovering":    dict(heart_rate=140, systolic=80, diastolic=45, spo2=86,
                                    temperature=39.5, resp_rate=32)}
    print(f"Hour 6 for all three: {last}\n")
    for name, start in starts.items():
        hist = [VitalSigns(time=i + 1, **{k: round(start[k] + (last[k] - start[k]) * i / 5, 1)
                                         for k in last}) for i in range(6)]
        o = pred.predict(Patient(name, age=60, history=hist))
        print(f"{name:14} ML: {o['ML'].level:8} {o['ML'].probability:.2f}    "
              f"DL: {o['DL'].level:8} {o['DL'].probability:.2f}   "
              f"(DL P(Critical)={o['DL'].probs['Critical']:.2f})")
    print("\nThe ML sees one instant: identical answers. The GRU uses the history; it")
    print("learned that a patient who was very unstable in the last hours is still at")
    print("high risk (relapse), even when the last values look better.")

    # ------------------------------------------------------------------
    title("5. INTEGRATION: predictions -> Knowledge Base -> logical reasoning")
    r = Reasoner(env)
    print(f"Before Block 3: CriticalRisk(P1)? {r.holds('CriticalRisk(P1)')}   "
          f"RequiresICU(P1)? {r.holds('RequiresICU(P1)')}   (rules alone say HighRisk)")
    pred.attach(env)                      # writes ML_Risk / DL_Risk for every patient
    print("\nFacts inserted by Block 3 (source ML / DL):")
    for f in risk_facts(env):
        print("  " + f)
    print(f"\nAfter Block 3 (Block 2 rules unchanged: Critical needs p >= 0.8):")
    print(f"  CriticalRisk(P1)? {r.holds('CriticalRisk(P1)')}   RequiresICU(P1)? {r.holds('RequiresICU(P1)')}")
    print("\nWhy does P1 need the ICU?")
    print(r.why("RequiresICU(P1)"))
    print()
    print(r.describe_requirements())

    # ------------------------------------------------------------------
    title("6. DYNAMIC ENVIRONMENT: the models re-run after every change")
    for ev in load_events(os.path.join(ROOT, "scenario_dynamic_events.json")):
        before = set(risk_facts(env))
        apply_event(env, ev)
        who = ev.get("patient") or ev.get("resource") or ev.get("time")
        who = who.get("id") if isinstance(who, dict) else who
        new = sorted(set(risk_facts(env)) - before)
        print(f"\n>>> event: {ev['type']}({who})")
        print("    new ML/DL facts: " + (", ".join(new) if new else "none (no patient data changed)"))
    print()
    print(r.describe_requirements())


if __name__ == "__main__":
    main()
