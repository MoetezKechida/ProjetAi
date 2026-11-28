"""
test_block3.py - Unit tests for Block 3 (no dataset needed).

    python -m unittest learning.test_block3
"""
import io
import os
import unittest
import warnings

import numpy as np

from learning.data_loader import read_psv, VITALS
from learning.dataset import patient_matrix, window_at
from learning.labeling import CRITICAL, HIGH, LOW, MEDIUM, future_labels, news2_level

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nan = np.nan


class TestNEWS2(unittest.TestCase):
    def test_normal_patient_is_low(self):
        score, level = news2_level([75], [120], [98], [37.0], [16])
        self.assertEqual((score[0], level[0]), (0, LOW))

    def test_band_boundaries(self):
        # HR 91 -> 1, SBP 100 -> 2, SpO2 93 -> 2, Temp 38.1 -> 1, RR 21 -> 2  => 8
        score, level = news2_level([91], [100], [93], [38.1], [21])
        self.assertEqual((score[0], level[0]), (8, HIGH))

    def test_single_red_score_is_medium(self):
        score, level = news2_level([75], [120], [91], [37.0], [16])   # SpO2 91 -> 3
        self.assertEqual((score[0], level[0]), (3, MEDIUM))

    def test_missing_counts_as_normal(self):
        score, _ = news2_level([nan], [nan], [nan], [nan], [nan])
        self.assertEqual(score[0], 0)


class TestLabels(unittest.TestCase):
    def test_future_not_present(self):
        level = np.array([HIGH, LOW, LOW, MEDIUM, LOW])
        y = future_labels(np.zeros(5), level, np.zeros(5, int), horizon=2)
        # t=0 looks at t1..t2 (Low, Low): the current High is NOT used
        self.assertEqual(list(y), [LOW, MEDIUM, MEDIUM, -1, -1])

    def test_sepsis_is_critical_even_without_future(self):
        y = future_labels(np.zeros(3), np.array([LOW, LOW, LOW]), np.array([0, 0, 1]), horizon=2)
        self.assertEqual(y[2], CRITICAL)

    def test_severe_future_news2_is_critical(self):
        level = np.array([LOW, HIGH, LOW]); score = np.array([0, 10, 0])   # NEWS2 10 >= 9
        y = future_labels(score, level, np.zeros(3, int), horizon=1)
        self.assertEqual(y[0], CRITICAL)


class TestSplit(unittest.TestCase):
    def test_stratified_split_by_patient(self):
        import pandas as pd
        from learning.dataset import split_patients
        strata = pd.Series(["A-0-1"] * 80 + ["A-1-2"] * 20, index=[f"p{i}" for i in range(100)])
        tr, va, te = split_patients(strata)
        self.assertFalse(tr & va or tr & te or va & te)           # no patient in two sets
        self.assertEqual(len(tr | va | te), 100)
        for part in (tr, va, te):                                 # same 20 % profile everywhere
            self.assertAlmostEqual(strata.loc[sorted(part)].eq("A-1-2").mean(), 0.2, delta=0.05)


class TestPreprocessing(unittest.TestCase):
    MED = dict(zip(VITALS, [80, 120, 70, 97, 37, 18]))

    def test_forward_fill_then_median(self):
        raw = np.array([[nan, 110, nan, 95, nan, nan],
                        [100, nan, nan, nan, 38, nan]], float)
        filled, mask = patient_matrix(raw, self.MED)
        self.assertEqual(list(filled[1]), [100, 110, 70, 95, 38, 18])
        self.assertEqual(list(mask[1]), [1, 0, 0, 0, 1, 0])

    def test_window_padding(self):
        filled, mask = patient_matrix(np.ones((2, 6)), self.MED)
        w = window_at(filled, mask, 1, window=6)
        self.assertEqual(w.shape, (6, 12))
        self.assertTrue((w[:4, 6:] == 0).all())      # padded hours flagged as not measured
        self.assertTrue((w[4:, 6:] == 1).all())

    def test_read_psv_rebuilds_diastolic_from_map(self):
        header = "HR|O2Sat|Temp|SBP|MAP|DBP|Resp|Age|ICULOS|SepsisLabel\n"
        df = read_psv(io.StringIO(header + "90|95|NaN|120|80|NaN|18|70|1|0\n"), "p1")
        self.assertAlmostEqual(df.diastolic[0], 60.0)       # (3*80 - 120) / 2
        self.assertEqual(list(df.columns), ["pid"] + VITALS + ["age", "hour", "sepsis"])


try:
    import torch  # noqa: F401
    HAS_TORCH = os.path.exists(os.path.join(ROOT, "learning", "models", "dl_gru.pt"))
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch or trained models not available")
class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore")
        from learning import RiskPredictor
        cls.pred = RiskPredictor.load()

    def env(self):
        from environment import Environment
        from interface import load_scenario
        e = Environment()
        load_scenario(e, os.path.join(ROOT, "scenario_example.json"))
        return e

    def test_predictions_are_distributions(self):
        out = self.pred.predict(self.env().patient("P1"))
        for p in out.values():
            self.assertAlmostEqual(sum(p.probs.values()), 1.0, places=2)
            self.assertAlmostEqual(p.probability, max(p.probs.values()), places=2)

    def test_p1_more_at_risk_than_p3(self):
        env = self.env()
        p1, p3 = self.pred.predict(env.patient("P1")), self.pred.predict(env.patient("P3"))
        for src in ("ML", "DL"):
            self.assertGreater(p1[src].probs["Critical"], p3[src].probs["Critical"])

    def test_attach_writes_facts_and_refreshes_them(self):
        from model import VitalSigns
        env = self.env()
        self.pred.attach(env)
        self.assertEqual(len(env.kb.query("ML_Risk(P3,?l,?p)")), 1)
        self.assertEqual(len(env.kb.query("DL_Risk(P3,?l,?p)")), 1)
        env.update_vitals("P3", VitalSigns(heart_rate=140, systolic=80, spo2=85,
                                           temperature=39.5, resp_rate=30))
        ml = env.kb.query("ML_Risk(P3,?l,?p)")
        self.assertEqual(len(ml), 1)                       # stale fact replaced, not duplicated
        self.assertNotEqual(ml[0]["?l"], "Low")

    def test_bp_category_without_number(self):
        from model import Patient, VitalSigns
        p = Patient("X", age=50, history=[VitalSigns(heart_rate=80, spo2=97, bp_category="Low")])
        self.assertIsNotNone(self.pred.predict(p))


if __name__ == "__main__":
    unittest.main()
