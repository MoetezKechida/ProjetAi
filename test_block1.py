import json
import unittest

from environment import Environment
from interface import apply_event, dump_scenario, load_events, load_scenario
from kb import F, Fact, KnowledgeBase
from model import Patient, Resource, VitalSigns


def make_env() -> Environment:
    env = Environment()
    load_scenario(env, "scenario_example.json")
    return env


class TestKB(unittest.TestCase):
    def test_parse_and_query(self):
        kb = KnowledgeBase()
        kb.add("HasSymptom(P1,ChestPain)")
        kb.add(F("HasSymptom", "P2", "Fever"))
        self.assertEqual(Fact.parse("Age(P1,72)"), F("Age", "P1", 72))
        self.assertEqual(kb.query("HasSymptom(?x,ChestPain)"), [{"?x": "P1"}])
        self.assertEqual(len(kb.query("HasSymptom(?x,?s)")), 2)
        self.assertTrue(kb.holds("HasSymptom(P2,Fever)"))

    def test_no_duplicates_and_version(self):
        kb = KnowledgeBase()
        self.assertTrue(kb.add("A(x)"))
        v = kb.version
        self.assertFalse(kb.add("A(x)"))
        self.assertEqual(kb.version, v)

    def test_variables_rejected_in_store(self):
        with self.assertRaises(ValueError):
            KnowledgeBase().add("A(?x)")


class TestEnvironment(unittest.TestCase):
    def test_scenario_facts(self):
        env = make_env()
        kb = env.kb
        for f in ["Patient(P1)", "Age(P1,72)", "LowBloodPressure(P1)", "OxygenSaturation(P1,87)",
                  "HasSymptom(P1,ChestPain)", "HasSymptom(P1,ShortnessOfBreath)", "Doctor(D1)",
                  "Specialty(D1,Cardiology)", "Available(D1)", "Bed(ICU1)", "Type(ICU1,ICU)",
                  "PatientWaiting(P1)"]:
            self.assertTrue(kb.holds(f), f)
        self.assertEqual(len(env.patient("P1").history), 5)

    def test_assignment_updates_state_and_kb(self):
        env = make_env()
        env.assign("P1", "D1")
        self.assertTrue(env.kb.holds("DoctorAssigned(P1,D1)"))
        self.assertTrue(env.kb.holds("NotAvailable(D1)"))
        self.assertFalse(env.kb.holds("Available(D1)"))
        self.assertNotIn("D1", env.snapshot().free)
        env.release("P1", "D1")
        self.assertTrue(env.kb.holds("Available(D1)"))
        with self.assertRaises(ValueError):
            env.assign("P2", "D1"); env.assign("P3", "D1")

    def test_state_is_immutable_and_hashable(self):
        env = make_env()
        s0 = env.snapshot()
        s1 = s0.assign("P1", "ICU1")
        self.assertIn("ICU1", s0.free)
        self.assertNotIn("ICU1", s1.free)
        self.assertEqual(len({s0, s1, s0.assign("P1", "ICU1")}), 2)
        with self.assertRaises(ValueError):
            s1.assign("P2", "ICU1")

    def test_evidence_invalidated_when_vitals_change(self):
        env = make_env()
        env.record_risk("P1", "DL", "Critical", 0.91)
        self.assertTrue(env.kb.holds("DL_Risk(P1,Critical,0.91)"))
        env.update_vitals("P1", VitalSigns(heart_rate=140, systolic=80, spo2=84))
        self.assertEqual(env.kb.facts("DL_Risk"), [])
        self.assertTrue(env.kb.holds("HeartRate(P1,140)"))
        self.assertFalse(env.kb.holds("HeartRate(P1,132)"))

    def test_resource_failure_breaks_assignment(self):
        env = make_env()
        seen = []
        env.subscribe(lambda e: seen.append(e.kind))
        env.assign("P1", "ICU1")
        env.set_resource_available("ICU1", False, "failure")
        self.assertIn("assignment_broken", seen)
        self.assertEqual(env.patient("P1").assigned, set())
        self.assertTrue(env.kb.holds("NotAvailable(ICU1)"))
        self.assertFalse(env.kb.holds("BedAssigned(P1,ICU1)"))

    def test_new_patient_event(self):
        env = make_env()
        apply_event(env, {"type": "add_patient", "patient": {"id": "P9", "age": 50, "symptoms": ["fainting"]}})
        self.assertTrue(env.kb.holds("HasSymptom(P9,Fainting)"))
        self.assertIn("P9", env.snapshot().waiting)

    def test_record_risk_replaces_old(self):
        env = make_env()
        env.record_risk("P1", "ML", "High", 0.87)
        env.record_risk("P1", "ML", "Critical", 0.9)
        self.assertEqual(len(env.kb.facts("ML_Risk")), 1)
        self.assertEqual(env.kb.source_of("ML_Risk(P1,Critical,0.9)"), "ML")

    def test_validation(self):
        with self.assertRaises(ValueError):
            VitalSigns(spo2=150)
        with self.assertRaises(ValueError):
            Resource("X1", "Spaceship")
        with self.assertRaises(ValueError):
            apply_event(make_env(), {"type": "nonsense"})

    def test_roundtrip(self):
        env = make_env()
        env2 = Environment()
        load_scenario(env2, json.loads(json.dumps(dump_scenario(env))))
        self.assertEqual([str(f) for f in env.kb.facts()], [str(f) for f in env2.kb.facts()])

    def test_dynamic_events_file(self):
        env = make_env()
        for ev in load_events("scenario_dynamic_events.json"):
            apply_event(env, ev)
        self.assertEqual(env.clock, 10)
        self.assertTrue(env.kb.holds("NotAvailable(ICU1)"))
        self.assertTrue(env.kb.holds("Patient(P4)"))


if __name__ == "__main__":
    unittest.main()
