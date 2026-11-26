"""Block 2 tests. From the repository root:  python -m unittest reasoning.test_block2"""
import os
import unittest

from environment import Environment
from interface import apply_event, load_events, load_scenario
from kb import F
from model import VitalSigns
from reasoning import RULES_FILE, Reasoner
from reasoning.forward import explain, forward_chain
from reasoning.rules import eval_builtin, load_rules, parse_rule, parse_rules, stratify
from reasoning.unify import (Compound, compose, fresh_mapping, rename, substitute, unify,
                             unify_with_trace)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scenario files live here


class TestUnify(unittest.TestCase):
    def test_pdf_example(self):
        # RequiresSpecialist(x,Cardiology) vs RequiresSpecialist(P1,Cardiology) -> {x/P1}
        theta = unify(F("RequiresSpecialist", "?x", "Cardiology"),
                      F("RequiresSpecialist", "P1", "Cardiology"))
        self.assertEqual(theta, {"?x": "P1"})

    def test_two_way(self):
        # variables on both sides: P(?x, B) vs P(A, ?y)
        self.assertEqual(unify(F("P", "?x", "B"), F("P", "A", "?y")), {"?x": "A", "?y": "B"})

    def test_constant_clash_and_arity(self):
        self.assertIsNone(unify(F("Specialty", "D1", "Cardiology"), F("Specialty", "D1", "General")))
        self.assertIsNone(unify(F("P", "?x"), F("P", "A", "B")))
        self.assertIsNone(unify(F("P", "?x"), F("Q", "A")))

    def test_repeated_variable(self):
        self.assertIsNone(unify(F("P", "?x", "?x"), F("P", "A", "B")))
        self.assertEqual(unify(F("P", "?x", "?x"), F("P", "A", "A")), {"?x": "A"})

    def test_chained_bindings(self):
        theta = unify(F("P", "?x", "?y"), F("P", "?y", "A"))
        self.assertEqual(substitute("?x", theta), "A")
        self.assertEqual(substitute("?y", theta), "A")

    def test_occurs_check(self):
        self.assertIsNone(unify("?x", Compound("f", ("?x",))))
        self.assertIsNone(unify(F("P", "?x", "?x"), F("P", "?y", Compound("f", ("?y",)))))
        # nested compound that does unify
        theta = unify(Compound("g", ("?x", Compound("f", ("?y",)))),
                      Compound("g", (Compound("h", ("A",)), Compound("f", ("B",)))))
        self.assertEqual(substitute("?x", theta), Compound("h", ("A",)))
        self.assertEqual(substitute("?y", theta), "B")

    def test_numbers(self):
        self.assertEqual(unify(F("DL_Risk", "P1", "Critical", "?p"),
                               F("DL_Risk", "P1", "Critical", 0.91)), {"?p": 0.91})

    def test_existing_theta_not_mutated(self):
        start = {"?x": "P1"}
        self.assertIsNone(unify(F("P", "?x"), F("P", "P2"), start))
        self.assertEqual(unify(F("P", "?x"), F("P", "P1"), start), {"?x": "P1"})
        self.assertEqual(start, {"?x": "P1"})

    def test_compose_and_rename(self):
        t = compose({"?x": "?y"}, {"?y": "P1"})
        self.assertEqual(substitute(F("R", "?x"), t), F("R", "P1"))
        m = fresh_mapping([F("P", "?x", "?y")], suffix=7)
        self.assertEqual(rename(F("P", "?x", "?y"), m), F("P", "?x_7", "?y_7"))

    def test_trace(self):
        theta, trace = unify_with_trace("?x", Compound("f", ("?x",)))
        self.assertIsNone(theta)
        self.assertTrue(any("occurs check" in line for line in trace))


class TestRules(unittest.TestCase):
    def test_parse_rule(self):
        r = parse_rule("Q6: Active(?x) & not CriticalRisk(?x) & HighRisk(?x) => RequiresEmergencyBed(?x)  # c")
        self.assertEqual(r.id, "Q6")
        self.assertEqual(r.head, F("RequiresEmergencyBed", "?x"))
        self.assertEqual(r.comment, "c")
        # positive literals are moved before negations
        self.assertEqual([str(l) for l in r.body],
                         ["Active(?x)", "HighRisk(?x)", "not CriticalRisk(?x)"])

    def test_builtins(self):
        r = parse_rule("S1: Less(?s,90) & OxygenSaturation(?x,?s) => Hypoxemia(?x)")
        self.assertEqual(r.body[0].atom.predicate, "OxygenSaturation")  # builtin moved after binder
        self.assertTrue(r.body[1].builtin)
        self.assertTrue(eval_builtin(F("Less", 87, 90)))
        self.assertFalse(eval_builtin(F("GreaterEq", 0.75, 0.8)))
        self.assertFalse(eval_builtin(F("Less", "P1", 90)))  # incomparable -> False
        with self.assertRaises(ValueError):
            eval_builtin(F("Less", "?s", 90))

    def test_unsafe_rules_rejected(self):
        for bad in ["X: A(?x) => B(?y)",                 # head variable not bound
                    "X: A(?x) & not B(?y) => C(?x)",     # negated variable not bound
                    "X: A(?x) & Less(?y,3) => C(?x)",    # builtin variable not bound
                    "X: A(?x) & not Less(?x,3) => C(?x)",  # negated builtin
                    "X: A(?x) => Less(?x,3)",            # builtin head
                    "X A(?x) => B(?x)"]:                 # syntax
            with self.assertRaises(ValueError, msg=bad):
                parse_rule(bad)

    def test_stratification(self):
        rules = parse_rules("""
            A1: P(?x) & not Q(?x) => R(?x)
            A2: S(?x) => Q(?x)
            A3: R(?x) => T(?x)
        """)
        strata = stratify(rules)
        self.assertEqual([[r.id for r in s] for s in strata], [["A2"], ["A1", "A3"]])

    def test_negative_cycle_rejected(self):
        with self.assertRaises(ValueError):
            stratify(parse_rules("A1: P(?x) & not B(?x) => B(?x)"))
        with self.assertRaises(ValueError):   # indirect: B -> not C -> B
            stratify(parse_rules("A1: P(?x) & not C(?x) => B(?x)\nA2: B(?x) => C(?x)"))
        # positive recursion is fine
        stratify(parse_rules("A1: Edge(?x,?y) => Path(?x,?y)\nA2: Path(?x,?y) & Edge(?y,?z) => Path(?x,?z)"))

    def test_hospital_rule_file(self):
        rules = load_rules(RULES_FILE)
        self.assertEqual(len(rules), 33)
        strata = stratify(rules)
        self.assertEqual(len(strata), 2)
        # a rule goes to stratum 1 when its HEAD predicate depends on a 'not'
        # (e.g. Q3 shares the head RequiresSpecialist with Q4, which uses 'not')
        self.assertEqual({r.id for r in strata[1]},
                         {"Q3", "Q4", "Q6", "Q7", "Q8", "B1", "B2", "B3", "B4", "B5",
                          "PR1", "PR2", "PR3"})
        self.assertIn("R7", {r.id for r in strata[0]})


def make_env():
    env = Environment()
    load_scenario(env, os.path.join(ROOT, "scenario_example.json"))
    return env


def infer(env):
    return forward_chain(env.kb, stratify(load_rules(RULES_FILE)))


class TestForwardChaining(unittest.TestCase):
    def setUp(self):
        self.env = make_env()
        self.kb = self.env.kb

    def derived(self, pid):
        return {str(f) for f in self.kb.about(pid) if self.kb.source_of(f) == "derived"}

    def test_nothing_derived_before_inference(self):
        self.assertFalse(self.derived("P1"))
        self.assertFalse(self.kb.holds("HighRisk(P1)"))

    def test_scenario_without_ml_dl(self):
        infer(self.env)
        self.assertTrue(self.kb.holds("HighRisk(P1)"))
        self.assertFalse(self.kb.holds("CriticalRisk(P1)"))   # only DL makes P1 critical
        self.assertFalse(self.kb.holds("HighRisk(P3)"))
        self.assertEqual(self.kb.source_of("HighRisk(P1)"), "derived")
        self.assertTrue({"RequiresResource(P1,Bed,Emergency)", "RequiresResource(P1,Doctor,Cardiology)",
                         "RequiresResource(P1,Equipment,ECG)", "Priority(P1,2)"} <= self.derived("P1"))
        for p in ("P2", "P3"):
            self.assertIn(f"RequiresResource({p},Room,Examination)", self.derived(p))
            self.assertIn(f"Priority({p},3)", self.derived(p))

    def test_fixpoint(self):
        first = infer(self.env)
        self.assertTrue(first)
        self.assertEqual(infer(self.env), {})   # running again adds nothing

    def test_dl_evidence_makes_p1_critical(self):
        self.env.record_risk("P1", "DL", "Critical", 0.91)
        just = infer(self.env)
        for f in ("CriticalRisk(P1)", "RequiresICU(P1)", "RequiresResource(P1,Bed,ICU)", "Priority(P1,1)"):
            self.assertTrue(self.kb.holds(f), f)
        self.assertEqual(just[F("CriticalRisk", "P1")].rule.id, "R7")
        # negation as failure: a critical patient does not get an emergency bed / priority 2
        self.assertFalse(self.kb.holds("RequiresEmergencyBed(P1)"))
        self.assertFalse(self.kb.holds("Priority(P1,2)"))

    def test_low_confidence_evidence_ignored(self):
        self.env.record_risk("P1", "DL", "Critical", 0.6)
        infer(self.env)
        self.assertFalse(self.kb.holds("CriticalRisk(P1)"))

    def test_ml_evidence(self):
        self.env.record_risk("P3", "ML", "High", 0.87)
        infer(self.env)
        self.assertTrue(self.kb.holds("HighRisk(P3)"))
        self.assertTrue(self.kb.holds("RequiresSpecialist(P3,General)"))  # Q4: no chest pain
        self.assertFalse(self.kb.holds("RequiresExamRoom(P3)"))

    def test_candidate_doctor_two_variables(self):
        infer(self.env)
        self.assertTrue(self.kb.holds("CandidateDoctor(P1,D1)"))
        self.assertFalse(self.kb.holds("CandidateDoctor(P1,D2)"))   # D2 is General

    def test_explain(self):
        just = infer(self.env)
        text = explain(F("RequiresResource", "P1", "Doctor", "Cardiology"), self.kb, just)
        self.assertIn("[B3]", text)
        self.assertIn("[R1]", text)
        self.assertIn("LowBloodPressure(P1)  [env]", text)
        self.assertIn("not HighRisk(P3)", explain(F("Priority", "P3", 3), self.kb, just))
        self.assertIn("not in the KB", explain(F("HighRisk", "P3"), self.kb, just))

    def test_dynamic_events(self):
        for ev in load_events(os.path.join(ROOT, "scenario_dynamic_events.json")):
            apply_event(self.env, ev)
        infer(self.env)
        self.assertTrue(self.kb.holds("CriticalRisk(P4)"))          # R4 shock pattern, no DL needed
        self.assertTrue(self.kb.holds("RequiresResource(P4,Bed,ICU)"))
        self.assertTrue(self.kb.holds("HighRisk(P2)"))              # R3 after deterioration
        self.assertTrue(self.kb.holds("RequiresResource(P2,Bed,Emergency)"))
        self.assertFalse(self.kb.holds("CriticalRisk(P1)"))


class TestReasoner(unittest.TestCase):
    def setUp(self):
        self.env = make_env()
        self.r = Reasoner(self.env)

    def test_infers_on_creation(self):
        self.assertTrue(self.r.holds("HighRisk(P1)"))
        self.assertTrue(all(self.env.kb.source_of(f) == "derived" for f in self.r.derived()))

    def test_requirements_contract(self):
        self.assertEqual(self.r.requirements(), {
            "P1": {"priority": 2, "urgent_evaluation": True,
                   "needs": [("Bed", "Emergency"), ("Doctor", "Cardiology"), ("Equipment", "ECG")]},
            "P2": {"priority": 3, "urgent_evaluation": False, "needs": [("Room", "Examination")]},
            "P3": {"priority": 3, "urgent_evaluation": False, "needs": [("Room", "Examination")]},
        })
        # every (kind, subtype) is something env.free_resources understands
        for req in self.r.requirements().values():
            for kind, subtype in req["needs"]:
                self.assertTrue(self.env.free_resources(kind, subtype), (kind, subtype))

    def test_dl_evidence_updates_automatically(self):
        self.env.record_risk("P1", "DL", "Critical", 0.91)      # no manual run()
        req = self.r.requirements()["P1"]
        self.assertEqual(req["priority"], 1)
        self.assertIn(("Bed", "ICU"), req["needs"])
        self.assertNotIn(("Bed", "Emergency"), req["needs"])    # retracted, not stale
        self.assertIn("[R7]", self.r.why("CriticalRisk(P1)"))
        self.assertIn("[DL]", self.r.why("CriticalRisk(P1)"))

    def test_truth_maintenance_on_new_vitals(self):
        self.env.record_risk("P1", "DL", "Critical", 0.91)
        self.assertTrue(self.r.holds("RequiresICU(P1)"))
        # P1 improves: new vitals delete the DL fact (Block 1) and no rule fires any more
        self.env.update_vitals("P1", VitalSigns(heart_rate=85, systolic=125, spo2=97, temperature=37.0))
        for f in ("RequiresICU(P1)", "CriticalRisk(P1)", "HighRisk(P1)", "Hypoxemia(P1)"):
            self.assertFalse(self.r.holds(f), f)
        self.assertEqual(self.r.requirements()["P1"]["priority"], 3)
        last = self.r.history[-1]
        self.assertIn("vitals_updated", last.cause)

    def test_resource_availability_changes_candidates(self):
        self.assertTrue(self.r.holds("CandidateDoctor(P1,D1)"))
        self.env.assign("P2", "D1")
        self.assertFalse(self.r.holds("CandidateDoctor(P1,D1)"))
        self.env.release("P2", "D1")
        self.assertTrue(self.r.holds("CandidateDoctor(P1,D1)"))

    def test_discharged_patient_has_no_requirements(self):
        self.env.discharge("P3")
        self.assertNotIn("P3", self.r.requirements())
        self.assertFalse(self.r.derived("P3"))

    def test_dynamic_scenario(self):
        self.env.record_risk("P1", "DL", "Critical", 0.91)
        for ev in load_events(os.path.join(ROOT, "scenario_dynamic_events.json")):
            apply_event(self.env, ev)
        req = self.r.requirements()
        self.assertEqual(list(req)[:2], ["P1", "P4"])           # both critical, sorted first
        self.assertIn(("Bed", "ICU"), req["P4"]["needs"])        # P1 and P4 compete for ICU1
        self.assertEqual(req["P2"]["priority"], 2)               # deteriorated
        self.assertIn(("Bed", "Emergency"), req["P2"]["needs"])
        self.assertEqual(req["P3"]["priority"], 3)

    def test_manual_mode(self):
        env = make_env()
        r = Reasoner(env, auto=False)
        env.record_risk("P1", "DL", "Critical", 0.91)
        self.assertFalse(r.holds("RequiresICU(P1)"))    # not followed automatically
        result = r.run("after DL")
        self.assertTrue(r.holds("RequiresICU(P1)"))
        self.assertIn(F("RequiresICU", "P1"), result.added)
        self.assertIn(F("RequiresEmergencyBed", "P1"), result.removed)


class TestResolution(unittest.TestCase):
    def setUp(self):
        self.env = make_env()
        self.r = Reasoner(self.env)
        self.env.record_risk("P1", "DL", "Critical", 0.91)

    def test_pdf_query_proved(self):
        res = self.r.ask("RequiresICU(P1)")
        self.assertTrue(res)
        self.assertFalse(res.proof[-1].resolvent)                     # ends with the empty clause
        partners = " ".join(s.partner for s in res.proof)
        self.assertIn("[Q2]", partners)
        self.assertIn("DL_Risk(P1,Critical,0.91) [DL]", partners)     # uses the learned evidence
        self.assertIn("PROVED", res.format())

    def test_failed_query(self):
        res = self.r.ask("RequiresICU(P3)")
        self.assertFalse(res)
        self.assertFalse(res.limit_reached)                          # search space exhausted
        self.assertIn("dead end", res.format())
        self.assertIn("NOT PROVED", res.format())

    def test_answer_extraction(self):
        self.assertEqual(self.r.ask("RequiresICU(?x)").answers, [{"?x": "P1"}])
        self.assertEqual(self.r.ask("CandidateDoctor(P1,?d)").answers, [{"?d": "D1"}])
        self.assertEqual(self.r.ask("Bogus(?x)").answers, [])

    def test_does_not_use_derived_facts(self):
        # HighRisk(P1) is in the KB as a derived fact, but the proof rebuilds it from base facts
        res = self.r.ask("HighRisk(P1)")
        self.assertTrue(res)
        self.assertGreater(len(res.proof), 1)
        self.assertTrue(all("[derived]" not in s.partner for s in res.proof))

    def test_low_probability_blocks_proof(self):
        self.env.record_risk("P1", "DL", "Critical", 0.5)   # replaces 0.91
        res = self.r.ask("RequiresICU(P1)")
        self.assertFalse(res)
        self.assertIn("GreaterEq(0.5,0.8) is false", res.format())

    def test_agrees_with_forward_chaining(self):
        """Soundness + completeness check on the Horn part: for every predicate whose
        rules use no 'not', resolution proves exactly the facts forward chaining derived."""
        for ev in load_events(os.path.join(ROOT, "scenario_dynamic_events.json")):
            apply_event(self.env, ev)
        horn_preds = ["Hypoxemia", "Tachycardia", "Fever", "Elderly", "Active", "HighRisk",
                      "CriticalRisk", "RequiresUrgentEvaluation", "RequiresICU", "RequiresEquipment"]
        for pred in horn_preds:
            derived = {f.args[0] for f in self.env.kb.facts(pred)}
            pattern = "?x" if pred != "RequiresEquipment" else "?x,ECG"
            answers = {a["?x"] for a in self.r.ask(f"{pred}({pattern})").answers}
            self.assertEqual(answers, derived, pred)


if __name__ == "__main__":
    unittest.main()
