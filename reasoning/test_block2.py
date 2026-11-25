"""Block 2 tests. From the repository root:  python -m unittest reasoning.test_block2"""
import os
import unittest

from environment import Environment
from interface import apply_event, load_events, load_scenario
from kb import F
from reasoning import RULES_FILE
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


if __name__ == "__main__":
    unittest.main()
