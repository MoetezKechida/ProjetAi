"""
reasoning - Block 2: Logical Reasoning, Deduction, Unification and Resolution.

Run everything from the repository root (Block 1 modules such as kb.py and
environment.py live there):

    python -m unittest reasoning.test_block2

Modules
    unify.py            unification with occurs check, substitutions, renaming
    rules.py            rule language, builtins, safety check, stratification
    rules_hospital.txt  the rule base (editable without touching the code)
    forward.py          forward chaining (deduction) + justifications + explain()
    resolution.py       resolution refutation prover with traces and answer extraction
    reasoner.py         Reasoner facade: truth maintenance, why(), ask(), requirements()
"""
import os

RULES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules_hospital.txt")

from .forward import Justification, explain, forward_chain  # noqa: E402
from .reasoner import InferenceResult, Reasoner  # noqa: E402
from .resolution import ProofResult, prove  # noqa: E402
from .rules import Rule, load_rules, stratify  # noqa: E402
from .unify import Compound, substitute, unify, unify_with_trace  # noqa: E402
