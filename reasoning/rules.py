"""
rules.py - Rule language, builtins and stratification (Block 2).

Rule syntax (one per line in a .txt file, '#' starts a comment):

    ID: Lit1 & Lit2 & ... => Head
    Q6: Active(?x) & HighRisk(?x) & not CriticalRisk(?x) => RequiresEmergencyBed(?x)

A literal is an atom  Pred(args)  optionally preceded by  'not'
(negation as failure).  Atoms whose predicate is a BUILTIN (Less, GreaterEq...)
are not looked up in the KB: they are evaluated once their arguments are ground.

Safety (range restriction): every variable of the head, of a builtin or of a
negated literal must appear in a positive non-builtin body literal. This
guarantees that forward chaining only produces ground facts and that 'not'
and builtins are always evaluated on ground atoms.

Stratification: a rule set with 'not' is only meaningful if no predicate
depends negatively on itself (directly or through other rules). `stratify`
computes the strata; forward chaining runs them in order so that, when a rule
asks "not CriticalRisk(P1)", every CriticalRisk fact has already been derived.
"""
from __future__ import annotations

import operator
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

from kb import Fact
from .unify import variables

# --------------------------------------------------------------------------
# Builtins
# --------------------------------------------------------------------------
BUILTINS = {
    "Less": operator.lt,
    "LessEq": operator.le,
    "Greater": operator.gt,
    "GreaterEq": operator.ge,
    "Eq": operator.eq,
    "Neq": operator.ne,
}


def is_builtin(atom: Fact) -> bool:
    return atom.predicate in BUILTINS


def eval_builtin(atom: Fact) -> bool:
    """Evaluate a GROUND builtin atom. Comparing incomparable values is False."""
    if not atom.is_ground:
        raise ValueError(f"Builtin {atom} evaluated before its variables were bound")
    if len(atom.args) != 2:
        raise ValueError(f"Builtin {atom} must have 2 arguments")
    try:
        return bool(BUILTINS[atom.predicate](*atom.args))
    except TypeError:
        return False


# --------------------------------------------------------------------------
# Literals and rules
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Literal:
    atom: Fact
    negated: bool = False

    @property
    def builtin(self) -> bool:
        return is_builtin(self.atom)

    def __str__(self) -> str:
        return ("not " if self.negated else "") + str(self.atom)

    __repr__ = __str__


@dataclass
class Rule:
    id: str
    body: List[Literal]
    head: Fact
    comment: str = ""
    stratum: int = field(default=0, compare=False)

    @property
    def has_negation(self) -> bool:
        return any(l.negated for l in self.body)

    def __str__(self) -> str:
        return f"{self.id}: {' & '.join(map(str, self.body))} => {self.head}"

    __repr__ = __str__


_RULE_RE = re.compile(r"^\s*(\w+)\s*:\s*(.+?)\s*=>\s*(.+?)\s*$")


def _split_conjunction(text: str) -> List[str]:
    """Split on '&' at parenthesis depth 0."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "&" and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    if any(not p for p in parts):
        raise ValueError(f"Empty literal in {text!r}")
    return parts


def parse_literal(text: str) -> Literal:
    text = text.strip()
    negated = False
    if text.startswith("not ") or text.startswith("not\t"):
        negated, text = True, text[4:].strip()
    atom = Fact.parse(text)
    if negated and is_builtin(atom):
        raise ValueError(f"Negated builtin not allowed (use the opposite comparison): {text}")
    return Literal(atom, negated)


def parse_rule(line: str) -> Rule:
    """Parse 'ID: A(?x) & not B(?x) => C(?x)  # comment'."""
    comment = ""
    if "#" in line:
        line, comment = line.split("#", 1)
        comment = comment.strip()
    m = _RULE_RE.match(line)
    if not m:
        raise ValueError(f"Cannot parse rule: {line!r}")
    rid, body_txt, head_txt = m.groups()
    body = [parse_literal(t) for t in _split_conjunction(body_txt)]
    head = Fact.parse(head_txt)
    if is_builtin(head):
        raise ValueError(f"{rid}: a builtin cannot be a rule head")
    rule = Rule(rid, _order_body(body), head, comment)
    _check_safe(rule)
    return rule


def _order_body(body: List[Literal]) -> List[Literal]:
    """Positive KB literals first (they bind variables), then builtins, then
    negations. Forward chaining and resolution both process literals left to
    right, so builtins/negations are reached only when they are ground."""
    pos = [l for l in body if not l.negated and not l.builtin]
    blt = [l for l in body if l.builtin]
    neg = [l for l in body if l.negated]
    return pos + blt + neg


def _check_safe(rule: Rule) -> None:
    bound = set()
    for l in rule.body:
        if not l.negated and not l.builtin:
            bound.update(variables(l.atom))
    for what, atom in [("head", rule.head)] + [("literal " + str(l), l.atom)
                                               for l in rule.body if l.negated or l.builtin]:
        free = [v for v in variables(atom) if v not in bound]
        if free:
            raise ValueError(f"{rule.id}: unsafe rule, variable(s) {free} in {what} "
                             f"do not appear in a positive body literal")


def parse_rules(text: str) -> List[Rule]:
    rules, ids = [], set()
    for n, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        try:
            r = parse_rule(raw)
        except ValueError as e:
            raise ValueError(f"line {n}: {e}") from None
        if r.id in ids:
            raise ValueError(f"line {n}: duplicate rule id {r.id}")
        ids.add(r.id)
        rules.append(r)
    return rules


def load_rules(path: Union[str, Path]) -> List[Rule]:
    return parse_rules(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Stratification
# --------------------------------------------------------------------------
def predicate_strata(rules: List[Rule]) -> Dict[str, int]:
    """stratum(head) >= stratum(p) for every positive body predicate p,
    stratum(head) >  stratum(q) for every negated body predicate q.
    Base predicates (never a head) are in stratum 0.
    Raises ValueError if a predicate depends negatively on itself."""
    preds = {r.head.predicate for r in rules}
    for r in rules:
        preds.update(l.atom.predicate for l in r.body if not l.builtin)
    s = {p: 0 for p in preds}
    limit = len(preds)
    changed = True
    while changed:
        changed = False
        for r in rules:
            h = r.head.predicate
            for l in r.body:
                if l.builtin:
                    continue
                need = s[l.atom.predicate] + (1 if l.negated else 0)
                if s[h] < need:
                    s[h] = need
                    changed = True
                    if s[h] > limit:
                        raise ValueError(
                            f"Rules are not stratifiable: '{h}' depends negatively on itself "
                            f"(cycle through 'not', see rule {r.id})")
    return s


def stratify(rules: List[Rule]) -> List[List[Rule]]:
    """Group rules by the stratum of their head predicate (sets rule.stratum)."""
    s = predicate_strata(rules)
    if not rules:
        return []
    n = max(s[r.head.predicate] for r in rules) + 1
    strata: List[List[Rule]] = [[] for _ in range(n)]
    for r in rules:
        r.stratum = s[r.head.predicate]
        strata[r.stratum].append(r)
    return [st for st in strata if st]


def describe(rules: List[Rule]) -> str:
    lines = []
    for i, st in enumerate(stratify(rules)):
        lines.append(f"--- stratum {i} ---")
        for r in st:
            lines.append(f"  {r}" + (f"    # {r.comment}" if r.comment else ""))
    return "\n".join(lines)
