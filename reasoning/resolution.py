"""
resolution.py - Proof by resolution refutation (Block 2).

To prove a query Q from the knowledge base KB we show that  KB AND NOT Q  is
unsatisfiable: we add the clause {~Q} and resolve until the EMPTY CLAUSE []
is derived.

Clausal form (all variables implicitly universally quantified)
    base fact   HasSymptom(P1,ChestPain)            -> { HasSymptom(P1,ChestPain) }
    rule        A(?x) & B(?x) => H(?x)               -> { ~A(?x) | ~B(?x) | H(?x) }
    negated goal  RequiresICU(P1)                    -> { ~RequiresICU(P1) }

Choices (and why)
  * Only BASE facts (source env / ML / DL) are used, never "derived" facts:
    otherwise RequiresICU(P1) would already be a unit clause and the proof
    would be trivial. Resolution must rebuild the chain from the evidence.
  * Only HORN rules without 'not' are used. Negation as failure is not
    classical negation, so rules such as Q6 (... & not CriticalRisk(?x))
    cannot be turned into sound clauses. Those conclusions are handled by
    stratified forward chaining instead.
  * Strategy = linear input resolution with SET OF SUPPORT: every step
    resolves the most recent clause (a descendant of the negated goal) with
    an input clause (a fact or a rule). The leftmost negative literal is
    selected (SLD resolution). This is complete for Horn clauses and avoids
    the combinatorial explosion of resolving every pair of clauses.
  * Rules are STANDARDIZED APART (fresh variables ?x_7) at every use.
  * Duplicate literals in a resolvent are merged (factoring).
  * Builtin literals (~GreaterEq(0.91,0.8)) are evaluated once ground: a TRUE
    comparison is removed from the clause (it resolves with the "theory of
    arithmetic"), a FALSE one closes the branch.
  * ANSWER EXTRACTION: for a query with variables, RequiresICU(?x), the goal
    clause is {~RequiresICU(?x) | Answer(?x)}; reaching {Answer(P1)} instead of
    [] gives the binding ?x = P1.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from kb import Fact, KnowledgeBase, as_fact

from .forward import DERIVED
from .rules import Rule, eval_builtin, is_builtin
from .unify import Theta, fmt_theta, fresh_mapping, rename, substitute, unify, variables

Lit = Tuple[bool, Fact]          # (positive?, atom)
Clause = Tuple[Lit, ...]
ANSWER = "Answer"


def fmt_lit(lit: Lit) -> str:
    pos, atom = lit
    return ("" if pos else "~") + str(atom)


def fmt_clause(clause: Sequence[Lit]) -> str:
    return "[] (empty clause)" if not clause else "{" + " | ".join(fmt_lit(l) for l in clause) + "}"


def rule_clause(rule: Rule) -> Clause:
    return tuple((False, l.atom) for l in rule.body) + ((True, rule.head),)


def _dedupe(clause: Sequence[Lit]) -> Clause:
    seen, out = set(), []
    for lit in clause:
        if lit not in seen:
            seen.add(lit)
            out.append(lit)
    return tuple(out)


# --------------------------------------------------------------------------
# Result objects
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    clause: Clause            # clause before the step
    selected: Lit             # literal resolved upon
    partner: str              # "[Q2] {...}"  or  "fact X [env]"  or  "builtin"
    theta: Optional[Theta]
    resolvent: Clause

    def format(self, n: int) -> str:
        lines = [f"{n:2}. {fmt_clause(self.clause)}"]
        if self.partner == "builtin":
            lines.append(f"    builtin {self.selected[1]} is true -> literal removed")
        else:
            lines.append(f"    resolve {fmt_lit(self.selected)} with {self.partner}")
            lines.append(f"    theta = {fmt_theta(self.theta)}")
        lines.append(f"    => {fmt_clause(self.resolvent)}")
        return "\n".join(lines)


@dataclass
class ProofResult:
    query: Fact
    proved: bool
    answers: List[Dict[str, object]] = field(default_factory=list)
    proof: List[Step] = field(default_factory=list)      # steps of the FIRST proof found
    search_log: List[str] = field(default_factory=list)  # every attempt (for failures)
    explored: int = 0
    n_facts: int = 0
    n_rules: int = 0
    n_excluded: int = 0
    limit_reached: bool = False

    def __bool__(self) -> bool:
        return self.proved

    def format(self, show_search: Optional[bool] = None) -> str:
        """Readable resolution trace. By default the search tree is shown only
        when the proof failed (the successful path is shown otherwise)."""
        show_search = (not self.proved) if show_search is None else show_search
        vars_ = variables(self.query)
        goal = [(False, self.query)] + ([(True, Fact(ANSWER, tuple(vars_)))] if vars_ else [])
        out = [f"Query: {self.query}?",
               f"Negated goal (set of support): {fmt_clause(goal)}",
               f"Input clauses: {self.n_facts} base facts + {self.n_rules} Horn rules "
               f"({self.n_excluded} rules with 'not' excluded)"]
        if self.proved:
            out.append("Proof:")
            out += [s.format(i) for i, s in enumerate(self.proof, start=1)]
        if show_search:
            out.append("Search (every resolution attempted):")
            out += ["  " + line for line in self.search_log]
        if self.proved and vars_:
            out.append("Answers: " + "; ".join(fmt_theta(a) for a in self.answers))
            out.append(f"=> {self.query} holds for every answer above (answer extraction)")
        elif self.proved:
            out.append(f"=> Empty clause derived: KB AND ~{self.query} is unsatisfiable, "
                       f"so {self.query} is PROVED")
        else:
            why = " (search limit reached)" if self.limit_reached else ""
            out.append(f"=> No empty clause can be derived{why}: {self.query} is NOT PROVED "
                       f"from the current knowledge")
        out.append(f"   ({self.explored} resolution steps explored)")
        return "\n".join(out)

    __str__ = format


class _Limit(Exception):
    pass


# --------------------------------------------------------------------------
# Prover
# --------------------------------------------------------------------------
def prove(query: Union[Fact, str], kb: KnowledgeBase, rules: List[Rule],
          find_all: Optional[bool] = None, max_depth: int = 60,
          max_steps: int = 20000, max_log: int = 400) -> ProofResult:
    """Prove `query` by resolution refutation from the base facts of kb and the
    Horn rules. For a query with variables all answers are collected
    (find_all defaults to True), otherwise the search stops at the first proof."""
    goal = as_fact(query)
    vars_ = variables(goal)
    if find_all is None:
        find_all = bool(vars_)

    facts: Dict[str, List[Fact]] = defaultdict(list)
    for f in kb.facts():
        if kb.source_of(f) != DERIVED:
            facts[f.predicate].append(f)
    horn = [r for r in rules if not r.has_negation]
    by_head: Dict[str, List[Rule]] = defaultdict(list)
    for r in horn:
        by_head[r.head.predicate].append(r)

    res = ProofResult(goal, False, n_facts=sum(map(len, facts.values())),
                      n_rules=len(horn), n_excluded=len(rules) - len(horn))
    counter = [0]

    def log(depth: int, text: str) -> None:
        if len(res.search_log) < max_log:
            res.search_log.append("  " * depth + text)
        elif len(res.search_log) == max_log:
            res.search_log.append("... (search log truncated)")

    def record_answer(clause: Clause, path: Tuple[Step, ...]) -> bool:
        """Called when only the Answer literal (or nothing) remains. True = stop."""
        if vars_:
            ans_atom = next(a for p, a in clause if p and a.predicate == ANSWER)
            answer = dict(zip(vars_, ans_atom.args))
            if answer not in res.answers:
                res.answers.append(answer)
        if not res.proved:
            res.proved, res.proof = True, list(path)
        return not find_all

    def select(clause: Clause) -> Optional[int]:
        """Leftmost negative literal that can be processed now (a builtin must be ground)."""
        for i, (pos, atom) in enumerate(clause):
            if not pos and (not is_builtin(atom) or atom.is_ground):
                return i
        return None

    def dfs(clause: Clause, depth: int, path: Tuple[Step, ...]) -> bool:
        if not any(not pos for pos, _ in clause):
            log(depth, f"{fmt_clause(clause)}  <- empty clause / answer found")
            return record_answer(clause, path)
        if depth >= max_depth:
            log(depth, "depth limit -> abandon branch")
            return False
        i = select(clause)
        if i is None:
            log(depth, f"{fmt_clause(clause)}: unbound builtin, dead end")
            return False
        lit = clause[i]
        atom = lit[1]
        rest = clause[:i] + clause[i + 1:]

        if is_builtin(atom):
            res.explored += 1
            if not eval_builtin(atom):
                log(depth, f"builtin {atom} is false -> dead end")
                return False
            log(depth, f"builtin {atom} is true -> {fmt_clause(rest)}")
            step = Step(clause, lit, "builtin", None, rest)
            return dfs(rest, depth + 1, path + (step,))

        candidates = 0
        for fact in facts.get(atom.predicate, ()):
            theta = unify(atom, fact)
            if theta is None:
                continue
            candidates += 1
            res.explored += 1
            if res.explored > max_steps:
                raise _Limit
            resolvent = _dedupe([(p, substitute(a, theta)) for p, a in rest])
            log(depth, f"{fmt_lit(lit)} + fact {fact} [{kb.source_of(fact)}] "
                       f"{fmt_theta(theta)} -> {fmt_clause(resolvent)}")
            step = Step(clause, lit, f"fact {fact} [{kb.source_of(fact)}]", theta, resolvent)
            if dfs(resolvent, depth + 1, path + (step,)):
                return True

        for rule in by_head.get(atom.predicate, ()):
            counter[0] += 1
            mapping = fresh_mapping([l.atom for l in rule.body] + [rule.head], counter[0])
            rc = tuple((p, rename(a, mapping)) for p, a in rule_clause(rule))
            theta = unify(atom, rc[-1][1])
            if theta is None:
                continue
            candidates += 1
            res.explored += 1
            if res.explored > max_steps:
                raise _Limit
            resolvent = _dedupe([(p, substitute(a, theta)) for p, a in rc[:-1] + rest])
            log(depth, f"{fmt_lit(lit)} + [{rule.id}] {fmt_theta(theta)} -> {fmt_clause(resolvent)}")
            step = Step(clause, lit, f"[{rule.id}] {fmt_clause(rc)}", theta, resolvent)
            if dfs(resolvent, depth + 1, path + (step,)):
                return True

        if not candidates:
            log(depth, f"{fmt_lit(lit)}: no fact or rule head unifies -> dead end")
        return False

    start: Clause = ((False, goal),) + (((True, Fact(ANSWER, tuple(vars_))),) if vars_ else ())
    try:
        dfs(start, 0, ())
    except _Limit:
        res.limit_reached = True
    return res
