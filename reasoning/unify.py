"""
unify.py - First-order unification (Block 2).

Terms
  variable  : a string starting with '?'            e.g. ?x
  constant  : any other str / int / float           e.g. P1, Cardiology, 0.91
  compound  : Compound(functor, args)  or  a kb.Fact (predicate, args)
              e.g. f(?x)   RequiresSpecialist(?x,Cardiology)

A substitution (theta) is a dict {variable: term}. Bindings may be chained
(?x -> ?y -> P1); `substitute` follows chains to the end.

Algorithm: Robinson / Russell & Norvig UNIFY with occurs check. Unlike
kb.match (one-way, pattern vs ground fact), unification is symmetric:
both sides may contain variables.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional

from kb import Fact, is_var

Theta = Dict[str, object]


@dataclass(frozen=True)
class Compound:
    """A function term f(t1,...,tn). Only needed to show the occurs check;
    hospital facts are flat."""

    functor: str
    args: tuple = ()

    def __str__(self) -> str:
        return f"{self.functor}({','.join(str(a) for a in self.args)})"

    __repr__ = __str__


def _is_compound(t) -> bool:
    return isinstance(t, (Compound, Fact))


def _parts(t):
    """(functor, args) of a compound term or a Fact."""
    return (t.predicate, t.args) if isinstance(t, Fact) else (t.functor, t.args)


def _rebuild(t, args: tuple):
    return Fact(t.predicate, args) if isinstance(t, Fact) else Compound(t.functor, args)


def fmt_theta(theta: Optional[Theta]) -> str:
    if theta is None:
        return "FAIL"
    return "{" + ", ".join(f"{k}/{v}" for k, v in theta.items()) + "}"


# --------------------------------------------------------------------------
# Substitution
# --------------------------------------------------------------------------
def substitute(term, theta: Theta):
    """Apply theta to a term, following chained bindings."""
    if is_var(term):
        seen = set()
        while is_var(term) and term in theta and term not in seen:
            seen.add(term)
            term = theta[term]
        return substitute(term, theta) if _is_compound(term) else term
    if _is_compound(term):
        _, args = _parts(term)
        return _rebuild(term, tuple(substitute(a, theta) for a in args))
    return term


def compose(theta1: Theta, theta2: Theta) -> Theta:
    """theta1 then theta2:  substitute(t, compose(a,b)) == substitute(substitute(t,a), b)."""
    out = {v: substitute(t, theta2) for v, t in theta1.items()}
    for v, t in theta2.items():
        out.setdefault(v, t)
    return {v: t for v, t in out.items() if v != t}


def variables(term) -> List[str]:
    """Variables of a term, in order of first appearance."""
    if is_var(term):
        return [term]
    if _is_compound(term):
        out: List[str] = []
        for a in _parts(term)[1]:
            for v in variables(a):
                if v not in out:
                    out.append(v)
        return out
    return []


# --------------------------------------------------------------------------
# Unification
# --------------------------------------------------------------------------
def occurs_check(var: str, term, theta: Theta) -> bool:
    """True if `var` occurs inside `term` (after applying theta)."""
    if var == term:
        return True
    if is_var(term) and term in theta:
        return occurs_check(var, theta[term], theta)
    if _is_compound(term):
        return any(occurs_check(var, a, theta) for a in _parts(term)[1])
    return False


def unify(x, y, theta: Optional[Theta] = None,
          trace: Optional[List[str]] = None) -> Optional[Theta]:
    """Most general unifier of x and y that extends theta, or None on failure.
    theta is never mutated. If `trace` is a list, one line per step is appended."""
    return _unify(dict(theta or {}), x, y, trace, 0)


def _unify(theta: Theta, x, y, trace, depth) -> Optional[Theta]:
    pad = "  " * (depth + 1)
    if x == y:
        return theta
    if is_var(x):
        return _unify_var(x, y, theta, trace, depth)
    if is_var(y):
        return _unify_var(y, x, theta, trace, depth)
    if _is_compound(x) and _is_compound(y):
        (fx, ax), (fy, ay) = _parts(x), _parts(y)
        if fx != fy or len(ax) != len(ay):
            if trace is not None:
                trace.append(f"{pad}{x} vs {y}: functor/arity mismatch -> FAIL")
            return None
        for a, b in zip(ax, ay):
            theta = _unify(theta, a, b, trace, depth + 1)
            if theta is None:
                return None
        return theta
    if trace is not None:
        trace.append(f"{pad}{x} vs {y}: different constants -> FAIL")
    return None


def _unify_var(var: str, t, theta: Theta, trace, depth) -> Optional[Theta]:
    pad = "  " * (depth + 1)
    if var in theta:
        return _unify(theta, theta[var], t, trace, depth)
    if is_var(t) and t in theta:
        return _unify(theta, var, theta[t], trace, depth)
    if occurs_check(var, t, theta):
        if trace is not None:
            trace.append(f"{pad}occurs check: {var} occurs in {substitute(t, theta)} -> FAIL")
        return None
    if trace is not None:
        trace.append(f"{pad}bind {var}/{t}")
    new = dict(theta)
    new[var] = t
    return new


def unify_with_trace(x, y, theta: Optional[Theta] = None):
    """Convenience: returns (theta_or_None, list_of_trace_lines)."""
    trace: List[str] = []
    result = _unify(dict(theta or {}), x, y, trace, 0)
    trace.insert(0, f"unify({x}, {y})")
    trace.append(f"=> {fmt_theta(result)}")
    return result, trace


# --------------------------------------------------------------------------
# Standardizing apart
# --------------------------------------------------------------------------
_counter = itertools.count(1)


def fresh_suffix() -> int:
    return next(_counter)


def rename(term, mapping: Dict[str, str]):
    """Rename variables according to mapping (used by standardize_apart)."""
    if is_var(term):
        return mapping.get(term, term)
    if _is_compound(term):
        _, args = _parts(term)
        return _rebuild(term, tuple(rename(a, mapping) for a in args))
    return term


def fresh_mapping(terms, suffix: Optional[int] = None) -> Dict[str, str]:
    """Map every variable in `terms` to a fresh one: ?x -> ?x_7."""
    n = fresh_suffix() if suffix is None else suffix
    out: Dict[str, str] = {}
    for t in terms:
        for v in variables(t):
            out.setdefault(v, f"{v}_{n}")
    return out
