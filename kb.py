"""
kb.py - Symbolic Knowledge Base (Block 1, reused by Blocks 2-6).

Facts are first-order ground atoms such as  HeartRate(P1,125)  or
HasSymptom(P1,ChestPain).  Variables are written with a leading '?'
(e.g. ?x) and are only allowed in *patterns* used for querying.

Every fact remembers its SOURCE so that reasoning traces can say where a
fact came from:
    "env"      -> produced by the environment model (user input / state)
    "ML"       -> Machine Learning prediction (Block 3)
    "DL"       -> Deep Learning prediction (Block 3)
    "derived"  -> deduced by the reasoning engine (Block 2)
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Union

_FACT_RE = re.compile(r"^\s*(\w+)\s*\((.*)\)\s*$")


def is_var(term) -> bool:
    """A term is a variable if it is a string starting with '?'."""
    return isinstance(term, str) and term.startswith("?")


def _coerce(token: str):
    token = token.strip()
    for cast in (int, float):
        try:
            return cast(token)
        except ValueError:
            pass
    return token


@dataclass(frozen=True)
class Fact:
    """An atom Predicate(arg1, ..., argN). Immutable and hashable."""

    predicate: str
    args: tuple = ()

    def __str__(self) -> str:
        return f"{self.predicate}({','.join(str(a) for a in self.args)})"

    __repr__ = __str__

    @property
    def arity(self) -> int:
        return len(self.args)

    @property
    def is_ground(self) -> bool:
        return not any(is_var(a) for a in self.args)

    @staticmethod
    def parse(text: str) -> "Fact":
        """Parse 'HasSymptom(P1,ChestPain)' into a Fact."""
        m = _FACT_RE.match(text)
        if not m:
            raise ValueError(f"Cannot parse fact/pattern: {text!r}")
        inner = m.group(2).strip()
        args = tuple(_coerce(a) for a in inner.split(",")) if inner else ()
        return Fact(m.group(1), args)


def F(predicate: str, *args) -> Fact:
    """Shorthand constructor: F('Age', 'P1', 72)."""
    return Fact(predicate, tuple(args))


def as_fact(x: Union[Fact, str]) -> Fact:
    return x if isinstance(x, Fact) else Fact.parse(x)


def match(pattern: Fact, fact: Fact, bindings: Optional[dict] = None) -> Optional[dict]:
    """
    One-way matching of a pattern (may contain ?variables) against a ground
    fact. Returns the extended bindings or None. (Full two-way unification of
    rules is the job of Block 2.)
    """
    if pattern.predicate != fact.predicate or pattern.arity != fact.arity:
        return None
    b = dict(bindings or {})
    for p, a in zip(pattern.args, fact.args):
        if is_var(p):
            if p in b and b[p] != a:
                return None
            b[p] = a
        elif p != a:
            return None
    return b


class KnowledgeBase:
    """A set of ground facts with sources, indexing, change log and listeners."""

    def __init__(self) -> None:
        self._facts: Dict[Fact, str] = {}
        self._by_pred: Dict[str, set] = defaultdict(set)
        self._listeners: List[Callable[[str, Fact, str], None]] = []
        self.version = 0  # incremented on every real change

    # ---- modification ---------------------------------------------------
    def add(self, fact: Union[Fact, str], source: str = "env") -> bool:
        """Add a ground fact. Returns True if the KB actually changed."""
        fact = as_fact(fact)
        if not fact.is_ground:
            raise ValueError(f"Only ground facts can be stored, got {fact}")
        if fact in self._facts:
            return False
        self._facts[fact] = source
        self._by_pred[fact.predicate].add(fact)
        self.version += 1
        self._notify("add", fact, source)
        return True

    def retract(self, fact: Union[Fact, str]) -> bool:
        fact = as_fact(fact)
        if fact not in self._facts:
            return False
        source = self._facts.pop(fact)
        self._by_pred[fact.predicate].discard(fact)
        self.version += 1
        self._notify("retract", fact, source)
        return True

    def retract_where(self, pattern: Union[Fact, str], source: Optional[str] = None) -> List[Fact]:
        """Retract every fact matching the pattern (optionally only from a source)."""
        pattern = as_fact(pattern)
        removed = [f for f in list(self._by_pred.get(pattern.predicate, ()))
                   if match(pattern, f) is not None and (source is None or self._facts[f] == source)]
        for f in removed:
            self.retract(f)
        return removed

    def retract_subject(self, subject, source: Optional[str] = None) -> List[Fact]:
        """Retract all facts whose first argument is `subject`."""
        removed = [f for f in list(self._facts)
                   if f.args and f.args[0] == subject and (source is None or self._facts[f] == source)]
        for f in removed:
            self.retract(f)
        return removed

    # ---- querying -------------------------------------------------------
    def holds(self, fact: Union[Fact, str]) -> bool:
        return as_fact(fact) in self._facts

    def query(self, pattern: Union[Fact, str]) -> List[dict]:
        """All variable bindings under which the pattern matches a stored fact.
        A ground pattern returns [{}] if it holds, [] otherwise."""
        pattern = as_fact(pattern)
        out = []
        for f in sorted(self._by_pred.get(pattern.predicate, ()), key=str):
            b = match(pattern, f)
            if b is not None:
                out.append(b)
        return out

    def facts(self, predicate: Optional[str] = None) -> List[Fact]:
        if predicate is None:
            return sorted(self._facts, key=str)
        return sorted(self._by_pred.get(predicate, ()), key=str)

    def about(self, subject) -> List[Fact]:
        """All facts mentioning `subject` as first argument."""
        return sorted((f for f in self._facts if f.args and f.args[0] == subject), key=str)

    def source_of(self, fact: Union[Fact, str]) -> Optional[str]:
        return self._facts.get(as_fact(fact))

    def __contains__(self, fact) -> bool:
        return self.holds(fact)

    def __len__(self) -> int:
        return len(self._facts)

    def __iter__(self) -> Iterable[Fact]:
        return iter(self.facts())

    # ---- listeners ------------------------------------------------------
    def subscribe(self, callback: Callable[[str, Fact, str], None]) -> None:
        """callback(action, fact, source) with action in {'add','retract'}."""
        self._listeners.append(callback)

    def _notify(self, action: str, fact: Fact, source: str) -> None:
        for cb in self._listeners:
            cb(action, fact, source)

    def dump(self) -> str:
        return "\n".join(f"{f}    [{self._facts[f]}]" for f in self.facts())
