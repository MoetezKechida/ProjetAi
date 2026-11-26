"""
cli_commands.py - Block 2 commands for the interactive CLI (cli.py).

Kept separate so cli.py only needs to forward unknown commands here.
"""
from __future__ import annotations

from typing import List

from kb import Fact

from .reasoner import Reasoner
from .unify import unify_with_trace

HELP = """Reasoning (Block 2)
  rules                         rule base, grouped by stratum
  infer                         re-run forward chaining and print the trace
  derived [Id]                  derived facts (all, or about one patient/resource)
  why <Fact>                    proof tree, e.g.  why RequiresICU(P1)
  ask <Query>                   resolution proof, e.g.  ask RequiresICU(P1)  /  ask RequiresICU(?x)
  unify <Term1> <Term2>         unification trace, e.g.  unify Pred(?x,B) Pred(A,?y)
  risk <P> <ML|DL> <Level> <p>  insert a model prediction, e.g.  risk P1 DL Critical 0.91
  reqs                          requirements sent to Search (Block 4)
  changes                       what each environment change added/removed
"""

COMMANDS = {"rules", "infer", "derived", "why", "ask", "unify", "risk", "reqs", "changes"}


def handle(reasoner: Reasoner, cmd: str, args: List[str]) -> bool:
    """Execute a Block 2 command. Returns False if `cmd` is not one of ours."""
    if cmd not in COMMANDS:
        return False
    if cmd == "rules":
        print(reasoner.describe_rules())
    elif cmd == "infer":
        result = reasoner.run("command 'infer'")
        print("\n".join(reasoner.last_trace))
        print(f"  {len(reasoner.derived())} derived facts"
              + ("" if result is None or not result.changed else f"\n{result}"))
    elif cmd == "derived":
        facts = reasoner.derived(args[0] if args else None)
        print("  " + ("\n  ".join(map(str, facts)) if facts else "nothing derived"))
    elif cmd == "why":
        print(reasoner.why(_one_term(args, "why <Fact>")))
    elif cmd == "ask":
        print(reasoner.ask(_one_term(args, "ask <Query>")).format())
    elif cmd == "unify":
        if len(args) != 2:
            raise ValueError("usage: unify <Term1> <Term2>   (no spaces inside a term)")
        _, trace = unify_with_trace(Fact.parse(args[0]), Fact.parse(args[1]))
        print("\n".join(trace))
    elif cmd == "risk":
        if len(args) != 4:
            raise ValueError("usage: risk <P> <ML|DL> <Level> <probability>")
        n = len(reasoner.history)
        fact = reasoner.env.record_risk(args[0], args[1], args[2], float(args[3]))
        print(f"  {fact} added")
        for res in reasoner.history[n:]:
            print(res)
        if len(reasoner.history) == n:
            print("  no derived fact changed")
    elif cmd == "reqs":
        print(reasoner.describe_requirements())
    elif cmd == "changes":
        for res in reasoner.history:
            print(res)
    return True


def _one_term(args: List[str], usage: str) -> str:
    if not args:
        raise ValueError(f"usage: {usage}")
    return "".join(args)     # tolerate spaces after commas: why Pred(P1, X)
