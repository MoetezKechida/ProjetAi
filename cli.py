"""
cli.py - Interactive command-line interface for the emergency environment.

    python cli.py [scenario.json]

Type 'help' for the list of commands.
"""
from __future__ import annotations

import json
import shlex
import sys

from environment import Environment
from interface import (apply_event, build_patient, load_events, load_scenario,
                       save_scenario)
from model import Patient

HELP = """Commands
  load <file.json>              load a scenario (resources + patients)
  events <file.json>            apply every event of a file, one by one
  show                          describe the environment
  kb                            print all facts (with their sources)
  facts <Predicate>             print facts of one predicate
  query <Pattern>               e.g. query HasSymptom(?p,ChestPain)
  about <Id>                    all facts about a patient/resource
  patient                       guided form to add a patient
  vitals <P> <HR> <SBP> <DBP> <SpO2> <Temp> <RR>   add a measurement ('-' = unknown)
  symptom <P> <name>            add a symptom
  down <R> / up <R>             make a resource unavailable / available
  assign <P> <R> / release <P> [R]
  state                         immutable State snapshot (for Search/Planning)
  log                           event log
  save <file.json>              save the scenario
  quit
"""


def _ask(prompt: str, cast=str, default=None):
    raw = input(f"  {prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
    if not raw:
        return default
    return cast(raw)


def _form_patient(env: Environment) -> None:
    print("New patient (empty answer = unknown)")
    d = {"id": _ask("id"), "age": _ask("age", int)}
    d["symptoms"] = [s.strip() for s in (_ask("symptoms (comma separated)", str, "") or "").split(",") if s.strip()]
    v = {}
    for key, label, cast in [("heart_rate", "heart rate", float), ("systolic", "systolic BP", float),
                             ("diastolic", "diastolic BP", float), ("spo2", "SpO2 %", float),
                             ("temperature", "temperature C", float), ("resp_rate", "respiratory rate", float)]:
        val = _ask(label, cast)
        if val is not None:
            v[key] = val
    if v:
        d["vitals"] = v
    env.add_patient(build_patient(d))
    print(f"  -> {d['id']} added, {len(env.kb.about(d['id']))} facts created")


def _opt(x: str):
    return None if x == "-" else float(x)


def run(env: Environment, line: str) -> bool:
    parts = shlex.split(line)
    if not parts:
        return True
    cmd, args = parts[0].lower(), parts[1:]
    if cmd in ("quit", "exit"):
        return False
    elif cmd == "help":
        print(HELP)
    elif cmd == "load":
        load_scenario(env, args[0]); print(env.describe())
    elif cmd == "events":
        for ev in load_events(args[0]):
            apply_event(env, ev)
        for e in env.events[-10:]:
            print(" ", e)
    elif cmd == "show":
        print(env.describe())
    elif cmd == "kb":
        print(env.kb.dump())
    elif cmd == "facts":
        for f in env.kb.facts(args[0]):
            print(" ", f)
    elif cmd == "query":
        res = env.kb.query(" ".join(args))
        print("  " + ("\n  ".join(map(str, res)) if res else "no match"))
    elif cmd == "about":
        for f in env.kb.about(args[0]):
            print(" ", f)
    elif cmd == "patient":
        _form_patient(env)
    elif cmd == "vitals":
        from model import VitalSigns
        hr, sbp, dbp, spo2, temp, rr = (_opt(a) for a in args[1:7])
        env.update_vitals(args[0], VitalSigns(heart_rate=hr, systolic=sbp, diastolic=dbp,
                                              spo2=spo2, temperature=temp, resp_rate=rr))
    elif cmd == "symptom":
        env.add_symptom(args[0], " ".join(args[1:]))
    elif cmd == "down":
        env.set_resource_available(args[0], False, "set from CLI")
    elif cmd == "up":
        env.set_resource_available(args[0], True, "set from CLI")
    elif cmd == "assign":
        env.assign(args[0], args[1])
    elif cmd == "release":
        env.release(args[0], args[1] if len(args) > 1 else None)
    elif cmd == "state":
        print(" ", env.snapshot())
    elif cmd == "log":
        for e in env.events:
            print(" ", e)
    elif cmd == "save":
        save_scenario(env, args[0]); print("  saved")
    else:
        print("Unknown command. Type 'help'.")
    return True


def main() -> None:
    env = Environment()
    if len(sys.argv) > 1:
        load_scenario(env, sys.argv[1])
        print(env.describe())
    print("Emergency environment - type 'help'")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        try:
            if not run(env, line):
                break
        except (ValueError, KeyError, IndexError, json.JSONDecodeError, FileNotFoundError) as e:
            print(f"  error: {e}")


if __name__ == "__main__":
    main()
