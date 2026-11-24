"""
interface.py - Structured input interface (Block 1).

Converts JSON/dict input into entities, then into KB facts via Environment.
No LLM involved: a strict, validated schema.

Scenario file:
{
  "resources": [{"id":"D1","kind":"Doctor","subtype":"Cardiology"}, ...],
  "patients":  [{"id":"P1","age":72,"symptoms":["chest pain"],
                 "vitals": {"heart_rate":125,"blood_pressure":"Low","spo2":88,"temperature":39},
                 "vitals_history": [ {...}, {...} ]}, ...]
}

Event (applied at run time, used for the dynamic environment of Block 6):
  {"type":"add_patient", "patient": {...}}
  {"type":"update_vitals", "patient":"P1", "vitals": {...}}
  {"type":"add_symptom" | "remove_symptom", "patient":"P1", "symptom":"fainting"}
  {"type":"add_resource", "resource": {...}}
  {"type":"set_available", "resource":"ICU1", "available": false, "reason":"..."}
  {"type":"discharge" | "start_treatment", "patient":"P1"}
  {"type":"advance_time", "time": 12}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

from environment import Environment, Event
from model import Patient, Resource, VitalSigns

_VITAL_KEYS = {"time", "heart_rate", "systolic", "diastolic", "spo2", "temperature", "resp_rate"}


def _require(d: dict, key: str, ctx: str):
    if key not in d:
        raise ValueError(f"{ctx}: missing required field '{key}'")
    return d[key]


def build_vitals(d: dict) -> VitalSigns:
    unknown = set(d) - _VITAL_KEYS - {"blood_pressure", "bp_category"}
    if unknown:
        raise ValueError(f"vitals: unknown field(s) {sorted(unknown)}")
    kw = {k: d[k] for k in _VITAL_KEYS if k in d}
    bp = d.get("blood_pressure", d.get("bp_category"))
    if isinstance(bp, str):
        kw["bp_category"] = bp
    elif bp is not None:
        raise ValueError("vitals.blood_pressure must be 'Low', 'Normal' or 'High'")
    return VitalSigns(**kw)


def build_patient(d: dict) -> Patient:
    pid = str(_require(d, "id", "patient"))
    p = Patient(id=pid, age=int(_require(d, "age", f"patient {pid}")),
                name=d.get("name", ""), symptoms=set(d.get("symptoms", [])))
    if "vitals_history" in d:
        p.history = [build_vitals(v) for v in d["vitals_history"]]
    elif "vitals" in d:
        p.history = [build_vitals(d["vitals"])]
    return p


def build_resource(d: dict) -> Resource:
    rid = str(_require(d, "id", "resource"))
    return Resource(id=rid, kind=_require(d, "kind", f"resource {rid}"),
                    subtype=d.get("subtype", ""), available=bool(d.get("available", True)))


def load_scenario(env: Environment, source: Union[str, Path, dict]) -> Environment:
    """Populate `env` from a JSON file path or an already-parsed dict."""
    data = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    for r in data.get("resources", []):
        env.add_resource(build_resource(r))
    for p in data.get("patients", []):
        env.add_patient(build_patient(p))
    return env


def load_events(source: Union[str, Path, list, dict]) -> List[dict]:
    if isinstance(source, list):
        return source
    data = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    return data["events"] if isinstance(data, dict) else data


def apply_event(env: Environment, ev: dict) -> Optional[Event]:
    """Apply ONE external change to the environment. Returns the last Event
    emitted (None if nothing changed)."""
    kind = _require(ev, "type", "event")
    before = len(env.events)
    if kind == "add_patient":
        env.add_patient(build_patient(_require(ev, "patient", kind)))
    elif kind == "update_vitals":
        env.update_vitals(_require(ev, "patient", kind), build_vitals(_require(ev, "vitals", kind)))
    elif kind == "add_symptom":
        env.add_symptom(_require(ev, "patient", kind), _require(ev, "symptom", kind))
    elif kind == "remove_symptom":
        env.remove_symptom(_require(ev, "patient", kind), _require(ev, "symptom", kind))
    elif kind == "add_resource":
        env.add_resource(build_resource(_require(ev, "resource", kind)))
    elif kind == "set_available":
        env.set_resource_available(_require(ev, "resource", kind), bool(_require(ev, "available", kind)),
                                   ev.get("reason", ""))
    elif kind == "discharge":
        env.discharge(_require(ev, "patient", kind))
    elif kind == "start_treatment":
        env.start_treatment(_require(ev, "patient", kind))
    elif kind == "advance_time":
        env.advance_time(float(_require(ev, "time", kind)))
    else:
        raise ValueError(f"Unknown event type {kind!r}")
    return env.events[-1] if len(env.events) > before else None


def dump_scenario(env: Environment) -> dict:
    """Export the current environment back to the scenario JSON format."""
    return {
        "resources": [{"id": r.id, "kind": r.kind, "subtype": r.subtype, "available": r.available}
                      for r in env.resources.values()],
        "patients": [{"id": p.id, "age": p.age, "name": p.name, "symptoms": sorted(p.symptoms),
                      "vitals_history": [{**{k: v for k, v in vs.as_dict().items() if k != "bp_category"},
                                          **({"blood_pressure": vs.bp_category} if vs.bp_category else {})}
                                         for vs in p.history]}
                     for p in env.patients.values()],
    }


def save_scenario(env: Environment, path: Union[str, Path]) -> None:
    Path(path).write_text(json.dumps(dump_scenario(env), indent=2), encoding="utf-8")
