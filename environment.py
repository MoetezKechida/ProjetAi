"""
environment.py - The emergency-department world model (Block 1).

Responsibilities
  * hold patients and resources (model.py)
  * keep the Knowledge Base synchronised with the world (every change
    re-generates the affected symbolic facts)
  * expose an immutable State snapshot for Search / Planning (Blocks 4-5)
  * emit Events on every change so the agent (Block 6) can detect that the
    environment changed and that earlier decisions may be obsolete
  * accept ML / DL predictions as facts (Block 3 -> Block 2 bridge)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple

from kb import F, Fact, KnowledgeBase
from model import Patient, Resource, VitalSigns, camel

ENV = "env"  # KB source tag for facts produced by the environment


def _num(x):
    """125.0 -> 125 so facts print as HeartRate(P1,125)."""
    return int(x) if isinstance(x, float) and x.is_integer() else x


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Event:
    """A change in the environment.

    kind in: patient_arrived, vitals_updated, symptom_changed, resource_added,
             resource_unavailable, resource_restored, assignment_broken,
             assigned, released, treatment_started, patient_discharged,
             risk_recorded, evidence_invalidated
    """
    version: int
    time: float
    kind: str
    subject: str
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        d = ", ".join(f"{k}={v}" for k, v in self.detail.items())
        return f"[v{self.version} t={self.time}] {self.kind}({self.subject})" + (f" {d}" if d else "")


# --------------------------------------------------------------------------
# Immutable snapshot used by Search and Planning
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class State:
    """Hashable state of the world: who waits, who is treated, which resources
    are free and who holds what. Transitions return NEW states."""

    waiting: FrozenSet[str]
    in_treatment: FrozenSet[str]
    free: FrozenSet[str]
    assignments: FrozenSet[Tuple[str, str]]  # (patient_id, resource_id)

    def is_free(self, rid: str) -> bool:
        return rid in self.free

    def assigned_to(self, pid: str) -> FrozenSet[str]:
        return frozenset(r for p, r in self.assignments if p == pid)

    def assign(self, pid: str, rid: str) -> "State":
        """AssignDoctor / ReserveICU ... all reduce to this transition."""
        if rid not in self.free:
            raise ValueError(f"Resource {rid} is not free")
        if pid not in self.waiting and pid not in self.in_treatment:
            raise ValueError(f"Patient {pid} is not active")
        return State(self.waiting, self.in_treatment, self.free - {rid},
                     self.assignments | {(pid, rid)})

    def release(self, pid: str, rid: str) -> "State":
        if (pid, rid) not in self.assignments:
            raise ValueError(f"{rid} is not assigned to {pid}")
        return State(self.waiting, self.in_treatment, self.free | {rid},
                     self.assignments - {(pid, rid)})

    def start_treatment(self, pid: str) -> "State":
        if pid not in self.waiting:
            raise ValueError(f"Patient {pid} is not waiting")
        return State(self.waiting - {pid}, self.in_treatment | {pid},
                     self.free, self.assignments)

    def __str__(self) -> str:
        return (f"State(waiting={sorted(self.waiting)}, in_treatment={sorted(self.in_treatment)}, "
                f"free={sorted(self.free)}, assignments={sorted(self.assignments)})")


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
class Environment:
    def __init__(self, kb: Optional[KnowledgeBase] = None) -> None:
        self.kb = kb if kb is not None else KnowledgeBase()
        self.patients: Dict[str, Patient] = {}
        self.resources: Dict[str, Resource] = {}
        self.clock: float = 0.0
        self.version = 0
        self.events: List[Event] = []
        self._subs: List[Callable[[Event], None]] = []

    # ---- events ---------------------------------------------------------
    def subscribe(self, callback: Callable[[Event], None]) -> None:
        self._subs.append(callback)

    def _emit(self, kind: str, subject: str, **detail) -> Event:
        self.version += 1
        ev = Event(self.version, self.clock, kind, subject, detail)
        self.events.append(ev)
        for cb in self._subs:
            cb(ev)
        return ev

    def advance_time(self, t: float) -> None:
        if t < self.clock:
            raise ValueError("time cannot go backwards")
        self.clock = t

    # ---- lookups --------------------------------------------------------
    def patient(self, pid: str) -> Patient:
        if pid not in self.patients:
            raise KeyError(f"Unknown patient {pid}")
        return self.patients[pid]

    def resource(self, rid: str) -> Resource:
        if rid not in self.resources:
            raise KeyError(f"Unknown resource {rid}")
        return self.resources[rid]

    def patients_with_status(self, status: str) -> List[Patient]:
        return [p for p in self.patients.values() if p.status == status]

    def free_resources(self, kind: Optional[str] = None, subtype: Optional[str] = None) -> List[Resource]:
        kind = camel(kind) if kind else None
        subtype = camel(subtype) if subtype else None
        return [r for r in self.resources.values()
                if r.is_free and (kind is None or r.kind == kind)
                and (subtype is None or r.subtype == subtype)]

    # ---- patients -------------------------------------------------------
    def add_patient(self, patient: Patient) -> Patient:
        if patient.id in self.patients:
            raise ValueError(f"Patient {patient.id} already exists")
        if patient.arrival_time is None:
            patient.arrival_time = self.clock
        for i, v in enumerate(patient.history, start=1):
            if v.time is None:
                v.time = i
        self.patients[patient.id] = patient
        self._sync_patient(patient.id)
        self._emit("patient_arrived", patient.id, age=patient.age)
        return patient

    def update_vitals(self, pid: str, vitals: VitalSigns) -> None:
        """New measurement. Old ML/DL predictions become stale and are removed."""
        p = self.patient(pid)
        if vitals.time is None:
            vitals.time = len(p.history) + 1
        p.history.append(vitals)
        self._invalidate_evidence(pid)
        self._sync_patient(pid)
        self._emit("vitals_updated", pid, time=vitals.time)

    def add_symptom(self, pid: str, symptom: str) -> None:
        p = self.patient(pid)
        s = camel(symptom)
        if s not in p.symptoms:
            p.symptoms.add(s)
            self._invalidate_evidence(pid)
            self._sync_patient(pid)
            self._emit("symptom_changed", pid, added=s)

    def remove_symptom(self, pid: str, symptom: str) -> None:
        p = self.patient(pid)
        s = camel(symptom)
        if s in p.symptoms:
            p.symptoms.discard(s)
            self._invalidate_evidence(pid)
            self._sync_patient(pid)
            self._emit("symptom_changed", pid, removed=s)

    def start_treatment(self, pid: str) -> None:
        p = self.patient(pid)
        if p.status != "Waiting":
            raise ValueError(f"{pid} is {p.status}, not Waiting")
        p.status = "UnderTreatment"
        self._sync_patient(pid)
        self._emit("treatment_started", pid)

    def discharge(self, pid: str) -> None:
        p = self.patient(pid)
        for rid in list(p.assigned):
            self._release(pid, rid)
        p.status = "Discharged"
        self._sync_patient(pid)
        self._emit("patient_discharged", pid)

    # ---- resources ------------------------------------------------------
    def add_resource(self, resource: Resource) -> Resource:
        if resource.id in self.resources:
            raise ValueError(f"Resource {resource.id} already exists")
        self.resources[resource.id] = resource
        self._sync_resource(resource.id)
        self._emit("resource_added", resource.id, resource_kind=resource.kind, subtype=resource.subtype)
        return resource

    def set_resource_available(self, rid: str, available: bool, reason: str = "") -> None:
        """Make a resource (un)usable. If it was assigned, the assignment is
        broken -> an 'assignment_broken' event tells the agent its plan is obsolete."""
        r = self.resource(rid)
        if r.available == available:
            return
        r.available = available
        victim = r.assigned_to
        if not available and victim:
            self._release(victim, rid)
        self._sync_resource(rid)
        self._emit("resource_restored" if available else "resource_unavailable", rid, reason=reason)
        if not available and victim:
            self._emit("assignment_broken", victim, resource=rid)

    # ---- assignments ----------------------------------------------------
    def assign(self, pid: str, rid: str) -> None:
        p, r = self.patient(pid), self.resource(rid)
        if p.status == "Discharged":
            raise ValueError(f"{pid} is discharged")
        if not r.is_free:
            raise ValueError(f"{rid} is not free")
        r.assigned_to = pid
        p.assigned.add(rid)
        self._sync_patient(pid)
        self._sync_resource(rid)
        self._emit("assigned", pid, resource=rid)

    def release(self, pid: str, rid: Optional[str] = None) -> None:
        """Release one resource (or all of the patient's resources)."""
        p = self.patient(pid)
        targets = [rid] if rid else sorted(p.assigned)
        for t in targets:
            if t not in p.assigned:
                raise ValueError(f"{t} is not assigned to {pid}")
            self._release(pid, t)
            self._sync_resource(t)
            self._emit("released", pid, resource=t)
        self._sync_patient(pid)

    def _release(self, pid: str, rid: str) -> None:
        self.patients[pid].assigned.discard(rid)
        self.resources[rid].assigned_to = None
        self._sync_patient(pid)

    # ---- learned evidence (Block 3 -> Block 2 bridge) -------------------
    def record_risk(self, pid: str, source: str, level: str, probability: float) -> Fact:
        """Store a model prediction as ML_Risk(P,Level,Prob) or DL_Risk(...).
        Replaces the previous prediction of the same source for that patient."""
        self.patient(pid)
        source = source.upper()
        if source not in ("ML", "DL"):
            raise ValueError("source must be 'ML' or 'DL'")
        level = camel(level)
        if level not in ("Low", "Medium", "High", "Critical"):
            raise ValueError("level must be Low, Medium, High or Critical")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        pred = f"{source}_Risk"
        self.kb.retract_where(F(pred, pid, "?l", "?p"), source=source)
        fact = F(pred, pid, level, round(float(probability), 4))
        self.kb.add(fact, source=source)
        self._emit("risk_recorded", pid, source=source, level=level, probability=probability)
        return fact

    def _invalidate_evidence(self, pid: str) -> None:
        removed = []
        for pred, src in (("ML_Risk", "ML"), ("DL_Risk", "DL")):
            removed += self.kb.retract_where(F(pred, pid, "?l", "?p"), source=src)
        if removed:
            self._emit("evidence_invalidated", pid, removed=[str(f) for f in removed])

    # ---- state snapshot -------------------------------------------------
    def snapshot(self) -> State:
        return State(
            waiting=frozenset(p.id for p in self.patients.values() if p.status == "Waiting"),
            in_treatment=frozenset(p.id for p in self.patients.values() if p.status == "UnderTreatment"),
            free=frozenset(r.id for r in self.resources.values() if r.is_free),
            assignments=frozenset((r.assigned_to, r.id) for r in self.resources.values() if r.assigned_to),
        )

    # ---- KB synchronisation --------------------------------------------
    def _patient_facts(self, p: Patient) -> Iterable[Fact]:
        pid = p.id
        yield F("Patient", pid)
        yield F("Age", pid, p.age)
        v = p.current
        if v is not None:
            if v.heart_rate is not None:
                yield F("HeartRate", pid, _num(v.heart_rate))
            if v.systolic is not None:
                yield F("SystolicBP", pid, _num(v.systolic))
            if v.diastolic is not None:
                yield F("DiastolicBP", pid, _num(v.diastolic))
            if v.bp_level:
                yield F(f"{v.bp_level}BloodPressure", pid)       # e.g. LowBloodPressure(P1)
            if v.spo2 is not None:
                yield F("OxygenSaturation", pid, _num(v.spo2))
            if v.temperature is not None:
                yield F("Temperature", pid, _num(v.temperature))
            if v.resp_rate is not None:
                yield F("RespiratoryRate", pid, _num(v.resp_rate))
        for s in sorted(p.symptoms):
            yield F("HasSymptom", pid, s)
        yield F({"Waiting": "PatientWaiting", "UnderTreatment": "PatientUnderTreatment",
                 "Discharged": "PatientDischarged"}[p.status], pid)
        for rid in sorted(p.assigned):
            yield F(f"{self.resources[rid].kind}Assigned", pid, rid)  # DoctorAssigned(P1,D1)

    def _resource_facts(self, r: Resource) -> Iterable[Fact]:
        yield F(r.kind, r.id)                                          # Doctor(D1), Bed(ICU1)
        if r.subtype:
            yield F("Specialty" if r.kind in ("Doctor", "Nurse") else "Type", r.id, r.subtype)
        yield F("Available" if r.is_free else "NotAvailable", r.id)

    def _sync_patient(self, pid: str) -> None:
        self.kb.retract_subject(pid, source=ENV)
        for f in self._patient_facts(self.patients[pid]):
            self.kb.add(f, source=ENV)

    def _sync_resource(self, rid: str) -> None:
        self.kb.retract_subject(rid, source=ENV)
        for f in self._resource_facts(self.resources[rid]):
            self.kb.add(f, source=ENV)

    def resync_all(self) -> None:
        for pid in self.patients:
            self._sync_patient(pid)
        for rid in self.resources:
            self._sync_resource(rid)

    # ---- display --------------------------------------------------------
    def describe(self) -> str:
        lines = [f"Environment v{self.version}, t={self.clock}"]
        lines.append("Patients:")
        for p in self.patients.values():
            v = p.current
            vit = (f"HR={v.heart_rate} BP={v.systolic}/{v.diastolic}({v.bp_level}) "
                   f"SpO2={v.spo2} T={v.temperature} RR={v.resp_rate}") if v else "no vitals"
            lines.append(f"  {p.id} age={p.age} [{p.status}] {vit} symptoms={sorted(p.symptoms)} "
                         f"holds={sorted(p.assigned)} samples={len(p.history)}")
        lines.append("Resources:")
        for r in self.resources.values():
            st = "free" if r.is_free else (f"used by {r.assigned_to}" if r.assigned_to else "UNAVAILABLE")
            lines.append(f"  {r.id:6} {r.kind:9} {r.subtype:12} {st}")
        return "\n".join(lines)
