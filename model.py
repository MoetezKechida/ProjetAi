"""
model.py - Entities of the emergency-department world (Block 1).

Plain data classes with validation. They carry NO reasoning logic; the
Environment turns them into symbolic facts for the Knowledge Base.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

RESOURCE_KINDS = ("Doctor", "Nurse", "Bed", "Equipment", "Room")
PATIENT_STATUSES = ("Waiting", "UnderTreatment", "Discharged")
BP_LEVELS = ("Low", "Normal", "High")

# Thresholds used only to categorise a numeric systolic pressure.
LOW_SYSTOLIC = 90.0
HIGH_SYSTOLIC = 140.0


def camel(text: str) -> str:
    """'chest pain' / 'chest_pain' -> 'ChestPain' (symbol-friendly name)."""
    parts = re.split(r"[\s_\-]+", str(text).strip())
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def _check(name: str, value, lo: float, hi: float) -> None:
    if value is not None and not (lo <= value <= hi):
        raise ValueError(f"{name}={value} outside plausible range [{lo}, {hi}]")


@dataclass
class VitalSigns:
    """One physiological measurement (one time step). Also the unit of the
    time series that the Deep Learning model (Block 3) will consume."""

    time: Optional[float] = None
    heart_rate: Optional[float] = None
    systolic: Optional[float] = None
    diastolic: Optional[float] = None
    spo2: Optional[float] = None          # oxygen saturation, %
    temperature: Optional[float] = None   # degrees Celsius
    resp_rate: Optional[float] = None     # breaths / minute
    bp_category: Optional[str] = None     # explicit 'Low'/'Normal'/'High'

    def __post_init__(self) -> None:
        _check("heart_rate", self.heart_rate, 0, 300)
        _check("systolic", self.systolic, 30, 300)
        _check("diastolic", self.diastolic, 20, 200)
        _check("spo2", self.spo2, 0, 100)
        _check("temperature", self.temperature, 25, 45)
        _check("resp_rate", self.resp_rate, 0, 80)
        if self.bp_category is not None:
            self.bp_category = str(self.bp_category).capitalize()
            if self.bp_category not in BP_LEVELS:
                raise ValueError(f"bp_category must be one of {BP_LEVELS}")

    @property
    def bp_level(self) -> Optional[str]:
        """Explicit category wins; otherwise derived from systolic pressure."""
        if self.bp_category:
            return self.bp_category
        if self.systolic is None:
            return None
        if self.systolic < LOW_SYSTOLIC:
            return "Low"
        if self.systolic >= HIGH_SYSTOLIC:
            return "High"
        return "Normal"

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def as_row(self) -> list:
        """Fixed-order numeric row [HR, SBP, DBP, SpO2, Temp, RR] (None kept)
        - handy for building ML/DL feature matrices."""
        return [self.heart_rate, self.systolic, self.diastolic,
                self.spo2, self.temperature, self.resp_rate]


@dataclass
class Patient:
    id: str
    age: int
    name: str = ""
    symptoms: Set[str] = field(default_factory=set)
    history: List[VitalSigns] = field(default_factory=list)  # chronological
    status: str = "Waiting"
    arrival_time: Optional[float] = None
    assigned: Set[str] = field(default_factory=set)  # ids of resources held

    def __post_init__(self) -> None:
        _check("age", self.age, 0, 120)
        self.symptoms = {camel(s) for s in self.symptoms}
        if self.status not in PATIENT_STATUSES:
            raise ValueError(f"status must be one of {PATIENT_STATUSES}")

    @property
    def current(self) -> Optional[VitalSigns]:
        return self.history[-1] if self.history else None

    def feature_dict(self) -> dict:
        """Flat structured view of the patient's CURRENT state (for Block 3 ML)."""
        v = self.current or VitalSigns()
        d = {"age": self.age, "heart_rate": v.heart_rate, "systolic": v.systolic,
             "diastolic": v.diastolic, "spo2": v.spo2, "temperature": v.temperature,
             "resp_rate": v.resp_rate, "bp_level": v.bp_level}
        d.update({f"symptom_{s}": 1 for s in sorted(self.symptoms)})
        return d


@dataclass
class Resource:
    """Any limited hospital resource.

    kind    : Doctor | Nurse | Bed | Equipment | Room
    subtype : Doctor/Nurse -> specialty (Cardiology, General...)
              Bed          -> ICU | Emergency
              Room         -> Examination | Surgery ...
              Equipment    -> Ventilator | ECG ...
    """

    id: str
    kind: str
    subtype: str = ""
    available: bool = True                  # physically usable (not broken/off-duty)
    assigned_to: Optional[str] = None       # patient id currently using it

    def __post_init__(self) -> None:
        self.kind = camel(self.kind)
        if self.kind not in RESOURCE_KINDS:
            raise ValueError(f"kind must be one of {RESOURCE_KINDS}, got {self.kind!r}")
        self.subtype = camel(self.subtype) if self.subtype else ""

    @property
    def is_free(self) -> bool:
        return self.available and self.assigned_to is None
