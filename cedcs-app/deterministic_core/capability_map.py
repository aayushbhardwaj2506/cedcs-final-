"""Canonical map from a triage capability key to the resource record(s) that
prove it. This is the ONLY vocabulary bridge between what the Triage agent may
emit (see config/tasks.yaml) and what the resource service actually stores
(see resource_service/seed.py). Keep the three in sync."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from schemas.resource import ScoredResourceRecord

Status = Literal["PRESENT", "ABSENT", "UNKNOWN"]

# Capabilities whose REQUIRED status implies the patient will be admitted.
ADMISSION_CAPABILITIES = frozenset({"ICU", "GENERAL_BED", "EMERGENCY_BED", "HDU", "PEDIATRIC_BED"})

# Capabilities that also have a specialist_<X> availability record.
SPECIALIST_CAPABILITIES = frozenset({"CARDIOLOGY", "NEUROLOGY", "ORTHOPEDICS", "PEDIATRICS"})


@dataclass(frozen=True)
class CheckResult:
    status: Status
    confidence: float = 0.0
    keys_used: tuple = field(default_factory=tuple)


def latest(records: list[ScoredResourceRecord], key: str) -> Optional[ScoredResourceRecord]:
    matching = [r for r in records if r.resource_key == key]
    return max(matching, key=lambda r: r.updated_at) if matching else None


def _flag(records, key: str, field_name: str) -> CheckResult:
    rec = latest(records, key)
    if rec is None or not isinstance(rec.value, dict) or rec.value.get(field_name) is None:
        return CheckResult("UNKNOWN", 0.0, (key,) if rec else ())
    return CheckResult("PRESENT" if rec.value[field_name] else "ABSENT", rec.confidence, (key,))


@dataclass(frozen=True)
class DepartmentCheck:
    code: str

    def evaluate(self, records: list[ScoredResourceRecord]) -> CheckResult:
        return _flag(records, f"department_{self.code}", "active")


@dataclass(frozen=True)
class EquipmentCheck:
    code: str

    def evaluate(self, records: list[ScoredResourceRecord]) -> CheckResult:
        return _flag(records, f"equipment_{self.code}", "operational")


@dataclass(frozen=True)
class BloodBankCheck:
    def evaluate(self, records: list[ScoredResourceRecord]) -> CheckResult:
        return _flag(records, "blood_bank", "available")


@dataclass(frozen=True)
class BedCheck:
    key: str

    def evaluate(self, records: list[ScoredResourceRecord]) -> CheckResult:
        rec = latest(records, self.key)
        if rec is None or not isinstance(rec.value, dict) or rec.value.get("available") is None:
            return CheckResult("UNKNOWN", 0.0, (self.key,) if rec else ())
        return CheckResult("PRESENT" if rec.value["available"] > 0 else "ABSENT", rec.confidence, (self.key,))


CapabilityCheck = Union[DepartmentCheck, EquipmentCheck, BloodBankCheck, BedCheck]

CAPABILITY_RESOURCE_MAP: dict[str, CapabilityCheck] = {
    "EMERGENCY_DEPARTMENT": DepartmentCheck("EMERGENCY_DEPARTMENT"),
    "CARDIOLOGY": DepartmentCheck("CARDIOLOGY"),
    "NEUROLOGY": DepartmentCheck("NEUROLOGY"),
    "TRAUMA": DepartmentCheck("TRAUMA"),
    "TRAUMA_BAY": DepartmentCheck("TRAUMA"),  # alias
    "ORTHOPEDICS": DepartmentCheck("ORTHOPEDICS"),
    "PEDIATRICS": DepartmentCheck("PEDIATRICS"),
    "OBSTETRICS": DepartmentCheck("OBSTETRICS"),
    "ONCOLOGY": DepartmentCheck("ONCOLOGY"),
    "NEPHROLOGY": DepartmentCheck("NEPHROLOGY"),
    "GENERAL_MEDICINE": DepartmentCheck("GENERAL_MEDICINE"),
    "ICU": BedCheck("icu_beds"),
    "GENERAL_BED": BedCheck("general_beds"),
    "EMERGENCY_BED": BedCheck("emergency_beds"),
    "HDU": BedCheck("hdu_beds"),
    "PEDIATRIC_BED": BedCheck("pediatric_beds"),
    "VENTILATOR": BedCheck("ventilators"),
    "CT_SCAN": EquipmentCheck("CT_SCAN"),
    "MRI": EquipmentCheck("MRI"),
    "X_RAY": EquipmentCheck("X_RAY"),
    "USG": EquipmentCheck("USG"),
    "CATH_LAB": EquipmentCheck("CATH_LAB"),
    "OT": EquipmentCheck("OT"),
    "LAB": EquipmentCheck("LAB"),
    "DIALYSIS": EquipmentCheck("DIALYSIS"),
    "CARDIAC_MONITORING": EquipmentCheck("CARDIAC_MONITOR"),
    "BLOOD_BANK": BloodBankCheck(),
}


def evaluate_capability(capability: str, records: list[ScoredResourceRecord]) -> CheckResult:
    """Unmapped capability keys are UNKNOWN (cannot be verified), never PRESENT."""
    check = CAPABILITY_RESOURCE_MAP.get(capability)
    if check is None:
        return CheckResult("UNKNOWN", 0.0, ())
    return check.evaluate(records)


def specialist_score(capability: str, records: list[ScoredResourceRecord]) -> Optional[tuple]:
    """(score, confidence) from specialist_<capability>, or None if the
    capability has no specialist counterpart. 1.0 on_site / 0.6 on_call / 0.0 else."""
    if capability not in SPECIALIST_CAPABILITIES:
        return None
    rec = latest(records, f"specialist_{capability}")
    if rec is None or not isinstance(rec.value, dict):
        return 0.0, 0.0
    return {"on_site": 1.0, "on_call": 0.6}.get(rec.value.get("status"), 0.0), rec.confidence
