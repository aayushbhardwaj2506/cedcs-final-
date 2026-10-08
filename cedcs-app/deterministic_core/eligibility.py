"""Hard eligibility gate (design doc 7.1). Filters; never ranks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from deterministic_core import facility_hints
from deterministic_core.capability_map import (
    ADMISSION_CAPABILITIES,
    CAPABILITY_RESOURCE_MAP,
    evaluate_capability,
)
from schemas.hospital import CandidateHospital
from schemas.requirement import RequirementProfile
from schemas.resource import ScoredResourceRecord

logger = logging.getLogger("cedcs.eligibility")


@dataclass
class EligibilityResult:
    eligible: list[CandidateHospital]
    provisional_ids: set = field(default_factory=set)  # eligible, but a REQUIRED capability was UNKNOWN
    rejections: dict = field(default_factory=dict)  # hospital_id -> plain-language reason


def _only_inferred(cap: str, records) -> bool:
    """True when the latest record behind this capability comes from a map tag (INFERRED), not from the hospital."""
    latest = [max((r for r in records if r.resource_key == k), key=lambda r: r.updated_at)
              for k in evaluate_capability(cap, records).keys_used if any(r.resource_key == k for r in records)]
    return bool(latest) and all(r.source == "INFERRED" for r in latest)


def filter_eligible(
    candidates: list[CandidateHospital],
    required: RequirementProfile,
    scored_records_by_hospital: dict[str, list[ScoredResourceRecord]],
    categories: frozenset = frozenset(),
) -> EligibilityResult:
    result = EligibilityResult(eligible=[])
    required_caps = sorted(required.required())
    admission_case = any(c in ADMISSION_CAPABILITIES for c in required_caps)

    for cap in required_caps:
        if cap not in CAPABILITY_RESOURCE_MAP:
            logger.warning("required capability %r has no resource mapping; treated as UNKNOWN", cap)

    for cand in candidates:
        hid = cand.hospital_id
        if cand.operating_status != "OPERATIONAL":
            result.rejections[hid] = f"facility is {cand.operating_status}"
            continue
        if admission_case and not cand.ipd_accepting:
            result.rejections[hid] = "not currently accepting admissions"
            continue

        if getattr(cand, "source", "registry") == "osm":
            # A real hospital's capabilities are unknown, but its NAME can rule out an eye/dental/skin clinic or a specialty
            # hospital outside its field. Inference from a name, labelled as such.
            why = facility_hints.rejection_reason(cand.name, set(categories))
            if why:
                result.rejections[hid] = why
                continue
        records = scored_records_by_hospital.get(hid, [])
        rejected_for = None
        provisional = False
        for cap in required_caps:
            status = evaluate_capability(cap, records).status
            if status == "ABSENT":
                rejected_for = cap
                break
            if status == "UNKNOWN":
                provisional = True
            elif status == "PRESENT" and _only_inferred(cap, records):
                provisional = True  # a map tag says so, nobody at the hospital has confirmed it
        if rejected_for:
            result.rejections[hid] = f"required capability {rejected_for} is not available"
            continue

        result.eligible.append(cand)
        if provisional:
            result.provisional_ids.add(hid)
    return result
