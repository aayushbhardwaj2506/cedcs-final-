"""confidence = SOURCE_TRUST[source] * exp(-age_minutes / tau); tau per resource type."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional

from schemas.resource import SOURCE_TRUST, Freshness, ResourceRecord, ScoredResourceRecord

# (prefix, tau in minutes); first match wins.
_TAU_RULES: list = [
    ("icu_beds", 45),
    ("emergency_beds", 30),
    ("general_beds", 60),
    ("hdu_beds", 60),
    ("pediatric_beds", 60),
    ("ventilators", 90),
    ("specialist_", 240),
    ("blood_bank", 360),
    ("equipment_", 1440),
    ("department_", 43200),
    ("opd", 30),
    ("ipd_admission_delay_est_min", 30),  # not in the design doc's table; assumed same order as bed data
]
DEFAULT_TAU = 60.0


def tau_for(resource_key: str) -> float:
    for prefix, tau in _TAU_RULES:
        if resource_key.startswith(prefix):
            return float(tau)
    return DEFAULT_TAU


def is_null_fact(value: Any) -> bool:
    """True when the fact itself is unknown (e.g. {"total": 10, "available": null}), which is not the same as stale."""
    if value is None:
        return True
    return isinstance(value, dict) and "available" in value and value["available"] is None


def classify(confidence: float) -> Freshness:
    if confidence >= 0.80:
        return "FRESH"
    if confidence >= 0.50:
        return "RECENT"
    if confidence >= 0.25:
        return "STALE"
    return "UNKNOWN"


def score_records(records: list[ResourceRecord], now: Optional[datetime] = None) -> list[ScoredResourceRecord]:
    now = now or datetime.now(timezone.utc)
    scored: list[ScoredResourceRecord] = []
    for rec in records:
        updated = rec.updated_at if rec.updated_at.tzinfo else rec.updated_at.replace(tzinfo=timezone.utc)
        age_min = max(0.0, (now - updated).total_seconds() / 60.0)
        if is_null_fact(rec.value):
            confidence, freshness = 0.0, "UNKNOWN"
        else:
            confidence = min(1.0, SOURCE_TRUST[rec.source] * math.exp(-age_min / tau_for(rec.resource_key)))
            freshness = classify(confidence)
        scored.append(
            ScoredResourceRecord(
                **rec.model_dump(), confidence=confidence, freshness=freshness, age_minutes=age_min
            )
        )
    return scored
