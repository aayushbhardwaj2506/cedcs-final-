"""Bed availability as a ranking metric (used only when the Bed Checker is on).

Which bed types matter depends on what the patient needs: an ICU requirement makes ICU beds the deciding resource, a
ventilator requirement makes ventilators count, and the emergency department's own beds always matter on arrival. Each
relevant type scores min(free beds / BED_TARGET, 1), discounted by the confidence of the record it came from, so a
hospital with 4 ICU beds reported a minute ago beats one with 4 reported three hours ago, and one with 0 or unknown
scores nothing. Unknown is never treated as available.
"""

from __future__ import annotations

from deterministic_core.capability_map import latest
from schemas.resource import ScoredResourceRecord

BED_KEYS = ["icu_beds", "emergency_beds", "general_beds", "hdu_beds", "pediatric_beds", "ventilators"]
BED_LABELS = {"icu_beds": "ICU", "emergency_beds": "ER", "general_beds": "General", "hdu_beds": "HDU",
              "pediatric_beds": "Paeds", "ventilators": "Vent"}
BED_TARGET = 3  # this many free beds of a type counts as fully comfortable
BED_WEIGHT = 0.35  # share of the total score when the Bed Checker is on; the other weights are scaled by 1 - BED_WEIGHT.
# 0.35 keeps beds the LARGEST single weight at every priority: the biggest original weight is 0.48 (LOW, accessibility)
# and 0.48 * (1 - 0.35) = 0.312 < 0.35.

_CAP_TO_BED = {"ICU": "icu_beds", "GENERAL_BED": "general_beds", "EMERGENCY_BED": "emergency_beds", "HDU": "hdu_beds",
               "PEDIATRIC_BED": "pediatric_beds", "VENTILATOR": "ventilators"}


def relevant_bed_keys(required: set, preferred: set) -> list:
    keys = ["emergency_beds"]  # every emergency arrives through the ED
    for cap in sorted(required | preferred):
        key = _CAP_TO_BED.get(cap)
        if key and key not in keys:
            keys.append(key)
    return keys


def bed_detail(records: list[ScoredResourceRecord], keys: list) -> dict:
    """{bed_key: {available, total, confidence, age_min}} for the given keys (available is None when unknown/missing)."""
    out = {}
    for k in keys:
        rec = latest(records, k)
        v = rec.value if rec is not None and isinstance(rec.value, dict) else {}
        out[k] = {"available": v.get("available"), "total": v.get("total"),
                  "confidence": rec.confidence if rec is not None else 0.0,
                  "age_min": round(rec.age_minutes, 1) if rec is not None else None}
    return out


def bed_score(records: list[ScoredResourceRecord], keys: list) -> float:
    detail = bed_detail(records, keys)
    parts = []
    for d in detail.values():
        avail = d["available"]
        parts.append(0.0 if avail is None else min(max(avail, 0) / BED_TARGET, 1.0) * d["confidence"])
    return sum(parts) / len(parts) if parts else 0.0
