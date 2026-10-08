"""
Resource schemas — the provenance-preserving EAV contract (design doc P3:
"every resource fact is stored as (value, updated_at, source, confidence),
never a bare value").

Matches config/tasks.yaml (`resource_interpretation_task.expected_output`)
and cedcs_resource_stub/schema.sql's resource_records table shape.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

ResourceSource = Literal["HOSPITAL_CONSOLE", "FHIR", "GOVT", "INFERRED", "SEED"]

Freshness = Literal["FRESH", "RECENT", "STALE", "UNKNOWN"]

# Trust weight per source, used by freshness.py's
# confidence = source_trust * exp(-age_minutes / tau) model.
SOURCE_TRUST = {
    "HOSPITAL_CONSOLE": 1.0,
    "FHIR": 0.9,
    "GOVT": 0.75,
    "INFERRED": 0.5,
    "SEED": 0.3,
}


class ResourceRecord(BaseModel):
    """As normalised by the Hospital Data Normaliser agent — no confidence
    or freshness field here. Those are computed downstream by freshness.py,
    never by an agent (P1: LLM interprets, never decides)."""

    resource_key: str
    value: Any
    updated_at: datetime
    source: ResourceSource
    reporter_id: Optional[str] = None


class ResourceSnapshot(BaseModel):
    hospital_id: str
    records: list[ResourceRecord] = Field(default_factory=list)


class ScoredResourceRecord(ResourceRecord):
    """Output of freshness.py — the same record, with confidence/freshness
    attached by deterministic Python, never by an LLM."""

    confidence: float = Field(ge=0.0, le=1.0)
    freshness: Freshness
    age_minutes: float
