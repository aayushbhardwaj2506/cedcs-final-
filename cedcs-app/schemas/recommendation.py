"""
The final, validated recommendation — the frozen, read-only object handed to
the Recommendation Narrator agent as {validated_recommendation}.

This is the single most safety-critical schema in the system: it is the
ONLY thing the Explanation Agent (recommendation_narrator) is allowed to see
and talk about. It is produced exclusively by the deterministic core
(eligibility.py -> ranking.py -> engine.py), never by an LLM, and once
constructed it is immutable — see guardrails/frozen.py for the enforcement
mechanism (P1 + P7: no agent may alter a decision already made).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RankedHospital(BaseModel):
    model_config = ConfigDict(frozen=True)

    hospital_id: str
    name: str
    rank: int
    eta_min: Optional[float] = None
    eta_confidence: float
    capability_match: list[str] = Field(default_factory=list)
    resource_confidence: float
    overall_score: float
    ranking_reason: str  # plain-language, pre-written by ranking.py — narrator explains, doesn't invent


class RejectedHospital(BaseModel):
    model_config = ConfigDict(frozen=True)

    hospital_id: str
    name: str
    rejection_reason: str  # e.g. "ICU reported full", "no cardiology capability"
    rejection_stage: str  # "ELIGIBILITY" | "ESCALATION" | "RANKING_CUTOFF"


class Caveat(BaseModel):
    model_config = ConfigDict(frozen=True)

    caveat_type: str  # e.g. "ETA_TIER", "DATA_FRESHNESS", "LOW_CONFIDENCE"
    detail: str


class ValidatedRecommendation(BaseModel):
    """Immutable by construction (frozen=True cascades to all nested models).
    Any attempt by downstream code — including an agent's tool call — to
    mutate a field raises pydantic.ValidationError at the type level, not
    just by convention."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    primary: RankedHospital
    alternatives: list[RankedHospital] = Field(default_factory=list)
    rejected: list[RejectedHospital] = Field(default_factory=list)
    caveats: list[Caveat] = Field(default_factory=list)
    escalated: bool = False
    escalation_reason: Optional[str] = None
    confidence: float = 0.0
    confidence_level: str = "LOW"  # HIGH | MODERATE | LOW
    confidence_terms: dict[str, float] = Field(default_factory=dict)  # how confidence was derived
    generated_at: str
    engine_version: str
