"""Output contracts for the advisory AI agents (critic, ranking advisor).

Both are ADVISORY inputs to the deterministic core, never decisions:
  * the critic can only raise priority, add PREFERRED capabilities, and request human review;
  * the advisor can only nudge scores of hospitals that already passed the hard eligibility gate,
    by at most MAX_AI_ADJUSTMENT, and every nudge is logged and shown to the operator.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from schemas.triage import Priority, find_diagnosis_terms

MAX_AI_ADJUSTMENT = 0.05  # absolute cap on any advisor score nudge


def _no_diagnosis(text: str) -> str:
    hit = find_diagnosis_terms(text)
    if hit:
        raise ValueError(f"text appears to contain diagnosis terms: {hit}")
    return text


class CriticReview(BaseModel):
    concerns: list[str] = Field(default_factory=list, max_length=6)
    contradictions: list[str] = Field(default_factory=list, max_length=4)
    suggested_priority: Optional[Priority] = None  # only ever applied if HIGHER than the current one
    additional_preferred_capabilities: list[str] = Field(default_factory=list)
    needs_human_review: bool = False
    rationale: str = ""

    @field_validator("concerns", "contradictions")
    @classmethod
    def _items_no_diagnosis(cls, v: list[str]) -> list[str]:
        return [_no_diagnosis(x) for x in v]

    @field_validator("rationale")
    @classmethod
    def _rationale_no_diagnosis(cls, v: str) -> str:
        return _no_diagnosis(v)


class AdvisorOpinion(BaseModel):
    preferred_hospital_id: Optional[str] = None
    adjustments: dict[str, float] = Field(default_factory=dict)  # hospital_id -> score nudge, clipped by the core
    reasoning: str = ""
    concerns: list[str] = Field(default_factory=list, max_length=5)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("reasoning")
    @classmethod
    def _reasoning_no_diagnosis(cls, v: str) -> str:
        return _no_diagnosis(v)

    @field_validator("concerns")
    @classmethod
    def _concerns_no_diagnosis(cls, v: list[str]) -> list[str]:
        return [_no_diagnosis(x) for x in v]
