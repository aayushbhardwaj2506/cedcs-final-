"""
Triage schema — output contract of the Emergency Triage Assessor agent.

Matches config/tasks.yaml (`triage_assessment_task.expected_output`) in the
exported cedcs_ai_multi_agent_layer project verbatim, so the orchestrator's
Pydantic validation and the agent's actual output shape never drift apart.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Priority = Literal["CRITICAL", "HIGH", "MODERATE", "LOW"]

# P1 guardrail support: ordinal scale used by the escalate-only red-flag rule
# (never used to *downgrade* a priority — see guardrails/red_flags.py)
PRIORITY_ORDER: dict[str, int] = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}

CAPABILITY_CATEGORIES = {
    "CARDIAC",
    "NEUROLOGICAL",
    "TRAUMA",
    "RESPIRATORY",
    "OBSTETRIC",
    "PAEDIATRIC",
    "BURNS",
    "TOXICOLOGY",
    "RENAL",
    "PSYCHIATRIC",
}

# Diagnosis-shaped terms the triage rationale/category_set must never contain.
# Enforced again, independently, by guardrails/diagnostic_filter.py — this
# list is duplicated there deliberately (defense in depth, not DRY).
_DIAGNOSIS_DENYLIST = {
    "heart attack",
    "stroke",
    "mi",
    "infarction",
    "sepsis",
    "appendicitis",
    "myocardial",
    "aneurysm",
    "embolism",
    "meningitis",
}


def find_diagnosis_terms(text: str) -> list[str]:
    """Whole-word match: a plain substring test flags "mi" inside "minutes", "imminent", "administer"."""
    lowered = text.lower()
    return [t for t in sorted(_DIAGNOSIS_DENYLIST) if re.search(r"\b" + re.escape(t) + r"\b", lowered)]


class TriageResult(BaseModel):
    priority: Priority
    category_set: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    preferred_capabilities: list[str] = Field(default_factory=list)
    triage_confidence: float = Field(ge=0.0, le=1.0)
    rationale: str

    @field_validator("category_set")
    @classmethod
    def _categories_must_be_known(cls, v: list[str]) -> list[str]:
        unknown = [c for c in v if c not in CAPABILITY_CATEGORIES]
        if unknown:
            raise ValueError(
                f"category_set contains non-capability labels (possible diagnosis leak): {unknown}"
            )
        return v

    @field_validator("rationale")
    @classmethod
    def _rationale_must_not_diagnose(cls, v: str) -> str:
        hit = find_diagnosis_terms(v)
        if hit:
            raise ValueError(f"rationale appears to contain diagnosis terms: {hit}")
        return v
