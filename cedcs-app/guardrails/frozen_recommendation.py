"""
Read-only/frozen-object enforcement for the Explanation Agent's input.

P1/P7 compliance: the recommendation_narrator agent must be able to
DESCRIBE the final recommendation but never CHANGE it — not reorder
hospitals, not add one that isn't there, not soften a rejection reason.

The design doc calls for enforcing this at the type level, not just via
prompt instruction (tasks.yaml's explanation_task already asks nicely —
"ABSOLUTE RULES: Do not reorder the hospitals..." — but a prompt is not a
guarantee). This module is the actual guarantee:

1. schemas/recommendation.py's ValidatedRecommendation already uses
   `model_config = ConfigDict(frozen=True)`, so any attempt to set an
   attribute on it after construction raises pydantic.ValidationError
   immediately — including from inside a tool the narrator agent might call.
2. This module adds a second, independent guarantee: a content-diff check
   run AFTER the narrator produces its explanation, verifying that every
   hospital name/id mentioned in the recommendation still appears, that no
   hospital NOT in the recommendation was introduced, and that every
   rejection reason's hospital is still referenced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from schemas.recommendation import ValidatedRecommendation


@dataclass
class IntegrityCheckResult:
    passed: bool
    missing_hospitals: list[str] = field(default_factory=list)
    unexpected_hospitals: list[str] = field(default_factory=list)
    missing_rejections: list[str] = field(default_factory=list)


def _names_mentioned_in(text: str, candidate_names: list[str]) -> set[str]:
    text_lower = text.lower()
    return {name for name in candidate_names if name.lower() in text_lower}


def check_explanation_integrity(
    recommendation: ValidatedRecommendation, explanation_text: str, extra_known_names: tuple = ()
) -> IntegrityCheckResult:
    """
    Verify the narrator's free-text explanation is consistent with the
    frozen recommendation it was given — catches an agent that silently
    dropped an alternative, invented a hospital, or omitted a rejection.

    This is a coarse name-mention check, not a semantic one — it cannot
    catch subtler tone drift (e.g. softened rejection wording), which is
    why the ABSOLUTE RULES in tasks.yaml still matter as a first line of
    defense. This is the backstop, not the whole guardrail.
    """
    expected_names = [recommendation.primary.name] + [h.name for h in recommendation.alternatives]
    rejected_names = [h.name for h in recommendation.rejected]

    mentioned = _names_mentioned_in(explanation_text, expected_names)
    missing_hospitals = [n for n in expected_names if n not in mentioned]

    missing_rejections = [
        n for n in rejected_names if n.lower() not in explanation_text.lower()
    ]

    # Detect hospital-shaped names in the explanation that don't belong to
    # ANY hospital in the recommendation (primary, alternatives, or
    # rejected) — a crude "Hospital"/"Medical Center"/"Clinic" heuristic,
    # good enough to catch an outright invented facility.
    # `extra_known_names`: other hospitals genuinely considered for this case (an advisory note may compare the top choice with
    # the fifth-ranked one). Mentioning a real candidate is not inventing a hospital.
    known_names = {n.lower() for n in expected_names + rejected_names + list(extra_known_names)}
    candidate_mentions = re.findall(
        r"[A-Z][A-Za-z.&' -]{2,60}(?:Hospital|Medical Center|Clinic|Trust|Nursing Home)\b",
        explanation_text,
    )
    # The pattern also matches a fragment of a longer real name ("Bethesda Hospital" inside "Bethesda Hospital and Child
    # Care Centre", "Sankara Eye Hospital" inside "Sankara Eye Hospital, Kanchipuram"), so a mention counts as known when
    # it is part of a known name, or contains one (the pattern also swallows preceding capitalised words: "Although LPR Hospital"). Real (OpenStreetMap) hospital names are free text, unlike the registry's.
    unexpected_hospitals = sorted(
        {m.strip() for m in candidate_mentions
         if not any(m.strip().lower() in k or k in m.strip().lower() for k in known_names)}
    )

    passed = not missing_hospitals and not unexpected_hospitals and not missing_rejections
    return IntegrityCheckResult(
        passed=passed,
        missing_hospitals=missing_hospitals,
        unexpected_hospitals=unexpected_hospitals,
        missing_rejections=missing_rejections,
    )
