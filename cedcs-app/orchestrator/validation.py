"""
Orchestrator validation gate — five checks the pipeline must pass before a
ValidatedRecommendation is allowed to reach the Explanation Agent or the
user. Each check is independent and named so a failure is traceable in the
audit log to a specific safety property, not a generic "something broke."

NOTE ON PROVENANCE: these five checks operationalise the safety properties
described across the design doc's P1-P8 principles and the orchestrator
section of the build prompt. The original build prompt text that first
listed "5 orchestrator validation checks" is not reproduced verbatim here
(it lives in CEDCS_CrewAI_Build_Prompt.md / the project docs) — this module
is my best-effort implementation of that intent, grounded in what's
consistently emphasised everywhere else in the project: P1 (no LLM
decision-making), P3 (provenance), and the escalate-only red-flag rule.
Cross-check this against §9/§10 of the build prompt before treating it as
final for the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from schemas.hospital import CandidateHospital
from schemas.recommendation import ValidatedRecommendation
from schemas.resource import ScoredResourceRecord


@dataclass
class ValidationCheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    checks: list[ValidationCheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(ValidationCheckResult(name=name, passed=passed, detail=detail))

    def failures(self) -> list[ValidationCheckResult]:
        return [c for c in self.checks if not c.passed]


def check_no_hallucinated_hospitals(
    recommendation: ValidatedRecommendation, known_hospital_ids: set[str]
) -> ValidationCheckResult:
    """V1 — every hospital_id in the recommendation (primary, alternatives,
    rejected) must trace back to a hospital that facility discovery actually
    returned. Catches a downstream bug/hallucination introducing a hospital
    nobody looked up."""
    ids_in_recommendation = {recommendation.primary.hospital_id}
    ids_in_recommendation |= {h.hospital_id for h in recommendation.alternatives}
    ids_in_recommendation |= {h.hospital_id for h in recommendation.rejected}

    unknown = ids_in_recommendation - known_hospital_ids
    return ValidationCheckResult(
        name="no_hallucinated_hospitals",
        passed=not unknown,
        detail=f"unknown hospital_ids: {sorted(unknown)}" if unknown else "ok",
    )


def check_eligibility_not_bypassed(
    recommendation: ValidatedRecommendation, eligible_hospital_ids: set[str]
) -> ValidationCheckResult:
    """V2 — the primary and every alternative must be a member of the set
    eligibility.py actually marked eligible. This is the core P1 gate: no
    hospital reaches the recommendation without having passed the
    deterministic eligibility filter."""
    recommended_ids = {recommendation.primary.hospital_id} | {
        h.hospital_id for h in recommendation.alternatives
    }
    bypassed = recommended_ids - eligible_hospital_ids
    return ValidationCheckResult(
        name="eligibility_not_bypassed",
        passed=not bypassed,
        detail=f"recommended but not eligible: {sorted(bypassed)}" if bypassed else "ok",
    )


def check_rank_ordering_consistent(recommendation: ValidatedRecommendation) -> ValidationCheckResult:
    """V3 — ranks must be strictly increasing starting at 1 across
    primary + alternatives, and overall_score must be non-increasing in
    rank order. Catches a ranking.py bug or a narrator/orchestrator step
    that silently reordered the list."""
    ordered = [recommendation.primary] + list(recommendation.alternatives)
    ranks = [h.rank for h in ordered]
    scores = [h.overall_score for h in ordered]

    ranks_ok = ranks == sorted(ranks) and ranks[0] == 1 and len(set(ranks)) == len(ranks)
    scores_ok = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    passed = ranks_ok and scores_ok
    detail_parts = []
    if not ranks_ok:
        detail_parts.append(f"ranks not strictly increasing from 1: {ranks}")
    if not scores_ok:
        detail_parts.append(f"overall_score not non-increasing with rank: {scores}")
    return ValidationCheckResult(
        name="rank_ordering_consistent", passed=passed, detail="; ".join(detail_parts) or "ok"
    )


def check_red_flag_escalation_preserved(
    recommendation: ValidatedRecommendation, red_flags_applied: list[str]
) -> ValidationCheckResult:
    """V4 — if any red-flag rule fired during triage (see
    guardrails/red_flags.py), the recommendation's `escalated` flag must be
    True and carry a non-empty escalation_reason. Structurally prevents a
    red flag from being computed but then silently dropped before it
    reaches the recommendation object."""
    if not red_flags_applied:
        return ValidationCheckResult(name="red_flag_escalation_preserved", passed=True, detail="no red flags fired")

    passed = recommendation.escalated and bool(recommendation.escalation_reason)
    return ValidationCheckResult(
        name="red_flag_escalation_preserved",
        passed=passed,
        detail=(
            "ok"
            if passed
            else f"red flags {red_flags_applied} fired but recommendation.escalated={recommendation.escalated}"
        ),
    )


def check_confidence_from_freshness_only(
    recommendation: ValidatedRecommendation,
    scored_records_by_hospital: dict[str, list[ScoredResourceRecord]],
) -> ValidationCheckResult:
    """V5 — resource_confidence on every ranked hospital must be traceable
    to freshness.py's own computed confidence values for that hospital's
    records (never a value an LLM could have invented). We check it falls
    within [min, max] of that hospital's scored record confidences as a
    tolerant but meaningful bound, since ranking.py may aggregate
    (e.g. average/min) rather than pass one record through untouched."""
    ordered = [recommendation.primary] + list(recommendation.alternatives)
    problems = []
    for h in ordered:
        records = scored_records_by_hospital.get(h.hospital_id)
        if not records:
            # A hospital that has reported nothing (a real hospital nobody has set up a console for) has no data to derive
            # confidence from. That is fine, provided the confidence is exactly zero: it must never be invented.
            if h.resource_confidence != 0.0:
                problems.append(f"{h.hospital_id}: no resource records but resource_confidence={h.resource_confidence} (must be 0.0)")
            continue
        confidences = [r.confidence for r in records]
        lo, hi = min(confidences), max(confidences)
        if not (lo - 1e-6 <= h.resource_confidence <= hi + 1e-6):
            problems.append(
                f"{h.hospital_id}: resource_confidence={h.resource_confidence} outside "
                f"observed record confidence range [{lo}, {hi}]"
            )

    return ValidationCheckResult(
        name="confidence_from_freshness_only",
        passed=not problems,
        detail="; ".join(problems) or "ok",
    )


def run_all_checks(
    recommendation: ValidatedRecommendation,
    *,
    known_hospital_ids: set[str],
    eligible_hospital_ids: set[str],
    red_flags_applied: list[str],
    scored_records_by_hospital: dict[str, list[ScoredResourceRecord]],
) -> ValidationReport:
    report = ValidationReport()
    for result in (
        check_no_hallucinated_hospitals(recommendation, known_hospital_ids),
        check_eligibility_not_bypassed(recommendation, eligible_hospital_ids),
        check_rank_ordering_consistent(recommendation),
        check_red_flag_escalation_preserved(recommendation, red_flags_applied),
        check_confidence_from_freshness_only(recommendation, scored_records_by_hospital),
    ):
        report.checks.append(result)
    return report
