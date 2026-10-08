"""Assembles the frozen ValidatedRecommendation. Fail-closed: never emits a
recommendation whose primary fails independent re-verification."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from deterministic_core.capability_map import evaluate_capability
from deterministic_core.ranking import RankedCandidate
from schemas.hospital import CandidateHospital
from schemas.recommendation import Caveat, RankedHospital, RejectedHospital, ValidatedRecommendation

logger = logging.getLogger("cedcs.engine")

ENGINE_VERSION = "cedcs-core-1.0"
MAX_ALTERNATIVES = 3
# Design doc: ETA tiers 4/5 cap overall confidence (haversine / seed ETAs are guesses).
ETA_TIER_CONFIDENCE_CAP = {4: 0.70, 5: 0.50}


class SafetyInvariantViolation(Exception):
    """A ranked hospital failed independent re-verification of REQUIRED capabilities."""


class NoEligibleHospitalError(Exception):
    """Nothing to recommend (escalation rung 4: operator handoff). The frozen
    ValidatedRecommendation requires a primary, so the orchestrator halts on this."""


def classify_confidence(value: float) -> str:
    if value >= 0.75:
        return "HIGH"
    if value >= 0.45:
        return "MODERATE"
    return "LOW"


# Deviation from the design doc's raw Conf_margin = score gap: with many equally capable
# hospitals the gap is ~0.03, which would floor every recommendation at LOW. A near-tie between
# two eligible options means either is acceptable, so it lowers confidence but does not floor it.
MARGIN_FLOOR = 0.6
MARGIN_SATURATION = 0.10  # score gap at which the margin term stops mattering


def margin_confidence(gap: float) -> float:
    return MARGIN_FLOOR + (1.0 - MARGIN_FLOOR) * min(max(gap, 0.0) / MARGIN_SATURATION, 1.0)


def compute_confidence(
    intake_completeness: float, triage_confidence: float, primary_data_confidence: float, gap: float
) -> tuple:
    conf = min(intake_completeness * triage_confidence, primary_data_confidence, margin_confidence(gap))
    return conf, classify_confidence(conf)


def _still_satisfies_required(rc: RankedCandidate) -> bool:
    for cap in rc.required:
        status = evaluate_capability(cap, rc.records).status
        if status == "ABSENT" or (status == "UNKNOWN" and not rc.provisional):
            return False
    return True


def build_recommendation(
    *,
    case_id: str,
    ranked: list[RankedCandidate],
    rejected: list[CandidateHospital],
    escalated: bool,
    escalation_reason: Optional[str],
    rejection_reasons: dict,
    intake_completeness: float,
    triage_confidence: float,
    extra_caveats: Optional[list] = None,
) -> ValidatedRecommendation:
    if not ranked:
        raise NoEligibleHospitalError("no eligible hospital to recommend")

    verified = [rc for rc in ranked if _still_satisfies_required(rc)]
    failed = [rc for rc in ranked if rc not in verified]
    if not verified:
        raise SafetyInvariantViolation(
            f"all {len(ranked)} ranked candidates failed re-verification of required capabilities"
        )

    def to_ranked(rc: RankedCandidate, rank: int) -> RankedHospital:
        return RankedHospital(
            hospital_id=rc.hospital_id,
            name=rc.candidate.name or rc.hospital_id,
            rank=rank,
            eta_min=rc.candidate.eta_min,
            eta_confidence=rc.candidate.eta_confidence,
            capability_match=rc.capability_match,
            resource_confidence=rc.resource_confidence,
            overall_score=rc.score,
            ranking_reason=rc.reason,
        )

    ordered = [to_ranked(rc, i) for i, rc in enumerate(verified[: 1 + MAX_ALTERNATIVES], start=1)]
    primary_rc = verified[0]

    rejected_out = []
    for cand in rejected:
        reason = rejection_reasons.get(cand.hospital_id)
        if reason is None:
            logger.warning("no rejection reason recorded for %s", cand.hospital_id)
            reason = "did not satisfy eligibility"
        rejected_out.append(
            RejectedHospital(
                hospital_id=cand.hospital_id, name=cand.name or cand.hospital_id,
                rejection_reason=reason, rejection_stage="ELIGIBILITY",
            )
        )
    for rc in failed:
        rejected_out.append(
            RejectedHospital(
                hospital_id=rc.hospital_id, name=rc.candidate.name or rc.hospital_id,
                rejection_reason="failed independent re-verification of required capabilities",
                rejection_stage="ESCALATION",
            )
        )

    margin = (verified[0].score - verified[1].score) if len(verified) >= 2 else MARGIN_SATURATION
    conf, level = compute_confidence(intake_completeness, triage_confidence, primary_rc.resource_confidence, margin)
    cap = ETA_TIER_CONFIDENCE_CAP.get(primary_rc.candidate.eta_source_tier)
    if cap is not None and conf > cap:
        conf, level = cap, classify_confidence(cap)

    terms = {
        "input": round(intake_completeness * triage_confidence, 4),
        "data": round(primary_rc.resource_confidence, 4),
        "margin": round(margin_confidence(margin), 4),
        "score_gap": round(margin, 4),
        "eta_cap": cap if cap is not None else 1.0,
    }
    caveats = [
        Caveat(
            caveat_type="RECOMMENDATION_CONFIDENCE",
            detail=(
                f"{level} confidence ({conf:.2f}): case information x triage certainty="
                f"{intake_completeness * triage_confidence:.2f}, resource data={primary_rc.resource_confidence:.2f}, "
                f"score margin over next option={margin:.2f}."
            ),
        )
    ]
    caveats.extend(extra_caveats or [])  # advisory AI notes (critic concerns, advisor reasoning)
    if primary_rc.provisional:
        caveats.append(
            Caveat(
                caveat_type="PROVISIONAL_VERIFICATION_NEEDED",
                detail="At least one required capability at the primary hospital could not be confirmed from "
                "current data; verify by phone before dispatch.",
            )
        )
    if primary_rc.candidate.eta_source_tier >= 4:
        caveats.append(
            Caveat(
                caveat_type="ETA_TIER",
                detail=f"Travel time is an estimate (tier {primary_rc.candidate.eta_source_tier}), not live traffic data.",
            )
        )

    return ValidatedRecommendation(
        case_id=case_id,
        primary=ordered[0],
        alternatives=ordered[1:],
        rejected=rejected_out,
        caveats=caveats,
        escalated=escalated,
        escalation_reason=escalation_reason,
        confidence=round(conf, 4),
        confidence_level=level,
        confidence_terms=terms,
        generated_at=datetime.now(timezone.utc).isoformat(),
        engine_version=ENGINE_VERSION,
    )
