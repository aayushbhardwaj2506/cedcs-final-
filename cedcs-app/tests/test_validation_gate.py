"""Unit tests for the five orchestrator validation-gate checks
(orchestrator/validation.py) — each test isolates exactly one check's
failure mode so a regression points at the right safety property."""

from __future__ import annotations

from datetime import datetime, timezone

from schemas.recommendation import RankedHospital, RejectedHospital, ValidatedRecommendation
from schemas.resource import ScoredResourceRecord
from orchestrator.validation import (
    check_confidence_from_freshness_only,
    check_eligibility_not_bypassed,
    check_no_hallucinated_hospitals,
    check_rank_ordering_consistent,
    check_red_flag_escalation_preserved,
    run_all_checks,
)


def _ranked(hospital_id: str, rank: int, score: float, resource_confidence: float = 0.6) -> RankedHospital:
    return RankedHospital(
        hospital_id=hospital_id,
        name=f"Hospital {hospital_id}",
        rank=rank,
        eta_min=10.0,
        eta_confidence=0.8,
        capability_match=["ICU"],
        resource_confidence=resource_confidence,
        overall_score=score,
        ranking_reason="test",
    )


def _recommendation(
    primary_id="H1", alt_ids=(), rejected_ids=(), escalated=False, escalation_reason=None,
    resource_confidences=None,
):
    resource_confidences = resource_confidences or {}
    primary = _ranked(primary_id, 1, 0.9, resource_confidence=resource_confidences.get(primary_id, 0.6))
    alternatives = [
        _ranked(hid, i + 2, 0.9 - (i + 1) * 0.1, resource_confidence=resource_confidences.get(hid, 0.6))
        for i, hid in enumerate(alt_ids)
    ]
    rejected = [
        RejectedHospital(hospital_id=hid, name=f"Hospital {hid}", rejection_reason="no capability", rejection_stage="ELIGIBILITY")
        for hid in rejected_ids
    ]
    return ValidatedRecommendation(
        case_id="case-1",
        primary=primary,
        alternatives=alternatives,
        rejected=rejected,
        caveats=[],
        escalated=escalated,
        escalation_reason=escalation_reason,
        generated_at="2026-09-13T00:00:00Z",
        engine_version="0.1.0",
    )


def _scored_record(hospital_id: str, confidence: float) -> ScoredResourceRecord:
    return ScoredResourceRecord(
        resource_key="icu_beds_available",
        value=2,
        updated_at=datetime.now(timezone.utc),
        source="SEED",
        reporter_id=None,
        confidence=confidence,
        freshness="RECENT",
        age_minutes=30.0,
    )


class TestNoHallucinatedHospitals:
    def test_all_known_passes(self):
        rec = _recommendation(primary_id="H1", alt_ids=["H2"], rejected_ids=["H3"])
        result = check_no_hallucinated_hospitals(rec, known_hospital_ids={"H1", "H2", "H3"})
        assert result.passed

    def test_unknown_hospital_fails(self):
        rec = _recommendation(primary_id="H1", alt_ids=["H2"])
        result = check_no_hallucinated_hospitals(rec, known_hospital_ids={"H1"})
        assert not result.passed
        assert "H2" in result.detail


class TestEligibilityNotBypassed:
    def test_recommended_within_eligible_passes(self):
        rec = _recommendation(primary_id="H1", alt_ids=["H2"])
        result = check_eligibility_not_bypassed(rec, eligible_hospital_ids={"H1", "H2", "H3"})
        assert result.passed

    def test_recommended_outside_eligible_fails(self):
        rec = _recommendation(primary_id="H1", alt_ids=["H2"])
        result = check_eligibility_not_bypassed(rec, eligible_hospital_ids={"H1"})
        assert not result.passed
        assert "H2" in result.detail


class TestRankOrderingConsistent:
    def test_strictly_increasing_ranks_and_non_increasing_scores_passes(self):
        rec = _recommendation(primary_id="H1", alt_ids=["H2", "H3"])
        result = check_rank_ordering_consistent(rec)
        assert result.passed

    def test_score_increasing_with_rank_fails(self):
        rec = _recommendation(primary_id="H1", alt_ids=["H2"])
        # tamper: alternative scores higher than primary
        bumped_alt = rec.alternatives[0].model_copy(update={"overall_score": 0.99})
        rec = rec.model_copy(update={"alternatives": [bumped_alt]})
        result = check_rank_ordering_consistent(rec)
        assert not result.passed


class TestRedFlagEscalationPreserved:
    def test_no_red_flags_always_passes(self):
        rec = _recommendation()
        result = check_red_flag_escalation_preserved(rec, red_flags_applied=[])
        assert result.passed

    def test_red_flags_fired_but_not_escalated_fails(self):
        rec = _recommendation(escalated=False)
        result = check_red_flag_escalation_preserved(rec, red_flags_applied=["absent_breathing"])
        assert not result.passed

    def test_red_flags_fired_and_escalated_passes(self):
        rec = _recommendation(escalated=True, escalation_reason="absent breathing red flag")
        result = check_red_flag_escalation_preserved(rec, red_flags_applied=["absent_breathing"])
        assert result.passed


class TestConfidenceFromFreshnessOnly:
    def test_confidence_within_observed_range_passes(self):
        rec = _recommendation(primary_id="H1")
        rec.primary.__dict__  # no-op; primary.resource_confidence = 0.6 from _ranked default
        scored = {"H1": [_scored_record("H1", 0.5), _scored_record("H1", 0.7)]}
        result = check_confidence_from_freshness_only(rec, scored)
        assert result.passed

    def test_confidence_outside_observed_range_fails(self):
        rec = _recommendation(primary_id="H1")
        scored = {"H1": [_scored_record("H1", 0.05), _scored_record("H1", 0.1)]}
        result = check_confidence_from_freshness_only(rec, scored)
        assert not result.passed

    def test_no_records_with_nonzero_confidence_fails(self):
        """Confidence with nothing behind it would be invented."""
        rec = _recommendation(primary_id="H1")  # helper gives resource_confidence 0.6
        result = check_confidence_from_freshness_only(rec, {})
        assert not result.passed and "must be 0.0" in result.detail

    def test_no_records_with_zero_confidence_passes(self):
        """A real hospital that has reported nothing: unknown is allowed, as long as it is scored as unknown."""
        rec = _recommendation(primary_id="H1", resource_confidences={"H1": 0.0})
        assert check_confidence_from_freshness_only(rec, {}).passed


class TestRunAllChecks:
    def test_all_pass_end_to_end(self):
        rec = _recommendation(
            primary_id="H1", alt_ids=["H2"], rejected_ids=["H3"],
            resource_confidences={"H1": 0.6, "H2": 0.5},
        )
        scored = {
            "H1": [_scored_record("H1", 0.6)],
            "H2": [_scored_record("H2", 0.5)],
        }
        report = run_all_checks(
            rec,
            known_hospital_ids={"H1", "H2", "H3"},
            eligible_hospital_ids={"H1", "H2"},
            red_flags_applied=[],
            scored_records_by_hospital=scored,
        )
        assert report.passed
        assert len(report.checks) == 5

    def test_failure_is_isolated_to_the_right_check(self):
        rec = _recommendation(
            primary_id="H1", alt_ids=["H2"], rejected_ids=["H3"],
            resource_confidences={"H1": 0.6, "H2": 0.5},
        )
        scored = {"H1": [_scored_record("H1", 0.6)], "H2": [_scored_record("H2", 0.5)]}
        report = run_all_checks(
            rec,
            known_hospital_ids={"H1", "H2", "H3"},
            eligible_hospital_ids={"H1"},  # H2 recommended but not eligible
            red_flags_applied=[],
            scored_records_by_hospital=scored,
        )
        assert not report.passed
        failed_names = {f.name for f in report.failures()}
        assert failed_names == {"eligibility_not_bypassed"}
