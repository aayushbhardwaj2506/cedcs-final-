"""Unit tests for the guardrails package: schema-retry-with-fallback,
diagnostic-assertion filter, escalate-only red-flag rule, and the frozen-
recommendation explanation-integrity check."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from guardrails.diagnostic_filter import check_text
from guardrails.frozen_recommendation import check_explanation_integrity
from guardrails.red_flags import apply_red_flags
from guardrails.schema_retry import SchemaRetryExhausted, validate_list_with_retry, validate_with_retry
from schemas.case import Location, Patient, StructuredCase
from schemas.hospital import CandidateHospital
from schemas.recommendation import RankedHospital, RejectedHospital, ValidatedRecommendation


class _Widget(BaseModel):
    name: str
    count: int


class TestSchemaRetry:
    def test_valid_json_first_try(self):
        outcome = validate_with_retry(_Widget, '{"name": "a", "count": 1}')
        assert outcome.attempts == 1
        assert not outcome.used_fallback
        assert outcome.value.count == 1

    def test_json_wrapped_in_markdown_fence_is_extracted(self):
        raw = 'Here is the result:\n```json\n{"name": "a", "count": 2}\n```\nDone.'
        outcome = validate_with_retry(_Widget, raw)
        assert outcome.value.count == 2

    def test_retry_fn_is_invoked_on_first_failure(self):
        calls = []

        def retry_fn(error_msg: str) -> str:
            calls.append(error_msg)
            return '{"name": "fixed", "count": 3}'

        outcome = validate_with_retry(_Widget, "not json at all", retry_fn=retry_fn)
        assert len(calls) == 1
        assert outcome.attempts == 2
        assert outcome.value.name == "fixed"

    def test_exhausted_raises_without_fallback(self):
        with pytest.raises(SchemaRetryExhausted):
            validate_with_retry(_Widget, "garbage", retry_fn=lambda e: "still garbage")

    def test_fallback_used_when_both_attempts_fail(self):
        fallback = _Widget(name="default", count=0)
        outcome = validate_with_retry(
            _Widget, "garbage", retry_fn=lambda e: "still garbage", fallback=fallback
        )
        assert outcome.used_fallback
        assert outcome.value.name == "default"

    def test_validate_list_with_retry_happy_path(self):
        raw = json.dumps(
            [
                {"hospital_id": "H1", "eta_source_tier": 4, "eta_confidence": 0.4},
                {"hospital_id": "H2", "eta_source_tier": 5, "eta_confidence": 0.25},
            ]
        )
        results = validate_list_with_retry(CandidateHospital, raw)
        assert len(results) == 2
        assert results[0].hospital_id == "H1"

    def test_validate_list_with_retry_rejects_non_array(self):
        with pytest.raises(SchemaRetryExhausted):
            validate_list_with_retry(CandidateHospital, '{"hospital_id": "H1"}')


class TestDiagnosticFilter:
    def test_clean_text_passes(self):
        result = check_text("Requires cardiac monitoring capability and ICU bed availability.")
        assert result.passed

    def test_diagnosis_term_flagged(self):
        result = check_text("The patient is likely having a heart attack.")
        assert not result.passed
        assert "heart attack" in result.flagged_terms

    def test_allowlisted_cardiac_capability_not_flagged(self):
        result = check_text("Needs cardiac monitoring and a cardiac bay on arrival.")
        assert result.passed

    def test_word_boundary_avoids_false_positive_substring(self):
        # "mi " as a denylist term should not match inside "admission"
        result = check_text("Recorded on admission to the ward.")
        assert result.passed


class TestRedFlags:
    def _case(self, **patient_kwargs) -> StructuredCase:
        return StructuredCase(
            patient=Patient(**patient_kwargs),
            location=Location(lat=0.0, lng=0.0, address="x", source="GPS"),
        )

    def test_no_red_flags_leaves_priority_unchanged(self):
        case = self._case()
        final, applied = apply_red_flags(case, "LOW")
        assert final == "LOW"
        assert applied == []

    def test_absent_breathing_forces_critical(self):
        case = self._case(breathing="absent")
        final, applied = apply_red_flags(case, "LOW")
        assert final == "CRITICAL"
        assert "absent_breathing" in applied

    def test_red_flag_never_downgrades(self):
        # severe_bleeding would force HIGH, but triage already said CRITICAL —
        # max() must never downgrade CRITICAL to HIGH.
        case = self._case(bleeding="severe")
        final, applied = apply_red_flags(case, "CRITICAL")
        assert final == "CRITICAL"
        assert "severe_bleeding" in applied

    def test_multiple_red_flags_take_the_highest(self):
        case = self._case(breathing="absent", bleeding="severe")
        final, applied = apply_red_flags(case, "LOW")
        assert final == "CRITICAL"
        assert set(applied) == {"absent_breathing", "severe_bleeding"}


class TestFrozenRecommendationIntegrity:
    def _recommendation(self) -> ValidatedRecommendation:
        primary = RankedHospital(
            hospital_id="H1",
            name="Alpha Hospital",
            rank=1,
            eta_min=10.0,
            eta_confidence=0.8,
            capability_match=["ICU"],
            resource_confidence=0.7,
            overall_score=0.9,
            ranking_reason="Nearest eligible hospital.",
        )
        rejected = [
            RejectedHospital(
                hospital_id="H2",
                name="Beta Clinic",
                rejection_reason="ICU reported full",
                rejection_stage="ELIGIBILITY",
            )
        ]
        return ValidatedRecommendation(
            case_id="case-1",
            primary=primary,
            alternatives=[],
            rejected=rejected,
            caveats=[],
            generated_at="2026-09-13T00:00:00Z",
            engine_version="0.1.0",
        )

    def test_consistent_explanation_passes(self):
        rec = self._recommendation()
        explanation = (
            "**Recommended Destination:** Alpha Hospital\n"
            "Alpha Hospital was chosen for its confirmed ICU capacity.\n\n"
            "**Why Nearby Hospitals Were Not Recommended:**\n"
            "Beta Clinic was excluded because its ICU was reported full."
        )
        result = check_explanation_integrity(rec, explanation)
        assert result.passed

    def test_missing_rejection_mention_fails(self):
        rec = self._recommendation()
        explanation = "**Recommended Destination:** Alpha Hospital\nGood ICU capacity."
        result = check_explanation_integrity(rec, explanation)
        assert not result.passed
        assert "Beta Clinic" in result.missing_rejections

    def test_invented_hospital_fails(self):
        rec = self._recommendation()
        explanation = (
            "**Recommended Destination:** Alpha Hospital\nGood ICU capacity.\n"
            "**Why Nearby Hospitals Were Not Recommended:**\n"
            "Beta Clinic was excluded because its ICU was reported full.\n"
            "Gamma Medical Center was also considered."
        )
        result = check_explanation_integrity(rec, explanation)
        assert not result.passed
        assert "Gamma Medical Center" in result.unexpected_hospitals
