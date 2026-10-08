"""Unit tests for the Pydantic schema layer — shape correctness and the
guardrail-relevant validators (diagnosis-term rejection, eta tier/confidence
plausibility, frozen recommendation immutability)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.case import Location, Patient, StructuredCase
from schemas.hospital import CandidateHospital
from schemas.recommendation import Caveat, RankedHospital, RejectedHospital, ValidatedRecommendation
from schemas.resource import ResourceRecord, ResourceSnapshot
from schemas.triage import TriageResult


def _make_case(**patient_overrides) -> StructuredCase:
    return StructuredCase(
        patient=Patient(**patient_overrides),
        location=Location(lat=13.08, lng=80.27, address="Chennai", source="GPS"),
    )


class TestStructuredCase:
    def test_minimal_valid_case(self):
        case = _make_case()
        assert case.patient.consciousness == "unknown"
        assert case.location.lat == 13.08

    def test_invalid_consciousness_value_rejected(self):
        with pytest.raises(ValidationError):
            _make_case(consciousness="sort-of-awake")


class TestTriageResult:
    def test_valid_result(self):
        result = TriageResult(
            priority="HIGH",
            category_set=["CARDIAC"],
            required_capabilities=["ICU", "CARDIOLOGY"],
            preferred_capabilities=[],
            triage_confidence=0.7,
            rationale="Irregular pulse pattern with chest pain requires cardiac monitoring capability.",
        )
        assert result.priority == "HIGH"

    def test_diagnosis_name_in_rationale_rejected(self):
        with pytest.raises(ValidationError):
            TriageResult(
                priority="CRITICAL",
                category_set=["CARDIAC"],
                required_capabilities=["ICU"],
                preferred_capabilities=[],
                triage_confidence=0.9,
                rationale="Patient is having a heart attack and needs immediate cardiology.",
            )

    def test_unknown_category_label_rejected(self):
        with pytest.raises(ValidationError):
            TriageResult(
                priority="LOW",
                category_set=["FLU"],  # not a recognised capability category
                required_capabilities=[],
                preferred_capabilities=[],
                triage_confidence=0.5,
                rationale="Mild symptoms reported.",
            )


class TestCandidateHospital:
    def test_seed_tier_with_plausible_confidence(self):
        c = CandidateHospital(hospital_id="H1", eta_source_tier=5, eta_confidence=0.25)
        assert c.eta_source_tier == 5

    def test_seed_tier_with_implausibly_high_confidence_rejected(self):
        with pytest.raises(ValidationError):
            CandidateHospital(hospital_id="H1", eta_source_tier=5, eta_confidence=0.95)


class TestValidatedRecommendationFrozen:
    def _make_recommendation(self) -> ValidatedRecommendation:
        primary = RankedHospital(
            hospital_id="H1",
            name="Test General Hospital",
            rank=1,
            eta_min=12.0,
            eta_confidence=0.8,
            capability_match=["ICU"],
            resource_confidence=0.7,
            overall_score=0.9,
            ranking_reason="Closest eligible hospital with confirmed ICU capacity.",
        )
        return ValidatedRecommendation(
            case_id="case-1",
            primary=primary,
            alternatives=[],
            rejected=[],
            caveats=[],
            generated_at="2026-09-13T00:00:00Z",
            engine_version="0.1.0",
        )

    def test_recommendation_is_frozen(self):
        rec = self._make_recommendation()
        with pytest.raises(ValidationError):
            rec.escalated = True  # type: ignore[misc]

    def test_nested_ranked_hospital_is_frozen(self):
        rec = self._make_recommendation()
        with pytest.raises(ValidationError):
            rec.primary.rank = 2  # type: ignore[misc]


def test_triage_rationale_filter_uses_whole_words():
    from schemas.triage import TriageResult

    ok = "imminent need for monitoring within minutes; administer oxygen; diminished responsiveness"
    TriageResult(priority="HIGH", triage_confidence=0.8, rationale=ok)  # must not trip on "mi" inside other words
    for bad in ("likely MI with sepsis", "signs of a stroke"):
        with pytest.raises(Exception):
            TriageResult(priority="HIGH", triage_confidence=0.8, rationale=bad)
