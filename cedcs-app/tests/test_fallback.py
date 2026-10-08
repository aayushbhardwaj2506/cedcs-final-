"""Rule-based fallback: intake parsing, triage rules, safety properties, explanation integrity."""

from __future__ import annotations

import pytest

from deterministic_core.capability_map import CAPABILITY_RESOURCE_MAP
from fallback import rule_based
from guardrails.diagnostic_filter import check_fields
from guardrails.red_flags import apply_red_flags

WORKED = {
    "emergency_report": "60 year old male, sudden collapse, confused, chest pain, diabetic",
    "consciousness": "confused", "breathing": "laboured", "bleeding": "none",
    "location_lat": 12.92, "location_lng": 80.1, "location_address": "Tambaram",
}


def triage_for(**over):
    return rule_based.assess_triage(rule_based.parse_intake({**WORKED, **over}))


def test_intake_extracts_worked_example_fields():
    c = rule_based.parse_intake(WORKED)
    assert c.patient.age == 60 and c.patient.gender == "male"
    assert {"chest pain", "collapse", "confusion"} <= set(c.patient.symptoms)
    assert c.patient.medical_history == ["diabetes"]
    assert "consciousness" not in c.missing_fields and 0.5 < c.intake_completeness <= 1.0


def test_intake_sparse_report_reports_missing_fields_by_impact():
    c = rule_based.parse_intake({"emergency_report": "someone is unwell", "location_lat": 1, "location_lng": 2})
    assert c.missing_fields[0] == "consciousness" and c.intake_completeness < 0.3


def test_worked_example_triage_is_critical_cardiac_with_icu():
    t = triage_for()
    assert t.priority == "CRITICAL" and "CARDIAC" in t.category_set
    assert {"EMERGENCY_DEPARTMENT", "ICU", "CARDIAC_MONITORING"} <= set(t.required_capabilities)
    assert "CARDIOLOGY" in t.preferred_capabilities


@pytest.mark.parametrize("report", [
    WORKED["emergency_report"], "road accident, fracture of leg, bleeding heavily",
    "pregnant woman in labour", "child 5 years old with fever and seizure", "he swallowed poison",
])
def test_triage_only_uses_known_capabilities_and_never_diagnoses(report):
    t = triage_for(emergency_report=report)
    assert set(t.required_capabilities + t.preferred_capabilities) <= set(CAPABILITY_RESOURCE_MAP)
    assert not set(t.required_capabilities) & set(t.preferred_capabilities)
    assert check_fields({"rationale": t.rationale}, ["rationale"])["rationale"].passed


def test_trauma_and_obstetric_and_paediatric_routing():
    assert "TRAUMA" in triage_for(emergency_report="road accident, bleeding heavily", bleeding="severe").category_set
    assert "OBSTETRICS" in triage_for(emergency_report="pregnant woman in labour").required_capabilities
    assert "PAEDIATRIC" in triage_for(emergency_report="child 5 years old with fever").category_set


def test_absent_breathing_is_critical_and_needs_ventilator():
    t = triage_for(breathing="absent", emergency_report="man not breathing")
    assert t.priority == "CRITICAL" and "VENTILATOR" in t.required_capabilities


def test_low_information_gives_low_priority_and_low_confidence():
    t = triage_for(emergency_report="mild fever", consciousness="alert", breathing="normal", bleeding="none")
    assert t.priority in ("LOW", "MODERATE") and t.triage_confidence < 0.9


def test_red_flags_still_only_escalate_over_fallback_triage():
    case = rule_based.parse_intake({**WORKED, "breathing": "absent"})
    final, applied = apply_red_flags(case, "LOW")
    assert final == "CRITICAL" and applied == ["absent_breathing"]
