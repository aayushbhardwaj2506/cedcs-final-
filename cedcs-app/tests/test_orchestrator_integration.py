"""
Integration test for the full pipeline (worked-example, matching the
definition-of-done in the original build prompt: run one realistic case
end to end and check the recommendation shape).

This test is SKIPPED unless both CEDCS_CREW_SRC (the exported CrewAI
Studio project's src/ dir) and CEDCS_DETERMINISTIC_CORE_SRC (the directory
containing your deterministic_core package) are set as environment
variables — see tests/conftest.py. It also needs a real LLM key
(OPENAI_API_KEY or whatever the crew's LLM(...) calls expect) since it
actually kicks off the CrewAI agents, so it is NOT part of the fast unit
test suite and should be run deliberately, not in every CI push, unless
you're comfortable with the token cost.

I have not been able to run this test myself in this conversation — I've
never been given your deterministic_core code or crew credentials — so
treat a first run of this file as the real verification step, not this
scaffold's presence.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not (os.environ.get("CEDCS_CREW_SRC") and os.environ.get("CEDCS_DETERMINISTIC_CORE_SRC")),
    reason=(
        "Set CEDCS_CREW_SRC and CEDCS_DETERMINISTIC_CORE_SRC to run the full "
        "pipeline integration test (see this file's docstring)."
    ),
)


WORKED_EXAMPLE_INPUTS = {
    "emergency_report": (
        "42 year old male, found collapsed at home, family says he complained "
        "of severe chest pain before collapsing. Currently conscious but "
        "confused, breathing is laboured."
    ),
    "consciousness": "confused",
    "breathing": "laboured",
    "bleeding": "none",
    "location_lat": 13.0827,
    "location_lng": 80.2707,
    "location_address": "Anna Salai, Chennai",
    "location_source": "GPS",
}


def test_full_pipeline_worked_example():
    from orchestrator.orchestrator import CedcsOrchestrator

    orchestrator = CedcsOrchestrator()
    result = orchestrator.run_case(WORKED_EXAMPLE_INPUTS)

    assert not result.halted, f"pipeline halted: {result.halt_reason}"
    assert result.recommendation is not None
    assert result.recommendation.primary.hospital_id
    assert result.explanation_text
    assert "**Recommended Destination:**" in result.explanation_text

    # Every stage should have logged at least a start/end pair to the audit trail.
    stages_seen = {e.stage for e in result.audit.events}
    for expected_stage in (
        "PIPELINE", "FRONT_CREW", "INTAKE", "TRIAGE",
        "FACILITY_DISCOVERY", "RESOURCE_INTERPRETATION",
        "DETERMINISTIC_CORE", "VALIDATION_GATE", "EXPLANATION",
    ):
        assert expected_stage in stages_seen, f"missing audit events for stage {expected_stage}"


def test_red_flag_case_is_escalated():
    from orchestrator.orchestrator import CedcsOrchestrator

    inputs = dict(WORKED_EXAMPLE_INPUTS)
    inputs["breathing"] = "absent"

    orchestrator = CedcsOrchestrator()
    result = orchestrator.run_case(inputs)

    assert not result.halted, f"pipeline halted: {result.halt_reason}"
    assert result.recommendation.escalated is True
    red_flag_events = [
        e for e in result.audit.events if e.event_type == "ESCALATION" and e.stage == "RED_FLAGS"
    ]
    assert red_flag_events, "expected a RED_FLAGS escalation audit event"
    assert "absent_breathing" in red_flag_events[0].detail.get("rules", [])
