"""End-to-end orchestrator run with the two CrewAI passes replaced by
test doubles (no LLM key, no crewai install). Everything between them — red
flags, deterministic core, validation gate, explanation integrity check —
is the real code."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orchestrator.orchestrator import CedcsOrchestrator
from schemas.case import Location, Patient, StructuredCase
from schemas.hospital import CandidateHospital
from schemas.resource import ResourceRecord, ResourceSnapshot
from schemas.triage import TriageResult

NOW = datetime.now(timezone.utc)


def _snapshot(hid, icu=3, cardiology=True):
    def r(key, value):
        return ResourceRecord(resource_key=key, value=value, updated_at=NOW - timedelta(minutes=5), source="HOSPITAL_CONSOLE")

    return ResourceSnapshot(hospital_id=hid, records=[
        r("department_EMERGENCY_DEPARTMENT", {"active": True}),
        r("department_CARDIOLOGY", {"active": cardiology}),
        r("icu_beds", {"total": 8, "available": icu}),
        r("emergency_beds", {"total": 8, "available": 4}),
        r("ipd_admission_delay_est_min", {"minutes": 30}),
        r("specialist_CARDIOLOGY", {"status": "on_site"}),
        r("equipment_CARDIAC_MONITOR", {"operational": True}),
    ])


@pytest.fixture
def orch(monkeypatch):
    def install(breathing="laboured"):
        case = StructuredCase(
            patient=Patient(age=60, gender="male", consciousness="confused", breathing=breathing, bleeding="none"),
            location=Location(lat=12.92, lng=80.12), intake_completeness=0.8,
        )
        triage = TriageResult(
            priority="HIGH", category_set=["CARDIAC"], triage_confidence=0.85,
            required_capabilities=["EMERGENCY_DEPARTMENT", "ICU", "CARDIAC_MONITORING"],
            preferred_capabilities=["CARDIOLOGY"], rationale="altered consciousness needs monitoring capability",
        )
        cands = [
            CandidateHospital(hospital_id="H1", name="Alpha", eta_min=8, eta_source_tier=4, eta_confidence=0.4),
            CandidateHospital(hospital_id="H2", name="Beta", eta_min=15, eta_source_tier=4, eta_confidence=0.4),
            CandidateHospital(hospital_id="H3", name="Gamma", eta_min=5, eta_source_tier=4, eta_confidence=0.4, operating_status="DIVERTING"),
            CandidateHospital(hospital_id="H4", name="Delta", eta_min=9, eta_source_tier=4, eta_confidence=0.4),
        ]
        snaps = [_snapshot("H1"), _snapshot("H2"), _snapshot("H3"), _snapshot("H4", icu=0)]
        monkeypatch.setattr(CedcsOrchestrator, "_run_front_crew", lambda self, raw, cid: (case, triage, cands, snaps))

        def fake_explain(self, rec, raw):
            names = [rec.primary.name] + [a.name for a in rec.alternatives] + [r.name for r in rec.rejected]
            return "**Recommended Destination:** " + rec.primary.name + "\n" + "\n".join(names)

        monkeypatch.setattr(CedcsOrchestrator, "_run_explanation", fake_explain)
        return CedcsOrchestrator()

    return install


def test_worked_example_produces_valid_recommendation(orch):
    result = orch().run_case({"case_id": "T1"})
    assert not result.halted, result.halt_reason
    rec = result.recommendation
    assert rec.primary.hospital_id in {"H1", "H2"}
    assert {r.hospital_id: r.rejection_reason for r in rec.rejected} == {
        "H3": "facility is DIVERTING", "H4": "required capability ICU is not available",
    }
    assert rec.escalated is False
    stages = {e.stage for e in result.audit.events}
    assert {"DETERMINISTIC_CORE", "VALIDATION_GATE"} <= stages


def test_red_flag_case_is_escalated_even_with_eligible_hospitals(orch):
    result = orch(breathing="absent").run_case({"case_id": "T2"})
    assert not result.halted, result.halt_reason
    assert result.recommendation.escalated is True
    assert result.recommendation.escalation_reason.startswith("RED_FLAG_ESCALATION")
    events = [e for e in result.audit.events if e.event_type == "ESCALATION" and e.stage == "RED_FLAGS"]
    assert "absent_breathing" in events[0].detail["rules"]


def test_no_eligible_hospital_halts_for_operator_handoff(orch, monkeypatch):
    o = orch()
    # Make every hospital's ICU full so nobody is eligible.
    snaps = [_snapshot(h, icu=0) for h in ("H1", "H2", "H3", "H4")]
    orig = CedcsOrchestrator._run_front_crew
    monkeypatch.setattr(
        CedcsOrchestrator, "_run_front_crew",
        lambda self, raw, cid: (*orig(self, raw, cid)[:3], snaps),
    )
    result = o.run_case({"case_id": "T3"})
    assert result.halted and "operator handoff" in result.halt_reason
