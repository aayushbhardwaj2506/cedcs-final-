"""API, streaming, timing spans, decision trace, metrics, negation handling.
Uses the rule-based front end with discovery/resources monkeypatched (no DB, no network)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from fallback import rule_based
from orchestrator.audit import AuditLog
from schemas.hospital import CandidateHospital
from schemas.resource import ResourceRecord, ResourceSnapshot

NOW = datetime.now(timezone.utc)
BODY = {
    "emergency_report": "60 year old male, sudden collapse, confused, chest pain, diabetic",
    "consciousness": "confused", "breathing": "laboured", "bleeding": "none",
    "location_lat": 12.9, "location_lng": 80.1,
}


def _snap(hid, icu=3):
    def r(k, v):
        return ResourceRecord(resource_key=k, value=v, updated_at=NOW - timedelta(minutes=3), source="HOSPITAL_CONSOLE")

    return ResourceSnapshot(hospital_id=hid, records=[
        r("department_EMERGENCY_DEPARTMENT", {"active": True}), r("department_CARDIOLOGY", {"active": True}),
        r("icu_beds", {"total": 8, "available": icu}), r("emergency_beds", {"total": 8, "available": 4}),
        r("equipment_CARDIAC_MONITOR", {"operational": True}), r("specialist_CARDIOLOGY", {"status": "on_site"}),
        r("ipd_admission_delay_est_min", {"minutes": 20}),
    ])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("CEDCS_MODE", "offline")
    api_main._RUNS.clear()

    def disc(case, **kw):
        return [
            CandidateHospital(hospital_id=h, name=f"{h} Hospital", eta_min=eta, eta_source_tier=4, eta_confidence=0.4,
                              lat=12.9 + i * 0.01, lng=80.1, distance_km=eta / 2)
            for i, (h, eta) in enumerate([("Alpha", 10), ("Beta", 20)])
        ], 8

    monkeypatch.setattr(rule_based, "discover_facilities", disc)
    monkeypatch.setattr(rule_based, "fetch_snapshots", lambda cands: [_snap(c.hospital_id) for c in cands])
    return TestClient(api_main.app)


def test_cases_returns_timeline_and_trace(client):
    d = client.post("/cases", json=BODY).json()
    assert not d["halted"] and d["mode"] == "RULE_BASED_FALLBACK" and d["total_ms"] > 0
    stages = {s["stage"]: s for s in d["timeline"]}
    for name in ("PIPELINE", "INTAKE", "TRIAGE", "FACILITY_DISCOVERY", "RESOURCE_INTERPRETATION", "RED_FLAGS",
                 "FRESHNESS", "ELIGIBILITY", "RANKING", "ENGINE", "VALIDATION_GATE", "EXPLANATION"):
        assert name in stages, name
    # nesting: children sit inside their parent's time window
    p, f = stages["PIPELINE"], stages["FRONT_CREW"]
    assert f["depth"] == p["depth"] + 1 and f["start_ms"] >= p["start_ms"]
    assert f["start_ms"] + f["duration_ms"] <= p["start_ms"] + p["duration_ms"] + 0.01
    for key in ("intake", "triage", "red_flags", "eligibility", "ranking", "confidence", "validation", "freshness"):
        assert key in d["trace"], key
    assert all(c["passed"] for c in d["trace"]["validation"])
    assert d["trace"]["confidence"]["terms"]["input"] > 0


def test_response_and_stream_carry_the_orchestration_messages(client):
    d = client.post("/cases", json=BODY).json()
    assert d["messages"] and {"from", "to", "kind", "label", "t_ms"} <= set(d["messages"][0])
    assert d["messages"][0]["from"] == "ORCH" and d["messages"][0]["kind"] == "command"
    assert [m["t_ms"] for m in d["messages"]] == sorted(m["t_ms"] for m in d["messages"])
    with client.stream("POST", "/cases/stream", json=BODY) as resp:
        types = [json.loads(l[6:])["type"] for l in resp.iter_lines() if l.startswith("data: ")]
    assert types.count("message") == len(d["messages"]) and types.index("message") < types.index("result")
    assert "plan" in d["trace"] and "clarification" in d["trace"]


def test_ranking_trace_contributions_sum_to_score(client):
    r = client.post("/cases", json=BODY).json()["trace"]["ranking"]
    for x in r["ranked"]:
        assert sum(x["contributions"].values()) * x["penalty"] == pytest.approx(x["score"])


def test_stream_emits_live_events_then_result(client):
    with client.stream("POST", "/cases/stream", json=BODY) as resp:
        events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
    types = [e["type"] for e in events]
    assert types[-1] == "result" and "stage_start" in types and "stage_end" in types and "trace" in types
    assert types.index("stage_start") < types.index("result")
    assert events[-1]["data"]["recommendation"]["primary"]["name"]


def test_halt_returns_handoff_and_closed_timeline(client, monkeypatch):
    monkeypatch.setattr(rule_based, "fetch_snapshots", lambda cands: [_snap(c.hospital_id, icu=0) for c in cands])
    d = client.post("/cases", json=BODY).json()
    assert d["halted"] and d["handoff_required"] and d["recommendation"] is None
    assert all("duration_ms" in s for s in d["timeline"])  # no span left open


def test_metrics_aggregate_and_reset(client):
    for _ in range(3):
        client.post("/cases", json=BODY)
    m = client.get("/metrics").json()
    assert m["runs"] == 3 and m["stages"]["RANKING"]["count"] == 3
    assert m["total"]["p50_ms"] <= m["total"]["p95_ms"] <= m["total"]["max_ms"]
    client.delete("/metrics")
    assert client.get("/metrics").json()["runs"] == 0


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "CEDCS Command View" in r.text


def test_audit_spans_nest_and_close_open():
    a = AuditLog()
    a.stage_start("OUTER"); a.stage_start("INNER"); a.stage_end("INNER")
    a.close_open("halted")
    spans = {s["stage"]: s for s in a.spans}
    assert spans["INNER"]["depth"] == 1 and spans["OUTER"]["status"] == "halted"


def test_audit_broken_callback_never_breaks_pipeline():
    a = AuditLog(on_event=lambda e: 1 / 0)
    a.stage_start("X"); a.stage_end("X")
    assert a.spans[0]["stage"] == "X"


# ---------------- negation ----------------
@pytest.mark.parametrize("text,has", [
    ("no chest pain", False), ("denies chest pain", False), ("without chest pain, but collapsed", False),
    ("chest pain", True), ("severe chest pain, no fever", True), ("has chest pain and no fever", True),
])
def test_negated_symptoms_are_not_extracted(text, has):
    syms = rule_based.parse_intake({"emergency_report": text}).patient.symptoms
    assert ("chest pain" in syms) == has
    if "no fever" in text:
        assert "fever" not in syms
