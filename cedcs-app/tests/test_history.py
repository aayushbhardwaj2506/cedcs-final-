"""Persistent run history: every run saved with its latency report, events around it, fallback and failure handling."""

from __future__ import annotations

import pytest

import api.main as api_main
from services import history
from tests.test_dispatch import ADMIN_H, BODY, add_family, client, run_case  # noqa: F401  (client is a fixture)


@pytest.fixture
def hist(tmp_path, monkeypatch):
    monkeypatch.setenv("CEDCS_HISTORY", "on")
    monkeypatch.setenv("CEDCS_HISTORY_BACKEND", "sqlite")
    monkeypatch.setenv("CEDCS_HISTORY_DB", str(tmp_path / "h.db"))
    history.reset()
    yield history
    history.flush(5)
    history.reset()


def _response(case_id="c1", halted=False):
    return {"case_id": case_id, "mode": "GROQ_LLM", "total_ms": 1234.5, "halted": halted, "halt_reason": "x" if halted else None,
            "explanation_text": "report", "messages": [{"from": "ORCH"}], "audit": [{"e": 1}],
            "recommendation": None if halted else {"primary": {"name": "City Hospital", "hospital_id": "H1"}, "alternatives": []},
            "timeline": [{"stage": "PIPELINE", "start_ms": 0, "duration_ms": 1234.5, "depth": 0},
                         {"stage": "TRIAGE", "start_ms": 10, "duration_ms": 400.0, "depth": 1, "status": "ok"}],
            "trace": {"red_flags": {"final_priority": "CRITICAL"},
                      "eligibility": {"candidates": [{"status": "ELIGIBLE"}, {"status": "PROVISIONAL"}, {"status": "REJECTED"}]},
                      "llm": [{"stage": "TRIAGE", "status": "ok", "provider": "nvidia"}, {"stage": "CRITIC", "status": "fallback"}]}}


def test_a_run_is_saved_with_its_full_report_and_timings(hist):
    hist.record_case({"emergency_report": "chest pain"}, _response(), hospital_source="real")
    assert hist.flush(5)
    r = hist.recent_runs()[0]
    assert r["case_id"] == "c1" and r["priority"] == "CRITICAL" and r["primary_hospital"] == "City Hospital" and r["total_ms"] == 1234.5
    assert (r["n_candidates"], r["n_eligible"], r["n_provisional"], r["ai_ok"], r["ai_fallback"], r["providers"]) == (3, 1, 1, 1, 1, "nvidia")
    full = hist.get_run("c1")
    assert full["request"] == {"emergency_report": "chest pain"} and full["trace"]["red_flags"]["final_priority"] == "CRITICAL"
    assert full["timeline"][1]["stage"] == "TRIAGE" and full["audit"] == [{"e": 1}] and full["explanation"] == "report"
    assert hist.stage_runs() == [{"total_ms": 1234.5, "stages": {"PIPELINE": 1234.5, "TRIAGE": 400.0}}]


def test_a_halted_run_is_saved_too_and_resaving_replaces(hist):
    hist.record_case({}, _response("h1", halted=True))
    hist.record_case({}, _response("h1", halted=True))
    assert hist.flush(5)
    runs = hist.recent_runs()
    assert len(runs) == 1 and runs[0]["halted"] == 1 and runs[0]["halt_reason"] == "x"
    assert hist.stage_runs() == []  # halted runs do not count towards latency figures


def test_events_are_saved_and_filtered_by_case(hist):
    hist.event("dispatch_sent", case_id="c1", hospital_id="H1", mode="outbox")
    hist.event("hospital_report", hospital_id="H2", value={"available": 3})
    assert hist.flush(5)
    assert [e["kind"] for e in hist.recent_events(case_id="c1")] == ["dispatch_sent"]
    assert {e["kind"] for e in hist.recent_events()} == {"dispatch_sent", "hospital_report"}
    assert hist.recent_events(case_id="c1")[0]["detail"]["mode"] == "outbox"


def test_disabled_saves_nothing(hist, monkeypatch):
    monkeypatch.setenv("CEDCS_HISTORY", "off")
    hist.record_case({}, _response("x"))
    hist.event("k")
    hist.flush(2)
    assert hist.recent_runs() == [] and hist.recent_events() == []


def test_falls_back_to_the_local_file_when_the_primary_store_fails(hist, monkeypatch):
    class Broken:
        name = "neon"

        def execute(self, *a, **k):
            raise RuntimeError("connection lost")

    monkeypatch.setattr(history, "_primary", Broken())
    hist.record_case({}, _response("fb"))
    assert hist.flush(5)
    st = hist.status()
    assert st["fell_back"] == 1 and st["saved_runs"] == 1 and "saved to the local file" in st["last_error"]
    assert history._fallback is not None and history._fallback.execute("SELECT case_id FROM case_runs").fetchall()[0]["case_id"] == "fb"


def test_recording_never_raises(hist):
    hist.record_case({}, {"no": "case id"})  # malformed: must be swallowed
    hist.event("k", detail=object())
    assert hist.flush(5)


def test_pg_translate_qualifies_tables_and_placeholders():
    q = history.pg_translate("SELECT * FROM case_runs WHERE case_id=? AND x LIKE '%a'", "cedcs_history")
    assert q == "SELECT * FROM cedcs_history.case_runs WHERE case_id=%s AND x LIKE '%%a'"


def test_a_real_case_and_its_dispatch_are_saved_through_the_api(hist, client):
    add_family(client, 1)
    case = run_case(client)
    d = client.post("/dispatch", json={"case_id": case["case_id"], "confirm": True}).json()
    msg = next(m for m in d["messages"] if m["kind"] == "HOSPITAL")
    client.post(f"/dispatch/{d['id']}/simulate-ack", json={"message_id": msg["id"], "response": "accept"})
    assert hist.flush(10)

    assert client.get("/history/runs").status_code == 403  # patient descriptions: admin only
    runs = client.get("/history/runs", headers=ADMIN_H).json()["runs"]
    assert runs[0]["case_id"] == case["case_id"] and runs[0]["total_ms"] > 0 and runs[0]["primary_hospital"]
    full = client.get(f"/history/runs/{case['case_id']}", headers=ADMIN_H).json()
    assert full["request"]["emergency_report"] == BODY["emergency_report"] and full["timeline"] and full["trace"]["eligibility"]
    kinds = [e["kind"] for e in client.get("/history/events", headers=ADMIN_H, params={"case_id": case["case_id"]}).json()["events"]]
    assert "dispatch_sent" in kinds and "hospital_response" in kinds
    assert "fam0@mail.test" not in str(full["events"])  # family contact details are never saved here
    lat = client.get("/history/latency", headers=ADMIN_H).json()
    assert lat["runs"] == 1 and "PIPELINE" in lat["stages"]
    assert client.get("/history/runs/nope", headers=ADMIN_H).status_code == 404
    assert client.get("/history/status").json()["stored_runs"] == 1


def test_settings_changes_are_logged(hist, client):
    client.put("/settings", json={"ack_timeout_s": 90}, headers=ADMIN_H)
    assert hist.flush(5)
    ev = hist.recent_events()[0]
    assert ev["kind"] == "settings_changed" and ev["detail"] == {"ack_timeout_s": 90}


# ------------------------------------------------------------------ analytics and research exports
def _seed_runs(hist):
    def resp(cid, ms, prio, hosp, llm, halted=False):
        r = _response(cid, halted)
        r["total_ms"] = ms
        r["timeline"] = [{"stage": "PIPELINE", "start_ms": 0, "duration_ms": ms, "depth": 0}, {"stage": "TRIAGE", "start_ms": 5, "duration_ms": ms / 4, "depth": 2}]
        r["trace"]["red_flags"] = {"final_priority": prio}
        r["trace"]["llm"] = llm
        r["recommendation"] = None if halted else {"primary": {"name": hosp, "hospital_id": "H", "eta_min": 6.0}, "alternatives": [], "confidence_level": "LOW"}
        return r

    ok = lambda p, ms: {"stage": "TRIAGE", "status": "ok", "provider": p, "ms": ms, "tokens": 100}
    hist.record_case({}, resp("a", 10000, "CRITICAL", "City Hospital", [ok("groq", 800), ok("nvidia", 2000)]))
    hist.record_case({}, resp("b", 20000, "HIGH", "City Hospital", [ok("nvidia", 3000), {"stage": "CRITIC", "status": "fallback"}]))
    hist.record_case({}, resp("c", 30000, "CRITICAL", "Other Hospital", [ok("groq", 900)]))
    hist.record_case({}, resp("d", 500, "CRITICAL", None, [], halted=True))
    hist.event("dispatch_sent", case_id="a", hospital_id="H")
    assert hist.flush(5)


def test_analytics_numbers_are_computed_from_the_saved_runs(hist):
    from services import analytics

    _seed_runs(hist)
    a = analytics.analytics()
    assert a["window"]["runs"] == 4 and a["decisions"]["completed"] == 3 and a["decisions"]["handoffs"] == 1
    t = a["latency"]["total"]
    assert (t["n"], t["min"], t["p50"], t["max"]) == (3, 10000.0, 20000.0, 30000.0)  # the handoff does not count towards latency
    assert a["ai"]["calls"] == 5 and a["ai"]["fallbacks"] == 1 and a["ai"]["ok_rate"] == 0.8 and a["ai"]["runs_fully_ai"] == 2
    assert a["ai"]["providers"]["groq"]["calls"] == 2 and a["ai"]["providers"]["nvidia"]["latency_ms"]["p50"] == 2500.0
    assert a["decisions"]["priority"] == {"CRITICAL": 3, "HIGH": 1} and a["decisions"]["top_hospitals"][0] == {"name": "City Hospital", "count": 2}
    assert a["decisions"]["primary_eta_min"]["p50"] == 6.0 and a["decisions"]["confidence"] == {"LOW": 3}
    assert [s["stage"] for s in a["latency"]["stages"]] == ["PIPELINE", "TRIAGE"] and sum(b["count"] for b in a["latency"]["histogram"]["bins"]) == 3
    assert a["dispatch"]["dispatches"] == 1 and len(a["runs"]) == 4


def test_analytics_time_window_and_empty_history(hist):
    from services import analytics

    assert analytics.analytics()["window"]["runs"] == 0 and analytics.analytics()["latency"]["total"] is None
    _seed_runs(hist)
    import time

    assert analytics.analytics(since=time.time() + 60)["window"]["runs"] == 0


def test_exports_are_csv_with_one_row_per_run_stage_and_event(hist):
    import csv
    import io

    from services import analytics

    _seed_runs(hist)
    runs = list(csv.DictReader(io.StringIO(analytics.export_csv("runs"))))
    assert len(runs) == 4 and {"case_id", "total_ms", "priority", "primary_eta_min"} <= set(runs[0])
    assert len(list(csv.DictReader(io.StringIO(analytics.export_csv("stages"))))) == 8
    ev = list(csv.DictReader(io.StringIO(analytics.export_csv("events"))))
    assert len(ev) == 1 and ev[0]["kind"] == "dispatch_sent"
    assert analytics.export_csv("nonsense") is None


def test_analytics_endpoints_need_the_admin_token(hist, client):
    _seed_runs(hist)
    assert client.get("/history/analytics").status_code == 403 and client.get("/history/export/runs").status_code == 403
    j = client.get("/history/analytics", headers=ADMIN_H).json()
    assert j["window"]["runs"] == 4
    r = client.get("/history/export/runs", headers=ADMIN_H)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv") and "attachment" in r.headers["content-disposition"]
    assert client.get("/history/export/bogus", headers=ADMIN_H).status_code == 404
    assert client.get("/analytics").status_code == 200  # the page itself is static; its data needs the token
