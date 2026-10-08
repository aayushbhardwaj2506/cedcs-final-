"""Bed Checker: manual gating, bed scoring, ranking as a primary metric, live scan merge, resource-service endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from deterministic_core import freshness, ranking
from deterministic_core.beds import BED_TARGET, BED_WEIGHT, bed_detail, bed_score, relevant_bed_keys
from dispatch import store
from fallback import rule_based
from orchestrator.orchestrator import CedcsOrchestrator
from schemas.hospital import CandidateHospital
from schemas.resource import ResourceRecord, ResourceSnapshot
from schemas.triage import TriageResult
from tests.test_groq_agents import RAW

NOW = datetime.now(timezone.utc)


def rec(key, value, age_min=1.0, source="HOSPITAL_CONSOLE"):
    return ResourceRecord(resource_key=key, value=value, updated_at=NOW - timedelta(minutes=age_min), source=source)


def scored(*records):
    return freshness.score_records(list(records), now=NOW)


def cand(hid, eta=10.0):
    return CandidateHospital(hospital_id=hid, name=f"{hid} Hospital", eta_min=eta, eta_source_tier=4, eta_confidence=0.4, lat=12.9, lng=80.1, distance_km=eta / 2)


def tri(required=("ICU",), priority="HIGH"):
    return TriageResult(priority=priority, triage_confidence=0.9, required_capabilities=list(required), rationale="needs monitoring")


# ---------------------------------------------------------------- settings: OFF by default, manual only
def test_bed_checker_is_off_by_default_and_cannot_switch_itself_on():
    st = store.get_settings()
    assert st["bed_checker_enabled"] is False and st["allow_user_toggle"] is True and st["auto_dispatch"] is False
    o = CedcsOrchestrator()
    o._decide_bed_check(dict(RAW))  # user did not ask
    assert o.bed_active is False and o.trace["bed_check"] == {"active": False, "by": None}


def test_user_can_turn_it_on_for_a_case_only_if_the_admin_allows_user_toggles():
    o = CedcsOrchestrator()
    o._decide_bed_check(dict(RAW, bed_check=True))
    assert o.bed_active and o.trace["bed_check"]["by"] == "user"
    store.put_settings({"allow_user_toggle": False})
    o2 = CedcsOrchestrator()
    o2._decide_bed_check(dict(RAW, bed_check=True))
    assert not o2.bed_active and "administrator" in o2.trace["bed_check"]["blocked"]


def test_admin_switch_enables_it_for_everyone():
    store.put_settings({"bed_checker_enabled": True, "allow_user_toggle": False})
    o = CedcsOrchestrator()
    o._decide_bed_check(dict(RAW))
    assert o.bed_active and o.trace["bed_check"]["by"] == "admin"


def test_settings_reject_unknown_keys_and_wrong_types():
    store.put_settings({"bed_checker_enabled": "yes", "ack_timeout_s": True, "nonsense": 1, "auto_dispatch": True})
    st = store.get_settings()
    assert st["bed_checker_enabled"] is False and st["ack_timeout_s"] == 120 and "nonsense" not in st and st["auto_dispatch"] is True


# ---------------------------------------------------------------- scoring
def test_relevant_beds_follow_the_patient_requirements():
    assert relevant_bed_keys({"ICU", "VENTILATOR"}, set()) == ["emergency_beds", "icu_beds", "ventilators"]
    assert relevant_bed_keys({"CT_SCAN"}, {"HDU"}) == ["emergency_beds", "hdu_beds"]


def test_bed_score_saturates_discounts_stale_data_and_never_treats_unknown_as_free():
    keys = ["icu_beds"]
    full = scored(rec("icu_beds", {"total": 10, "available": BED_TARGET + 5}, age_min=1))
    one = scored(rec("icu_beds", {"total": 10, "available": 1}, age_min=1))
    old = scored(rec("icu_beds", {"total": 10, "available": BED_TARGET + 5}, age_min=300))
    none_, unknown, missing = (scored(rec("icu_beds", {"total": 10, "available": 0})),
                               scored(rec("icu_beds", {"total": 10, "available": None})), [])
    assert bed_score(full, keys) > bed_score(one, keys) > 0
    assert bed_score(old, keys) < bed_score(full, keys)  # same beds, older report -> lower
    assert bed_score(none_, keys) == 0 and bed_score(unknown, keys) == 0 and bed_score(missing, keys) == 0
    assert bed_detail(unknown, keys)["icu_beds"]["available"] is None


# ---------------------------------------------------------------- ranking
def _two_hospitals(a_beds, b_beds, a_eta=5, b_eta=15):
    cands = [cand("A", a_eta), cand("B", b_eta)]
    base = lambda beds: scored(rec("icu_beds", {"total": 10, "available": beds}), rec("emergency_beds", {"total": 10, "available": beds}),
                               rec("ipd_admission_delay_est_min", {"minutes": 20}))
    return cands, {"A": base(a_beds), "B": base(b_beds)}


def test_without_the_bed_checker_ranking_is_unchanged():
    cands, recs = _two_hospitals(1, 9)
    plain = ranking.rank_candidates(cands, recs, tri())
    assert "BEDS" not in plain[0].components and "BEDS" not in plain[0].contributions


def test_bed_availability_can_flip_the_ranking_from_nearest_to_best_supplied():
    cands, recs = _two_hospitals(a_beds=1, b_beds=9)  # A is closer, B has far more free beds
    off = ranking.rank_candidates(cands, recs, tri(priority="HIGH"))
    on = ranking.rank_candidates(cands, recs, tri(priority="HIGH"), bed_metric=True)
    assert off[0].hospital_id == "A"  # proximity wins without the bed checker
    assert on[0].hospital_id == "B"   # beds now dominate
    assert "Beds free:" in on[0].reason and "ICU 9" in on[0].reason
    assert on[0].contributions["BEDS"] > on[0].contributions["ACC"]


@pytest.mark.parametrize("priority", ["CRITICAL", "HIGH", "MODERATE", "LOW"])
def test_beds_are_the_largest_weight_at_every_priority_and_weights_still_sum_to_one(priority):
    scaled = {k: v * (1 - BED_WEIGHT) for k, v in zip(("CAP", "RES", "CAPY", "SPEC", "ACC", "FRESH"), ranking.WEIGHTS[priority])}
    assert BED_WEIGHT > max(scaled.values())  # "primary metric", whatever the urgency
    assert sum(scaled.values()) + BED_WEIGHT == pytest.approx(1.0)


# ---------------------------------------------------------------- orchestrator: live scan replaces cached beds
def _run(monkeypatch, bed_check, scan=None):
    monkeypatch.setenv("CEDCS_MODE", "offline")
    cands = [cand("H1", 8), cand("H2", 9)]
    cached = lambda hid: ResourceSnapshot(hospital_id=hid, records=[
        rec("department_EMERGENCY_DEPARTMENT", {"active": True}), rec("icu_beds", {"total": 8, "available": 1}),
        rec("emergency_beds", {"total": 8, "available": 1}), rec("equipment_CARDIAC_MONITOR", {"operational": True}),
        rec("ipd_admission_delay_est_min", {"minutes": 20})])
    monkeypatch.setattr(rule_based, "discover_facilities", lambda c, **k: (cands, 8))
    monkeypatch.setattr(rule_based, "fetch_snapshots", lambda cs: [cached(c.hospital_id) for c in cs])
    calls = []

    def fake_scan(ids):
        calls.append(ids)
        if isinstance(scan, Exception):
            raise scan
        fresh = lambda hid, n: ResourceSnapshot(hospital_id=hid, records=[rec("icu_beds", {"total": 8, "available": n}, 0.1),
                                                                            rec("emergency_beds", {"total": 8, "available": n}, 0.1)])
        return {"snapshots": [fresh("H1", 0 if False else 2), fresh("H2", 7)], "scanned_at": NOW.isoformat(), "took_ms": 42}

    monkeypatch.setattr(rule_based, "scan_beds", fake_scan)
    o = CedcsOrchestrator()
    res = o.run_case(dict(RAW, emergency_report="chest pain", bed_check=bed_check))
    return o, res, calls


def test_scan_does_not_run_unless_switched_on(monkeypatch):
    o, res, calls = _run(monkeypatch, bed_check=False)
    assert calls == [] and "beds" not in o.trace and not any(s["stage"] == "BED_CHECK" for s in o.audit.spans)
    assert not any(m["to"] == "BEDCHECK" for m in o.audit.messages)


def test_scan_runs_when_switched_on_and_its_fresh_data_drives_the_ranking(monkeypatch):
    o, res, calls = _run(monkeypatch, bed_check=True)
    assert len(calls) == 1 and set(calls[0]) == {"H1", "H2"}
    assert o.trace["beds"]["took_ms"] == 42 and {h["id"] for h in o.trace["beds"]["hospitals"]} == {"H1", "H2"}
    assert o.trace["beds"]["hospitals"][1]["beds"]["icu_beds"]["available"] == 7
    assert o.trace["ranking"]["weights"]["BEDS"] == BED_WEIGHT == max(o.trace["ranking"]["weights"].values())
    assert res.recommendation.primary.hospital_id == "H2"  # the hospital the scan showed to have 7 free beds
    assert any(s["stage"] == "BED_CHECK" for s in o.audit.spans)
    assert any(m["from"] == "BEDCHECK" and m["kind"] == "result" for m in o.audit.messages)


def test_a_failed_scan_degrades_to_last_known_beds_without_failing_the_case(monkeypatch):
    o, res, calls = _run(monkeypatch, bed_check=True, scan=RuntimeError("registry timeout"))
    assert not res.halted and "error" in o.trace["beds"]
    assert any(m["from"] == "BEDCHECK" and m["kind"] == "error" for m in o.audit.messages)


# ---------------------------------------------------------------- resource service endpoints (fake database)
class FakeCursor:
    def __init__(self, db):
        self.db, self.rows = db, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        self.db.log.append((sql.split()[0], params))
        if sql.lstrip().startswith("SELECT value FROM"):
            self.rows = [{"value": self.db.value}] if self.db.value is not None else []
        elif sql.lstrip().startswith("SELECT id, value"):
            self.rows = [{"id": 1, "value": self.db.value}] if self.db.value is not None else []
        elif sql.lstrip().startswith("UPDATE"):
            self.db.value = params[0].adapted

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class FakeConn:
    def __init__(self, db):
        self.db = db

    def cursor(self):
        return FakeCursor(self.db)

    def commit(self):
        self.db.commits += 1

    def close(self):
        pass


class FakeDB:
    def __init__(self, value):
        self.value, self.log, self.commits = value, [], 0


@pytest.fixture
def svc(monkeypatch):
    import resource_service.main as rs

    db = FakeDB({"total": 6, "available": 2})
    monkeypatch.setattr(rs, "get_conn", lambda: FakeConn(db))
    monkeypatch.setattr(rs, "WRITE_KEY", "secret")
    return TestClient(rs.app), db


def test_reserve_decrements_never_goes_below_zero_and_refuses_unknown(svc):
    client, db = svc
    hdr = {"X-Write-Key": "secret"}
    body = {"hospital_id": "H1", "bed_key": "icu_beds", "token": "abcdef123456"}
    assert client.post("/beds/reserve", json=body, headers=hdr).json() == {"ok": True, "before": 2, "after": 1}
    assert db.value["available"] == 1 and db.value["total"] == 6
    client.post("/beds/reserve", json=body, headers=hdr)
    r = client.post("/beds/reserve", json=body, headers=hdr).json()
    assert r["ok"] is False and r["reason"] == "no free bed" and db.value["available"] == 0
    db.value = {"total": 6, "available": None}
    assert client.post("/beds/reserve", json=body, headers=hdr).json()["reason"] == "bed count unknown"


def test_writes_need_the_write_key_and_a_valid_bed_type(svc):
    client, _ = svc
    body = {"hospital_id": "H1", "bed_key": "icu_beds", "token": "t"}
    assert client.post("/beds/reserve", json=body).status_code == 403
    assert client.post("/beds/reserve", json=body, headers={"X-Write-Key": "wrong"}).status_code == 403
    bad = dict(body, bed_key="parking_spaces")
    assert client.post("/beds/reserve", json=bad, headers={"X-Write-Key": "secret"}).status_code == 400
    assert client.post("/beds/report", json={"hospital_id": "H1", "bed_key": "icu_beds", "available": 3}).status_code == 403


def test_hospital_bed_report_updates_availability(svc):
    client, db = svc
    r = client.post("/beds/report", json={"hospital_id": "H1", "bed_key": "icu_beds", "available": 5, "reporter_id": "dr-rao"},
                    headers={"X-Write-Key": "secret"}).json()
    assert r["ok"] and db.value["available"] == 5 and db.commits >= 1
