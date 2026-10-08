"""Current-location support: source/accuracy carried into the case, live nearby search, reverse geocoding, and the
discovery trace that feeds the live search visual."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from fallback import rule_based
from orchestrator.orchestrator import CedcsOrchestrator
from services import locationiq
from tests.test_ai_orchestration import RAW, _cands, _snap

FACILITIES = [
    {"hospital_id": f"H{i}", "name": f"Hospital {i}", "lat": 12.9 + i / 100, "lng": 80.1, "distance_km": 1.0 + i,
     "operating_status": "DIVERTING" if i == 2 else "OPERATIONAL", "ipd_accepting": True, "trauma_level": "II", "address": "x"}
    for i in range(4)
]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("CEDCS_MODE", "offline")
    return TestClient(api_main.app)


def test_gps_source_and_accuracy_reach_the_case():
    case = rule_based.parse_intake(dict(RAW, location_source="GPS", location_accuracy_m=18.0))
    assert case.location.source == "GPS" and case.location.accuracy_m == 18.0
    assert rule_based.parse_intake(dict(RAW)).location.source == "MANUAL"  # no longer defaults to a GPS claim


def test_api_accepts_source_and_accuracy(client, monkeypatch):
    seen = {}
    real = CedcsOrchestrator.run_case
    monkeypatch.setattr(CedcsOrchestrator, "run_case", lambda self, raw: (seen.update(raw), real(self, raw))[1])
    cands = _cands(8, 15)
    monkeypatch.setattr(rule_based, "discover_facilities", lambda c, **k: (cands, 8))
    monkeypatch.setattr(rule_based, "fetch_snapshots", lambda cs: [_snap(c.hospital_id) for c in cs])
    body = {"emergency_report": "chest pain", "location_lat": 12.9, "location_lng": 80.1, "location_source": "GPS", "location_accuracy_m": 25}
    client.post("/cases", json=body)
    assert seen["location_source"] == "GPS" and seen["location_accuracy_m"] == 25


def test_poor_gps_accuracy_adds_a_caveat_and_good_accuracy_does_not(monkeypatch):
    monkeypatch.setenv("CEDCS_MODE", "offline")
    cands = _cands(8, 15)
    monkeypatch.setattr(rule_based, "discover_facilities", lambda c, **k: (cands, 8))
    monkeypatch.setattr(rule_based, "fetch_snapshots", lambda cs: [_snap(c.hospital_id) for c in cs])
    bad = CedcsOrchestrator().run_case(dict(RAW, location_source="GPS", location_accuracy_m=900))
    assert any(c.caveat_type == "LOCATION_ACCURACY" and "900" in c.detail for c in bad.recommendation.caveats)
    good = CedcsOrchestrator().run_case(dict(RAW, location_source="GPS", location_accuracy_m=15))
    assert not any(c.caveat_type == "LOCATION_ACCURACY" for c in good.recommendation.caveats)


def test_discovery_trace_lists_the_hospitals_found_for_the_live_search_visual(monkeypatch):
    monkeypatch.setenv("CEDCS_MODE", "offline")
    cands = _cands(8, 15, 20, 25)
    monkeypatch.setattr(rule_based, "discover_facilities", lambda c, **k: (cands, 8))
    monkeypatch.setattr(rule_based, "fetch_snapshots", lambda cs: [_snap(c.hospital_id) for c in cs])
    o = CedcsOrchestrator()
    o.run_case(dict(RAW))
    found = o.trace["discovery"]["hospitals"]
    assert len(found) == 4 and {"id", "name", "lat", "lng", "distance_km", "operating_status"} <= set(found[0])


def test_nearby_endpoint_returns_hospitals_and_radius(client, monkeypatch):
    monkeypatch.setattr(rule_based, "nearby_raw", lambda lat, lng, start_radius=8: (FACILITIES, 8))
    d = client.get("/nearby", params={"lat": 12.9, "lng": 80.1}).json()
    assert d["ok"] and d["radius_km"] == 8 and len(d["hospitals"]) == 4
    assert d["hospitals"][2]["operating_status"] == "DIVERTING" and d["hospitals"][0].get("source") in (None, "registry")


def test_nearby_endpoint_degrades_when_the_registry_is_down(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(rule_based, "nearby_raw", boom)
    d = client.get("/nearby", params={"lat": 12.9, "lng": 80.1}).json()
    assert d["ok"] is False and d["hospitals"] == []


def test_reverse_geocoding_shortens_and_caches(client, monkeypatch):
    monkeypatch.setenv("LOCATIONIQ_KEY", "k")
    monkeypatch.setattr(locationiq.time, "sleep", lambda s: None)
    locationiq._REVERSE_CACHE.clear()
    calls = []

    class R:
        status_code = 200

        def json(self):
            return {"display_name": "Anna Salai, Teynampet, Chennai, Tamil Nadu, 600018, India"}

    monkeypatch.setattr(locationiq._session, "get", lambda *a, **k: (calls.append(1), R())[1])
    assert client.get("/reverse", params={"lat": 13.04, "lng": 80.25}).json() == {"address": "Anna Salai, Teynampet, Chennai"}
    client.get("/reverse", params={"lat": 13.04, "lng": 80.25})
    assert len(calls) == 1
