"""Real (OpenStreetMap) hospital mode: discovery, name hints, reported data, consoles, and the honesty invariants."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from deterministic_core import facility_hints
from deterministic_core.eligibility import filter_eligible
from dispatch import store
from guardrails.frozen_recommendation import check_explanation_integrity
from schemas.hospital import CandidateHospital
from schemas.requirement import RequirementProfile
from services import hospitals_osm, registry

ADMIN = {"X-Admin-Token": api_main.ADMIN_TOKEN}


def _place(osm_id, name, lat, lon, typ="node"):
    return {"osm_id": str(osm_id), "osm_type": typ, "name": name, "lat": str(lat), "lon": str(lon), "display_name": f"{name}, Road, Area, City"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("CEDCS_MODE", "offline")
    return TestClient(api_main.app)


@pytest.fixture(autouse=True)
def _clean_cache():
    hospitals_osm.clear_cache()
    yield
    hospitals_osm.clear_cache()


# ------------------------------------------------------------------ discovery
def test_to_facility_marks_capacity_unknown_and_drops_unnamed():
    f = hospitals_osm.to_facility(_place(7, "City Hospital", 12.95, 80.1), 12.9, 80.1)
    assert f["hospital_id"] == "OSM-node-7" and f["source"] == "osm" and f["status_verified"] is False and f["phone"] is None
    assert hospitals_osm.to_facility(_place(8, "", 12.95, 80.1), 12.9, 80.1) is None
    assert hospitals_osm.to_facility({"name": "X", "osm_id": "1", "lat": "bad", "lon": "1"}, 0, 0) is None


def test_nearby_dedupes_same_building_and_respects_radius(monkeypatch):
    places = [_place(1, "Apollo Hospital", 12.9500, 80.1000), _place(2, "Apollo  Hospital", 12.9501, 80.1001, "way"),
              _place(3, "Far Hospital", 13.5, 80.1)]
    monkeypatch.setattr(hospitals_osm, "_query", lambda lat, lng: places)
    out = hospitals_osm.nearby(12.95, 80.1, radius_km=8)
    assert [f["name"] for f in out] == ["Apollo Hospital"]


def test_nearby_none_only_when_provider_is_down_everywhere(monkeypatch):
    monkeypatch.setattr(hospitals_osm, "_query", lambda lat, lng: None)
    assert hospitals_osm.nearby(12.95, 80.1) is None
    monkeypatch.setattr(hospitals_osm, "_query", lambda lat, lng: [])
    assert hospitals_osm.nearby(12.95, 80.1) == []


def test_sample_points_cover_a_wide_radius():
    pts = hospitals_osm.sample_points(12.9, 80.1, 8)
    assert len(pts) > 1 and (12.9, 80.1) in [(round(a, 4), round(b, 4)) for a, b in pts] or len(pts) > 1


# ------------------------------------------------------------------ registry switch
def test_source_is_synthetic_without_the_map_key():
    store.put_settings({"hospital_source": "real"})
    assert registry.source() == "synthetic"  # tests run with a blank LOCATIONIQ_KEY


def test_source_real_with_key(monkeypatch):
    monkeypatch.setenv("LOCATIONIQ_KEY", "k")
    store.put_settings({"hospital_source": "real"})
    assert registry.source() == "real"
    store.put_settings({"hospital_source": "synthetic"})
    assert registry.source() == "synthetic"
    assert registry.is_real_id("OSM-node-1") and not registry.is_real_id("H001")


# ------------------------------------------------------------------ name hints
@pytest.mark.parametrize("name", ["Sankara Eye Hospital", "Sugam homeopathic hospital", "Dr KK dental clinic"])
def test_excluded_facility_names(name):
    assert facility_hints.rejection_reason(name, set())


def test_specialty_hospital_allowed_only_for_its_field():
    assert facility_hints.rejection_reason("Samyuktaa Maternity Hospital", set())
    assert facility_hints.rejection_reason("Ordinary General Hospital", set()) is None


def _cand(hid, name, source="osm"):
    return CandidateHospital(hospital_id=hid, name=name, lat=1.0, lng=1.0, source=source, distance_km=1.0, eta_source_tier=2, eta_confidence=0.5)


def test_eligibility_uses_hints_for_osm_only():
    req = RequirementProfile()
    r = filter_eligible([_cand("OSM-node-1", "Sankara Eye Hospital"), _cand("OSM-node-2", "City Hospital"),
                         _cand("H1", "Eye Hospital", source="registry")], req, {})
    ids = {c.hospital_id for c in r.eligible}
    assert ids == {"OSM-node-2", "H1"} and "OSM-node-1" in r.rejections


# ------------------------------------------------------------------ reports & consoles
def test_reports_roundtrip_and_reserve():
    hid = "OSM-node-9"
    assert store.reserve_reported_bed(hid, "icu_beds")["ok"] is False  # nothing reported: refuse, never invent
    store.upsert_report(hid, "icu_beds", {"available": 1, "total": 4}, "t")
    assert store.reports_for([hid])[hid][0]["value"]["available"] == 1
    assert store.reserve_reported_bed(hid, "icu_beds") == {"ok": True, "before": 1, "after": 0}
    assert store.reserve_reported_bed(hid, "icu_beds")["reason"] == "no free bed"


def test_console_endpoints_need_admin_and_save_reports(client):
    body = {"hospital_id": "OSM-node-5", "name": "City Hospital", "email": "desk@city.org"}
    assert client.post("/admin/consoles", json=body).status_code == 403
    c = client.post("/admin/consoles", json=body, headers=ADMIN).json()
    assert client.post("/admin/consoles", json={**body, "email": "nope"}, headers=ADMIN).status_code == 422
    assert client.get("/console/wrongtoken").status_code == 404
    assert client.get(c["url"].split("/console/")[0] and f"/console/{c['token']}").status_code == 200
    r = client.post(f"/console/{c['token']}", data={"icu_beds_avail": "2", "icu_beds_total": "5"})
    assert r.status_code == 200
    assert store.reports_for(["OSM-node-5"])["OSM-node-5"][0]["value"] == {"available": 2, "total": 5}
    assert store.contact_for("OSM-node-5")["email"] == "desk@city.org"
    assert client.get("/admin/consoles").status_code == 403


# ------------------------------------------------------------------ guardrail on free-text names
def test_integrity_check_allows_fragments_of_real_names():
    from tests.test_validation_gate import _recommendation

    rec = _recommendation(primary_id="H1", rejected_ids=("H2",))
    text = f"{rec.primary.name} is best. Hospital H2 excluded."
    assert check_explanation_integrity(rec, text).passed
    assert not check_explanation_integrity(rec, text + " Also consider Invented General Hospital.").passed


# ------------------------------------------------------------------ OSM facility tags -> inferred evidence
from services import osm_tags  # noqa: E402


def test_tags_become_inferred_departments():
    d = osm_tags.departments({"emergency": "yes", "healthcare:speciality": "cardiology;paediatrics"})
    assert d["department_EMERGENCY_DEPARTMENT"] == {"active": True}
    assert d["department_CARDIOLOGY"] == {"active": True} and d["department_PEDIATRICS"] == {"active": True}
    assert osm_tags.departments({"emergency": "no"})["department_EMERGENCY_DEPARTMENT"] == {"active": False}
    assert osm_tags.departments({"name": "X"}) == {}  # a missing tag says nothing


def test_records_are_inferred_source_and_aged_from_check_date():
    rec = osm_tags.records("OSM-node-1", {"emergency": "yes", "check_date": "2025-01-01"})[0]
    assert rec.source == "INFERRED" and rec.updated_at.year == 2025
    from deterministic_core.freshness import score_records

    assert 0 < score_records([rec])[0].confidence <= 0.5


def test_details_extracts_contact_and_labels():
    d = osm_tags.details({"emergency": "yes", "phone": "044 1", "opening_hours": "24/7", "beds": "120"})
    assert d["phone"] == "044 1" and d["hours"] == "24/7" and d["beds_total"] == 120 and d["inferred"] == ["Emergency department"]


def test_inferred_only_capability_is_provisional_and_ranks_above_blank():
    from deterministic_core.freshness import score_records
    from schemas.requirement import RequirementProfile

    rec = score_records(osm_tags.records("OSM-node-2", {"emergency": "yes"}))
    req = RequirementProfile(capabilities={"EMERGENCY_DEPARTMENT": "REQUIRED"})
    r = filter_eligible([_cand("OSM-node-2", "Be Well Hospital"), _cand("OSM-node-3", "Other Hospital")], req, {"OSM-node-2": rec})
    assert {c.hospital_id for c in r.eligible} == {"OSM-node-2", "OSM-node-3"} and r.provisional_ids == {"OSM-node-2", "OSM-node-3"}
    declared_absent = score_records(osm_tags.records("OSM-node-4", {"emergency": "no"}))
    r = filter_eligible([_cand("OSM-node-4", "Vaidhyasala")], req, {"OSM-node-4": declared_absent})
    assert not r.eligible and "OSM-node-4" in r.rejections


# ------------------------------------------------------------------ an advisory note naming another real candidate must not halt a case
def test_integrity_check_knows_other_candidates_of_the_case():
    from tests.test_validation_gate import _recommendation

    rec = _recommendation(primary_id="H1")
    text = f"{rec.primary.name} is best. The advisor compared it with Aashiana Hospital."
    assert not check_explanation_integrity(rec, text).passed  # unknown to the check: looks invented
    assert check_explanation_integrity(rec, text, extra_known_names=("Aashiana Hospital",)).passed  # a real candidate of this case
    assert not check_explanation_integrity(rec, text + " Also Invented General Hospital.", extra_known_names=("Aashiana Hospital",)).passed


def test_template_can_leave_out_ai_notes():
    from fallback import rule_based
    from schemas.recommendation import Caveat
    from tests.test_validation_gate import _recommendation

    rec = _recommendation(primary_id="H1")
    rec = rec.model_copy(update={"caveats": [Caveat(caveat_type="AI_SECOND_OPINION", detail="Prefer Aashiana Hospital."),
                                             Caveat(caveat_type="DATA_FRESHNESS", detail="Data is old.")]})
    assert "Aashiana" in rule_based.explain(rec)
    plain = rule_based.explain(rec, include_ai_notes=False)
    assert "Aashiana" not in plain and "Data is old." in plain  # only AI-contributed notes are dropped
