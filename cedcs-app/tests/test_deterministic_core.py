"""Unit tests for deterministic_core: one per eligibility rule, freshness decay,
ranking weights, confidence boundaries, escalation."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from deterministic_core import capability_map, eligibility, engine, escalation, freshness, ranking
from deterministic_core.requirements import compute_requirements
from schemas.hospital import CandidateHospital
from schemas.resource import ResourceRecord
from schemas.triage import TriageResult

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def rec(key, value, age_min=0.0, source="HOSPITAL_CONSOLE"):
    return ResourceRecord(resource_key=key, value=value, updated_at=NOW - timedelta(minutes=age_min), source=source)


def scored(*records):
    return freshness.score_records(list(records), now=NOW)


def cand(hid, eta=10.0, tier=4, status="OPERATIONAL", ipd=True):
    return CandidateHospital(
        hospital_id=hid, name=f"H-{hid}", operating_status=status, ipd_accepting=ipd,
        eta_min=eta, eta_source_tier=tier, eta_confidence=0.4,
    )


def triage(required=(), preferred=(), priority="HIGH", conf=0.9):
    return TriageResult(
        priority=priority, category_set=["CARDIAC"], required_capabilities=list(required),
        preferred_capabilities=list(preferred), triage_confidence=conf, rationale="urgent monitoring needed",
    )


# ---------------- requirements ----------------
def test_required_wins_over_preferred():
    p = compute_requirements(triage(required=["ICU"], preferred=["ICU", "CT_SCAN"]))
    assert p.required() == {"ICU"} and p.preferred() == {"CT_SCAN"}


# ---------------- freshness ----------------
@pytest.mark.parametrize("key,age,tau", [("icu_beds", 45, 45), ("emergency_beds", 60, 30), ("specialist_CARDIOLOGY", 240, 240)])
def test_decay_formula(key, age, tau):
    s = scored(rec(key, {"available": 3}, age_min=age))[0]
    assert s.confidence == pytest.approx(math.exp(-age / tau))
    assert s.age_minutes == pytest.approx(age)


def test_source_trust_and_default_tau():
    s = scored(rec("mystery_key", {"x": 1}, age_min=0, source="GOVT"))[0]
    assert s.confidence == pytest.approx(0.75)
    assert freshness.tau_for("mystery_key") == 60


@pytest.mark.parametrize("c,label", [(0.80, "FRESH"), (0.79, "RECENT"), (0.50, "RECENT"), (0.49, "STALE"), (0.25, "STALE"), (0.24, "UNKNOWN")])
def test_freshness_classification_boundaries(c, label):
    assert freshness.classify(c) == label


def test_null_fact_is_unknown_regardless_of_age():
    s = scored(rec("icu_beds", {"total": 10, "available": None}, age_min=0))[0]
    assert s.confidence == 0.0 and s.freshness == "UNKNOWN"


# ---------------- capability map ----------------
def test_capability_statuses():
    r = scored(rec("icu_beds", {"total": 5, "available": 2}), rec("ventilators", {"total": 5, "available": 0}),
               rec("equipment_CARDIAC_MONITOR", {"operational": True}), rec("department_TRAUMA", {"active": True}),
               rec("blood_bank", {"available": True}), rec("hdu_beds", {"total": 5, "available": None}))
    ev = lambda c: capability_map.evaluate_capability(c, r).status
    assert ev("ICU") == "PRESENT" and ev("VENTILATOR") == "ABSENT"
    assert ev("CARDIAC_MONITORING") == "PRESENT" and ev("TRAUMA_BAY") == "PRESENT" and ev("BLOOD_BANK") == "PRESENT"
    assert ev("HDU") == "UNKNOWN" and ev("MRI") == "UNKNOWN" and ev("NOT_A_CAPABILITY") == "UNKNOWN"


# ---------------- eligibility (one test per rule) ----------------
def _full(icu=2):
    return scored(rec("icu_beds", {"total": 5, "available": icu}), rec("department_CARDIOLOGY", {"active": True}))


def test_rule_operating_status_rejects():
    res = eligibility.filter_eligible([cand("A", status="DIVERTING")], compute_requirements(triage(["CARDIOLOGY"])), {"A": _full()})
    assert not res.eligible and res.rejections["A"] == "facility is DIVERTING"


def test_rule_not_accepting_admissions_only_for_admission_cases():
    admit = eligibility.filter_eligible([cand("A", ipd=False)], compute_requirements(triage(["ICU"])), {"A": _full()})
    assert admit.rejections["A"] == "not currently accepting admissions"
    no_admit = eligibility.filter_eligible([cand("A", ipd=False)], compute_requirements(triage(["CARDIOLOGY"])), {"A": _full()})
    assert [c.hospital_id for c in no_admit.eligible] == ["A"]


def test_rule_absent_required_capability_rejects_with_name():
    res = eligibility.filter_eligible([cand("A")], compute_requirements(triage(["ICU", "CARDIOLOGY"])), {"A": _full(icu=0)})
    assert "ICU" in res.rejections["A"]


def test_rule_unknown_required_is_provisional_not_rejected():
    recs = scored(rec("icu_beds", {"total": 5, "available": None}))
    res = eligibility.filter_eligible([cand("A"), cand("B")], compute_requirements(triage(["ICU"])), {"A": recs, "B": _full()})
    assert {c.hospital_id for c in res.eligible} == {"A", "B"} and res.provisional_ids == {"A"}


def test_no_records_at_all_is_provisional():
    res = eligibility.filter_eligible([cand("A")], compute_requirements(triage(["ICU"])), {})
    assert res.provisional_ids == {"A"}


# ---------------- escalation ----------------
def test_red_flag_always_escalates_even_with_eligible():
    esc, reason = escalation.apply_escalation(triage(), [cand("A")], ["absent_breathing"])
    assert esc and reason.startswith("RED_FLAG_ESCALATION")


def test_no_eligible_escalates_and_red_flag_reason_preferred():
    esc, reason = escalation.apply_escalation(triage(), [], [])
    assert esc and reason.startswith("RUNG_4")
    assert escalation.apply_escalation(triage(), [], ["unresponsive"])[1].startswith("RED_FLAG")
    assert escalation.apply_escalation(triage(), [cand("A")], []) == (False, None)


# ---------------- ranking ----------------
def _hospital_records(icu_avail, eb, delay=20, age=0.0):
    return scored(
        rec("icu_beds", {"total": 10, "available": icu_avail}, age),
        rec("emergency_beds", {"total": 10, "available": eb}, age),
        rec("ipd_admission_delay_est_min", {"minutes": delay}, age),
        rec("department_CARDIOLOGY", {"active": True}, age),
        rec("specialist_CARDIOLOGY", {"status": "on_site"}, age),
        rec("equipment_CATH_LAB", {"operational": True}, age),
    )


def test_weight_table_rows_sum_to_one():
    for w in ranking.WEIGHTS.values():
        assert sum(w) == pytest.approx(1.0)


def test_priority_shifts_ranking_between_close_and_capable():
    near = cand("NEAR", eta=5)
    far = cand("FAR", eta=30)
    records = {"NEAR": scored(rec("icu_beds", {"total": 10, "available": 1}), rec("emergency_beds", {"total": 10, "available": 0}),
                              rec("ipd_admission_delay_est_min", {"minutes": 120})),
               "FAR": _hospital_records(8, 10, 0)}
    low = ranking.rank_candidates([near, far], records, triage(["ICU"], priority="LOW"))
    crit = ranking.rank_candidates([near, far], records, triage(["ICU"], priority="CRITICAL"))
    assert low[0].hospital_id == "NEAR"      # accessibility dominates when LOW
    assert crit[0].hospital_id == "FAR"      # capacity/resources dominate when CRITICAL


def test_ranks_strictly_increasing_and_scores_non_increasing():
    cands = [cand(f"H{i}", eta=5 + i) for i in range(5)]
    recs = {c.hospital_id: _hospital_records(3, 4) for c in cands}
    out = ranking.rank_candidates(cands, recs, triage(["ICU"], ["CATH_LAB", "CARDIOLOGY"]))
    assert [r.rank for r in out] == [1, 2, 3, 4, 5]
    assert all(out[i].score >= out[i + 1].score for i in range(4))


def test_provisional_ranks_below_equivalent_fresh_hospital():
    a, b = cand("PROV"), cand("FRESH")
    recs = {"PROV": _hospital_records(3, 4), "FRESH": _hospital_records(3, 4)}
    out = ranking.rank_candidates([a, b], recs, triage(["ICU"]), provisional_ids={"PROV"})
    assert [r.hospital_id for r in out] == ["FRESH", "PROV"]
    assert out[1].score == pytest.approx(out[0].score * ranking.PROVISIONAL_PENALTY)


def test_stale_data_lowers_score():
    a, b = cand("OLD"), cand("NEW")
    recs = {"OLD": _hospital_records(3, 4, age=200), "NEW": _hospital_records(3, 4, age=1)}
    out = ranking.rank_candidates([a, b], recs, triage(["ICU"]))
    assert out[0].hospital_id == "NEW"


def test_resource_confidence_within_record_range():
    recs = {"A": _hospital_records(3, 4, age=30)}
    r = ranking.rank_candidates([cand("A")], recs, triage(["ICU"]))[0]
    confs = [x.confidence for x in recs["A"]]
    assert min(confs) <= r.resource_confidence <= max(confs)


# ---------------- confidence + engine ----------------
@pytest.mark.parametrize("v,level", [(0.75, "HIGH"), (0.7499, "MODERATE"), (0.45, "MODERATE"), (0.4499, "LOW")])
def test_confidence_classification_boundaries(v, level):
    assert engine.classify_confidence(v) == level


def test_confidence_is_min_of_three_terms():
    conf, level = engine.compute_confidence(0.9, 0.9, 0.95, 0.0)  # dead tie -> margin term 0.6
    assert conf == pytest.approx(0.6) and level == "MODERATE"
    conf, level = engine.compute_confidence(0.9, 0.9, 0.95, 0.5)  # clear winner -> input term 0.81 binds
    assert conf == pytest.approx(0.81) and level == "HIGH"
    assert engine.compute_confidence(0.9, 0.9, 0.2, 0.5)[1] == "LOW"  # weak data binds


def _build(ranked, rejected=(), reasons=None, **kw):
    return engine.build_recommendation(
        case_id="C1", ranked=ranked, rejected=list(rejected), escalated=False, escalation_reason=None,
        rejection_reasons=reasons or {}, intake_completeness=0.9, triage_confidence=0.9, **kw,
    )


def test_engine_builds_frozen_recommendation_with_rejections_and_caveats():
    cands = [cand("A", eta=5), cand("B", eta=15, tier=5)]
    recs = {c.hospital_id: _hospital_records(3, 4) for c in cands}
    ranked = ranking.rank_candidates(cands, recs, triage(["ICU"]), provisional_ids={"A"})
    rec_ = _build(ranked, [cand("X", status="CLOSED")], {"X": "facility is CLOSED"})
    assert rec_.primary.rank == 1 and [a.rank for a in rec_.alternatives] == [2]
    assert rec_.rejected[0].rejection_reason == "facility is CLOSED"
    types = {c.caveat_type for c in rec_.caveats}
    assert {"RECOMMENDATION_CONFIDENCE", "ETA_TIER"} <= types or "PROVISIONAL_VERIFICATION_NEEDED" in types
    with pytest.raises(Exception):
        rec_.primary.rank = 9  # frozen


def test_engine_missing_rejection_reason_falls_back():
    ranked = ranking.rank_candidates([cand("A")], {"A": _hospital_records(3, 4)}, triage(["ICU"]))
    rec_ = _build(ranked, [cand("X")])
    assert rec_.rejected[0].rejection_reason == "did not satisfy eligibility"


def test_engine_reverification_skips_bad_primary_then_fails_closed():
    cands = [cand("A", eta=5), cand("B", eta=6)]
    recs = {"A": _hospital_records(3, 4), "B": _hospital_records(3, 4)}
    ranked = ranking.rank_candidates(cands, recs, triage(["ICU"]))
    # corrupt the top-ranked candidate's records after ranking: ICU now full
    ranked[0].records = scored(rec("icu_beds", {"total": 10, "available": 0}))
    out = _build(ranked)
    assert out.primary.hospital_id == ranked[1].hospital_id and out.primary.rank == 1
    assert any(r.rejection_stage == "ESCALATION" for r in out.rejected)
    ranked[1].records = scored(rec("icu_beds", {"total": 10, "available": 0}))
    with pytest.raises(engine.SafetyInvariantViolation):
        _build(ranked)


def test_engine_no_ranked_raises_no_eligible():
    with pytest.raises(engine.NoEligibleHospitalError):
        _build([])


def test_recommendation_carries_confidence_and_tier_cap():
    ranked = ranking.rank_candidates([cand("A")], {"A": _hospital_records(3, 4)}, triage(["ICU"]))
    out = engine.build_recommendation(
        case_id="C", ranked=ranked, rejected=[], escalated=False, escalation_reason=None,
        rejection_reasons={}, intake_completeness=1.0, triage_confidence=1.0,
    )
    assert out.confidence <= engine.ETA_TIER_CONFIDENCE_CAP[4]  # tier-4 ETA caps confidence
    assert out.confidence_level == engine.classify_confidence(out.confidence)
