"""Advisory AI agents and the orchestration bus: escalate-only critic, bounded advisor,
adaptive invocation, parallel spans, and the message stream that drives the live diagram."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from deterministic_core import ranking
from fallback import rule_based
from llm import groq_agents
from orchestrator.audit import AuditLog
from orchestrator.orchestrator import CedcsOrchestrator
from schemas.agents import MAX_AI_ADJUSTMENT, AdvisorOpinion, CriticReview
from schemas.hospital import CandidateHospital
from schemas.resource import ResourceRecord, ResourceSnapshot
from schemas.triage import TriageResult
from tests.test_groq_agents import RAW, TRIAGE_OK, FakeResp, groq  # noqa: F401  (fixture)

NOW = datetime.now(timezone.utc)


def _snap(hid, icu=3):
    def r(k, v):
        return ResourceRecord(resource_key=k, value=v, updated_at=NOW - timedelta(minutes=2), source="HOSPITAL_CONSOLE")

    return ResourceSnapshot(hospital_id=hid, records=[
        r("department_EMERGENCY_DEPARTMENT", {"active": True}), r("icu_beds", {"total": 8, "available": icu}),
        r("emergency_beds", {"total": 8, "available": 4}), r("equipment_CARDIAC_MONITOR", {"operational": True}),
        r("ipd_admission_delay_est_min", {"minutes": 20}),
    ])


def _cands(*etas):
    return [CandidateHospital(hospital_id=f"H{i}", name=f"H{i} Hospital", eta_min=e, eta_source_tier=4, eta_confidence=0.4,
                              lat=12.9 + i * .01, lng=80.1, distance_km=e / 2) for i, e in enumerate(etas)]


# ---------------- ranking adjustments are bounded ----------------
def test_ai_adjustment_is_clipped_and_reorders_only_within_reach():
    from deterministic_core import freshness

    cands = _cands(10, 10)
    recs = {c.hospital_id: freshness.score_records(_snap(c.hospital_id).records) for c in cands}
    tri = TriageResult(priority="HIGH", triage_confidence=0.9, required_capabilities=["ICU"], rationale="needs monitoring")
    base = ranking.rank_candidates(cands, recs, tri)
    loser = base[1].hospital_id
    nudged = ranking.rank_candidates(cands, recs, tri, ai_adjustments={loser: 5.0})  # absurd request
    assert nudged[0].hospital_id == loser and nudged[0].ai_adjustment == MAX_AI_ADJUSTMENT
    assert nudged[0].score == pytest.approx(base[1].score + MAX_AI_ADJUSTMENT)
    assert [r.rank for r in nudged] == [1, 2]


def test_advisor_opinion_and_critic_reject_diagnosis_language():
    with pytest.raises(Exception):
        CriticReview(concerns=["possible sepsis"])
    with pytest.raises(Exception):
        AdvisorOpinion(reasoning="probable stroke, pick H1")
    CriticReview(concerns=["imminent deterioration within minutes"])  # whole-word matching: allowed


# ---------------- critic is escalate-only ----------------
def test_critic_can_raise_but_never_lower_and_flags_human_review():
    o = CedcsOrchestrator()
    tri = TriageResult(priority="HIGH", triage_confidence=0.9, required_capabilities=["ICU"], preferred_capabilities=["CT_SCAN"],
                       rationale="needs monitoring")
    lower = o._apply_critic(tri, CriticReview(suggested_priority="LOW", concerns=["fine"]))
    assert lower.priority == "HIGH"
    higher = o._apply_critic(tri, CriticReview(suggested_priority="CRITICAL", additional_preferred_capabilities=["BLOOD_BANK", "ICU"],
                                                needs_human_review=True, concerns=["breathing contradicts report"]))
    assert higher.priority == "CRITICAL" and higher.preferred_capabilities == ["CT_SCAN", "BLOOD_BANK"]
    assert higher.required_capabilities == ["ICU"]  # never touches REQUIRED
    assert o.ai_flags and any(c.caveat_type == "AI_CRITIC" for c in o.extra_caveats)


def test_bare_human_review_request_is_ignored_but_real_contradiction_is_honoured():
    tri = TriageResult(priority="HIGH", triage_confidence=0.9, required_capabilities=["ICU"], rationale="needs monitoring")
    o = CedcsOrchestrator()
    o._apply_critic(tri, CriticReview(needs_human_review=True, concerns=["no vitals"]))
    assert not o.ai_flags and o.trace["critic"]["applied"]["human_review_ignored"]
    o2 = CedcsOrchestrator()
    o2._apply_critic(tri, CriticReview(needs_human_review=True, contradictions=["text says not breathing, field says normal"]))
    assert o2.ai_flags


def test_ai_escalation_is_labelled_separately_from_hard_red_flags():
    from deterministic_core.escalation import apply_escalation

    tri = TriageResult(priority="HIGH", triage_confidence=0.9, rationale="x")
    assert apply_escalation(tri, [object()], ["AI_CRITIC_REVIEW: x"])[1].startswith("AI_REVIEW_ESCALATION")
    assert apply_escalation(tri, [object()], ["absent_breathing", "AI_CRITIC_REVIEW: x"])[1].startswith("RED_FLAG_ESCALATION")


# ---------------- adaptive advisor ----------------
def _core_setup(monkeypatch, etas, icu=3):
    monkeypatch.setenv("CEDCS_MODE", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    cands = _cands(*etas)
    snaps = [_snap(c.hospital_id, icu=icu) for c in cands]
    case = rule_based.parse_intake(RAW)
    tri = TriageResult(priority="HIGH", triage_confidence=0.9, required_capabilities=["ICU"], rationale="needs monitoring")
    return case, tri, cands, snaps


def _run_core(o, case, tri, cands, snaps):
    return o._run_deterministic_core(case_id="C", structured_case=case, triage=tri, candidates=cands, snapshots=snaps, red_flags_applied=[])


def test_advisor_skipped_on_clear_winner(monkeypatch, groq):
    case, tri, cands, snaps = _core_setup(monkeypatch, [5, 60])  # far apart -> accessibility gap is large
    o = CedcsOrchestrator()
    rec, _ = _run_core(o, case, tri, cands, snaps)
    assert o.trace["advisor"]["called"] is False and not groq["seen"]  # no LLM call spent
    assert any(m["kind"] == "skip" and m["to"] == "ADVISOR" for m in o.audit.messages)
    assert any("Advisor skipped" in d["decision"] for d in o.plan)


def test_advisor_consulted_on_near_tie_and_can_change_top_within_bounds(monkeypatch, groq):
    case, tri, cands, snaps = _core_setup(monkeypatch, [10, 10])  # identical hospitals -> dead tie
    o = CedcsOrchestrator()
    # discover which one the engine ranks second, then have the advisor prefer it
    from deterministic_core import freshness
    recs = {c.hospital_id: freshness.score_records(s.records) for c, s in zip(cands, snaps)}
    second = ranking.rank_candidates(cands, recs, tri)[1].hospital_id
    groq["route"] = {"Ranking Advisor": FakeResp(json.dumps(
        {"preferred_hospital_id": second, "adjustments": {second: 0.9}, "reasoning": "Safer choice at equal ETA.", "confidence": 0.7}))}
    rec, gate = _run_core(o, case, tri, cands, snaps)
    adv = o.trace["advisor"]
    assert adv["called"] and adv["changed_top"] and rec.primary.hospital_id == second
    assert adv["adjustments"][second] == MAX_AI_ADJUSTMENT  # 0.9 asked, clipped
    assert any(c.caveat_type == "AI_SECOND_OPINION" for c in rec.caveats)
    assert second in gate["eligible_hospital_ids"]  # the advisor can never introduce a non-eligible hospital
    assert any(s["stage"] == "RE_RANK" for s in o.audit.spans)


def test_advisor_failure_keeps_engine_ranking(monkeypatch, groq):
    case, tri, cands, snaps = _core_setup(monkeypatch, [10, 10])
    groq["route"] = {"Ranking Advisor": FakeResp(status=500)}
    o = CedcsOrchestrator()
    rec, _ = _run_core(o, case, tri, cands, snaps)
    assert "error" in o.trace["advisor"] and rec.primary.hospital_id in {"H0", "H1"}
    assert any(m["kind"] == "error" and m["from"] == "ADVISOR" for m in o.audit.messages)


# ---------------- parallel spans + message bus ----------------
def test_parallel_threads_keep_correct_depths():
    a = AuditLog()
    a.stage_start("ROOT")
    base = a.current_depth()
    barrier = threading.Barrier(2)

    def worker(name):
        a.adopt_depth(base)
        a.stage_start(name)
        barrier.wait()  # both are open at the same moment
        a.stage_end(name)

    ts = [threading.Thread(target=worker, args=(n,)) for n in ("A", "B")]
    [t.start() for t in ts]
    [t.join() for t in ts]
    a.stage_end("ROOT")
    depth = {s["stage"]: s["depth"] for s in a.spans}
    assert depth["ROOT"] == 0 and depth["A"] == depth["B"] == 1


def test_full_run_emits_paired_messages_and_parallel_stages(monkeypatch, groq):
    monkeypatch.setenv("CEDCS_MODE", "offline")
    cands = _cands(8, 15)
    monkeypatch.setattr(rule_based, "discover_facilities", lambda c, **k: (cands, 8))
    monkeypatch.setattr(rule_based, "fetch_snapshots", lambda cs: [_snap(c.hospital_id) for c in cs])
    events = []
    o = CedcsOrchestrator(audit=AuditLog(on_event=events.append))
    res = o.run_case(dict(RAW))
    assert not res.halted, res.halt_reason
    msgs = o.audit.messages
    # every command to an agent/service gets a reply or skip back
    for to in ("INTAKE", "TRIAGE", "CLARIFY", "REGISTRY", "RED_FLAGS", "CORE", "VALIDATOR", "NARRATOR"):
        assert any(m["to"] == to and m["kind"] == "command" for m in msgs), to
        assert any(m["from"] == to and m["kind"] in ("result", "error") for m in msgs), to
    assert [m for m in events if m["type"] == "message"]  # streamed live too
    spans = {s["stage"]: s for s in o.audit.spans}
    assert spans["TRIAGE"]["depth"] == spans["CLARIFICATION"]["depth"]  # siblings, not nested
    assert len(o.plan) >= 2 and "background" in o.plan[0]["decision"] and "parallel" in o.plan[1]["decision"]
    assert o.trace["clarification"]["questions"] is not None


# ---------------- text vs toggle reconciliation (escalate-only) ----------------
@pytest.mark.parametrize("text,field,toggle,expected", [
    ("man not breathing after collapse", "breathing", "slow", "absent"),
    ("he is unconscious and not responding", "consciousness", "alert", "unresponsive"),
    ("bleeding heavily from the leg", "bleeding", "none", "severe"),
    ("gasping for air", "breathing", "normal", "laboured"),
])
def test_alarming_text_overrides_milder_toggle(text, field, toggle, expected):
    from schemas.case import Patient

    p, conflicts = rule_based.reconcile_vitals(text, Patient(**{field: toggle}))
    assert getattr(p, field) == expected and conflicts[0]["toggle"] == toggle and conflicts[0]["text"] == expected


@pytest.mark.parametrize("text,field,toggle", [
    ("he is breathing normally, no bleeding", "breathing", "absent"),   # text milder than toggle: toggle kept (never downgrade)
    ("not unresponsive, talking clearly", "consciousness", "alert"),    # negated phrase must not trigger
    ("breathing fine", "breathing", "normal"),
])
def test_reconciliation_never_downgrades_and_respects_negation(text, field, toggle):
    from schemas.case import Patient

    p, conflicts = rule_based.reconcile_vitals(text, Patient(**{field: toggle}))
    assert getattr(p, field) == toggle and not conflicts


def test_not_breathing_text_with_slow_toggle_now_fires_the_hard_red_flag(monkeypatch):
    monkeypatch.setenv("CEDCS_MODE", "offline")
    cands = _cands(8, 15)
    monkeypatch.setattr(rule_based, "discover_facilities", lambda c, **k: (cands, 8))
    monkeypatch.setattr(rule_based, "fetch_snapshots", lambda cs: [_snap(c.hospital_id) for c in cs])
    o = CedcsOrchestrator()
    res = o.run_case(dict(RAW, emergency_report="man not breathing after collapse", breathing="slow", consciousness="confused"))
    assert "absent_breathing" in o.trace["red_flags"]["rules"] and o.trace["red_flags"]["final_priority"] == "CRITICAL"
    assert res.recommendation.escalated and res.recommendation.escalation_reason.startswith("RED_FLAG_ESCALATION")
    assert any(c.caveat_type == "INPUT_CONFLICT" for c in res.recommendation.caveats)
    assert o.trace["input_conflicts"][0]["text"] == "absent"


# ---------------- adaptive widening ----------------
def test_search_widens_when_too_few_eligible_and_keeps_candidates_when_it_finds_nothing_new(monkeypatch):
    monkeypatch.setenv("CEDCS_MODE", "offline")
    near, far = _cands(8, 15)[:1], _cands(8, 15, 20, 30, 40)
    calls = []

    def disc(case, start_radius=8):
        calls.append(start_radius)
        return (far, 25) if start_radius == 25 else (near, 8)

    monkeypatch.setattr(rule_based, "discover_facilities", disc)
    monkeypatch.setattr(rule_based, "fetch_snapshots", lambda cs: [_snap(c.hospital_id) for c in cs])
    o = CedcsOrchestrator()
    res = o.run_case(dict(RAW))
    assert calls == [8, 25] and o.trace["discovery"]["widened_from"] == 8 and o.trace["discovery"]["radius_km"] == 25
    assert len(res.recommendation.alternatives) >= 1  # widening produced alternatives
    assert any("Widen" in d["decision"] for d in o.plan)

    calls.clear()
    monkeypatch.setattr(rule_based, "discover_facilities", lambda c, start_radius=8: (near, 8))
    o2 = CedcsOrchestrator()
    o2.run_case(dict(RAW))
    assert o2.trace["discovery"]["radius_km"] == 8  # nothing new found: original result kept


def test_no_widening_when_enough_eligible(monkeypatch):
    monkeypatch.setenv("CEDCS_MODE", "offline")
    cands = _cands(8, 12, 15, 20)
    seen = []
    monkeypatch.setattr(rule_based, "discover_facilities", lambda c, start_radius=8: (seen.append(start_radius), (cands, 8))[1])
    monkeypatch.setattr(rule_based, "fetch_snapshots", lambda cs: [_snap(c.hospital_id) for c in cs])
    CedcsOrchestrator().run_case(dict(RAW))
    assert seen == [8]


def test_advisor_preferring_the_current_leader_adds_no_score(monkeypatch, groq):
    case, tri, cands, snaps = _core_setup(monkeypatch, [10, 10])
    from deterministic_core import freshness
    recs = {c.hospital_id: freshness.score_records(s.records) for c, s in zip(cands, snaps)}
    leader = ranking.rank_candidates(cands, recs, tri)[0].hospital_id
    groq["route"] = {"Ranking Advisor": FakeResp(json.dumps(
        {"preferred_hospital_id": leader, "adjustments": {}, "reasoning": "Keep the leader.", "confidence": 0.9}))}
    o = CedcsOrchestrator()
    rec, _ = _run_core(o, case, tri, cands, snaps)
    assert o.trace["advisor"]["adjustments"] == {} and not o.trace["advisor"]["changed_top"]
    assert rec.primary.hospital_id == leader
    assert not any(s["stage"] == "RE_RANK" for s in o.audit.spans)  # nothing to re-rank
