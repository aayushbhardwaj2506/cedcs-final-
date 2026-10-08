"""Token-budget tracking: header parsing, headroom-aware model choice, skipping optional agents when tight."""

from __future__ import annotations

import json

import pytest

from fallback import rule_based
from llm import groq_agents
from llm.groq_agents import BUDGET
from orchestrator.orchestrator import CedcsOrchestrator
from tests.test_groq_agents import RAW, TRIAGE_OK, FakeResp, groq  # noqa: F401  (fixture)

M120, M20, QW = "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"  # QW: budget bookkeeping only; no plan uses it now


@pytest.mark.parametrize("text,secs", [("38.28s", 38.28), ("547ms", 0.547), ("1m3.2s", 63.2), ("2m", 120.0), ("", 0.0)])
def test_reset_header_parsing(text, secs):
    assert BUDGET._parse_reset(text) == pytest.approx(secs)


def test_headers_update_headroom_and_window_reset_restores_it(monkeypatch):
    BUDGET.update(M120, {"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "1200", "x-ratelimit-reset-tokens": "30s"}, 200)
    assert BUDGET.headroom(M120) == 1200 and BUDGET.headroom(M20) == 8000  # unseen model: full budget assumed
    BUDGET.update(QW, {"retry-after": "12"}, 429)
    assert BUDGET.headroom(QW) == 0
    import time
    real = time.monotonic
    monkeypatch.setattr(groq_agents.time, "monotonic", lambda: real() + 61)  # window has rolled over
    assert BUDGET.headroom(M120) == 8000 and BUDGET.headroom(QW) == 8000


def test_model_choice_prefers_models_with_headroom_but_keeps_preference_order(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("NVIDIA_API_KEY", "n")
    G120, G20, NV = groq_agents.GQ_120B, groq_agents.GQ_20B, groq_agents.NV_SUPER
    assert groq_agents._ordered_chain("TRIAGE", 2000) == [G120, NV, G20]  # the plan, untouched while everything has room
    BUDGET.update(M120, {"x-ratelimit-remaining-tokens": "300", "x-ratelimit-reset-tokens": "30s"}, 200)
    order = groq_agents._ordered_chain("TRIAGE", 2000)
    assert order == [NV, G20, G120]  # the exhausted model goes last; the other PROVIDER is next in line


def test_call_goes_straight_to_the_model_with_headroom_without_wasting_a_429(groq):
    BUDGET.update(M120, {"x-ratelimit-remaining-tokens": "100", "x-ratelimit-reset-tokens": "40s"}, 200)
    groq["queue"].append(FakeResp(json.dumps(TRIAGE_OK)))
    t, info = groq_agents.triage(rule_based.parse_intake(RAW))
    assert [b["model"] for b in groq["seen"]] == [M20] and info.model == M20  # no failed call to the exhausted model


def test_optional_agents_skipped_when_budget_tight_but_essential_never(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")  # Groq-only pool: 2 buckets
    for m in groq_agents.ALL_MODELS:
        BUDGET.update(m, {"x-ratelimit-remaining-tokens": "1500", "x-ratelimit-reset-tokens": "40s"}, 200)
    ok, why = groq_agents.can_afford("CRITIC")
    assert not ok and "budget" in why
    assert groq_agents.can_afford("TRIAGE")[0] and groq_agents.can_afford("EXPLANATION")[0]
    BUDGET.reset()
    assert groq_agents.can_afford("CRITIC")[0]


def test_a_healthy_second_provider_keeps_optional_agents_running_when_groq_is_tight(monkeypatch):
    """The point of sharing load: with Groq's buckets low, NVIDIA's headroom keeps the optional agents (and the reserve) afloat."""
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("NVIDIA_API_KEY", "n")
    for m in groq_agents.ALL_MODELS:
        BUDGET.update(m, {"x-ratelimit-remaining-tokens": "1500", "x-ratelimit-reset-tokens": "40s"}, 200)
    assert groq_agents.can_afford("CRITIC")[0]  # critic's own plan starts on NVIDIA
    groq_agents.HEALTH.rate_limited("nvidia", 30)  # ...but not once NVIDIA is cooling down too
    assert not groq_agents.can_afford("CRITIC")[0]


def test_orchestrator_skips_critic_and_uses_rule_questions_when_budget_is_tight(monkeypatch, groq):
    monkeypatch.setenv("CEDCS_MODE", "groq")
    for m in groq_agents.ALL_MODELS:
        BUDGET.update(m, {"x-ratelimit-remaining-tokens": "1500", "x-ratelimit-reset-tokens": "40s"}, 200)
    monkeypatch.setattr(rule_based, "discover_facilities", lambda c, **k: ([], 8))
    monkeypatch.setattr(rule_based, "fetch_snapshots", lambda c: [])
    groq["route"] = {"Intake": FakeResp(json.dumps({"patient": {}})), "Triage": FakeResp(json.dumps(TRIAGE_OK))}
    o = CedcsOrchestrator()
    o._run_front_fallback(dict(RAW), "T")
    assert any(m["kind"] == "skip" and m["to"] == "CRITIC" for m in o.audit.messages)
    assert any(c["stage"] == "CRITIC" and c["status"] == "skipped" for c in o.llm_calls)
    assert o.trace["clarification"]["source"] == "rules"  # free rule-based questions instead of an LLM call
    assert not any("Triage Critic" in b["messages"][0]["content"] for b in groq["seen"])  # the critic was never called


def test_rate_limit_fallback_is_labelled_as_such(monkeypatch, groq):
    monkeypatch.setenv("CEDCS_MODE", "groq")
    o = CedcsOrchestrator()
    groq["queue"] += [FakeResp(status=429, headers={"retry-after": "40"})] * 3
    _, used = o._llm_stage("TRIAGE", lambda: groq_agents.triage(rule_based.parse_intake(RAW)), lambda: None)
    assert used is False and o.llm_calls[0]["reason"] == "rate_limited"


def test_explanation_payload_groups_excluded_hospitals_but_keeps_every_name():
    from datetime import datetime, timezone

    from schemas.recommendation import RankedHospital, RejectedHospital, ValidatedRecommendation

    r = lambda i: RejectedHospital(hospital_id=f"H{i}", name=f"Hosp {i}", rejection_reason="facility is DIVERTING", rejection_stage="ELIGIBILITY")
    p = RankedHospital(hospital_id="P", name="Prime", rank=1, eta_min=5, eta_confidence=0.8, resource_confidence=0.7, overall_score=0.6, ranking_reason="ok")
    rec = ValidatedRecommendation(case_id="c", primary=p, rejected=[r(i) for i in range(6)], generated_at=datetime.now(timezone.utc).isoformat(), engine_version="t")
    groups = groq_agents._compact(rec)["excluded_hospitals_by_reason"]
    assert len(groups) == 1 and groups[0]["reason"] == "facility is DIVERTING" and len(groups[0]["hospitals"]) == 6


def test_essential_stage_waits_for_a_short_reset_instead_of_falling_back(groq):
    limited = lambda: FakeResp(status=429, headers={"retry-after": "5"})
    groq["queue"] += [limited(), limited(), limited(), FakeResp(json.dumps(TRIAGE_OK))]  # all 3 models limited, then reset
    t, info = groq_agents.triage(rule_based.parse_intake(RAW))
    assert t.priority == "HIGH" and info.extra["waited_for_budget_s"] == 5.0


def test_optional_stage_never_waits(groq):
    groq["queue"] += [FakeResp(status=429, headers={"retry-after": "5"})] * 3
    case = rule_based.parse_intake(RAW)
    with pytest.raises(groq_agents.LLMError):
        groq_agents.clarify(case)


def test_prime_budget_learns_real_headroom_from_headers(groq, monkeypatch):
    hdr = lambda rem: {"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": str(rem), "x-ratelimit-reset-tokens": "30s"}
    groq["queue"] += [FakeResp("x", headers=hdr(500)), FakeResp("x", headers=hdr(7900))]
    snap = groq_agents.prime_budget()
    assert snap[M120] == 500 and snap[M20] == 7900 and QW not in snap


def test_optional_agent_must_leave_room_for_the_explanation(monkeypatch, groq):
    def setb(m, rem):
        BUDGET.update(m, {"x-ratelimit-remaining-tokens": str(rem), "x-ratelimit-reset-tokens": "40s"}, 200)

    monkeypatch.setattr(groq_agents, "ADVISORY_MIN_TOTAL", 0)  # isolate the reserve rule from the overall-total rule
    # The clarifier could run on the 20b bucket (4000 >= 2500), but spending 1500 there would leave the explanation's best
    # remaining bucket with only 2500 < 2600, so the optional agent is refused.
    setb(M20, 4000); setb(M120, 2000)
    ok, why = groq_agents.can_afford("CLARIFICATION")
    assert not ok and "explanation" in why
    setb(M20, 6000)  # the explanation now has its own comfortable bucket -> allowed
    assert groq_agents.can_afford("CLARIFICATION")[0]


def test_a_daily_limit_429_blocks_the_model_for_as_long_as_retry_after_says():
    """Groq's per-DAY token cap answers 429 with a full per-minute bucket and a 1 ms reset: only retry-after is truthful."""
    hdr = {"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "8000", "x-ratelimit-reset-tokens": "1ms", "retry-after": "556"}
    BUDGET.update(M20, hdr, 429)
    assert BUDGET.headroom(M20) == 0  # not "8000 because the minute bucket is full"
    import time

    real = time.monotonic
    groq_agents.time.monotonic  # same module object
    from llm import providers

    orig = providers.time.monotonic
    providers.time.monotonic = lambda: real() + 300  # 5 minutes later: still inside the 556 s block
    try:
        assert BUDGET.headroom(M20) == 0
        providers.time.monotonic = lambda: real() + 600  # past it
        assert BUDGET.headroom(M20) == 8000
    finally:
        providers.time.monotonic = orig
