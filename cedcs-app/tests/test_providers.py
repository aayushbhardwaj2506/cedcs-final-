"""Two providers (Groq + NVIDIA): routing, cross-provider failover, circuit breaker, retired models, request shaping,
warm-up, and the guarantee that no key ever leaks. HTTP is faked for both."""

from __future__ import annotations

import json

import pytest
import requests

from fallback import rule_based
from schemas.triage import TriageResult
from llm import groq_agents, providers
from orchestrator.orchestrator import CedcsOrchestrator
from tests.test_groq_agents import RAW, TRIAGE_OK, FakeResp

G120, G20, NV, GEM = groq_agents.GQ_120B, groq_agents.GQ_20B, groq_agents.NV_SUPER, groq_agents.NV_GEMMA
OK = lambda body=TRIAGE_OK: FakeResp(json.dumps(body))
INTAKE_OK = {"patient": {"age": 60, "symptoms": ["chest pain"]}, "intake_completeness": 0.6}


@pytest.fixture
def both(monkeypatch):
    """Both providers configured; each session returns scripted responses (a callable may raise). Calls are recorded."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-secret")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-secret")
    monkeypatch.setattr(groq_agents.time, "sleep", lambda s: None)
    rig = {"calls": [], "script": {"groq": [], "nvidia": []}}

    def make(prov):
        def post(url, headers=None, json=None, timeout=None):
            rig["calls"].append({"provider": prov, "model": json["model"], "body": json, "headers": headers, "url": url, "timeout": timeout})
            item = rig["script"][prov].pop(0) if rig["script"][prov] else OK()
            if callable(item):
                return item()
            return item
        return post

    monkeypatch.setattr(providers.SESSIONS["groq"], "post", make("groq"))
    monkeypatch.setattr(providers.SESSIONS["nvidia"], "post", make("nvidia"))
    rig["models"] = lambda: [(c["provider"], c["model"].split("/")[-1]) for c in rig["calls"]]
    return rig


def raise_timeout():
    raise requests.Timeout("read timed out")


# ---------------------------------------------------------------- plans and routing
def test_default_plans_split_the_work_across_both_providers():
    provs = {st: providers.split(plan[0])[0] for st, plan in groq_agents.STAGE_PLAN.items()}
    assert provs == {"INTAKE": "nvidia", "CLARIFICATION": "nvidia", "TRIAGE": "groq", "CRITIC": "nvidia", "ADVISOR": "groq", "EXPLANATION": "groq"}
    for plan in groq_agents.STAGE_PLAN.values():  # every plan can survive the loss of either provider
        assert {providers.split(x)[0] for x in plan} == {"groq", "nvidia"}
    assert not any("qwen" in x for plan in groq_agents.STAGE_PLAN.values() for x in plan)  # the model that kept overloading


def test_each_stage_goes_to_its_primary_provider(both):
    both["script"]["nvidia"] += [OK(INTAKE_OK)]
    case, info = groq_agents.intake(dict(RAW))
    assert both["models"]() == [("nvidia", "nemotron-3-super-120b-a12b")] and info.provider == "nvidia" and case.patient.age == 60
    t, tinfo = groq_agents.triage(case)
    assert both["models"]()[-1] == ("groq", "gpt-oss-120b") and tinfo.provider == "groq" and t.priority == "HIGH"


def test_a_missing_key_removes_that_provider_from_every_plan(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    assert all(providers.split(x)[0] == "groq" for st in groq_agents.STAGE_PLAN for x in groq_agents._plan(st))
    monkeypatch.setenv("NVIDIA_API_KEY", "n")
    monkeypatch.setenv("CEDCS_LLM_PROVIDERS", "nvidia")  # explicit restriction
    assert all(providers.split(x)[0] == "nvidia" for st in groq_agents.STAGE_PLAN for x in groq_agents._plan(st))
    monkeypatch.setenv("CEDCS_LLM_PROVIDERS", "")
    assert set(providers.configured()) == {"groq", "nvidia"}


def test_no_provider_configured_raises_a_clear_error(monkeypatch):
    for k in ("GROQ_API_KEY", "NVIDIA_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert not groq_agents.enabled()
    with pytest.raises(groq_agents.LLMError, match="no LLM provider"):
        groq_agents.triage(rule_based.parse_intake(RAW))


# ---------------------------------------------------------------- failover across providers
def test_groq_rate_limit_fails_over_to_nvidia_not_straight_to_rules(both):
    both["script"]["groq"] += [FakeResp(status=429, headers={"retry-after": "40"})]
    t, info = groq_agents.triage(rule_based.parse_intake(RAW))
    assert both["models"]() == [("groq", "gpt-oss-120b"), ("nvidia", "nemotron-3-super-120b-a12b")]
    assert info.provider == "nvidia" and t.priority == "HIGH" and info.extra["failover"] == ["gpt-oss-120b rate limited"]


def test_nvidia_timeout_fails_over_to_groq(both):
    both["script"]["nvidia"] += [raise_timeout]
    case, info = groq_agents.intake(dict(RAW))
    assert both["models"]()[:2] == [("nvidia", "nemotron-3-super-120b-a12b"), ("groq", "gpt-oss-20b")] and info.provider == "groq"
    assert "nvidia:nemotron-3-super-120b-a12b Timeout" in info.extra["failover"]


def test_nvidia_is_given_a_short_timeout_so_a_cold_start_cannot_stall_the_case(both):
    groq_agents.intake(dict(RAW))
    nv = next(c for c in both["calls"] if c["provider"] == "nvidia")
    assert nv["timeout"][1] <= 20 and both["calls"][0]["provider"] == "nvidia"
    groq_agents.triage(rule_based.parse_intake(RAW))
    assert next(c for c in both["calls"] if c["provider"] == "groq")["timeout"][1] >= 30


def test_an_empty_answer_counts_as_a_failure_and_moves_on(both):
    both["script"]["nvidia"] += [FakeResp(None)]  # a reasoning model that spent its whole budget thinking: content is null
    case, info = groq_agents.intake(dict(RAW))
    assert info.provider == "groq" and "nvidia:nemotron-3-super-120b-a12b empty answer" in info.extra["failover"]
    assert providers.HEALTH.fails["nvidia"] == 1


def test_everything_failing_raises_llmerror_with_the_status_code(both):
    both["script"]["groq"] += [FakeResp(status=500)] * 4
    both["script"]["nvidia"] += [FakeResp(status=503)] * 6
    with pytest.raises(groq_agents.LLMError, match="HTTP 50"):
        groq_agents.triage(rule_based.parse_intake(RAW))


# ---------------------------------------------------------------- health: circuit breaker, cooldown, retired models
def test_circuit_opens_after_repeated_failures_and_closes_on_recovery(both, monkeypatch):
    both["script"]["nvidia"] += [FakeResp(status=500)] * providers.FAIL_THRESHOLD
    for _ in range(providers.FAIL_THRESHOLD):
        groq_agents.intake(dict(RAW))
    assert providers.HEALTH.state("nvidia") == "circuit open" and not providers.HEALTH.available("nvidia")
    both["calls"].clear()
    groq_agents.intake(dict(RAW))  # while open, NVIDIA is not even tried
    assert all(c["provider"] == "groq" for c in both["calls"])
    real = providers.time.monotonic
    monkeypatch.setattr(providers.time, "monotonic", lambda: real() + providers.CIRCUIT_OPEN_S + 1)  # cooled off: half-open probe
    assert providers.HEALTH.available("nvidia")
    both["calls"].clear()
    groq_agents.intake(dict(RAW))
    assert both["calls"][0]["provider"] == "nvidia" and providers.HEALTH.state("nvidia") == "healthy"


def test_nvidia_rate_limit_puts_the_whole_provider_on_cooldown(both):
    both["script"]["nvidia"] += [FakeResp(status=429, headers={"retry-after": "30"})]
    groq_agents.intake(dict(RAW))
    assert "cooling down" in providers.HEALTH.state("nvidia") and providers.HEALTH.status()["nvidia"]["retry_in_s"] > 0
    both["calls"].clear()
    groq_agents.critic(rule_based.parse_intake(RAW), "x", TriageResult.model_validate(dict(TRIAGE_OK, category_set=["CARDIAC"])), "HIGH")
    assert both["calls"][0]["provider"] == "groq"  # the critic normally starts on NVIDIA; now it routes around the cooldown


def test_a_groq_rate_limit_does_not_cool_down_the_whole_provider(both):
    both["script"]["groq"] += [FakeResp(status=429, headers={"retry-after": "40"})]
    groq_agents.triage(rule_based.parse_intake(RAW))
    assert providers.HEALTH.available("groq") and providers.HEALTH.state("groq") == "healthy"  # limits are per model, not per provider
    assert providers.BUDGET.headroom("openai/gpt-oss-120b") == 0 and providers.BUDGET.headroom("openai/gpt-oss-20b") > 0


def test_a_model_the_account_cannot_use_is_retired_after_one_404(both):
    both["script"]["nvidia"] += [FakeResp(status=404)]
    groq_agents.intake(dict(RAW))
    assert providers.HEALTH.status()["nvidia"]["retired_models"] == ["nvidia/nemotron-3-super-120b-a12b"]
    both["calls"].clear()
    groq_agents.intake(dict(RAW))
    assert not any(c["provider"] == "nvidia" and "nemotron" in c["model"] for c in both["calls"])  # not retried for 10 minutes


def test_soft_request_rate_cap_protects_the_free_tier(both):
    for _ in range(providers.PROVIDERS["nvidia"]["rpm"]):
        providers.HEALTH.note_call("nvidia")
    assert not providers.HEALTH.available("nvidia")
    both["calls"].clear()
    groq_agents.intake(dict(RAW))
    assert both["calls"][0]["provider"] == "groq"


# ---------------------------------------------------------------- request shaping
def test_nvidia_nemotron_is_called_with_thinking_off_and_json_mode():
    b = providers.build_body("nvidia", "nvidia/nemotron-3-super-120b-a12b", "sys", "usr", True, 900)
    assert b["chat_template_kwargs"] == {"enable_thinking": False} and b["response_format"] == {"type": "json_object"}
    assert b["max_tokens"] == 900 and "max_completion_tokens" not in b and b["messages"][0]["role"] == "system"


def test_groq_body_keeps_its_own_parameters():
    b = providers.build_body("groq", "openai/gpt-oss-120b", "s", "u", True, 900)
    assert b["max_completion_tokens"] == 900 and b["reasoning_effort"] == "low" and "chat_template_kwargs" not in b
    assert "response_format" not in providers.build_body("groq", "openai/gpt-oss-20b", "s", "u", False, 50)


def test_a_provider_that_rejects_json_mode_is_retried_once_without_it_and_remembered(both):
    both["script"]["nvidia"] += [FakeResp(status=400)]
    groq_agents.intake(dict(RAW))
    nv = [c for c in both["calls"] if c["provider"] == "nvidia"]
    assert len(nv) == 2 and "response_format" in nv[0]["body"] and "response_format" not in nv[1]["body"]
    assert "nvidia/nemotron-3-super-120b-a12b" in providers.NO_JSON_MODE


def test_keys_go_only_in_the_authorization_header_and_never_into_results(both):
    both["script"]["nvidia"] += [OK(INTAKE_OK)]
    case, info = groq_agents.intake(dict(RAW))
    nv = next(c for c in both["calls"] if c["provider"] == "nvidia")
    assert nv["headers"]["Authorization"] == "Bearer nvapi-test-secret" and "integrate.api.nvidia.com" in nv["url"]
    blob = json.dumps({"info": info.as_dict(), "status": groq_agents.status(), "case": case.model_dump(mode="json")})
    assert "nvapi-test-secret" not in blob and "gsk-test-secret" not in blob


# ---------------------------------------------------------------- warm-up and status
def test_prime_warms_every_nvidia_model_in_use_and_reads_groq_budgets(both):
    hdr = {"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "5000", "x-ratelimit-reset-tokens": "20s"}
    both["script"]["groq"] += [FakeResp("x", headers=hdr), FakeResp("x", headers=hdr)]
    snap = groq_agents.prime_budget()
    assert snap["openai/gpt-oss-120b"] == 5000
    warmed = {c["model"] for c in both["calls"] if c["provider"] == "nvidia"}
    assert warmed == {"nvidia/nemotron-3-super-120b-a12b"}  # gemma is only a late fallback, so it is not kept warm
    assert all(c["body"]["messages"][0]["content"] for c in both["calls"])  # a non-empty system message: an empty one can be rejected
    assert both["calls"][-1]["timeout"][1] >= 30  # warm-up may wait out a cold start
    assert providers.HEALTH.status()["nvidia"]["avg_ms"] is not None


def test_prime_retires_models_that_are_404_for_the_account(both):
    both["script"]["nvidia"] += [FakeResp(status=404)]  # the first warmed model
    groq_agents.prime_budget()
    assert providers.HEALTH.status()["nvidia"]["retired_models"]


def test_status_lists_health_budgets_and_every_stage_plan(both):
    s = groq_agents.status()
    assert set(s["providers"]) == {"groq", "nvidia"} and s["providers"]["nvidia"]["state"] == "healthy"
    assert set(s["plans"]) == set(groq_agents.STAGE_PLAN) and s["plans"]["TRIAGE"][0] == {"spec": G120, "usable": True}
    assert set(s["groq_budget"]) == set(providers.GROQ_MODELS)


# ---------------------------------------------------------------- end to end: the orchestrator sees which provider answered
def test_llm_trace_records_provider_and_failover_per_agent(monkeypatch, both):
    monkeypatch.setenv("CEDCS_MODE", "groq")
    monkeypatch.setattr(rule_based, "discover_facilities", lambda c, **k: ([], 8))
    monkeypatch.setattr(rule_based, "fetch_snapshots", lambda c: [])
    both["script"]["nvidia"] += [raise_timeout]  # intake: NVIDIA times out -> Groq answers
    o = CedcsOrchestrator()
    o._run_front_fallback(dict(RAW), "T")
    by = {c["stage"]: c for c in o.trace["llm"]}
    assert by["INTAKE"]["provider"] == "groq" and by["INTAKE"]["failover"]
    assert by["TRIAGE"]["provider"] == "groq" and by["CLARIFICATION"]["provider"] in ("nvidia", "groq")


def test_keep_warm_pings_nvidia_periodically_and_stops_on_request(both, monkeypatch):
    import threading

    monkeypatch.setattr(groq_agents, "KEEP_WARM_EVERY_S", 0.01)
    stop = threading.Event()
    th = threading.Thread(target=groq_agents.keep_warm_forever, args=(stop,), daemon=True)
    th.start()
    for _ in range(200):
        if len([c for c in both["calls"] if c["provider"] == "nvidia"]) >= 2:
            break
        threading.Event().wait(0.01)
    stop.set()
    th.join(timeout=2)
    assert len([c for c in both["calls"] if c["provider"] == "nvidia"]) >= 2 and not th.is_alive()
    assert not any(c["provider"] == "groq" for c in both["calls"])  # only NVIDIA needs it


def test_clarification_leads_with_the_reliable_nvidia_model_not_the_cold_starting_one():
    assert groq_agents.STAGE_PLAN["CLARIFICATION"][0] == groq_agents.NV_SUPER
    assert groq_agents.NV_GEMMA == groq_agents.STAGE_PLAN["CLARIFICATION"][-1]


def test_a_transient_nvidia_503_is_retried_once_before_it_counts_as_a_failure(both):
    both["script"]["nvidia"] += [FakeResp(status=503)]  # then the default (success) answers the retry
    case, info = groq_agents.intake(dict(RAW))
    assert info.provider == "nvidia" and [c["provider"] for c in both["calls"]] == ["nvidia", "nvidia"]
    assert providers.HEALTH.fails["nvidia"] == 0 and not info.extra.get("failover")


def test_a_persistent_5xx_counts_as_one_failure_after_the_retry(both):
    both["script"]["nvidia"] += [FakeResp(status=503), FakeResp(status=503)]
    groq_agents.intake(dict(RAW))
    assert providers.HEALTH.fails["nvidia"] == 1


def test_when_every_circuit_is_open_the_plan_is_probed_anyway_instead_of_giving_up(both):
    providers.HEALTH.open_until["nvidia"] = providers.time.monotonic() + 999
    providers.HEALTH.open_until["groq"] = providers.time.monotonic() + 999
    case, info = groq_agents.intake(dict(RAW))
    assert info.extra.get("last_resort_probe") is True and both["calls"]  # answered despite both circuits being open
    assert providers.HEALTH.state("nvidia") == "healthy"  # and a successful probe closes the circuit


def test_a_missing_key_is_never_probed_even_as_a_last_resort(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    assert not providers.HEALTH.available("nvidia", force=True) and not providers.HEALTH.available("groq", force=True)


def test_nvidia_waits_longer_when_groq_cannot_take_the_call(both):
    providers.BUDGET.update("openai/gpt-oss-120b", {"retry-after": "600"}, 429)
    providers.BUDGET.update("openai/gpt-oss-20b", {"retry-after": "600"}, 429)  # Groq's daily cap
    groq_agents.intake(dict(RAW))
    nv = next(c for c in both["calls"] if c["provider"] == "nvidia")
    assert nv["timeout"][1] >= 30  # nobody to fail over to: be patient
    both["calls"].clear()
    providers.BUDGET.reset()
    groq_agents.intake(dict(RAW))
    assert next(c for c in both["calls"] if c["provider"] == "nvidia")["timeout"][1] <= 15  # Groq is available: fail over quickly


def test_status_reports_an_exhausted_groq_quota_and_when_it_returns(both):
    for m in providers.GROQ_MODELS:
        providers.BUDGET.update(m, {"retry-after": "556", "x-ratelimit-remaining-tokens": "8000", "x-ratelimit-reset-tokens": "1ms"}, 429)
    s = groq_agents.status()["providers"]
    assert s["groq"]["state"] == "quota exhausted" and 500 <= s["groq"]["retry_in_s"] <= 556
    assert s["nvidia"]["state"] == "healthy"  # the other provider is untouched
    providers.BUDGET.reset()
    assert groq_agents.status()["providers"]["groq"]["state"] == "healthy"
