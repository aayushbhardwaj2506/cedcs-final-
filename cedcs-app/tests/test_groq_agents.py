"""Groq agent layer with the HTTP call mocked: no key, no network."""

from __future__ import annotations

import json

import pytest

from fallback import rule_based
from llm import groq_agents
from orchestrator import orchestrator as orch_mod
from orchestrator.orchestrator import CedcsOrchestrator, _mode

RAW = {
    "emergency_report": "60 year old male, sudden collapse, confused, chest pain, diabetic",
    "consciousness": "confused", "breathing": "laboured", "bleeding": "none",
    "location_lat": 12.9, "location_lng": 80.1, "location_address": "Tambaram",
}


class FakeResp:
    def __init__(self, content=None, status=200, headers=None):
        self.status_code, self.headers = status, headers or {}
        self._j = {"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 100}}
        self.text = json.dumps(self._j)

    def json(self):
        return self._j


@pytest.fixture
def groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("CEDCS_LLM_MODEL", raising=False)
    calls = {"queue": [], "seen": []}

    def post(url, headers=None, json=None, timeout=None):
        calls["seen"].append(json)
        route = calls.get("route")
        if route is not None:  # agent-keyed responses (stages run in parallel, so call order is not fixed)
            system = json["messages"][0]["content"]
            hit = next((v for k, v in route.items() if k in system), None)
            if hit is not None:
                return hit() if callable(hit) else hit
        item = calls["queue"].pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(groq_agents._session, "post", post)
    monkeypatch.setattr(groq_agents.time, "sleep", lambda s: None)
    return calls


TRIAGE_OK = {
    "priority": "HIGH", "category_set": ["CARDIAC", "MADE_UP_CATEGORY"],
    "required_capabilities": ["EMERGENCY_DEPARTMENT", "ICU", "NOT_A_CAPABILITY"],
    "preferred_capabilities": ["CARDIOLOGY", "ICU"], "triage_confidence": 0.8,
    "rationale": "Imminent need for monitoring within minutes; administer oxygen.",
}


def test_prompts_come_from_the_crewai_yaml():
    system, desc, expected = groq_agents._prompts("emergency_triage_assessor", "triage_assessment_task", {})
    assert "Triage" in system and "closed vocabulary" in desc and "required_capabilities" in expected


def test_per_stage_models_and_overrides(monkeypatch):
    monkeypatch.delenv("CEDCS_LLM_MODEL", raising=False)
    assert len({groq_agents.model_for(s) for s in ("INTAKE", "TRIAGE", "EXPLANATION")}) == 3  # spread over separate buckets
    monkeypatch.setenv("CEDCS_LLM_MODEL_TRIAGE", "groq/openai/gpt-oss-20b")
    assert groq_agents.model_for("TRIAGE") == "openai/gpt-oss-20b"
    monkeypatch.setenv("CEDCS_LLM_MODEL", "openai/gpt-5.4-mini")  # the old CrewAI default is ignored
    from llm import providers

    assert groq_agents.model_for("INTAKE") == providers.split(groq_agents.STAGE_PLAN["INTAKE"][0])[1]


def test_triage_sanitises_capabilities_and_categories(groq):
    groq["queue"].append(FakeResp(json.dumps(TRIAGE_OK)))
    case = rule_based.parse_intake(RAW)
    t, info = groq_agents.triage(case)
    assert t.category_set == ["CARDIAC"]
    assert t.required_capabilities == ["EMERGENCY_DEPARTMENT", "ICU"] and t.preferred_capabilities == ["CARDIOLOGY"]
    assert info.extra["dropped_capabilities"] == ["NOT_A_CAPABILITY"] and info.attempts == 1 and info.tokens == 100


def test_triage_retries_once_with_validation_error(groq):
    bad = dict(TRIAGE_OK, rationale="patient is having a heart attack")
    groq["queue"] += [FakeResp(json.dumps(bad)), FakeResp(json.dumps(TRIAGE_OK))]
    t, info = groq_agents.triage(rule_based.parse_intake(RAW))
    assert info.attempts == 2 and "diagnosis" in info.extra["first_attempt_error"]
    assert "failed validation" in groq["seen"][1]["messages"][1]["content"]  # error fed back to the model


def test_triage_gives_up_after_two_failures(groq):
    bad = json.dumps(dict(TRIAGE_OK, rationale="likely sepsis"))
    groq["queue"] += [FakeResp(bad), FakeResp(bad)]
    with pytest.raises(groq_agents.LLMError):
        groq_agents.triage(rule_based.parse_intake(RAW))


def test_intake_toggles_and_location_are_authoritative(groq):
    llm_case = {"patient": {"age": 60, "consciousness": "alert", "breathing": "normal", "symptoms": ["chest pain"]},
                "location": {"lat": 1.0, "lng": 2.0}, "case_id": "LLM-INVENTED", "intake_completeness": 0.6}
    groq["queue"].append(FakeResp(json.dumps(llm_case)))
    case, _ = groq_agents.intake(RAW)
    assert case.patient.consciousness == "confused" and case.patient.breathing == "laboured"
    assert (case.location.lat, case.location.lng) == (12.9, 80.1) and case.case_id is None


def test_429_short_wait_then_success(groq):
    groq["queue"] += [FakeResp(status=429, headers={"retry-after": "2"}), FakeResp(json.dumps(TRIAGE_OK))]
    t, info = groq_agents.triage(rule_based.parse_intake(RAW))
    assert info.extra["rate_limit_wait_s"] == 2.0 and t.priority == "HIGH"


def test_429_fails_over_to_another_model_instead_of_dropping_to_rules(groq):
    groq["queue"] += [FakeResp(status=429, headers={"retry-after": "40"}), FakeResp(json.dumps(TRIAGE_OK))]
    t, info = groq_agents.triage(rule_based.parse_intake(RAW))
    assert t.priority == "HIGH" and info.model != groq_agents.model_for("TRIAGE")
    assert info.extra["failover"] == ["gpt-oss-120b rate limited"]
    assert [b["model"] for b in groq["seen"]] == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def test_429_on_every_model_raises(groq):
    groq["queue"] += [FakeResp(status=429, headers={"retry-after": "40"})] * 3
    with pytest.raises(groq_agents.LLMError):
        groq_agents.triage(rule_based.parse_intake(RAW))


def test_mode_selection(monkeypatch):
    for k, v in {"CEDCS_MODE": "auto", "GROQ_API_KEY": ""}.items():
        monkeypatch.setenv(k, v)
    assert _mode() == "offline"
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert _mode() == "groq"
    monkeypatch.setenv("CEDCS_MODE", "offline")
    assert _mode() == "offline"


def test_llm_failure_falls_back_to_rules_per_stage(groq, monkeypatch):
    monkeypatch.setenv("CEDCS_MODE", "groq")
    o = CedcsOrchestrator()
    groq["queue"] += [FakeResp(status=500), FakeResp(status=500)]  # intake fails on both Groq models it may use
    case, used = o._llm_stage("INTAKE", lambda: groq_agents.intake(RAW), lambda: rule_based.parse_intake(RAW))
    assert used is False and case.patient.age == 60  # rule-based result
    assert o.llm_calls[0]["status"] == "fallback" and "500" in o.llm_calls[0]["error"]
    assert any(e.event_type == "GUARDRAIL_FAIL" and e.stage == "INTAKE" for e in o.audit.events)


def test_triage_rule_floor_only_raises(groq, monkeypatch):
    """LLM says LOW for an absent-breathing case: the rule engine's CRITICAL floor wins."""
    monkeypatch.setenv("CEDCS_MODE", "groq")
    monkeypatch.setattr(rule_based, "discover_facilities", lambda c, **k: ([], 8))
    monkeypatch.setattr(rule_based, "fetch_snapshots", lambda c: [])
    low = dict(TRIAGE_OK, priority="LOW")
    groq["route"] = {
        "Intake": FakeResp(json.dumps({"patient": {}})),
        "Triage Critic": FakeResp(json.dumps({"concerns": [], "rationale": "sound"})),
        "Clarification": FakeResp(json.dumps({"questions": [], "round_rationale": "none"})),
        "Triage": FakeResp(json.dumps(low)),
    }
    o = CedcsOrchestrator()
    raw = dict(RAW, breathing="absent", consciousness="unresponsive", emergency_report="man not breathing")
    _, triage, _, _ = o._run_front_fallback(raw, "T")
    assert triage.priority == "CRITICAL" and o.trace["triage"]["raised_by_rule_floor"] == "CRITICAL"
