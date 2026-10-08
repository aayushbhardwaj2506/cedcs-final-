"""LLM agents on two providers, Groq and NVIDIA NIM (both OpenAI-compatible), with load shared between them.

The prompts are NOT re-written here: role / goal / backstory come from
cedcs_ai_multi_agent_layer/config/agents.yaml and the task description /
expected_output from config/tasks.yaml, so the exported CrewAI project stays the
single source of truth for what each agent is told.

Every agent has a PLAN: an ordered list of "provider:model" choices. The call goes to the first one that is currently healthy
and has headroom; if it is rate limited, times out or fails, the next one is tried, on the other provider if need be. Every
output is validated against the same Pydantic schemas as the CrewAI path, retried once with the validation error, and any
failure raises LLMError so the orchestrator can fall back to the rule-based stage (P8: degrade, don't fail). The LLM never
sees eligibility or ranking logic and never chooses a hospital (P1).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests
import yaml

from llm import providers

from deterministic_core.capability_map import CAPABILITY_RESOURCE_MAP
from guardrails.schema_retry import SchemaRetryExhausted, validate_with_retry
from schemas.case import StructuredCase
from schemas.recommendation import ValidatedRecommendation
from schemas.triage import CAPABILITY_CATEGORIES, TriageResult

CONFIG = Path(__file__).resolve().parent.parent / "cedcs_ai_multi_agent_layer" / "config"

NV_SUPER = "nvidia:nvidia/nemotron-3-super-120b-a12b"  # fast (~1.5-3 s), reliable JSON with thinking off
NV_GEMMA = "nvidia:google/gemma-4-31b-it"
GQ_120B = "groq:openai/gpt-oss-120b"
GQ_20B = "groq:openai/gpt-oss-20b"

# Who does what by default. Qwen (which kept overloading Groq's output-token limit) is out; its light stages (intake,
# clarification) moved to NVIDIA, and the two providers split the remaining work so neither is the single bottleneck:
#   Groq   : triage, advisor (gpt-oss-120b)  +  explanation (gpt-oss-20b)
#   NVIDIA : intake, clarification, critic (nemotron-3-super)
# Each plan continues with the other provider's models, so an outage or rate limit on either side is absorbed by the other.
STAGE_PLAN = {
    "INTAKE": [NV_SUPER, GQ_20B, GQ_120B],
    "CLARIFICATION": [NV_SUPER, GQ_20B, NV_GEMMA],  # (gemma-4 cold-starts too often to lead)
    "TRIAGE": [GQ_120B, NV_SUPER, GQ_20B],
    "CRITIC": [NV_SUPER, GQ_120B, GQ_20B],
    "ADVISOR": [GQ_120B, NV_SUPER, GQ_20B],
    "EXPLANATION": [GQ_20B, NV_SUPER, GQ_120B],
}
DEFAULT_PLAN = [GQ_120B, NV_SUPER, GQ_20B]

ALL_MODELS = providers.GROQ_MODELS  # models whose token budget is tracked
ESSENTIAL_STAGES = {"INTAKE", "TRIAGE", "EXPLANATION"}  # never skipped for budget reasons
ADVISORY_MIN_TOTAL = 7000  # tokens of headroom across the pool needed before an OPTIONAL agent may run
ADVISORY_MIN_MODEL = 2500  # ...and at least one model in its plan must have this much
OPTIONAL_CALL_TOKENS = 1500  # typical cost of one optional agent call
EXPLANATION_RESERVE = 2600  # tokens kept free for the (essential) explanation
MAX_ESSENTIAL_WAIT_S = 8.0  # essential stages may wait this long for a rate-limit window to reset
NVIDIA_TOKEN_HEADROOM = 8000  # NVIDIA is limited by request rate, not tokens; while healthy it counts as a full bucket

BUDGET = providers.BUDGET
Budget = providers.Budget
HEALTH = providers.HEALTH
_session = providers.SESSIONS["groq"]  # (tests patch .post on this object)


class LLMError(Exception):
    """The LLM path failed for this stage; caller should use the rule-based stage."""


@dataclass
class CallInfo:
    stage: str
    model: str
    provider: str = ""
    ms: float = 0.0
    tokens: int = 0
    attempts: int = 0
    calls: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"stage": self.stage, "model": self.model, "provider": self.provider, "ms": round(self.ms, 1), "tokens": self.tokens,
                "attempts": self.attempts, "status": "ok", **self.extra}


def enabled() -> bool:
    """True if at least one LLM provider has a key and is permitted."""
    return bool(providers.configured())


def _plan_all(stage: str) -> list:
    """The stage's plan with any environment override applied (regardless of which providers are currently usable)."""
    base = list(STAGE_PLAN.get(stage, DEFAULT_PLAN))
    forced = os.environ.get(f"CEDCS_LLM_MODEL_{stage}", "").strip()
    if not forced:
        g_ = os.environ.get("CEDCS_LLM_MODEL", "").strip()
        forced = g_ if g_ and not g_.startswith("openai/gpt-5") else ""
    if forced:
        spec = providers.normalise(forced)
        base = [spec] + [x for x in base if x != spec]
    return base


def _plan(stage: str) -> list:
    """The plan restricted to providers that are configured and permitted."""
    return [x for x in _plan_all(stage) if providers.provider_ok(providers.split(x)[0])]


def model_for(stage: str) -> str:
    """Model name the stage prefers (the first entry of its plan)."""
    return providers.split(_plan_all(stage)[0])[1]


def provider_for(stage: str) -> str:
    return providers.split(_plan_all(stage)[0])[0]


def model_name() -> str:
    return model_for("TRIAGE")


def _yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG / name).read_text(encoding="utf-8"))


_AGENTS: dict = {}
_TASKS: dict = {}


def _prompts(agent: str, task: str, variables: dict) -> tuple:
    if not _AGENTS:
        _AGENTS.update(_yaml("agents.yaml"))
        _TASKS.update(_yaml("tasks.yaml"))
    a, t = _AGENTS[agent], _TASKS[task]
    desc = t["description"]
    for k, v in variables.items():  # plain replace: the YAML also contains literal JSON braces
        desc = desc.replace("{" + k + "}", str(v))
    system = f"You are the {a['role']}.\nGOAL: {a['goal']}\nBACKSTORY: {a['backstory']}"
    return system, desc, t["expected_output"]


def _headroom(spec: str) -> int:
    """Token headroom of one plan entry: Groq per-model budget from response headers; NVIDIA a full bucket while healthy."""
    prov, model = providers.split(spec)
    if not HEALTH.available(prov, model):
        return 0
    return BUDGET.headroom(model) if prov == "groq" else NVIDIA_TOKEN_HEADROOM


def _pool_total() -> int:
    total = sum(BUDGET.headroom(m) for m in ALL_MODELS if HEALTH.available("groq", m))
    if HEALTH.available("nvidia"):
        total += NVIDIA_TOKEN_HEADROOM
    return total


def can_afford(stage: str) -> tuple:
    """(ok, reason). Essential stages always run; OPTIONAL agents are skipped when the shared token budget is tight."""
    if stage in ESSENTIAL_STAGES:
        return True, ""
    plan = _plan(stage)
    best = max((_headroom(x) for x in plan), default=0)
    total = _pool_total()
    if total < ADVISORY_MIN_TOTAL or best < ADVISORY_MIN_MODEL:
        return False, f"LLM budget is low ({total} tokens left across providers); saving it for triage and the explanation"
    # Reserve: after this agent spends its share, the explanation (essential, runs last) must still be affordable.
    spend_on = next(x for x in plan if _headroom(x) >= ADVISORY_MIN_MODEL)
    left = {x: _headroom(x) - (OPTIONAL_CALL_TOKENS if x == spend_on else 0) for x in set(plan) | set(_plan("EXPLANATION"))}
    if max((left[x] for x in _plan("EXPLANATION")), default=0) < EXPLANATION_RESERVE:
        return False, "running this optional agent would leave too little budget for the explanation"
    return True, ""


def _ordered_chain(stage: str, need: int) -> list:
    """Plan entries, best first: those with headroom for this call keep their plan order, the rest go last."""
    plan = _plan(stage)
    return sorted(plan, key=lambda x: (_headroom(x) < need, plan.index(x)))


def _short(model: str) -> str:
    return model.split("/")[-1]


PATIENT_TIMEOUT = (4, 30)  # used when the other provider cannot take the call: waiting beats falling back to rules


def _attempt(prov: str, model: str, system: str, user: str, json_mode: bool, max_tokens: int, info: CallInfo, patient: bool = False) -> tuple:
    """One model, one or two HTTP calls. Returns (outcome, response, retry_after): outcome is ok | rate | fail | dead."""
    r = None
    for attempt in (1, 2):
        try:
            r, ms = providers.post(prov, model, system, user, json_mode, max_tokens, timeout=PATIENT_TIMEOUT if (patient and prov == "nvidia") else None)
        except requests.RequestException as exc:
            HEALTH.failure(prov)
            info.extra.setdefault("failover", []).append(f"{prov}:{_short(model)} {type(exc).__name__}")
            return "fail", r, 0.0
        if prov == "groq":
            BUDGET.update(model, r.headers, r.status_code)
        try:
            wait = float(r.headers.get("retry-after") or 0)
        except (TypeError, ValueError):
            wait = 0.0
        if r.status_code == 429:
            if prov == "groq" and attempt == 1 and 0 < wait <= 3:
                info.extra["rate_limit_wait_s"] = wait  # a very short wait is cheaper than switching model
                time.sleep(wait)
                continue
            if prov != "groq":
                HEALTH.rate_limited(prov, wait)
            return "rate", r, wait
        if r.status_code == 404:  # this account cannot use that model: retire it for a while
            HEALTH.model_unavailable(prov, model)
            return "dead", r, 0.0
        if r.status_code == 400 and prov == "nvidia" and json_mode and attempt == 1 and model not in providers.NO_JSON_MODE:
            providers.NO_JSON_MODE.add(model)  # provider rejected response_format: retry once without it
            continue
        if r.status_code in (502, 503, 504) and prov == "nvidia" and attempt == 1:
            time.sleep(0.4)  # NVIDIA's free endpoints throw the odd transient 503: one quick retry before it counts as a failure
            continue
        if r.status_code >= 500:
            HEALTH.failure(prov)
            return "fail", r, 0.0
        if r.status_code != 200:
            return "fail", r, 0.0
        HEALTH.success(prov, ms)
        return "ok", r, 0.0
    return "fail", r, 0.0


def _chat(system: str, user: str, info: CallInfo, json_mode: bool, max_tokens: int = 1500) -> str:
    """Send the call to the best available model across both providers, failing over on rate limits, timeouts, errors and
    empty answers. `info.model` / `info.provider` end up as the ones that actually answered. If nothing looks available (every
    circuit open / cooling down) the plan is probed anyway rather than giving up. If everything is rate limited and the stage is
    essential, wait (up to MAX_ESSENTIAL_WAIT_S) for the soonest reset and go round once more."""
    t0 = time.perf_counter()
    last: tuple = ("groq", None)
    try:
        need = (len(system) + len(user)) // 3 + max_tokens // 2  # rough token estimate for this call
        for round_ in (1, 2):
            waits: list = []
            chain = _ordered_chain(info.stage, need)
            if not chain:
                raise LLMError("no LLM provider is configured (set GROQ_API_KEY and/or NVIDIA_API_KEY)")
            tried = 0
            for force in (False, True):
                if force and tried:  # the normal pass found something to try: no last-resort probing
                    break
                for spec in chain:
                    prov, model = providers.split(spec)
                    if not HEALTH.available(prov, model, force=force):
                        continue
                    tried += 1
                    others = any(providers.split(x)[0] != prov and HEALTH.available(*providers.split(x)) and _headroom(x) >= need for x in chain)
                    outcome, r, wait = _attempt(prov, model, system, user, json_mode, max_tokens, info, patient=not others)
                    last = (prov, r)
                    if outcome == "ok":
                        j = r.json()
                        try:
                            text = j["choices"][0]["message"]["content"] or ""
                        except (KeyError, IndexError, TypeError):
                            text = ""
                        if not text.strip():  # e.g. a reasoning model that spent its whole budget thinking
                            HEALTH.failure(prov)
                            info.extra.setdefault("failover", []).append(f"{prov}:{_short(model)} empty answer")
                            continue
                        info.model, info.provider = model, prov
                        if force:
                            info.extra["last_resort_probe"] = True
                        info.calls += 1
                        info.tokens += int((j.get("usage") or {}).get("total_tokens") or 0)
                        return text
                    if outcome == "rate":
                        waits.append(wait)
                        info.extra.setdefault("failover", []).append(f"{_short(model)} rate limited" if prov == "groq" else f"{prov}:{_short(model)} rate limited")
                    elif outcome == "dead":
                        info.extra.setdefault("failover", []).append(f"{prov}:{_short(model)} unavailable")
                    else:
                        info.extra.setdefault("failover", []).append(f"{prov}:{_short(model)} failed")
            soonest = min([w for w in waits if w > 0], default=0)
            if round_ == 1 and info.stage in ESSENTIAL_STAGES and 0 < soonest <= MAX_ESSENTIAL_WAIT_S:
                info.extra["waited_for_budget_s"] = round(soonest, 1)
                time.sleep(soonest + 0.2)
                continue
            break
    finally:
        info.ms += (time.perf_counter() - t0) * 1000
    prov, r = last
    info.calls += 1
    if r is None:
        raise LLMError("no LLM provider answered (all skipped or unreachable)")
    raise LLMError(f"{prov.capitalize() if prov != 'nvidia' else 'NVIDIA'} HTTP {r.status_code}: {r.text[:200]}")


WARM_TIMEOUT = (5, 60)  # a warm-up may wait out a cold start: nobody is waiting on it


def _nvidia_models_in_use() -> list:
    return list(dict.fromkeys(providers.split(x)[1] for plan in STAGE_PLAN.values() for x in plan
                              if x.startswith("nvidia:") and x != NV_GEMMA))  # gemma is only a late fallback: not worth keeping warm


def _warm_nvidia() -> None:
    for model in _nvidia_models_in_use():
        try:
            r, ms = providers.post("nvidia", model, "You are a health check.", "Reply with the single word: ok", False, 4, timeout=WARM_TIMEOUT)
            if r.status_code == 200:
                HEALTH.success("nvidia", ms)
            elif r.status_code == 404:
                HEALTH.model_unavailable("nvidia", model)
        except requests.RequestException:
            pass  # still cold or unreachable: the real calls will fail over


def prime_budget() -> dict:
    """Learn each provider's real state at startup: Groq's remaining token budget per model (1-token calls), and a warm-up call to
    each NVIDIA model in use (their free endpoints cold-start in 25-40 s, so the first real request should not pay that).
    Cheap and failure-tolerant."""
    snap: dict = {}
    if providers.provider_ok("groq"):
        for m in ALL_MODELS:
            try:
                r, ms = providers.post("groq", m, "You are a health check.", "ok", False, 1)
                BUDGET.update(m, r.headers, r.status_code)
                if r.status_code == 200:
                    HEALTH.success("groq", ms)
            except requests.RequestException:
                pass
        snap = BUDGET.snapshot()
    if providers.provider_ok("nvidia"):
        _warm_nvidia()
    return snap


KEEP_WARM_EVERY_S = 240.0


def keep_warm_forever(stop=None) -> None:
    """Background loop: NVIDIA's endpoints go cold after a few idle minutes; a tiny ping every 4 minutes keeps them fast."""
    stop = stop or threading.Event()
    while not stop.wait(KEEP_WARM_EVERY_S):
        if providers.provider_ok("nvidia"):
            _warm_nvidia()


def status() -> dict:
    """Provider health, Groq budgets and each stage's plan (no secrets), for the UI."""
    return {
        "providers": HEALTH.status(),
        "groq_budget": BUDGET.snapshot(),
        "plans": {st: [{"spec": x, "usable": providers.provider_ok(providers.split(x)[0]) and HEALTH.available(*providers.split(x))}
                       for x in _plan_all(st)] for st in STAGE_PLAN},
    }


def _structured(schema, system: str, user: str, info: CallInfo, sanitize=None):
    """One call + one retry that feeds the validation error back, like the CrewAI guardrail."""

    def post(raw: str) -> str:
        if sanitize is None:
            return raw
        try:
            return json.dumps(sanitize(json.loads(raw)))
        except (json.JSONDecodeError, TypeError, AttributeError):
            return raw

    first = post(_chat(system, user, info, json_mode=True))
    retry = lambda err: post(_chat(system, user + f"\n\nYour previous answer failed validation: {err}\nReturn corrected JSON only.", info, True))
    try:
        outcome = validate_with_retry(schema, first, retry_fn=retry)
    except SchemaRetryExhausted as exc:
        raise LLMError(f"{schema.__name__} failed validation twice: {exc.errors[-1][:200]}") from exc
    info.attempts = outcome.attempts
    if outcome.validation_errors:
        info.extra["first_attempt_error"] = outcome.validation_errors[0].replace("\n", " ")[:160]
    return outcome.value


# ------------------------------------------------------------------ agents

def intake(raw: dict[str, Any]) -> tuple:
    """free text + toggles -> StructuredCase. Toggles and location are authoritative, never LLM-invented."""
    info = CallInfo("INTAKE", model_for("INTAKE"), provider_for("INTAKE"))
    variables = {
        "emergency_report": raw.get("emergency_report", ""), "consciousness": raw.get("consciousness", "unknown"),
        "breathing": raw.get("breathing", "unknown"), "bleeding": raw.get("bleeding", "unknown"),
        "location_lat": raw.get("location_lat"), "location_lng": raw.get("location_lng"),
        "location_address": raw.get("location_address", ""), "location_source": raw.get("location_source", "MANUAL"),
    }
    system, desc, expected = _prompts("emergency_intake_specialist", "structured_intake_task", variables)
    user = f"{desc}\n\nEXPECTED OUTPUT:\n{expected}\n\nReturn ONLY the JSON object."

    def sanitize(d: dict) -> dict:
        p = d.setdefault("patient", {})
        for k in ("consciousness", "breathing", "bleeding"):
            if raw.get(k, "unknown") != "unknown":
                p[k] = raw[k]  # the caller's toggle wins over the model's reading of the text
        d["location"] = {"lat": raw.get("location_lat"), "lng": raw.get("location_lng"),
                         "address": raw.get("location_address"), "source": raw.get("location_source", "MANUAL"),
                         "accuracy_m": raw.get("location_accuracy_m")}
        d["case_id"] = None
        d["attachments"] = []
        return d

    case = _structured(StructuredCase, system, user, info, sanitize)
    return case, info


def triage(case: StructuredCase) -> tuple:
    """StructuredCase -> TriageResult. Capability keys outside the checkable vocabulary are dropped and reported."""
    info = CallInfo("TRIAGE", model_for("TRIAGE"), provider_for("TRIAGE"))
    system, desc, expected = _prompts("emergency_triage_assessor", "triage_assessment_task", {})
    user = (f"{desc}\n\nSTRUCTURED CASE (JSON):\n{case.model_dump_json(exclude={'case_id'})}\n\n"
            f"EXPECTED OUTPUT:\n{expected}\n\nReturn ONLY the JSON object.")
    dropped: list = []

    def sanitize(d: dict) -> dict:
        d["category_set"] = [c for c in d.get("category_set", []) if c in CAPABILITY_CATEGORIES]
        for key in ("required_capabilities", "preferred_capabilities"):
            keep = [c for c in d.get(key, []) if c in CAPABILITY_RESOURCE_MAP]
            dropped.extend(c for c in d.get(key, []) if c not in CAPABILITY_RESOURCE_MAP)
            d[key] = list(dict.fromkeys(keep))
        d["preferred_capabilities"] = [c for c in d["preferred_capabilities"] if c not in d["required_capabilities"]]
        return d

    result = _structured(TriageResult, system, user, info, sanitize)
    if not result.required_capabilities:
        raise LLMError("triage returned no checkable required capabilities")
    info.extra["dropped_capabilities"] = sorted(set(dropped))
    return result, info


def _group_rejections(rec: ValidatedRecommendation) -> list:
    """One entry per distinct reason, listing every hospital name (all must still be mentioned) - far fewer tokens."""
    groups: dict = {}
    for x in rec.rejected:
        groups.setdefault(x.rejection_reason, []).append(x.name)
    return [{"reason": reason, "hospitals": names} for reason, names in groups.items()]


def _compact(rec: ValidatedRecommendation) -> dict:
    """Only what the narrator needs: keeps the prompt small (token limits) without dropping any hospital or reason."""
    h = lambda x: {"rank": x.rank, "name": x.name, "eta_min": x.eta_min, "why_ranked": x.ranking_reason}
    return {
        "primary": {**h(rec.primary), "confirmed_capabilities": rec.primary.capability_match},
        "alternatives": [h(x) for x in rec.alternatives],
        "excluded_hospitals_by_reason": _group_rejections(rec),
        "caveats": [c.detail for c in rec.caveats],
        "confidence": {"level": rec.confidence_level, "value": rec.confidence},
        "escalated": rec.escalated, "escalation_reason": rec.escalation_reason,
    }


def explain(rec: ValidatedRecommendation, avoid: tuple = ()) -> tuple:
    info = CallInfo("EXPLANATION", model_for("EXPLANATION"), provider_for("EXPLANATION"))
    system, desc, expected = _prompts(
        "recommendation_narrator", "explanation_task", {"validated_recommendation": json.dumps(_compact(rec))}
    )
    user = (f"{desc}\n\nEXPECTED OUTPUT:\n{expected}\n\n"
            "Use every hospital name exactly as written in the recommendation. Plain text with the bold headings shown above. "
            "Never name a medical condition or diagnosis: describe only the observed signs and the capabilities needed."
            + (f" Your previous draft was rejected for naming: {', '.join(avoid)}. Do not use those words." if avoid else ""))
    text = _chat(system, user, info, json_mode=False, max_tokens=1800).strip()
    info.attempts = 1
    if not text:
        raise LLMError("empty explanation")
    return text, info


# ------------------------------------------------------------------ advisory agents

def clarify(case: StructuredCase) -> tuple:
    """What to ask the caller next. Prompt comes from the CrewAI YAML (clarification_specialist)."""
    from schemas.case import ClarificationBatch

    info = CallInfo("CLARIFICATION", model_for("CLARIFICATION"), provider_for("CLARIFICATION"))
    system, desc, expected = _prompts("clarification_specialist", "clarification_task", {})
    user = (f"{desc}\n\nSTRUCTURED CASE (JSON):\n{case.model_dump_json(exclude={'case_id'})}\n\n"
            f"EXPECTED OUTPUT:\n{expected}\n\nReturn ONLY the JSON object.")

    def sanitize(d: dict) -> dict:
        d["questions"] = list(d.get("questions", []))[:3]
        return d

    return _structured(ClarificationBatch, system, user, info, sanitize), info


CRITIC_SYSTEM = (
    "You are the Triage Critic: an experienced emergency physician acting as devil's advocate over an automated "
    "triage. You do NOT diagnose and never name a medical condition. You look for ways the triage could be WRONG or "
    "UNSAFE: under-estimated urgency, capabilities the patient will need that were not requested, and contradictions "
    "between the free-text report and the structured fields (for example the text says 'not breathing' but the "
    "breathing field says normal). Be concise, specific and skeptical; if the triage is sound say so and return few "
    "or no concerns. You may only ever ask for MORE caution, never less. Missing pulse, blood pressure and oxygen "
    "readings are NORMAL for a bystander's phone report: do not list them as concerns and do not request human "
    "review because of them alone. Ask for human review only for a specific, unresolved safety risk that a human "
    "operator could actually resolve (for example a real contradiction, or an ambiguous life-threatening presentation)."
)


def _redact_diagnoses(text: str) -> tuple:
    """(text, terms) with any condition name replaced by a neutral phrase. For ADVISORY notes only: the review's value is the
    risk it points at, so one slipped-in condition name should not cost the whole review."""
    from schemas.triage import find_diagnosis_terms

    hits = find_diagnosis_terms(text)
    for term in hits:
        text = re.sub(r"(?i)\b" + re.escape(term) + r"\b", "a serious condition", text)
    return text, hits


def critic(case: StructuredCase, report_text: str, triage: TriageResult, rule_priority: str) -> tuple:
    info = CallInfo("CRITIC", model_for("CRITIC"), provider_for("CRITIC"))
    vocab = ", ".join(sorted(CAPABILITY_RESOURCE_MAP))
    user = (
        f"FREE-TEXT REPORT:\n{report_text}\n\nSTRUCTURED CASE:\n{case.model_dump_json(exclude={'case_id'})}\n\n"
        f"CURRENT TRIAGE:\n{triage.model_dump_json()}\n\nA simple rule engine independently rated this case: {rule_priority}.\n\n"
        "Review the triage critically. Return ONLY this JSON object:\n"
        '{"concerns": [up to 5 short strings], "contradictions": [up to 3 short strings between report and fields], '
        '"suggested_priority": "CRITICAL"|"HIGH"|"MODERATE"|"LOW"|null (only if the current priority is too LOW), '
        '"additional_preferred_capabilities": [keys from the list below that would help and are missing], '
        '"needs_human_review": true|false (true if the case is ambiguous or high-stakes with poor data), '
        '"rationale": "one or two sentences, no diagnosis names"}\n\n'
        "In every text field describe the observed signs and the risk. Never write a condition name or abbreviation "
        "(for example MI, stroke, sepsis, heart attack): such text is rejected.\n"
        f"Capability keys allowed: {vocab}."
    )

    def sanitize(d: dict) -> dict:
        d["additional_preferred_capabilities"] = [
            c for c in d.get("additional_preferred_capabilities", []) if c in CAPABILITY_RESOURCE_MAP
        ]
        removed: list = []
        for key in ("concerns", "contradictions"):
            if isinstance(d.get(key), list):
                cleaned = []
                for item in d[key]:
                    text, hits = _redact_diagnoses(item) if isinstance(item, str) else (item, [])
                    removed += hits
                    cleaned.append(text)
                d[key] = cleaned
        if isinstance(d.get("rationale"), str):
            d["rationale"], hits = _redact_diagnoses(d["rationale"])
            removed += hits
        if removed:
            info.extra["redacted_terms"] = sorted(set(removed))  # shown in the run details: the notes were edited, not rejected
        return d

    from schemas.agents import CriticReview

    return _structured(CriticReview, CRITIC_SYSTEM, user, info, sanitize), info


ADVISOR_SYSTEM = (
    "You are the Ranking Advisor giving a SECOND OPINION on hospitals that a deterministic engine has ALREADY "
    "verified as eligible and ranked. You cannot add, remove or override hospitals. You may nudge scores by at most "
    "+-0.05 to reflect a real trade-off the formula handles poorly: a much longer trip for a marginal capability gain, "
    "unverified (provisional) data on a life-critical requirement, a stale key resource, or a near-tie where one option "
    "is clearly safer. If the top hospital is clearly best, return no adjustments. Reason briefly and concretely from "
    "the numbers given. Never mention a diagnosis."
)


def advise(case: StructuredCase, triage: TriageResult, top: list[dict]) -> tuple:
    from schemas.agents import AdvisorOpinion

    info = CallInfo("ADVISOR", model_for("ADVISOR"), provider_for("ADVISOR"))
    user = (
        f"PATIENT: priority {triage.priority}; required {triage.required_capabilities}; preferred {triage.preferred_capabilities}; "
        f"consciousness {case.patient.consciousness}, breathing {case.patient.breathing}, bleeding {case.patient.bleeding}.\n\n"
        f"TOP ELIGIBLE HOSPITALS (already ranked by the engine; score in [0,1], higher is better):\n{json.dumps(top)}\n\n"
        "Return ONLY this JSON object:\n"
        '{"preferred_hospital_id": "<id from the list or null>", "adjustments": {"<id>": <number between -0.05 and 0.05>}, '
        '"reasoning": "2-3 sentences citing the numbers", "concerns": [up to 3 short strings], "confidence": <0..1>}'
    )
    ids = {h["id"] for h in top}

    def sanitize(d: dict) -> dict:
        d["adjustments"] = {k: max(-0.05, min(0.05, float(v))) for k, v in (d.get("adjustments") or {}).items() if k in ids}
        if d.get("preferred_hospital_id") not in ids:
            d["preferred_hospital_id"] = None
        return d

    return _structured(AdvisorOpinion, ADVISOR_SYSTEM, user, info, sanitize), info
