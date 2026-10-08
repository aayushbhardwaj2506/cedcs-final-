"""
CEDCS orchestrator — the plain-Python state machine that sits between the
CrewAI Studio-generated crew (`cedcs_ai_multi_agent_layer`) and the
deterministic decision core (`requirements.py`, `freshness.py`,
`eligibility.py`, `escalation.py`, `ranking.py`, `engine.py`).

WHY THIS EXISTS (do not delete / do not fold into a single crew.kickoff()):
CrewAI Studio exported ONE Crew with all six tasks wired
Process.sequential end to end, including explanation_task. But
explanation_task needs {validated_recommendation} as an input — and that
value does not exist until AFTER eligibility/ranking/escalation have run on
the outputs of facility_discovery_task and resource_interpretation_task.
Those are deterministic Python steps, not agent steps (P1: "LLM interprets,
never decides"). So a single end-to-end crew.kickoff() call is structurally
impossible to use correctly here — engine.py has to run in the middle.

This module therefore runs the crew in TWO passes:
  PASS 1 ("front crew"): intake -> clarification -> triage -> facility
          discovery -> resource interpretation. Five agents, five tasks.
  [deterministic core runs here, in plain Python, outside CrewAI entirely]
  PASS 2 ("explanation crew"): recommendation_narrator only, fed the
          engine's ValidatedRecommendation as {validated_recommendation}.

Every agent output is passed through guardrails.schema_retry before the
orchestrator trusts it as a typed object, and the final recommendation is
independently re-checked by orchestrator.validation before pass 2 runs.

INTEGRATION NOTE — READ BEFORE RUNNING:
This module imports the deterministic core as `deterministic_core.*`
(requirements, freshness, eligibility, escalation, ranking, engine). That
package is listed as [BUILT] in your own Orchestration Matrix, but I have
never been shown its actual code in this conversation, so the four
functions below (`compute_requirements`, `score_freshness`,
`filter_eligible`, `apply_escalation`, `rank_candidates`,
`build_recommendation`) are called with the signatures your matrix
implies. If your real module's function names or signatures differ,
adjust the calls in `_run_deterministic_core` below — that function is the
ONLY place this file talks to your existing code, by design, so the fix is
localised to one function.
"""

from __future__ import annotations

import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

from guardrails.diagnostic_filter import check_fields
from guardrails.frozen_recommendation import check_explanation_integrity
from guardrails.red_flags import apply_red_flags
from guardrails.schema_retry import SchemaRetryExhausted, validate_list_with_retry, validate_with_retry
from orchestrator.audit import AuditLog
from orchestrator.validation import run_all_checks
from schemas.case import ClarificationBatch, StructuredCase
from schemas.hospital import CandidateHospital
from schemas.recommendation import ValidatedRecommendation
from schemas.resource import ResourceSnapshot, ScoredResourceRecord
from schemas.triage import TriageResult

logger = logging.getLogger("cedcs.orchestrator")

MAX_CLARIFICATION_ROUNDS = 2


class OrchestratorHaltError(Exception):
    """Raised when the pipeline cannot safely continue — e.g. schema
    validation exhausted its retry budget with no safe fallback, or the
    final validation gate failed. The caller (API layer / CLI) is expected
    to surface this as "escalate to a human", never as a silent 500."""

    def __init__(self, stage: str, reason: str, audit: AuditLog):
        self.stage = stage
        self.reason = reason
        self.audit = audit
        super().__init__(f"Orchestrator halted at stage '{stage}': {reason}")


@dataclass
class PipelineResult:
    case_id: str
    recommendation: Optional[ValidatedRecommendation]
    explanation_text: Optional[str]
    audit: AuditLog
    halted: bool = False
    halt_reason: Optional[str] = None
    trace: dict = field(default_factory=dict)  # what the system decided at each step, and why


def _import_crew():
    """Import the CrewAI Studio export lazily so this package can be unit
    tested (see tests/) without the crewai dependency or API keys present.
    Expects the exported project's `src/` to be on PYTHONPATH — see
    README / tests/conftest.py for how CI wires this up."""
    try:
        from cedcs_ai_multi_agent_layer.crew import CedcsAiMultiAgentLayerCrew
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "Could not import CedcsAiMultiAgentLayerCrew. Make sure the "
            "exported CrewAI Studio project's 'src/' directory is on "
            "PYTHONPATH (pip install -e it, or add to sys.path)."
        ) from exc
    return CedcsAiMultiAgentLayerCrew


def _mode() -> str:
    """'groq' (LLM agents via Groq), 'crewai' (exported CrewAI crew), or 'offline' (rule-based).
    CEDCS_MODE: offline | groq | crewai | llm (= groq if a key is set, else crewai) | auto (default).
    auto: Groq if GROQ_API_KEY is set, else CrewAI if installed and configured, else offline."""
    from llm import groq_agents

    mode = os.environ.get("CEDCS_MODE", "auto").lower()
    if mode == "offline":
        return "offline"
    if mode == "crewai":
        return "crewai"
    if mode in ("groq", "llm"):
        return "groq" if groq_agents.enabled() else ("crewai" if mode == "llm" else "offline")
    if groq_agents.enabled():
        return "groq"
    import importlib.util

    # find_spec only checks that the package exists; actually importing crewai takes ~20 s and _mode() runs on every case.
    if importlib.util.find_spec("crewai") is None:
        return "offline"
    return "crewai" if (os.environ.get("OPENAI_API_KEY") or os.environ.get("CEDCS_LLM_BASE_URL")) else "offline"


def _use_rule_based_fallback() -> bool:
    """True when the orchestrator runs its own direct path (Groq or rules) instead of the CrewAI crew."""
    return _mode() != "crewai"


_MODE_LABEL = {"groq": "GROQ_LLM", "crewai": "LLM_AGENTS", "offline": "RULE_BASED_FALLBACK"}


class CedcsOrchestrator:
    def __init__(self, audit: Optional[AuditLog] = None):
        self.audit = audit or AuditLog()
        self.trace: dict[str, Any] = {}
        self.llm_calls: list[dict] = []
        self.plan: list[dict] = []  # the orchestrator's own decisions and why (shown in the UI)
        self.ai_flags: list[str] = []  # escalation reasons raised by advisory AI agents
        self.extra_caveats: list = []  # caveats contributed by advisory AI agents
        self._routed = False  # True once the direct path has already applied routed ETAs
        self.bed_active = False  # the Bed Checker runs only when a person (or admin) switched it on

    # ---------------- message bus (drives the live orchestration diagram)
    def _send(self, to: str, label: str, **d: Any) -> dict:
        return self.audit.message("ORCH", to, "command", label, **d)

    def _reply(self, frm: str, label: str, kind: str = "result", **d: Any) -> dict:
        return self.audit.message(frm, "ORCH", kind, label, **d)

    def _decide(self, decision: str, why: str) -> None:
        self.plan.append({"decision": decision, "why": why, "t_ms": round(self.audit._ms(), 1)})
        self._trace("plan", list(self.plan))

    def _trace(self, key: str, data: Any) -> None:
        self.trace[key] = data
        self.audit._emit({"type": "trace", "key": key, "data": data})

    @contextmanager
    def _span(self, stage: str, **detail: Any):
        self.audit.stage_start(stage, **detail)
        try:
            yield
        finally:
            self.audit.stage_end(stage)

    def _apply_routed_etas(self, structured_case: StructuredCase, candidates: list[CandidateHospital]) -> list[CandidateHospital]:
        """Upgrade tier-4 straight-line ETAs to tier-2 routed ETAs (LocationIQ). Never fails the case."""
        from services import locationiq

        loc = structured_case.location
        if not candidates or not locationiq.enabled() or loc.lat is None or loc.lng is None:
            self._trace("routing", {"provider": None, "routed": 0, "fallback": len(candidates),
                                    "reason": "no candidates" if not candidates else "not configured"})
            return candidates
        # One matrix call covers 24 destinations: route the nearest ones, keep tier-4 estimates for the rest.
        nearest = sorted((c for c in candidates if c.lat is not None), key=lambda c: c.distance_km if c.distance_km is not None else 1e9)
        routed_ids = {c.hospital_id for c in nearest[: locationiq.MAX_COORDS - 1]}
        self._send("ROUTER", f"route {len(routed_ids)} hospitals by road")
        with self._span("ETA_ROUTING"):
            mins_by_id = None
            routable = [c for c in candidates if c.hospital_id in routed_ids]
            mins = locationiq.route_minutes((loc.lat, loc.lng), [(c.lat, c.lng) for c in routable])
            if mins is not None:
                mins_by_id = {c.hospital_id: m for c, m in zip(routable, mins)}
        if mins_by_id is None:
            self._reply("ROUTER", "unavailable, keeping straight-line estimates", kind="error")
            self.audit.guardrail_fail("ETA_ROUTING", "routing_service", detail="unavailable; keeping tier-4 estimates")
            self._trace("routing", {"provider": "LocationIQ", "routed": 0, "fallback": len(candidates), "error": "unavailable"})
            return candidates
        out, routed = [], 0
        for c in candidates:
            m = mins_by_id.get(c.hospital_id)
            if m is None:
                out.append(c)
            else:
                routed += 1
                out.append(c.model_copy(update={"eta_min": m, "eta_source_tier": 2, "eta_confidence": 0.80}))
        self._trace("routing", {"provider": "LocationIQ", "routed": routed, "fallback": len(candidates) - routed})
        self._reply("ROUTER", f"{routed} road ETAs (tier 2)")
        return out

    def _trace_intake(self, structured_case: StructuredCase) -> None:
        p = structured_case.patient
        self._trace("intake", {
            "age": p.age, "gender": p.gender, "consciousness": p.consciousness, "breathing": p.breathing,
            "bleeding": p.bleeding, "symptoms": p.symptoms, "history": p.medical_history, "onset": p.onset_time,
            "missing_fields": structured_case.missing_fields, "completeness": structured_case.intake_completeness,
        })

    def _trace_front(self, structured_case: StructuredCase, triage: TriageResult) -> None:
        if "intake" not in self.trace:
            self._trace_intake(structured_case)
        if "triage" not in self.trace:
            self._trace("triage", {
                "priority": triage.priority, "categories": triage.category_set, "required": triage.required_capabilities,
                "preferred": triage.preferred_capabilities, "confidence": triage.triage_confidence, "rationale": triage.rationale,
            })

    def _llm_stage(self, stage: str, llm_fn, rule_fn):
        """Run an LLM stage; on any LLM failure record it and use the rule-based stage instead (P8)."""
        from llm import groq_agents

        try:
            value, info = llm_fn()
            self.llm_calls.append(info.as_dict())
            return value, True
        except groq_agents.LLMError as exc:
            self.llm_calls.append({"stage": stage, "model": groq_agents.model_for(stage), "status": "fallback", "error": str(exc)[:200], "reason": "rate_limited" if "429" in str(exc) else "error"})
            self.audit.guardrail_fail(stage, "llm_fallback", detail=str(exc)[:200])
            return rule_fn(), False
        finally:
            self._trace("llm", list(self.llm_calls))

    # ---------------- agents of the direct (Groq / rule-based) path
    def _agent_intake(self, raw: dict, mode: str, case_id: str) -> StructuredCase:
        from fallback import rule_based
        from llm import groq_agents

        self._send("INTAKE", "structure the report into a case")
        with self._span("INTAKE"):
            used = True
            if mode == "groq":
                case, used = self._llm_stage("INTAKE", lambda: groq_agents.intake(raw), lambda: rule_based.parse_intake(raw))
            else:
                case = rule_based.parse_intake(raw)
            case = case.model_copy(update={"case_id": case_id})
        case = self._reconcile_inputs(case, raw)
        self._trace_intake(case)
        acc = case.location.accuracy_m
        if acc is not None and acc > 300:
            from schemas.recommendation import Caveat

            self.extra_caveats.append(Caveat(caveat_type="LOCATION_ACCURACY",
                                             detail=f"Patient location is only accurate to about {int(acc)} m; travel times may be off."))
        self._reply("INTAKE", f"case structured, {int(case.intake_completeness * 100)}% complete" + ("" if used else " (rules)"),
                    kind="result" if used else "error")
        return case

    def _reconcile_inputs(self, case: StructuredCase, raw: dict) -> StructuredCase:
        """Escalate-only: text more alarming than a toggle overrides the toggle (see rule_based.reconcile_vitals)."""
        from fallback import rule_based
        from schemas.recommendation import Caveat

        patient, conflicts = rule_based.reconcile_vitals(str(raw.get("emergency_report", "")), case.patient)
        if not conflicts:
            return case
        for c in conflicts:
            self.extra_caveats.append(Caveat(
                caveat_type="INPUT_CONFLICT",
                detail=f"Report says '{c['phrase']}' but {c['field']} was set to '{c['toggle']}': treated as '{c['text']}' (worst case).",
            ))
        self._decide("Text overrode a toggle", "; ".join(f"{c['field']}: {c['toggle']} to {c['text']} ('{c['phrase']}')" for c in conflicts))
        self._trace("input_conflicts", conflicts)
        return case.model_copy(update={"patient": patient})

    def _agent_triage(self, case: StructuredCase, mode: str, base: Optional[int] = None) -> tuple:
        from fallback import rule_based
        from llm import groq_agents
        from schemas.triage import PRIORITY_ORDER

        if base is not None:  # only worker threads adopt a depth; the calling thread already has its own stack
            self.audit.adopt_depth(base)
        self._send("TRIAGE", "assess urgency and needed capabilities")
        with self._span("TRIAGE"):
            rules = rule_based.assess_triage(case)
            triage, used = rules, False
            if mode == "groq":
                triage, used = self._llm_stage("TRIAGE", lambda: groq_agents.triage(case), lambda: rules)
            floor = None
            if used and PRIORITY_ORDER[rules.priority] > PRIORITY_ORDER[triage.priority]:
                # escalate-only safety floor: the rule engine can raise the LLM's priority, never lower it
                triage = triage.model_copy(update={"priority": rules.priority})
                floor = rules.priority
            self._trace("triage", {
                "priority": triage.priority, "categories": triage.category_set, "required": triage.required_capabilities,
                "preferred": triage.preferred_capabilities, "confidence": triage.triage_confidence,
                "rationale": triage.rationale, "source": "llm" if used else "rules",
                "rule_priority": rules.priority, "raised_by_rule_floor": floor,
            })
        self._reply("TRIAGE", f"{triage.priority}, {len(triage.required_capabilities)} required capabilities",
                    kind="result" if (used or mode != "groq") else "error")
        return triage, rules

    def _agent_clarify(self, case: StructuredCase, mode: str, base: int) -> None:
        """Non-critical: what to ask the caller next. Never blocks the case."""
        from fallback import rule_based
        from llm import groq_agents

        self.audit.adopt_depth(base)
        self._send("CLARIFY", "what should we ask the caller next?")
        if mode == "groq":
            ok, why = groq_agents.can_afford("CLARIFICATION")
            if not ok:  # advisory: rule-based questions cost no tokens
                self._decide("Clarifier uses rules", why)
                mode = "offline"
        try:
            with self._span("CLARIFICATION"):
                used = True
                if mode == "groq":
                    batch, used = self._llm_stage("CLARIFICATION", lambda: groq_agents.clarify(case), lambda: rule_based.clarify(case))
                else:
                    batch = rule_based.clarify(case)
            self._trace("clarification", {"questions": [q.model_dump() for q in batch.questions],
                                          "rationale": batch.round_rationale, "source": "llm" if (mode == "groq" and used) else "rules"})
            self._reply("CLARIFY", f"{len(batch.questions)} follow-up question(s)")
        except Exception as exc:
            self._reply("CLARIFY", f"failed: {str(exc)[:60]}", kind="error")

    def _agent_critic(self, case: StructuredCase, raw: dict, triage: TriageResult, rules: TriageResult, mode: str, base: int):
        from llm import groq_agents

        self.audit.adopt_depth(base)
        if mode != "groq":
            self._decide("Critic skipped", "no LLM available in this mode")
            self.audit.message("ORCH", "CRITIC", "skip", "skipped: no LLM in this mode")
            return None
        ok, why = groq_agents.can_afford("CRITIC")
        if not ok:
            self._decide("Critic skipped", why)
            self.audit.message("ORCH", "CRITIC", "skip", "skipped: LLM budget low")
            self.llm_calls.append({"stage": "CRITIC", "model": groq_agents.model_for("CRITIC"), "status": "skipped", "reason": "budget"})
            self._trace("llm", list(self.llm_calls))
            return None
        self._send("CRITIC", "challenge the triage: what could be wrong or unsafe?")
        try:
            with self._span("CRITIC"):
                review, info = groq_agents.critic(case, str(raw.get("emergency_report", "")), triage, rules.priority)
            self.llm_calls.append(info.as_dict())
        except groq_agents.LLMError as exc:
            self.llm_calls.append({"stage": "CRITIC", "model": groq_agents.model_for("CRITIC"), "status": "fallback", "error": str(exc)[:200], "reason": "rate_limited" if "429" in str(exc) else "error"})
            self.audit.guardrail_fail("CRITIC", "llm_fallback", detail=str(exc)[:200])
            self._reply("CRITIC", "unavailable, triage kept as is", kind="error")
            return None
        finally:
            self._trace("llm", list(self.llm_calls))
        n = len(review.concerns) + len(review.contradictions)
        self._reply("CRITIC", f"{n} concern(s)" + (f", suggests {review.suggested_priority}" if review.suggested_priority else "")
                    + (", wants human review" if review.needs_human_review else ""))
        return review

    def _apply_critic(self, triage: TriageResult, review) -> TriageResult:
        """Escalate-only: the critic may raise priority, add PREFERRED capabilities and request human review."""
        from schemas.recommendation import Caveat
        from schemas.triage import PRIORITY_ORDER

        if review is None:
            return triage
        upd: dict = {}
        applied: dict = {}
        if review.suggested_priority and PRIORITY_ORDER[review.suggested_priority] > PRIORITY_ORDER[triage.priority]:
            upd["priority"] = review.suggested_priority
            applied["priority_raised_to"] = review.suggested_priority
            self._decide(f"Priority raised {triage.priority} to {review.suggested_priority}", "the critic argued triage was too low (escalate-only)")
        added = [c for c in review.additional_preferred_capabilities
                 if c not in triage.required_capabilities and c not in triage.preferred_capabilities]
        if added:
            upd["preferred_capabilities"] = triage.preferred_capabilities + added
            applied["added_preferred"] = added
        # Alert-fatigue guard: a bare "please review" (usually just missing vitals) is not honoured. It needs a real
        # contradiction between report and fields, or an argument for higher priority.
        if review.needs_human_review and (review.contradictions or "priority_raised_to" in applied):
            self.ai_flags.append("AI_CRITIC_REVIEW: critic requests human review")
            applied["human_review"] = True
        elif review.needs_human_review:
            applied["human_review_ignored"] = "no contradiction or priority argument"
        for text in (review.contradictions + review.concerns)[:2]:
            self.extra_caveats.append(Caveat(caveat_type="AI_CRITIC", detail=text))
        self._trace("critic", {**review.model_dump(), "applied": applied})
        if "triage" in self.trace and upd:
            self._trace("triage", {**self.trace["triage"], **{k: v for k, v in (
                ("priority", upd.get("priority")), ("preferred", upd.get("preferred_capabilities"))) if v},
                "raised_by_critic": upd.get("priority")})
        return triage.model_copy(update=upd) if upd else triage

    def _effective_weights(self, priority: str) -> dict:
        from deterministic_core import ranking
        from deterministic_core.beds import BED_WEIGHT

        w = dict(zip(("CAP", "RES", "CAPY", "SPEC", "ACC", "FRESH"), ranking.WEIGHTS[priority]))
        if self.bed_active:
            w = {k: round(v * (1 - BED_WEIGHT), 4) for k, v in w.items()}
            w["BEDS"] = BED_WEIGHT
        return w

    def _decide_bed_check(self, raw: dict) -> None:
        """Bed Checker is MANUAL: it runs only if an admin enabled it globally, or a user switched it on for this case
        (and the admin allows user toggles). It never turns itself on."""
        from dispatch import store

        st = store.get_settings()
        by = "admin" if st["bed_checker_enabled"] else ("user" if raw.get("bed_check") and st["allow_user_toggle"] else None)
        self.bed_active = by is not None
        info = {"active": self.bed_active, "by": by}
        if raw.get("bed_check") and not st["allow_user_toggle"] and not st["bed_checker_enabled"]:
            info["blocked"] = "the administrator has not allowed users to switch the bed checker on"
        self._trace("bed_check", info)
        if self.bed_active:
            self._decide("Bed Checker on", f"switched on manually by the {by}; bed availability becomes the largest ranking metric")

    def _bed_scan(self, candidates, snapshots):
        """Live scan of nearby hospitals' beds (bypassing caches); fresh bed facts replace the cached ones."""
        from fallback import rule_based

        self._send("BEDCHECK", f"scan live bed availability at {len(candidates)} hospitals")
        try:
            with self._span("BED_CHECK"):
                scan = rule_based.scan_beds([c.hospital_id for c in candidates])
        except Exception as exc:
            self._reply("BEDCHECK", "scan failed, using last known beds", kind="error")
            self.audit.guardrail_fail("BED_CHECK", "bed_scan", detail=str(exc)[:160])
            self._trace("beds", {"error": str(exc)[:160], "hospitals": []})
            return snapshots
        fresh = {s.hospital_id: s.records for s in scan["snapshots"]}
        from deterministic_core.beds import BED_KEYS

        merged = []
        for snap in snapshots:
            keep = [r for r in snap.records if r.resource_key not in BED_KEYS]
            merged.append(snap.model_copy(update={"records": keep + fresh.get(snap.hospital_id, [])}))
        names = {c.hospital_id: c.name for c in candidates}
        self._trace("beds", {"scanned_at": scan["scanned_at"], "took_ms": scan["took_ms"], "hospitals": [
            {"id": s.hospital_id, "name": names.get(s.hospital_id, s.hospital_id),
             "beds": {r.resource_key: {"available": (r.value or {}).get("available"), "total": (r.value or {}).get("total"),
                                       "updated_at": r.updated_at.isoformat(), "source": r.source}
                      for r in s.records if r.resource_key in BED_KEYS}} for s in scan["snapshots"]]})
        self._reply("BEDCHECK", f"{len(fresh)} hospitals scanned")
        return merged

    @staticmethod
    def _found(candidates) -> list:
        """What the search found, for the live map (before eligibility has judged anything)."""
        return [{"id": c.hospital_id, "name": c.name, "lat": c.lat, "lng": c.lng, "distance_km": c.distance_km,
                 "operating_status": c.operating_status, "source": c.source, "status_verified": c.status_verified,
                 "phone": c.phone, "address": c.address, "osm_url": c.osm_url, "hours": c.hours, "website": c.website,
                 "inferred": c.inferred} for c in candidates if c.lat is not None]

    MIN_ELIGIBLE = 3  # fewer eligible hospitals than this triggers a wider search

    def _eligible_count(self, candidates, triage: TriageResult, snapshots) -> int:
        from deterministic_core import eligibility, freshness, requirements

        scored = {s.hospital_id: freshness.score_records(s.records) for s in snapshots}
        return len(eligibility.filter_eligible(candidates, requirements.compute_requirements(triage), scored, frozenset(triage.category_set)).eligible)

    def _data_branch(self, case: StructuredCase, base: int, triage: Optional[TriageResult] = None) -> tuple:
        """Hospital search -> road ETAs -> resource records. Independent of triage, so it runs beside the critic."""
        from fallback import rule_based

        self.audit.adopt_depth(base)
        self._send("REGISTRY", "find hospitals near the patient")
        with self._span("FACILITY_DISCOVERY"):
            candidates, radius = rule_based.discover_facilities(case)
        self._trace("discovery", {
            "radius_km": radius, "count": len(candidates), "patient": {"lat": case.location.lat, "lng": case.location.lng},
            "hospitals": self._found(candidates), "source": "osm" if any(c.source == "osm" for c in candidates) else "registry",
        })
        self._reply("REGISTRY", f"{len(candidates)} hospitals within {radius} km")
        candidates = self._apply_routed_etas(case, candidates)
        self._send("REGISTRY", f"fetch resource records for {len(candidates)} hospitals")
        with self._span("RESOURCE_INTERPRETATION"):
            snapshots = rule_based.fetch_snapshots(candidates)
        self._reply("REGISTRY", f"{sum(len(x.records) for x in snapshots)} records")
        if self.bed_active:
            snapshots = self._bed_scan(candidates, snapshots)

        # Escalation rung 1: too few ELIGIBLE hospitals nearby -> widen the search before ranking. The eligibility
        # test is the same deterministic function the core uses; only the search area changes.
        if triage is not None and radius < 25:
            n = self._eligible_count(candidates, triage, snapshots)
            if n < self.MIN_ELIGIBLE:
                self._decide("Widen the search to 25 km",
                             f"only {n} eligible hospital(s) within {radius} km; alternatives matter, so look further out")
                self._send("REGISTRY", "widen the search to 25 km")
                with self._span("DISCOVERY_WIDEN"):
                    wide, wide_radius = rule_based.discover_facilities(case, start_radius=25)
                if len(wide) > len(candidates):
                    self._reply("REGISTRY", f"{len(wide)} hospitals within {wide_radius} km")
                    candidates = self._apply_routed_etas(case, wide)
                    self._send("REGISTRY", f"fetch resource records for {len(candidates)} hospitals")
                    with self._span("RESOURCE_INTERPRETATION_WIDE"):
                        snapshots = rule_based.fetch_snapshots(candidates)
                    self._reply("REGISTRY", f"{sum(len(x.records) for x in snapshots)} records")
                    if self.bed_active:
                        snapshots = self._bed_scan(candidates, snapshots)
                    self._trace("discovery", {
                        "radius_km": wide_radius, "count": len(candidates), "widened_from": radius, "eligible_before": n,
                        "patient": {"lat": case.location.lat, "lng": case.location.lng}, "hospitals": self._found(candidates),
                    })
                else:
                    self._reply("REGISTRY", "no additional hospitals found", kind="error")
        return candidates, snapshots

    def _run_front_fallback(self, raw_inputs: dict[str, Any], case_id: str):
        mode = _mode()
        self._trace("mode", _MODE_LABEL[mode])
        self.audit.stage_start("FRONT_CREW", mode=_MODE_LABEL[mode])
        try:
            case = self._agent_intake(raw_inputs, mode, case_id)
            base = self.audit.current_depth()
            self._decide("Start clarification in the background while triage runs", "both need only the structured case; clarification must not delay triage")
            with ThreadPoolExecutor(max_workers=4) as ex:
                # Clarification is advisory and slow-ish: it runs in the background and is only joined at the end,
                # so it never delays triage or the hospital search.
                f_clar = ex.submit(self._agent_clarify, case, mode, base)
                triage, rules = self._agent_triage(case, mode)
                self._trace_front(case, triage)
                self._decide("Run critic review and hospital search in parallel",
                             "the critic needs only the triage; finding hospitals does not depend on it")
                f_crit = ex.submit(self._agent_critic, case, raw_inputs, triage, rules, mode, base)
                f_data = ex.submit(self._data_branch, case, base, triage)
                review = f_crit.result()
                candidates, snapshots = f_data.result()
                f_clar.result()
            triage = self._apply_critic(triage, review)
            self._routed = True
        except OrchestratorHaltError:
            raise
        except Exception as exc:  # network / validation problems in the direct path
            self.audit.error("FRONT_CREW", str(exc))
            raise OrchestratorHaltError("FRONT_CREW", f"front stages failed: {exc}", self.audit)
        self.audit.stage_end(
            "FRONT_CREW", intake_completeness=case.intake_completeness, priority=triage.priority,
            candidate_count=len(candidates),
        )
        return case, triage, candidates, snapshots

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run_case(self, raw_inputs: dict[str, Any]) -> PipelineResult:
        case_id = raw_inputs.get("case_id") or str(uuid.uuid4())
        self.audit.case_id = case_id
        self.audit.stage_start("PIPELINE", case_id=case_id)
        self._decide_bed_check(raw_inputs)

        try:
            structured_case, triage, candidates, snapshots = self._run_front_crew(raw_inputs, case_id)

            self._trace_front(structured_case, triage)
            if not self._routed:  # the CrewAI path leaves routing to the orchestrator
                candidates = self._apply_routed_etas(structured_case, candidates)
                self._trace("mode", _MODE_LABEL[_mode()])
            agent_priority = triage.priority
            self._send("RED_FLAGS", "apply hard safety rules")
            with self._span("RED_FLAGS"):
                final_priority, red_flags_applied = apply_red_flags(structured_case, triage.priority)
            self._reply("RED_FLAGS", f"{len(red_flags_applied)} rule(s) fired" + (f", now {final_priority}" if red_flags_applied else ""))
            self._trace("red_flags", {"agent_priority": agent_priority, "final_priority": final_priority, "rules": red_flags_applied})
            if red_flags_applied:
                self.audit.escalation(
                    "RED_FLAGS", "red flag rule(s) fired", rules=red_flags_applied,
                    original_priority=triage.priority, final_priority=final_priority,
                )
            triage = triage.model_copy(update={"priority": final_priority})

            recommendation, gate_context = self._run_deterministic_core(
                case_id=case_id,
                structured_case=structured_case,
                triage=triage,
                candidates=candidates,
                snapshots=snapshots,
                red_flags_applied=red_flags_applied + self.ai_flags,  # AI flags escalate exactly like red flags
            )

            self._run_validation_gate(recommendation, gate_context)

            explanation_text = self._run_explanation(recommendation, raw_inputs)

            self.audit.stage_end("PIPELINE", case_id=case_id, outcome="SUCCESS")
            return PipelineResult(
                case_id=case_id,
                recommendation=recommendation,
                explanation_text=explanation_text,
                audit=self.audit,
                trace=self.trace,
            )

        except OrchestratorHaltError as halt:
            self.audit.error("PIPELINE", str(halt), halt_stage=halt.stage)
            self.audit.close_open("halted")
            return PipelineResult(
                case_id=case_id,
                recommendation=None,
                explanation_text=None,
                audit=self.audit,
                halted=True,
                halt_reason=f"{halt.stage}: {halt.reason}",
                trace=self.trace,
            )

    # ------------------------------------------------------------------
    # PASS 1 — front crew stages
    # ------------------------------------------------------------------
    def _build_front_crew(self, crew_base):
        """Assemble a Crew covering only intake -> clarification -> triage
        -> facility discovery -> resource interpretation, reusing the
        @agent / @task factory methods CrewAI Studio generated on the
        exported CrewBase class rather than the combined @crew method
        (which also wires in explanation_task — see module docstring)."""
        from crewai import Crew, Process

        agents = [
            crew_base.emergency_intake_specialist(),
            crew_base.clarification_specialist(),
            crew_base.emergency_triage_assessor(),
            crew_base.facility_discovery_coordinator(),
            crew_base.hospital_data_normaliser(),
        ]
        tasks = [
            crew_base.structured_intake_task(),
            crew_base.clarification_task(),
            crew_base.triage_assessment_task(),
            crew_base.facility_discovery_task(),
            crew_base.resource_interpretation_task(),
        ]
        return Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=True)

    def _run_front_crew(
        self, raw_inputs: dict[str, Any], case_id: str
    ) -> tuple[StructuredCase, TriageResult, list[CandidateHospital], list[ResourceSnapshot]]:
        """Runs the single front-crew kickoff (intake -> clarification ->
        triage -> facility discovery -> resource interpretation) and
        validates each of the five task outputs independently, in task
        order, against its schema. tasks_output is indexed positionally
        because CrewAI's TaskOutput list is returned in the same order the
        tasks were declared on the Crew — see _build_front_crew, which
        declares them in exactly this order."""
        if _use_rule_based_fallback():
            return self._run_front_fallback(raw_inputs, case_id)
        self.audit.stage_start("FRONT_CREW")
        CrewClass = _import_crew()
        crew_base = CrewClass()
        front_crew = self._build_front_crew(crew_base)

        crew_output = front_crew.kickoff(inputs=raw_inputs)
        outputs = crew_output.tasks_output
        if len(outputs) != 5:
            raise OrchestratorHaltError(
                "FRONT_CREW",
                f"expected 5 task outputs (intake, clarification, triage, "
                f"discovery, resource interpretation), got {len(outputs)}",
                self.audit,
            )
        intake_raw, clarification_raw, triage_raw, discovery_raw, resource_raw = (
            o.raw for o in outputs
        )

        # --- intake -----------------------------------------------------
        self.audit.stage_start("INTAKE")
        try:
            intake_outcome = validate_with_retry(StructuredCase, intake_raw)
        except SchemaRetryExhausted as exc:
            self.audit.guardrail_fail("INTAKE", "schema_retry", errors=exc.errors)
            raise OrchestratorHaltError("INTAKE", "StructuredCase failed schema validation", self.audit)
        self.audit.guardrail_pass("INTAKE", "schema_retry", attempts=intake_outcome.attempts)
        structured_case = intake_outcome.value.model_copy(update={"case_id": case_id})
        self.audit.stage_end("INTAKE", intake_completeness=structured_case.intake_completeness)

        # --- clarification (non-critical: log and continue on failure) --
        self.audit.stage_start("CLARIFICATION")
        try:
            clar_outcome = validate_with_retry(ClarificationBatch, clarification_raw)
            self.audit.guardrail_pass("CLARIFICATION", "schema_retry", attempts=clar_outcome.attempts)
        except SchemaRetryExhausted as exc:
            self.audit.guardrail_fail("CLARIFICATION", "schema_retry", errors=exc.errors)
        self.audit.stage_end("CLARIFICATION")

        # --- triage -------------------------------------------------------
        self.audit.stage_start("TRIAGE")
        try:
            triage_outcome = validate_with_retry(TriageResult, triage_raw)
        except SchemaRetryExhausted as exc:
            self.audit.guardrail_fail("TRIAGE", "schema_retry", errors=exc.errors)
            raise OrchestratorHaltError("TRIAGE", "TriageResult failed schema validation", self.audit)
        self.audit.guardrail_pass("TRIAGE", "schema_retry", attempts=triage_outcome.attempts)

        # Independent second pass of the diagnostic-assertion check (the
        # first pass already runs inside TriageResult's own field
        # validator — see schemas/triage.py's docstring on why both exist).
        filter_results = check_fields(triage_outcome.value.model_dump(), ["rationale"])
        for field_name, result in filter_results.items():
            if not result.passed:
                self.audit.guardrail_fail("TRIAGE", "diagnostic_filter", flagged=result.flagged_terms)
                raise OrchestratorHaltError(
                    "TRIAGE", f"diagnostic filter flagged terms in {field_name}: {result.flagged_terms}", self.audit
                )
        triage = triage_outcome.value
        self.audit.stage_end("TRIAGE", priority=triage.priority)

        # --- facility discovery -------------------------------------------
        self.audit.stage_start("FACILITY_DISCOVERY")
        try:
            candidates = validate_list_with_retry(CandidateHospital, discovery_raw)
        except (SchemaRetryExhausted, ValueError) as exc:
            self.audit.guardrail_fail("FACILITY_DISCOVERY", "schema_retry", errors=[str(exc)])
            raise OrchestratorHaltError("FACILITY_DISCOVERY", "CandidateHospital list failed schema validation", self.audit)
        self.audit.guardrail_pass("FACILITY_DISCOVERY", "schema_retry", count=len(candidates))
        self.audit.stage_end("FACILITY_DISCOVERY", candidate_count=len(candidates))

        # --- resource interpretation ---------------------------------------
        self.audit.stage_start("RESOURCE_INTERPRETATION")
        try:
            snapshots = validate_list_with_retry(ResourceSnapshot, resource_raw)
        except (SchemaRetryExhausted, ValueError) as exc:
            self.audit.guardrail_fail("RESOURCE_INTERPRETATION", "schema_retry", errors=[str(exc)])
            raise OrchestratorHaltError(
                "RESOURCE_INTERPRETATION", "ResourceSnapshot list failed schema validation", self.audit
            )
        self.audit.guardrail_pass("RESOURCE_INTERPRETATION", "schema_retry", count=len(snapshots))
        self.audit.stage_end("RESOURCE_INTERPRETATION", snapshot_count=len(snapshots))

        self.audit.stage_end("FRONT_CREW")
        return structured_case, triage, candidates, snapshots

    # ------------------------------------------------------------------
    # Deterministic core call — the ONLY place this file talks to your
    # existing eligibility/ranking/escalation/engine code.
    # ------------------------------------------------------------------
    def _run_deterministic_core(
        self,
        *,
        case_id: str,
        structured_case: StructuredCase,
        triage: TriageResult,
        candidates: list[CandidateHospital],
        snapshots: list[ResourceSnapshot],
        red_flags_applied: list[str],
    ) -> tuple[ValidatedRecommendation, dict[str, Any]]:
        self._send("CORE", "filter eligible hospitals, score, rank, verify")
        self.audit.stage_start("DETERMINISTIC_CORE")

        # Deferred import: keeps this whole package importable/testable
        # even if deterministic_core isn't on PYTHONPATH yet.
        from deterministic_core import eligibility, engine, escalation, freshness, ranking, requirements

        from deterministic_core.capability_map import evaluate_capability
        from deterministic_core.eligibility import _only_inferred

        with self._span("REQUIREMENTS"):
            required = requirements.compute_requirements(triage)
        with self._span("FRESHNESS"):
            scored_records_by_hospital: dict[str, list[ScoredResourceRecord]] = {
                snap.hospital_id: freshness.score_records(snap.records) for snap in snapshots
            }
        counts = {"FRESH": 0, "RECENT": 0, "STALE": 0, "UNKNOWN": 0}
        for recs in scored_records_by_hospital.values():
            for r in recs:
                counts[r.freshness] += 1
        self._trace("freshness", {"records": sum(counts.values()), **counts})

        with self._span("ELIGIBILITY"):
            elig = eligibility.filter_eligible(candidates, required, scored_records_by_hospital, frozenset(triage.category_set))
        eligible = elig.eligible
        eligible_ids = {c.hospital_id for c in eligible}
        req_caps = sorted(required.required())
        self._trace("eligibility", {
            "required": req_caps,
            "candidates": [
                {
                    "id": c.hospital_id, "name": c.name, "lat": c.lat, "lng": c.lng, "eta_min": c.eta_min,
                    "distance_km": c.distance_km, "operating_status": c.operating_status, "source": c.source,
                    "status_verified": c.status_verified, "phone": c.phone, "address": c.address, "osm_url": c.osm_url,
                    "hours": c.hours, "website": c.website, "inferred": c.inferred,
                    "status": (
                        "REJECTED" if c.hospital_id not in eligible_ids
                        else "PROVISIONAL" if c.hospital_id in elig.provisional_ids else "ELIGIBLE"
                    ),
                    "reason": elig.rejections.get(c.hospital_id),
                    "checks": {
                        cap: evaluate_capability(cap, scored_records_by_hospital.get(c.hospital_id, [])).status
                        for cap in req_caps
                    },
                    # capabilities whose only evidence is a map tag (declared, not confirmed by the hospital)
                    "inferred_checks": [cap for cap in req_caps if evaluate_capability(cap, scored_records_by_hospital.get(c.hospital_id, [])).status == "PRESENT"
                                        and _only_inferred(cap, scored_records_by_hospital.get(c.hospital_id, []))],
                }
                for c in candidates
            ],
        })

        with self._span("ESCALATION"):
            escalated, escalation_reason = escalation.apply_escalation(
                triage=triage, eligible=eligible, red_flags_applied=red_flags_applied
            )

        with self._span("RANKING"):
            ranked = ranking.rank_candidates(
                eligible, scored_records_by_hospital, triage, provisional_ids=elig.provisional_ids, bed_metric=self.bed_active
            )
        ranked = self._advisor_step(structured_case, triage, ranked, eligible, scored_records_by_hospital, elig)
        self._trace("ranking", {
            "priority": triage.priority,
            "weights": self._effective_weights(triage.priority),
            "provisional_penalty": ranking.PROVISIONAL_PENALTY,
            "ranked": [
                {"rank": r.rank, "id": r.hospital_id, "name": r.candidate.name, "score": r.score,
                 "contributions": r.contributions, "components": r.components, "penalty": r.penalty,
                 "provisional": r.provisional, "data_confidence": r.resource_confidence,
                 "ai_adjustment": r.ai_adjustment}
                for r in ranked
            ],
        })

        try:
            with self._span("ENGINE"):
                recommendation = engine.build_recommendation(
                    case_id=case_id,
                    ranked=ranked,
                    rejected=[c for c in candidates if c.hospital_id not in eligible_ids],
                    escalated=escalated,
                    escalation_reason=escalation_reason,
                    rejection_reasons=elig.rejections,
                    intake_completeness=structured_case.intake_completeness,
                    triage_confidence=triage.triage_confidence,
                    extra_caveats=self.extra_caveats,
                )
        except (engine.NoEligibleHospitalError, engine.SafetyInvariantViolation) as exc:
            # No safe recommendation exists: hand off to a human operator (rung 4).
            self.audit.escalation("DETERMINISTIC_CORE", str(exc), escalation_reason=escalation_reason)
            raise OrchestratorHaltError("DETERMINISTIC_CORE", f"operator handoff required: {exc}", self.audit)
        self._reply("CORE", f"{len(eligible)} eligible, top: {recommendation.primary.name}, confidence {recommendation.confidence_level}")
        self._trace("confidence", {
            "level": recommendation.confidence_level, "value": recommendation.confidence,
            "terms": recommendation.confidence_terms, "escalated": escalated, "escalation_reason": escalation_reason,
        })

        self.audit.stage_end(
            "DETERMINISTIC_CORE",
            eligible_count=len(eligible),
            escalated=escalated,
        )

        gate_context = {
            "known_hospital_ids": {c.hospital_id for c in candidates},
            "eligible_hospital_ids": eligible_ids,
            "red_flags_applied": red_flags_applied,
            "scored_records_by_hospital": scored_records_by_hospital,
        }
        return recommendation, gate_context

    ADVISOR_GAP = 0.03  # consult the advisor only when the engine's top two are this close, or the top is shaky

    def _advisor_step(self, case, triage, ranked, eligible, scored_records_by_hospital, elig):
        """Adaptive second opinion. Called only when the choice is genuinely close or the top pick is shaky;
        otherwise skipped (saves an LLM call). The advisor can only nudge scores of already-eligible hospitals
        by a bounded amount; the core re-ranks and re-verifies."""
        from deterministic_core import ranking
        from llm import groq_agents

        if len(ranked) < 2:
            return ranked
        gap = ranked[0].score - ranked[1].score
        top = ranked[:5]
        etas = [r.candidate.eta_min for r in top if r.candidate.eta_min is not None]
        reasons = []
        if gap <= self.ADVISOR_GAP:
            reasons.append(f"near-tie: top two are only {gap:.3f} apart")
        if ranked[0].provisional:
            reasons.append("the top hospital has an unverified required capability")
        if etas and ranked[0].candidate.eta_min is not None and ranked[0].candidate.eta_min > 1.5 * min(etas) + 3:
            reasons.append(f"the top hospital's ETA ({ranked[0].candidate.eta_min:.0f} min) is much slower than the best in the shortlist ({min(etas):.0f} min)")
        if _mode() != "groq":
            self.audit.message("ORCH", "ADVISOR", "skip", "skipped: no LLM in this mode")
            return ranked
        if not reasons:
            self._decide("Advisor skipped", f"clear winner (score gap {gap:.3f} > {self.ADVISOR_GAP}), no extra opinion needed")
            self.audit.message("ORCH", "ADVISOR", "skip", f"skipped: clear winner (gap {gap:.2f})")
            self._trace("advisor", {"called": False, "reason": f"clear winner, gap {gap:.3f}"})
            return ranked
        ok, why = groq_agents.can_afford("ADVISOR")
        if not ok:
            self._decide("Advisor skipped", why)
            self.audit.message("ORCH", "ADVISOR", "skip", "skipped: LLM budget low")
            self.llm_calls.append({"stage": "ADVISOR", "model": groq_agents.model_for("ADVISOR"), "status": "skipped", "reason": "budget"})
            self._trace("llm", list(self.llm_calls))
            self._trace("advisor", {"called": False, "reason": why})
            return ranked
        self._decide("Advisor consulted", "; ".join(reasons))
        payload = [{
            "id": r.hospital_id, "name": r.candidate.name, "rank": r.rank, "score": round(r.score, 3),
            "eta_min": r.candidate.eta_min, "eta_tier": r.candidate.eta_source_tier,
            "data_confidence": round(r.resource_confidence, 2), "provisional": r.provisional,
            "components": {k: round(v, 2) for k, v in r.components.items()},
            "missing_preferred": sorted(set(triage.preferred_capabilities) - set(r.capability_match)),
        } for r in top]
        self._send("ADVISOR", "second opinion on the shortlist", reasons=reasons)
        try:
            with self._span("ADVISOR"):
                opinion, info = groq_agents.advise(case, triage, payload)
            self.llm_calls.append(info.as_dict())
        except groq_agents.LLMError as exc:
            self.llm_calls.append({"stage": "ADVISOR", "model": groq_agents.model_for("ADVISOR"), "status": "fallback", "error": str(exc)[:200], "reason": "rate_limited" if "429" in str(exc) else "error"})
            self.audit.guardrail_fail("ADVISOR", "llm_fallback", detail=str(exc)[:200])
            self._reply("ADVISOR", "unavailable, engine ranking kept", kind="error")
            self._trace("advisor", {"called": True, "reasons": reasons, "error": str(exc)[:160]})
            return ranked
        finally:
            self._trace("llm", list(self.llm_calls))
        adj = dict(opinion.adjustments)
        # A stated preference for a hospital that is NOT already first, with no explicit nudge, becomes a small nudge.
        # A preference for the current leader means "keep it": it must not add score.
        pref = opinion.preferred_hospital_id
        if pref and pref != ranked[0].hospital_id and adj.get(pref, 0.0) <= 0:
            adj[pref] = 0.03
        adj = {k: v for k, v in adj.items() if abs(v) > 1e-9}
        before = ranked[0].hospital_id
        if adj:
            with self._span("RE_RANK"):
                ranked = ranking.rank_candidates(
                    eligible, scored_records_by_hospital, triage,
                    provisional_ids=elig.provisional_ids, ai_adjustments=adj, bed_metric=self.bed_active,
                )
        changed = ranked[0].hospital_id != before
        self._trace("advisor", {
            "called": True, "reasons": reasons, "opinion": opinion.model_dump(), "adjustments": adj,
            "top_before": before, "top_after": ranked[0].hospital_id, "changed_top": changed,
        })
        if adj:
            from schemas.recommendation import Caveat
            self.extra_caveats.append(Caveat(caveat_type="AI_SECOND_OPINION", detail=opinion.reasoning))
        self._reply("ADVISOR", ("changed the top choice: " if changed else "kept the top choice: ") + (opinion.reasoning[:70] or "no comment"))
        return ranked

    # ------------------------------------------------------------------
    # Validation gate
    # ------------------------------------------------------------------
    def _run_validation_gate(self, recommendation: ValidatedRecommendation, gate_context: dict[str, Any]) -> None:
        self._send("VALIDATOR", "run the five safety checks")
        self.audit.stage_start("VALIDATION_GATE")
        report = run_all_checks(recommendation, **gate_context)
        self._trace("validation", [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in report.checks])
        if not report.passed:
            for failure in report.failures():
                self.audit.guardrail_fail("VALIDATION_GATE", failure.name, detail=failure.detail)
            raise OrchestratorHaltError(
                "VALIDATION_GATE",
                f"{len(report.failures())} check(s) failed: "
                + "; ".join(f"{f.name}: {f.detail}" for f in report.failures()),
                self.audit,
            )
        self.audit.stage_end("VALIDATION_GATE", checks_passed=len(report.checks))
        self._reply("VALIDATOR", f"{len(report.checks)}/{len(report.checks)} checks passed")

    # ------------------------------------------------------------------
    # PASS 2 — explanation crew
    # ------------------------------------------------------------------
    def _candidate_names(self) -> tuple:
        """Names of every hospital this case considered (so an advisory note naming one is not mistaken for an invented hospital)."""
        return tuple(c["name"] for c in (self.trace.get("eligibility") or {}).get("candidates", []) if c.get("name"))

    def _run_explanation(self, recommendation: ValidatedRecommendation, raw_inputs: dict[str, Any]) -> str:
        self._send("NARRATOR", "explain the frozen recommendation in plain language")
        self.audit.stage_start("EXPLANATION")
        if _use_rule_based_fallback():
            from fallback import rule_based
            from llm import groq_agents

            text, source = None, "rules"
            if _mode() == "groq":
                try:
                    candidate, info = groq_agents.explain(recommendation)
                    checked = check_fields({"explanation": candidate}, ["explanation"])
                    if any(not r.passed for r in checked.values()):  # named a diagnosis: ask once more, telling it which words to avoid
                        bad = tuple(sorted({t for r in checked.values() for t in r.flagged_terms}))
                        self.audit.guardrail_fail("EXPLANATION", "diagnostic_filter", detail="retrying without: " + ", ".join(bad))
                        candidate, info = groq_agents.explain(recommendation, avoid=bad)
                    problems = [n for n, r in check_fields({"explanation": candidate}, ["explanation"]).items() if not r.passed]
                    integrity = check_explanation_integrity(recommendation, candidate, self._candidate_names())
                    if problems or not integrity.passed:
                        info.extra["rejected_by"] = "diagnostic_filter" if problems else "integrity_check"
                        raise groq_agents.LLMError(f"explanation rejected by {info.extra['rejected_by']}")
                    text, source = candidate, "llm"
                    self.llm_calls.append(info.as_dict())
                except groq_agents.LLMError as exc:
                    self.llm_calls.append({"stage": "EXPLANATION", "model": groq_agents.model_for("EXPLANATION"), "status": "fallback", "error": str(exc)[:200], "reason": "rate_limited" if "429" in str(exc) else "error"})
                    self.audit.guardrail_fail("EXPLANATION", "llm_fallback", detail=str(exc)[:200])
                self._trace("llm", list(self.llm_calls))
            if text is None:
                names = self._candidate_names()
                text = rule_based.explain(recommendation)
                if not check_explanation_integrity(recommendation, text, names).passed:
                    # an AI-written note in the caveats named something unknown: leave the AI notes out rather than stop the case
                    self.audit.guardrail_fail("EXPLANATION", "template_ai_notes_dropped", detail="an AI note failed the integrity check")
                    text = rule_based.explain(recommendation, include_ai_notes=False)
                    if not check_explanation_integrity(recommendation, text, names).passed:
                        raise OrchestratorHaltError("EXPLANATION", "template explanation failed integrity check", self.audit)
            self.audit.stage_end("EXPLANATION", mode=_MODE_LABEL[_mode()], source=source)
            self._reply("NARRATOR", "explanation written" + ("" if source == "llm" else " (template)"),
                        kind="result" if (source == "llm" or _mode() != "groq") else "error")
            return text
        from crewai import Crew, Process

        CrewClass = _import_crew()
        crew_base = CrewClass()

        explanation_crew = Crew(
            agents=[crew_base.recommendation_narrator()],
            tasks=[crew_base.explanation_task()],
            process=Process.sequential,
            verbose=True,
        )

        inputs = dict(raw_inputs)
        inputs["validated_recommendation"] = recommendation.model_dump_json()
        crew_output = explanation_crew.kickoff(inputs=inputs)
        explanation_text = str(crew_output.raw)

        filter_results = check_fields({"explanation": explanation_text}, ["explanation"])
        for field_name, result in filter_results.items():
            if not result.passed:
                self.audit.guardrail_fail("EXPLANATION", "diagnostic_filter", flagged=result.flagged_terms)
                raise OrchestratorHaltError(
                    "EXPLANATION",
                    f"diagnostic filter flagged terms in {field_name}: {result.flagged_terms}",
                    self.audit,
                )

        integrity = check_explanation_integrity(recommendation, explanation_text, self._candidate_names())
        if not integrity.passed:
            self.audit.guardrail_fail(
                "EXPLANATION",
                "frozen_recommendation_integrity",
                missing_hospitals=integrity.missing_hospitals,
                unexpected_hospitals=integrity.unexpected_hospitals,
                missing_rejections=integrity.missing_rejections,
            )
            raise OrchestratorHaltError(
                "EXPLANATION",
                "explanation text is inconsistent with the frozen ValidatedRecommendation",
                self.audit,
            )

        self.audit.stage_end("EXPLANATION")
        return explanation_text
