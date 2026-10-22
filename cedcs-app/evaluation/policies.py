"""Destination policies compared in the evaluation: baselines B1-B4, the full CEDCS decision core, ablations and an oracle.

CEDCS here is the project's REAL deterministic core (freshness scoring, eligibility gate, priority-dependent ranking, confidence,
the 8 km -> 25 km widening and the re-verification in the engine). The language-model stages (intake, triage, critic, advisor,
narrator) are not part of it: experiments E1-E3, E5, E6 hand the core a ground-truth triage so that the decision layer is measured on
its own; E4 feeds it the rule-based triage of a noisy text. B5 (an LLM choosing without a gate) is in experiments.py.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from deterministic_core import engine, ranking
from deterministic_core.eligibility import filter_eligible
from deterministic_core.freshness import score_records
from deterministic_core.requirements import compute_requirements
from evaluation.world import T0, Case, Hospital, distance_km, eta_minutes, true_status
from schemas.hospital import CandidateHospital
from schemas.triage import TriageResult

NEAR_KM, WIDE_KM, MIN_ELIGIBLE = 8.0, 25.0, 3  # the application's adaptive radius and widening rule


@dataclass
class Decision:
    hid: Optional[str]  # None: nothing was recommended (operator handoff)
    level: Optional[str] = None  # HIGH | MODERATE | LOW confidence (CEDCS only)
    rung: str = "none"  # none | widened | handoff
    n_eligible: int = 0
    n_provisional: int = 0
    ms: float = 0.0
    top_score: Optional[float] = None


def triage_for(case: Case, *, drop_required: int = 0, priority: Optional[str] = None, rng=None) -> TriageResult:
    """The ground-truth triage (or a deliberately corrupted one, for the safety stress test)."""
    t = case.template
    req = list(t.required)
    if drop_required and rng is not None:
        droppable = [c for c in req if c != "EMERGENCY_DEPARTMENT"]
        for _ in range(min(drop_required, len(droppable))):
            req.remove(rng.choice([c for c in droppable if c in req]))
    return TriageResult(priority=priority or t.priority, category_set=list(t.categories), required_capabilities=req,
                        preferred_capabilities=list(t.preferred), triage_confidence=1.0, rationale="synthetic ground-truth profile")


def _candidate(h: Hospital, case: Case) -> CandidateHospital:
    d = distance_km(case, h)
    return CandidateHospital(hospital_id=h.hid, name=h.name, operating_status="OPERATIONAL", ipd_accepting=True, lat=h.lat, lng=h.lng,
                             distance_km=round(d, 2), eta_min=eta_minutes(d), eta_source_tier=2, eta_confidence=0.80)  # road-routed ETA, as in the deployed app


_scored_cache: dict = {}


def scored_records(h: Hospital, freshness: bool = True) -> list:
    """What the system holds about a hospital: its reports, scored with their age (freshness on) or as if all were current (off)."""
    key = (id(h), freshness)
    hit = _scored_cache.get(key)
    if hit is not None and hit[0] is h:
        return hit[1]
    recs = list(h.reported.values())
    out = score_records(recs, T0) if freshness else [s for r in recs for s in score_records([r], r.updated_at)]
    _scored_cache[key] = (h, out)
    return out


def clear_cache() -> None:
    _scored_cache.clear()


@contextmanager
def weights(override: Optional[dict]):
    """Temporarily replace the ranking weight vectors (the sensitivity experiment)."""
    if not override:
        yield
        return
    saved = dict(ranking.WEIGHTS)
    ranking.WEIGHTS.update(override)
    try:
        yield
    finally:
        ranking.WEIGHTS.clear()
        ranking.WEIGHTS.update(saved)


# ---------------------------------------------------------------- CEDCS and its ablations
def cedcs(case: Case, hospitals: list, triage: TriageResult, *, freshness: bool = True, gate: bool = True) -> Decision:
    t0 = time.perf_counter()
    profile = compute_requirements(triage)
    elig = cands = scored = None
    radius_used = NEAR_KM
    for radius in (NEAR_KM, WIDE_KM):
        radius_used = radius
        cands = [_candidate(h, case) for h in hospitals if distance_km(case, h) <= radius]
        scored = {c.hospital_id: scored_records(next(h for h in hospitals if h.hid == c.hospital_id), freshness) for c in cands}
        if gate:
            elig = filter_eligible(cands, profile, scored)
            if len(elig.eligible) >= MIN_ELIGIBLE:
                break
        elif len(cands) >= MIN_ELIGIBLE:
            break
    if gate:
        eligible, prov, reasons = elig.eligible, frozenset(elig.provisional_ids), elig.rejections
    else:  # ablation: no hard gate, every candidate is ranked (a soft score only)
        eligible, prov, reasons = cands, frozenset(), {}
    if not eligible:
        return Decision(None, None, "handoff", 0, 0, (time.perf_counter() - t0) * 1000)
    ranked = ranking.rank_candidates(eligible, scored, triage, prov)
    if not gate:
        # No gate of any kind: the top of the soft ranking is the answer. (The engine's final re-verification is itself a gate,
        # so it must not be applied here; it is part of what CEDCS adds.)
        return Decision(ranked[0].hospital_id, None, "widened" if radius_used == WIDE_KM else "none", len(eligible), 0, (time.perf_counter() - t0) * 1000,
                        ranked[0].score)
    eligible_ids = {c.hospital_id for c in eligible}
    try:
        rec = engine.build_recommendation(case_id=case.cid, ranked=ranked, rejected=[c for c in cands if c.hospital_id not in eligible_ids],
                                          escalated=False, escalation_reason=None, rejection_reasons=reasons, intake_completeness=1.0,
                                          triage_confidence=1.0)
    except (engine.NoEligibleHospitalError, engine.SafetyInvariantViolation):
        return Decision(None, None, "handoff", len(eligible), len(prov), (time.perf_counter() - t0) * 1000)
    return Decision(rec.primary.hospital_id, rec.confidence_level, "widened" if radius_used == WIDE_KM else "none", len(eligible), len(prov),
                    (time.perf_counter() - t0) * 1000, rec.primary.overall_score)


# ---------------------------------------------------------------- baselines (they only see the reports, like any system would)
def _reported_active(h: Hospital, key: str, field: str) -> bool:
    r = h.reported.get(key)
    return bool(r is not None and isinstance(r.value, dict) and r.value.get(field))


def _nearest(case: Case, hospitals: list, ok=lambda h: True) -> Optional[Hospital]:
    pool = [h for h in hospitals if distance_km(case, h) <= WIDE_KM and ok(h)]
    return min(pool, key=lambda h: distance_km(case, h), default=None)


def b1_nearest(case: Case, hospitals: list, triage=None) -> Decision:
    h = _nearest(case, hospitals)
    return Decision(h.hid if h else None)


def b2_nearest_with_ed(case: Case, hospitals: list, triage=None) -> Decision:
    h = _nearest(case, hospitals, lambda h: _reported_active(h, "department_EMERGENCY_DEPARTMENT", "active"))
    return Decision(h.hid if h else None)


def b3_distance_plus_specialty(case: Case, hospitals: list, triage=None) -> Decision:
    spec = case.template.specialty

    def ok(h):
        return _reported_active(h, "department_EMERGENCY_DEPARTMENT", "active") and (spec is None or _reported_active(h, f"department_{spec}", "active"))

    h = _nearest(case, hospitals, ok)
    return Decision(h.hid if h else None)


def b4_soft_ranking(case: Case, hospitals: list, triage: TriageResult) -> Decision:
    """The multi-criteria score of CEDCS, but no hard gate and no freshness modelling (every report is taken as current)."""
    return cedcs(case, hospitals, triage, freshness=False, gate=False)


def oracle(case: Case, hospitals: list, triage=None) -> Decision:
    """Nearest hospital that is TRULY appropriate (the evaluator sees the true state). The best any policy could do on distance."""
    h = _nearest(case, hospitals, lambda h: all(true_status(h, c) == "PRESENT" for c in case.template.required))
    return Decision(h.hid if h else None)


POLICIES = {
    "B1 nearest": b1_nearest,
    "B2 nearest with ED": b2_nearest_with_ed,
    "B3 distance + specialty": b3_distance_plus_specialty,
    "B4 soft ranking, no gate, no freshness": b4_soft_ranking,
    "CEDCS": lambda case, hospitals, triage: cedcs(case, hospitals, triage),
}
