"""Ranking (design doc 7.2). Runs only on hospitals that already passed eligibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from deterministic_core.capability_map import (
    CAPABILITY_RESOURCE_MAP,
    SPECIALIST_CAPABILITIES,
    BedCheck,
    EquipmentCheck,
    evaluate_capability,
    latest,
    specialist_score,
)
from deterministic_core.beds import BED_WEIGHT, BED_LABELS, bed_detail, bed_score, relevant_bed_keys
from deterministic_core.requirements import compute_requirements
from schemas.hospital import CandidateHospital
from schemas.resource import ScoredResourceRecord
from schemas.triage import TriageResult

# (w_cap, w_res, w_capy, w_spec, w_acc, w_fresh)
WEIGHTS: dict = {
    "CRITICAL": (0.28, 0.22, 0.15, 0.13, 0.12, 0.10),
    "HIGH": (0.24, 0.20, 0.15, 0.12, 0.19, 0.10),
    "MODERATE": (0.18, 0.15, 0.14, 0.10, 0.33, 0.10),
    "LOW": (0.12, 0.10, 0.12, 0.08, 0.48, 0.10),
}

from schemas.agents import MAX_AI_ADJUSTMENT

PROVISIONAL_PENALTY = 0.6  # tunable; not in the design doc verbatim
NEEDED_UNITS = 1  # units of each bed/equipment resource a case needs
CAPY_EMERGENCY_BED_CAP = 10  # this many free emergency beds = full capacity score
CAPY_IPD_DELAY_CAP_MIN = 120  # this admission delay = zero score


@dataclass
class RankedCandidate:
    candidate: CandidateHospital
    rank: int
    score: float
    components: dict
    capability_match: list
    resource_confidence: float
    reason: str
    provisional: bool
    required: frozenset  # carried so engine.py can re-verify independently
    records: list = field(default_factory=list)
    contributions: dict = field(default_factory=dict)  # weighted, post-FRESH terms that sum to score/penalty
    penalty: float = 1.0
    ai_adjustment: float = 0.0  # advisor nudge, clipped to +-MAX_AI_ADJUSTMENT, applied after the penalty

    @property
    def hospital_id(self) -> str:
        return self.candidate.hospital_id


def _mean(xs: list, default: float) -> float:
    return sum(xs) / len(xs) if xs else default


def _res_component(caps: set, records) -> float:
    """Mean of min(available/needed, 1) over required+preferred bed/equipment resources."""
    vals = []
    for cap in caps:
        check = CAPABILITY_RESOURCE_MAP.get(cap)
        if not isinstance(check, (BedCheck, EquipmentCheck)):
            continue
        res = evaluate_capability(cap, records)
        if isinstance(check, BedCheck):
            rec = latest(records, check.key)
            avail = rec.value.get("available") if rec and isinstance(rec.value, dict) else None
            vals.append(min((avail or 0) / NEEDED_UNITS, 1.0))
        else:
            vals.append(1.0 if res.status == "PRESENT" else 0.0)
    return _mean(vals, 1.0)


def _capy_component(records) -> float:
    parts = []
    beds = latest(records, "emergency_beds")
    if beds and isinstance(beds.value, dict) and beds.value.get("available") is not None:
        parts.append(min(beds.value["available"] / CAPY_EMERGENCY_BED_CAP, 1.0))
    delay = latest(records, "ipd_admission_delay_est_min")
    if delay and isinstance(delay.value, dict) and delay.value.get("minutes") is not None:
        parts.append(1.0 - min(delay.value["minutes"] / CAPY_IPD_DELAY_CAP_MIN, 1.0))
    return _mean(parts, 0.0)


def _spec_component(caps: set, records) -> float:
    vals = []
    for cap in caps:
        s = specialist_score(cap, records)
        if s is not None:
            vals.append(s[0])
    return _mean(vals, 1.0)


def rank_candidates(
    eligible: list[CandidateHospital],
    scored_records_by_hospital: dict[str, list[ScoredResourceRecord]],
    triage: TriageResult,
    provisional_ids: frozenset = frozenset(),
    ai_adjustments: dict = None,
    bed_metric: bool = False,
) -> list[RankedCandidate]:
    if not eligible:
        return []
    profile = compute_requirements(triage)
    required, preferred = profile.required(), profile.preferred()
    w_cap, w_res, w_capy, w_spec, w_acc, w_fresh = WEIGHTS[triage.priority]
    w_beds = 0.0
    bed_keys: list = []
    if bed_metric:
        # Bed availability becomes the single largest metric; the priority-dependent weights share what is left.
        w_beds = BED_WEIGHT
        w_cap, w_res, w_capy, w_spec, w_acc, w_fresh = (w * (1 - BED_WEIGHT) for w in (w_cap, w_res, w_capy, w_spec, w_acc, w_fresh))
        bed_keys = relevant_bed_keys(required, preferred)

    etas = [c.eta_min for c in eligible if c.eta_min is not None]
    eta_lo, eta_hi = (min(etas), max(etas)) if etas else (0.0, 0.0)

    scored: list = []
    for cand in eligible:
        records = scored_records_by_hospital.get(cand.hospital_id, [])
        satisfied = {c for c in required | preferred if evaluate_capability(c, records).status == "PRESENT"}

        # Share of required + preferred capabilities positively evidenced. A required one that is merely UNKNOWN (a real hospital
        # nobody has reported on) earns nothing here, so declared/confirmed facilities lift a hospital above blank ones.
        wanted = required | preferred
        cap = len(satisfied & wanted) / len(wanted) if wanted else 1.0
        res = _res_component(required | preferred, records)
        capy = _capy_component(records)
        spec = _spec_component(required | preferred, records)

        if cand.eta_min is None:
            acc = 0.0
        elif eta_hi == eta_lo:
            acc = 1.0
        else:
            acc = 1.0 - (cand.eta_min - eta_lo) / (eta_hi - eta_lo)

        used_keys = {"emergency_beds", "ipd_admission_delay_est_min"}
        for c in required | preferred:
            used_keys.update(evaluate_capability(c, records).keys_used)
            if c in SPECIALIST_CAPABILITIES:
                used_keys.add(f"specialist_{c}")
        used = [r.confidence for k in used_keys if (r := latest(records, k)) is not None]
        fresh = _mean(used, _mean([r.confidence for r in records], 0.0))

        beds = bed_score(records, bed_keys) if bed_metric else 0.0
        score = (
            w_cap * cap + w_res * res * fresh + w_capy * capy * fresh
            + w_spec * spec * fresh + w_acc * acc + w_fresh * fresh + w_beds * beds
        )
        contrib = {
            "CAP": w_cap * cap, "RES": w_res * res * fresh, "CAPY": w_capy * capy * fresh,
            "SPEC": w_spec * spec * fresh, "ACC": w_acc * acc, "FRESH": w_fresh * fresh,
        }
        comps = {"CAP": cap, "RES": res, "CAPY": capy, "SPEC": spec, "ACC": acc, "FRESH": fresh}
        if bed_metric:
            contrib["BEDS"] = w_beds * beds
            comps["BEDS"] = beds
        provisional = cand.hospital_id in provisional_ids
        penalty = PROVISIONAL_PENALTY if provisional else 1.0
        score *= penalty
        ai_adj = max(-MAX_AI_ADJUSTMENT, min(MAX_AI_ADJUSTMENT, float((ai_adjustments or {}).get(cand.hospital_id, 0.0))))
        score += ai_adj

        eta_txt = f"ETA about {cand.eta_min:.0f} min (tier {cand.eta_source_tier} estimate)" if cand.eta_min is not None else "ETA unknown"
        reason = (
            f"Meets all {len(required)} required capabilities"
            f"{' (some pending verification)' if provisional else ''}; "
            f"{len(satisfied & preferred)}/{len(preferred)} preferred confirmed; "
            f"{eta_txt}; resource data confidence {fresh:.2f}."
        )
        if bed_metric:
            det = bed_detail(records, bed_keys)
            reason += " Beds free: " + ", ".join(
                f"{BED_LABELS[k]} {'?' if d['available'] is None else d['available']}" for k, d in det.items()) + "."
        scored.append((cand, score, comps,
                       sorted(satisfied), fresh, reason, provisional, records, contrib, penalty, ai_adj))

    scored.sort(key=lambda t: (-t[1], t[0].eta_min if t[0].eta_min is not None else float("inf"), t[0].hospital_id))
    return [
        RankedCandidate(
            candidate=c, rank=i, score=s, components=comp, capability_match=match,
            resource_confidence=fr, reason=rs, provisional=prov,
            required=frozenset(required), records=recs, contributions=contrib, penalty=pen, ai_adjustment=aj,
        )
        for i, (c, s, comp, match, fr, rs, prov, recs, contrib, pen, aj) in enumerate(scored, start=1)
    ]
