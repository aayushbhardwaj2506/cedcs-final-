"""The experiments of the paper's Table 4 (E1-E6), plus the latency decomposition. Each returns plain dicts of numbers."""

from __future__ import annotations

import json
import random
import re
import time
from typing import Optional

from deterministic_core import ranking
from deterministic_core.capability_map import evaluate_capability
from evaluation import metrics, policies, world
from evaluation.policies import Decision, cedcs, triage_for
from evaluation.world import Case, T0

PRIORITIES = ["LOW", "MODERATE", "HIGH", "CRITICAL"]


def _worlds(seeds, cases_per_world, **wkw):
    for seed in seeds:
        hs = world.build_world(seed, **wkw)
        yield seed, hs, world.build_cases(seed, cases_per_world)
        policies.clear_cache()


def run_policies(seeds, cases_per_world, pols: dict, *, with_oracle=True, triage_fn=None, **wkw) -> dict:
    """{policy name: [judged row, ...]} over every case of every world. The same cases go to every policy (paired comparison)."""
    rows = {k: [] for k in pols} | ({"oracle": []} if with_oracle else {})
    for _, hs, cs in _worlds(seeds, cases_per_world, **wkw):
        for c in cs:
            tr = triage_fn(c) if triage_fn else triage_for(c)
            for name, fn in pols.items():
                rows[name].append(metrics.judge(c, fn(c, hs, tr), hs))
            if with_oracle:
                rows["oracle"].append(metrics.judge(c, policies.oracle(c, hs), hs))
    return rows


def _fails(rows: list) -> list:
    return [bool(r["issued"] and not r["appropriate"]) for r in rows]


# ---------------------------------------------------------------- E1 main comparison
def e1_main(seeds, cases_per_world, loads=("relaxed", "normal", "crowded")) -> dict:
    out = {}
    for load in loads:
        rows = run_policies(seeds, cases_per_world, policies.POLICIES, load=load)
        summ = {k: metrics.summarise(v) for k, v in rows.items()}
        base = _fails(rows["CEDCS"])
        paired = {k: metrics.mcnemar_exact(_fails(rows[k]), base) for k in rows if k not in ("CEDCS", "oracle")}
        by_pri = {}
        for k in ("B3 distance + specialty", "B4 soft ranking, no gate, no freshness", "CEDCS"):
            by_pri[k] = {p: metrics.summarise([r for r in rows[k] if r["priority"] == p])["undertriage"] for p in PRIORITIES}
        out[load] = {"summary": summ, "mcnemar_p_vs_cedcs": paired, "calibration": metrics.calibration(rows["CEDCS"]), "undertriage_by_priority": by_pri,
                     "rungs": _count([r["rung"] for r in rows["CEDCS"]]), "n_provisional_mean": _mean([r["n_provisional"] for r in rows["CEDCS"]]),
                     "n_eligible_mean": _mean([r["n_eligible"] for r in rows["CEDCS"]])}
    return out


def _count(xs):
    d: dict = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


# ---------------------------------------------------------------- B5: an LLM chooses a hospital, with no gate (subset, real model calls)
def b5_llm(seeds, n_cases: int, load="normal") -> dict:
    from llm import groq_agents

    rows, errors = [], 0
    pols = {"CEDCS": policies.POLICIES["CEDCS"], "B4 soft ranking, no gate, no freshness": policies.POLICIES["B4 soft ranking, no gate, no freshness"]}
    cmp_rows = {k: [] for k in pols}
    done = 0
    for _, hs, cs in _worlds(seeds, max(1, n_cases // len(list(seeds))), load=load):
        for c in cs:
            if done >= n_cases:
                break
            decision = _llm_choose(groq_agents, c, hs)
            if decision is None:
                errors += 1
                continue
            rows.append(metrics.judge(c, decision, hs))
            for k, fn in pols.items():
                cmp_rows[k].append(metrics.judge(c, fn(c, hs, triage_for(c)), hs))
            done += 1
    return {"n": len(rows), "errors": errors, "B5": metrics.summarise(rows) if rows else None,
            "same_cases": {k: metrics.summarise(v) for k, v in cmp_rows.items() if v}}


def _llm_choose(groq_agents, case: Case, hospitals: list) -> Optional[Decision]:
    near = sorted((h for h in hospitals if world.distance_km(case, h) <= policies.WIDE_KM), key=lambda h: world.distance_km(case, h))[:12]
    if not near:
        return None
    lines = []
    for h in near:
        deps = [k.split("_", 1)[1] for k, r in h.reported.items() if k.startswith("department_") and r.value.get("active")]
        eq = [k.split("_", 1)[1] for k, r in h.reported.items() if k.startswith("equipment_") and r.value.get("operational")]
        beds = {k.replace("_beds", ""): r.value.get("available") for k, r in h.reported.items() if k in ("icu_beds", "emergency_beds")}
        ages = [(T0 - r.updated_at).total_seconds() / 3600 for r in h.reported.values()]
        lines.append(f"- {h.hid}: {world.eta_minutes(world.distance_km(case, h))} min away; departments {deps}; equipment {eq}; free beds {beds}; "
                     f"blood bank {h.reported['blood_bank'].value['available'] if 'blood_bank' in h.reported else 'unknown'}; "
                     f"data last reported {round(sum(ages) / len(ages), 1) if ages else '?'} h ago on average")
    system = ("You help an emergency dispatcher choose ONE hospital for a patient. Use only the data given. Reply with JSON only: "
              '{"hospital_id": "<id from the list>", "reason": "<one sentence>"}')
    user = f"PATIENT REPORT: {case.text}\n\nCANDIDATE HOSPITALS:\n" + "\n".join(lines)
    info = groq_agents.CallInfo("INTAKE", groq_agents.model_for("INTAKE"), groq_agents.provider_for("INTAKE"))
    t0 = time.perf_counter()
    try:
        text = groq_agents._chat(system, user, info, json_mode=True, max_tokens=300)
        hid = json.loads(text[text.find("{"): text.rfind("}") + 1]).get("hospital_id")
    except Exception:
        return None
    if hid not in {h.hid for h in near}:
        return None
    return Decision(hid, ms=(time.perf_counter() - t0) * 1000)


# ---------------------------------------------------------------- E2 freshness ablation
def e2_freshness(seeds, cases_per_world, ages=(0.0, 1.0, 4.0, 12.0), load="normal") -> dict:
    pols = {"CEDCS (freshness on)": lambda c, hs, t: cedcs(c, hs, t, freshness=True),
            "CEDCS (freshness off)": lambda c, hs, t: cedcs(c, hs, t, freshness=False),
            "B4 soft ranking, no gate, no freshness": policies.POLICIES["B4 soft ranking, no gate, no freshness"]}
    out = {}
    for a in ages:
        rows = run_policies(seeds, cases_per_world, pols, with_oracle=False, fixed_age_h=a, load=load)
        out[str(a)] = {"summary": {k: metrics.summarise(v) for k, v in rows.items()},
                       "mcnemar_p_on_vs_off": metrics.mcnemar_exact(_fails(rows["CEDCS (freshness on)"]), _fails(rows["CEDCS (freshness off)"])),
                       "calibration_on": metrics.calibration(rows["CEDCS (freshness on)"]), "calibration_off": metrics.calibration(rows["CEDCS (freshness off)"])}
    return out


# ---------------------------------------------------------------- E3 safety stress test: wrong triage is injected
def e3_safety(seeds, cases_per_world, probs=(0.0, 0.1, 0.2, 0.3, 0.5), load="normal") -> dict:
    out = {}
    for p in probs:
        rng = random.Random(int(p * 1000) + 3)
        given_viol = {"CEDCS": 0, "B4 soft ranking, no gate, no freshness": 0}
        issued = {"CEDCS": 0, "B4 soft ranking, no gate, no freshness": 0}
        corrupted = {}
        fns = {"CEDCS": cedcs, "B4 soft ranking, no gate, no freshness": policies.b4_soft_ranking, "B3 distance + specialty": policies.b3_distance_plus_specialty}
        rows = {"CEDCS": [], "B4 soft ranking, no gate, no freshness": [], "B3 distance + specialty": []}
        for _, hs, cs in _worlds(seeds, cases_per_world, load=load):
            for c in cs:
                bad = rng.random() < p
                tr = triage_for(c, drop_required=1 if bad else 0, priority=_down(c.template.priority) if bad else None, rng=rng)
                corrupted[c.cid] = bad
                for name in rows:
                    d = fns[name](c, hs, tr)
                    rows[name].append(metrics.judge(c, d, hs))
                    if name in given_viol and d.hid is not None:
                        issued[name] += 1
                        h = next(x for x in hs if x.hid == d.hid)
                        if any(evaluate_capability(cap, policies.scored_records(h, True)).status == "ABSENT" for cap in tr.required_capabilities):
                            given_viol[name] += 1
        out[str(p)] = {"summary": {k: metrics.summarise(v) for k, v in rows.items()},
                       "violations_of_given_profile": {k: {"count": given_viol[k], "of": issued[k]} for k in given_viol},
                       "share_corrupted": sum(corrupted.values()) / len(corrupted),
                       "undertriage_when_corrupted": {k: metrics.summarise([r for r in v if corrupted[r["case"]]])["undertriage"] for k, v in rows.items()}}
    return out


def _down(p: str) -> str:
    return PRIORITIES[max(0, PRIORITIES.index(p) - 1)]


# ---------------------------------------------------------------- E4 input-noise robustness (rule-based triage of a noisy text)
FILLERS = ["please hurry", "oh god he is dying", "HELP HELP", "i dont know what to do", "come fast", "my husband", "he fell", "ambulance please"]


def noisy(text: str, level: int, rng: random.Random, other_texts: list) -> str:
    if level <= 0:
        return text
    clauses = [c.strip() for c in text.split(",") if c.strip()]
    if len(clauses) > 2:  # level 1: lose information (age or a symptom)
        clauses.pop(0 if rng.random() < 0.5 else -1)
    out = ", ".join(clauses)
    if level >= 2 and rng.random() < 0.35:  # level 2: a factual error, the wrong complaint
        out = rng.choice(other_texts)
    if level >= 3:  # level 3: panic phrasing, cut off
        words = out.split()
        words = words[: max(3, int(len(words) * 0.6))]
        words.insert(rng.randrange(len(words) + 1), rng.choice(FILLERS))
        out = " ".join(w.upper() if rng.random() < 0.3 else w for w in words)
    return out


def e4_noise(seeds, cases_per_world, levels=(0, 1, 2, 3), load="normal") -> dict:
    from fallback import rule_based
    from schemas.triage import TriageResult

    out = {}
    all_texts = [t for tpl in world.TEMPLATES for t in tpl.texts]
    for lvl in levels:
        rng = random.Random(40 + lvl)
        rows, recalls, under_pri, n = [], [], 0, 0
        for _, hs, cs in _worlds(seeds, cases_per_world, load=load):
            for c in cs:
                text = noisy(c.text, lvl, rng, [t for t in all_texts if t not in c.template.texts])
                raw = {"emergency_report": text, "consciousness": "unknown", "breathing": "unknown", "bleeding": "unknown",
                       "location_lat": c.lat, "location_lng": c.lng, "location_source": "MANUAL"}
                derived = rule_based.assess_triage(rule_based.parse_intake(raw))
                req = set(derived.required_capabilities)
                truth = set(c.template.required)
                recalls.append(len(req & truth) / len(truth))
                under_pri += PRIORITIES.index(derived.priority) < PRIORITIES.index(c.template.priority)
                n += 1
                tr = TriageResult(priority=derived.priority, category_set=[x for x in derived.category_set], required_capabilities=sorted(req) or ["EMERGENCY_DEPARTMENT"],
                                  preferred_capabilities=list(derived.preferred_capabilities), triage_confidence=derived.triage_confidence, rationale="rule-based triage")
                rows.append(metrics.judge(c, cedcs(c, hs, tr), hs))
        s = metrics.summarise(rows)
        out[str(lvl)] = {"summary": s, "required_recall_mean": _mean(recalls), "priority_underestimated": under_pri / n, "n": n,
                         "calibration": metrics.calibration(rows)}
    return out


def e4_llm(seeds, n_cases: int, levels=(0, 3), load="normal") -> dict:
    """The same degraded texts, triaged by the application's real AI agents (intake + triage), on a subset of cases."""
    from fallback import rule_based
    from llm import groq_agents
    from schemas.triage import TriageResult

    all_texts = [t for tpl in world.TEMPLATES for t in tpl.texts]
    out = {}
    for lvl in levels:
        rng = random.Random(40 + lvl)  # the same texts as the rule-based run, so the two are paired
        rows, rows_rules, recalls, recalls_rules, under, under_rules, n, errors = [], [], [], [], 0, 0, 0, 0
        for _, hs, cs in _worlds(seeds, max(1, n_cases // len(list(seeds))), load=load):
            for c in cs:
                if n >= n_cases:
                    break
                text = noisy(c.text, lvl, rng, [t for t in all_texts if t not in c.template.texts])
                raw = {"emergency_report": text, "consciousness": "unknown", "breathing": "unknown", "bleeding": "unknown",
                       "location_lat": c.lat, "location_lng": c.lng, "location_source": "MANUAL"}
                try:
                    case_obj, _ = groq_agents.intake(raw)
                    ai, _ = groq_agents.triage(case_obj)
                except Exception:
                    errors += 1
                    continue
                rules = rule_based.assess_triage(rule_based.parse_intake(raw))
                truth = set(c.template.required)
                for derived, rr, rc_list in ((ai, rows, recalls), (rules, rows_rules, recalls_rules)):
                    req = set(derived.required_capabilities)
                    rc_list.append(len(req & truth) / len(truth))
                    tr = TriageResult(priority=derived.priority, category_set=list(derived.category_set), required_capabilities=sorted(req) or ["EMERGENCY_DEPARTMENT"],
                                      preferred_capabilities=list(derived.preferred_capabilities), triage_confidence=derived.triage_confidence, rationale="triage")
                    rr.append(metrics.judge(c, cedcs(c, hs, tr), hs))
                under += PRIORITIES.index(ai.priority) < PRIORITIES.index(c.template.priority)
                under_rules += PRIORITIES.index(rules.priority) < PRIORITIES.index(c.template.priority)
                n += 1
        out[str(lvl)] = {"n": n, "errors": errors,
                         "ai": {"summary": metrics.summarise(rows), "recall": _mean(recalls), "priority_underestimated": under / n if n else float("nan")},
                         "rules_same_cases": {"summary": metrics.summarise(rows_rules), "recall": _mean(recalls_rules), "priority_underestimated": under_rules / n if n else float("nan")}}
    return out


# ---------------------------------------------------------------- E5 weight sensitivity
def e5_weights(seeds, cases_per_world, draws=20, spread=0.2, load="normal") -> dict:
    base_rows = {}
    base_choice = {}
    cases = []
    for _, hs, cs in _worlds(seeds, cases_per_world, load=load):
        for c in cs:
            d = cedcs(c, hs, triage_for(c))
            base_choice[c.cid] = d.hid
            base_rows[c.cid] = metrics.judge(c, d, hs)
        cases.append((hs, cs))
    base_under = metrics.summarise(list(base_rows.values()))["undertriage"]
    rng = random.Random(55)
    results = []
    default = dict(ranking.WEIGHTS)
    for _ in range(draws):
        w = {}
        for pri, vec in default.items():
            v = [x * rng.uniform(1 - spread, 1 + spread) for x in vec]
            s = sum(v)
            w[pri] = tuple(x / s * sum(vec) for x in v)
        same = tot = 0
        rows = []
        with policies.weights(w):
            policies.clear_cache()
            for hs, cs in cases:
                for c in cs:
                    d = cedcs(c, hs, triage_for(c))
                    rows.append(metrics.judge(c, d, hs))
                    same += d.hid == base_choice[c.cid]
                    tot += 1
        s = metrics.summarise(rows)
        results.append({"undertriage": s["undertriage"], "top1_agreement": same / tot, "excess_min": s["excess_min"][0]})
        policies.clear_cache()
    u = [r["undertriage"] for r in results]
    ag = [r["top1_agreement"] for r in results]
    return {"baseline_undertriage": base_under, "draws": draws, "spread": spread, "undertriage_min": min(u), "undertriage_max": max(u), "undertriage_mean": _mean(u),
            "top1_agreement_min": min(ag), "top1_agreement_mean": _mean(ag), "top1_agreement_max": max(ag), "per_draw": results}


# ---------------------------------------------------------------- E6 escalation under constrained supply
KNOCKOUT = [("icu_beds", {"available": 0}), ("equipment_CT_SCAN", {"operational": False}), ("department_CARDIOLOGY", {"active": False}),
            ("department_NEUROLOGY", {"active": False}), ("equipment_OT", {"operational": False}), ("blood_bank", {"available": False})]


def _degrade(hospitals: list, q: float, rng: random.Random) -> None:
    """Remove capabilities from hospitals in BOTH the true state and their reports (the system knows about the shortage)."""
    for h in hospitals:
        for key, patch in KNOCKOUT:
            if rng.random() < q:
                if key in h.true:
                    h.true[key] = {**h.true[key], **patch}
                if key in h.reported:
                    r = h.reported[key]
                    h.reported[key] = r.model_copy(update={"value": {**r.value, **patch}})
        h.__dict__.pop("_truth", None)


def e6_escalation(seeds, cases_per_world, qs=(0.0, 0.2, 0.4, 0.6, 0.8, 0.95), load="normal") -> dict:
    out = {}
    for q in qs:
        rng = random.Random(int(q * 100) + 9)
        rows = []
        for seed in seeds:
            hs = world.build_world(seed, load=load)
            _degrade(hs, q, rng)
            for c in world.build_cases(seed, cases_per_world):
                rows.append(metrics.judge(c, cedcs(c, hs, triage_for(c)), hs))
            policies.clear_cache()
        s = metrics.summarise(rows)
        issued = [r for r in rows if r["issued"]]
        out[str(q)] = {"n": len(rows), "handoff_rate": s["handoff_rate"], "widened_rate": sum(1 for r in rows if r["rung"] == "widened") / len(rows),
                       "recommended_rate": len(issued) / len(rows), "undertriage": s["undertriage"], "violations": s["violations"],
                       "avoidable_share": sum(1 for r in rows if r["avoidable"]) / len(rows),
                       "handoff_when_avoidable": sum(1 for r in rows if r["handoff"] and r["avoidable"]) / max(1, sum(1 for r in rows if r["avoidable"])),
                       "handoff_when_unavoidable": sum(1 for r in rows if r["handoff"] and not r["avoidable"]) / max(1, sum(1 for r in rows if not r["avoidable"])),
                       "rungs": _count([r["rung"] for r in rows])}
    return out


# ---------------------------------------------------------------- latency
def latency_core(seeds, cases_per_world, load="normal") -> dict:
    """Wall-clock time of the deterministic decision core alone (freshness, gate, ranking, confidence) per case."""
    ms = []
    for _, hs, cs in _worlds(seeds, cases_per_world, load=load):
        for c in cs:
            ms.append(cedcs(c, hs, triage_for(c)).ms)
    ms.sort()
    return {"n": len(ms), "mean_ms": sum(ms) / len(ms), "p50_ms": ms[len(ms) // 2], "p95_ms": ms[int(len(ms) * 0.95) - 1], "max_ms": ms[-1]}
