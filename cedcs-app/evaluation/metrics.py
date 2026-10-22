"""Judging a recommendation against the hospital's TRUE state, and summarising many of them with honest uncertainty.

Definitions (they are the ones in the paper's Table 3):
  appropriate        every REQUIRED capability of the case's ground-truth profile is truly present at the chosen hospital at T0.
  undertriage        a recommendation that is not appropriate: the patient would arrive somewhere that lacks a REQUIRED capability
                     and be transferred on (an UNPLANNED secondary transfer). Reported over issued recommendations.
  avoidable case     some hospital within 25 km was truly appropriate, so a perfect policy would have avoided the transfer.
  handoff            the policy issued no recommendation (CEDCS: nothing eligible, a human takes over: a planned pathway, not an error).
  safety violation   a recommendation for a hospital whose REPORTED data says a REQUIRED capability is absent (the gate must make this 0).
  overtriage         a LOW/MODERATE case sent to a tertiary or trauma centre although a closer, truly appropriate smaller hospital existed.
  excess travel      ETA of the chosen hospital minus ETA of the nearest truly appropriate one (avoidable cases only).
  cost-weighted err  L_UNDER * undertriage + L_OVER * overtriage + L_TIME * excess minutes (per issued recommendation).
"""

from __future__ import annotations

import math
import random
from typing import Optional

from deterministic_core.capability_map import evaluate_capability
from evaluation.policies import WIDE_KM, Decision, scored_records
from evaluation.world import Case, Hospital, distance_km, eta_minutes, true_status

L_UNDER, L_OVER, L_TIME = 10.0, 1.0, 0.05  # lambda_under >> lambda_over > lambda_time, per minute (the paper's ordering; magnitudes assumed)
BIG = ("tertiary", "trauma")


def judge(case: Case, d: Decision, hospitals: list) -> dict:
    """One recommendation, judged. `hospitals` is the whole world (the evaluator needs it for the oracle comparison)."""
    req, pref = case.template.required, case.template.preferred
    near = [h for h in hospitals if distance_km(case, h) <= WIDE_KM]
    good = [h for h in near if all(true_status(h, c) == "PRESENT" for c in req)]
    best_eta = min((eta_minutes(distance_km(case, h)) for h in good), default=None)
    row = {"case": case.cid, "template": case.template.name, "priority": case.template.priority, "avoidable": bool(good), "issued": d.hid is not None,
           "handoff": d.hid is None, "level": d.level, "rung": d.rung, "ms": d.ms, "n_eligible": d.n_eligible, "n_provisional": d.n_provisional}
    if d.hid is None:
        return row | {"appropriate": None, "req_cov": None, "full_cov": None, "violation": None, "over": None, "excess": None, "cost": None}
    h = next(x for x in hospitals if x.hid == d.hid)
    req_ok = [true_status(h, c) == "PRESENT" for c in req]
    all_ok = [true_status(h, c) == "PRESENT" for c in (*req, *pref)]
    appropriate = all(req_ok)
    reported = scored_records(h, True)
    violation = any(evaluate_capability(c, reported).status == "ABSENT" for c in req)
    eta = eta_minutes(distance_km(case, h))
    over = (case.template.priority in ("LOW", "MODERATE") and h.archetype in BIG
            and any(g.archetype not in BIG and distance_km(case, g) < distance_km(case, h) for g in good))
    excess = max(0.0, eta - best_eta) if (good and best_eta is not None) else None
    cost = L_UNDER * (not appropriate) + L_OVER * over + L_TIME * (excess or 0.0)
    return row | {"appropriate": appropriate, "req_cov": sum(req_ok) / len(req), "full_cov": sum(all_ok) / len(all_ok), "violation": violation,
                  "over": over, "excess": excess, "cost": cost, "eta": eta, "archetype": h.archetype}


# ---------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """95 % Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    lo, hi = max(0.0, c - h), min(1.0, c + h)
    return (0.0 if k == 0 else min(lo, p), 1.0 if k == n else max(hi, p))  # exact at the ends: no rounding can put p outside its interval


def mean_ci(xs: list, reps: int = 1000, seed: int = 1) -> tuple:
    """(mean, lo, hi): bootstrap 95 % interval of the mean."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    m = sum(xs) / len(xs)
    boots = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(reps))
    return (m, boots[int(0.025 * reps)], boots[int(0.975 * reps) - 1])


def mcnemar_exact(a_fail: list, b_fail: list) -> float:
    """Two-sided exact McNemar p-value for paired binary failures (a_fail[i], b_fail[i] for the same case)."""
    b = sum(1 for x, y in zip(a_fail, b_fail) if x and not y)
    c = sum(1 for x, y in zip(a_fail, b_fail) if y and not x)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summarise(rows: list) -> dict:
    """The Table-3 metrics for one policy over a list of judged recommendations."""
    n = len(rows)
    issued = [r for r in rows if r["issued"]]
    av = [r for r in issued if r["avoidable"]]
    fail = sum(1 for r in issued if not r["appropriate"])
    fail_av = sum(1 for r in av if not r["appropriate"])
    out = {"n": n, "issued": len(issued), "handoff_rate": (n - len(issued)) / n if n else float("nan"),
           "undertriage": fail / len(issued) if issued else float("nan"), "undertriage_ci": wilson(fail, len(issued)),
           "undertriage_avoidable": fail_av / len(av) if av else float("nan"), "undertriage_avoidable_ci": wilson(fail_av, len(av)), "n_avoidable": len(av),
           "violations": sum(1 for r in issued if r["violation"]), "violation_rate": sum(1 for r in issued if r["violation"]) / len(issued) if issued else float("nan"),
           "overtriage": sum(1 for r in issued if r["over"]) / len(issued) if issued else float("nan"),
           "req_coverage": mean_ci([r["req_cov"] for r in issued]), "full_coverage": mean_ci([r["full_cov"] for r in issued]),
           "excess_min": mean_ci([r["excess"] for r in av]), "cost": mean_ci([r["cost"] for r in issued]),
           "ms_mean": sum(r["ms"] for r in rows) / n if n else float("nan")}
    return out


def calibration(rows: list) -> dict:
    """By stated confidence level: how many recommendations were truly appropriate."""
    out = {}
    for lvl in ("HIGH", "MODERATE", "LOW"):
        sel = [r for r in rows if r["issued"] and r["level"] == lvl]
        k = sum(1 for r in sel if r["appropriate"])
        out[lvl] = {"n": len(sel), "appropriate": k / len(sel) if sel else float("nan"), "ci": wilson(k, len(sel))}
    return out
