"""Aggregations over the saved run history, for the Analytics dashboard and for research exports.

Everything here is computed from what services.history stored; nothing is estimated. Latency statistics use completed runs
only (a run that handed off to a human is counted separately), so one early halt does not drag the figures down.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from typing import Optional

from services import history

# case_runs summary columns, in the order they appear in the CSV export
RUN_COLUMNS = ["case_id", "started_at", "mode", "hospital_source", "total_ms", "halted", "halt_reason", "priority", "primary_hospital",
               "primary_hospital_id", "n_candidates", "n_eligible", "n_provisional", "ai_ok", "ai_fallback", "providers"]


def pct(sorted_vals: list, q: float) -> float:
    """Percentile with linear interpolation between neighbouring values (the usual definition, e.g. numpy's default)."""
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo))


def stats(vals: list) -> Optional[dict]:
    v = sorted(float(x) for x in vals if x is not None)
    if not v:
        return None
    return {"n": len(v), "mean": round(sum(v) / len(v), 1), "p50": round(pct(v, 0.5), 1), "p95": round(pct(v, 0.95), 1),
            "min": round(v[0], 1), "max": round(v[-1], 1)}


def _is_pg() -> bool:
    return history._get_primary().name == "neon"


def _extras(since: float) -> dict:
    """Per-run values that live inside the JSON columns (AI calls, confidence level, ETA of the primary hospital)."""
    if _is_pg():
        sql = ("SELECT case_id, trace->'llm' AS llm, recommendation->>'confidence_level' AS conf, "
               "recommendation->'primary'->>'eta_min' AS eta FROM case_runs WHERE started_at>=?")
    else:
        sql = ("SELECT case_id, json_extract(trace,'$.llm') AS llm, json_extract(recommendation,'$.confidence_level') AS conf, "
               "json_extract(recommendation,'$.primary.eta_min') AS eta FROM case_runs WHERE started_at>=?")
    out = {}
    for r in history._rows(sql, (since,)):
        llm = history._unj(r.get("llm")) or []
        out[r["case_id"]] = {"llm": llm if isinstance(llm, list) else [], "conf": r.get("conf"), "eta": r.get("eta")}
    return out


def _histogram(vals: list, bins: int = 10) -> dict:
    if not vals:
        return {"bins": []}
    lo, hi = min(vals), max(vals)
    width = (hi - lo) / bins or 1.0
    counts = [0] * bins
    for v in vals:
        counts[min(bins - 1, int((v - lo) / width))] += 1
    return {"bins": [{"lo": round(lo + i * width), "hi": round(lo + (i + 1) * width), "count": c} for i, c in enumerate(counts)]}


def analytics(since: float = 0.0, limit: int = 2000) -> dict:
    runs = history._rows(f"SELECT {', '.join(RUN_COLUMNS)} FROM case_runs WHERE started_at>=? ORDER BY started_at DESC LIMIT ?", (since, int(limit)))
    runs.sort(key=lambda r: float(r["started_at"]))
    extras = _extras(since)
    ids = {r["case_id"] for r in runs}
    for r in runs:
        x = extras.get(r["case_id"], {})
        r["confidence_level"], r["primary_eta_min"] = x.get("conf"), (float(x["eta"]) if x.get("eta") not in (None, "") else None)

    done = [r for r in runs if not r["halted"]]
    out: dict = {"window": {"since": since or None, "runs": len(runs), "first": runs[0]["started_at"] if runs else None,
                            "last": runs[-1]["started_at"] if runs else None}, "storage": history.status()}

    # ---- latency
    totals = [r["total_ms"] for r in done]
    stage_vals: dict = defaultdict(list)
    depth_of: dict = {}
    for s in history._rows("SELECT s.case_id, s.stage, s.depth, s.duration_ms FROM stage_timings s JOIN case_runs r ON r.case_id=s.case_id "
                           "WHERE r.started_at>=? AND r.halted=0", (since,)):
        if s["case_id"] in ids:
            stage_vals[s["stage"]].append(s["duration_ms"])
            depth_of[s["stage"]] = s["depth"]
    out["latency"] = {
        "total": stats(totals),
        "stages": sorted(({"stage": k, "depth": depth_of[k], **stats(v)} for k, v in stage_vals.items()), key=lambda x: (x["depth"], -x["p50"])),
        "histogram": _histogram([float(t) for t in totals if t is not None]),
        "series": [{"t": r["started_at"], "ms": r["total_ms"], "case_id": r["case_id"], "halted": bool(r["halted"]), "priority": r["priority"]} for r in runs],
    }

    # ---- AI behaviour
    by_provider: dict = defaultdict(lambda: {"calls": 0, "ms": [], "tokens": 0})
    by_stage: dict = defaultdict(Counter)
    calls = fallbacks = 0
    clean_runs = 0
    for r in runs:
        llm = extras.get(r["case_id"], {}).get("llm", [])
        if llm and all(c.get("status") == "ok" for c in llm):
            clean_runs += 1
        for c in llm:
            calls += 1
            if c.get("status") == "ok":
                p = by_provider[c.get("provider") or "unknown"]
                p["calls"] += 1
                p["ms"].append(c.get("ms"))
                p["tokens"] += int(c.get("tokens") or 0)
                by_stage[c.get("stage")][c.get("provider") or "unknown"] += 1
            else:
                fallbacks += 1
                by_stage[c.get("stage")]["fallback"] += 1
    out["ai"] = {
        "calls": calls, "fallbacks": fallbacks, "ok_rate": round(1 - fallbacks / calls, 4) if calls else None,
        "runs_fully_ai": clean_runs, "providers": {k: {"calls": v["calls"], "tokens": v["tokens"], "latency_ms": stats(v["ms"])} for k, v in by_provider.items()},
        "by_stage": {k: dict(v) for k, v in sorted(by_stage.items())},
    }

    # ---- decisions
    n_c = sum(r["n_candidates"] or 0 for r in done)
    n_e = sum(r["n_eligible"] or 0 for r in done)
    n_p = sum(r["n_provisional"] or 0 for r in done)
    out["decisions"] = {
        "completed": len(done), "handoffs": len(runs) - len(done),
        "priority": dict(Counter(r["priority"] or "unknown" for r in runs)), "hospital_source": dict(Counter(r["hospital_source"] or "unknown" for r in runs)),
        "mode": dict(Counter(r["mode"] or "unknown" for r in runs)), "confidence": dict(Counter(r["confidence_level"] or "n/a" for r in done)),
        "funnel": {"avg_candidates": round(n_c / len(done), 1) if done else 0, "avg_eligible": round(n_e / len(done), 1) if done else 0,
                   "avg_provisional": round(n_p / len(done), 1) if done else 0,
                   "avg_rejected": round((n_c - n_e - n_p) / len(done), 1) if done else 0},
        "primary_eta_min": stats([r["primary_eta_min"] for r in done]),
        "top_hospitals": [{"name": k, "count": v} for k, v in Counter(r["primary_hospital"] for r in done if r["primary_hospital"]).most_common(8)],
    }

    # ---- dispatch loop, from the event log
    ev = history._rows("SELECT ts, kind, case_id, hospital_id, detail FROM events WHERE ts>=? ORDER BY ts LIMIT 20000", (since,))
    sent: dict = {}
    first_reply: dict = {}
    responses: Counter = Counter()
    kinds: Counter = Counter()
    for e in ev:
        kinds[e["kind"]] += 1
        d = history._unj(e.get("detail")) or {}
        if e["kind"] == "dispatch_sent" and e["case_id"]:
            sent.setdefault(e["case_id"], float(e["ts"]))
        elif e["kind"] == "hospital_response" and e["case_id"]:
            responses[d.get("response", "?")] += 1
            first_reply.setdefault(e["case_id"], float(e["ts"]))
    waits = [first_reply[c] - sent[c] for c in first_reply if c in sent and first_reply[c] >= sent[c]]
    out["dispatch"] = {
        "events": dict(kinds), "dispatches": len(sent), "responses": dict(responses),
        "accept_rate": round(responses.get("ACCEPTED", 0) / len(sent), 3) if sent else None,
        "time_to_first_reply_s": stats(waits),
    }
    out["runs"] = [{k: r[k] for k in RUN_COLUMNS + ["confidence_level", "primary_eta_min"]} for r in reversed(runs)]
    return out


# ---------------------------------------------------------------- research exports
def _csv(header: list, rows: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for r in rows:
        w.writerow(["" if v is None else v for v in r])
    return buf.getvalue()


def export_csv(kind: str, since: float = 0.0) -> Optional[str]:
    """CSV text for 'runs' (one row per run), 'stages' (one row per stage per run) or 'events'; None for an unknown kind."""
    if kind == "runs":
        a = analytics(since)
        cols = RUN_COLUMNS + ["confidence_level", "primary_eta_min"]
        return _csv(cols, [[r[c] for c in cols] for r in a["runs"]])
    if kind == "stages":
        rows = history._rows("SELECT s.case_id, s.seq, s.stage, s.depth, s.start_ms, s.duration_ms, r.started_at, r.priority FROM stage_timings s "
                             "JOIN case_runs r ON r.case_id=s.case_id WHERE r.started_at>=? ORDER BY r.started_at, s.seq", (since,))
        cols = ["case_id", "seq", "stage", "depth", "start_ms", "duration_ms", "started_at", "priority"]
        return _csv(cols, [[r[c] for c in cols] for r in rows])
    if kind == "events":
        rows = history._rows("SELECT ts, kind, case_id, hospital_id, detail FROM events WHERE ts>=? ORDER BY ts", (since,))
        return _csv(["ts", "kind", "case_id", "hospital_id", "detail"],
                    [[r["ts"], r["kind"], r["case_id"], r["hospital_id"], json.dumps(history._unj(r["detail"]), default=str)] for r in rows])
    return None
