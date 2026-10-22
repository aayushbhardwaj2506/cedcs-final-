"""Run the whole evaluation:  python -m evaluation.run [--worlds 10] [--cases 100] [--seed 42] [--llm-baseline 0] [--real-latency]

Writes evaluation/results/: results.json (everything), CSV files, figures/ and RESULTS.md (paper-ready tables). Deterministic for a given
seed, except B5 (a real language model) and the real-latency table (it reads runs saved by the live application).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation import experiments as E  # noqa: E402
from evaluation import report  # noqa: E402


def _clean(o):
    """JSON-safe copy (tuples -> lists, NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and o != o:
        return None
    return o


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=10)
    ap.add_argument("--cases", type=int, default=100, help="cases per world")
    ap.add_argument("--hospitals", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42, help="first world seed; world i uses seed+i")
    ap.add_argument("--llm-baseline", type=int, default=0, help="also run B5 (a real LLM, no gate) on this many cases")
    ap.add_argument("--llm-triage", type=int, default=0, help="also run E4 with the real AI triage on this many cases per noise level (0 and 3)")
    ap.add_argument("--real-latency", action="store_true", help="include the latency of real pipeline runs saved in the history")
    ap.add_argument("--only", default="", help="comma-separated steps (e.g. e4_llm,b5) to redo on top of the existing results.json")
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "results"))
    a = ap.parse_args(argv)
    seeds = list(range(a.seed, a.seed + a.worlds))
    t0 = time.time()
    res: dict = {}
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    if only:  # redo some steps and keep everything else from the previous run
        res = {k: v for k, v in json.loads((Path(a.out) / "results.json").read_text(encoding="utf-8")).items() if k != "config"}
    steps = [("e1", lambda: E.e1_main(seeds, a.cases)), ("e2", lambda: E.e2_freshness(seeds, a.cases)), ("e3", lambda: E.e3_safety(seeds, a.cases)),
             ("e4", lambda: E.e4_noise(seeds, a.cases)), ("e5", lambda: E.e5_weights(seeds, a.cases)), ("e6", lambda: E.e6_escalation(seeds, a.cases)),
             ("latency_core", lambda: E.latency_core(seeds, a.cases))]
    for name, fn in steps:
        if only and name not in only:
            continue
        t = time.time()
        res[name] = fn()
        print(f"{name}: {time.time() - t:.1f}s", flush=True)
    if a.llm_baseline and (not only or "b5" in only):
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        t = time.time()
        res["b5"] = E.b5_llm(seeds, a.llm_baseline)
        print(f"b5: {res['b5']['n']} cases, {res['b5']['errors']} unusable, {time.time() - t:.1f}s", flush=True)
    if a.llm_triage and (not only or "e4_llm" in only):
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        t = time.time()
        res["e4_llm"] = E.e4_llm(seeds, a.llm_triage)
        print(f"e4_llm: {time.time() - t:.1f}s", flush=True)
    if a.real_latency and (not only or "latency_real" in only):
        try:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env")
            from services import analytics

            res["latency_real"] = analytics.analytics()
        except Exception as exc:  # the history database may be unreachable: the rest of the evaluation does not need it
            print("real latency unavailable:", type(exc).__name__, flush=True)
    out = Path(a.out)
    config = {"seed": a.seed, "worlds": a.worlds, "cases": a.cases, "hospitals": a.hospitals}
    md = report.build(res, out, config)
    (out / "RESULTS.md").write_text(md, encoding="utf-8")
    keep = {k: v for k, v in res.items() if k != "latency_real"}
    if "latency_real" in res:
        keep["latency_real"] = {"window": res["latency_real"]["window"], "latency": {k: v for k, v in res["latency_real"]["latency"].items() if k != "series"},
                                "ai": res["latency_real"]["ai"]}
    (out / "results.json").write_text(json.dumps(_clean({"config": config, **keep}), indent=1), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
