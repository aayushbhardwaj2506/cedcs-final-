"""Persistent run history: every case that runs, with its full latency report, reasoning trace and outcome, plus an event log of
what happens around it (dispatches, hospital answers, bed reports, setting changes).

Stored in Postgres (Neon) when HISTORY_DATABASE_URL, or DATABASE_URL, is set, in its own schema (`cedcs_history`) so the hospital
tables are never touched. Otherwise, or when Postgres cannot be reached, in a local SQLite file. Writes go through a background
worker so a slow database never slows a case. Set CEDCS_HISTORY=off to disable.

Privacy: a saved run contains the caller's description of the emergency and the patient's coordinates. Family e-mail addresses
are never stored here.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("cedcs.history")

SCHEMA = "cedcs_history"
LOCAL_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "cedcs_history.db"
MAX_RUNS_DEFAULT = 2000
_TABLES = ("case_runs", "stage_timings", "events")
_TABLE_RE = re.compile(r"\b(" + "|".join(_TABLES) + r")\b")

# {J}: JSON column (JSONB in Postgres, so it can be queried in Neon's SQL editor), {F}: 8-byte float
_DDL = [
    """CREATE TABLE IF NOT EXISTS case_runs(
        case_id TEXT PRIMARY KEY, started_at {F} NOT NULL, mode TEXT, hospital_source TEXT, total_ms {F}, halted INTEGER,
        halt_reason TEXT, priority TEXT, primary_hospital TEXT, primary_hospital_id TEXT, n_candidates INTEGER, n_eligible INTEGER,
        n_provisional INTEGER, ai_ok INTEGER, ai_fallback INTEGER, providers TEXT, explanation TEXT,
        request {J}, recommendation {J}, trace {J}, timeline {J}, messages {J}, audit {J})""",
    """CREATE TABLE IF NOT EXISTS stage_timings(
        case_id TEXT NOT NULL, seq INTEGER NOT NULL, stage TEXT, start_ms {F}, duration_ms {F}, depth INTEGER, attrs {J},
        PRIMARY KEY(case_id, seq))""",
    """CREATE TABLE IF NOT EXISTS events(
        id TEXT PRIMARY KEY, ts {F} NOT NULL, kind TEXT NOT NULL, case_id TEXT, hospital_id TEXT, detail {J})""",
    "CREATE INDEX IF NOT EXISTS idx_runs_started ON case_runs(started_at)",
    "CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)",
    "CREATE INDEX IF NOT EXISTS idx_events_case ON events(case_id)",
]


def enabled() -> bool:
    return os.environ.get("CEDCS_HISTORY", "on").strip().lower() not in ("off", "0", "false", "no")


def _pg_url() -> str:
    if os.environ.get("CEDCS_HISTORY_BACKEND", "").strip().lower() == "sqlite":
        return ""
    return (os.environ.get("HISTORY_DATABASE_URL") or os.environ.get("APP_DATABASE_URL") or os.environ.get("DATABASE_URL") or "").strip()


def pg_translate(sql: str, schema: str) -> str:
    out = sql.replace("%", "%%").replace("?", "%s")
    return _TABLE_RE.sub(lambda m: f"{schema}.{m.group(1)}", out)


def _ddl(pg: bool) -> list:
    return [s.format(J="JSONB" if pg else "TEXT", F="DOUBLE PRECISION" if pg else "REAL") for s in _DDL]


class _Pg:
    name = "neon"

    def __init__(self, url: str, schema: str = SCHEMA):
        self.url, self.schema, self._c = url, schema, None
        self._connect()
        for stmt in _ddl(True):
            self.execute(stmt)

    def _connect(self) -> None:
        import psycopg2

        if self._c is not None:
            try:
                self._c.close()
            except Exception:
                pass
        self._c = psycopg2.connect(self.url, connect_timeout=10)
        self._c.autocommit = True
        with self._c.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")

    def execute(self, sql: str, params=()):
        import psycopg2
        from psycopg2.extras import RealDictCursor

        q = pg_translate(sql, self.schema)
        for attempt in (0, 1):
            try:
                cur = self._c.cursor(cursor_factory=RealDictCursor)
                cur.execute(q, tuple(params))
                return cur
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                if attempt:
                    raise
                self._connect()


class _Lite:
    name = "local file"

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._c = sqlite3.connect(path, check_same_thread=False)
        self._c.row_factory = sqlite3.Row
        for stmt in _ddl(False):
            self._c.execute(stmt)
        self._c.commit()

    def execute(self, sql: str, params=()):
        cur = self._c.execute(sql, tuple(params))
        self._c.commit()
        return cur


_lock = threading.RLock()
_primary = None
_fallback = None
_q: "queue.Queue" = queue.Queue()
_worker: Optional[threading.Thread] = None
_stats = {"saved_runs": 0, "saved_events": 0, "failed": 0, "fell_back": 0, "last_error": None}
_inserts = 0


def _local_path() -> Path:
    return Path(os.environ.get("CEDCS_HISTORY_DB") or LOCAL_DEFAULT)


def _get_primary():
    global _primary
    with _lock:
        if _primary is None:
            url = _pg_url()
            try:
                _primary = _Pg(url) if url else _Lite(_local_path())
            except Exception as exc:
                logger.warning("history: could not open the primary store (%s); using a local file", type(exc).__name__)
                _stats["last_error"] = f"primary store unavailable: {type(exc).__name__}"
                _primary = _get_fallback()
        return _primary


def _get_fallback():
    global _fallback
    with _lock:
        if _fallback is None:
            _fallback = _Lite(_local_path())
        return _fallback


def reset() -> None:
    """Forget open connections and queued work (tests, and after changing the environment)."""
    global _primary, _fallback, _inserts
    flush(2)
    with _lock:
        _primary = _fallback = None
        _inserts = 0
        _stats.update(saved_runs=0, saved_events=0, failed=0, fell_back=0, last_error=None)


def _j(v: Any) -> Any:
    return json.dumps(v, default=str)


def _unj(v: Any) -> Any:
    if isinstance(v, (str, bytes)):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


# ---------------------------------------------------------------- writing
def _ensure_worker() -> None:
    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_loop, name="history-writer", daemon=True)
            _worker.start()


def _loop() -> None:
    while True:
        item = _q.get()
        try:
            kind, payload = item
            _write_with_fallback(kind, payload)
        except Exception as exc:  # never let one bad item kill the writer
            _stats["failed"] += 1
            _stats["last_error"] = f"{type(exc).__name__}: {str(exc)[:100]}"
            logger.warning("history write failed: %s", exc)
        finally:
            _q.task_done()


def _write_with_fallback(kind: str, payload: dict) -> None:
    primary = _get_primary()
    try:
        _write(primary, kind, payload)
    except Exception as exc:
        fb = _get_fallback()
        if fb is primary:
            raise
        _stats["fell_back"] += 1
        _stats["last_error"] = f"{primary.name} write failed ({type(exc).__name__}); saved to the local file instead"
        _write(fb, kind, payload)
    _stats["saved_runs" if kind == "run" else "saved_events"] += 1


def _write(db, kind: str, p: dict) -> None:
    global _inserts
    if kind == "event":
        db.execute("INSERT INTO events(id,ts,kind,case_id,hospital_id,detail) VALUES(?,?,?,?,?,?)",
                   (p["id"], p["ts"], p["kind"], p["case_id"], p["hospital_id"], _j(p["detail"])))
        return
    r = p["row"]
    cols = list(r)
    db.execute(f"DELETE FROM stage_timings WHERE case_id=?", (r["case_id"],))
    db.execute(f"DELETE FROM case_runs WHERE case_id=?", (r["case_id"],))
    db.execute(f"INSERT INTO case_runs({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
               [(_j(v) if c in _JSON_COLS else v) for c, v in r.items()])
    for i, s in enumerate(p["stages"]):
        db.execute("INSERT INTO stage_timings(case_id,seq,stage,start_ms,duration_ms,depth,attrs) VALUES(?,?,?,?,?,?,?)",
                   (r["case_id"], i, s.get("stage"), s.get("start_ms"), s.get("duration_ms"), s.get("depth"),
                    _j({k: v for k, v in s.items() if k not in ("stage", "start_ms", "duration_ms", "depth")})))
    _inserts += 1
    if _inserts % 25 == 0:
        _prune(db)


_JSON_COLS = {"request", "recommendation", "trace", "timeline", "messages", "audit"}


def _prune(db) -> None:
    keep = int(os.environ.get("HISTORY_MAX_RUNS", MAX_RUNS_DEFAULT))
    old = db.execute("SELECT case_id FROM case_runs ORDER BY started_at DESC LIMIT 100000 OFFSET ?", (keep,)).fetchall()
    for row in old:
        cid = row["case_id"]
        db.execute("DELETE FROM stage_timings WHERE case_id=?", (cid,))
        db.execute("DELETE FROM case_runs WHERE case_id=?", (cid,))


def _enqueue(kind: str, payload: dict) -> None:
    if not enabled():
        return
    _ensure_worker()
    _q.put((kind, payload))


def record_case(request: dict, response: dict, hospital_source: str = "") -> None:
    """Queue one finished run (halted or not) for saving. `response` is the CaseResponse as a dict."""
    try:
        trace = response.get("trace") or {}
        rec = response.get("recommendation") or {}
        primary = rec.get("primary") or {}
        elig = (trace.get("eligibility") or {}).get("candidates") or []
        llm = trace.get("llm") or []
        _enqueue("run", {"row": {
            "case_id": response["case_id"], "started_at": time.time(), "mode": response.get("mode"), "hospital_source": hospital_source,
            "total_ms": response.get("total_ms"), "halted": int(bool(response.get("halted"))), "halt_reason": response.get("halt_reason"),
            "priority": (trace.get("red_flags") or {}).get("final_priority") or (trace.get("triage") or {}).get("priority"),
            "primary_hospital": primary.get("name"), "primary_hospital_id": primary.get("hospital_id"),
            "n_candidates": len(elig), "n_eligible": sum(c.get("status") == "ELIGIBLE" for c in elig),
            "n_provisional": sum(c.get("status") == "PROVISIONAL" for c in elig),
            "ai_ok": sum(c.get("status") == "ok" for c in llm), "ai_fallback": sum(c.get("status") == "fallback" for c in llm),
            "providers": ",".join(sorted({c.get("provider") for c in llm if c.get("status") == "ok" and c.get("provider")})),
            "explanation": response.get("explanation_text"),
            "request": request, "recommendation": rec or None, "trace": trace, "timeline": response.get("timeline") or [],
            "messages": response.get("messages") or [], "audit": response.get("audit") or [],
        }, "stages": response.get("timeline") or []})
    except Exception as exc:  # recording must never break a case
        _stats["failed"] += 1
        _stats["last_error"] = f"{type(exc).__name__}: {str(exc)[:100]}"


def event(kind: str, case_id: Optional[str] = None, hospital_id: Optional[str] = None, **detail: Any) -> None:
    """Queue one thing that happened (dispatch sent, hospital answered, beds reported, ...). Never raises."""
    try:
        _enqueue("event", {"id": uuid.uuid4().hex[:16], "ts": time.time(), "kind": kind, "case_id": case_id, "hospital_id": hospital_id, "detail": detail})
    except Exception:
        pass


def flush(timeout: float = 15.0) -> bool:
    """Wait until everything queued has been written (tests, shutdown). True if the queue drained in time."""
    end = time.time() + timeout
    while _q.unfinished_tasks and time.time() < end:
        time.sleep(0.02)
    return not _q.unfinished_tasks


# ---------------------------------------------------------------- reading
def _stores() -> list:
    """Where history may live: the primary store, then the local fallback if anything was written there."""
    out = [_get_primary()]
    if _fallback is not None and _fallback is not out[0]:
        out.append(_fallback)
    return out


def _rows(sql: str, params=()) -> list:
    """Rows from every store that may hold history; an identical row present in two stores is returned once."""
    seen: dict = {}
    for db in _stores():
        try:
            for r in db.execute(sql, params).fetchall():
                row = dict(r)
                seen.setdefault(json.dumps(row, sort_keys=True, default=str), row)
        except Exception as exc:
            _stats["last_error"] = f"read failed on {db.name}: {type(exc).__name__}"
    return list(seen.values())


_SUMMARY = ("case_id,started_at,mode,hospital_source,total_ms,halted,halt_reason,priority,primary_hospital,primary_hospital_id,"
            "n_candidates,n_eligible,n_provisional,ai_ok,ai_fallback,providers")


def recent_runs(limit: int = 50) -> list:
    rows = _rows(f"SELECT {_SUMMARY} FROM case_runs ORDER BY started_at DESC LIMIT ?", (int(limit),))
    return sorted(rows, key=lambda r: -float(r["started_at"]))[: int(limit)]


def get_run(case_id: str) -> Optional[dict]:
    rows = _rows("SELECT * FROM case_runs WHERE case_id=?", (case_id,))
    if not rows:
        return None
    r = rows[0]
    for c in _JSON_COLS:
        r[c] = _unj(r.get(c))
    r["events"] = recent_events(200, case_id=case_id)
    return r


def recent_events(limit: int = 100, case_id: Optional[str] = None) -> list:
    if case_id:
        rows = _rows("SELECT id,ts,kind,case_id,hospital_id,detail FROM events WHERE case_id=? ORDER BY ts DESC LIMIT ?", (case_id, int(limit)))
    else:
        rows = _rows("SELECT id,ts,kind,case_id,hospital_id,detail FROM events ORDER BY ts DESC LIMIT ?", (int(limit),))
    for r in rows:
        r["detail"] = _unj(r.get("detail"))
    return sorted(rows, key=lambda r: -float(r["ts"]))[: int(limit)]


def stage_runs(limit: int = 500) -> list:
    """The latency history in the shape the live latency panel uses: [{total_ms, stages: {stage: ms}}], oldest first."""
    runs = _rows("SELECT case_id,total_ms FROM case_runs WHERE halted=0 ORDER BY started_at DESC LIMIT ?", (int(limit),))
    if not runs:
        return []
    ids = [r["case_id"] for r in runs]
    marks = ",".join("?" * len(ids))
    spans: dict = {}
    for s in _rows(f"SELECT case_id,seq,stage,duration_ms FROM stage_timings WHERE case_id IN ({marks})", ids):
        spans.setdefault(s["case_id"], {})[s["stage"]] = float(s["duration_ms"] or 0)
    return [{"total_ms": float(r["total_ms"] or 0), "stages": spans.get(r["case_id"], {})} for r in reversed(runs)]


def status() -> dict:
    db = None
    try:
        db = _get_primary()
        runs = int(db.execute("SELECT COUNT(*) AS n FROM case_runs").fetchone()["n"])
        events = int(db.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"])
    except Exception:
        runs = events = None
    return {"enabled": enabled(), "backend": db.name if db else "unavailable", "stored_runs": runs, "stored_events": events,
            "queued": _q.unfinished_tasks, **_stats}
