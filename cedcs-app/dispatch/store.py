"""App-owned persistent state: settings, emergency contacts, dispatches, messages and hospital acknowledgements.

SQLite by default (a file in ./data). With APP_DATABASE_URL set (a Postgres URL, e.g. Neon) the same tables live in a separate
Postgres schema instead, so the data survives a host that wipes its disk on restart (Render's free plan).

Kept separate from the hospital/resource database on purpose: hospital data belongs to the resource service, while
contacts, outbox and acks are this application's own records. The file lives in ./data (git-ignored).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "cedcs_app.db"

DEFAULT_SETTINGS: dict[str, Any] = {
    "bed_checker_enabled": False,  # global switch, admin only. OFF: nothing scans unless a user turns it on for a case.
    "allow_user_toggle": True,  # may an individual user switch the bed checker on for their own cases?
    "auto_dispatch": False,  # send emergency alerts automatically for CRITICAL/HIGH cases (off: a person must confirm)
    "auto_escalate": True,  # notify the next-ranked hospital when the first declines or does not answer in time
    "ack_timeout_s": 120,  # how long to wait for a hospital's confirmation before escalating
    "hospital_source": "real",  # "real": OpenStreetMap hospitals (capacity unknown until reported); "synthetic": the fictional seeded network
}
HOSPITAL_SOURCES = ("real", "synthetic")

_TABLES = ("contacts", "settings", "dispatches", "messages", "acks", "hospital_reports", "hospital_consoles")
_TABLE_RE = re.compile(r"\b(" + "|".join(_TABLES) + r")\b")


def pg_translate(sql: str, schema: str) -> str:
    """SQLite-flavoured SQL -> Postgres: %s placeholders, schema-qualified tables (works behind any connection pooler), no rowid,
    8-byte floats for timestamps."""
    out = sql.replace("%", "%%").replace("?", "%s")
    out = re.sub(r",\s*rowid\b", "", out)
    out = _TABLE_RE.sub(lambda m: f"{schema}.{m.group(1)}", out)
    return re.sub(r"\bREAL\b", "DOUBLE PRECISION", out)


class _PgConn:
    """The few sqlite3.Connection methods this module uses, over psycopg2 (autocommit; reconnects if the server dropped us)."""

    def __init__(self, url: str, schema: str):
        if not re.fullmatch(r"[a-z_][a-z0-9_]{0,40}", schema):
            raise ValueError("invalid schema name")
        self.url, self.schema, self._c = url, schema, None
        self._connect()
        for stmt in [s.strip() for s in _SCHEMA.split(";") if s.strip()]:
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

    def commit(self) -> None:  # autocommit
        pass

    def close(self) -> None:
        try:
            self._c.close()
        except Exception:
            pass


_lock = threading.RLock()
_conn = None
_path: Optional[Path] = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts(id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL, relation TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS dispatches(id TEXT PRIMARY KEY, case_id TEXT NOT NULL, created_at REAL, mode TEXT, status TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY, dispatch_id TEXT NOT NULL, kind TEXT NOT NULL, hospital_id TEXT, to_name TEXT,
  to_email TEXT, actual_to TEXT, subject TEXT, body_text TEXT, body_html TEXT, status TEXT, token TEXT UNIQUE, created_at REAL, sent_at REAL, error TEXT);
CREATE TABLE IF NOT EXISTS acks(id TEXT PRIMARY KEY, message_id TEXT NOT NULL, received_at REAL, response TEXT, note TEXT, eta_min REAL, beds TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS hospital_reports(hospital_id TEXT NOT NULL, resource_key TEXT NOT NULL, value TEXT NOT NULL, updated_at REAL NOT NULL,
  reporter TEXT, PRIMARY KEY(hospital_id, resource_key));
CREATE TABLE IF NOT EXISTS hospital_consoles(token TEXT PRIMARY KEY, hospital_id TEXT NOT NULL, name TEXT, email TEXT, phone TEXT, created_at REAL);
CREATE INDEX IF NOT EXISTS idx_msg_dispatch ON messages(dispatch_id);
CREATE INDEX IF NOT EXISTS idx_ack_msg ON acks(message_id);
"""


def use(path: Optional[Path | str] = None) -> None:
    """(Re)open the database at `path` (tests use a temp file). Default: env CEDCS_APP_DB or ./data/cedcs_app.db."""
    global _conn, _path
    with _lock:
        if _conn is not None:
            _conn.close()
        url = os.environ.get("APP_DATABASE_URL", "").strip()
        if url and not path and not os.environ.get("CEDCS_APP_DB"):
            _conn = _PgConn(url, os.environ.get("APP_DATABASE_SCHEMA", "cedcs_app"))
            return
        _path = Path(path or os.environ.get("CEDCS_APP_DB") or DEFAULT_PATH)
        _path.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(_path, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(_SCHEMA)
        _conn.commit()


def db():
    if _conn is None:
        use()
    return _conn


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------- settings
def get_settings() -> dict[str, Any]:
    with _lock:
        rows = db().execute("SELECT key, value FROM settings").fetchall()
    out = dict(DEFAULT_SETTINGS)
    for r in rows:
        if r["key"] in DEFAULT_SETTINGS:
            out[r["key"]] = json.loads(r["value"])
    return out


def put_settings(changes: dict[str, Any]) -> dict[str, Any]:
    """Only known keys with the right type are accepted; anything else is ignored."""
    changes = {k: v for k, v in changes.items() if k != "hospital_source" or v in HOSPITAL_SOURCES}
    with _lock:
        for k, v in changes.items():
            if k in DEFAULT_SETTINGS and isinstance(v, type(DEFAULT_SETTINGS[k])) and not (isinstance(v, bool) != isinstance(DEFAULT_SETTINGS[k], bool)):
                db().execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (k, json.dumps(v)))
        db().commit()
    return get_settings()


# ---------------------------------------------------------------- contacts
def add_contact(name: str, email: str, relation: str = "") -> dict:
    cid = _new_id()
    with _lock:
        db().execute("INSERT INTO contacts VALUES(?,?,?,?,?)", (cid, name.strip(), email.strip(), relation.strip(), time.time()))
        db().commit()
    return {"id": cid, "name": name.strip(), "email": email.strip(), "relation": relation.strip()}


def list_contacts() -> list[dict]:
    with _lock:
        rows = db().execute("SELECT id,name,email,relation FROM contacts ORDER BY created_at").fetchall()
    return [dict(r) for r in rows]


def delete_contact(cid: str) -> bool:
    with _lock:
        n = db().execute("DELETE FROM contacts WHERE id=?", (cid,)).rowcount
        db().commit()
    return n > 0


# ---------------------------------------------------------------- dispatches / messages / acks
def create_dispatch(case_id: str, mode: str, payload: dict) -> str:
    did = _new_id()
    with _lock:
        db().execute("INSERT INTO dispatches VALUES(?,?,?,?,?,?)", (did, case_id, time.time(), mode, "OPEN", json.dumps(payload, default=str)))
        db().commit()
    return did


def update_dispatch(did: str, *, status: Optional[str] = None, payload: Optional[dict] = None) -> None:
    with _lock:
        if status is not None:
            db().execute("UPDATE dispatches SET status=? WHERE id=?", (status, did))
        if payload is not None:
            db().execute("UPDATE dispatches SET payload=? WHERE id=?", (json.dumps(payload, default=str), did))
        db().commit()


def dispatch_for_case(case_id: str) -> Optional[str]:
    with _lock:
        r = db().execute("SELECT id FROM dispatches WHERE case_id=? ORDER BY created_at DESC LIMIT 1", (case_id,)).fetchone()
    return r["id"] if r else None


def add_message(did: str, kind: str, to_name: str, to_email: str, subject: str, body_text: str, body_html: str,
                hospital_id: Optional[str] = None) -> dict:
    mid, token = _new_id(), uuid.uuid4().hex
    with _lock:
        db().execute("INSERT INTO messages(id,dispatch_id,kind,hospital_id,to_name,to_email,actual_to,subject,body_text,body_html,status,token,created_at)"
                     " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (mid, did, kind, hospital_id, to_name, to_email, to_email, subject, body_text, body_html, "QUEUED", token, time.time()))
        db().commit()
    return {"id": mid, "token": token}


def update_message(mid: str, **fields: Any) -> None:
    allowed = {"status", "actual_to", "sent_at", "error", "subject", "body_text", "body_html"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return
    with _lock:
        db().execute(f"UPDATE messages SET {', '.join(k + '=?' for k in sets)} WHERE id=?", (*sets.values(), mid))
        db().commit()


def message_by_token(token: str) -> Optional[dict]:
    with _lock:
        r = db().execute("SELECT * FROM messages WHERE token=?", (token,)).fetchone()
    return dict(r) if r else None


def add_ack(message_id: str, response: str, note: str = "", eta_min: Optional[float] = None, beds: Optional[dict] = None,
            source: str = "LINK") -> dict:
    aid = _new_id()
    with _lock:
        db().execute("INSERT INTO acks VALUES(?,?,?,?,?,?,?,?)", (aid, message_id, time.time(), response, note, eta_min, json.dumps(beds or {}), source))
        db().commit()
    return {"id": aid}


def get_dispatch(did: str) -> Optional[dict]:
    with _lock:
        d = db().execute("SELECT * FROM dispatches WHERE id=?", (did,)).fetchone()
        if d is None:
            return None
        msgs = [dict(r) for r in db().execute("SELECT * FROM messages WHERE dispatch_id=? ORDER BY created_at, rowid", (did,)).fetchall()]
        acks = [dict(r) for r in db().execute(
            "SELECT a.* FROM acks a JOIN messages m ON m.id=a.message_id WHERE m.dispatch_id=? ORDER BY a.received_at", (did,)).fetchall()]
    out = dict(d)
    out["payload"] = json.loads(out["payload"] or "{}")
    for m in msgs:
        m.pop("body_html", None)  # large; fetched separately when needed
        m["acks"] = [dict(a, beds=json.loads(a["beds"] or "{}")) for a in acks if a["message_id"] == m["id"]]
        m["ack"] = m["acks"][-1] if m["acks"] else None
    out["messages"] = msgs
    return out


def list_outbox(limit: int = 50) -> list[dict]:
    with _lock:
        rows = db().execute("SELECT id,dispatch_id,kind,to_name,to_email,actual_to,subject,status,created_at,sent_at,error FROM messages "
                            "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- data reported by real hospitals (their own console)
REPORT_KEYS_BEDS = ("icu_beds", "emergency_beds", "general_beds", "hdu_beds", "pediatric_beds", "ventilators")


def upsert_report(hospital_id: str, resource_key: str, value: dict, reporter: str = "") -> None:
    with _lock:
        db().execute("INSERT INTO hospital_reports VALUES(?,?,?,?,?) ON CONFLICT(hospital_id,resource_key) DO UPDATE SET "
                     "value=excluded.value, updated_at=excluded.updated_at, reporter=excluded.reporter",
                     (hospital_id, resource_key, json.dumps(value), time.time(), reporter))
        db().commit()
    from services import history

    history.event("hospital_report", hospital_id=hospital_id, resource_key=resource_key, value=value, reporter=reporter)


def reports_for(hospital_ids: list) -> dict:
    """{hospital_id: [{resource_key, value, updated_at(epoch), reporter}]} for hospitals that have reported anything."""
    if not hospital_ids:
        return {}
    marks = ",".join("?" * len(hospital_ids))
    with _lock:
        rows = db().execute(f"SELECT hospital_id,resource_key,value,updated_at,reporter FROM hospital_reports WHERE hospital_id IN ({marks})",
                            list(hospital_ids)).fetchall()
    out: dict = {}
    for r in rows:
        out.setdefault(r["hospital_id"], []).append({"resource_key": r["resource_key"], "value": json.loads(r["value"]),
                                                     "updated_at": r["updated_at"], "reporter": r["reporter"]})
    return out


def reserve_reported_bed(hospital_id: str, bed_key: str, reporter: str = "") -> dict:
    """available - 1 on a hospital's own reported count (never below zero; unreported is refused)."""
    with _lock:
        row = db().execute("SELECT value FROM hospital_reports WHERE hospital_id=? AND resource_key=?", (hospital_id, bed_key)).fetchone()
        if row is None:
            return {"ok": False, "reason": "this hospital has not reported its beds", "before": None, "after": None}
        value = json.loads(row["value"])
        before = value.get("available")
        if before is None:
            return {"ok": False, "reason": "bed count unknown", "before": None, "after": None}
        if before <= 0:
            return {"ok": False, "reason": "no free bed", "before": before, "after": before}
        value["available"] = before - 1
        db().execute("UPDATE hospital_reports SET value=?, updated_at=?, reporter=? WHERE hospital_id=? AND resource_key=?",
                     (json.dumps(value), time.time(), reporter, hospital_id, bed_key))
        db().commit()
    from services import history

    history.event("bed_reserved", hospital_id=hospital_id, resource_key=bed_key, before=before, after=before - 1, reporter=reporter)
    return {"ok": True, "before": before, "after": before - 1}


def create_console(hospital_id: str, name: str = "", email: str = "", phone: str = "") -> dict:
    """A private link for one hospital's staff to report beds/facilities. The token in the URL is the credential."""
    token = uuid.uuid4().hex + uuid.uuid4().hex[:8]
    with _lock:
        db().execute("INSERT INTO hospital_consoles VALUES(?,?,?,?,?,?)", (token, hospital_id, name, email.strip(), phone.strip(), time.time()))
        db().commit()
    return {"token": token, "hospital_id": hospital_id}


def console_by_token(token: str) -> Optional[dict]:
    with _lock:
        r = db().execute("SELECT * FROM hospital_consoles WHERE token=?", (token,)).fetchone()
    return dict(r) if r else None


def contact_for(hospital_id: str) -> Optional[dict]:
    """Latest registered contact (email/phone) for a hospital, if an administrator set one up."""
    with _lock:
        r = db().execute("SELECT name,email,phone FROM hospital_consoles WHERE hospital_id=? ORDER BY created_at DESC LIMIT 1", (hospital_id,)).fetchone()
    return dict(r) if r else None


def list_consoles() -> list:
    with _lock:
        rows = db().execute("SELECT hospital_id,name,email,phone,created_at, substr(token,1,6) AS token_hint FROM hospital_consoles ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def hospital_inbox(hospital_id: str, limit: int = 30) -> list:
    """Incoming-patient alerts addressed to one hospital, newest first: {message, dispatch} pairs (dispatch has payload + acks)."""
    with _lock:
        rows = db().execute("SELECT id,dispatch_id FROM messages WHERE kind='HOSPITAL' AND hospital_id=? ORDER BY created_at DESC LIMIT ?",
                            (hospital_id, limit)).fetchall()
    out = []
    for r in rows:
        d = get_dispatch(r["dispatch_id"])
        m = next((x for x in (d or {}).get("messages", []) if x["id"] == r["id"]), None)
        if d and m:
            out.append({"message": m, "dispatch": d})
    return out


def console_for_hospital(hospital_id: str) -> Optional[dict]:
    with _lock:
        r = db().execute("SELECT * FROM hospital_consoles WHERE hospital_id=? ORDER BY created_at DESC LIMIT 1", (hospital_id,)).fetchone()
    return dict(r) if r else None


def hospital_inbox(hospital_id: str, limit: int = 30) -> list:
    """Incoming-patient alerts addressed to one hospital, newest first: {message, dispatch} pairs (dispatch has payload + acks)."""
    with _lock:
        rows = db().execute("SELECT id,dispatch_id FROM messages WHERE kind='HOSPITAL' AND hospital_id=? ORDER BY created_at DESC LIMIT ?",
                            (hospital_id, limit)).fetchall()
    out = []
    for r in rows:
        d = get_dispatch(r["dispatch_id"])
        m = next((x for x in (d or {}).get("messages", []) if x["id"] == r["id"]), None)
        if d and m:
            out.append({"message": m, "dispatch": d})
    return out


def console_for_hospital(hospital_id: str) -> Optional[dict]:
    with _lock:
        r = db().execute("SELECT * FROM hospital_consoles WHERE hospital_id=? ORDER BY created_at DESC LIMIT 1", (hospital_id,)).fetchone()
    return dict(r) if r else None
