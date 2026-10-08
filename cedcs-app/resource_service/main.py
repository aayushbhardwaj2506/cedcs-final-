"""
CEDCS Resource Retrieval Service (M7).

Returns a ResourceSnapshot[] for a list of hospital_ids, each record carrying
value + updated_at + source + reporter_id (never a bare value — P3). It does NOT
compute freshness, eligibility, or ranking; that stays in the deterministic core,
strictly downstream of this service.

Also serves the facility registry: nearby search (radius) and lookup by id.

Read path: an in-memory copy of hospitals + resource records, refreshed from the
database in the background every CACHE_REFRESH_S seconds (default 30). Requests
are answered from memory in ~1 ms, so a slow or cold remote database (e.g. Neon)
never sits on the emergency request path. Each record still carries its own
updated_at, so the freshness model correctly discounts anything older. Until the
first load finishes (or with CACHE_REFRESH_S=0) requests go straight to Postgres.

Run locally:
    export DATABASE_URL="postgresql://user:pass@host/db?sslmode=require"
    uvicorn resource_service.main:app --port 8000
"""

import math
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL")
CACHE_REFRESH_S = float(os.environ.get("CACHE_REFRESH_S", "30"))


# ---------------------------------------------------------------- schemas

class ResourceRecordOut(BaseModel):
    resource_key: str
    value: Any
    updated_at: datetime
    source: Literal["HOSPITAL_CONSOLE", "FHIR", "GOVT", "INFERRED", "SEED"]
    reporter_id: Optional[str] = None


class ResourceSnapshotOut(BaseModel):
    hospital_id: str
    records: list[ResourceRecordOut]


class SnapshotRequest(BaseModel):
    hospital_ids: list[str]


class BatchRequest(BaseModel):
    hospital_ids: list[str]


class CandidateOut(BaseModel):
    hospital_id: str
    name: str
    lat: float
    lng: float
    address: Optional[str] = None
    operating_status: str
    ipd_accepting: bool = True
    trauma_level: str
    distance_km: Optional[float] = None
    place_id: Optional[str] = None
    seed_eta_min: Optional[float] = None


# ---------------------------------------------------------------- database

_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


class _PooledConn:
    """close() returns the connection to the pool, so `finally: conn.close()` call sites keep working."""

    def __init__(self, pool, conn):
        self._pool, self._conn = pool, conn

    def cursor(self, *a, **kw):
        return self._conn.cursor(*a, **kw)

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.rollback()  # end the transaction (a no-op after commit) before reuse
        self._pool.putconn(self._conn)


def get_conn():
    global _pool
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL is not configured on this service")
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(1, 10, DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return _PooledConn(_pool, _pool.getconn())


def _haversine_km(lat1, lng1, lat2, lng2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(a))


def _candidate(h: dict, distance_km: Optional[float] = None) -> CandidateOut:
    return CandidateOut(
        hospital_id=h["hospital_id"], name=h["name"], lat=h["lat"], lng=h["lng"], address=h["address"],
        operating_status=h["operating_status"], ipd_accepting=h["ipd_accepting"], trauma_level=h["trauma_level"],
        distance_km=None if distance_km is None else round(distance_km, 2),
    )


# ---------------------------------------------------------------- cache

class _Cache:
    def __init__(self):
        self.hospitals: dict = {}
        self.records: dict = {}
        self.loaded_at: Optional[float] = None
        self.load_seconds: Optional[float] = None
        self.last_error: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self.loaded_at is not None

    def refresh(self) -> None:
        t0 = time.time()
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT hospital_id, name, lat, lng, address, operating_status, ipd_accepting, trauma_level FROM hospitals"
                )
                hospitals = {r["hospital_id"]: dict(r) for r in cur.fetchall()}
                cur.execute(
                    "SELECT hospital_id, resource_key, value, updated_at, source, reporter_id "
                    "FROM resource_records ORDER BY hospital_id, resource_key"
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        records: dict = {hid: [] for hid in hospitals}
        for r in rows:
            records.setdefault(r["hospital_id"], []).append(
                ResourceRecordOut(resource_key=r["resource_key"], value=r["value"], updated_at=r["updated_at"],
                                  source=r["source"], reporter_id=r["reporter_id"])
            )
        self.hospitals, self.records = hospitals, records  # atomic swap of references
        self.loaded_at, self.load_seconds, self.last_error = time.time(), round(time.time() - t0, 2), None

    def loop(self) -> None:
        while True:
            try:
                self.refresh()
            except Exception as exc:  # keep serving the previous copy; retry next tick
                self.last_error = str(exc)
                print(f"[resource_service] cache refresh failed: {exc}")
            time.sleep(max(CACHE_REFRESH_S, 5))


cache = _Cache()


@asynccontextmanager
async def _lifespan(_app):
    if DATABASE_URL and CACHE_REFRESH_S > 0:
        threading.Thread(target=cache.loop, daemon=True).start()
    yield


app = FastAPI(title="CEDCS Resource Retrieval Service", version="0.3.0", lifespan=_lifespan)


# ---------------------------------------------------------------- endpoints

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/cache")
def cache_status():
    return {
        "enabled": CACHE_REFRESH_S > 0, "ready": cache.ready,
        "age_s": None if not cache.ready else round(time.time() - cache.loaded_at, 1),
        "last_load_s": cache.load_seconds, "hospitals": len(cache.hospitals),
        "records": sum(len(v) for v in cache.records.values()), "last_error": cache.last_error,
    }


@app.post("/snapshots", response_model=list[ResourceSnapshotOut])
def resource_snapshots(req: SnapshotRequest):
    if not req.hospital_ids:
        raise HTTPException(status_code=400, detail="hospital_ids must not be empty")

    if cache.ready:
        return [ResourceSnapshotOut(hospital_id=hid, records=cache.records.get(hid, [])) for hid in req.hospital_ids]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT hospital_id, resource_key, value, updated_at, source, reporter_id
                FROM resource_records
                WHERE hospital_id = ANY(%s)
                ORDER BY hospital_id, resource_key
                """,
                (req.hospital_ids,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    by_hospital: dict[str, list[ResourceRecordOut]] = {hid: [] for hid in req.hospital_ids}
    for row in rows:
        by_hospital.setdefault(row["hospital_id"], []).append(
            ResourceRecordOut(
                resource_key=row["resource_key"], value=row["value"], updated_at=row["updated_at"],
                source=row["source"], reporter_id=row["reporter_id"],
            )
        )
    return [ResourceSnapshotOut(hospital_id=hid, records=recs) for hid, recs in by_hospital.items()]


@app.get("/facilities/nearby", response_model=list[CandidateOut])
def facilities_nearby(
    lat: float = Query(...),
    lng: float = Query(...),
    radius_km: float = Query(8.0, description="8 km urban / 25 km rural per the design doc"),
    limit: int = Query(15, le=50),
):
    """Radius search (Matrix stage 9, tier 2 fallback for the Facility Discovery agent)."""
    if cache.ready:
        found = []
        for h in cache.hospitals.values():
            if h["operating_status"] == "CLOSED":
                continue
            d = _haversine_km(lat, lng, h["lat"], h["lng"])
            if d <= radius_km:
                found.append((d, h))
        found.sort(key=lambda t: t[0])
        return [_candidate(h, d) for d, h in found[:limit]]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT hospital_id, name, lat, lng, address, operating_status, ipd_accepting, trauma_level,
                       ST_Distance(geog, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography) / 1000.0 AS distance_km
                FROM hospitals
                WHERE operating_status != 'CLOSED'
                  AND ST_DWithin(geog, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s * 1000.0)
                ORDER BY distance_km ASC
                LIMIT %s
                """,
                (lng, lat, lng, lat, radius_km, limit),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [_candidate(r, r["distance_km"]) for r in rows]


@app.post("/facilities/batch", response_model=list[CandidateOut])
def facilities_batch(req: BatchRequest):
    """Lookup by hospital_id; same shape as /facilities/nearby (no distance)."""
    if not req.hospital_ids:
        raise HTTPException(status_code=400, detail="hospital_ids must not be empty")

    if cache.ready:
        return [_candidate(cache.hospitals[h]) for h in sorted(req.hospital_ids) if h in cache.hospitals]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT hospital_id, name, lat, lng, address, operating_status, ipd_accepting, trauma_level
                FROM hospitals WHERE hospital_id = ANY(%s) ORDER BY hospital_id
                """,
                (req.hospital_ids,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [_candidate(r) for r in rows]


# ---------------------------------------------------------------- live bed scan / hospital bed reports / reservations
BED_KEYS = ("icu_beds", "emergency_beds", "general_beds", "hdu_beds", "pediatric_beds", "ventilators")
WRITE_KEY = os.environ.get("RESOURCE_WRITE_KEY", "")


def _require_write_key(key: Optional[str]) -> None:
    if WRITE_KEY and key != WRITE_KEY:
        raise HTTPException(status_code=403, detail="write key required")


class BedScanRequest(BaseModel):
    hospital_ids: list[str]


class BedReport(BaseModel):
    hospital_id: str
    bed_key: str
    available: int
    total: Optional[int] = None
    reporter_id: Optional[str] = None


class BedReserve(BaseModel):
    hospital_id: str
    bed_key: str
    token: str = ""


def _refresh_cache_for(hospital_ids, rows) -> None:
    """Fold freshly read rows into the in-memory copy so normal reads see them immediately, not after the next refresh."""
    if not cache.ready:
        return
    for hid in hospital_ids:
        keep = [r for r in cache.records.get(hid, []) if r.resource_key not in BED_KEYS]
        cache.records[hid] = keep + [
            ResourceRecordOut(resource_key=r["resource_key"], value=r["value"], updated_at=r["updated_at"], source=r["source"], reporter_id=r["reporter_id"])
            for r in rows if r["hospital_id"] == hid
        ]


@app.post("/beds/scan")
def beds_scan(req: BedScanRequest):
    """LIVE read of bed availability straight from the database, bypassing the in-memory copy. Used by the Bed Checker."""
    if not req.hospital_ids:
        raise HTTPException(status_code=400, detail="hospital_ids must not be empty")
    t0 = time.time()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (hospital_id, resource_key) hospital_id, resource_key, value, updated_at, source, reporter_id
                FROM resource_records
                WHERE hospital_id = ANY(%s) AND resource_key = ANY(%s)
                ORDER BY hospital_id, resource_key, updated_at DESC
                """,
                (req.hospital_ids, list(BED_KEYS)),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    _refresh_cache_for(req.hospital_ids, rows)
    by: dict = {hid: [] for hid in req.hospital_ids}
    for r in rows:
        by[r["hospital_id"]].append(ResourceRecordOut(resource_key=r["resource_key"], value=r["value"], updated_at=r["updated_at"],
                                                      source=r["source"], reporter_id=r["reporter_id"]))
    return {"scanned_at": datetime.utcnow().isoformat() + "Z", "took_ms": round((time.time() - t0) * 1000),
            "snapshots": [ResourceSnapshotOut(hospital_id=h, records=recs).model_dump(mode="json") for h, recs in by.items()]}


def _write_bed(cur, hospital_id: str, bed_key: str, available: int, total: Optional[int], reporter: str) -> dict:
    cur.execute("SELECT id, value FROM resource_records WHERE hospital_id=%s AND resource_key=%s ORDER BY updated_at DESC LIMIT 1 FOR UPDATE",
                (hospital_id, bed_key))
    row = cur.fetchone()
    value = dict(row["value"]) if row and isinstance(row["value"], dict) else {}
    value["available"] = max(0, int(available))
    if total is not None:
        value["total"] = int(total)
    value.setdefault("total", value["available"])
    if row:
        cur.execute("UPDATE resource_records SET value=%s, updated_at=now(), source='HOSPITAL_CONSOLE', reporter_id=%s WHERE id=%s",
                    (psycopg2.extras.Json(value), reporter, row["id"]))
    else:
        cur.execute("INSERT INTO resource_records(hospital_id,resource_key,value,updated_at,source,reporter_id) VALUES(%s,%s,%s,now(),'HOSPITAL_CONSOLE',%s)",
                    (hospital_id, bed_key, psycopg2.extras.Json(value), reporter))
    return value


@app.post("/beds/report")
def beds_report(req: BedReport, x_write_key: Optional[str] = Header(None)):
    """A hospital (or admin) reports its current free beds: stored as a fresh HOSPITAL_CONSOLE fact, so it outranks older data."""
    _require_write_key(x_write_key)
    if req.bed_key not in BED_KEYS:
        raise HTTPException(status_code=400, detail=f"bed_key must be one of {BED_KEYS}")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            value = _write_bed(cur, req.hospital_id, req.bed_key, req.available, req.total, req.reporter_id or "hospital-report")
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "hospital_id": req.hospital_id, "bed_key": req.bed_key, "value": value}


@app.post("/beds/reserve")
def beds_reserve(req: BedReserve, x_write_key: Optional[str] = Header(None)):
    """Reserve one bed after a hospital ACCEPTS an incoming patient: available - 1, never below zero, unknown is refused."""
    _require_write_key(x_write_key)
    if req.bed_key not in BED_KEYS:
        raise HTTPException(status_code=400, detail=f"bed_key must be one of {BED_KEYS}")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM resource_records WHERE hospital_id=%s AND resource_key=%s ORDER BY updated_at DESC LIMIT 1 FOR UPDATE",
                        (req.hospital_id, req.bed_key))
            row = cur.fetchone()
            before = row["value"].get("available") if row and isinstance(row["value"], dict) else None
            if before is None:
                conn.commit()
                return {"ok": False, "reason": "bed count unknown", "before": None, "after": None}
            if before <= 0:
                conn.commit()
                return {"ok": False, "reason": "no free bed", "before": before, "after": before}
            _write_bed(cur, req.hospital_id, req.bed_key, before - 1, None, f"cedcs-reserve:{req.token[:8]}")
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "before": before, "after": before - 1}
