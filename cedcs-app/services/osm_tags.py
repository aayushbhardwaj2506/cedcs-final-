"""Facilities and services a real hospital's OpenStreetMap entry actually declares (Overpass API).

This is crowd-mapped, sparse and unverified: it says what a mapper recorded, not what the hospital can do right now. It is
therefore turned into INFERRED-source resource records (trust 0.5, aged from the entry's check_date, or a month when it has
none), and a capability supported only by such a record still counts as provisional in eligibility. Hospital-reported data
(console) always outranks it because it is fresher and trusted fully.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

OVERPASS = "https://overpass-api.de/api/interpreter"
UA = "cedcs-research/0.1 (emergency routing research prototype)"
CACHE_TTL_S = 24 * 3600
DEFAULT_AGE_DAYS = 30  # no check_date on the entry: assume a month old

_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()  # (lat2, lng2, radius) -> (timestamp, {hospital_id: tags})
_TAGS: dict = {}  # hospital_id -> tags, filled by every successful fetch
_LOCK = threading.Lock()
_session = requests.Session()
_session.headers["User-Agent"] = UA

# keyword in healthcare:speciality -> department key(s)
_SPECIALTY_DEPT = [
    (r"cardi", ["CARDIOLOGY"]), (r"neuro", ["NEUROLOGY"]), (r"ortho|bone|joint", ["ORTHOPEDICS"]), (r"trauma|accident", ["TRAUMA"]),
    (r"paed|pedia|child", ["PEDIATRICS"]), (r"obstet|gyn|matern|women", ["OBSTETRICS"]), (r"onco|cancer", ["ONCOLOGY"]),
    (r"nephro|renal|kidney", ["NEPHROLOGY"]), (r"general|multi|internal|emergency", ["GENERAL_MEDICINE"]),
]
_LABEL = {"EMERGENCY_DEPARTMENT": "Emergency department", "CARDIOLOGY": "Cardiology", "NEUROLOGY": "Neurology", "ORTHOPEDICS": "Orthopaedics",
          "TRAUMA": "Trauma care", "PEDIATRICS": "Paediatrics", "OBSTETRICS": "Obstetrics", "ONCOLOGY": "Oncology", "NEPHROLOGY": "Nephrology",
          "GENERAL_MEDICINE": "General medicine"}


def enabled() -> bool:
    import os

    return bool(os.environ.get("CEDCS_OSM_TAGS", "1") not in ("0", "false")) and bool(os.environ.get("LOCATIONIQ_KEY"))


def _key(el: dict) -> str:
    return f"OSM-{el.get('type', 'node')}-{el.get('id')}"


def fetch(lat: float, lng: float, radius_km: float = 8.0) -> dict:
    """{hospital_id: tags} for hospitals around a point (one Overpass call, cached). {} if the service is unavailable."""
    if not enabled():
        return {}
    ck = (round(lat, 2), round(lng, 2), round(radius_km))
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(ck)
        if hit and now - hit[0] < CACHE_TTL_S:
            return hit[1]
    q = f'[out:json][timeout:25];nwr["amenity"="hospital"](around:{int(radius_km * 1000)},{lat},{lng});out tags center 500;'
    try:
        r = _session.post(OVERPASS, data={"data": q}, timeout=30)
        if r.status_code != 200:
            return {}
        elements = r.json().get("elements", [])
    except (requests.RequestException, ValueError):
        return {}
    tags = {_key(el): el.get("tags") or {} for el in elements if el.get("id")}
    with _LOCK:
        _CACHE[ck] = (now, tags)
        while len(_CACHE) > 100:
            _CACHE.popitem(last=False)
        _TAGS.update(tags)
    return tags


def tags_for(hospital_id: str, lat: Optional[float] = None, lng: Optional[float] = None) -> dict:
    with _LOCK:
        t = _TAGS.get(hospital_id)
    if t is None and lat is not None and lng is not None:
        fetch(lat, lng, 3)
        with _LOCK:
            t = _TAGS.get(hospital_id)
    return t or {}


def _first(tags: dict, *keys) -> Optional[str]:
    for k in keys:
        v = (tags.get(k) or "").strip()
        if v:
            return v
    return None


def departments(tags: dict) -> dict:
    """{resource_key: {"active": bool}} of what the entry declares. Absence of a tag says nothing, so nothing is emitted."""
    out: dict = {}
    em = (tags.get("emergency") or "").lower()
    if em in ("yes", "24/7"):
        out["department_EMERGENCY_DEPARTMENT"] = {"active": True}
    elif em == "no":
        out["department_EMERGENCY_DEPARTMENT"] = {"active": False}
    spec = (tags.get("healthcare:speciality") or tags.get("speciality") or "").lower()
    for token in re.split(r"[;,]", spec):
        for pat, depts in _SPECIALTY_DEPT:
            if re.search(pat, token):
                for d in depts:
                    out[f"department_{d}"] = {"active": True}
    if "emergency" in spec and out.get("department_EMERGENCY_DEPARTMENT") is None:
        out["department_EMERGENCY_DEPARTMENT"] = {"active": True}
    if re.search(r"dialysis", spec):
        out["equipment_DIALYSIS"] = {"operational": True}
    return out


def details(tags: dict) -> dict:
    """Contact and descriptive facts for the UI: phone, website, hours, operator type, bed count if mapped."""
    beds = _first(tags, "beds", "capacity:beds")
    return {"phone": _first(tags, "phone", "contact:phone"), "website": _first(tags, "website", "contact:website"),
            "hours": _first(tags, "opening_hours"), "operator_type": _first(tags, "operator:type"),
            "beds_total": int(beds) if beds and beds.isdigit() else None,
            "inferred": [_LABEL[k.split("_", 1)[1]] + ("" if v.get("active") else " (declared absent)")
                         for k, v in departments(tags).items() if k.startswith("department_") and k.split("_", 1)[1] in _LABEL]}


def _stamp(tags: dict) -> datetime:
    cd = _first(tags, "check_date", "survey:date")
    if cd:
        try:
            return datetime.fromisoformat(cd[:10]).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc) - timedelta(days=DEFAULT_AGE_DAYS)


def records(hospital_id: str, tags: dict) -> list:
    from schemas.resource import ResourceRecord

    at = _stamp(tags)
    return [ResourceRecord(resource_key=k, value=v, updated_at=at, source="INFERRED", reporter_id="osm-tags")
            for k, v in departments(tags).items()]


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
        _TAGS.clear()
