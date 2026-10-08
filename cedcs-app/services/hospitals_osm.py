"""Real hospital discovery from OpenStreetMap, through LocationIQ's Nearby API.

What this gives: real hospital names, locations and addresses.
What it does NOT give: beds, departments, equipment, specialists, whether a hospital is open or diverting, phone numbers or
e-mail addresses. Those stay "not reported" until the hospital reports them itself (hospital console) — nothing is invented.

The Nearby API returns at most 10 places per call, so a search samples several points around the patient (centre plus a ring)
and merges the results; each sampled point is cached for a day because hospitals do not move.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import OrderedDict
from typing import Optional

import requests

from services import locationiq

NEARBY = "https://us1.locationiq.com/v1/nearby"
CACHE_TTL_S = 24 * 3600
CACHE_MAX = 400
_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()  # (lat3, lng3) -> (timestamp, [places])
_LOCK = threading.Lock()
MAX_RESULTS = 40


def enabled() -> bool:
    return locationiq.enabled()


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(a))


def offset(lat: float, lng: float, bearing_deg: float, km: float) -> tuple:
    b = math.radians(bearing_deg)
    return lat + km * math.cos(b) / 111.32, lng + km * math.sin(b) / (111.32 * math.cos(math.radians(lat)))


def sample_points(lat: float, lng: float, radius_km: float) -> list:
    """Where to ask: the centre, a ring at ~55% of the radius, and for wide searches a second ring at ~85%. Each call returns the
    10 nearest to ITS point, so together they cover the disc."""
    pts = [(lat, lng)]
    pts += [offset(lat, lng, b, radius_km * 0.55) for b in (0, 90, 180, 270)]
    if radius_km > 10:
        pts += [offset(lat, lng, b, radius_km * 0.85) for b in (45, 135, 225, 315)]
    return pts


def _query(lat: float, lng: float) -> Optional[list]:
    """One Nearby call (cached). None if the service could not be reached."""
    key = (round(lat, 3), round(lng, 3))
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < CACHE_TTL_S:
            _CACHE.move_to_end(key)
            return hit[1]
    import os

    locationiq._throttle()
    try:
        r = locationiq._session.get(NEARBY, params={"key": os.environ["LOCATIONIQ_KEY"], "lat": lat, "lon": lng,
                                                     "tag": "amenity:hospital", "radius": 30000, "format": "json"}, timeout=10)
        if r.status_code == 404:  # "no results" is an empty list for this provider
            places: list = []
        elif r.status_code != 200:
            return None
        else:
            places = r.json()
            if not isinstance(places, list):
                return None
    except (requests.RequestException, ValueError):
        return None
    with _LOCK:
        _CACHE[key] = (now, places)
        if len(_CACHE) > CACHE_MAX:
            _CACHE.popitem(last=False)
    return places


def _short_address(p: dict) -> str:
    parts = [x.strip() for x in (p.get("display_name") or "").split(",") if x.strip()]
    return ", ".join(parts[1:4]) if len(parts) > 1 else ""


def to_facility(p: dict, from_lat: float, from_lng: float) -> Optional[dict]:
    name = (p.get("name") or "").strip()
    if not name or not p.get("osm_id"):
        return None  # unnamed map features are not something a person can be sent to
    try:
        lat, lng = float(p["lat"]), float(p["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    hid = f"OSM-{p.get('osm_type', 'node')}-{p['osm_id']}"
    return {
        "hospital_id": hid, "name": name, "lat": lat, "lng": lng, "address": _short_address(p),
        "operating_status": "OPERATIONAL", "ipd_accepting": True, "trauma_level": "unknown",  # unknown: NOT verified
        "distance_km": round(haversine_km(from_lat, from_lng, lat, lng), 2), "source": "osm", "status_verified": False, "phone": None,
        "osm_url": f"https://www.openstreetmap.org/{p.get('osm_type', 'node')}/{p['osm_id']}",
    }


def nearby(lat: float, lng: float, radius_km: float = 8.0, limit: int = MAX_RESULTS) -> Optional[list]:
    """Real hospitals within radius_km of the point, nearest first. None if the provider was unreachable for EVERY sample point
    (an empty list means it answered and there are none)."""
    seen: dict = {}
    answered = False
    for plat, plng in sample_points(lat, lng, radius_km):
        places = _query(plat, plng)
        if places is None:
            continue
        answered = True
        for p in places:
            f = to_facility(p, lat, lng)
            if f and f["distance_km"] <= radius_km:
                seen.setdefault(f["hospital_id"], f)
    if not answered:
        return None
    # the same building is often mapped twice (a node and a way): drop near-duplicates with the same name
    out: list = []
    for f in sorted(seen.values(), key=lambda x: x["distance_km"]):
        key = re.sub(r"\W+", "", f["name"].lower())
        if any(re.sub(r"\W+", "", o["name"].lower()) == key and haversine_km(f["lat"], f["lng"], o["lat"], o["lng"]) < 0.15 for o in out):
            continue
        out.append(f)
    return out[:limit]


def clear_cache() -> int:
    with _LOCK:
        n = len(_CACHE)
        _CACHE.clear()
    return n
