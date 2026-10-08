"""LocationIQ client: address search/autocomplete and routed travel times.

Routed ETAs replace the straight-line estimate (ETA tier 4) with real road
durations (tier 2: live route, no traffic). Any failure -> caller keeps its tier-4
estimate (design principle P8: degrade, don't fail). The key is read from the
environment and never sent to the browser.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from typing import Optional

import requests

_session = requests.Session()
SEARCH = "https://api.locationiq.com/v1/autocomplete"
MATRIX = "https://us1.locationiq.com/v1/matrix/driving"
MAX_COORDS = 25  # provider limit per matrix call: 1 origin + 24 destinations
MIN_INTERVAL_S = 0.55  # free tier allows 2 requests/second
_last_call = 0.0
_ROUTE_CACHE: dict = {}  # (origin, destinations) -> (timestamp, minutes); routes barely change minute to minute
ROUTE_CACHE_TTL_S = 300


def enabled() -> bool:
    return bool(os.environ.get("LOCATIONIQ_KEY"))


def _throttle() -> None:
    global _last_call
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


TILE_HOSTS = ("a", "b", "c")
TILE_STYLES = {"streets", "dark", "light"}
_TILE_CACHE: "OrderedDict[tuple, bytes]" = OrderedDict()  # small LRU: a map view is ~20 tiles of 5-20 KB each
TILE_CACHE_MAX = 800
DIRECTIONS = "https://us1.locationiq.com/v1/directions/driving"
_ROUTE_GEOM_CACHE: dict = {}


def tile(style: str, z: int, x: int, y: int) -> Optional[bytes]:
    """Map tile via the provider, cached in memory. The API key stays on the server."""
    if not enabled() or style not in TILE_STYLES or not (0 <= z <= 19):
        return None
    key = (style, z, x, y)
    hit = _TILE_CACHE.get(key)
    if hit is not None:
        _TILE_CACHE.move_to_end(key)
        return hit
    host = TILE_HOSTS[(x + y) % len(TILE_HOSTS)]
    try:
        r = _session.get(f"https://{host}-tiles.locationiq.com/v3/{style}/r/{z}/{x}/{y}.png",
                         params={"key": os.environ["LOCATIONIQ_KEY"]}, timeout=8)
    except requests.RequestException:
        return None
    if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image"):
        return None
    _TILE_CACHE[key] = r.content
    if len(_TILE_CACHE) > TILE_CACHE_MAX:
        _TILE_CACHE.popitem(last=False)
    return r.content


def route_geometry(origin: tuple, dest: tuple) -> Optional[dict]:
    """Road route between two (lat, lng) points: {coords: [[lat, lng], ...], km, minutes}, or None."""
    if not enabled():
        return None
    ck = ((round(origin[0], 4), round(origin[1], 4)), (round(dest[0], 4), round(dest[1], 4)))
    hit = _ROUTE_GEOM_CACHE.get(ck)
    if hit and time.time() - hit[0] < ROUTE_CACHE_TTL_S:
        return hit[1]
    _throttle()
    try:
        r = _session.get(
            f"{DIRECTIONS}/{origin[1]},{origin[0]};{dest[1]},{dest[0]}",
            params={"key": os.environ["LOCATIONIQ_KEY"], "overview": "full", "geometries": "geojson", "steps": "false"},
            timeout=10,
        )
        j = r.json()
        if r.status_code != 200 or j.get("code") != "Ok":
            return None
        route = j["routes"][0]
        out = {
            "coords": [[lat, lng] for lng, lat in route["geometry"]["coordinates"]],  # GeoJSON is [lng, lat]
            "km": round(route["distance"] / 1000, 1), "minutes": round(route["duration"] / 60, 1),
        }
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return None
    if len(_ROUTE_GEOM_CACHE) > 200:
        _ROUTE_GEOM_CACHE.clear()
    _ROUTE_GEOM_CACHE[ck] = (time.time(), out)
    return out


_REVERSE_CACHE: dict = {}


def reverse(lat: float, lng: float) -> Optional[str]:
    """Short human-readable address for a point, e.g. 'Anna Salai, Teynampet, Chennai'. None if unavailable."""
    if not enabled():
        return None
    ck = (round(lat, 4), round(lng, 4))
    if ck in _REVERSE_CACHE:
        return _REVERSE_CACHE[ck]
    _throttle()
    try:
        r = _session.get("https://us1.locationiq.com/v1/reverse",
                         params={"key": os.environ["LOCATIONIQ_KEY"], "lat": lat, "lon": lng, "format": "json", "zoom": 17}, timeout=8)
        if r.status_code != 200:
            return None
        name = r.json().get("display_name", "")
    except (requests.RequestException, ValueError):
        return None
    parts = [p.strip() for p in name.split(",") if p.strip()]
    short = ", ".join(parts[:3]) if parts else None
    if len(_REVERSE_CACHE) > 300:
        _REVERSE_CACHE.clear()
    _REVERSE_CACHE[ck] = short
    return short


def search(query: str, limit: int = 5) -> list[dict]:
    """Address / place autocomplete, biased to India. Returns [{name, lat, lng}]."""
    if not enabled() or len(query.strip()) < 3:
        return []
    _throttle()
    r = _session.get(
        SEARCH,
        params={"key": os.environ["LOCATIONIQ_KEY"], "q": query, "limit": limit, "countrycodes": "in", "format": "json"},
        timeout=8,
    )
    if r.status_code != 200:
        return []
    return [{"name": x["display_name"], "lat": float(x["lat"]), "lng": float(x["lon"])} for x in r.json()]


def route_minutes(origin: tuple, destinations: list) -> Optional[list]:
    """Driving minutes from origin=(lat, lng) to each destination=(lat, lng), same order.
    Entries are None where no route was found. Returns None if the service is unavailable."""
    if not enabled() or not destinations:
        return None
    out: list = []
    key = os.environ["LOCATIONIQ_KEY"]
    ck = ((round(origin[0], 3), round(origin[1], 3)), tuple((round(a, 4), round(b, 4)) for a, b in destinations))
    hit = _ROUTE_CACHE.get(ck)
    if hit and time.time() - hit[0] < ROUTE_CACHE_TTL_S:
        return list(hit[1])
    try:
        for i in range(0, len(destinations), MAX_COORDS - 1):
            chunk = destinations[i : i + MAX_COORDS - 1]
            coords = ";".join(f"{lng},{lat}" for lat, lng in [origin] + chunk)
            _throttle()
            r = _session.get(f"{MATRIX}/{coords}", params={"key": key, "sources": "0", "annotations": "duration"}, timeout=10)
            if r.status_code != 200 or r.json().get("code") != "Ok":
                return None
            row = r.json()["durations"][0][1:]
            out.extend(None if d is None else round(d / 60.0, 1) for d in row)
    except (requests.RequestException, KeyError, ValueError):
        return None
    if len(_ROUTE_CACHE) > 200:
        _ROUTE_CACHE.clear()
    _ROUTE_CACHE[ck] = (time.time(), out)
    return out
