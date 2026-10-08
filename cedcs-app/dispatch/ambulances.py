"""Synthetic ambulance directory and nearest-unit selection.

Like the hospitals, these stations are fictional (Chennai area) and carry reserved-domain addresses (example.org), so
nothing can ever reach a real service by accident. The nearest available unit is chosen by straight-line distance, then
the closest few are re-ranked by real road time (LocationIQ) when available.
"""

from __future__ import annotations

import math
import random
from typing import Optional

CENTER = (13.0500, 80.2000)
AREAS = ["Tambaram", "Adyar", "Anna Nagar", "T Nagar", "Velachery", "Porur", "Perambur", "Guindy", "Egmore", "Sholinganallur",
         "Ambattur", "Mylapore", "Kilpauk", "Chromepet"]
AVG_SPEED_KMH = 35.0  # urban ambulance estimate used when road routing is unavailable


def _build() -> list:
    rng = random.Random(108)  # fixed seed: the directory is the same on every run
    out = []
    for i, area in enumerate(AREAS, start=1):
        out.append({
            "id": f"AMB-{i:03d}", "name": f"{area} Ambulance Station", "type": "ALS" if i % 3 else "BLS",
            "lat": CENTER[0] + rng.uniform(-0.13, 0.13), "lng": CENTER[1] + rng.uniform(-0.13, 0.13),
            "phone": f"+91 44 {rng.randint(2000, 2999)} {rng.randint(1000, 9999)}",
            "email": f"dispatch.amb{i:03d}@ambulance.example.org", "available": (i % 7) != 0,
        })
    return out


AMBULANCES = _build()


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(a))


def hospital_email(hospital_id: str) -> str:
    """Synthetic emergency-desk address for a hospital (reserved domain: cannot reach a real mailbox)."""
    return f"emergency.{hospital_id.lower()}@hospitals.example.org"


def nearest(lat: float, lng: float, road_minutes=None) -> Optional[dict]:
    """Nearest AVAILABLE ambulance: {**unit, distance_km, eta_min, method}. `road_minutes(origin, dests)` may refine the ETA."""
    units = [a for a in AMBULANCES if a["available"]]
    if not units:
        return None
    ranked = sorted(units, key=lambda a: haversine_km(lat, lng, a["lat"], a["lng"]))[:3]
    est = [(a, haversine_km(lat, lng, a["lat"], a["lng"])) for a in ranked]
    mins = None
    if road_minutes is not None:
        try:
            mins = road_minutes((lat, lng), [(a["lat"], a["lng"]) for a, _ in est])
        except Exception:
            mins = None
    best = None
    for i, (a, d) in enumerate(est):
        eta = mins[i] if mins and mins[i] is not None else round(d / AVG_SPEED_KMH * 60 * 1.3, 1)
        cand = {**a, "distance_km": round(d, 1), "eta_min": eta, "method": "road" if mins and mins[i] is not None else "estimate",
                "simulated": True}  # the ambulance directory is fictional whatever the hospital source
        if best is None or cand["eta_min"] < best["eta_min"]:
            best = cand
    return best
