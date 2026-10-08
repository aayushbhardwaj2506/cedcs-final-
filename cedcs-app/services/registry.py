"""Which hospital network the system works on: real (OpenStreetMap) or the fictional seeded one."""

from __future__ import annotations

from dispatch import store
from services import hospitals_osm


def source() -> str:
    """The effective source. "real" needs the map-data key; without it the system falls back to the synthetic network."""
    want = store.get_settings().get("hospital_source", "real")
    return "real" if want == "real" and hospitals_osm.enabled() else "synthetic"


def is_real_id(hospital_id: str) -> bool:
    return str(hospital_id).startswith("OSM-")
