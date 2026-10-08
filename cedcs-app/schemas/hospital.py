"""
Facility-discovery schemas — CandidateHospital and its ETA-tier model.

Matches config/tasks.yaml (`facility_discovery_task.expected_output`).
eta_source_tier follows the 5-tier ETA confidence ladder from the design
doc: 1=live traffic, 2=live-no-traffic, 3=recent-cached, 4=haversine-estimate,
5=seed_eta_min fallback. Only tiers 4-5 are ever assigned by the
Facility Discovery Coordinator agent itself; tiers 1-3 are filled in by the
orchestrator's own DistanceMatrixTool call, never by an agent guessing.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

ETA_TIER_CONFIDENCE = {
    1: 0.95,  # live traffic-aware
    2: 0.80,  # live, no traffic
    3: 0.55,  # recent cached matrix result
    4: 0.40,  # haversine estimate
    5: 0.25,  # seed_eta_min fallback
}


class CandidateHospital(BaseModel):
    hospital_id: str
    name: str = ""
    operating_status: Literal["OPERATIONAL", "DIVERTING", "CLOSED"] = "OPERATIONAL"
    ipd_accepting: bool = True
    lat: Optional[float] = None
    lng: Optional[float] = None
    source: str = "registry"  # "osm" = a real hospital from OpenStreetMap; "registry" = the seeded network
    status_verified: bool = True  # False: open/accepting status is ASSUMED (real hospitals only report it themselves)
    phone: Optional[str] = None
    address: Optional[str] = None
    osm_url: Optional[str] = None
    hours: Optional[str] = None
    website: Optional[str] = None
    inferred: list = Field(default_factory=list)  # facilities the map entry declares (unverified)
    place_id: Optional[str] = None
    distance_km: Optional[float] = None
    eta_min: Optional[float] = None
    eta_source_tier: int = Field(ge=1, le=5)
    eta_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _confidence_matches_tier_band(self) -> "CandidateHospital":
        # Not a strict equality check (the orchestrator's own DistanceMatrixTool
        # pass may compute a slightly different value within the tier's band) —
        # just guards against an agent inventing an eta_confidence that
        # contradicts the tier it claims, e.g. tier 5 (seed) claiming 0.9 confidence.
        expected = ETA_TIER_CONFIDENCE.get(self.eta_source_tier)
        if expected is not None and self.eta_confidence > expected + 0.15:
            raise ValueError(
                f"eta_confidence={self.eta_confidence} is implausibly high for "
                f"eta_source_tier={self.eta_source_tier} (expected around {expected})"
            )
        return self


class FacilityCapability(BaseModel):
    key: str
    active: bool = True


class Facility(BaseModel):
    """A registry record for one hospital, as returned by
    FacilityRegistryLookupTool / the resource-stub's /facilities/batch."""

    hospital_id: str
    place_id: Optional[str] = None
    name: str
    lat: float
    lng: float
    address: Optional[str] = None
    capabilities: list[str] = Field(default_factory=list)
    seed_eta_min: Optional[float] = None
