
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List, Optional, Any
import requests
import json
import os


class FacilityRegistryLookupInput(BaseModel):
    """Input schema for FacilityRegistryLookupTool."""

    hospital_ids: Optional[List[str]] = Field(
        default=None,
        description="List of hospital IDs to look up directly. If provided, performs a batch lookup.",
    )
    lat: Optional[float] = Field(
        default=None,
        description="Centre latitude for radius-based search.",
    )
    lng: Optional[float] = Field(
        default=None,
        description="Centre longitude for radius-based search.",
    )
    radius_km: Optional[float] = Field(
        default=25.0,
        description="Search radius in kilometres (default: 25.0). Used only when lat/lng are provided.",
    )


class FacilityRegistryLookupTool(BaseTool):
    """Tool for querying the Facility Registry REST API to retrieve CEDCS hospital records.

    Supports two lookup modes:
      1. Batch lookup by a list of hospital_ids (POST /facilities/batch).
      2. Radius search by geographic coordinates (GET /facilities/nearby).

    Returns facility records containing hospital_id, place_id, name, lat, lng,
    address, capabilities, and seed ETA data.
    """

    name: str = "FacilityRegistryLookupTool"
    description: str = (
        "Looks up CEDCS hospital records from the Facility Registry (PostgreSQL+PostGIS). "
        "Supports two modes: (1) lookup by hospital_ids list, (2) radius search by lat/lng/radius_km. "
        "Returns facility records including hospital_id, place_id, name, lat, lng, address, "
        "capabilities, and seed ETA data."
    )
    args_schema: Type[BaseModel] = FacilityRegistryLookupInput

    def _run(
        self,
        hospital_ids: Optional[List[str]] = None,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        radius_km: Optional[float] = 25.0,
    ) -> Any:
        """
        Execute the Facility Registry lookup.

        Args:
            hospital_ids: Optional list of hospital IDs for batch lookup.
            lat: Optional latitude for radius search.
            lng: Optional longitude for radius search.
            radius_km: Search radius in km (default 25.0).

        Returns:
            List of facility dicts on success, or an error dict on failure.
        """
        # --- Retrieve base URL from environment ---
        base_url = os.environ.get("RESOURCE_SERVICE_URL", "").rstrip("/")
        if not base_url:
            return {
                "error": "Environment variable RESOURCE_SERVICE_URL is not set or empty.",
                "results": [],
            }

        # --- Validate input: at least one mode must be provided ---
        has_hospital_ids = hospital_ids is not None and len(hospital_ids) > 0
        has_coordinates = lat is not None and lng is not None

        if not has_hospital_ids and not has_coordinates:
            return {"error": "Must provide either hospital_ids or lat+lng+radius_km"}

        # --- Mode 1: Batch lookup by hospital_ids ---
        if has_hospital_ids:
            return self._lookup_by_ids(base_url, hospital_ids)

        # --- Mode 2: Radius search by lat/lng ---
        return self._lookup_by_radius(base_url, lat, lng, radius_km)

    def _lookup_by_ids(self, base_url: str, hospital_ids: List[str]) -> Any:
        """POST /facilities/batch with a list of hospital IDs."""
        endpoint = f"{base_url}/facilities/batch"
        payload = {"hospital_ids": hospital_ids}

        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            return self._parse_response(response)

        except requests.exceptions.HTTPError as http_err:
            return {
                "error": f"HTTP error during batch lookup: {http_err} — Response: {http_err.response.text if http_err.response else 'N/A'}",
                "results": [],
            }
        except requests.exceptions.ConnectionError as conn_err:
            return {
                "error": f"Connection error during batch lookup: {conn_err}",
                "results": [],
            }
        except requests.exceptions.Timeout:
            return {
                "error": "Request timed out during batch lookup.",
                "results": [],
            }
        except Exception as exc:
            return {
                "error": f"Unexpected error during batch lookup: {exc}",
                "results": [],
            }

    def _lookup_by_radius(
        self, base_url: str, lat: float, lng: float, radius_km: float
    ) -> Any:
        """GET /facilities/nearby with lat, lng, and radius_km query parameters."""
        endpoint = f"{base_url}/facilities/nearby"
        params = {
            "lat": lat,
            "lng": lng,
            "radius_km": radius_km if radius_km is not None else 25.0,
        }

        try:
            response = requests.get(
                endpoint,
                params=params,
                headers={"Accept": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            return self._parse_response(response)

        except requests.exceptions.HTTPError as http_err:
            return {
                "error": f"HTTP error during radius search: {http_err} — Response: {http_err.response.text if http_err.response else 'N/A'}",
                "results": [],
            }
        except requests.exceptions.ConnectionError as conn_err:
            return {
                "error": f"Connection error during radius search: {conn_err}",
                "results": [],
            }
        except requests.exceptions.Timeout:
            return {
                "error": "Request timed out during radius search.",
                "results": [],
            }
        except Exception as exc:
            return {
                "error": f"Unexpected error during radius search: {exc}",
                "results": [],
            }

    def _parse_response(self, response: requests.Response) -> Any:
        """
        Parse and normalize the API response into the expected facility record format.

        Expected output per record:
            {
                "hospital_id": str,
                "place_id": str | None,
                "name": str,
                "lat": float,
                "lng": float,
                "address": str,
                "capabilities": list[str],
                "seed_eta_min": float | None,
            }
        """
        try:
            raw_data = response.json()
        except ValueError:
            return {
                "error": f"Failed to parse JSON response. Raw content: {response.text[:500]}",
                "results": [],
            }

        # Handle both a top-level list and a dict wrapping a list (e.g. {"results": [...]})
        if isinstance(raw_data, list):
            records = raw_data
        elif isinstance(raw_data, dict):
            # Try common wrapper keys
            records = raw_data.get("results") or raw_data.get("data") or raw_data.get("facilities") or []
            if not isinstance(records, list):
                # If the dict itself is a single record, wrap it
                records = [raw_data]
        else:
            return {
                "error": f"Unexpected response format from Facility Registry: {type(raw_data).__name__}",
                "results": [],
            }

        normalized = []
        for item in records:
            if not isinstance(item, dict):
                continue  # Skip malformed entries

            normalized.append({
                "hospital_id": str(item.get("hospital_id", "")),
                "place_id": item.get("place_id"),  # str or None
                "name": str(item.get("name", "")),
                "lat": float(item.get("lat", item.get("latitude", 0.0))),
                "lng": float(item.get("lng", item.get("longitude", 0.0))),
                "address": str(item.get("address", "")),
                "capabilities": list(item.get("capabilities", [])),
                "operating_status": item.get("operating_status", "OPERATIONAL"),
                "ipd_accepting": bool(item.get("ipd_accepting", True)),
                "trauma_level": item.get("trauma_level"),
                "distance_km": item.get("distance_km"),
                "seed_eta_min": (
                    float(item["seed_eta_min"])
                    if item.get("seed_eta_min") is not None
                    else None
                ),
            })

        return normalized
