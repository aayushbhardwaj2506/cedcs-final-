
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Any
import os


class PlacesNearbyToolInput(BaseModel):
    """Input schema for PlacesNearbyTool."""

    lat: float = Field(..., description="Latitude of the search centre.")
    lng: float = Field(..., description="Longitude of the search centre.")
    radius_m: int = Field(
        ...,
        description="Search radius in metres (e.g. 8000 for urban, 25000 for rural; max 500000).",
    )


class PlacesNearbyTool(BaseTool):
    """Tool for searching hospitals near a given location using the LocationIQ Nearby API."""

    name: str = "PlacesNearbyTool"
    description: str = (
        "Searches for hospitals near a given lat/lng within a specified radius using the LocationIQ Nearby API. "
        "Returns a list of places with place_id, name, lat, lng, business_status, and vicinity."
    )
    args_schema: Type[BaseModel] = PlacesNearbyToolInput

    # ------------------------------------------------------------------ #
    #  Main run method                                                     #
    # ------------------------------------------------------------------ #

    def _run(self, lat: float, lng: float, radius_m: int) -> Any:
        """
        Call the LocationIQ Nearby API for hospitals around (lat, lng)
        within radius_m metres (capped at 500,000 m).
        """
        api_key = os.environ.get("LOCATIONIQ_API_KEY", "")
        if not api_key:
            return {"error": "LOCATIONIQ_API_KEY environment variable is not set.", "results": []}

        # Cap radius as per LocationIQ limits
        capped_radius = min(radius_m, 500_000)

        params = {
            "key": api_key,
            "lat": lat,
            "lon": lng,
            "tag": "hospital",
            "radius": capped_radius,
            "limit": 50,
            "format": "json",
        }

        try:
            response = requests.get(
                "https://us1.locationiq.com/v1/nearby",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            raw_results = response.json()

            # LocationIQ returns a list directly on success
            if not isinstance(raw_results, list):
                return {"error": f"Unexpected response format: {raw_results}", "results": []}

            mapped = []
            for result in raw_results:
                try:
                    mapped.append(
                        {
                            "place_id": str(result.get("place_id", "")),
                            "name": result.get("name") or result.get("display_name", ""),
                            "lat": float(result["lat"]),
                            "lng": float(result["lon"]),
                            "business_status": "OPERATIONAL",
                            "vicinity": result.get("display_name", ""),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    # Skip malformed entries rather than crashing the whole call
                    continue

            return mapped

        except requests.exceptions.Timeout:
            return {"error": "Request to LocationIQ API timed out.", "results": []}
        except requests.exceptions.HTTPError as exc:
            return {"error": f"HTTP error: {exc}", "results": []}
        except requests.exceptions.RequestException as exc:
            return {"error": f"Request error: {exc}", "results": []}
        except Exception as exc:
            return {"error": f"Unexpected error: {exc}", "results": []}
