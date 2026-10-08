from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import requests
import os


class GeocodingToolInput(BaseModel):
    """Input schema for GeocodingTool."""

    address: str = Field(
        ...,
        description="The free-text address or landmark to geocode.",
    )


class GeocodingTool(BaseTool):
    """Tool for converting a free-text address into geographic coordinates using the LocationIQ Geocoding API."""

    name: str = "GeocodingTool"
    description: str = (
        "Converts a free-text address or landmark into geographic coordinates using the LocationIQ Geocoding API. "
        "Returns a dict with keys: lat, lng, accuracy_m, formatted_address."
    )
    args_schema: Type[BaseModel] = GeocodingToolInput

    def _run(self, address: str) -> dict:
        """
        Geocode the given address using the LocationIQ Geocoding API.

        Returns a dict with keys:
            - lat (float): Latitude of the location.
            - lng (float): Longitude of the location.
            - accuracy_m (None): Not available from LocationIQ search endpoint.
            - formatted_address (str): Human-readable address returned by the API.

        On empty results returns:
            {"error": "No results returned", "lat": None, "lng": None, "accuracy_m": None}

        On HTTP error or any exception returns:
            {"error": str, "lat": None, "lng": None, "accuracy_m": None}
        """
        error_response = {"error": None, "lat": None, "lng": None, "accuracy_m": None}

        try:
            api_key = os.environ.get("LOCATIONIQ_API_KEY", "")

            params = {
                "q": address,
                "format": "json",
                "key": api_key,
            }

            response = requests.get(
                "https://us1.locationiq.com/v1/search",
                params=params,
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()

            # LocationIQ returns a list; handle empty results
            if not data:
                error_response["error"] = "No results returned"
                return error_response

            # Take the first result
            result = data[0]
            lat = float(result["lat"])
            lng = float(result["lon"])
            formatted_address = result.get("display_name", "")

            return {
                "lat": lat,
                "lng": lng,
                "accuracy_m": None,
                "formatted_address": formatted_address,
            }

        except Exception as exc:
            error_response["error"] = str(exc)
            return error_response
