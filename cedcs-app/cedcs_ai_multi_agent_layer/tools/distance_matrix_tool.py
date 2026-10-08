
import os
import time
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Module-level circuit breaker state
# ---------------------------------------------------------------------------
_CB_FAILURE_THRESHOLD = 3          # consecutive failures before opening
_CB_RECOVERY_TIMEOUT  = 60         # seconds to wait before allowing retry

_cb_consecutive_failures: int = 0
_cb_open_since: Optional[float] = None   # epoch timestamp when circuit opened


def _is_circuit_open() -> bool:
    """Return True if the circuit breaker is currently open (blocking calls)."""
    global _cb_open_since
    if _cb_open_since is None:
        return False
    if time.time() - _cb_open_since >= _CB_RECOVERY_TIMEOUT:
        # Recovery timeout elapsed — move to half-open (allow one retry)
        _cb_open_since = None
        return False
    return True


def _record_success() -> None:
    """Reset circuit breaker counters after a successful call."""
    global _cb_consecutive_failures, _cb_open_since
    _cb_consecutive_failures = 0
    _cb_open_since = None


def _record_failure() -> None:
    """Increment failure counter and open circuit if threshold is reached."""
    global _cb_consecutive_failures, _cb_open_since
    _cb_consecutive_failures += 1
    if _cb_consecutive_failures >= _CB_FAILURE_THRESHOLD:
        _cb_open_since = time.time()


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------
class DistanceMatrixInput(BaseModel):
    """Input schema for DistanceMatrixTool."""

    origin_lat: float = Field(
        ...,
        description="Latitude of the origin point.",
    )
    origin_lng: float = Field(
        ...,
        description="Longitude of the origin point.",
    )
    destinations: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "List of hospital destination dicts, each containing "
            "'hospital_id' (str), 'lat' (float), and 'lng' (float). "
            "Maximum 24 items; extras are silently ignored."
        ),
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
class DistanceMatrixTool(BaseTool):
    """Tool for computing travel time and distance to hospital destinations
    via the LocationIQ Driving Matrix API, with a built-in circuit breaker."""

    name: str = "DistanceMatrixTool"
    description: str = (
        "Computes travel time (seconds) and distance (metres) from one origin "
        "to up to 24 hospital destinations using the LocationIQ Driving Matrix API. "
        "Returns per-destination results with duration_in_traffic_seconds, "
        "distance_metres, and status."
    )
    args_schema: Type[BaseModel] = DistanceMatrixInput

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def _run(
        self,
        origin_lat: float,
        origin_lng: float,
        destinations: List[Dict[str, Any]],
    ) -> Any:
        """
        Query the LocationIQ OSRM Matrix API and return structured results.

        Parameters
        ----------
        origin_lat:   Latitude of the starting point.
        origin_lng:   Longitude of the starting point.
        destinations: Up to 24 hospital dicts with hospital_id / lat / lng.

        Returns
        -------
        A list of result dicts, or an error dict if something goes wrong.
        """

        # ── 1. Circuit breaker check ────────────────────────────────────────
        if _is_circuit_open():
            return {
                "error": "Circuit breaker open",
                "results": [],
            }

        # ── 2. Read API key ──────────────────────────────────────────────────
        api_key = os.environ.get("LOCATIONIQ_API_KEY", "")
        if not api_key:
            return {
                "error": "Environment variable LOCATIONIQ_API_KEY is not set.",
                "results": [],
            }

        # ── 3. Clamp destinations to 24 ─────────────────────────────────────
        destinations = destinations[:24]
        if not destinations:
            return {"error": "No destinations provided.", "results": []}

        # ── 4. Build coordinates path segment ───────────────────────────────
        # Format: "<origin_lng>,<origin_lat>;<d1_lng>,<d1_lat>;..."
        # Origin is always index 0; destinations are indices 1..N
        coord_parts = [f"{origin_lng},{origin_lat}"]
        for d in destinations:
            coord_parts.append(f"{d['lng']},{d['lat']}")
        coordinates = ";".join(coord_parts)

        # ── 5. Build destination indices string ──────────────────────────────
        # Indices 1-based for all destinations (origin is source=0)
        dest_indices = ";".join(str(i) for i in range(1, len(destinations) + 1))

        # ── 6. Call the LocationIQ OSRM Matrix API ───────────────────────────
        url = f"https://us1.locationiq.com/v1/matrix/driving/{coordinates}"
        params: Dict[str, str] = {
            "sources":      "0",
            "destinations": dest_indices,
            "annotations":  "true",
            "key":          api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            _record_failure()
            return {
                "error": "Request to LocationIQ Matrix API timed out.",
                "results": [],
            }
        except requests.exceptions.RequestException as exc:
            _record_failure()
            return {
                "error": f"HTTP request failed: {exc}",
                "results": [],
            }
        except ValueError:
            _record_failure()
            return {
                "error": "Failed to parse JSON response from LocationIQ Matrix API.",
                "results": [],
            }

        # ── 7. Parse durations and distances ────────────────────────────────
        # durations and distances are 2D arrays; take row [0] (from the single source)
        try:
            durations_row: List[Optional[float]] = data["durations"][0]
            distances_row: List[Optional[float]] = data["distances"][0]
        except (KeyError, IndexError, TypeError) as exc:
            _record_failure()
            return {
                "error": f"Unexpected response structure: {exc}",
                "results": [],
            }

        # ── 8. Map results back to hospital IDs ─────────────────────────────
        results: List[Dict[str, Any]] = []
        for idx, dest in enumerate(destinations):
            raw_duration = durations_row[idx] if idx < len(durations_row) else None
            raw_distance = distances_row[idx] if idx < len(distances_row) else None

            if raw_duration is None or raw_distance is None:
                results.append(
                    {
                        "hospital_id":                 dest.get("hospital_id", ""),
                        "duration_in_traffic_seconds": None,
                        "distance_metres":             None,
                        "status":                      "error",
                    }
                )
            else:
                results.append(
                    {
                        "hospital_id":                 dest.get("hospital_id", ""),
                        "duration_in_traffic_seconds": int(raw_duration),
                        "distance_metres":             int(raw_distance),
                        "status":                      "ok",
                    }
                )

        # ── 9. Record success and return ─────────────────────────────────────
        _record_success()
        return results
