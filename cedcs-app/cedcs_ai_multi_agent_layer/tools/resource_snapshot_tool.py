from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List, Any
import requests
import os


class ResourceSnapshotToolInput(BaseModel):
    """Input schema for ResourceSnapshotTool."""

    hospital_ids: List[str] = Field(
        ...,
        description="The CEDCS hospital IDs to fetch resources for.",
    )


class ResourceSnapshotTool(BaseTool):
    """Tool for retrieving raw hospital resource feeds from the Resource Retrieval Service."""

    name: str = "ResourceSnapshotTool"
    description: str = (
        "Retrieves raw heterogeneous resource feeds (beds, ICU, ventilators, blood bank, specialists, etc.) "
        "for a list of hospital IDs from the Resource Retrieval Service. Returns per-hospital raw feed data "
        "including resource_key, value, updated_at, source, and reporter_id."
    )
    args_schema: Type[BaseModel] = ResourceSnapshotToolInput

    def _run(self, hospital_ids: List[str]) -> Any:
        """
        POST to the Resource Retrieval Service and return raw hospital resource snapshots.

        Args:
            hospital_ids: List of CEDCS hospital IDs to retrieve resource snapshots for.

        Returns:
            A list of per-hospital snapshot dicts on success, or an error dict on failure.
        """
        base_url = os.environ.get("RESOURCE_SERVICE_URL", "").rstrip("/")

        try:
            response = requests.post(
                url=f"{base_url}/snapshots",
                json={"hospital_ids": hospital_ids},
                timeout=10,
            )

            # Raise an HTTPError for 4xx/5xx responses
            if not response.ok:
                return {
                    "error": f"{response.status_code}: {response.text}",
                    "snapshots": [],
                }

            return response.json()

        except Exception as e:
            return {
                "error": str(e),
                "snapshots": [],
            }
