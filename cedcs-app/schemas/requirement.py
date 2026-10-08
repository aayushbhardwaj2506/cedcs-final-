"""RequirementProfile — the capability set eligibility/ranking evaluate against.

Produced mechanically from TriageResult by deterministic_core.requirements;
no agent chooses these strengths beyond the required/preferred split it
already made in triage (P1).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Strength = Literal["REQUIRED", "PREFERRED", "OPTIONAL"]


class RequirementProfile(BaseModel):
    capabilities: dict[str, Strength] = Field(default_factory=dict)

    def required(self) -> set[str]:
        return {k for k, v in self.capabilities.items() if v == "REQUIRED"}

    def preferred(self) -> set[str]:
        return {k for k, v in self.capabilities.items() if v == "PREFERRED"}
