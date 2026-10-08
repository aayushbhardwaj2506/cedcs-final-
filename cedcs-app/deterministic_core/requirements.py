from __future__ import annotations

from schemas.requirement import RequirementProfile, Strength
from schemas.triage import TriageResult


def compute_requirements(triage: TriageResult) -> RequirementProfile:
    """Mechanical reshape of the agent's required/preferred split. REQUIRED wins on overlap."""
    caps: dict[str, Strength] = {k: "PREFERRED" for k in triage.preferred_capabilities}
    for k in triage.required_capabilities:
        caps[k] = "REQUIRED"
    return RequirementProfile(capabilities=caps)
