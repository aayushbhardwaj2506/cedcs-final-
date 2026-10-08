"""Escalation flagging.

Implements only the red-flag rule and rung-4 (operator handoff) identification:
escalated == bool(red_flags) or no eligible hospital.

TODO(next pass): rungs 1-3 (widen radius / drop PREFERRED / stabilise-transfer)
need discovery to be re-invoked by the orchestrator, not from in here.
"""

from __future__ import annotations

from typing import Optional

from schemas.hospital import CandidateHospital
from schemas.triage import TriageResult


def apply_escalation(
    triage: TriageResult, eligible: list[CandidateHospital], red_flags_applied: list[str]
) -> tuple:
    """Returns (escalated, reason). A red-flag reason wins over no-eligible when both apply."""
    hard = [f for f in red_flags_applied if not f.startswith("AI_")]
    ai = [f for f in red_flags_applied if f.startswith("AI_")]
    if hard:
        return True, f"RED_FLAG_ESCALATION: {', '.join(hard + ai)}"
    if ai:
        return True, f"AI_REVIEW_ESCALATION: {', '.join(ai)}"
    if not eligible:
        return True, "RUNG_4_NO_ELIGIBLE_HOSPITAL: no candidate satisfied all required capabilities; operator handoff"
    return False, None
