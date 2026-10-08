"""
Canonical case schemas — StructuredCase and its components.

Mirrors §3.5 (Canonical Data Model) of the CEDCS Final Design Document and
§3 of the CrewAI build prompt. These are the contracts every agent output
is validated against before the orchestrator trusts it.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Patient(BaseModel):
    age: Optional[int] = None
    age_confidence: float = 0.0
    gender: Optional[str] = None
    consciousness: Literal["alert", "confused", "unresponsive", "unknown"] = "unknown"
    breathing: Literal["normal", "laboured", "slow", "absent", "unknown"] = "unknown"
    bleeding: Literal["none", "mild", "severe", "unknown"] = "unknown"
    pulse: Optional[float] = None
    spo2: Optional[float] = None
    bp: Optional[str] = None
    symptoms: list[str] = Field(default_factory=list)
    onset_time: Optional[str] = None
    medical_history: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    blood_group: Optional[str] = None


class Incident(BaseModel):
    mechanism: Optional[str] = None
    location_description: Optional[str] = None


class Location(BaseModel):
    lat: Optional[float] = None
    lng: Optional[float] = None
    address: Optional[str] = None
    accuracy_m: Optional[float] = None
    source: Literal["GPS", "GEOCODED", "MANUAL"] = "GPS"


class StructuredCase(BaseModel):
    case_id: Optional[str] = None  # assigned by the Emergency Case Manager, not by any agent
    patient: Patient
    incident: Incident = Field(default_factory=Incident)
    location: Location
    attachments: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)  # ranked by decision impact
    field_confidence: dict[str, float] = Field(default_factory=dict)
    intake_completeness: float = 0.0


class ClarificationQuestion(BaseModel):
    field: str
    question: str
    impact: str


class ClarificationBatch(BaseModel):
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    round_rationale: str = ""
