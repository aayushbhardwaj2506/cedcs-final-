"""What the scene-analysis agent may say about a photo or a video frame.

Deliberately narrow: it describes what is VISIBLE and nothing more. It never names a condition (the text is filtered like every
other agent's), never identifies anyone, and every field falls back to "unclear" rather than guessing. Advisory only: nothing
here changes a case unless a person adds the sentence to the report.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

POSITIONS = ("lying", "sitting", "standing", "unclear", "no person")
BLEEDING = ("none", "minor", "heavy", "unclear")
MOVEMENT = ("moving", "no visible movement", "unclear")
URGENCY = ("CRITICAL", "HIGH", "MODERATE", "LOW", "unclear")
QUALITY = ("good", "poor", "unusable")


def _choice(v: Any, allowed: tuple, default: str = "unclear") -> str:
    s = str(v or "").strip()
    for a in allowed:
        if s.lower() == a.lower():
            return a
    return default


def _items(v: Any, n: int = 6, width: int = 120) -> list:
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    return [str(x).strip()[:width] for x in v if str(x).strip()][:n]


class SceneAnalysis(BaseModel):
    people_visible: int | None = Field(default=None, ge=0, le=50)
    person_position: str = "unclear"
    visible_bleeding: str = "unclear"
    visible_bleeding_where: str = ""
    visible_injuries: list = Field(default_factory=list)
    movement_cues: str = "unclear"
    hazards: list = Field(default_factory=list)
    urgency_hint: str = "unclear"
    summary: str = ""
    report_sentence: str = ""
    image_quality: str = "good"
    limits: str = "Breathing, pulse and consciousness cannot be judged reliably from an image."

    @field_validator("people_visible", mode="before")
    @classmethod
    def _people(cls, v):
        try:
            return max(0, min(50, int(v)))
        except (TypeError, ValueError):
            return None

    @field_validator("person_position", mode="before")
    @classmethod
    def _pos(cls, v):
        return _choice(v, POSITIONS)

    @field_validator("visible_bleeding", mode="before")
    @classmethod
    def _bleed(cls, v):
        return _choice(v, BLEEDING)

    @field_validator("movement_cues", mode="before")
    @classmethod
    def _move(cls, v):
        return _choice(v, MOVEMENT)

    @field_validator("urgency_hint", mode="before")
    @classmethod
    def _urg(cls, v):
        return _choice(v, URGENCY)

    @field_validator("image_quality", mode="before")
    @classmethod
    def _qual(cls, v):
        return _choice(v, QUALITY, "good")

    @field_validator("visible_injuries", "hazards", mode="before")
    @classmethod
    def _lists(cls, v):
        return _items(v)

    @field_validator("summary", "report_sentence", "limits", "visible_bleeding_where", mode="before")
    @classmethod
    def _text(cls, v):
        return str(v or "").strip()[:400]
