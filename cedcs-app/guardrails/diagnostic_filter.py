"""
Diagnostic-assertion denylist filter.

P1 ("LLM interprets, never decides") has a narrower sibling rule specific to
triage: the LLM must never assert a diagnosis. tasks.yaml already instructs
the Emergency Triage Assessor not to do this, and schemas/triage.py already
rejects known diagnosis terms at the Pydantic-validation layer. This module
is the SAME check applied a second, independent time, deliberately
duplicated rather than imported from schemas/triage.py:

    - schemas/triage.py's validator only ever sees the *rationale* field,
      because that's the only free-text field on TriageResult.
    - This filter is reusable against ANY agent's free text (clarification
      questions, facility discovery notes, explanation narration) — every
      surface where an LLM could leak a diagnosis, not just triage.

Defense in depth: two independently-maintained lists are less likely to
both miss the same wording than one shared list both layers trust blindly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Condition/diagnosis names and strong diagnostic phrasing that must never
# appear, asserted as fact, in any agent-generated text shown to a human
# decision-maker (ambulance crew, receiving hospital, family).
DIAGNOSIS_DENYLIST = [
    "heart attack",
    "myocardial infarction",
    "mi ",
    "stroke",
    "cva",
    "sepsis",
    "septic shock",
    "appendicitis",
    "aneurysm",
    "pulmonary embolism",
    "embolism",
    "meningitis",
    "diabetic ketoacidosis",
    "dka",
    "anaphylaxis",
    "pneumothorax",
    "subdural hematoma",
    "subarachnoid",
    "cardiac arrest",  # note: distinct from "in arrest" symptom description — see ALLOWLIST_PHRASES
]

# Phrases that legitimately contain a denylisted substring but describe an
# observed sign/capability requirement rather than asserting a diagnosis.
# Checked before flagging, so e.g. "cardiac monitoring capability required"
# doesn't false-positive on "cardiac".
ALLOWLIST_PHRASES = [
    "cardiac monitoring",
    "cardiac capability",
    "cardiac bay",
    "cardiac department",
]


@dataclass
class FilterResult:
    passed: bool
    flagged_terms: list[str] = field(default_factory=list)
    matched_spans: list[tuple[int, int]] = field(default_factory=list)


def _is_allowlisted_context(text_lower: str, start: int, end: int) -> bool:
    window = text_lower[max(0, start - 5) : end + 20]
    return any(phrase in window for phrase in ALLOWLIST_PHRASES)


def check_text(text: str) -> FilterResult:
    """Scan a single piece of agent-generated free text for diagnosis
    assertions. Word-boundary regex to avoid matching inside unrelated
    words (e.g. 'mi ' requires a trailing space/boundary, not a substring
    of 'admission')."""
    lowered = text.lower()
    flagged: list[str] = []
    spans: list[tuple[int, int]] = []

    for term in DIAGNOSIS_DENYLIST:
        pattern = r"\b" + re.escape(term.strip()) + r"\b"
        for m in re.finditer(pattern, lowered):
            if _is_allowlisted_context(lowered, m.start(), m.end()):
                continue
            flagged.append(term.strip())
            spans.append((m.start(), m.end()))

    return FilterResult(passed=len(flagged) == 0, flagged_terms=flagged, matched_spans=spans)


def check_fields(obj: dict, text_fields: list[str]) -> dict[str, FilterResult]:
    """Run check_text over the named string fields of a dict-shaped agent
    output (e.g. a TriageResult.model_dump() or an explanation payload).
    Non-string / missing fields are skipped silently."""
    results = {}
    for field_name in text_fields:
        value = obj.get(field_name)
        if isinstance(value, str) and value:
            results[field_name] = check_text(value)
    return results


class DiagnosticLeakError(Exception):
    """Raised by the orchestrator when a diagnostic-filter check fails and
    no retry budget remains — this is a hard stop, never silently
    stripped-and-continued, because silently editing agent output would
    make the audit trail lie about what the agent actually said."""

    def __init__(self, field_name: str, result: FilterResult):
        self.field_name = field_name
        self.result = result
        super().__init__(
            f"Diagnostic-assertion terms found in '{field_name}': {result.flagged_terms}"
        )
