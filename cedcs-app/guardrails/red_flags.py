"""
Escalate-only red-flag rule layer.

Structural safety property: red flags may only ever RAISE priority, never
lower it. This is enforced by construction — the combining function is a
max() over an ordinal scale — so there is no code path, however buggy,
that could cause a red-flag check to downgrade urgency. If you find
yourself wanting to make this take the red flag's suggestion directly
instead of max()-ing it against the existing priority, don't: that would
reintroduce the exact bug class this module exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas.triage import PRIORITY_ORDER, Priority

# Inverse lookup: ordinal -> label
_ORDER_TO_PRIORITY: dict[int, Priority] = {v: k for k, v in PRIORITY_ORDER.items()}


@dataclass
class RedFlagRule:
    name: str
    predicate_field: str  # dotted path into StructuredCase, e.g. "patient.breathing"
    trigger_values: set[str]
    forced_priority: Priority
    rationale: str


# Vital-sign / reported-symptom combinations that must force a minimum
# priority regardless of what the Triage Assessor agent concluded.
# These mirror Matrix A's red-flag table — kept in plain deterministic
# Python specifically so an LLM error can never suppress one.
RED_FLAG_RULES: list[RedFlagRule] = [
    RedFlagRule(
        name="absent_breathing",
        predicate_field="patient.breathing",
        trigger_values={"absent"},
        forced_priority="CRITICAL",
        rationale="Reported absent breathing forces CRITICAL regardless of agent triage output.",
    ),
    RedFlagRule(
        name="unresponsive",
        predicate_field="patient.consciousness",
        trigger_values={"unresponsive"},
        forced_priority="CRITICAL",
        rationale="Reported unresponsiveness forces CRITICAL regardless of agent triage output.",
    ),
    RedFlagRule(
        name="severe_bleeding",
        predicate_field="patient.bleeding",
        trigger_values={"severe"},
        forced_priority="HIGH",
        rationale="Reported severe bleeding forces a minimum of HIGH priority.",
    ),
]


def _get_dotted(obj, dotted_path: str):
    node = obj
    for part in dotted_path.split("."):
        node = getattr(node, part, None)
        if node is None:
            return None
    return node


def apply_red_flags(structured_case, triage_priority: Priority) -> tuple[Priority, list[str]]:
    """
    Combine the agent's triage priority with every red-flag rule via max()
    on the ordinal scale. Returns (final_priority, applied_rule_names).

    Never call this with the intention of using the rule's forced_priority
    directly — always route it through this function so the max() combine
    is the only path to a final value.
    """
    current_ordinal = PRIORITY_ORDER[triage_priority]
    applied: list[str] = []

    for rule in RED_FLAG_RULES:
        value = _get_dotted(structured_case, rule.predicate_field)
        if value in rule.trigger_values:
            forced_ordinal = PRIORITY_ORDER[rule.forced_priority]
            if forced_ordinal > current_ordinal:
                current_ordinal = forced_ordinal
            applied.append(rule.name)

    return _ORDER_TO_PRIORITY[current_ordinal], applied
