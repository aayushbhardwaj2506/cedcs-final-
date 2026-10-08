"""
Schema-retry-with-fallback guardrail.

CrewAI agents return free-text-shaped JSON from an LLM; that JSON can be
malformed, missing required keys, or (per P1) can smuggle in things it
shouldn't (a diagnosis name, a hospital recommendation from the wrong
agent). This wrapper is the single choke point every agent output passes
through before the orchestrator trusts it as a typed object.

Policy: validate -> on failure, ask the SAME agent to retry ONCE with the
validation error fed back verbatim -> on second failure, fall back to a
safe default (or escalate to human, for fields where no safe default
exists) rather than ever passing unvalidated data to the deterministic
core.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("cedcs.guardrails.schema_retry")


class SchemaRetryExhausted(Exception):
    """Raised when both the original attempt and the single retry fail
    schema validation and no safe fallback was supplied. The orchestrator
    must treat this as a hard stop for the case, not a silent skip."""

    def __init__(self, schema: type[BaseModel], errors: list[str]):
        self.schema = schema
        self.errors = errors
        super().__init__(
            f"Schema validation failed for {schema.__name__} after retry: {errors}"
        )


@dataclass
class RetryOutcome(Generic[T]):
    value: T
    attempts: int
    used_fallback: bool
    validation_errors: list[str]


def _extract_json(raw: str) -> dict:
    """Agents sometimes wrap JSON in prose or markdown fences despite
    instructions. Try straight parse first, then strip common wrappers."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    if "```" in raw:
        # take the content of the first fenced block
        parts = raw.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[len("json"):].strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    # last resort: slice between first '{' or '[' and the matching last brace
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = raw.find(open_ch)
        end = raw.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise json.JSONDecodeError("Could not locate valid JSON in agent output", raw, 0)


# Public alias — facility discovery / resource interpretation outputs are
# JSON *arrays*, not single objects, so orchestrator.py's list-validation
# helper needs the same tolerant extraction without going through
# validate_with_retry's single-object shape.
extract_json = _extract_json


def validate_list_with_retry(
    schema: type[T],
    raw_output: str,
    retry_fn: Optional[Callable[[str], str]] = None,
) -> list[T]:
    """Same tolerant JSON extraction as validate_with_retry, but for a
    JSON array of objects each conforming to `schema` (CandidateHospital
    list, ResourceSnapshot list). No per-item retry — if any single
    element fails validation, the whole batch is treated as invalid
    (partial hospital lists are worse than none: a silently dropped
    candidate is indistinguishable from one correctly filtered out)."""
    errors: list[str] = []

    for attempt in (1, 2):
        current_raw = raw_output if attempt == 1 else None
        if attempt == 2:
            if retry_fn is None:
                break
            current_raw = retry_fn("; ".join(errors))

        try:
            data = _extract_json(current_raw)
            if not isinstance(data, list):
                raise ValueError(f"expected a JSON array, got {type(data).__name__}")
            return [schema.model_validate(item) for item in data]
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            msg = str(exc)
            errors.append(msg)
            logger.warning(
                "validate_list_with_retry: attempt %d failed for list[%s]: %s",
                attempt, schema.__name__, msg,
            )

    raise SchemaRetryExhausted(schema, errors)


def validate_with_retry(
    schema: type[T],
    raw_output: str,
    retry_fn: Optional[Callable[[str], str]] = None,
    fallback: Optional[T] = None,
) -> RetryOutcome[T]:
    """
    Parameters
    ----------
    schema:      the Pydantic model the output must conform to.
    raw_output:  the agent's raw string output (first attempt).
    retry_fn:    optional callable(error_message) -> new raw_output string.
                 Should re-invoke the SAME agent/task with the validation
                 error appended to its prompt. If None, no retry is attempted.
    fallback:    an already-constructed safe default instance to fall back to
                 if both attempts fail. If None, failure raises SchemaRetryExhausted.

    Returns
    -------
    RetryOutcome with the validated (or fallback) instance and bookkeeping
    about how many attempts were needed — the orchestrator logs this to the
    audit trail regardless of outcome.
    """
    errors: list[str] = []

    for attempt in (1, 2):
        current_raw = raw_output if attempt == 1 else None
        if attempt == 2:
            if retry_fn is None:
                break
            current_raw = retry_fn("; ".join(errors))

        try:
            data = _extract_json(current_raw)
            instance = schema.model_validate(data)
            return RetryOutcome(
                value=instance,
                attempts=attempt,
                used_fallback=False,
                validation_errors=errors,
            )
        except (ValidationError, json.JSONDecodeError) as exc:
            msg = str(exc)
            errors.append(msg)
            logger.warning(
                "schema_retry: attempt %d failed for %s: %s", attempt, schema.__name__, msg
            )

    if fallback is not None:
        return RetryOutcome(
            value=fallback, attempts=len(errors), used_fallback=True, validation_errors=errors
        )

    raise SchemaRetryExhausted(schema, errors)
