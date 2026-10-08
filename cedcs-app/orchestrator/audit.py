"""
Audit-event logging.

Every stage transition, guardrail check, retry, and deterministic-core call
is recorded as a structured, append-only event. This is what makes the
system's decisions reconstructible after the fact — required for any
emergency-response tool, and explicitly called for in the design doc's
audit-trail requirement ("a full audit trail").

Kept intentionally dependency-free (stdlib only) so it can be swapped for a
real sink (Postgres table, structured log shipper) without touching call
sites — every call site only ever talks to AuditLog, never to a specific
storage backend.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger("cedcs.audit")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AuditEvent:
    event_id: str
    case_id: Optional[str]
    stage: str
    event_type: str  # STAGE_START | STAGE_END | GUARDRAIL_PASS | GUARDRAIL_FAIL | RETRY | ESCALATION | ERROR
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=True)


class AuditLog:
    """In-memory + logging-module sink. `events` accumulates the full
    trail for the current case so the orchestrator can attach it to the
    final result (or persist it) without a second query."""

    def __init__(self, case_id: Optional[str] = None, on_event: Optional[Callable[[dict], None]] = None):
        self.case_id = case_id
        self.events: list[AuditEvent] = []
        # Timing: every stage_start/stage_end pair becomes a span (nested via depth) so the UI
        # can draw a waterfall. perf_counter is monotonic, so spans are safe against clock changes.
        self._t0 = time.perf_counter()
        self._lock = threading.RLock()
        self._tl = threading.local()  # per-thread nesting, so parallel agents keep correct depths
        self._open: list[dict] = []
        self.spans: list[dict] = []
        self.messages: list[dict] = []  # orchestrator <-> agent/service messages, for the live diagram
        self.on_event = on_event  # live sink, e.g. the SSE stream

    def _emit(self, payload: dict) -> None:
        if self.on_event:
            try:
                self.on_event(payload)
            except Exception:  # a broken UI stream must never break the pipeline
                logger.exception("audit on_event callback failed")

    def _ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def _stack(self) -> list:
        if not hasattr(self._tl, "stack"):
            self._tl.stack, self._tl.base = [], 0
        return self._tl.stack

    def adopt_depth(self, base: int) -> None:
        """Call at the start of a worker thread: its spans nest under the spawning thread's current depth."""
        self._stack()
        self._tl.base = base

    def current_depth(self) -> int:
        return getattr(self._tl, "base", 0) + len(self._stack())

    def message(self, frm: str, to: str, kind: str, label: str, **detail: Any) -> dict:
        """One message on the orchestration bus. kind: command | result | data | skip | error."""
        msg = {"type": "message", "t_ms": round(self._ms(), 3), "from": frm, "to": to, "kind": kind, "label": label, **detail}
        with self._lock:
            self.messages.append(msg)
        self._emit(msg)
        return msg

    def _record(self, stage: str, event_type: str, detail: Optional[dict[str, Any]] = None) -> AuditEvent:
        evt = AuditEvent(
            event_id=str(uuid.uuid4()),
            case_id=self.case_id,
            stage=stage,
            event_type=event_type,
            detail=detail or {},
        )
        with self._lock:
            self.events.append(evt)
        logger.info("AUDIT %s", evt.to_json())
        return evt

    def stage_start(self, stage: str, **detail: Any) -> AuditEvent:
        span = {"stage": stage, "start_ms": self._ms(), "depth": self.current_depth(), "detail": dict(detail)}
        self._stack().append(span)
        with self._lock:
            self._open.append(span)
        self._emit({"type": "stage_start", "stage": stage, "start_ms": round(span["start_ms"], 3), "depth": span["depth"]})
        return self._record(stage, "STAGE_START", detail)

    def _close(self, stage: str, detail: dict, status: str) -> None:
        with self._lock:
            span = next((sp for sp in reversed(self._open) if sp["stage"] == stage), None)
            if span is None:
                return
            self._open.remove(span)
            for st in (self._stack(),):
                if span in st:
                    st.remove(span)
            span["duration_ms"] = round(self._ms() - span["start_ms"], 3)
            span["start_ms"] = round(span["start_ms"], 3)
            span["status"] = status
            span["detail"].update(detail)
            self.spans.append(span)
        self._emit({"type": "stage_end", **{k: span[k] for k in ("stage", "start_ms", "duration_ms", "depth", "status")},
                    "detail": span["detail"]})

    def stage_end(self, stage: str, **detail: Any) -> AuditEvent:
        self._close(stage, detail, "ok")
        return self._record(stage, "STAGE_END", detail)

    def close_open(self, status: str = "halted") -> None:
        """Close any stage left open by a halt so the timeline stays complete."""
        with self._lock:
            open_now = list(reversed(self._open))
        for span in open_now:
            self._close(span["stage"], {}, status)

    def timeline(self) -> list[dict]:
        with self._lock:
            return sorted(self.spans, key=lambda s: (s["start_ms"], s["depth"]))

    def guardrail_pass(self, stage: str, guardrail: str, **detail: Any) -> AuditEvent:
        return self._record(stage, "GUARDRAIL_PASS", {"guardrail": guardrail, **detail})

    def guardrail_fail(self, stage: str, guardrail: str, **detail: Any) -> AuditEvent:
        return self._record(stage, "GUARDRAIL_FAIL", {"guardrail": guardrail, **detail})

    def retry(self, stage: str, reason: str, **detail: Any) -> AuditEvent:
        return self._record(stage, "RETRY", {"reason": reason, **detail})

    def escalation(self, stage: str, reason: str, **detail: Any) -> AuditEvent:
        return self._record(stage, "ESCALATION", {"reason": reason, **detail})

    def error(self, stage: str, message: str, **detail: Any) -> AuditEvent:
        return self._record(stage, "ERROR", {"message": message, **detail})

    def as_list(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self.events]
