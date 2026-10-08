"""Two OpenAI-compatible LLM providers (Groq, NVIDIA NIM) and the bookkeeping that lets the agents share load between them.

  * request building per provider (each has its own quirks: Groq wants max_completion_tokens and reasoning_effort, NVIDIA's
    Nemotron needs its thinking mode switched off or a structured answer takes ~12 s instead of ~2 s);
  * a health tracker with a circuit breaker per provider: repeated failures or timeouts take a provider out of rotation for
    a short while, a rate limit puts it on cooldown, a model that answers 404 for this account is retired for the session;
  * Groq's per-model token budget, read from its response headers.

Keys come from the environment only (GROQ_API_KEY, NVIDIA_API_KEY) and are never logged or returned.
CEDCS_LLM_PROVIDERS=groq,nvidia limits which providers may be used (default: every provider that has a key).
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections import deque
from typing import Any, Optional

import requests

PROVIDERS: dict = {
    "groq": {"url": "https://api.groq.com/openai/v1/chat/completions", "key": "GROQ_API_KEY", "timeout": (5, 45), "rpm": 900},
    # NVIDIA's free hosted endpoints have cold starts (25-40 s) and roughly 40 requests/minute. A short read timeout makes a cold
    # call fail over to the other provider quickly (the request still wakes the endpoint, so the next call is fast), warm-up and
    # keep-warm pings avoid most cold starts, and a soft request-rate cap stays below the published limit.
    "nvidia": {"url": "https://integrate.api.nvidia.com/v1/chat/completions", "key": "NVIDIA_API_KEY", "timeout": (4, 12), "rpm": 35},
}
SESSIONS: dict = {"groq": requests.Session(), "nvidia": requests.Session()}
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

FAIL_THRESHOLD = 3  # consecutive failures that open a provider's circuit (a flaky-but-working provider gets a fair chance)
CIRCUIT_OPEN_S = 25.0
DEAD_MODEL_S = 600.0
NO_JSON_MODE: set = set()  # models that rejected response_format: retried without it, and remembered


def split(spec: str) -> tuple:
    """'nvidia:nvidia/nemotron-3-super-120b-a12b' -> ('nvidia', 'nvidia/nemotron-3-super-120b-a12b'). Bare model names mean Groq."""
    head, sep, rest = spec.partition(":")
    if sep and head in PROVIDERS:
        return head, rest
    return "groq", spec


def normalise(spec: str) -> str:
    if spec.startswith("groq/") and spec != "groq/compound":
        spec = spec[len("groq/"):]
    p, m = split(spec)
    return f"{p}:{m}"


def allowed_providers() -> list:
    raw = os.environ.get("CEDCS_LLM_PROVIDERS", "").strip()
    if not raw:
        return list(PROVIDERS)
    return [p.strip() for p in raw.split(",") if p.strip() in PROVIDERS]


def has_key(provider: str) -> bool:
    return bool(os.environ.get(PROVIDERS[provider]["key"], "").strip())


def provider_ok(provider: str) -> bool:
    """Configured (has a key) and permitted, regardless of current health."""
    return provider in allowed_providers() and has_key(provider)


def configured() -> list:
    return [p for p in PROVIDERS if provider_ok(p)]


# ------------------------------------------------------------------ Groq token budget
class Budget:
    """Client-side view of Groq's per-model tokens-per-minute budget, refreshed from the x-ratelimit-* response headers."""

    DEFAULT_LIMIT = 8000

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._m: dict = {}  # model -> (remaining_tokens, limit, monotonic time when the window resets)

    @staticmethod
    def _parse_reset(v: str) -> float:
        """'38.28s', '547ms', '1m3.2s' -> seconds."""
        total = 0.0
        for num, unit in re.findall(r"([\d.]+)(ms|s|m|h)", v or ""):
            total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
        return total

    def update(self, model: str, headers, status: int) -> None:
        try:
            limit = int(headers.get("x-ratelimit-limit-tokens") or self.DEFAULT_LIMIT)
            rem = headers.get("x-ratelimit-remaining-tokens")
            reset = self._parse_reset(headers.get("x-ratelimit-reset-tokens") or headers.get("retry-after") or "")
        except (AttributeError, ValueError):
            return
        if status == 429:
            rem = 0
            # A per-DAY limit (tokens per day) still reports a full per-minute bucket and a 1 ms reset, so trust retry-after:
            # it says when this model may be used again (seconds to hours).
            try:
                reset = max(reset, float(headers.get("retry-after") or 0))
            except (TypeError, ValueError):
                pass
        if rem is None:
            return
        with self._lock:
            self._m[model] = (int(float(rem)), limit, time.monotonic() + (reset or 60.0))

    def headroom(self, model: str) -> int:
        with self._lock:
            entry = self._m.get(model)
        if entry is None:
            return self.DEFAULT_LIMIT
        rem, limit, reset_at = entry
        return limit if time.monotonic() >= reset_at else rem

    def retry_in(self, model: str) -> int:
        """Seconds until this model has any budget again (0 if it has some now)."""
        with self._lock:
            entry = self._m.get(model)
        if entry is None or entry[0] > 0:
            return 0
        return max(0, round(entry[2] - time.monotonic()))

    def total(self) -> int:
        return sum(self.headroom(m) for m in GROQ_MODELS)

    def snapshot(self) -> dict:
        return {m: self.headroom(m) for m in GROQ_MODELS}


BUDGET = Budget()


# ------------------------------------------------------------------ health / circuit breaker
class Health:
    def __init__(self):
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.fails = {p: 0 for p in PROVIDERS}
            self.open_until = {p: 0.0 for p in PROVIDERS}
            self.cool_until = {p: 0.0 for p in PROVIDERS}
            self.ewma_ms: dict = {p: None for p in PROVIDERS}
            self.calls = {p: deque() for p in PROVIDERS}
            self.dead: dict = {}  # (provider, model) -> monotonic time until which it is retired
            self.totals = {p: {"ok": 0, "failed": 0, "rate_limited": 0} for p in PROVIDERS}
            NO_JSON_MODE.clear()

    def _rpm(self, p: str) -> int:
        now = time.monotonic()
        with self._lock:
            q = self.calls[p]
            while q and now - q[0] > 60:
                q.popleft()
            return len(q)

    def note_call(self, p: str) -> None:
        with self._lock:
            self.calls[p].append(time.monotonic())

    def available(self, provider: str, model: Optional[str] = None, force: bool = False) -> bool:
        """force=True is the last-resort probe: ignore an open circuit, a cooldown and the request-rate cap (a provider that is
        merely flaky beats no provider at all), but never a missing key or a model this account cannot use."""
        if not provider_ok(provider):
            return False
        now = time.monotonic()
        with self._lock:
            if model is not None and now < self.dead.get((provider, model), 0.0):
                return False
            if force:
                return True
            if now < self.open_until[provider] or now < self.cool_until[provider]:
                return False
        return self._rpm(provider) < PROVIDERS[provider]["rpm"]

    def success(self, p: str, ms: float) -> None:
        with self._lock:
            self.fails[p] = 0
            self.open_until[p] = 0.0
            self.ewma_ms[p] = ms if self.ewma_ms[p] is None else 0.7 * self.ewma_ms[p] + 0.3 * ms
            self.totals[p]["ok"] += 1

    def failure(self, p: str) -> None:
        with self._lock:
            self.fails[p] += 1
            self.totals[p]["failed"] += 1
            if self.fails[p] >= FAIL_THRESHOLD:
                self.open_until[p] = time.monotonic() + CIRCUIT_OPEN_S

    def rate_limited(self, p: str, retry_after: float = 0.0) -> None:
        """Provider-wide cooldown (NVIDIA limits per account). Groq's limits are per model and handled by BUDGET."""
        with self._lock:
            self.totals[p]["rate_limited"] += 1
            self.cool_until[p] = time.monotonic() + (retry_after if 0 < retry_after <= 90 else 20.0)

    def model_unavailable(self, p: str, model: str) -> None:
        with self._lock:
            self.dead[(p, model)] = time.monotonic() + DEAD_MODEL_S

    def state(self, p: str) -> str:
        if not provider_ok(p):
            return "not configured"
        now = time.monotonic()
        with self._lock:
            if now < self.open_until[p]:
                return "circuit open"
            if now < self.cool_until[p]:
                return "cooling down (rate limited)"
            if self.fails[p]:
                return "degraded"
        if p == "groq" and all(BUDGET.headroom(m) == 0 for m in GROQ_MODELS):
            return "quota exhausted"  # e.g. Groq's free tier caps tokens per DAY per model; the other provider carries the load
        return "healthy"

    def status(self) -> dict:
        now = time.monotonic()
        out = {}
        for p in PROVIDERS:
            with self._lock:
                out[p] = {
                    "state": self.state(p), "configured": provider_ok(p), "avg_ms": None if self.ewma_ms[p] is None else round(self.ewma_ms[p]),
                    "calls_last_min": self._rpm(p), "limit_per_min": PROVIDERS[p]["rpm"], **self.totals[p],
                    "retired_models": sorted(m for (pp, m), t in self.dead.items() if pp == p and t > now),
                    "retry_in_s": round(max(self.open_until[p], self.cool_until[p]) - now) if max(self.open_until[p], self.cool_until[p]) > now else 0,
                }
        if out["groq"]["state"] == "quota exhausted":
            out["groq"]["retry_in_s"] = min(BUDGET.retry_in(m) for m in GROQ_MODELS)
        return out


HEALTH = Health()


# ------------------------------------------------------------------ requests
def build_body(provider: str, model: str, system: str, user: str, json_mode: bool, max_tokens: int) -> dict:
    body: dict[str, Any] = {"model": model, "temperature": 0.1,
                            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    body["max_completion_tokens" if provider == "groq" else "max_tokens"] = max_tokens
    if json_mode and model not in NO_JSON_MODE:
        body["response_format"] = {"type": "json_object"}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"  # short reasoning: latency matters in an emergency
    elif provider == "groq" and "qwen" in model:
        body["reasoning_effort"] = "none"
    elif provider == "nvidia" and "nemotron" in model:
        body["chat_template_kwargs"] = {"enable_thinking": False}  # ~2.5 s instead of ~12 s, and valid JSON far more reliably
    return body


def post(provider: str, model: str, system: str, user: str, json_mode: bool, max_tokens: int, timeout: Optional[tuple] = None):
    """One HTTP call. Returns (response, elapsed_ms); raises requests.RequestException on network errors/timeouts."""
    cfg = PROVIDERS[provider]
    HEALTH.note_call(provider)
    t0 = time.perf_counter()
    r = SESSIONS[provider].post(cfg["url"], headers={"Authorization": f"Bearer {os.environ[cfg['key']]}"},
                                json=build_body(provider, model, system, user, json_mode, max_tokens), timeout=timeout or cfg["timeout"])
    return r, (time.perf_counter() - t0) * 1000
