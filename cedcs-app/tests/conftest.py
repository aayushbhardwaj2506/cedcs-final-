"""
Pytest configuration — wires up PYTHONPATH so `schemas`, `guardrails`, and
`orchestrator` import as top-level packages (matching the import style used
throughout orchestrator/orchestrator.py), and adds a place to point at the
exported CrewAI Studio project + deterministic_core package once they exist
on disk.

To actually exercise the CrewAI-dependent code paths (anything that calls
_import_crew() or imports deterministic_core), set these two environment
variables before running pytest:

    CEDCS_CREW_SRC=/path/to/cedcs_ai_multi_agent_layer_v1_crewai-project/src
    CEDCS_DETERMINISTIC_CORE_SRC=/path/to/your/deterministic_core/parent/dir

Without them, tests that need those imports are skipped (see
tests/test_orchestrator_integration.py) — the schema, guardrail, and
validation-gate unit tests below don't need either and always run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Hermetic tests: blank any real secrets so a developer's .env can never trigger paid/network calls.
for _k in ("LOCATIONIQ_KEY", "GROQ_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY", "CEDCS_LLM_BASE_URL", "CEDCS_LLM_PROVIDERS", "APP_DATABASE_URL", "HISTORY_DATABASE_URL"):
    os.environ[_k] = ""

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_llm_budget():
    """The Groq token budget is process-wide state; every test starts from a full budget."""
    from llm import groq_agents

    groq_agents.BUDGET.reset()
    groq_agents.HEALTH.reset()
    yield
    groq_agents.BUDGET.reset()
    groq_agents.HEALTH.reset()


@pytest.fixture(autouse=True)
def _history_off(monkeypatch):
    """Run history is off in tests (they must never write to a real database); history tests switch it on with a temp file."""
    from services import history

    monkeypatch.setenv("CEDCS_HISTORY", "off")
    monkeypatch.setenv("CEDCS_HISTORY_BACKEND", "sqlite")
    history.reset()
    yield
    history.reset()


@pytest.fixture(autouse=True)
def _isolated_app_db(tmp_path):
    """Settings, contacts, outbox and acks live in SQLite; each test gets its own empty database."""
    from dispatch import store

    store.use(tmp_path / "app.db")
    yield


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for env_var in ("CEDCS_CREW_SRC", "CEDCS_DETERMINISTIC_CORE_SRC"):
    path = os.environ.get(env_var)
    if path and path not in sys.path:
        sys.path.insert(0, path)
