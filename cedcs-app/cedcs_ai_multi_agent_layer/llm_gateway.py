"""Single place the crew's LLM is configured, so swapping models is an env change.

Local Qwen via Ollama later:
    CEDCS_LLM_MODEL=ollama/qwen2.5:7b  CEDCS_LLM_BASE_URL=http://localhost:11434
"""

import os

from crewai import LLM


def get_llm() -> LLM:
    model = os.environ.get("CEDCS_LLM_MODEL", "openai/gpt-5.4-mini")
    base_url = os.environ.get("CEDCS_LLM_BASE_URL")
    kwargs = {"model": model}
    if base_url:
        kwargs["base_url"] = base_url
    return LLM(**kwargs)
