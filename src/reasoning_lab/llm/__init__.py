"""Factory for the configured model backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..config import LLMSettings
from .base import LLMClient
from .cache import CompletionCache
from .mock import MockLLM, MockSolution
from .openai_compat import OpenAICompatClient


def build_llm(settings: LLMSettings, *, solutions: Sequence[MockSolution | dict[str, Any]] | None = None) -> LLMClient:
    if settings.is_mock:
        return MockLLM(list(solutions or []), latency_ms=settings.mock_latency_ms)
    settings.require_credentials()
    cache = CompletionCache(settings.cache_dir) if settings.cache_dir else None
    return OpenAICompatClient(settings, cache=cache)


def mock_solution_from_problem(problem: dict[str, Any]) -> MockSolution:
    return MockSolution.model_validate(
        {
            "match": problem.get("match") or str(problem.get("question", ""))[:40],
            "steps": problem.get("steps") or [],
            "answer": problem.get("answer", ""),
            "flawed": problem.get("flawed") or [],
        }
    )


__all__ = [
    "CompletionCache",
    "LLMClient",
    "MockLLM",
    "MockSolution",
    "OpenAICompatClient",
    "build_llm",
    "mock_solution_from_problem",
]
