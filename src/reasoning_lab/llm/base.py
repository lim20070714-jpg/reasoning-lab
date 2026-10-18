"""Model-agnostic generation interface.

Strategies only depend on :class:`LLMClient`, so the same search code runs
against an OpenAI-compatible endpoint, a scripted mock in CI, or a cached
record of a previous run.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from ..schema import GenerationParams, Message


class LLMError(RuntimeError):
    """Raised when a provider cannot produce a completion."""


class Prompt(BaseModel):
    messages: list[Message]
    tag: str = ""

    @property
    def text(self) -> str:
        return "\n\n".join(f"[{m.role}]\n{m.content}" for m in self.messages)

    def fingerprint(self, model: str, params: GenerationParams) -> str:
        payload = json.dumps(
            {"model": model, "messages": [m.model_dump() for m in self.messages], "params": params.model_dump()},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Completion(BaseModel):
    text: str
    model: str = ""
    params: GenerationParams = Field(default_factory=GenerationParams)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    fingerprint: str = ""
    tag: str = ""
    finish_reason: str | None = None
    provider_meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Cheap token proxy used when a provider omits usage fields."""
        return max(1, round(len(text) / 4))


class LLMClient(ABC):
    """Minimal async chat-completion contract used across the harness."""

    model: str = ""
    supports_seed: bool = True

    @abstractmethod
    async def generate(
        self, prompt: Prompt, params: GenerationParams | None = None, *, tag: str = ""
    ) -> Completion: ...

    async def generate_many(
        self, prompt: Prompt, n: int, params: GenerationParams | None = None, *, tag: str = ""
    ) -> list[Completion]:
        """Produce ``n`` samples. Providers without batch sampling fan out calls."""
        return [await self.generate(prompt, params, tag=tag) for _ in range(n)]

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
