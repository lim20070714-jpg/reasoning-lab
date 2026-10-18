"""Async client for OpenAI-compatible ``/chat/completions`` endpoints.

Deliberately uses httpx directly instead of the official SDK so one code path
covers OpenAI, DeepSeek, DashScope compat mode, Moonshot, OpenRouter, vLLM,
llama.cpp and Ollama's ``/v1`` surface, including providers that reject
optional sampling fields.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

import httpx

from ..config import LLMSettings
from ..schema import GenerationParams
from .base import Completion, LLMClient, LLMError, Prompt
from .cache import CompletionCache

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatClient(LLMClient):
    def __init__(self, settings: LLMSettings, cache: CompletionCache | None = None):
        self.settings = settings
        self.model = settings.model
        base = settings.base_url.rstrip("/")
        self._url = f"{base}/chat/completions"
        headers = {"User-Agent": settings.user_agent}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        self._client = httpx.AsyncClient(
            timeout=settings.timeout_s,
            headers=headers,
            limits=httpx.Limits(max_connections=max(1, settings.max_concurrency * 2), max_keepalive_connections=8),
        )
        self._semaphore = asyncio.Semaphore(max(1, settings.max_concurrency))
        self._cache = cache

    async def generate(
        self, prompt: Prompt, params: GenerationParams | None = None, *, tag: str = ""
    ) -> Completion:
        p = params or GenerationParams(temperature=self.settings.temperature, max_tokens=self.settings.max_tokens)
        if p.max_tokens in (None, 0):
            p = p.model_copy(update={"max_tokens": self.settings.max_tokens})
        fingerprint = prompt.fingerprint(self.model, p)
        if self._cache is not None:
            hit = self._cache.get(fingerprint)
            if hit is not None:
                hit.tag = tag or hit.tag
                return hit

        payload = self._payload(prompt, p)
        started = time.perf_counter()
        text, usage, finish_reason, meta = await self._post_with_retry(payload)
        latency_ms = (time.perf_counter() - started) * 1000.0

        completion = Completion(
            text=text,
            model=meta.get("model", self.model),
            params=p,
            prompt_tokens=int(usage.get("prompt_tokens") or Completion.estimate_tokens(prompt.text)),
            completion_tokens=int(usage.get("completion_tokens") or Completion.estimate_tokens(text)),
            latency_ms=round(latency_ms, 2),
            fingerprint=fingerprint,
            tag=tag or prompt.tag,
            finish_reason=finish_reason,
            provider_meta={k: meta[k] for k in ("id", "created") if k in meta},
        )
        if self._cache is not None:
            self._cache.put(completion)
        return completion

    async def generate_many(
        self, prompt: Prompt, n: int, params: GenerationParams | None = None, *, tag: str = ""
    ) -> list[Completion]:
        return list(await asyncio.gather(*(self.generate(prompt, params, tag=tag) for _ in range(n))))

    def _payload(self, prompt: Prompt, p: GenerationParams) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in prompt.messages],
            "temperature": p.temperature,
            "max_tokens": p.max_tokens,
        }
        if p.top_p is not None:
            payload["top_p"] = p.top_p
        if p.seed is not None and self.supports_seed:
            payload["seed"] = p.seed
        if p.stop:
            payload["stop"] = p.stop
        return payload

    async def _post_with_retry(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any], str | None, dict[str, Any]]:
        attempts = max(1, self.settings.max_retries + 1)
        last_error: str = "no attempt made"
        for attempt in range(attempts):
            try:
                async with self._semaphore:
                    response = await self._client.post(self._url, json=payload)
                if response.status_code in RETRYABLE_STATUS:
                    last_error = f"HTTP {response.status_code}: {response.text[:240]}"
                    if attempt + 1 < attempts:
                        await asyncio.sleep(self._backoff(attempt, response))
                        continue
                    raise LLMError(f"provider unavailable after {attempts} attempts — {last_error}")
                if response.status_code >= 400:
                    raise LLMError(f"request rejected (HTTP {response.status_code}): {response.text[:400]}")
                body = response.json()
                return self._parse(body)
            except (httpx.TransportError, httpx.InvalidURL) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 >= attempts:
                    raise LLMError(f"transport failure after {attempts} attempts — {last_error}") from exc
                await asyncio.sleep(self._backoff(attempt))
        raise LLMError(f"generation failed — {last_error}")

    def _backoff(self, attempt: int, response: httpx.Response | None = None) -> float:
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return min(60.0, float(retry_after))
                except ValueError:
                    pass
        return min(30.0, (2**attempt) * 0.5 + random.uniform(0, 0.4))

    @staticmethod
    def _parse(body: dict[str, Any]) -> tuple[str, dict[str, Any], str | None, dict[str, Any]]:
        choices = body.get("choices") or []
        if not choices:
            raise LLMError(f"provider returned no choices: {str(body)[:300]}")
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content")
        if not text:
            # o1-style and some reasoning models put usable output in reasoning_content
            text = message.get("reasoning_content") or message.get("text") or ""
        if not isinstance(text, str):
            text = str(text)
        meta = {k: body.get(k) for k in ("model", "id", "created") if body.get(k) is not None}
        return text.strip(), body.get("usage") or {}, choice.get("finish_reason"), meta

    async def aclose(self) -> None:
        await self._client.aclose()
