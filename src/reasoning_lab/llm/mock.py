"""Deterministic scripted model used by tests, CI and the offline demo.

MockLLM answers from a solution bank keyed on question substrings. It handles
both whole-solution prompts and continuation prompts (tree search), and can
inject arithmetic perturbations so process reward models get real negative
examples without a GPU or API spend.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from ..schema import GenerationParams
from .base import Completion, LLMClient, Prompt

QUESTION_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.S)
PRIOR_RE = re.compile(r"<prior_steps>\s*(.*?)\s*</prior_steps>", re.S)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
TAIL_VALUE_RE = re.compile(r"=\s*([^=;]+?)\s*[.。]?\s*$")


class MockSolution(BaseModel):
    """A reference chain plus optional flawed chains used as search negatives."""

    match: str
    steps: list[str] = Field(default_factory=list)
    answer: str = ""
    flawed: list[list[str]] = Field(default_factory=list)

    def chain_text(self, steps: Sequence[str] | None = None) -> str:
        """Render a chain and close it with the answer that chain actually derives.

        Stating the gold answer after a corrupted chain would make every
        strategy immune to bad reasoning, so the label follows the steps.
        """
        body = list(steps if steps is not None else self.steps)
        lines = [f"{i + 1}. {s}" for i, s in enumerate(body)]
        joined = "\n".join(lines)
        answer = self.answer if steps is None else chain_answer(body, self.answer)
        if answer and "final answer" not in joined.lower():
            lines.append(f"Final Answer: {answer}")
        return "\n".join(lines)


def chain_answer(steps: Sequence[str], fallback: str) -> str:
    """Value derived by the last step that assigns one, e.g. ``r = 312.`` -> ``312``."""
    for text in reversed(list(steps)):
        match = TAIL_VALUE_RE.search(text)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return fallback


def perturb(text: str, salt: str) -> str:
    """Corrupt exactly one numeric literal so the step becomes verifiably wrong."""
    matches = [m for m in NUMBER_RE.finditer(text) if m.group() not in ("1", "0")]
    if not matches:
        return text
    pick = matches[int(hashlib.sha256(salt.encode()).hexdigest(), 16) % len(matches)]
    value = float(pick.group())
    replacement = f"{value + 1:g}" if int(hashlib.sha256((salt + "d").encode()).hexdigest(), 16) % 2 else f"{value * 2:g}"
    return text[: pick.start()] + replacement + text[pick.end() :]


Resender = Callable[[Prompt, GenerationParams, int], str]


class MockLLM(LLMClient):
    def __init__(
        self,
        solutions: Sequence[MockSolution | Mapping[str, Any]] = (),
        *,
        model: str = "mock-reasoning",
        responder: Resender | None = None,
        latency_ms: float = 0.0,
        flawed_rate: float = 0.35,
        seed: int = 0,
    ):
        self.model = model
        self.solutions = [s if isinstance(s, MockSolution) else MockSolution.model_validate(s) for s in solutions]
        self.responder = responder
        self.latency_ms = latency_ms
        self.flawed_rate = flawed_rate
        self.seed = seed
        self.calls = 0

    def find_solution(self, question: str) -> MockSolution | None:
        best: MockSolution | None = None
        best_len = -1
        for sol in self.solutions:
            needle = sol.match.lower()
            if needle and needle in question.lower() and len(needle) > best_len:
                best, best_len = sol, len(needle)
        return best

    async def generate(
        self, prompt: Prompt, params: GenerationParams | None = None, *, tag: str = ""
    ) -> Completion:
        p = params or GenerationParams()
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000.0)
        self.calls += 1
        salt = hashlib.sha256(f"{self.seed}:{prompt.text}:{p.seed}:{p.temperature}".encode()).hexdigest()
        sample_index = int(salt[:8], 16) % 1000
        text = (
            self.responder(prompt, p, sample_index)
            if self.responder
            else self._default_response(prompt, p, sample_index)
        )
        return Completion(
            text=text,
            model=self.model,
            params=p,
            prompt_tokens=Completion.estimate_tokens(prompt.text),
            completion_tokens=Completion.estimate_tokens(text),
            latency_ms=self.latency_ms,
            fingerprint=prompt.fingerprint(self.model, p),
            tag=tag or prompt.tag,
            finish_reason="stop",
            provider_meta={"mock": True, "sample_index": sample_index},
        )

    async def generate_many(
        self, prompt: Prompt, n: int, params: GenerationParams | None = None, *, tag: str = ""
    ) -> list[Completion]:
        p = params or GenerationParams()
        out: list[Completion] = []
        for i in range(n):
            clone = p.model_copy(update={"seed": (p.seed or self.seed) + i})
            out.append(await self.generate(prompt, clone, tag=tag))
        return out

    # -- response construction ---------------------------------------
    def _default_response(self, prompt: Prompt, params: GenerationParams, sample_index: int) -> str:
        question_match = QUESTION_RE.search(prompt.text)
        if not question_match:
            return "Final Answer: 0"
        question = question_match.group(1)
        sol = self.find_solution(question)
        if sol is None:
            return "Final Answer: unknown"

        prior = PRIOR_RE.search(prompt.text)
        if prior:
            return self._continuation(sol, prior.group(1), params)

        variant = self._roll_variant(sol, sample_index, params)
        if variant == "flawed" and sol.flawed:
            return sol.chain_text(sol.flawed[sample_index % len(sol.flawed)])
        if variant == "flawed":
            return sol.chain_text([perturb(s, f"{sol.match}:{i}") for i, s in enumerate(sol.steps)])
        if variant == "truncated":
            return sol.chain_text(sol.steps[: max(1, len(sol.steps) - 1)])
        return sol.chain_text()

    def _roll_variant(self, sol: MockSolution, sample_index: int, params: GenerationParams) -> str:
        if params.temperature <= 0.01:
            return "correct"
        bucket = sample_index / 1000.0
        if bucket < self.flawed_rate:
            return "flawed"
        if bucket < self.flawed_rate + 0.12:
            return "truncated"
        return "correct"

    def _continuation(self, sol: MockSolution, prior_text: str, params: GenerationParams) -> str:
        """Return the next step of the reference chain given the steps so far."""
        used = 0
        for step in sol.steps:
            probe = step[: min(24, len(step))].strip().lower()
            if probe and probe in prior_text.lower():
                used += 1
        if used >= len(sol.steps):
            return f"Final Answer: {sol.answer}"
        nxt = sol.steps[used]
        if params.temperature > 0.5 and self._is_flawed_continuation(sol, prior_text, used, params):
            nxt = perturb(nxt, f"{sol.match}:{used}:{prior_text[:32]}")
        return nxt

    def _is_flawed_continuation(self, sol: MockSolution, prior_text: str, used: int, params: GenerationParams) -> bool:
        digest = hashlib.sha256(f"{sol.match}|{used}|{prior_text[-48:]}|{params.seed}".encode()).hexdigest()
        return (int(digest[:8], 16) / 0xFFFFFFFF) < self.flawed_rate
