"""Process reward model contract.

A PRM turns a reasoning prefix plus one candidate step into a probability that
the step is sound. Strategies consume this through the protocol below, so the
symbolic checker, a learned head and an LLM judge are interchangeable at run
time (``--prm symbolic|learned|judge|mixed|none``).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

from ...schema import Step, VerifyStatus

Aggregator = Literal["product", "mean", "min", "last", "geomean"]


class StepScore(BaseModel):
    score: float = 0.5
    status: VerifyStatus = VerifyStatus.UNKNOWN
    detail: str = ""
    source: str = ""
    checks: list[dict] = Field(default_factory=list)


class ProcessRewardModel(ABC):
    name: str = "prm"
    #: length-normalised product; see ``path_value``
    aggregator: Aggregator = "geomean"

    @abstractmethod
    async def score_step(self, problem_id: str, prefix: list[Step], step: Step) -> StepScore: ...

    async def score_path(self, problem_id: str, steps: list[Step]) -> list[StepScore]:
        scores: list[StepScore] = []
        for i, step in enumerate(steps):
            scores.append(await self.score_step(problem_id, steps[:i], step))
        return scores

    def path_value(self, scores: list[StepScore]) -> float:
        if not scores:
            return 0.5
        values = [max(1e-6, min(1.0, s.score)) for s in scores]
        if self.aggregator == "product":
            product = 1.0
            for value in values:
                product *= value
            return product
        if self.aggregator == "geomean":
            # product^(1/n): keeps "one rejected step condemns the chain" while
            # dropping the raw product's length bias, which ranks a correct
            # chain's own prefix above the completed chain
            log_sum = 0.0
            for value in values:
                log_sum += math.log(value)
            return math.exp(log_sum / len(values))
        if self.aggregator == "mean":
            return sum(values) / len(values)
        if self.aggregator == "min":
            return min(values)
        return values[-1]

    def rank(self, candidates: list[tuple[list[Step], list[StepScore]]]) -> list[float]:
        return [self.path_value(scores) for _, scores in candidates]


class NullPRM(ProcessRewardModel):
    """Uniform scores — the ablation baseline every comparison is run against."""

    name = "none"
    #: mean, not product: a flat prior must not penalise longer chains
    aggregator: Aggregator = "mean"

    async def score_step(self, problem_id: str, prefix: list[Step], step: Step) -> StepScore:
        return StepScore(score=0.5, status=VerifyStatus.UNKNOWN, source="none")
