"""Weighted ensemble of process reward models."""

from __future__ import annotations

import math

from ...schema import Step, VerifyStatus
from .base import Aggregator, ProcessRewardModel, StepScore


class MixedPRM(ProcessRewardModel):
    """Combines normalised log-odds of several PRMs.

    Used to show how much of the gain comes from the symbolic checker versus the
    learned head, and to let a run blend both without editing strategy code.
    """

    def __init__(self, parts: list[tuple[ProcessRewardModel, float]], *, aggregator: Aggregator = "geomean"):
        if not parts:
            raise ValueError("MixedPRM needs at least one component")
        total = sum(weight for _, weight in parts) or 1.0
        self.parts = [(p, weight / total) for p, weight in parts]
        self.aggregator = aggregator
        self.name = "mixed(" + "+".join(p.name for p, _ in self.parts) + ")"

    async def score_step(self, problem_id: str, prefix: list[Step], step: Step) -> StepScore:
        odds = 0.0
        components: list[StepScore] = []
        for prm, weight in self.parts:
            score = await prm.score_step(problem_id, prefix, step)
            components.append(score)
            p = max(1e-4, min(1 - 1e-4, score.score))
            odds += weight * math.log(p / (1.0 - p))
        combined = 1.0 / (1.0 + math.exp(-odds))
        status = next((c.status for c in components if c.status == VerifyStatus.FAIL), components[0].status)
        return StepScore(
            score=round(max(0.001, min(0.999, combined)), 5),
            status=status,
            detail=" | ".join(f"{c.source}={c.score:.3f}" for c in components),
            source=self.name,
            checks=components[0].checks,
        )
