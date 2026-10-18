"""Process reward models and the factory used by the CLI and the server."""

from __future__ import annotations

from pathlib import Path

from ...llm.base import LLMClient
from ...schema import Problem
from ..symbolic import SymbolicVerifier
from .base import Aggregator, NullPRM, ProcessRewardModel, StepScore
from .learned import LearnedPRM
from .llm_judge import LLMJudgePRM
from .mixed import MixedPRM
from .symbolic import SymbolicPRM

__all__ = [
    "LLMJudgePRM",
    "LearnedPRM",
    "MixedPRM",
    "NullPRM",
    "ProcessRewardModel",
    "StepScore",
    "SymbolicPRM",
    "SymbolicVerifier",
    "build_prm",
]

PRM_KINDS = ("none", "symbolic", "learned", "judge", "mixed")


def build_prm(
    kind: str = "symbolic",
    *,
    problem: Problem | None = None,
    llm: LLMClient | None = None,
    model_path: Path | str | None = None,
    aggregator: Aggregator = "geomean",
    weights: tuple[float, float] = (0.6, 0.4),
) -> ProcessRewardModel:
    """Resolve ``--prm`` into a concrete scorer.

    ``learned`` and ``mixed`` fall back to the symbolic verifier when no trained
    head has been saved yet, so a fresh clone is never left without a baseline.
    """
    if kind == "none":
        return NullPRM()
    if kind == "symbolic":
        return SymbolicPRM(aggregator=aggregator)
    if kind == "learned":
        if model_path and Path(model_path).is_file():
            return LearnedPRM.load(model_path)
        return SymbolicPRM(aggregator=aggregator)
    if kind == "judge":
        if problem is None or llm is None:
            raise ValueError("the judge PRM needs both a problem and a model client")
        return LLMJudgePRM(problem, llm, aggregator=aggregator)
    if kind == "mixed":
        symbolic = SymbolicPRM(aggregator=aggregator)
        learned = (
            LearnedPRM.load(model_path)
            if model_path and Path(model_path).is_file()
            else SymbolicPRM(aggregator=aggregator)
        )
        return MixedPRM([(symbolic, weights[0]), (learned, weights[1])], aggregator=aggregator)
    raise ValueError(f"unknown prm kind '{kind}', expected one of {PRM_KINDS}")
