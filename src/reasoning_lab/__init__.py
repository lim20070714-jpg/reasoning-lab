"""reasoning-lab: unified LLM reasoning strategy experiments with PRM verification."""

from __future__ import annotations

from .config import LLMSettings, Settings
from .llm.base import Completion, LLMClient, LLMError, Prompt
from .schema import (
    GenerationParams,
    Problem,
    ReasoningTree,
    RunMetrics,
    ScoreCard,
    SearchNode,
    Step,
    StrategyOutput,
    Verification,
    VerifyStatus,
)

__version__ = "0.2.0"

__all__ = [
    "Completion",
    "GenerationParams",
    "LLMClient",
    "LLMError",
    "LLMSettings",
    "Problem",
    "Prompt",
    "ReasoningTree",
    "RunMetrics",
    "ScoreCard",
    "SearchNode",
    "Settings",
    "Step",
    "StrategyOutput",
    "Verification",
    "VerifyStatus",
    "__version__",
]
