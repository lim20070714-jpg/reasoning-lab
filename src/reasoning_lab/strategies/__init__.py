"""Strategy registry shared by the CLI, the runner and the server."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..llm.base import LLMClient
from ..schema import Problem, StrategyOutput
from ..verification.prm.base import ProcessRewardModel
from .base import ReasoningStrategy, SearchConfig, StrategyContext
from .best_of_n import BestOfN
from .cot import ChainOfThought
from .mcts import MCTSSearch
from .self_consistency import SelfConsistency
from .tot import TreeOfThoughts

STRATEGIES: dict[str, ReasoningStrategy] = {
    cls.name: cls()
    for cls in (ChainOfThought, SelfConsistency, BestOfN, TreeOfThoughts, MCTSSearch)
}

ALIASES = {
    "chain_of_thought": "cot",
    "chain": "cot",
    "sc": "self_consistency",
    "self-consistency": "self_consistency",
    "bon": "best_of_n",
    "best-of-n": "best_of_n",
    "tree_of_thoughts": "tot",
    "tree-of-thoughts": "tot",
    "mcts_search": "mcts",
}

#: strategies whose behaviour is not seed-invariant, so reports sweep seeds
STOCHASTIC = ("cot", "self_consistency", "best_of_n", "mcts")


def resolve_name(name: str) -> str:
    key = name.strip().lower()
    return ALIASES.get(key, key)


def get_strategy(name: str) -> ReasoningStrategy:
    key = resolve_name(name)
    if key not in STRATEGIES:
        raise ValueError(f"unknown strategy '{name}'; available: {', '.join(available())}")
    return STRATEGIES[key]


def available() -> list[str]:
    return sorted(STRATEGIES)


def describe_all() -> list[dict[str, object]]:
    return [STRATEGIES[name].describe() for name in sorted(STRATEGIES)]


async def run_strategy(
    name: str,
    problem: Problem,
    llm: LLMClient,
    prm: ProcessRewardModel,
    config: SearchConfig | None = None,
    *,
    sink=None,
    gate=None,
    on_tree=None,
    extra: Mapping[str, Any] | None = None,
) -> StrategyOutput:
    """Single-problem entry point used by the server and the tests."""
    strategy = get_strategy(name)
    context = StrategyContext(
        problem=problem,
        llm=llm,
        prm=prm,
        config=config or SearchConfig(),
        sink=sink,
        gate=gate,
        on_tree=on_tree,
        extra=dict(extra or {}),
    )
    return await strategy.solve(problem, context)


__all__ = [
    "ALIASES",
    "STOCHASTIC",
    "BestOfN",
    "ChainOfThought",
    "MCTSSearch",
    "ReasoningStrategy",
    "SearchConfig",
    "SelfConsistency",
    "StrategyContext",
    "TreeOfThoughts",
    "available",
    "describe_all",
    "get_strategy",
    "resolve_name",
    "run_strategy",
]
