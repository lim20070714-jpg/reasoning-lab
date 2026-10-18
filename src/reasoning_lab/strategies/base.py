"""Strategy interface, search configuration and shared tree bookkeeping.

Every strategy obeys the same contract: it owns a :class:`ReasoningTree`, spends
model calls through the budget in :class:`SearchConfig`, scores each produced
step with the configured process reward model, and emits events as it goes so
the same run can be streamed live to the UI or replayed from a trace file.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ..llm.base import Completion, LLMClient, Prompt
from ..parser import extract_final_answer, split_steps
from ..schema import (
    CheckResult,
    EventKind,
    GenerationParams,
    NodeState,
    Problem,
    ReasoningTree,
    SearchNode,
    Step,
    StrategyOutput,
    TraceEvent,
    Verification,
    VerifyStatus,
)
from ..verification.prm.base import ProcessRewardModel, StepScore

Sink = Callable[[TraceEvent], None]


class BudgetExhausted(RuntimeError):
    pass


class Gate(Protocol):
    """Human-in-the-loop checkpoint consulted before each expansion."""

    async def checkpoint(self, node_id: str | None) -> None: ...


@dataclass
class SearchConfig:
    n_samples: int = 8
    branch_factor: int = 3
    max_depth: int = 8
    beam_width: int = 2
    temperature: float = 0.7
    top_p: float | None = None
    max_tokens: int = 768
    max_llm_calls: int = 120
    prune_below: float = 0.02
    answer_confidence_floor: float = 0.35
    propose_mode: Literal["batch", "individual"] = "batch"
    sc_selector: Literal["vote", "vote_prm"] = "vote"
    tot_mode: Literal["beam", "dfs"] = "beam"
    mcts_value: Literal["prm", "hybrid"] = "prm"
    mcts_final: Literal["visits", "value"] = "visits"
    few_shot: bool = False
    seed: int = 0
    mcts_iterations: int = 24
    mcts_c: float = 1.4
    mcts_rollout_depth: int = 5
    mcts_prior_weight: float = 0.5
    mcts_expand_children: int = 2

    def params(
        self, *, temperature: float | None = None, seed: int | None = None, max_tokens: int | None = None
    ) -> GenerationParams:
        return GenerationParams(
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p,
            max_tokens=max_tokens or self.max_tokens,
            seed=self.seed if seed is None else seed,
        )


@dataclass
class StrategyContext:
    """Per-run workspace shared by a strategy, the model client and the PRM."""

    problem: Problem
    llm: LLMClient
    prm: ProcessRewardModel
    config: SearchConfig = field(default_factory=SearchConfig)
    sink: Sink | None = None
    gate: Gate | None = None
    #: handed the tree as soon as search starts, so a live viewer can read it
    on_tree: Callable[[ReasoningTree], None] | None = None
    tree: ReasoningTree | None = None
    calls: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Strip gold answers and reference solutions from the strategy's view.

        Search must never see the label it is later scored on; oracle-style
        upper bounds are recomputed from traces in ``evaluation.metrics``.
        """
        if self.problem.answer or self.problem.solution or self.problem.steps or self.problem.flawed:
            self.problem = self.problem.model_copy(
                update={"answer": "", "solution": None, "steps": [], "flawed": []}
            )

    def start_tree(self, strategy: str) -> ReasoningTree:
        tree = ReasoningTree(problem_id=self.problem.id, strategy=strategy, sink=self.sink)
        self.tree = tree
        if self.on_tree is not None:
            self.on_tree(tree)
        tree.extra.update(self.extra)
        self.emit(EventKind.SEARCH_STARTED, tree.root_id, {"strategy": strategy, "budget": self.config.max_llm_calls})
        return tree

    def require_tree(self) -> ReasoningTree:
        if self.tree is None:
            raise RuntimeError("call start_tree() before using the context")
        return self.tree

    def emit(self, kind: EventKind, node_id: str | None, payload: dict[str, Any] | None = None) -> None:
        if self.tree is None:
            return
        self.tree.emit(kind, node_id, payload or {})

    @property
    def budget_left(self) -> int:
        return self.config.max_llm_calls - self.calls

    async def generate(
        self,
        prompt: Prompt,
        *,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
        tag: str = "",
    ) -> Completion:
        if self.budget_left <= 0:
            raise BudgetExhausted(f"hit max_llm_calls={self.config.max_llm_calls}")
        params = self.config.params(temperature=temperature, seed=seed, max_tokens=max_tokens)
        if seed is None:
            params = params.model_copy(update={"seed": self.config.seed + self.calls})
        self.calls += 1
        tree = self.require_tree()
        tree.metrics.llm_calls += 1
        completion = await self.llm.generate(prompt, params, tag=tag or prompt.tag)
        tree.metrics.cached_calls += int(completion.cached)
        tree.metrics.prompt_tokens += completion.prompt_tokens
        tree.metrics.completion_tokens += completion.completion_tokens
        tree.metrics.latency_ms += completion.latency_ms
        self.emit(
            EventKind.GENERATION_FINISHED,
            None,
            {"tag": completion.tag, "tokens": completion.total_tokens, "cached": completion.cached},
        )
        return completion

    async def sample(self, prompt: Prompt, n: int, **kwargs: Any) -> list[Completion]:
        """``n`` independent samples of one prompt, budget-aware."""
        return [await self.generate(prompt, **kwargs) for _ in range(min(n, max(0, self.budget_left)))]

    async def score(self, prefix: list[Step], step: Step, node_id: str | None = None) -> StepScore:
        score = await self.prm.score_step(self.problem.id, prefix, step)
        tree = self.require_tree()
        tree.metrics.verifications += 1
        if score.status == VerifyStatus.FAIL:
            tree.metrics.hard_failures += 1
        self.emit(
            EventKind.NODE_SCORED,
            node_id,
            {"score": score.score, "status": str(score.status), "source": score.source},
        )
        return score

    async def checkpoint(self, node_id: str | None = None) -> None:
        """Park until a human resume arrives; a pruned node aborts the branch."""
        if self.gate is not None:
            await self.gate.checkpoint(node_id)


# -- shared tree helpers ------------------------------------------------------
async def score_and_attach(
    ctx: StrategyContext,
    *,
    parent_id: str,
    step: Step,
    sample_index: int | None = None,
    meta: dict[str, Any] | None = None,
) -> SearchNode:
    """Create a node for ``step``, verify it, and record its process reward."""
    tree = ctx.require_tree()
    prefix = tree.steps_to(parent_id)
    score = await ctx.score(prefix, step)
    node = tree.add_node(
        parent_id,
        step,
        sample_index=sample_index,
        meta={**(meta or {}), "prm_source": score.source},
    )
    node.prm_score = score.score
    node.verification = Verification(
        status=score.status,
        score=score.score,
        checks=[c if isinstance(c, CheckResult) else CheckResult.model_validate(c) for c in score.checks],
        verifier=score.source,
        notes=score.detail,
    )
    if step.answer:
        node.final_answer = step.answer
    node.branch_score = path_value(ctx, node.id)
    ctx.emit(EventKind.NODE_SCORED, node.id, {"branch_score": node.branch_score, "status": str(score.status)})
    return node


def path_scores(tree: ReasoningTree, node_id: str) -> list[StepScore]:
    return [
        StepScore(score=n.verification.score, status=n.verification.status)
        for n in tree.path(node_id)
        if n.verification is not None
    ]


def path_value(ctx: StrategyContext, node_id: str) -> float:
    tree = ctx.require_tree()
    scores = path_scores(tree, node_id)
    return ctx.prm.path_value(scores) if scores else 0.5


async def build_verified_chain(
    ctx: StrategyContext,
    text: str,
    *,
    sample_index: int | None = None,
    parent_id: str | None = None,
) -> SearchNode | None:
    """Append a parsed chain under ``parent_id``, verifying every step."""
    tree = ctx.require_tree()
    steps = split_steps(text)
    answer = extract_final_answer(text)
    parent = parent_id or tree.root_id
    last: SearchNode | None = None
    for i, step in enumerate(steps):
        if i == len(steps) - 1 and answer and not step.answer:
            step.answer = answer
        last = await score_and_attach(ctx, parent_id=parent, step=step, sample_index=sample_index)
        parent = last.id
    if last is not None:
        last.final_answer = answer
        last.terminal = bool(answer)
        last.state = NodeState.TERMINAL if answer else NodeState.EXPANDED
        ctx.emit(EventKind.NODE_TERMINAL, last.id, {"answer": answer, "sample_index": sample_index})
    return last


def terminal_nodes(tree: ReasoningTree) -> list[SearchNode]:
    return [n for n in tree.nodes.values() if n.final_answer and n.terminal]


def vote_answers(nodes: list[SearchNode]) -> tuple[str | None, dict[str, int]]:
    """Deterministic majority vote over final answers; ties break on the string."""
    tally: dict[str, int] = {}
    for node in nodes:
        key = (node.final_answer or "").strip()
        if key:
            tally[key] = tally.get(key, 0) + 1
    if not tally:
        return None, {}
    best = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return best[0], tally


async def gather_limited(coros: list[Coroutine[Any, Any, Any]], limit: int) -> list[Any]:
    semaphore = asyncio.Semaphore(max(1, limit))

    async def run(coro: Awaitable[Any]) -> Any:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(run(c) for c in coros))


class ReasoningStrategy(ABC):
    """Base class implemented by every strategy in the registry."""

    name: str = "base"
    description: str = ""
    stochastic: bool = True

    @abstractmethod
    async def solve(self, problem: Problem, ctx: StrategyContext) -> StrategyOutput: ...

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "stochastic": self.stochastic}


def node_is_dead(tree: ReasoningTree, node_id: str) -> bool:
    """A human may prune any open branch; search must respect it."""
    node = tree.nodes.get(node_id)
    return node is None or node.state in (NodeState.PRUNED, NodeState.FAILED)


__all__ = [
    "BudgetExhausted",
    "Gate",
    "ReasoningStrategy",
    "SearchConfig",
    "SearchNode",
    "Step",
    "StrategyContext",
    "StrategyOutput",
    "build_verified_chain",
    "gather_limited",
    "node_is_dead",
    "path_scores",
    "path_value",
    "score_and_attach",
    "terminal_nodes",
    "vote_answers",
]
