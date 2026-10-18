"""Step proposal and chain completion shared by ToT, MCTS and CoT fallbacks."""

from __future__ import annotations

import re

from ..parser import explicit_final_answer, split_steps
from ..prompts import templates
from ..schema import EventKind, NodeState, SearchNode, Step
from .base import (
    BudgetExhausted,
    StrategyContext,
    build_verified_chain,
    gather_limited,
    node_is_dead,
    score_and_attach,
)

CANDIDATE_RE = re.compile(r"^\s*(?:\d+\s*[\.:\)]|[-*•])\s*(.+)$")


def parse_candidates(text: str, k: int) -> list[str]:
    """Read the numbered alternatives out of a batch proposal prompt."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: list[str] = []
    for line in lines:
        match = CANDIDATE_RE.match(line)
        candidate = (match.group(1) if match else line).strip()
        if len(candidate) < 3:
            continue
        if out and candidate.lower() == out[-1].lower():
            continue
        out.append(candidate)
        if len(out) >= k:
            return out
    if out:
        return out
    return [s.text for s in split_steps(text) if s.text.strip()][:k]


async def propose_children(
    ctx: StrategyContext, parent_id: str, *, k: int | None = None, meta: dict[str, object] | None = None
) -> list[SearchNode]:
    """Generate ``k`` candidate next steps for a node and attach them, verified."""
    config = ctx.config
    tree = ctx.require_tree()
    node_count = k or config.branch_factor
    if node_is_dead(tree, parent_id):
        return []
    steps = tree.steps_to(parent_id)
    created: list[SearchNode] = []

    if config.propose_mode == "batch" and node_count > 1:
        await ctx.checkpoint(parent_id)
        prompt = templates.chain_step_candidates(ctx.problem, steps, node_count)
        completion = await ctx.generate(prompt, tag="step_candidates")
        candidates = parse_candidates(completion.text, node_count)
        for candidate in candidates:
            created.append(await attach_candidate(ctx, parent_id, candidate, len(steps), meta=meta))
        return created

    for _ in range(node_count):
        await ctx.checkpoint(parent_id)
        prompt = templates.chain_step(ctx.problem, steps)
        completion = await ctx.generate(prompt, tag="step")
        body = (completion.text or "").strip()
        first = split_steps(body)
        candidate = first[0].text if first else body
        if candidate:
            created.append(await attach_candidate(ctx, parent_id, candidate, len(steps), meta=meta))
    return created


async def attach_candidate(
    ctx: StrategyContext, parent_id: str, text: str, depth: int, *, meta: dict[str, object] | None = None
) -> SearchNode:
    step = Step(index=depth, text=text)
    answer = explicit_final_answer(text)
    if answer:
        step.answer = answer
    node = await score_and_attach(ctx, parent_id=parent_id, step=step, meta=dict(meta or {}))
    if answer:
        node.final_answer = answer
        node.terminal = True
        node.state = NodeState.TERMINAL
        ctx.emit(EventKind.NODE_TERMINAL, node.id, {"answer": answer, "depth": depth})
    return node


def prune(node: SearchNode, ctx: StrategyContext, reason: str) -> None:
    if node.state == NodeState.TERMINAL:
        return
    node.state = NodeState.PRUNED
    ctx.require_tree().metrics.nodes_pruned += 1
    ctx.emit(EventKind.NODE_PRUNED, node.id, {"reason": reason, "branch_score": node.branch_score})


def tie_break(node: SearchNode) -> tuple[str, str, str]:
    """Content-derived tail for selection keys.

    Node ids are random per run, so ordering by id makes a rerun pick a
    different candidate whenever two tie on score. Step text and answer are
    stable, and candidates that agree on both are equivalent to select on.
    """
    return (node.text, node.final_answer or "", node.id)


def keep_order(nodes: list[SearchNode]) -> list[SearchNode]:
    """Highest branch score first, ties broken on chain content."""
    return sorted(nodes, key=lambda n: (-(n.branch_score or 0.0), *tie_break(n)))


async def extend_until_answer(
    ctx: StrategyContext,
    parent_id: str,
    *,
    max_extra_steps: int,
    rollout: bool = False,
) -> SearchNode:
    """Continue a branch until it emits a final answer or runs out of room."""
    tree = ctx.require_tree()
    current = tree.nodes[parent_id]
    for _ in range(max_extra_steps):
        if current.terminal or node_is_dead(tree, current.id) or ctx.budget_left <= 0:
            break
        created = await propose_children(
            ctx, current.id, k=1, meta={"rollout": True} if rollout else None
        )
        if not created:
            break
        current = created[0]
        if current.branch_score is not None and current.branch_score < ctx.config.prune_below:
            prune(current, ctx, "low branch score during rollout")
            break
    return current


def mark_best_terminal(nodes: list[SearchNode]) -> SearchNode | None:
    """Same ordering rule as ``keep_order``, so both pick the same candidate."""
    terminals = [n for n in nodes if n.final_answer]
    return keep_order(terminals)[0] if terminals else None


async def sample_chains(
    ctx: StrategyContext, n: int, *, concurrency: int = 4
) -> list[SearchNode | None]:
    """Draw ``n`` independent full chains, each verified as it is appended.

    Seeds are derived from the sample index rather than call order so a rerun of
    the same configuration reproduces the same sample set.
    """
    prompt = templates.cot(ctx.problem, few_shot=ctx.config.few_shot)

    async def one(index: int) -> SearchNode | None:
        try:
            completion = await ctx.generate(prompt, seed=ctx.config.seed + index, tag=f"sample_{index}")
        except BudgetExhausted:
            return None
        return await build_verified_chain(ctx, completion.text, sample_index=index)

    return await gather_limited([one(i) for i in range(n)], concurrency)
