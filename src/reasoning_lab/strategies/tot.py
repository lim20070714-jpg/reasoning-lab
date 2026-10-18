"""Tree-of-Thoughts: breadth or depth search over single-step thoughts."""

from __future__ import annotations

from ..schema import EventKind, NodeState, Problem, SearchNode, StrategyOutput
from .base import ReasoningStrategy, StrategyContext, node_is_dead, vote_answers
from .chain import extend_until_answer, keep_order, propose_children, prune


class TreeOfThoughts(ReasoningStrategy):
    name = "tot"
    description = "Expand k next steps per state, score each with the PRM, keep the strongest beam."
    stochastic = False

    async def solve(self, problem: Problem, ctx: StrategyContext) -> StrategyOutput:
        tree = ctx.start_tree(self.name)
        terminals: list[SearchNode] = []

        if ctx.config.tot_mode == "dfs":
            await self._dfs(ctx, tree.root_id, 0, terminals)
        else:
            await self._beam(ctx, terminals)

        if not terminals:
            open_nodes = [n for n in tree.frontier() if not n.meta.get("rollout") and n.id != tree.root_id]
            for candidate in keep_order(open_nodes)[:1]:
                extended = await extend_until_answer(ctx, candidate.id, max_extra_steps=ctx.config.max_depth)
                if extended.final_answer:
                    terminals.append(extended)

        answered = [n for n in terminals if n.final_answer]
        best = keep_order(answered)[0] if answered else None
        _, votes = vote_answers(answered)
        tree.answer = best.final_answer if best else None
        tree.answer_node_id = best.id if best else tree.root_id
        tree.extra["terminal_count"] = len(answered)
        ctx.emit(EventKind.ANSWER_RESOLVED, tree.answer_node_id, {"answer": tree.answer, "mode": ctx.config.tot_mode})
        return StrategyOutput(
            tree=tree,
            answer=tree.answer,
            rationale_node_id=tree.answer_node_id,
            votes=votes,
            extra={"mode": ctx.config.tot_mode, "terminals": len(answered)},
        )

    async def _beam(self, ctx: StrategyContext, terminals: list[SearchNode]) -> None:
        tree = ctx.require_tree()
        frontier = [tree.root_id]
        for _ in range(ctx.config.max_depth):
            if not frontier or ctx.budget_left <= 0:
                return
            children: list[SearchNode] = []
            for parent_id in frontier:
                if node_is_dead(tree, parent_id):
                    continue
                children.extend(await propose_children(ctx, parent_id))
            if not children:
                return
            keep: list[SearchNode] = []
            for child in keep_order(children):
                if child.terminal and child.final_answer:
                    if (child.branch_score or 0.0) >= ctx.config.prune_below:
                        terminals.append(child)
                    continue
                if len(keep) < ctx.config.beam_width:
                    keep.append(child)
                else:
                    prune(child, ctx, "outside beam")
            if not keep:
                return
            frontier = [c.id for c in keep]

    async def _dfs(
        self, ctx: StrategyContext, node_id: str, depth: int, terminals: list[SearchNode]
    ) -> bool:
        tree = ctx.require_tree()
        if depth >= ctx.config.max_depth or ctx.budget_left <= 0 or node_is_dead(tree, node_id):
            return False
        children = await propose_children(ctx, node_id)
        for child in keep_order(children)[: ctx.config.beam_width]:
            if child.state in (NodeState.PRUNED, NodeState.FAILED):
                continue
            if (child.branch_score or 0.0) < ctx.config.prune_below:
                prune(child, ctx, "below value floor")
                continue
            if child.terminal and child.final_answer:
                terminals.append(child)
                if (child.branch_score or 0.0) >= ctx.config.answer_confidence_floor:
                    return True
                continue
            if await self._dfs(ctx, child.id, depth + 1, terminals):
                return True
        return False
