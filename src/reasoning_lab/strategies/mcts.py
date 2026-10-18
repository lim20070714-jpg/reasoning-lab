"""Monte Carlo Tree Search over verified reasoning steps.

Selection uses UCT with an additive process-reward prior, expansion adds real
tree nodes, and evaluation runs a depth-limited rollout whose nodes are tagged
``rollout`` so they inform values without being chosen as the answer path.
"""

from __future__ import annotations

import math

from ..schema import EventKind, NodeState, Problem, ReasoningTree, SearchNode, StrategyOutput
from .base import ReasoningStrategy, StrategyContext, node_is_dead, path_value, vote_answers
from .chain import extend_until_answer, keep_order, propose_children, tie_break


class MCTSSearch(ReasoningStrategy):
    name = "mcts"
    description = "UCT tree search with PRM priors and depth-limited rollouts."
    stochastic = True

    async def solve(self, problem: Problem, ctx: StrategyContext) -> StrategyOutput:
        tree = ctx.start_tree(self.name)
        completed = 0
        for _ in range(ctx.config.mcts_iterations):
            if ctx.budget_left <= 0:
                tree.extra["budget_exhausted"] = True
                break
            leaf = self._select(ctx)
            await ctx.checkpoint(leaf.id)
            children = await self._expand(ctx, leaf)
            target = keep_order(children)[0] if children else leaf
            value = await self._evaluate(ctx, target)
            self._backprop(ctx, target, value)
            completed += 1

        best = self._choose(ctx)
        candidates = [n for n in tree.nodes.values() if n.final_answer and not n.meta.get("rollout")] or [
            n for n in tree.nodes.values() if n.final_answer
        ]
        _, votes = vote_answers(candidates)
        tree.answer = best.final_answer if best else None
        tree.answer_node_id = best.id if best else tree.root_id
        tree.extra["iterations"] = completed
        ctx.emit(
            EventKind.ANSWER_RESOLVED,
            tree.answer_node_id,
            {"answer": tree.answer, "iterations": completed, "value_mode": ctx.config.mcts_value},
        )
        return StrategyOutput(
            tree=tree,
            answer=tree.answer,
            rationale_node_id=tree.answer_node_id,
            votes=votes,
            extra={"iterations": completed, "value_mode": ctx.config.mcts_value},
        )

    # -- selection ---------------------------------------------------
    def _searchable(self, tree: ReasoningTree, node: SearchNode) -> list[SearchNode]:
        return [
            tree.nodes[cid]
            for cid in node.children
            if cid in tree.nodes and not tree.nodes[cid].meta.get("rollout") and tree.nodes[cid].state != NodeState.PRUNED
        ]

    def _select(self, ctx: StrategyContext) -> SearchNode:
        tree = ctx.require_tree()
        current = tree.nodes[tree.root_id]
        while True:
            children = self._searchable(tree, current)
            if current.terminal or current.depth >= ctx.config.max_depth:
                return current
            if len(children) < ctx.config.branch_factor:
                return current
            unvisited = [c for c in children if c.visits == 0]
            if unvisited:
                return keep_order(unvisited)[0]
            return max(children, key=lambda c: (self._uct(ctx, c, current), *tie_break(c)))

    def _uct(self, ctx: StrategyContext, child: SearchNode, parent: SearchNode) -> float:
        if child.visits == 0:
            return math.inf
        exploit = child.value if child.value is not None else (child.branch_score or 0.0)
        explore = ctx.config.mcts_c * math.sqrt(math.log(max(1, parent.visits)) / child.visits)
        prior = ctx.config.mcts_prior_weight * (child.prm_score or 0.5)
        return exploit + explore + prior / max(1, child.visits) ** 0.25

    # -- expansion ---------------------------------------------------
    async def _expand(self, ctx: StrategyContext, leaf: SearchNode) -> list[SearchNode]:
        tree = ctx.require_tree()
        if leaf.terminal or leaf.depth >= ctx.config.max_depth or node_is_dead(tree, leaf.id):
            return []
        room = ctx.config.branch_factor - len(self._searchable(tree, leaf))
        if room <= 0:
            return []
        return await propose_children(ctx, leaf.id, k=min(ctx.config.mcts_expand_children, room))

    # -- evaluation --------------------------------------------------
    async def _evaluate(self, ctx: StrategyContext, node: SearchNode) -> float:
        if node.terminal and node.final_answer:
            base = node.branch_score if node.branch_score is not None else path_value(ctx, node.id)
        else:
            if ctx.budget_left <= 0:
                return node.branch_score or 0.0
            extended = await extend_until_answer(
                ctx, node.id, max_extra_steps=ctx.config.mcts_rollout_depth, rollout=True
            )
            base = extended.branch_score if extended.branch_score is not None else path_value(ctx, extended.id)

        if ctx.config.mcts_value == "hybrid":
            tree = ctx.require_tree()
            terminals = [n for n in tree.nodes.values() if n.final_answer]
            agreed = [n for n in terminals if n.final_answer == node.final_answer] if node.final_answer else []
            agreement = len(agreed) / len(terminals) if terminals else 0.5
            return 0.5 * base + 0.5 * agreement
        return base

    # -- backpropagation ---------------------------------------------
    def _backprop(self, ctx: StrategyContext, node: SearchNode, value: float) -> None:
        tree = ctx.require_tree()
        for ancestor in tree.path(node.id):
            if ancestor.id == tree.root_id:
                ancestor.visits += 1
                continue
            previous = ancestor.value if ancestor.value is not None else 0.0
            ancestor.visits += 1
            ancestor.value = round((previous * (ancestor.visits - 1) + value) / ancestor.visits, 6)
        tree.metrics.nodes_expanded += 1

    # -- final choice ------------------------------------------------
    def _choose(self, ctx: StrategyContext) -> SearchNode | None:
        tree = ctx.require_tree()
        pool = [
            n
            for n in tree.nodes.values()
            if n.final_answer and n.state != NodeState.PRUNED and not n.meta.get("rollout")
        ]
        if not pool:
            pool = [n for n in tree.nodes.values() if n.final_answer and n.state != NodeState.PRUNED]
        if not pool:
            return None
        if ctx.config.mcts_final == "value":
            return max(pool, key=lambda n: (n.value or 0.0, n.branch_score or 0.0, *tie_break(n)))
        return max(pool, key=lambda n: (n.visits, n.value or 0.0, n.branch_score or 0.0, *tie_break(n)))
