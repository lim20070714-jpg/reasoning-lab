"""Self-Consistency: sample N chains and take the majority final answer."""

from __future__ import annotations

from collections import defaultdict

from ..schema import EventKind, Problem, StrategyOutput
from .base import ReasoningStrategy, StrategyContext, terminal_nodes, vote_answers
from .chain import keep_order, sample_chains


class SelfConsistency(ReasoningStrategy):
    name = "self_consistency"
    description = "Independent chains plus majority voting over final answers."
    stochastic = True

    async def solve(self, problem: Problem, ctx: StrategyContext) -> StrategyOutput:
        tree = ctx.start_tree(self.name)
        await sample_chains(ctx, ctx.config.n_samples)
        answered = terminal_nodes(tree)
        answer, votes = vote_answers(answered)

        if ctx.config.sc_selector == "vote_prm" and answered:
            # Weight each distinct answer by the process reward of the chains that
            # produced it, so a 4-3 split can be overturned by step quality.
            mass: dict[str, float] = defaultdict(float)
            for node in answered:
                key = (node.final_answer or "").strip()
                if key:
                    mass[key] += node.branch_score or 0.0
            if mass:
                answer = sorted(mass.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        representative = next(
            (
                n
                for n in keep_order(answered)
                if (n.final_answer or "").strip() == (answer or "").strip()
            ),
            None,
        )
        tree.answer = answer
        tree.answer_node_id = representative.id if representative else tree.root_id
        ctx.emit(
            EventKind.ANSWER_RESOLVED,
            tree.answer_node_id,
            {"answer": answer, "votes": votes, "selector": ctx.config.sc_selector},
        )
        return StrategyOutput(
            tree=tree,
            answer=answer,
            rationale_node_id=tree.answer_node_id,
            votes=votes,
            extra={"selector": ctx.config.sc_selector, "n_answered": len(answered)},
        )
