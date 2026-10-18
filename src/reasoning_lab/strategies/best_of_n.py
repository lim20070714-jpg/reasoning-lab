"""Best-of-N with process reward reranking."""

from __future__ import annotations

from ..schema import EventKind, Problem, StrategyOutput
from .base import ReasoningStrategy, StrategyContext, terminal_nodes, vote_answers
from .chain import keep_order, sample_chains


class BestOfN(ReasoningStrategy):
    name = "best_of_n"
    description = "Sample N chains, then select the one whose verified steps score highest."
    stochastic = True

    async def solve(self, problem: Problem, ctx: StrategyContext) -> StrategyOutput:
        tree = ctx.start_tree(self.name)
        await sample_chains(ctx, ctx.config.n_samples)
        answered = keep_order([n for n in terminal_nodes(tree) if n.final_answer])
        best = answered[0] if answered else None
        _, votes = vote_answers(list(answered))
        answer = best.final_answer if best else None

        tree.answer = answer
        tree.answer_node_id = best.id if best else tree.root_id
        ctx.emit(
            EventKind.ANSWER_RESOLVED,
            tree.answer_node_id,
            {"answer": answer, "selected_branch_score": best.branch_score if best else None, "votes": votes},
        )
        return StrategyOutput(
            tree=tree,
            answer=answer,
            rationale_node_id=tree.answer_node_id,
            votes=votes,
            extra={
                "ranking": [
                    {"node_id": n.id, "answer": n.final_answer, "branch_score": n.branch_score} for n in answered
                ]
            },
        )
