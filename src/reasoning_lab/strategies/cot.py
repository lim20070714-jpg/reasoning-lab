"""Chain-of-Thought: one sampled reasoning chain, verified step by step."""

from __future__ import annotations

from ..prompts import templates
from ..schema import EventKind, NodeState, Problem, StrategyOutput
from .base import BudgetExhausted, ReasoningStrategy, StrategyContext, build_verified_chain
from .chain import extend_until_answer


class ChainOfThought(ReasoningStrategy):
    name = "cot"
    description = "Single sampled chain of thought; steps are verified by the configured PRM."
    stochastic = True

    async def solve(self, problem: Problem, ctx: StrategyContext) -> StrategyOutput:
        tree = ctx.start_tree(self.name)
        prompt = templates.cot(problem, few_shot=ctx.config.few_shot)
        answer: str | None = None
        last = None
        try:
            completion = await ctx.generate(prompt, tag="cot")
            last = await build_verified_chain(ctx, completion.text)
            answer = (last.final_answer if last else None) or None
        except BudgetExhausted:
            tree.extra["budget_exhausted"] = True

        if answer is None and last is not None:
            try:
                last = await extend_until_answer(
                    ctx, last.id if last.id != tree.root_id else tree.root_id, max_extra_steps=ctx.config.max_depth
                )
                answer = last.final_answer
            except BudgetExhausted:
                tree.extra["budget_exhausted"] = True

        if last is not None and answer:
            last.terminal = True
            last.state = NodeState.TERMINAL
            last.final_answer = answer
        tree.answer = answer
        tree.answer_node_id = last.id if last is not None else tree.root_id
        ctx.emit(EventKind.ANSWER_RESOLVED, tree.answer_node_id, {"answer": answer, "source": "cot"})
        return StrategyOutput(
            tree=tree,
            answer=answer,
            rationale_node_id=tree.answer_node_id,
            extra={"budget_exhausted": bool(tree.extra.get("budget_exhausted"))},
        )
