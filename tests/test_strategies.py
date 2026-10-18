from __future__ import annotations

from dataclasses import replace

import pytest

from reasoning_lab.llm.mock import MockLLM, MockSolution
from reasoning_lab.schema import EventKind, ReasoningTree, Step
from reasoning_lab.strategies import SearchConfig, StrategyContext, available, run_strategy
from reasoning_lab.strategies.base import BudgetExhausted
from reasoning_lab.strategies.chain import keep_order, mark_best_terminal
from reasoning_lab.verification.prm import build_prm
from reasoning_lab.verification.prm.base import NullPRM

ALL = available()

SMALL = SearchConfig(
    n_samples=4,
    branch_factor=2,
    max_depth=4,
    beam_width=2,
    mcts_iterations=6,
    max_llm_calls=40,
    temperature=0.8,
    seed=3,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ALL)
async def test_every_strategy_solves_and_records(name, problem, mock_llm):
    output = await run_strategy(name, problem, mock_llm, build_prm("symbolic"), SMALL)
    assert output.tree.strategy == name
    assert output.tree.metrics.llm_calls >= 1
    assert output.tree.nodes, "search produced an empty tree"
    # a single stochastic sample may land on the flawed chain; tree search may too
    assert output.answer in {problem.answer, "8"}, f"{name} answered {output.answer!r}"
    kinds = {e.kind for e in output.tree.events}
    assert EventKind.SEARCH_STARTED in kinds and EventKind.ANSWER_RESOLVED in kinds
    assert output.tree.metrics.verifications >= 1


@pytest.mark.asyncio
async def test_search_steps_are_verified_nodes(problem, mock_llm):
    output = await run_strategy("cot", problem, mock_llm, build_prm("symbolic"), SMALL)
    chain = output.tree.steps_to(output.tree.answer_node_id)
    assert len(chain) >= 2
    scored = [n for n in output.tree.path(output.tree.answer_node_id) if n.prm_score is not None]
    assert scored and all(0.0 <= n.prm_score <= 1.0 for n in scored)


@pytest.mark.asyncio
async def test_best_of_n_reranking_prefers_the_uncontradicted_chain(problem, mock_llm):
    output = await run_strategy("best_of_n", problem, mock_llm, build_prm("symbolic"), SMALL)
    ranking = output.extra["ranking"]
    assert ranking, "no candidate chains were ranked"
    chosen = next(node for node in output.tree.nodes.values() if node.id == ranking[0]["node_id"])
    assert chosen.final_answer == problem.answer
    assert not any(c.status.value == "fail" for n in output.tree.path(chosen.id) if n.verification for c in n.verification.checks)


@pytest.mark.asyncio
async def test_self_consistency_reports_votes(problem, mock_llm):
    output = await run_strategy("self_consistency", problem, mock_llm, NullPRM(), SMALL)
    assert sum(output.votes.values()) >= 1
    assert output.answer in output.votes


@pytest.mark.asyncio
async def test_tree_search_prunes_outside_beam(problem, mock_llm):
    output = await run_strategy("tot", problem, mock_llm, build_prm("symbolic"), SMALL)
    pruned = [n for n in output.tree.nodes.values() if n.state.value == "pruned"]
    assert output.tree.metrics.nodes_pruned == len(pruned)
    assert any(n.meta.get("prm_source") for n in output.tree.nodes.values())


@pytest.mark.asyncio
async def test_mcts_records_visits_and_rollouts(problem, mock_llm):
    output = await run_strategy("mcts", problem, mock_llm, build_prm("symbolic"), SMALL)
    assert output.tree.extra["iterations"] >= 1
    assert any(n.visits > 0 for n in output.tree.nodes.values() if n.id != output.tree.root_id)
    rollouts = [n for n in output.tree.nodes.values() if n.meta.get("rollout")]
    assert all(n.state.value != "terminal" or n.final_answer for n in rollouts)


@pytest.mark.asyncio
async def test_null_prm_still_searches(problem, mock_llm):
    output = await run_strategy("mcts", problem, mock_llm, NullPRM(), SMALL)
    assert output.answer is not None


@pytest.mark.asyncio
async def test_budget_is_respected(problem, mock_llm):
    tight = replace(SMALL, max_llm_calls=1)
    output = await run_strategy("tot", problem, mock_llm, build_prm("symbolic"), tight)
    assert output.tree.metrics.llm_calls <= 1


@pytest.mark.asyncio
async def test_generate_raises_when_out_of_budget(problem, mock_llm):
    ctx = StrategyContext(problem=problem, llm=mock_llm, prm=NullPRM(), config=SMALL)
    ctx.start_tree("test")
    ctx.calls = ctx.config.max_llm_calls
    from reasoning_lab.prompts import templates

    with pytest.raises(BudgetExhausted):
        await ctx.generate(templates.cot(problem))


def test_context_hides_gold_from_strategies(problem):
    ctx = StrategyContext(problem=problem, llm=None, prm=NullPRM(), config=SMALL)  # type: ignore[arg-type]
    assert ctx.problem.answer == ""
    assert ctx.problem.steps == []
    assert ctx.problem.solution is None
    assert ctx.problem.question == problem.question


@pytest.mark.asyncio
async def test_events_carry_valid_node_ids(problem, mock_llm):
    node_ids: list[str] = []
    kinds: list[str] = []

    def sink(event) -> None:
        kinds.append(str(event.kind))
        if event.node_id:
            node_ids.append(event.node_id)

    output = await run_strategy("tot", problem, mock_llm, build_prm("symbolic"), SMALL, sink=sink)
    assert "node_added" in kinds and "node_scored" in kinds
    assert node_ids, "no event referenced a node"
    assert all(node_id in output.tree.nodes for node_id in node_ids)


@pytest.mark.asyncio
async def test_majority_voting_recovers_answer_when_flawed_chains_are_minority(problem):
    """Deterministic 60/40 split: voting must land on the correct answer."""
    solution = MockSolution(
        match="Mia buys all boxes",
        steps=list(problem.steps),
        answer=problem.answer,
        flawed=[["Total packed: 3*4=13.", "Mia eats 5, so 13-5=8.", "Final Answer: 8"]],
    )

    def responder(prompt, params, sample_index):
        # even sample seeds produce the correct chain, odd seeds the flawed one
        return solution.chain_text() if (params.seed or 0) % 2 else solution.chain_text(solution.flawed[0])

    balanced = MockLLM([solution], responder=responder)
    config = replace(SMALL, n_samples=5, seed=3)
    voting = await run_strategy("self_consistency", problem, balanced, NullPRM(), config)
    assert voting.answer == problem.answer
    assert set(voting.votes) == {problem.answer, "8"}

    single = await run_strategy("cot", problem, balanced, NullPRM(), replace(config, n_samples=1))
    assert single.answer == problem.answer


@pytest.mark.asyncio
async def test_unknown_strategy_error_lists_options(problem, mock_llm):
    with pytest.raises(ValueError, match="available"):
        await run_strategy("nope", problem, mock_llm, NullPRM(), SMALL)


def test_tied_candidates_select_by_content_not_node_id():
    """Node ids are random per run, so a score tie must not be broken by id.

    Selection order decided ~5% of best-of-n answers on the bundled set, which
    made an identical rerun report a different accuracy.
    """
    tree = ReasoningTree(problem_id="t", strategy="cot")
    first = tree.add_node("root", Step(index=0, text="2 + 2 = 4."))
    second = tree.add_node("root", Step(index=0, text="1 + 1 = 2."))
    first.branch_score = second.branch_score = 0.8
    first.final_answer = second.final_answer = "4"
    assert keep_order([first, second])[0].id == second.id
    assert keep_order([second, first])[0].id == second.id
    assert mark_best_terminal([first, second]).id == second.id


def test_problem_gold_stripping_is_idempotent(problem):
    stripped = problem.model_copy(update={"answer": "", "steps": [], "solution": None, "flawed": []})
    ctx = StrategyContext(problem=stripped, llm=None, prm=NullPRM())  # type: ignore[arg-type]
    assert ctx.problem is stripped
