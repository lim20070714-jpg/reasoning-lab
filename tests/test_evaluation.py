"""Tests for dataset loading, metrics, the grid runner and report rendering."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from reasoning_lab.evaluation.dataset import describe, load_problems, mock_solutions
from reasoning_lab.evaluation.metrics import (
    EvalRecord,
    auc,
    expected_calibration_error,
    rerank_table,
    select_answer,
    summarise,
    verifier_discrimination,
)
from reasoning_lab.evaluation.report import build_report
from reasoning_lab.evaluation.runner import GridSpec, run_grid
from reasoning_lab.llm.mock import MockLLM
from reasoning_lab.schema import Problem, RunMetrics, ScoreCard, VerifyStatus
from reasoning_lab.strategies.base import SearchConfig

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture(scope="module")
def problems() -> list[Problem]:
    return load_problems(data_dir=DATA)


def test_dataset_loads_and_is_balanced(problems):
    stats = describe(problems)
    assert stats["problems"] == len(problems)
    assert stats["with_reference_steps"] == len(problems), "every problem needs a reference chain"
    assert stats["with_flawed"] == len(problems), "every problem needs a negative for the PRM"
    assert set(stats["by_subject"]) == {"math", "physics"}
    assert len(stats["by_level"]) >= 12


def test_dataset_ids_unique_and_answers_are_pure_math(problems):
    from reasoning_lab.verification import math_text as mt

    assert len({p.id for p in problems}) == len(problems)
    for problem in problems:
        assert problem.answer == problem.answer.strip() and problem.answer
        assert mt.parse_math(problem.answer) is not None, f"answer carries prose or units: {problem.answer!r}"


def test_verifier_separates_reference_from_flawed(problems):
    stats = verifier_discrimination(problems)
    assert stats["reference_false_flag_pct"] == 0.0, "reference steps must never be rejected"
    assert stats["variant_detection_pct"] == 100.0, "each flawed chain must contain a rejected step"
    assert stats["decidable_pct"] > 60.0
    assert (stats["chain_auc"] or 0) > 0.7
    assert (stats["step_auc"] or 0) > 0.6


def test_mock_solutions_cover_every_problem(problems):
    bank = mock_solutions(problems)
    assert len(bank) == len(problems)
    assert all(entry["steps"] and entry["answer"] for entry in bank)


def test_auc_matches_manual_computation():
    assert auc([0.9, 0.8, 0.2], [True, True, False]) == 1.0
    assert auc([0.1, 0.9], [True, False]) == 0.0
    assert auc([0.5, 0.5], [True, True]) is None


def test_ece_stays_small_for_well_calibrated_confidence():
    records = [_toy_record(conf, conf >= 0.5) for conf in (0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 1.0)]
    out = expected_calibration_error(records, bins=5)
    assert out["n"] == 8
    overconfident = [_toy_record(0.95, c) for c in (True, True, True, False, False, False)]
    assert out["ece"] < expected_calibration_error(overconfident, bins=5)["ece"]


def _toy_record(confidence: float, correct: bool) -> EvalRecord:
    card = ScoreCard(
        problem_id="x",
        strategy="cot",
        answer="1" if correct else "2",
        gold="1",
        correct=correct,
        confidence=confidence,
        verify_status=VerifyStatus.PASS,
        metrics=RunMetrics(),
    )
    return EvalRecord(card=card)


def test_select_answer_policies_disagree():
    from reasoning_lab.evaluation.metrics import Candidate

    pool = [
        Candidate(node_id="a", answer="7", branch_score=0.3, correct=False),
        Candidate(node_id="b", answer="7", branch_score=0.4, correct=False),
        Candidate(node_id="c", answer="8", branch_score=0.9, correct=True),
    ]
    assert select_answer(pool, "first") == "7"
    assert select_answer(pool, "vote") == "7"
    assert select_answer(pool, "prm_mean") == "8"
    assert select_answer(pool, "prm_max") == "8"
    assert select_answer(pool, "oracle") == "8"
    assert select_answer([], "vote") is None


def test_rerank_table_reports_prm_gain_over_first():
    from reasoning_lab.evaluation.metrics import Candidate

    good = EvalRecord(
        card=ScoreCard(problem_id="a", strategy="best_of_n", answer="8", gold="8", correct=True),
        candidates=[
            Candidate(node_id="1", answer="7", branch_score=0.3),
            Candidate(node_id="2", answer="8", branch_score=0.9, correct=True),
        ],
    )
    table = rerank_table([good])
    assert table["first"]["accuracy"] == 0.0
    assert table["prm_mean"]["accuracy"] == 1.0
    assert table["oracle"]["accuracy"] == 1.0


@pytest.mark.asyncio
async def test_run_grid_writes_records_and_resumes(tmp_path, problems):
    subset = problems[:2]
    llm = MockLLM(mock_solutions(subset))
    config = replace(SearchConfig(), n_samples=4, max_llm_calls=24, max_depth=4, branch_factor=2)
    spec = GridSpec(strategies=("cot", "self_consistency"), prm="symbolic", concurrency=1, save_traces=True)
    result = await run_grid(subset, llm=llm, config=config, spec=spec, run_dir=tmp_path, run_id="unit")
    assert result.succeeded == 4
    assert not result.errors
    directory = tmp_path / "unit"
    lines = (directory / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    traces = (directory / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(traces) == 4
    payload = json.loads(traces[0])
    assert payload["tree"]["nodes"], "trace must carry the search tree for replay"
    assert (directory / "config.json").is_file()

    resumed = await run_grid(subset, llm=llm, config=config, spec=spec, run_dir=tmp_path, run_id="unit")
    assert resumed.skipped == 4, "a second pass must reuse the checkpoint"
    assert len(resumed.records) == 4
    await llm.aclose()


@pytest.mark.asyncio
async def test_oracle_is_an_upper_bound_of_accuracy(tmp_path, problems):
    subset = problems[:6]
    llm = MockLLM(mock_solutions(subset))
    result = await run_grid(
        subset,
        llm=llm,
        config=replace(SearchConfig(), n_samples=4, max_llm_calls=24, max_depth=5, branch_factor=2),
        spec=GridSpec(strategies=("self_consistency", "best_of_n"), save_traces=False, concurrency=2),
        run_dir=tmp_path,
        run_id="bound",
    )
    assert result.succeeded == 12
    summary = summarise(result.records)
    assert summary["accuracy"] <= summary["oracle_any_pct"] / 100.0 + 1e-9
    assert summary["rerank"]["prm_mean"]["n"] <= summary["n"]
    await llm.aclose()


def test_report_renders_all_sections(problems):
    records = [
        _toy_record(0.9, True),
        _toy_record(0.4, False),
    ]
    text = build_report(records, problems=problems[:4], metadata={"run_id": "unit-run"})
    assert "# Reasoning strategy evaluation" in text
    assert "## Strategies" in text
    assert "## Dataset" in text
    assert "unit-run" in text
    assert "NaN" not in text


def test_summary_counts_are_consistent():
    records = [_toy_record(0.8, True), _toy_record(0.3, False)]
    summary = summarise(records)
    assert summary["n"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["calibration"]["n"] == 2
