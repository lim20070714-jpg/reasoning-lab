"""Benchmark loading, grid running, metrics and report rendering."""

from __future__ import annotations

from .dataset import describe, load_problems, mock_solutions, read_jsonl
from .metrics import (
    Candidate,
    EvalRecord,
    auc,
    discrimination_stats,
    expected_calibration_error,
    group,
    record_from_output,
    rerank_table,
    select_answer,
    summarise,
    verifier_discrimination,
)
from .report import build_report, load_records, load_run_metadata, write_report
from .runner import GridJob, GridSpec, RunResult, default_run_id, run_grid

__all__ = [
    "Candidate",
    "EvalRecord",
    "GridJob",
    "GridSpec",
    "RunResult",
    "auc",
    "build_report",
    "default_run_id",
    "describe",
    "discrimination_stats",
    "expected_calibration_error",
    "group",
    "load_problems",
    "load_records",
    "load_run_metadata",
    "mock_solutions",
    "read_jsonl",
    "record_from_output",
    "rerank_table",
    "run_grid",
    "select_answer",
    "summarise",
    "verifier_discrimination",
    "write_report",
]
