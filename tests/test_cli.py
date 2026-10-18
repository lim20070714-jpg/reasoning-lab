"""Tests for the ``reasoning-lab`` command line surface.

Every case runs against the mock backend inside a temporary directory, so the
whole file stays offline, API-key free and leaves the repository untouched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from reasoning_lab.cli import _settings, build_parser, main

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

T1 = {
    "id": "t1",
    "question": "Evaluate 3*4 + 5*2 - 7.",
    "answer": "15",
    "subject": "math",
    "level": "arithmetic",
    "steps": ["3*4 = 12.", "5*2 = 10.", "12 + 10 = 22.", "22 - 7 = 15.", "x = 15."],
    "flawed": [["3*4 = 12.", "5*2 = 7.", "12 + 7 = 19.", "19 - 7 = 12.", "x = 12."]],
}
T2 = {
    "id": "t2",
    "question": "A 5 kg crate accelerates at 3 m/s^2. What is the net force, in newtons?",
    "answer": "15",
    "subject": "physics",
    "level": "mechanics",
    "steps": ["F = m*a.", "m = 5.", "a = 3.", "F = 5*3.", "F = 15."],
    "flawed": [["F = m*a.", "m = 5.", "a = 3.", "F = 5*3.", "F = 12."]],
}


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    """Commands write ``runs/`` relative to the cwd and read configuration from the environment.

    Both come from the developer's shell, so pin them per test instead of
    inheriting whatever a real evaluation run left behind.
    """
    for name in [key for key in os.environ if key.startswith("REASONING_LAB_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_dataset(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / "problems_math.jsonl").write_text(json.dumps(T1) + "\n", encoding="utf-8")
    (data / "problems_physics.jsonl").write_text(json.dumps(T2) + "\n", encoding="utf-8")
    return data


def json_blocks(text: str) -> list[dict]:
    """Split the concatenated pretty-printed objects that ``datasets`` emits."""
    decoder = json.JSONDecoder()
    out: list[dict] = []
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        value, cursor = decoder.raw_decode(text, cursor)
        out.append(value)
    return out


def test_strategies_command_prints_the_registry(capsys: CaptureFixture) -> None:
    assert main(["strategies"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {"cot", "self_consistency", "best_of_n", "tot", "mcts"} <= set(payload["available"])
    assert "symbolic" in payload["prm"]


def test_datasets_command_reports_coverage_and_discrimination(capsys: CaptureFixture) -> None:
    assert main(["datasets", "--data-dir", str(DATA_DIR)]) == 0
    stats, discrimination = json_blocks(capsys.readouterr().out)
    assert stats["problems"] == 40
    assert stats["by_subject"] == {"math": 24, "physics": 16}
    assert discrimination["reference_steps"] == 199
    assert discrimination["decidable_pct"] == pytest.approx(94.47, abs=0.05)


def test_run_command_solves_an_ad_hoc_question(capsys: CaptureFixture, tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    code = main(
        [
            "run",
            "--mock",
            "--prm",
            "symbolic",
            "--strategy",
            "cot",
            "--question",
            T1["question"],
            "--answer",
            "15",
            "--trace",
            str(trace),
        ]
    )
    assert code == 0
    assert "strategy=cot" in capsys.readouterr().out
    payload = json.loads(trace.read_text(encoding="utf-8"))
    assert payload["tree"]["strategy"] == "cot"
    assert payload["events"], "the sink must record the tree events"


def test_run_command_exits_two_without_a_problem(capsys: CaptureFixture) -> None:
    assert main(["run", "--mock", "--ids", "does-not-exist"]) == 2
    assert "provide --question" in capsys.readouterr().err


def test_eval_writes_records_traces_and_report(capsys: CaptureFixture, tmp_path: Path) -> None:
    data = write_dataset(tmp_path)
    code = main(
        [
            "eval",
            "--mock",
            "--prm",
            "symbolic",
            "--data-dir",
            str(data),
            "--strategies",
            "cot,self_consistency",
            "--n-samples",
            "3",
            "--max-calls",
            "20",
            "--run-id",
            "cli-test",
            "--no-appendix",
        ]
    )
    assert code == 0
    run = tmp_path / "runs" / "cli-test"
    records = (run / "records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(records) == 4, "2 strategies x 2 problems"
    report = (run / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# Reasoning strategy evaluation")
    assert "## Strategies" in report
    assert (run / "traces.jsonl").is_file()
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert config["grid"]["prm"] == "symbolic"
    assert config["search"]["n_samples"] == 3


def test_eval_resume_does_not_duplicate_records(capsys: CaptureFixture, tmp_path: Path) -> None:
    data = write_dataset(tmp_path)
    argv = [
        "eval",
        "--mock",
        "--data-dir",
        str(data),
        "--strategies",
        "cot",
        "--max-calls",
        "20",
        "--run-id",
        "resumed",
    ]
    assert main(argv) == 0
    assert main(argv) == 0
    capsys.readouterr()
    lines = (tmp_path / "runs" / "resumed" / "records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2, "the second pass must reuse the recorded jobs"


def test_report_command_re_renders_without_model_calls(capsys: CaptureFixture, tmp_path: Path) -> None:
    data = write_dataset(tmp_path)
    assert main(["eval", "--mock", "--data-dir", str(data), "--strategies", "cot", "--run-id", "r2"]) == 0
    capsys.readouterr()
    target = tmp_path / "again.md"
    assert main(["report", "--run", str(tmp_path / "runs" / "r2"), "--out", str(target), "--title", "Rerendered"]) == 0
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# Rerendered")
    assert "## Strategies" in text


def test_train_prm_fits_and_saves_a_head(capsys: CaptureFixture, tmp_path: Path) -> None:
    data = write_dataset(tmp_path)
    model = tmp_path / "prm.joblib"
    code = main(["train-prm", "--mock", "--data-dir", str(data), "--labels", "symbolic,outcome", "--out", str(model)])
    assert code == 0
    assert model.is_file()
    printed = capsys.readouterr().out
    assert "accuracy" in printed.lower()
    assert "| feature | weight |" in printed


def test_mock_latency_reaches_the_llm_settings(tmp_path: Path) -> None:
    """The interactive demo is only steerable because the mock really sleeps."""
    args = build_parser().parse_args(["serve", "--mock", "--mock-latency-ms", "450", "--data-dir", str(tmp_path)])
    settings = _settings(args)
    assert settings.llm.provider == "mock"
    assert settings.llm.mock_latency_ms == 450.0
