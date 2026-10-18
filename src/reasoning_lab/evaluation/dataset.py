"""Benchmark loading: bundled ``data/*.jsonl`` plus external JSONL files."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..schema import Problem

DEFAULT_DATA_DIR = Path("data")


def read_jsonl(path: Path) -> list[Problem]:
    problems: list[Problem] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
        problems.append(Problem.model_validate(payload))
    return problems


def load_problems(
    paths: Sequence[Path | str] | None = None,
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    subjects: Iterable[str] = (),
    levels: Iterable[str] = (),
    ids: Iterable[str] = (),
    limit: int | None = None,
) -> list[Problem]:
    """Load problems from explicit files, or every ``problems_*.jsonl`` in ``data_dir``.

    Later filters are applied in order: subject, level, id, then ``limit``.
    """
    files = [Path(p) for p in paths] if paths else sorted(Path(data_dir).glob("problems_*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"no problems_*.jsonl found in {data_dir}. Pass --dataset PATH or run scripts/check_dataset.py."
        )
    seen: set[str] = set()
    problems: list[Problem] = []
    for file in files:
        for problem in read_jsonl(file):
            if problem.id in seen:
                raise ValueError(f"duplicate problem id '{problem.id}' in {file}")
            seen.add(problem.id)
            problems.append(problem)

    subject_set = {s.strip().lower() for s in subjects if s.strip()}
    level_set = {s.strip().lower() for s in levels if s.strip()}
    id_set = {s.strip() for s in ids if s.strip()}
    if subject_set:
        problems = [p for p in problems if p.subject.lower() in subject_set]
    if level_set:
        problems = [p for p in problems if p.level.lower() in level_set]
    if id_set:
        problems = [p for p in problems if p.id in id_set]
    if limit is not None and limit >= 0:
        problems = problems[:limit]
    if not problems:
        raise ValueError("filtering produced an empty problem set")
    return problems


def mock_solutions(problems: Sequence[Problem]) -> list[dict[str, Any]]:
    """Solution bank for the offline mock backend (see ``llm/mock.py``)."""
    return [p.as_mock_solution() for p in problems if p.steps]


def describe(problems: Sequence[Problem]) -> dict[str, Any]:
    by_subject: dict[str, int] = {}
    by_level: dict[str, int] = {}
    for problem in problems:
        by_subject[problem.subject] = by_subject.get(problem.subject, 0) + 1
        by_level[problem.level or "(unset)"] = by_level.get(problem.level or "(unset)", 0) + 1
    return {
        "problems": len(problems),
        "with_reference_steps": sum(1 for p in problems if p.steps),
        "with_flawed": sum(1 for p in problems if p.flawed),
        "by_subject": dict(sorted(by_subject.items())),
        "by_level": dict(sorted(by_level.items())),
    }


__all__ = ["DEFAULT_DATA_DIR", "describe", "load_problems", "mock_solutions", "read_jsonl"]
