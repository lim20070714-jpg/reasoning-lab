#!/usr/bin/env python3
"""Validate the bundled problem sets.

Usage:
    uv run python scripts/check_dataset.py [data/*.jsonl ...]

Checks, for every record in ``data/problems_*.jsonl``:

1. required fields exist and are non-empty (``id``, ``question``, ``answer``, ``steps``);
2. every reference step is accepted by the symbolic verifier (PASS or UNKNOWN —
   a reference solution must never contain a step the checker rejects);
3. the last step's derived value matches ``answer`` under ``answers_match``;
4. each ``flawed`` variant contains at least one step the verifier rejects, and
   its final answer differs from ``answer`` (otherwise it is not a negative);
5. ids are unique and ``subject``/``level`` come from the documented vocabulary.

Exits non-zero with a per-problem report when anything fails. Also prints
verification coverage, which the README quotes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reasoning_lab.schema import Problem, Step, VerifyStatus  # noqa: E402
from reasoning_lab.verification.answer_check import answers_match  # noqa: E402
from reasoning_lab.verification.symbolic import SymbolicVerifier  # noqa: E402

ALLOWED_SUBJECTS = {"math", "physics"}
ALLOWED_LEVELS = {
    "arithmetic", "algebra", "geometry", "number-theory", "combinatorics",
    "precalculus", "calculus", "probability", "competition",
    "mechanics", "electricity", "thermodynamics", "waves", "optics", "modern",
}


def load_records(paths: list[Path]) -> list[tuple[Path, int, dict[str, Any]]]:
    records: list[tuple[Path, int, dict[str, Any]]] = []
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            records.append((path, lineno, payload))
    return records


def verify_chain(steps_text: list[str]) -> list[VerifyStatus]:
    verifier = SymbolicVerifier()
    steps = [Step(index=i, text=text) for i, text in enumerate(steps_text)]
    return [v.status for v in verifier.verify_path(steps)]


def check(payload: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    problems: list[str] = []
    stats = {"steps": 0, "decided": 0}
    try:
        problem = Problem.model_validate(payload)
    except Exception as exc:
        return [f"schema: {exc}"], stats

    if not problem.steps:
        problems.append("steps is empty")
    if problem.subject not in ALLOWED_SUBJECTS:
        problems.append(f"subject '{problem.subject}' not in {sorted(ALLOWED_SUBJECTS)}")
    if problem.level not in ALLOWED_LEVELS:
        problems.append(f"level '{problem.level}' not in the documented vocabulary")

    statuses = verify_chain(problem.steps)
    stats["steps"] += len(statuses)
    stats["decided"] += sum(1 for s in statuses if s in (VerifyStatus.PASS, VerifyStatus.FAIL))
    for index, status in enumerate(statuses):
        if status == VerifyStatus.FAIL:
            problems.append(f"reference step {index + 1} is rejected by the verifier: {problem.steps[index]!r}")

    derived = problem.steps[-1] if problem.steps else ""
    matched, method = answers_match(derived.split("=")[-1] if "=" in derived else derived, problem.answer)
    if not matched:
        problems.append(f"last step does not yield answer {problem.answer!r} (matched={matched}, method={method})")

    if not problem.flawed:
        problems.append("no flawed variant (needed as a PRM negative)")
    for variant_index, variant in enumerate(problem.flawed):
        if not variant:
            problems.append(f"flawed[{variant_index}] is empty")
            continue
        v_statuses = verify_chain(list(variant))
        if VerifyStatus.FAIL not in v_statuses:
            problems.append(f"flawed[{variant_index}] has no step the verifier rejects")
        final = variant[-1]
        answer_tail = final.split("=")[-1] if "=" in final else final
        neg_matched, _ = answers_match(answer_tail, problem.answer)
        if neg_matched:
            problems.append(f"flawed[{variant_index}] still reaches the gold answer")
    return problems, stats


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or sorted((ROOT / "data").glob("problems_*.jsonl"))
    if not paths:
        print("no problem files found", file=sys.stderr)
        return 1
    records = load_records(paths)
    seen_ids: dict[str, Path] = {}
    total_steps = 0
    decided_steps = 0
    failures = 0

    for path, lineno, payload in records:
        problems, stats = check(payload)
        total_steps += stats["steps"]
        decided_steps += stats["decided"]
        identifier = str(payload.get("id", ""))
        if identifier in seen_ids:
            problems.append(f"duplicate id (also in {seen_ids[identifier]})")
        elif identifier:
            seen_ids[identifier] = path
        if problems:
            failures += 1
            print(f"FAIL {path.name}:{lineno} id={identifier or '?'}")
            for problem in problems:
                print(f"     - {problem}")

    coverage = (decided_steps / total_steps * 100.0) if total_steps else 0.0
    print(
        f"\n{len(records)} problems, {len(seen_ids)} unique ids, "
        f"{total_steps} reference steps, {decided_steps} decidable ({coverage:.1f}%)"
    )
    if failures:
        print(f"{failures} problem(s) failed validation", file=sys.stderr)
        return 1
    print("dataset OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
