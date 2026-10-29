"""Markdown experiment report rendered from persisted records.

``reasoning-lab report --run runs/<id>`` rebuilds the document from
``records.jsonl`` alone, so publishing numbers never requires re-spending model
calls. Every figure in the output is computed from those records or from the
bundled dataset.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..schema import Problem
from .dataset import describe
from .metrics import EvalRecord, group, summarise, verifier_discrimination

NULL = "-"


def _fmt(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return NULL
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(fraction: Any) -> str:
    """Format a 0-1 fraction as a percentage."""
    return NULL if fraction is None else f"{100.0 * fraction:.1f}%"


def _already_pct(value: Any) -> str:
    """Format a value the metrics layer already expressed in percent."""
    return NULL if value is None else f"{value:.1f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def load_records(path: Path | str) -> list[EvalRecord]:
    file = Path(path)
    if file.is_dir():
        file = file / "records.jsonl"
    records: list[EvalRecord] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        try:
            records.append(EvalRecord.model_validate(payload))
        except Exception:
            continue
    return records


def load_run_metadata(directory: Path) -> dict[str, Any]:
    config = directory / "config.json"
    if not config.is_file():
        return {}
    try:
        return json.loads(config.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def strategy_rows(summaries: dict[str, dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for name, s in summaries.items():
        calibration = s.get("calibration") or {}
        discrimination = s.get("discrimination") or {}
        rows.append(
            [
                f"`{name}`",
                _fmt(s.get("n")),
                _pct(s.get("accuracy")),
                _already_pct(s.get("oracle_any_pct")),
                _already_pct(s.get("oracle_prm_pct")),
                _fmt(s.get("mean_llm_calls")),
                _fmt(s.get("mean_tokens"), digits=0),
                _fmt(s.get("tokens_per_correct"), digits=0),
                _fmt(s.get("mean_steps")),
                _fmt(s.get("mean_branch_score")),
                _fmt(calibration.get("ece")),
                _fmt(discrimination.get("auc_confidence")),
            ]
        )
    return rows


HEADERS = [
    "strategy",
    "n",
    "accuracy",
    "oracle any",
    "oracle PRM",
    "calls",
    "tokens",
    "tokens/correct",
    "steps",
    "mean score",
    "ECE",
    "AUC",
]


def build_report(
    records: Sequence[EvalRecord],
    *,
    problems: Sequence[Problem] | None = None,
    metadata: dict[str, Any] | None = None,
    title: str = "Reasoning strategy evaluation",
    include_appendix: bool = True,
) -> str:
    if not records:
        return f"# {title}\n\nNo records. Run `reasoning-lab eval` first.\n"
    meta = metadata or {}
    grid = meta.get("grid", {})
    search = meta.get("search", {})
    overall = summarise(records)
    by_strategy = group(records, "strategy")
    lines: list[str] = [f"# {title}", ""]

    facts: list[list[str]] = [
        ["run id", meta.get("run_id", NULL)],
        ["jobs evaluated", _fmt(overall["n"])],
        ["failed jobs", _fmt(overall["errors"])],
        ["PRM", f"`{grid.get('prm', 'symbolic')}`"],
        ["seeds", ", ".join(str(s) for s in grid.get("seeds", [0]))],
        ["budget per job", f"max_llm_calls={search.get('max_llm_calls', NULL)}"],
        ["sampling", f"n_samples={search.get('n_samples', NULL)}, branch={search.get('branch_factor', NULL)}, depth={search.get('max_depth', NULL)}"],
        ["model calls", _fmt(overall["llm_calls"])],
        ["cache hits", f"{_fmt(overall['cached_calls'])} ({_already_pct(overall['cache_hit_pct'])})"],
        ["tokens", _fmt(overall["total_tokens"])],
        ["wall time (s)", _fmt(overall["wall_time_s"], digits=1)],
    ]
    lines += ["## Run", "", _table(["field", "value"], facts), ""]

    if problems:
        stats = describe(problems)
        lines += [
            "## Dataset",
            "",
            _table(
                ["field", "value"],
                [
                    ["problems", _fmt(stats["problems"])],
                    ["with reference steps", _fmt(stats["with_reference_steps"])],
                    ["with flawed variants", _fmt(stats["with_flawed"])],
                    ["subjects", ", ".join(f"{k}: {v}" for k, v in stats["by_subject"].items())],
                ],
            ),
            "",
        ]
        steps = verifier_discrimination(problems)
        lines += [
            "### Symbolic verifier over the bundled data",
            "",
            _table(
                ["metric", "value"],
                [
                    ["reference steps", _fmt(steps["reference_steps"])],
                    ["flawed chains", _fmt(steps["flawed_variants"])],
                    ["decidable (PASS or FAIL)", _already_pct(steps["decidable_pct"])],
                    ["flawed chains with a rejected step", _already_pct(steps["variant_detection_pct"])],
                    ["reference steps wrongly rejected", _already_pct(steps["reference_false_flag_pct"])],
                    ["rejected steps inside flawed chains", _already_pct(steps["flawed_step_rejection_pct"])],
                    ["chain-level AUROC (reference vs flawed)", _fmt(steps["chain_auc"])],
                    ["step-level AUROC", _fmt(steps["step_auc"])],
                ],
            ),
            "",
        ]

    lines += [
        "## Strategies",
        "",
        _table(HEADERS, strategy_rows(by_strategy)),
        "",
        "`oracle any` is the ceiling for best-of-k over the candidate answers a run produced; "
        "`oracle PRM` is the accuracy of picking the highest branch-score candidate. "
        "Both are recomputed from traces and never consulted by the strategies themselves.",
        "",
    ]

    rerank_rows: list[list[str]] = []
    for name, summary in by_strategy.items():
        rerank = summary.get("rerank") or {}
        if not rerank:
            continue
        sample = next(iter(rerank.values()))
        rerank_rows.append(
            [
                f"`{name}`",
                _fmt(sample["n"]),
                *[_pct((rerank.get(mode) or {}).get("accuracy")) for mode in ("first", "vote", "prm_mean", "prm_max", "oracle")],
            ]
        )
    if rerank_rows:
        lines += [
            "## Answer selection policy",
            "",
            "Same recorded candidates, different selectors. A PRM-selected accuracy above the "
            "first-answer column is the process reward model paying for itself.",
            "",
            _table(["strategy", "n", "first", "majority vote", "PRM mean", "PRM max", "oracle"], rerank_rows),
            "",
        ]

    ttrl_rows: list[list[str]] = []
    for name, summary in by_strategy.items():
        stats = summary.get("ttrl") or {}
        if not stats.get("with_candidates"):
            continue
        ttrl_rows.append(
            [
                f"`{name}`",
                _fmt(stats["with_candidates"]),
                _pct(stats["majority_label_accuracy"]),
                _pct(stats["reward_accuracy"]),
                _fmt(stats["candidates_scored"]),
            ]
        )
    if ttrl_rows:
        lines += [
            "## TTRL-style self-labelling",
            "",
            "TTRL replaces the gold answer with a majority vote over the run's own candidates. "
            "`majority label acc` is how often that label is actually correct; `reward acc` is how "
            "often the implied reward (candidate matches the majority) agrees with the true verdict. "
            "Single-chain strategies have a one-vote majority, so read this table on the "
            "multi-candidate strategies.",
            "",
            _table(["strategy", "problems", "majority label acc", "reward acc", "candidates scored"], ttrl_rows),
            "",
        ]

    for name, summary in by_strategy.items():
        bins = (summary.get("calibration") or {}).get("bins") or []
        if not bins:
            continue
        lines += [
            f"## Calibration · `{name}`",
            "",
            _table(
                ["confidence bin", "n", "avg confidence", "accuracy", "gap"],
                [[b["bin"], _fmt(b["n"]), _fmt(b["avg_confidence"]), _fmt(b["accuracy"]), _fmt(b["gap"])] for b in bins],
            ),
            "",
        ]

    for key, label in (("subject", "By subject"), ("level", "By level")):
        buckets = group(records, key)
        if len(buckets) < 2:
            continue
        lines += [
            f"## {label}",
            "",
            _table(
                [key, "n", "accuracy", "mean score", "calls", "tokens"],
                [[k, _fmt(s["n"]), _pct(s["accuracy"]), _fmt(s["mean_branch_score"]),
                  _fmt(s["mean_llm_calls"]), _fmt(s["mean_tokens"], digits=0)] for k, s in buckets.items()],
            ),
            "",
        ]

    errors = [r for r in records if r.error]
    if errors:
        lines += [
            "## Failed jobs",
            "",
            _table(["job", "error"], [[f"`{r.card.strategy}` {r.card.problem_id}", r.error] for r in errors[:40]]),
            "",
        ]

    if include_appendix:
        lines += [
            "## Per-problem appendix",
            "",
            _table(
                ["problem", "strategy", "answer", "gold", "correct", "score", "status", "steps", "calls"],
                [
                    [
                        r.card.problem_id,
                        f"`{r.card.strategy}`",
                        str(r.card.answer or NULL),
                        str(r.card.gold or NULL),
                        _fmt(r.card.correct),
                        _fmt(r.card.confidence),
                        str(r.card.verify_status),
                        _fmt(r.card.steps),
                        _fmt(r.card.metrics.llm_calls),
                    ]
                    for r in sorted(records, key=lambda x: (x.card.problem_id, x.card.strategy))
                ],
            ),
            "",
        ]

    lines += [
        "## Reproduce",
        "",
        "```",
        "uv run reasoning-lab eval --strategies cot,self_consistency,best_of_n,tot,mcts --prm symbolic",
        "uv run reasoning-lab report --run runs/<run-id>",
        "```",
        "",
    ]
    return "\n".join(lines)


def write_report(records: Sequence[EvalRecord], path: Path | str, **kwargs: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_report(records, **kwargs), encoding="utf-8")
    return target


__all__ = ["build_report", "load_records", "load_run_metadata", "write_report"]
