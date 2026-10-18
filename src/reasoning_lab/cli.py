"""Command line entry point: ``reasoning-lab --help``.

Subcommands::

    run         one problem, one strategy, live tree events
    eval        strategy x problem grid, records + traces + Markdown report
    report      re-render a report from stored records (no model calls)
    train-prm   fit the shallow process-reward head on verifier-derived labels
    datasets    describe the bundled problems and report verifier coverage
    serve       FastAPI server with the interactive reasoning-tree UI
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import Settings
from .evaluation.dataset import describe, load_problems, mock_solutions
from .evaluation.metrics import EvalRecord, summarise, verifier_discrimination
from .evaluation.report import build_report, load_records, load_run_metadata, write_report
from .evaluation.runner import GridSpec, run_grid
from .llm import build_llm
from .llm.base import LLMClient
from .schema import Problem, Step
from .strategies import available, describe_all, run_strategy
from .strategies.base import SearchConfig
from .verification.prm import PRM_KINDS, build_prm
from .verification.prm.base import ProcessRewardModel

DEFAULT_STRATEGIES = ("cot", "self_consistency", "best_of_n", "tot", "mcts")


# -- shared plumbing ---------------------------------------------------------
def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env(getattr(args, "env_file", None))
    if getattr(args, "mock", False):
        settings.llm.provider = "mock"
    if getattr(args, "model", None):
        settings.llm.model = args.model
    if getattr(args, "base_url", None):
        settings.llm.base_url = args.base_url
    if getattr(args, "concurrency", None):
        settings.llm.max_concurrency = args.concurrency
    if getattr(args, "mock_latency_ms", None):
        settings.llm.mock_latency_ms = args.mock_latency_ms
    return settings.ensure_dirs()


def _search_config(args: argparse.Namespace) -> SearchConfig:
    return SearchConfig(
        n_samples=args.n_samples,
        branch_factor=args.branch_factor,
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_llm_calls=args.max_calls,
        prune_below=args.prune_below,
        propose_mode=args.propose_mode,
        sc_selector=args.sc_selector,
        tot_mode=args.tot_mode,
        mcts_value=args.mcts_value,
        mcts_final=args.mcts_final,
        mcts_iterations=args.mcts_iterations,
        seed=args.seed,
        few_shot=args.few_shot,
    )


def _problems(args: argparse.Namespace) -> list[Problem]:
    return load_problems(
        getattr(args, "dataset", None),
        data_dir=getattr(args, "data_dir", "data"),
        subjects=_split(getattr(args, "subject", "")),
        levels=_split(getattr(args, "level", "")),
        ids=_split(getattr(args, "ids", "")),
        limit=getattr(args, "limit", None),
    )


def _split(value: str | None) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def _llm(settings: Settings, problems: Sequence[Problem]) -> LLMClient:
    return build_llm(settings.llm, solutions=mock_solutions(problems) or None)


def _prm(kind: str, problem: Problem | None, llm: LLMClient | None, args: argparse.Namespace) -> ProcessRewardModel:
    return build_prm(
        kind,
        problem=problem,
        llm=llm,
        model_path=getattr(args, "prm_model", None),
        aggregator=getattr(args, "aggregator", "geomean"),
    )


def _print_tree_summary(output: Any) -> None:
    tree = output.tree
    print(f"\nstrategy={tree.strategy} problem={tree.problem_id} answer={output.answer!r}")
    print(
        f"nodes={len(tree.nodes) - 1} calls={tree.metrics.llm_calls} "
        f"tokens={tree.metrics.total_tokens} verifications={tree.metrics.verifications} "
        f"hard_failures={tree.metrics.hard_failures}"
    )
    if output.votes:
        print("votes: " + ", ".join(f"{k} x{v}" for k, v in sorted(output.votes.items(), key=lambda kv: -kv[1])))


# -- commands ---------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    settings = _settings(args)
    problems: list[Problem] = []
    if not args.question:
        try:
            problems = _problems(args)
        except FileNotFoundError:
            problems = []
    problem: Problem
    if args.question:
        problem = Problem(id="cli", question=args.question, answer=args.answer or "", steps=[])
    elif problems:
        problem = problems[0]
    else:
        print("provide --question, or --ids/--dataset to pick a bundled problem", file=sys.stderr)
        return 2
    config = _search_config(args)
    llm = _llm(settings, problems or [problem])
    prm = _prm(args.prm, problem, llm, args)

    events: list[dict[str, Any]] = []

    def sink(event) -> None:
        events.append(event.model_dump())
        if args.verbose:
            print(f"[{event.seq:>4}] {event.kind:<18} {event.node_id or '-'} {json.dumps(event.payload, default=str)}")

    async def go() -> Any:
        try:
            return await run_strategy(args.strategy, problem, llm, prm, config, sink=sink)
        finally:
            await llm.aclose()

    output = asyncio.run(go())
    _print_tree_summary(output)
    if args.trace:
        payload = {
            "tree": output.tree.model_dump(),
            "answer": output.answer,
            "votes": output.votes,
            "extra": output.extra,
            "events": events,
        }
        Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
        Path(args.trace).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"wrote {args.trace}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    settings = _settings(args)
    problems = _problems(args)
    strategies = tuple(_split(args.strategies) or list(DEFAULT_STRATEGIES))
    seeds = tuple(int(s) for s in _split(args.seeds) or [str(args.seed)])
    spec = GridSpec(
        strategies=strategies,
        prm=args.prm,
        prm_aggregator=args.aggregator,
        prm_model=args.prm_model,
        seeds=seeds,
        concurrency=args.concurrency or 2,
        save_traces=not args.no_traces,
        resume=not args.no_resume,
    )
    config = _search_config(args)
    llm = _llm(settings, problems)

    def progress(key: str, record: EvalRecord) -> None:
        card = record.card
        status = "ok" if card.correct else ("error" if record.error else "wrong")
        print(f"{status:>5} {key} answer={card.answer!r} gold={card.gold!r} calls={card.metrics.llm_calls}")

    async def go():
        try:
            return await run_grid(
                problems,
                llm=llm,
                config=config,
                spec=spec,
                run_dir=settings.run_dir,
                run_id=args.run_id,
                on_progress=progress if args.verbose else None,
            )
        finally:
            await llm.aclose()

    result = asyncio.run(go())
    print(
        f"\nrun {result.run_id}: {len(result.records)} records "
        f"({result.succeeded} evaluated, {len(result.errors)} errors, {result.skipped} resumed)"
    )
    for name in strategies:
        rows = [r for r in result.records if r.card.strategy == name]
        if not rows:
            continue
        summary = summarise(rows)
        print(
            f"{name:<18} acc={summary['accuracy']} oracle_any={summary['oracle_any_pct']}% "
            f"calls={summary['mean_llm_calls']} tokens={summary['mean_tokens']} ece={summary['calibration']['ece']}"
        )
    report_path = result.directory / "report.md"
    write_report(
        result.records,
        report_path,
        problems=problems,
        metadata=load_run_metadata(result.directory),
        include_appendix=not args.no_appendix,
    )
    print(f"records: {result.directory / 'records.jsonl'}")
    print(f"report:  {report_path}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"copied:  {args.out}")
    for error in result.errors[:10]:
        print(f"error: {error}", file=sys.stderr)
    return 1 if result.errors and args.fail_on_error else 0


def cmd_report(args: argparse.Namespace) -> int:
    directory = Path(args.run)
    records = load_records(directory)
    if not records:
        print(f"no records in {directory}", file=sys.stderr)
        return 2
    problems: list[Problem] = []
    with contextlib.suppress(FileNotFoundError, ValueError):
        problems = load_problems(data_dir=args.data_dir)
    text = build_report(
        records,
        problems=problems,
        metadata=load_run_metadata(directory if directory.is_dir() else directory.parent),
        title=args.title,
        include_appendix=not args.no_appendix,
    )
    target = Path(args.out) if args.out else (directory / "report.md" if directory.is_dir() else None)
    if target:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target}")
    else:
        print(text)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    settings = _settings(args)
    problems = _problems(args)
    kinds = _split(args.labels) or ["symbolic", "outcome"]
    llm = _llm(settings, problems) if "rollout" in kinds else None

    async def go() -> tuple[Any, Any]:
        from .verification.prm.labels import (
            outcome_examples,
            rollout_examples,
            symbolic_examples,
        )
        from .verification.prm.learned import LearnedPRM

        examples: list[Any] = []
        try:
            for problem in problems:
                prm = _prm(args.prm, problem, llm, args)
                reference = [Step(index=i, text=t) for i, t in enumerate(problem.steps)]
                if not reference:
                    continue
                scores = await prm.score_path(problem.id, reference)
                if "symbolic" in kinds:
                    examples += symbolic_examples(problem, reference, scores)
                if "outcome" in kinds:
                    examples += outcome_examples(problem, reference, scores, correct=True)
                    for variant in problem.flawed:
                        steps = [Step(index=i, text=t) for i, t in enumerate(variant)]
                        vscores = await prm.score_path(problem.id, steps)
                        examples += outcome_examples(problem, steps, vscores, correct=False)
                if "rollout" in kinds and llm is not None:
                    examples += await rollout_examples(problem, reference, llm, prm, k=args.rollout_k)
        finally:
            if llm is not None:
                await llm.aclose()

        model, report = LearnedPRM.fit(examples, estimator=args.estimator, C=args.C, seed=args.seed)
        return model, report

    model, report = asyncio.run(go())
    print(report.to_markdown())
    target = Path(args.out)
    print("saved:", model.save(target))
    return 0


def cmd_datasets(args: argparse.Namespace) -> int:
    problems = load_problems(args.dataset, data_dir=args.data_dir)
    stats = describe(problems)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(verifier_discrimination(problems), indent=2, ensure_ascii=False))
    return 0


def cmd_strategies(args: argparse.Namespace) -> int:
    print(json.dumps({"available": available(), "strategies": describe_all(), "prm": list(PRM_KINDS)}, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .serving.app import create_app

    settings = _settings(args)
    problems: list[Problem] = []
    with contextlib.suppress(FileNotFoundError):
        problems = _problems(args)
    app = create_app(settings=settings, problems=problems, config=_search_config(args))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


# -- parser -----------------------------------------------------------------
def _add_search_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-samples", type=int, default=8, help="samples for self-consistency / best-of-n")
    parser.add_argument("--branch-factor", type=int, default=3, help="children per expansion for ToT/MCTS")
    parser.add_argument("--max-depth", type=int, default=8, help="maximum chain length")
    parser.add_argument("--beam-width", type=int, default=2, help="beam width for ToT")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--max-calls", type=int, default=120, help="per-job LLM call budget")
    parser.add_argument("--prune-below", type=float, default=0.02, help="drop branches under this PRM score")
    parser.add_argument("--propose-mode", choices=("batch", "individual"), default="batch")
    parser.add_argument("--sc-selector", choices=("vote", "vote_prm"), default="vote")
    parser.add_argument("--tot-mode", choices=("beam", "dfs"), default="beam")
    parser.add_argument("--mcts-value", choices=("prm", "hybrid"), default="prm")
    parser.add_argument("--mcts-final", choices=("visits", "value"), default="visits")
    parser.add_argument("--mcts-iterations", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--few-shot", action="store_true", help="prepend worked examples to the prompt")


def _add_data_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", nargs="*", default=None, help="explicit JSONL files")
    parser.add_argument("--data-dir", default="data", help="directory holding problems_*.jsonl")
    parser.add_argument("--subject", default="", help="comma-separated filter")
    parser.add_argument("--level", default="", help="comma-separated filter")
    parser.add_argument("--ids", default="", help="comma-separated problem ids")
    parser.add_argument("--limit", type=int, default=None, help="keep at most N problems")


def _add_model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mock", action="store_true", help="use the deterministic offline backend")
    parser.add_argument("--model", default=None, help="override REASONING_LAB_MODEL")
    parser.add_argument("--base-url", default=None, help="override REASONING_LAB_API_BASE_URL")
    parser.add_argument("--env-file", default=None, help="path to a .env file")
    parser.add_argument("--prm", default="symbolic", choices=PRM_KINDS)
    parser.add_argument("--prm-model", default=None, help="path to a trained head for --prm learned/mixed")
    parser.add_argument("--aggregator", default="geomean", choices=("product", "geomean", "mean", "min", "last"))
    parser.add_argument("--concurrency", type=int, default=None, help="parallel requests / parallel jobs")
    parser.add_argument(
        "--mock-latency-ms",
        type=float,
        default=None,
        help="per-request delay for --mock, so the interactive search can be steered",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reasoning-lab", description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="solve one problem with one strategy")
    _add_model_flags(run)
    _add_search_flags(run)
    _add_data_flags(run)
    run.add_argument("--strategy", default="cot", help="one of " + ", ".join(available()))
    run.add_argument("--question", default="", help="ad-hoc problem text")
    run.add_argument("--answer", default="", help="gold answer, only used for scoring")
    run.add_argument("--trace", default=None, help="write tree + events to this JSON file")
    run.set_defaults(func=cmd_run)

    ev = sub.add_parser("eval", help="run the strategy x problem grid")
    _add_model_flags(ev)
    _add_search_flags(ev)
    _add_data_flags(ev)
    ev.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    ev.add_argument("--seeds", default="", help="comma-separated seeds; one grid per seed")
    ev.add_argument("--run-id", default=None, help="resume or name a run directory")
    ev.add_argument("--out", default=None, help="also copy the report here")
    ev.add_argument("--no-traces", action="store_true", help="skip writing traces.jsonl")
    ev.add_argument("--no-resume", action="store_true", help="ignore existing records")
    ev.add_argument("--no-appendix", action="store_true")
    ev.add_argument("--fail-on-error", action="store_true", help="exit non-zero when a job errors")
    ev.set_defaults(func=cmd_eval, concurrency=None)

    rp = sub.add_parser("report", help="re-render a Markdown report from records")
    rp.add_argument("--run", required=True, help="run directory or records.jsonl")
    rp.add_argument("--out", default=None)
    rp.add_argument("--data-dir", default="data")
    rp.add_argument("--title", default="Reasoning strategy evaluation")
    rp.add_argument("--no-appendix", action="store_true")
    rp.set_defaults(func=cmd_report)

    tr = sub.add_parser("train-prm", help="fit the shallow PRM head on verifier-derived labels")
    _add_model_flags(tr)
    _add_data_flags(tr)
    tr.add_argument("--labels", default="symbolic,outcome", help="comma-separated: symbolic,outcome,rollout")
    tr.add_argument("--rollout-k", type=int, default=4, help="continuations per prefix for rollout labels")
    tr.add_argument("--estimator", default="logistic", choices=("logistic", "gbm"))
    tr.add_argument("--C", type=float, default=1.0, help="inverse regularisation for the logistic head")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--out", default="models/prm.joblib")
    tr.set_defaults(func=cmd_train, limit=None)

    ds = sub.add_parser("datasets", help="describe the bundled problems and verifier coverage")
    _add_data_flags(ds)
    ds.set_defaults(func=cmd_datasets, limit=None)

    st = sub.add_parser("strategies", help="list strategies and PRM kinds")
    st.set_defaults(func=cmd_strategies)

    sv = sub.add_parser("serve", help="interactive visualisation server")
    _add_model_flags(sv)
    _add_search_flags(sv)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--data-dir", default="data")
    sv.set_defaults(func=cmd_serve, limit=None, subject="", level="", ids="", dataset=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
