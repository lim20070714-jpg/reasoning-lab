# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow semantic versioning.

## [0.1.0] - 2026-10-18

First release: strategy harness, PRM layer, evaluation runner, interactive
visualisation server, bundled dataset.

### Added

- **Schema layer** (`schema.py`): `Problem`, `ReasoningNode`, `ReasoningTree`,
  `TraceEvent`, `VerifyStatus`, `ScoreCard`, `RunMetrics`; node state machine with
  `pending / expanded / pruned / terminal / failed`.
- **Parser** (`parser.py`): step splitting for model output and final-answer
  extraction (`Final Answer:` markers, trailing `x = n`, multiple-choice letters).
- **Symbolic verifier** (`verification/symbolic.py`, `verification/math_text.py`):
  sympy re-computation of every arithmetic step with equality bindings carried
  forward, unit and multiple-choice comparison, three-valued PASS / FAIL / UNKNOWN.
- **PRM layer** (`verification/prm/`): `SymbolicPRM`, `LearnedPRM` (logistic or
  gradient-boosting head over 19 step features), `LLMJudgePRM`, `MixedPRM`,
  `NullPRM`; path aggregators `product / geomean / mean / min / last` with
  `geomean` as the default; label builders for reference, outcome and rollout
  supervision (`verification/prm/labels.py`).
- **Strategies** (`strategies/`): CoT, Self-Consistency, Best-of-N, ToT
  (beam and dfs), MCTS (UCT with PRM or hybrid value), all behind
  `run_strategy(name, problem, llm, prm, config, *, sink, gate, on_tree)` with one
  shared `SearchConfig` budget; aliases resolve `best-of-n`, `self-consistency`,
  `tree_of_thoughts`.
- **Gold isolation**: `StrategyContext.__post_init__` removes `answer`, `steps`,
  `solution` and `flawed` from the problem a strategy can read; oracle metrics are
  recomputed from traces instead.
- **Deterministic selection**: `strategies/chain.py` breaks score ties on step
  text and answer rather than on the random node id, and `GridJob.key` includes the
  PRM aggregator, so same-seed reruns reach the same candidates in the same order.
- **LLM backends** (`llm/`): OpenAI-compatible client with retries, semaphore
  concurrency and a content-keyed completion cache; deterministic `MockLLM` with a
  configurable per-request sleep (`mock_latency_ms`) so live searches can be steered.
- **Evaluation** (`evaluation/`): resumable grid runner writing
  `config.json / records.jsonl / traces.jsonl / report.md` per run; accuracy,
  oracle-any, oracle-PRM, cost and calibration (ECE, Brier, AUC) aggregates;
  Markdown report with per-strategy, per-subject and verifier sections.
- **Serving** (`serving/`): FastAPI app with `SearchSession` state machine,
  `Gate.checkpoint()` human-in-the-loop suspension, SSE fan-out with `last_seq`
  replay and a `stream_end` closing frame; interventions `pause / resume / stop /
  prune / edit / inject / answer`, where an operator's answer overrides the
  strategy's conclusion; trace-replay endpoints over recorded runs.
- **Front end** (`web/`): four-zone vanilla HTML + SVG client (problem bank,
  reasoning tree, step inspector, monitor) with pan/zoom, verification-coloured
  nodes and no build step or external assets.
- **Dataset** (`data/`): 40 curated problems (24 math, 16 physics) across 15
  levels, each with a reference derivation and flawed variants.
- **CLI** (`cli.py`): `run`, `eval`, `report`, `train-prm`, `datasets`,
  `strategies`, `serve`.
- **Tests**: 88 offline tests across parser, verifier, PRM, strategies,
  evaluation, serving (including a real-socket SSE round trip) and the CLI.
- **CI** (`.github/workflows/ci.yml`): ruff, mypy, dataset coverage check and the
  suite on Python 3.11–3.13, plus an `eval --mock` smoke job.

### Notes

- Every figure quoted in `README.md` is reproducible with the commands listed
  there and comes from the offline mock backend.
- Not included: GRPO / RLVR training, TTRL and PRIME reproductions.
