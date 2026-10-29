# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow semantic versioning.

## [0.2.0] - 2026-10-28

First run against a real model, the four parsing/search defects it exposed, and
the TTRL-style label-free reward metric.

### Added

- **TTRL-style self-labelling** (`evaluation/metrics.py:ttrl_stats`): takes the
  majority vote over the answers a run actually produced, treats it as the
  pseudo-label, and reports two numbers — `majority label acc` (how often that
  label is right against the gold answer) and `reward acc` (how often the implied
  reward, "candidate matches the majority", agrees with the true verdict). This
  measures the label-free reward a TTRL loop would train on without any gold
  answers. Surfaced as a `## TTRL-style self-labelling` section in the report and
  in `summarise()`; single-chain strategies show a one-vote majority and are not
  meaningful there.
- **Termination contract for step prompts** (`prompts/templates.py:STEP_TERMINATION`):
  the step prompt now asks the model to append `Final Answer: <answer>` on the
  line that completes a solution, and to say nothing about the answer otherwise.
  Candidate prompts get the same instruction with "a candidate completes".

### Fixed

- **ToT walked a real model to the depth limit without ever terminating.** With
  no instruction to close the chain, ToT (beam mode) expanded to depth 16, found
  no terminal node, and returned an empty answer; the recorded run showed 0
  candidates on the first real-model problem. With the termination contract the
  same problem answers `15` from 3 agreeing candidates in 15 calls.
- **The symbolic verifier counted LaTeX commands as prose.** `\times` and friends
  matched the "two or more letters" prose test, so arithmetic steps written in
  LaTeX were classified as non-math and came back UNKNOWN — 3 of 4 CoT steps in
  the first real-model trace. `looks_like_math` now strips `\command` sequences
  before the prose check, and a regression test pins it.
- **The verifier aborted a whole job when sympy could not invert the system.**
  `sp.solve` raises `NotImplementedError` on transcendental equations (gamma
  functions in a combinatorics chain), which escaped `solve_for` and was recorded
  as a job error — 3 of 56 jobs in the first real-model pass. Unresolved bindings
  now fall through to UNKNOWN, which is what the verifier contract already says.
- **Resume treated errored jobs as finished.** The checkpoint marked any job with
  a record as done, errors included, so a failed job was never retried. `_read_records`
  now keeps the latest record per job key and only lists jobs without an error as
  done, so a second pass replaces the failed record instead of duplicating or
  dropping it.
- **Greek control sequences were silently dropped before parsing.** `latex_to_source`
  strips every `\command`, so `4\pi` reached sympy as `4` and `2\theta+1` as `2+1`
  — a wrong parse is worse than no parse, because the verifier then "checked" a
  different expression. Greek names now map to their sympy identifiers and count as
  known math words rather than prose.
- **Answer wrappers counted as part of the answer.** A real-model run answered
  `\( 4\pi \)` to a gold `4*pi` and was scored wrong: `normalize_answer` stripped
  `$` but not `\(…\)`, `\[…\]` or `$$…$$`. All four wrappers are stripped now.
- Stray `SyntaxWarning` from sympy when re-parsing a step is suppressed inside
  `parse_math` instead of leaking into eval stderr.

### Changed

- Tests: 88 → 95, covering the TTRL metric (correct majority, incorrect majority,
  report section), the LaTeX/prose verifier case, the uninvertible-system case, the
  error-retrying resume and the two answer-parsing fixes.
- README now reports real-model numbers: run `local-qwen25-7b-pilot`, 12 problems
  × 5 strategies against a local `Qwen2.5-7B-Instruct-Q4_K_M` llama.cpp server,
  60 records, 0 errors, 874 calls, 343,115 tokens. The mock-backend table is kept
  as a separate subsection.

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
