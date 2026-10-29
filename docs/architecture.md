# Architecture

## Data flow

```
Problem (JSONL)
   │  load_problems()            data/*.jsonl → schema.Problem
   ▼
run_strategy(name, problem, llm, prm, config, sink=…, gate=…)
   │
   ├─ StrategyContext            strips gold fields, owns the budget counter
   │    ├─ Prompt templates      prompts/templates.py
   │    ├─ llm.generate()        OpenAI-compatible client or MockLLM
   │    ├─ parse.split_steps()   model text → Step[]
   │    └─ prm.score_step()      SymbolicPRM / LearnedPRM / LLMJudgePRM / MixedPRM
   │                              └─ verification/symbolic.py (sympy)
   ▼
ReasoningTree + TraceEvent stream
   │  ├─ sink: SSE fan-out (serving/session.py)     live view
   │  └─ StrategyOutput                             evaluation/runner.py → records.jsonl
   ▼
evaluation/metrics.py   accuracy, oracle bounds, cost, calibration, TTRL-style reward
evaluation/report.py    report.md
```

A search is a tree of `SearchNode`s: one reasoning step per node, `parent_id`
giving the prefix, `branch_score` the aggregated path value, `visits` and `value`
used by MCTS. Everything downstream — the live view, the recorded trace, the
report, the replay UI — reads that same structure.

## Modules

| module | responsibility | notable rule |
| --- | --- | --- |
| `schema.py` | pydantic models shared by every layer | `ReasoningTree.add_node()` assigns `uuid4` ids; ordering never depends on them |
| `parser.py` | model text → steps and final answer | answer extraction order: last `Final Answer:` → `\boxed{}` → "answer is" → trailing `x = n` → trailing number |
| `verification/symbolic.py` | path-aware sympy checks | bindings learned from an earlier step make later steps checkable; unresolvable text is `UNKNOWN`, never `FAIL` |
| `verification/prm/base.py` | `ProcessRewardModel` protocol, `path_value()` | aggregators `product / geomean / mean / min / last`, default `geomean` |
| `verification/prm/features.py` | 19 step features for the learned head | symbolic verdicts are features, so the head sits on top of the verifier instead of replacing it |
| `strategies/base.py` | `SearchConfig`, `StrategyContext`, budget enforcement | `BudgetExhausted` raised at `max_llm_calls`; per-call seed `config.seed + calls` |
| `strategies/chain.py` | shared expansion, pruning, scoring helpers | `tie_break()` orders equal scores by step content |
| `strategies/*.py` | the five searches | each implements `solve(problem, ctx)` and nothing else |
| `evaluation/runner.py` | grid, resume, records, traces | job key `strategy\|problem\|seed\|prm\|aggregator` |
| `evaluation/metrics.py` | scoring and oracle recomputation | strategies never see gold; oracle columns are derived from traces |
| `serving/session.py` | session state machine, interventions, SSE | interventions require `paused`; see `docs/visualization.md` |
| `llm/openai_compat.py` | HTTP client | semaphore concurrency, retries with backoff, content-keyed cache |
| `llm/mock.py` | deterministic offline backend | completions come from the dataset's reference and flawed chains |

## Invariants

**Gold isolation.** `StrategyContext.__post_init__` replaces the problem with a
copy whose `answer`, `solution`, `steps` and `flawed` are empty. A strategy that
wanted the label would have to reach around its own context. Consequence: the
`oracle any` and `oracle PRM` columns in a report are computed afterwards from the
candidate list each run recorded.

**Budget accounting.** Every generation goes through `ctx.generate()`, which
increments `ctx.calls`, the tree metrics and the token counters, then raises
`BudgetExhausted` when the budget is spent. Cost numbers in the report are
therefore counts of real calls, not estimates.

**Verification is advisory.** A `FAIL` lowers `branch_score` and can trigger
pruning at `--prune-below` (0.02); it never deletes a node. The symbolic PRM maps
verdicts to probabilities — pass 0.95, fail 0.05, and 0.75–0.85 for the
"nothing to check" cases — so an unparseable step costs a little without being
treated as an error. `answer_confidence_floor` (0.35) is a separate gate, used by
ToT to decide whether a terminal node is worth reporting as an answer at all.

**Aggregation.** `path_value()` combines step scores. `geomean` is the default:
length-normalised, so a five-step chain and a three-step chain are comparable, and
one rejected step still moves the score sharply. `product` punishes long chains;
`mean` lets a strong prefix hide a wrong conclusion. The aggregator is part of the
job key, so switching it never silently reuses cached records.

**Filtering is reversible.** Pruned, low-scoring and rejected branches remain in
the tree and in `traces.jsonl`; selection happens at the ranking layer
(`evaluation/metrics.py`) or in the UI, so an experiment can ask "what if the
threshold had been different" without a rerun.

## Extension points

- New strategy: subclass `ReasoningStrategy` in `strategies/`, implement
  `name`, `describe()` and `solve()`, register it in `STRATEGIES`
  (`strategies/__init__.py`), add an alias if it needs one. `run --strategy` and
  `eval --strategies` pick it up with no other change.
- New scorer: implement `ProcessRewardModel.score_step()` / `score_path()` and add
  the kind to `PRM_KINDS` and `build_prm()`.
- New dataset: append a JSONL file to `data/`; `load_problems()` globs
  `problems_*.jsonl` and `scripts/check_dataset.py` reports coverage.
- Custom model output parsing for a new prompt style belongs in `parser.py`, not in
  a strategy, so verification and metrics keep seeing the same `Step` shape.

## Deliberate limits

- The learned head is logistic regression or gradient-boosting over hand-built
  features. A neural PRM would change the cost profile of every experiment and is
  out of scope for a harness meant to run on one machine.
- `MockLLM` replays curated chains; it exercises the pipeline and cannot measure a
  model. Real numbers require `REASONING_LAB_API_KEY`.
- No RLVR loop yet: `verification/prm/labels.py` emits the step-level supervision a
  GRPO/TTRL-style training run would consume, but the training script itself is not
  part of this release.
