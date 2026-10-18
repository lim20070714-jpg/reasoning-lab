# reasoning-lab

Unified experiment harness for LLM reasoning strategies — CoT, Self-Consistency,
Best-of-N, Tree-of-Thoughts, MCTS — with process reward model verification, a
Markdown evaluation report, and an interactive reasoning-tree view that supports
human intervention during a live search.

中文版本见 [下方](#reasoning-lab-中文版)。

## Quickstart

```bash
uv sync --extra dev

# interactive demo, no API key needed
uv run python -m reasoning_lab.cli serve --mock --mock-latency-ms 450 --port 8000
# open http://127.0.0.1:8000/

# one problem, one strategy, tree events printed as they happen
uv run python -m reasoning_lab.cli -v run --mock --strategy tot --ids m01

# full grid over the bundled dataset
uv run python -m reasoning_lab.cli eval --mock --prm symbolic --data-dir data
```

Against a real model, copy `.env.example` to `.env` and set
`REASONING_LAB_API_KEY`, `REASONING_LAB_API_BASE_URL` and `REASONING_LAB_MODEL`;
every command above then works unchanged once `--mock` is dropped.

## Layout

| path | role |
| --- | --- |
| `src/reasoning_lab/schema.py` | pydantic models: `Problem`, `ReasoningTree`, `TraceEvent`, `ScoreCard` |
| `src/reasoning_lab/parser.py` | step splitting and final-answer extraction from model text |
| `src/reasoning_lab/verification/` | sympy verifier, `SymbolicPRM`, learned head, LLM-judge, mixed scorer |
| `src/reasoning_lab/strategies/` | the five searches behind one `run_strategy()` entry point |
| `src/reasoning_lab/evaluation/` | grid runner, metrics, oracle bounds, Markdown report |
| `src/reasoning_lab/serving/` | FastAPI app, `SearchSession` state machine, SSE fan-out |
| `web/` | zero-build HTML + SVG + `EventSource` client in four zones |
| `data/` | 40 curated math and physics problems with reference and flawed chains |

Deeper notes: [docs/architecture.md](docs/architecture.md) for the layering and
invariants, [docs/visualization.md](docs/visualization.md) for the HTTP surface and
the intervention walkthrough.

## Strategies

All five share `run_strategy(name, problem, llm, prm, config, *, sink, gate, on_tree)`
and the same budget fields (`n_samples`, `branch_factor`, `max_depth`, `beam_width`,
`max_llm_calls`), so a comparison changes the search and nothing else.

| strategy | search shape | PRM used for |
| --- | --- | --- |
| `cot` | one chain | per-step scores, no re-ranking |
| `self_consistency` | N independent chains | answer selection on top of majority vote (`--sc-selector vote_prm`) |
| `best_of_n` | N complete chains | selection of the highest path score |
| `tot` | beam over thought batches | step scoring and beam pruning (`--tot-mode dfs` also available) |
| `mcts` | UCT selection + rollout | node value and pruning (`--mcts-value hybrid`) |

Two invariants make the comparison trustworthy:

- `StrategyContext` strips `answer`, `steps` and `flawed` from the problem the
  strategy sees, so no search can read the label it is scored on. Oracle numbers
  are recomputed from recorded traces in `evaluation/metrics.py`.
- Selection ties break on chain content (`strategies/chain.py:98`), not on the
  random node id, and the PRM aggregator is part of the job key
  (`evaluation/runner.py:42`), so a rerun with the same seed reaches the same
  candidates in the same order. Two independent runs of the table below
  (`readme-baseline`, `readme-verify`) produced identical per-strategy numbers.

## Process reward models

| `--prm` | source |
| --- | --- |
| `none` | constant 0.5, isolates the search from verification |
| `symbolic` | sympy: recompute each step, carry bindings forward, PASS / FAIL / UNKNOWN |
| `learned` | logistic or gradient-boosting head over the features in `verification/prm/features.py` |
| `judge` | an LLM prompted to grade one step |
| `mixed` | symbolic and learned weights combined |

`--aggregator` turns step scores into a path score: `product`, `geomean` (default),
`mean`, `min`, `last`. `geomean` is the default because it is length-normalised —
a longer correct chain is not punished relative to a shorter one, and a single
rejected step still dominates the product.

`uv run python -m reasoning_lab.cli train-prm --mock` fits the head from
verifier-derived labels (reference steps positive, steps of the bundled `flawed`
variants negative) and writes `models/prm.joblib`. On the bundled dataset that is
785 labelled steps (561 positive) and a logistic head measuring accuracy 0.832,
AUC 0.899 on a held-out 25% stratified split.

## Interactive visualisation

`cli serve` exposes the search as it runs. The front end has no framework, no
build step and no CDN dependency.

- **Zone 1 — 题库 / problem bank**: filters, the run form, the strategy and PRM selectors.
- **Zone 2 — 推理树 / reasoning tree**: SVG layout, pan and wheel-zoom, colour by
  verification status, node labels carry PRM score, branch score, visit count.
- **Zone 3 — 步骤检查 / step inspector**: verification detail per step, then the
  intervention controls.
- **Zone 4 — monitor**: event stream, candidate ranking, metrics, trace replay of
  recorded runs.

Events reach the browser over `GET /api/sessions/{id}/events?last_seq=N` (SSE with
replay, so a reload does not lose history). Interventions are `POST`s against the
same session:

| action | effect |
| --- | --- |
| `pause` / `resume` / `stop` | the strategy calls `gate.checkpoint()` between expansions and blocks there |
| `edit` | replace a step, then re-verify that subtree and recompute branch scores upward |
| `inject` | attach a hand-written child to a node |
| `prune` | mark a subtree dead; dead nodes are kept and drawn, never silently dropped |
| `answer` | end the search with the selected chain; the operator's pick overrides the strategy |

Pruned and rejected branches stay in the tree, and `--mock-latency-ms` exists so a
search runs slowly enough to steer by hand.

## Measured results

All figures below come from `reasoning-lab eval --mock --prm symbolic --data-dir data`
(run id `readme-verify`, 200 records, 0 errors, 5,887 model calls, 773,101 tokens,
31.8 s wall time on 2-way concurrency). **The backend is the offline mock, so these
numbers characterise the harness, not any real model.** They exist to prove the
plumbing is deterministic and to give a fresh clone something to compare against.

The mock replays the dataset's reference chain, and with `flawed_rate=0.35` (plus a
12% truncated band) it substitutes a bundled flawed chain instead. `temperature=0`
always yields the reference chain.

| strategy | accuracy | oracle any | oracle PRM | calls | tokens | tokens/correct | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cot` | 72.5% | 72.5% | 72.5% | 1.000 | 132 | 183 | 0.133 |
| `self_consistency` | 90.0% | 100.0% | 97.5% | 8.000 | 1060 | 1178 | 0.099 |
| `best_of_n` | 87.5% | 100.0% | 97.5% | 8.000 | 1060 | 1211 | 0.069 |
| `tot` | 92.5% | 92.5% | 92.5% | 10.220 | 1486 | 1606 | 0.417 |
| `mcts` | 87.5% | 87.5% | 87.5% | 119.950 | 15590 | 17817 | 0.083 |

`oracle any` is the ceiling for best-of-k over the answers a run produced;
`oracle PRM` is the accuracy of picking the highest branch-score candidate.

### Symbolic verifier over the bundled dataset

`reasoning-lab datasets --data-dir data`, no model calls:

| metric | value |
| --- | --- |
| problems | 40 (math 24, physics 16) |
| reference steps | 199 |
| flawed variants / flawed steps | 80 / 398 |
| decidable (PASS or FAIL) | 94.47% |
| flawed variants with at least one rejected step | 100.0% |
| reference steps wrongly rejected | 0.0% |
| flawed steps rejected | 24.87% |
| mean step score, reference vs flawed | 0.9251 / 0.7013 |
| chain-level AUROC | 0.9905 |
| step-level AUROC | 0.6312 |

## Dataset

`data/problems_math.jsonl` and `data/problems_physics.jsonl`, one JSON object per line:

```json
{"id":"m01","question":"...","answer":"15","subject":"math","level":"arithmetic",
 "steps":["3*4 = 12.","..."],"flawed":[["3*4 = 12.","..."]],"tags":["order of operations"]}
```

`steps` is a correct derivation, `flawed` lists variants with an injected
arithmetic error — both are used for PRM labels and oracle metrics, never as
strategy input. `scripts/check_dataset.py` fails when a file is malformed or
coverage drops, and runs in CI.

## Configuration

| variable | default | meaning |
| --- | --- | --- |
| `REASONING_LAB_PROVIDER` | `openai` | `mock` selects the offline backend |
| `REASONING_LAB_API_BASE_URL` | OpenAI | any OpenAI-compatible endpoint |
| `REASONING_LAB_API_KEY` | – | required unless the provider is `mock` |
| `REASONING_LAB_MODEL` | `gpt-4o-mini` | chat model used for generation and `--prm judge` |
| `REASONING_LAB_MOCK_LATENCY_MS` | `0` | per-request sleep for the mock, for steerable demos |
| `REASONING_LAB_CACHE_DIR` | `cache/llm` | completion cache; empty string disables it |
| `REASONING_LAB_RUN_DIR` | `runs` | where `eval` writes records, traces and reports |

## Tests and CI

```bash
uv run python -m pytest tests -q      # 88 tests, offline
uv run ruff check src tests scripts
uv run mypy src/reasoning_lab
```

`.github/workflows/ci.yml` runs lint, mypy, the dataset check and the suite on
Python 3.11–3.13, plus an `eval --mock` smoke job that asserts a report is written.

## Not in this release

- GRPO / RLVR training (LoRA fine-tuning, TTRL and PRIME reproductions) is not
  implemented. `verification/prm/labels.py` already produces the step-level
  supervision such a loop would consume.
- The table above is mock-backend data. A real-model report needs an API key, and
  is deliberately not fabricated here.

## License

MIT.

---

## reasoning-lab 中文版

面向 LLM 推理策略的统一实验框架：CoT、Self-Consistency、Best-of-N、ToT、MCTS 共用
一个 `run_strategy()` 入口与同一套预算参数，接入过程奖励模型（PRM）做逐步验证与
重排序，在数学/物理题上批量评测，并把搜索过程实时可视化、支持人工介入。

### 上手

不需要 API key 即可跑通全部功能：`uv sync --extra dev` 之后执行
`uv run python -m reasoning_lab.cli serve --mock --mock-latency-ms 450 --port 8000`，
浏览器打开 `http://127.0.0.1:8000/`。接入真实模型时，把 `.env.example` 复制为
`.env` 并填写 key、base URL、模型名，命令去掉 `--mock` 即可。

### 设计要点

- **策略层**：五种策略只在搜索形态上不同，预算字段与调用方式完全一致，对比结果
  可以归因到搜索本身。
- **标签隔离**：`StrategyContext` 会把 `answer`、`steps`、`flawed` 从策略可见的
  问题对象中剥离，oracle 类指标一律在 `evaluation/metrics.py` 里由落盘的搜索轨迹
  重新计算，策略无法读到评分依据。
- **可复现**：打分相同的候选按步骤内容决胜（`strategies/chain.py`），不依赖随机
  节点 id，PRM 聚合方式写入 job key，同 seed 重跑得到同一批候选与同一排名。
- **PRM**：`symbolic` 用 sympy 逐步重算并向下传递等式绑定，输出 PASS/FAIL/UNKNOWN；
  另有 `learned`（逻辑回归/GBDT 浅层头）、`judge`、`mixed`、`none`。路径分默认用
  `geomean`，做了长度归一，长链不会因为更长而被惩罚。
- **可视化**：四个分区（题库与运行参数 / 推理树 / 步骤检查 / 事件与指标），
  前端零构建零依赖，事件通过 SSE 推送并支持 `last_seq` 断线重放。介入操作包括
  暂停/继续/终止、改写某一步并重验子树、注入手写子节点、剪枝（被剪分支保留在树中，
  不静默删除）、直接指定答案（人工选择优先于策略结论）。`--mock-latency-ms` 让离线
  demo 慢到可以手动介入。

### 实测数据

数据集共 40 题（数学 24、物理 16），199 条参考步骤、80 条含错变体（398 步）。
符号验证器可判定比例 94.47%，参考步骤误判 0.0%，含错变体检出率 100.0%，
链级 AUROC 0.9905、步级 AUROC 0.6312。
浅层学习头用验证器标签训练：785 条步级标签（561 正例），25% 分层留出集上
accuracy 0.832、AUC 0.899。

评测命令 `reasoning-lab eval --mock --prm symbolic --data-dir data`（200 条记录、
0 失败、5,887 次调用、773,101 tokens、2 并发耗时 31.8 s）结果：
cot 准确率 72.5%（1.0 次调用/题、132 tokens），self_consistency 90.0%（8.0、1060），
best_of_n 87.5%（8.0、1060），tot 92.5%（10.22、1486），mcts 87.5%（119.95、15590）。
**以上均为离线 mock 后端数据，用于验证框架链路与确定性，不代表任何真实模型的表现；
真实模型报告需配好 key 后再跑，此处不做编造。**

### 测试与 CI

`uv run python -m pytest tests -q`（88 项，全部离线）、
`uv run ruff check src tests scripts`、`uv run mypy src/reasoning_lab`。
GitHub Actions 在 Python 3.11–3.13 上跑上述三项加数据集校验，并额外执行一次
`eval --mock` 冒烟评测确认报告产出。

### 暂未实现

GRPO / RLVR 训练流程（LoRA 微调、TTRL 与 PRIME 复现）尚未实现；
`verification/prm/labels.py` 已经产出该流程所需的步级监督标签。
