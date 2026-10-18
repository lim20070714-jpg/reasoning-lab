# Interactive visualisation

`reasoning-lab serve` runs the search inside the server and streams it to the
browser. The client is `web/index.html`, `web/styles.css`, `web/app.js` — plain
HTML, SVG and `EventSource`, no bundler, no network fonts, nothing to install.

```bash
uv run python -m reasoning_lab.cli serve --mock --mock-latency-ms 450 --port 8000
# http://127.0.0.1:8000/
```

`--mock-latency-ms` makes the offline backend sleep that many milliseconds per
request. Without it a 120-call search finishes in well under a second and there is
nothing to steer; with the mock sleeping, the pause/edit/inject path is reachable
by hand. Rollouts run concurrently, so wall time is roughly
`rounds × latency`, not `calls × latency`.

## Zones

| zone | contents |
| --- | --- |
| 题库 / problem bank | subject and level filters, the problem list, the run form (strategy, PRM, `n_samples`, `branch_factor`, `max_depth`, `beam_width`, `max_llm_calls`, `temperature`), **Start search** |
| 推理树 / reasoning tree | SVG tree, one box per step: label, PRM score, branch score `b…`, MCTS visits `v…`, answer `=…`; colour by verification status; drag to pan, wheel to zoom |
| 步骤检查 / step inspector | node id, state, depth, scores, visits, answer, verifier status, then the per-check table (`kind / expression / status / detail`), then the intervention buttons and the step editor |
| monitor | `events` (every `TraceEvent` with its seq), `candidates` (answer ranking by branch score), `metrics` (session state, votes, tree metrics), `runs` (trace replay of recorded evaluations) |

The catalogue endpoint deliberately omits gold answers, and `steps` is returned as
a count rather than the reference derivation.

## HTTP surface

| endpoint | notes |
| --- | --- |
| `GET /api/strategies` | strategy descriptions and PRM kinds for the run form |
| `GET /api/problems?subject=&level=&limit=` | id, question, subject, level, tags, reference-step count |
| `POST /api/sessions` | `{problem_id, strategy, prm, config}` → 201 with `describe()`; unknown id 404, unknown strategy or PRM 400, unknown config field 400 |
| `GET /api/sessions` | every live session, `describe()` each |
| `GET /api/sessions/{id}` | `describe()`: state, answer, gold, `correct`, votes, interventions, tree summary |
| `GET /api/sessions/{id}/tree` | `snapshot()`: nested tree, reranked candidates, metrics |
| `GET /api/sessions/{id}/events?last_seq=N` | SSE; replays events after `N`, then streams live ones |
| `POST /api/sessions/{id}/pause` \| `resume` \| `stop` | accepted from any session; `stop` is terminal |
| `POST /api/sessions/{id}/edit` | `{node_id, text}` — **requires `paused`** |
| `POST /api/sessions/{id}/inject` | `{parent_id, text}` — **requires `paused`** |
| `POST /api/sessions/{id}/prune` | `{node_id}` — no pause required, marks the subtree dead |
| `POST /api/sessions/{id}/answer` | `{node_id}` — no pause required, ends the search with that chain |
| `GET /api/runs` | run ids and record counts under `REASONING_LAB_RUN_DIR` |
| `GET /api/runs/{run_id}/jobs` | job keys stored in `traces.jsonl` |
| `GET /api/runs/{run_id}/jobs/{key}` | the recorded tree for replay |

An intervention that arrives out of order returns 409 with the reason (`pause the
search before you edit a step`, `this session already finished`, …).

## Session states

`running → paused → running → finished | stopped | failed`.

Strategies do not poll the session. They call `gate.checkpoint(node_id)` between
expansions, and that call blocks on an `asyncio.Event` while the session is
`paused`, so a pause takes effect at the next checkpoint: requests already in
flight still finish and their nodes still appear. `stop` is different — it raises
`SearchStopped` out of the checkpoint, and the runner records the partial tree.

An operator's `answer` overrides whatever the strategy concluded: `describe()`
reports the chosen chain and scores it against gold.

## Event stream

Emitted `TraceEvent` kinds: `search_started`, `node_added`, `node_scored`,
`node_terminal`, `generation_finished`, `answer_resolved`, plus `node_pruned` and
`intervention` from the search or an operator. `EventKind` also reserves
`generation_started` and `search_finished`, which nothing emits today — the stream
is closed by the synthetic `stream_end` frame instead. Each event carries `seq`,
`ts`, `kind`, `node_id` and `payload`.

Counted over the 200-job mock run in `runs/readme-verify/traces.jsonl`:
17,018 `node_scored`, 8,509 `node_added`, 5,887 `generation_finished`,
988 `node_terminal`, 200 `search_started`, 200 `answer_resolved`.

The endpoint replays everything after `last_seq` before streaming, so a page reload
or a dropped connection catches up rather than restarting blind. A comment frame is
emitted every 5 s of silence to keep proxies from closing the connection, and a
terminal session sends `{"kind":"stream_end"}` so the client closes its
`EventSource` instead of reconnecting forever.

The client debounces tree redraws on incoming events (300 ms), keeps the inspector
in sync unless a step edit has focus (typing into the editor must not be
overwritten), and derives the pause/resume/stop button states from the server's
session state rather than from local assumptions.

## Replay

Switching to **Trace replay** lists runs, then the job keys of the selected run.
Loading one renders the recorded tree read-only — same layout, same scores, no
intervention controls — which is how a finished evaluation is inspected after the
fact. Job keys contain strategy, problem, seed, PRM kind and aggregator.

## Interaction walkthrough

Recorded against the mock backend at 450 ms latency, `mcts` on problem `m01`:

1. Start search; the tree grows node by node while the event log ticks.
2. Pause. Expansion stops at the next checkpoint.
3. Select a step, replace its text with `3*4 = 99`, **re-verify edit**. The
   inspector shows `equality | 3*4=99 | fail | 12 != 99`, the node goes
   `pass → fail`, PRM drops to 0.050 and branch scores recompute upward along the
   subtree.
4. **inject as child** with the same text: a new node appears, scored on
   verification, and the parent's branch score changes again.
5. **take this answer**: the session ends immediately, `describe()` reports the
   injected chain's answer against gold, and the intervention is logged.
6. Resume, then Stop: the status line reads `stopped`, and pause/resume/stop all
   disable because the server state is terminal.
