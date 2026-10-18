"""Tests for the live search session and the HTTP surface in front of it.

Interventions are checked against a hand-built tree so the state machine is
deterministic; the endpoints are checked over ASGI with the mock backend, which
keeps the whole file offline and API-key free.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

import pytest
import uvicorn
from httpx import ASGITransport, AsyncClient

from reasoning_lab.config import LLMSettings, Settings
from reasoning_lab.evaluation.runner import GridSpec, run_grid
from reasoning_lab.llm.mock import MockLLM
from reasoning_lab.schema import (
    EventKind,
    NodeState,
    Problem,
    ReasoningTree,
    Step,
    StrategyOutput,
    VerifyStatus,
)
from reasoning_lab.serving.app import WEB_DIR, create_app
from reasoning_lab.serving.session import InterventionError, SearchSession, SearchStopped
from reasoning_lab.strategies.base import SearchConfig, node_is_dead
from reasoning_lab.verification.prm.symbolic import SymbolicPRM

CLEAN = ["Total apples packed: 3*4=12.", "Mia eats 5, so 12-5=7.", "Left x = 7."]
WRONG_FIRST = "Total apples packed: 3*4=13."


class SlowMock(MockLLM):
    """``latency_ms`` is bookkeeping only, so steering mid-search needs a real await."""

    delay_s = 0.02

    async def generate(self, prompt, params=None, *, tag=""):  # type: ignore[override]
        await asyncio.sleep(self.delay_s)
        return await super().generate(prompt, params, tag=tag)


def session_for(problem: Problem, llm: MockLLM, strategy: str = "tot") -> SearchSession:
    return SearchSession(problem, llm, SymbolicPRM(), SearchConfig(n_samples=2, max_depth=4), strategy=strategy)


async def attach_chain(session: SearchSession, steps: Sequence[str]) -> list[str]:
    """Give a never-started session a scored chain, the way search would have built it."""
    tree = ReasoningTree(problem_id=session.problem.id, strategy=session.strategy, sink=session._sink)
    session.tree = tree
    ids: list[str] = []
    parent = tree.root_id
    for index, text in enumerate(steps):
        node = tree.add_node(parent, Step(index=index, text=text))
        if index == len(steps) - 1:
            node.final_answer = text.rsplit("=", 1)[1].strip().rstrip(".")
            node.terminal = True
            node.state = NodeState.TERMINAL
        ids.append(node.id)
        parent = node.id
    await session._reverify_subtree(tree.root_id)
    return ids


# -- session: interventions ------------------------------------------------
async def test_editing_a_step_reverifies_the_whole_subtree(problem, mock_llm):
    session = session_for(problem, mock_llm)
    ids = await attach_chain(session, CLEAN)
    tree = session.tree
    assert tree.nodes[ids[0]].verification.status is VerifyStatus.PASS
    leaf_before = tree.nodes[ids[-1]].branch_score

    session.pause()
    await session.edit_step(ids[0], WRONG_FIRST)

    assert tree.nodes[ids[0]].verification.status is VerifyStatus.FAIL
    assert tree.nodes[ids[-1]].branch_score < leaf_before
    assert [e.kind for e in session.events].count(EventKind.NODE_SCORED) >= 2


async def test_editing_a_leaf_updates_its_answer(problem, mock_llm):
    session = session_for(problem, mock_llm)
    ids = await attach_chain(session, CLEAN)
    session.pause()
    node = await session.edit_step(ids[-1], "Left x = 9.")
    assert node.final_answer == "9"
    assert session.rerank()[0]["answer"] == "9"


async def test_inject_requires_pause_and_scores_the_new_branch(problem, mock_llm):
    session = session_for(problem, mock_llm)
    ids = await attach_chain(session, CLEAN)
    with pytest.raises(InterventionError):
        await session.inject_step(ids[1], "Mia eats 5, so 12-5=7.")
    session.pause()
    node = await session.inject_step(ids[1], "Half of the rest is 7/2.")
    assert node.meta["injected"] is True
    assert node.prm_score is not None
    assert node.id in session.tree.nodes[ids[1]].children


async def test_prune_closes_the_branch_it_is_given(problem, mock_llm):
    session = session_for(problem, mock_llm)
    ids = await attach_chain(session, CLEAN)
    tree = session.tree
    rival = tree.add_node(ids[1], Step(index=2, text="Left x = 9."))
    session.pause()

    pruned = session.prune(ids[1])

    assert pruned == 2  # the requested node and the human rival; the terminal leaf stays
    assert tree.metrics.nodes_pruned == pruned
    assert node_is_dead(tree, ids[1]) and node_is_dead(tree, rival.id)
    assert not node_is_dead(tree, ids[-1])
    assert any(e.kind is EventKind.NODE_PRUNED for e in session.events)
    with pytest.raises(InterventionError):
        session.prune("missing-node")


async def test_take_answer_ends_the_search_with_that_chain(problem, mock_llm):
    session = session_for(problem, mock_llm)
    ids = await attach_chain(session, CLEAN)
    answer = session.take_answer(ids[-1])
    assert answer == "7"
    assert session.tree.answer_node_id == ids[-1]
    assert session.events[-1].kind is EventKind.ANSWER_RESOLVED
    described = session.describe()
    assert described["correct"] is True
    # a strategy that concludes a different answer afterwards must not overwrite the operator's pick
    session.output = StrategyOutput(tree=session.tree, answer="9")
    assert session.describe()["answer"] == "7"


async def test_gate_checkpoint_honours_pause_and_stop(problem, mock_llm):
    session = session_for(problem, mock_llm)
    await attach_chain(session, CLEAN)
    session.pause()
    waiting = asyncio.create_task(session.checkpoint("root"))
    await asyncio.sleep(0)
    assert not waiting.done()
    session.resume()
    await asyncio.wait_for(waiting, 1.0)

    await session.stop()
    assert session.state == "stopped"
    with pytest.raises(SearchStopped):
        await session.checkpoint()


async def test_interventions_reject_a_missing_tree(problem, mock_llm):
    session = session_for(problem, mock_llm)
    with pytest.raises(InterventionError):
        session.snapshot()
    with pytest.raises(InterventionError):
        session.prune("root")


# -- http: catalogue -------------------------------------------------------
def build_app(tmp_path: Path, problem: Problem, llm: MockLLM):
    settings = Settings(
        run_dir=tmp_path / "runs",
        data_dir=tmp_path / "data",
        llm=LLMSettings(provider="mock", base_url="mock://", model="mock"),
    )
    config = SearchConfig(
        n_samples=2,
        branch_factor=2,
        beam_width=2,
        max_depth=4,
        max_llm_calls=24,
        mcts_iterations=6,
        temperature=0.0,  # the mock only rolls flawed/truncated chains above 0.01
    )
    return create_app(settings=settings, problems=[problem], llm=llm, config=config, web_dir=tmp_path)


@pytest.fixture
def app(tmp_path, problem, mock_llm):
    return build_app(tmp_path, problem, mock_llm)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver", timeout=10.0) as http:
        yield http
    await app.state.hub.close()


async def test_bundled_ui_is_served_alongside_the_api(tmp_path, problem, mock_llm):
    """The front end lives at the repository root, not next to the package."""
    app = create_app(
        settings=Settings(
            run_dir=tmp_path / "runs",
            data_dir=tmp_path / "data",
            llm=LLMSettings(provider="mock", base_url="mock://", model="mock"),
        ),
        problems=[problem],
        llm=mock_llm,
    )
    assert (WEB_DIR / "index.html").is_file() and (WEB_DIR / "app.js").is_file()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver", timeout=5.0) as http:
        assert (await http.get("/")).status_code == 200
        assert (await http.get("/app.js")).status_code == 200
        assert (await http.get("/api/problems")).status_code == 200, "static mount must not shadow the API"


async def test_strategy_catalogue_and_problem_list_hide_the_gold(client):
    strategies = (await client.get("/api/strategies")).json()
    assert {"cot", "self_consistency", "best_of_n", "tot", "mcts"} <= {s["name"] for s in strategies["strategies"]}
    assert set(strategies["prm"]) == {"none", "symbolic", "learned", "judge", "mixed"}

    body = (await client.get("/api/problems")).json()
    assert body["count"] == 1
    row = body["problems"][0]
    assert {"id", "question", "subject", "level", "tags", "steps"} == set(row)
    assert row["steps"] == 3, "step count only; the reference chain and its answer stay server-side"


# -- http: a live session --------------------------------------------------
async def test_cot_session_runs_to_completion_and_snapshots(app, client):
    created = await client.post("/api/sessions", json={"problem_id": "t-apples", "strategy": "cot", "prm": "symbolic"})
    assert created.status_code == 201
    session_id = created.json()["id"]

    session = app.state.hub.sessions[session_id]
    await asyncio.wait_for(session.finished.wait(), 10.0)

    state = (await client.get(f"/api/sessions/{session_id}")).json()
    assert state["state"] == "finished"
    assert state["answer"] == "7" and state["correct"] is True
    assert state["summary"]["nodes"] >= len(CLEAN)

    snapshot = (await client.get(f"/api/sessions/{session_id}/tree")).json()
    assert snapshot["tree"]["id"] == "root"
    assert snapshot["candidates"] and snapshot["candidates"][0]["answer"] == "7"
    assert snapshot["metrics"]["llm_calls"] >= 1

    listed = (await client.get("/api/sessions")).json()["sessions"]
    assert [s["id"] for s in listed] == [session_id]


async def test_live_stream_and_human_edit_round_trip(tmp_path, problem):
    """Steer a running search over real HTTP.

    The event stream never closes on its own and the mock returns instantly, so
    this needs a listening server and a backend that actually awaits.
    """
    llm = SlowMock(
        [{"match": "Mia buys all boxes", "steps": list(problem.steps), "answer": problem.answer}],
        seed=1,
    )
    app = build_app(tmp_path, problem, llm)
    server, task, base = await _serve(app)
    try:
        async with AsyncClient(base_url=base, timeout=10.0) as http:
            created = await http.post(
                "/api/sessions", json={"problem_id": problem.id, "strategy": "tot", "prm": "symbolic"}
            )
            assert created.status_code == 201
            session_id = created.json()["id"]
            session = app.state.hub.sessions[session_id]

            streamed: list[dict] = []
            async with http.stream("GET", f"/api/sessions/{session_id}/events") as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[len("data: ") :])
                    streamed.append(event)
                    if event["kind"] != "node_added" or event["node_id"] in (None, "root"):
                        continue
                    assert (await http.post(f"/api/sessions/{session_id}/pause")).status_code == 200
                    edited = await http.post(
                        f"/api/sessions/{session_id}/edit",
                        json={"node_id": event["node_id"], "text": WRONG_FIRST},
                    )
                    assert edited.status_code == 200
                    assert edited.json()["status"] == VerifyStatus.FAIL.value
                    break

            assert (await http.get(f"/api/sessions/{session_id}")).json()["state"] == "paused"
            replayed_from = streamed[-1]["seq"]

            assert (await http.post(f"/api/sessions/{session_id}/resume")).status_code == 200
            await asyncio.wait_for(session.finished.wait(), 20.0)
            tail = await _read_events(http, f"/api/sessions/{session_id}/events?last_seq={replayed_from}")
            seqs = [e["seq"] for e in tail if e["kind"] != "stream_end"]
            assert seqs and seqs == list(range(replayed_from + 1, replayed_from + 1 + len(seqs)))
            assert tail[-1]["kind"] == "stream_end", "a finished session closes its stream"
            assert any(
                n.verification.status is VerifyStatus.FAIL for n in session.tree.nodes.values() if n.verification
            ), "the edited step stays rejected after the search resumes"
    finally:
        await _shutdown(server, task)


async def _serve(app) -> tuple[uvicorn.Server, asyncio.Task, str]:
    # bounded grace: an idle SSE handler would otherwise hold the shutdown open
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", timeout_graceful_shutdown=2)
    )
    task = asyncio.create_task(server.serve(), name="uvicorn")
    while not server.started:
        await asyncio.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, task, f"http://127.0.0.1:{port}"


async def _shutdown(server: uvicorn.Server, task: asyncio.Task) -> None:
    server.should_exit = True
    await asyncio.wait_for(task, 10.0)


async def _read_events(client: AsyncClient, url: str) -> list[dict]:
    """Collect an SSE body; the server ends the stream once the session is done."""
    events: list[dict] = []

    async def collect() -> None:
        async with client.stream("GET", url) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[len("data: ") :]))

    await asyncio.wait_for(collect(), 15.0)
    return events


async def test_write_back_endpoints_apply_to_the_tree(app, client):
    session_id = (
        await client.post("/api/sessions", json={"problem_id": "t-apples", "strategy": "cot", "prm": "symbolic"})
    ).json()["id"]
    session = app.state.hub.sessions[session_id]
    await asyncio.wait_for(session.finished.wait(), 10.0)
    nodes = list(session.tree.nodes)
    leaf = next(n for n in nodes if session.tree.nodes[n].final_answer)

    blocked = await client.post(f"/api/sessions/{session_id}/edit", json={"node_id": leaf, "text": WRONG_FIRST})
    assert blocked.status_code == 409, "a finished search is not a paused one"

    answered = await client.post(f"/api/sessions/{session_id}/answer", json={"node_id": leaf})
    assert answered.json()["answer"] == "7"

    pruned = await client.post(f"/api/sessions/{session_id}/prune", json={"node_id": nodes[1]})
    assert pruned.json()["pruned"] >= 1

    unknown = await client.post(f"/api/sessions/{session_id}/prune", json={"node_id": "nope"})
    assert unknown.status_code == 409


async def test_bad_requests_are_rejected(client):
    missing_problem = await client.post("/api/sessions", json={"problem_id": "no-such-problem"})
    assert missing_problem.status_code == 404
    bad_strategy = await client.post("/api/sessions", json={"problem_id": "t-apples", "strategy": "graph_explore"})
    assert bad_strategy.status_code == 400
    bad_prm = await client.post("/api/sessions", json={"problem_id": "t-apples", "prm": "oracle"})
    assert bad_prm.status_code == 400
    bad_config = await client.post("/api/sessions", json={"problem_id": "t-apples", "config": {"answer": "7"}})
    assert bad_config.status_code == 400
    assert "not settable" in bad_config.json()["detail"]
    assert (await client.get("/api/sessions/000000000000")).status_code == 404
    assert (await client.get("/api/runs/no-such-run/summary")).status_code == 404
    assert (await client.get(f"/api/runs/{quote('..escape')}/summary")).status_code == 400


# -- http: recorded runs ---------------------------------------------------
async def test_recorded_run_is_replayable(app, client, problem, mock_llm):
    result = await run_grid(
        [problem],
        llm=mock_llm,
        config=SearchConfig(n_samples=3, max_depth=4, max_llm_calls=24),
        spec=GridSpec(strategies=("cot", "best_of_n"), prm="symbolic", seeds=(0,), concurrency=1),
        run_dir=app.state.hub.settings.run_dir,
        run_id="demo-run",
    )
    assert result.succeeded == 2

    runs = (await client.get("/api/runs")).json()["runs"]
    assert {"run_id": "demo-run", "records": 2, "report": False} in runs

    summary = (await client.get("/api/runs/demo-run/summary")).json()
    assert 0.0 <= summary["accuracy"] <= 1.0

    jobs = (await client.get("/api/runs/demo-run/jobs")).json()
    assert jobs["total"] == 2
    key = jobs["jobs"][0]["key"]
    replay = (await client.get(f"/api/runs/demo-run/jobs/{quote(key, safe='')}")).json()
    assert replay["problem_id"] == problem.id
    assert len(replay["tree"]["nodes"]) >= 2, "traces stay flat; the front end rebuilds the nesting"
    assert (await client.get(f"/api/runs/demo-run/jobs/{quote('cot|nope|seed0|symbolic|geomean', safe='')}")).status_code == 404
