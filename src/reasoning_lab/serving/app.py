"""FastAPI surface for the interactive front end.

Three things the browser needs: the problem list, a live event stream per
session, and the intervention endpoints that write back into a running search.
Finished grid runs are replayable from their trace files, so a recorded run can
be reviewed with the same tree view as a live one.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import Settings
from ..evaluation.dataset import load_problems, mock_solutions
from ..evaluation.metrics import summarise
from ..evaluation.report import load_records
from ..llm import build_llm
from ..llm.base import LLMClient
from ..schema import Problem
from ..strategies import available, describe_all
from ..strategies.base import SearchConfig
from ..verification.prm import PRM_KINDS, build_prm
from .session import InterventionError, SearchSession

#: src/reasoning_lab/serving/app.py -> repository root, where web/ lives
ROOT = Path(__file__).resolve().parents[3]
WEB_DIR = ROOT / "web"

Payload = Annotated[dict[str, Any], Body()]

#: an idle SSE connection is re-checked for disconnects this often
KEEPALIVE_S = 5.0


class Hub:
    """Sessions, the shared model client and the problem pool."""

    def __init__(
        self,
        *,
        settings: Settings,
        problems: list[Problem],
        config: SearchConfig,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.problems = {p.id: p for p in problems}
        self.config = config
        self._llm = llm
        self.sessions: dict[str, SearchSession] = {}

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = build_llm(self.settings.llm, solutions=mock_solutions(list(self.problems.values())) or None)
        return self._llm

    def create(self, *, problem_id: str, strategy: str, prm: str, overrides: dict[str, Any]) -> SearchSession:
        problem = self.problems.get(problem_id)
        if problem is None:
            raise HTTPException(404, f"unknown problem id: {problem_id}")
        if strategy not in available():
            raise HTTPException(400, f"unknown strategy '{strategy}', available: {', '.join(available())}")
        if prm not in PRM_KINDS:
            raise HTTPException(400, f"unknown prm kind '{prm}', expected one of {', '.join(PRM_KINDS)}")
        config = _apply_overrides(self.config, overrides)
        session = SearchSession(
            problem,
            self.llm,
            build_prm(prm),
            config,
            strategy=strategy,
        )
        self.sessions[session.id] = session
        session.start()
        return session

    def get(self, session_id: str) -> SearchSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise HTTPException(404, f"no such session: {session_id}")
        return session

    async def close(self) -> None:
        for session in self.sessions.values():
            await session.stop()
        if self._llm is not None:
            await self._llm.aclose()


def _apply_overrides(config: SearchConfig, overrides: dict[str, Any]) -> SearchConfig:
    """Copy only the documented search knobs a client may change per session."""
    allowed = {f for f in SearchConfig.__dataclass_fields__}
    unknown = set(overrides) - allowed
    if unknown:
        raise HTTPException(400, f"config fields not settable: {', '.join(sorted(unknown))}")
    return SearchConfig(**{**_config_dict(config), **overrides})


def _config_dict(config: SearchConfig) -> dict[str, Any]:
    return asdict(config)


def create_app(
    *,
    settings: Settings | None = None,
    problems: list[Problem] | None = None,
    config: SearchConfig | None = None,
    llm: LLMClient | None = None,
    web_dir: Path | str | None = None,
) -> FastAPI:
    """Build the application. ``llm`` lets tests inject the mock backend."""
    settings = settings or Settings.from_env()
    problems = problems if problems is not None else _default_problems(settings)
    hub = Hub(settings=settings, problems=problems, config=config or SearchConfig(), llm=llm)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await hub.close()

    app = FastAPI(title="reasoning-lab", version="0.1.0", lifespan=lifespan)
    app.state.hub = hub

    @app.get("/api/strategies")
    async def strategies() -> dict[str, Any]:
        return {"strategies": describe_all(), "prm": list(PRM_KINDS)}

    @app.get("/api/problems")
    async def list_problems(
        subject: str = Query(default=""),
        level: str = Query(default=""),
        limit: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        rows = [
            {
                "id": p.id,
                "question": p.question,
                "subject": p.subject,
                "level": p.level,
                "tags": p.tags,
                "steps": len(p.steps),
            }
            for p in problems
            if (not subject or p.subject == subject) and (not level or p.level == level)
        ]
        return {"count": len(rows), "problems": rows[: limit or len(rows)]}

    @app.get("/api/sessions")
    async def list_sessions() -> dict[str, Any]:
        return {"sessions": [s.describe() for s in hub.sessions.values()]}

    @app.post("/api/sessions", status_code=201)
    async def create_session(payload: Payload) -> dict[str, Any]:
        session = hub.create(
            problem_id=str(payload.get("problem_id", "")),
            strategy=str(payload.get("strategy", "cot")),
            prm=str(payload.get("prm", "symbolic")),
            overrides=dict(payload.get("config") or {}),
        )
        return session.describe()

    @app.get("/api/sessions/{session_id}")
    async def session_state(session_id: str) -> dict[str, Any]:
        return hub.get(session_id).describe()

    @app.get("/api/sessions/{session_id}/tree")
    async def session_tree(session_id: str) -> dict[str, Any]:
        session = hub.get(session_id)
        try:
            return session.snapshot()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/sessions/{session_id}/events")
    async def session_events(
        session_id: str,
        request: Request,
        last_seq: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        session = hub.get(session_id)
        queue = session.subscribe()

        async def stream() -> AsyncIterator[str]:
            try:
                for event in session.events_after(last_seq):
                    yield _sse(event)
                if session.is_terminal:
                    # the closing frame was pushed before this client subscribed
                    yield _sse({"kind": "stream_end", "state": session.state, "seq": len(session.events)})
                    return
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_S)
                    except TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield _sse(payload)
                    if payload.get("kind") == "stream_end":
                        return
            finally:
                session.unsubscribe(queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/sessions/{session_id}/pause")
    async def pause(session_id: str) -> dict[str, Any]:
        return _call(hub.get(session_id).pause)

    @app.post("/api/sessions/{session_id}/resume")
    async def resume(session_id: str) -> dict[str, Any]:
        return _call(hub.get(session_id).resume)

    @app.post("/api/sessions/{session_id}/stop")
    async def stop(session_id: str) -> dict[str, Any]:
        await hub.get(session_id).stop()
        return hub.get(session_id).describe()

    @app.post("/api/sessions/{session_id}/prune")
    async def prune(session_id: str, payload: Payload) -> dict[str, Any]:
        session = hub.get(session_id)
        return _call(lambda: {"pruned": session.prune(str(payload.get("node_id", "")))})

    @app.post("/api/sessions/{session_id}/edit")
    async def edit(session_id: str, payload: Payload) -> dict[str, Any]:
        session = hub.get(session_id)
        node = await _await_call(session.edit_step(str(payload.get("node_id", "")), str(payload.get("text", ""))))
        return {"node_id": node.id, "score": node.prm_score, "status": str(node.verification.status)}

    @app.post("/api/sessions/{session_id}/inject")
    async def inject(session_id: str, payload: Payload) -> dict[str, Any]:
        session = hub.get(session_id)
        node = await _await_call(session.inject_step(str(payload.get("parent_id", "")), str(payload.get("text", ""))))
        return {"node_id": node.id, "score": node.prm_score, "status": str(node.verification.status)}

    @app.post("/api/sessions/{session_id}/answer")
    async def answer(session_id: str, payload: Payload) -> dict[str, Any]:
        session = hub.get(session_id)
        return _call(lambda: {"answer": session.take_answer(str(payload.get("node_id", "")))})

    @app.get("/api/runs")
    async def list_runs() -> dict[str, Any]:
        rows = []
        for directory in sorted(hub.settings.run_dir.glob("*/")) if hub.settings.run_dir.is_dir() else []:
            records = directory / "records.jsonl"
            if records.is_file():
                rows.append(
                    {
                        "run_id": directory.name,
                        "records": sum(1 for line in records.read_text(encoding="utf-8").splitlines() if line.strip()),
                        "report": (directory / "report.md").is_file(),
                    }
                )
        return {"runs": rows}

    @app.get("/api/runs/{run_id}/summary")
    async def run_summary(run_id: str) -> dict[str, Any]:
        records = load_records(_run_dir(hub.settings, run_id))
        if not records:
            raise HTTPException(404, f"run {run_id} has no records")
        return summarise(records)

    @app.get("/api/runs/{run_id}/jobs")
    async def run_jobs(run_id: str) -> dict[str, Any]:
        traces = _run_dir(hub.settings, run_id) / "traces.jsonl"
        if not traces.is_file():
            raise HTTPException(404, f"run {run_id} stored no traces")
        jobs = []
        for line in traces.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            jobs.append({"key": payload.get("key"), "problem_id": payload.get("problem_id")})
        return {"total": len(jobs), "jobs": jobs}

    @app.get("/api/runs/{run_id}/jobs/{key}")
    async def run_job(run_id: str, key: str) -> dict[str, Any]:
        directory = _run_dir(hub.settings, run_id)
        traces = directory / "traces.jsonl"
        if not traces.is_file():
            raise HTTPException(404, f"run {run_id} stored no traces")
        for line in traces.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("key") == key:
                return {"key": key, "tree": payload.get("tree"), "problem_id": payload.get("problem_id")}
        raise HTTPException(404, f"no job {key} in run {run_id}")

    directory = Path(web_dir) if web_dir else WEB_DIR
    if directory.is_dir():
        app.mount("/", StaticFiles(directory=str(directory), html=True), name="web")
    return app


def _default_problems(settings: Settings) -> list[Problem]:
    try:
        return load_problems(data_dir=settings.data_dir)
    except FileNotFoundError:
        return []


def _run_dir(settings: Settings, run_id: str) -> Path:
    if "/" in run_id or ".." in run_id:
        raise HTTPException(400, "invalid run id")
    directory = settings.run_dir / run_id
    if not directory.is_dir():
        raise HTTPException(404, f"no such run: {run_id}")
    return directory


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def _call(action: Any) -> dict[str, Any]:
    try:
        result = action()
    except InterventionError as exc:
        raise HTTPException(409, str(exc)) from exc
    return result if isinstance(result, dict) else {"ok": True}


async def _await_call(coroutine: Any) -> Any:
    try:
        return await coroutine
    except InterventionError as exc:
        raise HTTPException(409, str(exc)) from exc


__all__ = ["Hub", "create_app"]
