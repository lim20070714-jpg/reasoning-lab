"""One live search session: run a strategy, stream its events, intervene.

The session is also the :class:`~reasoning_lab.strategies.base.Gate` that tree
search consults before every expansion, so pause, resume and prune take effect
at the next checkpoint instead of requiring the run to restart. Editing a step
re-verifies that node and its whole subtree with the configured PRM, which is
the loop the UI exists to make possible.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from typing import Any

from ..llm.base import LLMClient
from ..parser import extract_final_answer
from ..schema import (
    CheckResult,
    EventKind,
    NodeState,
    Problem,
    ReasoningTree,
    SearchNode,
    Step,
    StrategyOutput,
    TraceEvent,
    Verification,
)
from ..strategies import run_strategy
from ..strategies.base import SearchConfig
from ..strategies.chain import keep_order
from ..verification.answer_check import answers_match
from ..verification.prm.base import ProcessRewardModel, StepScore

SessionState = str

RUNNING = "running"
PAUSED = "paused"
FINISHED = "finished"
STOPPED = "stopped"
FAILED = "failed"


class SearchStopped(RuntimeError):
    """Raised inside the strategy when a human stops the search."""


class InterventionError(ValueError):
    """A requested intervention does not apply to the session's current state."""


def _subtree(tree: ReasoningTree, node_id: str) -> list[SearchNode]:
    out: list[SearchNode] = []
    stack = [node_id]
    while stack:
        current = stack.pop()
        node = tree.nodes.get(current)
        if node is None:
            continue
        out.append(node)
        stack.extend(node.children)
    return out


class SearchSession:
    """A single problem, a single strategy, watched and steered from outside."""

    def __init__(
        self,
        problem: Problem,
        llm: LLMClient,
        prm: ProcessRewardModel,
        config: SearchConfig,
        *,
        strategy: str = "cot",
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.problem = problem
        self.llm = llm
        self.prm = prm
        self.config = config
        self.strategy = strategy
        self.created_at = time.time()
        self.state: SessionState = RUNNING
        self.tree: ReasoningTree | None = None
        self.output: StrategyOutput | None = None
        self.error: str | None = None
        #: an operator's pick wins over whatever the strategy concluded
        self.chosen_answer: str | None = None
        self.events: list[TraceEvent] = []
        self.interventions: list[dict[str, Any]] = []
        self.finished = asyncio.Event()
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self._resume = asyncio.Event()
        self._resume.set()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[Any] | None = None

    # -- lifecycle ----------------------------------------------------
    def start(self) -> None:
        if self._task is not None:
            raise InterventionError("this session already started")
        self._task = asyncio.create_task(self._run(), name=f"session-{self.id}")

    async def _run(self) -> None:
        try:
            self.output = await run_strategy(
                self.strategy,
                self.problem,
                self.llm,
                self.prm,
                self.config,
                sink=self._sink,
                gate=self,
                on_tree=self._attach,
            )
            self.state = FINISHED
        except asyncio.CancelledError:
            self.state = STOPPED
            raise
        except SearchStopped:
            self.state = STOPPED
        except Exception as exc:  # surfaced to the UI as a failed session
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = FAILED
        finally:
            self._resume.set()
            self.finished.set()
            self._close_subscribers()

    def _attach(self, tree: ReasoningTree) -> None:
        self.tree = tree

    def _sink(self, event: TraceEvent) -> None:
        self.events.append(event)
        payload = event.model_dump(mode="json")
        for queue in list(self._subscribers):
            queue.put_nowait(payload)

    # -- streaming ----------------------------------------------------
    @property
    def is_terminal(self) -> bool:
        """No further events will be produced, so a stream can be closed."""
        return self.state in (FINISHED, STOPPED, FAILED)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def _close_subscribers(self) -> None:
        """Tell every open stream that no more events are coming.

        Without this an EventSource for a finished session reconnects forever and
        replays the whole log each time.
        """
        payload = {"kind": "stream_end", "state": self.state, "seq": len(self.events)}
        for queue in list(self._subscribers):
            queue.put_nowait(payload)

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with contextlib.suppress(ValueError):
            self._subscribers.remove(queue)

    def events_after(self, seq: int) -> list[dict[str, Any]]:
        return [event.model_dump(mode="json") for event in self.events if event.seq > seq]

    # -- gate ---------------------------------------------------------
    async def checkpoint(self, node_id: str | None = None) -> None:
        if self._stop.is_set():
            raise SearchStopped("stopped by the operator")
        if not self._resume.is_set():
            self._record({"action": "wait", "node_id": node_id})
            await self._resume.wait()
            if self._stop.is_set():
                raise SearchStopped("stopped by the operator")

    # -- interventions ------------------------------------------------
    def pause(self) -> None:
        """Stop expanding; already-discovered nodes stay readable."""
        if self.state != RUNNING:
            raise InterventionError(f"session is {self.state}")
        self._resume.clear()
        self.state = PAUSED
        self._emit_intervention(node_id=None, action="pause")

    def resume(self) -> None:
        if self.state != PAUSED:
            raise InterventionError(f"session is {self.state}")
        self._resume.set()
        self.state = RUNNING
        self._emit_intervention(node_id=None, action="resume")

    async def stop(self) -> None:
        """Abandon the search, keeping the tree for inspection."""
        self._stop.set()
        self._resume.set()
        self._emit_intervention(node_id=None, action="stop")
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self.tree is not None:
            for node in self.tree.frontier():
                node.state = NodeState.PRUNED
        if self.state not in (FINISHED, FAILED):
            self.state = STOPPED

    def prune(self, node_id: str) -> int:
        """Close a branch and everything below it; search re-ranks around it."""
        tree = self._require_tree()
        self._node(node_id)
        affected = [n for n in _subtree(tree, node_id) if n.state not in (NodeState.PRUNED, NodeState.TERMINAL)]
        for target in affected:
            target.state = NodeState.PRUNED
            tree.metrics.nodes_pruned += 1
            tree.emit(EventKind.NODE_PRUNED, target.id, {"reason": "pruned by the operator"})
        self._record({"action": "prune", "node_id": node_id, "nodes": len(affected)})
        return len(affected)

    async def edit_step(self, node_id: str, text: str) -> SearchNode:
        """Replace a step and re-verify the node plus its whole subtree."""
        self._require_paused("edit a step")
        node = self._node(node_id)
        node.step = Step(index=node.step.index if node.step else 0, text=text)
        if not node.children:
            answer = extract_final_answer(text)
            node.final_answer = answer or None
            node.terminal = bool(answer)
        await self._reverify_subtree(node_id)
        self._record({"action": "edit", "node_id": node_id, "text": text})
        return node

    async def inject_step(self, parent_id: str, text: str) -> SearchNode:
        """Add a human step under ``parent_id``; its children are re-verified."""
        self._require_paused("inject a step")
        tree = self._require_tree()
        parent = self._node(parent_id)
        step = Step(index=len(tree.steps_to(parent_id)), text=text)
        score = await self.prm.score_step(self.problem.id, tree.steps_to(parent_id), step)
        node = tree.add_node(
            parent_id,
            step,
            meta={"injected": True, "prm_source": score.source},
        )
        node.prm_score = score.score
        node.verification = _verification(score)
        answer = extract_final_answer(text)
        if answer:
            node.final_answer = answer
            node.terminal = True
            node.state = NodeState.TERMINAL
        parent.state = NodeState.EXPANDED
        await self._reverify_subtree(node.id)
        self._record({"action": "inject", "node_id": node.id, "text": text})
        return node

    def take_answer(self, node_id: str) -> str:
        """End the search with this node's chain as the answer."""
        tree = self._require_tree()
        node = self._node(node_id)
        answer = node.final_answer or extract_final_answer(node.text) or node.text
        self.chosen_answer = answer
        tree.answer = answer
        tree.answer_node_id = node.id
        tree.emit(EventKind.ANSWER_RESOLVED, node.id, {"answer": answer, "source": "operator"})
        self._record({"action": "take_answer", "node_id": node_id, "answer": answer})
        return answer

    def rerank(self) -> list[dict[str, Any]]:
        """Candidates ordered by the current path scores, after any edit."""
        tree = self._require_tree()
        scored = [n for n in tree.nodes.values() if n.final_answer]
        return [
            {
                "node_id": n.id,
                "answer": n.final_answer,
                "branch_score": n.branch_score,
                "depth": n.depth,
                "status": str(n.verification.status) if n.verification else None,
            }
            for n in keep_order(scored)
        ]

    async def _reverify_subtree(self, node_id: str) -> None:
        """Recompute verification and branch scores along a subtree, path order."""
        tree = self._require_tree()
        for node in _subtree(tree, node_id):
            prefix = tree.steps_to(node.parent_id) if node.parent_id else []
            if node.step is not None:
                score = await self.prm.score_step(self.problem.id, prefix, node.step)
                node.prm_score = score.score
                node.verification = _verification(score)
                tree.metrics.verifications += 1
                tree.emit(
                    EventKind.NODE_SCORED,
                    node.id,
                    {"score": score.score, "status": str(score.status), "source": score.source},
                )
            node.branch_score = _path_value(self.prm, tree, node.id)
            tree.emit(
                EventKind.NODE_SCORED,
                node.id,
                {"branch_score": node.branch_score, "status": "recomputed"},
            )

    def _record(self, payload: dict[str, Any]) -> None:
        self.interventions.append({"ts": time.time(), **payload})

    def _emit_intervention(self, *, node_id: str | None, action: str) -> None:
        self._record({"action": action, "node_id": node_id})
        if self.tree is not None:
            self.tree.emit(EventKind.INTERVENTION, node_id, {"action": action})

    def _require_tree(self) -> ReasoningTree:
        if self.tree is None:
            raise InterventionError("search has not produced a tree yet")
        return self.tree

    def _node(self, node_id: str) -> SearchNode:
        tree = self._require_tree()
        node = tree.nodes.get(node_id)
        if node is None:
            raise InterventionError(f"unknown node: {node_id}")
        return node

    def _require_paused(self, what: str) -> None:
        if self.state != PAUSED:
            raise InterventionError(f"pause the search before you {what}")

    # -- view ---------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        tree = self.tree
        concluded = self.output.answer if self.output else (tree.answer if tree else None)
        answer = self.chosen_answer or concluded
        gold = self.problem.answer or None
        return {
            "id": self.id,
            "state": self.state,
            "strategy": self.strategy,
            "problem_id": self.problem.id,
            "question": self.problem.question,
            "created_at": self.created_at,
            "events": len(self.events),
            "interventions": list(self.interventions),
            "answer": answer,
            "gold": gold,
            "correct": answers_match(answer, gold)[0] if answer and gold else None,
            "votes": self.output.votes if self.output else {},
            "error": self.error,
            "summary": tree.summary() if tree else None,
        }

    def snapshot(self) -> dict[str, Any]:
        tree = self._require_tree()
        return {
            "id": self.id,
            "state": self.state,
            "tree": tree.to_nested(),
            "candidates": self.rerank(),
            "metrics": tree.metrics.model_dump(),
        }


def _verification(score: StepScore) -> Verification:
    return Verification(
        status=score.status,
        score=score.score,
        checks=[c if isinstance(c, CheckResult) else CheckResult.model_validate(c) for c in score.checks],
        verifier=score.source,
        notes=score.detail,
    )


def _path_value(prm: ProcessRewardModel, tree: ReasoningTree, node_id: str) -> float:
    scores = [
        StepScore(score=n.verification.score, status=n.verification.status)
        for n in tree.path(node_id)
        if n.verification is not None
    ]
    return prm.path_value(scores) if scores else 0.5


__all__ = [
    "FAILED",
    "FINISHED",
    "PAUSED",
    "RUNNING",
    "STOPPED",
    "InterventionError",
    "SearchSession",
    "SearchStopped",
]
