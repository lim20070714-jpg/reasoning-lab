"""Core data model shared by strategies, verifiers, the runner and the web UI.

Every object here is a pydantic model so a search tree can be streamed to the
browser and written to disk as JSONL without custom serialisation code.
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1"


class NodeState(StrEnum):
    PENDING = "pending"
    EXPANDED = "expanded"
    PRUNED = "pruned"
    TERMINAL = "terminal"
    FAILED = "failed"


class VerifyStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"
    NOT_CHECKED = "not_checked"


class EventKind(StrEnum):
    NODE_ADDED = "node_added"
    NODE_SCORED = "node_scored"
    NODE_PRUNED = "node_pruned"
    NODE_TERMINAL = "node_terminal"
    GENERATION_STARTED = "generation_started"
    GENERATION_FINISHED = "generation_finished"
    ANSWER_RESOLVED = "answer_resolved"
    INTERVENTION = "intervention"
    SEARCH_STARTED = "search_started"
    SEARCH_FINISHED = "search_finished"


class Problem(BaseModel):
    """One benchmark item. Reference material is optional but powers the mock
    backend and the verifier-labelled PRM training set."""

    id: str
    question: str
    answer: str
    subject: str = "math"
    level: str = ""
    solution: str | None = None
    steps: list[str] = Field(default_factory=list)
    flawed: list[list[str]] = Field(default_factory=list)
    source: str = ""
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def as_mock_solution(self, match: str | None = None) -> dict[str, Any]:
        return {
            "match": match or self.question[:40],
            "steps": list(self.steps),
            "answer": self.answer,
            "flawed": [list(f) for f in self.flawed],
        }


class Message(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


class GenerationParams(BaseModel):
    temperature: float = 0.7
    top_p: float | None = None
    max_tokens: int = 768
    seed: int | None = None
    stop: list[str] | None = None

    def merged_with(self, other: GenerationParams | None) -> GenerationParams:
        return other if other is not None else self


class CheckResult(BaseModel):
    """One atomic verification performed on a reasoning step."""

    kind: str
    expression: str | None = None
    status: VerifyStatus = VerifyStatus.UNKNOWN
    detail: str = ""
    weight: float = 1.0


class Verification(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: VerifyStatus = VerifyStatus.NOT_CHECKED
    score: float = 0.5
    checks: list[CheckResult] = Field(default_factory=list)
    verifier: str = ""
    notes: str = ""
    verified_at: float | None = None

    @property
    def decisive(self) -> bool:
        return self.status in (VerifyStatus.PASS, VerifyStatus.FAIL)

    @classmethod
    def from_checks(cls, checks: list[CheckResult], *, verifier: str) -> Verification:
        """Aggregate weighted checks into a status plus a score in [0, 1].

        Unknown checks pull the score toward the prior 0.5 instead of counting
        as failures, so partial parse coverage does not masquerade as evidence.
        """
        if not checks:
            return cls(status=VerifyStatus.NOT_CHECKED, score=0.5, verifier=verifier)
        total = sum(c.weight for c in checks) or 1.0
        credit = sum(
            c.weight * (1.0 if c.status == VerifyStatus.PASS else 0.0 if c.status == VerifyStatus.FAIL else 0.5)
            for c in checks
        )
        score = credit / total
        hard_fail = any(c.status == VerifyStatus.FAIL for c in checks if c.weight >= 1.0)
        decided = [c for c in checks if c.status != VerifyStatus.UNKNOWN]
        if hard_fail:
            status = VerifyStatus.FAIL
        elif decided and all(c.status == VerifyStatus.PASS for c in decided):
            status = VerifyStatus.PASS
        elif decided:
            status = VerifyStatus.UNKNOWN
        else:
            status = VerifyStatus.UNKNOWN
        return cls(
            status=status,
            score=round(0.05 + 0.9 * score, 4),
            checks=checks,
            verifier=verifier,
            verified_at=time.time(),
        )


class Step(BaseModel):
    index: int = 0
    text: str
    expressions: list[str] = Field(default_factory=list)
    answer: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class SearchNode(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    parent_id: str | None = None
    depth: int = 0
    state: NodeState = NodeState.PENDING
    step: Step | None = None
    children: list[str] = Field(default_factory=list)
    prm_score: float | None = None
    value: float | None = None
    visits: int = 0
    rollouts: int = 0
    terminal: bool = False
    final_answer: str | None = None
    verification: Verification | None = None
    sample_index: int | None = None
    branch_score: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.step.text if self.step else ""


class TraceEvent(BaseModel):
    seq: int
    ts: float
    kind: EventKind
    node_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class RunMetrics(BaseModel):
    llm_calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    wall_time_s: float = 0.0
    nodes_created: int = 0
    nodes_pruned: int = 0
    nodes_expanded: int = 0
    verifications: int = 0
    hard_failures: int = 0

    def add(self, other: RunMetrics) -> RunMetrics:
        for name in type(self).model_fields:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        return self

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ReasoningTree(BaseModel):
    """Mutable search tree; also the serialisable artifact consumed by the UI."""

    problem_id: str
    strategy: str
    root_id: str = "root"
    nodes: dict[str, SearchNode] = Field(default_factory=dict)
    events: list[TraceEvent] = Field(default_factory=list)
    answer: str | None = None
    answer_node_id: str | None = None
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    schema_version: str = SCHEMA_VERSION
    extra: dict[str, Any] = Field(default_factory=dict)
    #: live subscribers (server SSE) receive each event; excluded from dumps
    sink: Any | None = Field(default=None, exclude=True)

    _seq: int = 0

    def model_post_init(self, __context: Any) -> None:
        if self.root_id not in self.nodes:
            self.nodes[self.root_id] = SearchNode(id=self.root_id, state=NodeState.EXPANDED, depth=0)

    # -- construction -------------------------------------------------
    def add_node(
        self,
        parent_id: str,
        step: Step | None = None,
        *,
        sample_index: int | None = None,
        node_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> SearchNode:
        if parent_id not in self.nodes:
            raise KeyError(f"unknown parent node: {parent_id}")
        node = SearchNode(
            id=node_id or uuid.uuid4().hex[:10],
            parent_id=parent_id,
            depth=self.nodes[parent_id].depth + 1,
            step=step,
            sample_index=sample_index,
            meta=dict(meta or {}),
        )
        self.nodes[node.id] = node
        self.nodes[parent_id].children.append(node.id)
        self.nodes[parent_id].state = NodeState.EXPANDED
        self.metrics.nodes_created += 1
        self.emit(EventKind.NODE_ADDED, node.id, {"depth": node.depth, "sample_index": node.sample_index})
        return node

    def emit(self, kind: EventKind, node_id: str | None, payload: dict[str, Any] | None = None) -> TraceEvent:
        self._seq += 1
        event = TraceEvent(seq=self._seq, ts=time.time(), kind=kind, node_id=node_id, payload=dict(payload or {}))
        self.events.append(event)
        if self.sink is not None:
            self.sink(event)
        return event

    # -- queries ------------------------------------------------------
    def path(self, node_id: str) -> list[SearchNode]:
        chain: list[SearchNode] = []
        current: str | None = node_id
        seen: set[str] = set()
        while current and current in self.nodes and current not in seen:
            seen.add(current)
            chain.append(self.nodes[current])
            current = self.nodes[current].parent_id
        return list(reversed(chain))

    def steps_to(self, node_id: str) -> list[Step]:
        return [n.step for n in self.path(node_id) if n.step is not None]

    def leaves(self) -> list[SearchNode]:
        return [n for n in self.nodes.values() if not n.children and n.id != self.root_id]

    def frontier(self) -> list[SearchNode]:
        return [n for n in self.nodes.values() if n.state in (NodeState.PENDING, NodeState.EXPANDED)]

    def nodes_with_answers(self) -> list[SearchNode]:
        return [n for n in self.nodes.values() if n.final_answer]

    def to_nested(self, node_id: str | None = None) -> dict[str, Any]:
        root = self.nodes[node_id or self.root_id]
        return {
            "id": root.id,
            "depth": root.depth,
            "state": root.state.value,
            "text": root.text,
            "final_answer": root.final_answer,
            "prm_score": root.prm_score,
            "branch_score": root.branch_score,
            "value": root.value,
            "visits": root.visits,
            "verification": root.verification.model_dump() if root.verification else None,
            "children": [self.to_nested(c) for c in root.children],
        }

    def summary(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "strategy": self.strategy,
            "nodes": len(self.nodes) - 1,
            "depth": max((n.depth for n in self.nodes.values()), default=0),
            "pruned": self.metrics.nodes_pruned,
            "answer": self.answer,
            "metrics": self.metrics.model_dump(),
        }


class StrategyOutput(BaseModel):
    tree: ReasoningTree
    answer: str | None = None
    rationale_node_id: str | None = None
    votes: dict[str, int] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


class ScoreCard(BaseModel):
    """Per-problem result used by the metrics/report layer."""

    problem_id: str
    strategy: str
    subject: str = "math"
    level: str = ""
    answer: str | None = None
    gold: str | None = None
    correct: bool = False
    confidence: float | None = None
    steps: int = 0
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    verify_status: VerifyStatus = VerifyStatus.NOT_CHECKED
    latency_ms: float = 0.0
    notes: str = ""
