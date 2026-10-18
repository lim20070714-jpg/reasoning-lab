"""Scoring, calibration and oracle bounds computed from recorded traces.

Two design rules make the numbers defensible:

1. Strategies never see gold answers (``StrategyContext`` strips them), so
   "oracle" selection rates are recomputed here from the candidate list each
   run recorded, never inside a strategy.
2. Every aggregate is derived from :class:`EvalRecord` objects, which are
   written to disk by the runner. Reports re-read those records, so a report
   can be regenerated without spending model calls again.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..parser import split_steps
from ..schema import Problem, ScoreCard, Step, StrategyOutput, VerifyStatus
from ..strategies.base import terminal_nodes
from ..verification.answer_check import answers_match
from ..verification.symbolic import SymbolicVerifier

SelectionMode = Literal["first", "vote", "prm_mean", "prm_max", "oracle"]


class Candidate(BaseModel):
    node_id: str
    answer: str
    branch_score: float = 0.5
    prm_score: float | None = None
    depth: int = 0
    correct: bool = False
    rollout: bool = False


class EvalRecord(BaseModel):
    card: ScoreCard
    candidates: list[Candidate] = Field(default_factory=list)
    votes: dict[str, int] = Field(default_factory=dict)
    oracle_any: bool = False
    oracle_prm_correct: bool = False
    oracle_branch_score: float | None = None
    answer_branch_score: float | None = None
    error: str = ""

    @property
    def correct(self) -> bool:
        return self.card.correct


def _path_stats(tree, node_id: str | None) -> tuple[int, VerifyStatus, float | None]:
    """Steps, aggregate verification status and branch score for one answer node."""
    if node_id is None or node_id not in tree.nodes:
        return 0, VerifyStatus.NOT_CHECKED, None
    nodes = tree.path(node_id)
    steps = sum(1 for n in nodes if n.step is not None)
    statuses = [n.verification.status for n in nodes if n.verification is not None]
    if VerifyStatus.FAIL in statuses:
        status = VerifyStatus.FAIL
    elif statuses and all(s == VerifyStatus.PASS for s in statuses):
        status = VerifyStatus.PASS
    elif statuses:
        status = VerifyStatus.UNKNOWN
    else:
        status = VerifyStatus.NOT_CHECKED
    node = tree.nodes[node_id]
    branch = node.branch_score if node.branch_score is not None else node.prm_score
    return steps, status, branch


def select_answer(records_candidates: Sequence[Candidate], mode: SelectionMode) -> str | None:
    """Pick an answer from recorded candidates the way each policy would."""
    pool = [c for c in records_candidates if c.answer]
    if not pool:
        return None
    if mode == "first":
        return pool[0].answer
    if mode == "oracle":
        hits = [c for c in pool if c.correct]
        return hits[0].answer if hits else pool[0].answer
    tally: dict[str, list[Candidate]] = {}
    for candidate in pool:
        tally.setdefault(candidate.answer, []).append(candidate)
    if mode == "vote":
        ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        return ranked[0][0]
    # both PRM policies rank answers by branch score; they differ only in whether
    # an answer's evidence is averaged or taken from its strongest node
    aggregated = {
        answer: (sum(c.branch_score for c in cs) / len(cs) if mode == "prm_mean" else max(c.branch_score for c in cs))
        for answer, cs in tally.items()
    }
    ordered = sorted(aggregated.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[0][0]


def record_from_output(
    problem: Problem,
    output: StrategyOutput,
    *,
    latency_ms: float = 0.0,
    error: str = "",
) -> EvalRecord:
    tree = output.tree
    gold = problem.answer
    chosen = [c for c in tree.nodes_with_answers() if c.final_answer]
    candidates = [
        Candidate(
            node_id=node.id,
            answer=(node.final_answer or "").strip(),
            branch_score=node.branch_score if node.branch_score is not None else (node.prm_score or 0.5),
            prm_score=node.prm_score,
            depth=node.depth,
            correct=answers_match(node.final_answer, gold)[0],
            rollout=bool(node.meta.get("rollout")),
        )
        for node in chosen
    ]
    steps, status, branch = _path_stats(tree, output.rationale_node_id)
    correct, method = answers_match(output.answer, gold)
    terminals = terminal_nodes(tree)
    oracle_any = any(answers_match(t.final_answer, gold)[0] for t in terminals) or any(c.correct for c in candidates)
    best = max(candidates, key=lambda c: c.branch_score, default=None)
    card = ScoreCard(
        problem_id=problem.id,
        strategy=tree.strategy,
        subject=problem.subject,
        level=problem.level,
        answer=output.answer,
        gold=gold,
        correct=correct,
        confidence=branch,
        steps=steps,
        metrics=tree.metrics.model_copy(),
        verify_status=status,
        latency_ms=latency_ms,
        notes=method if not correct else "",
    )
    return EvalRecord(
        card=card,
        candidates=candidates,
        votes=dict(output.votes),
        oracle_any=oracle_any,
        oracle_prm_correct=bool(best and best.correct),
        oracle_branch_score=best.branch_score if best else None,
        answer_branch_score=branch,
        error=error,
    )


def expected_calibration_error(records: Sequence[EvalRecord], *, bins: int = 10) -> dict[str, Any]:
    """Reliability of ``confidence`` as a probability of being correct."""
    pairs = [(r.card.confidence, r.card.correct) for r in records if r.card.confidence is not None]
    if not pairs:
        return {"n": 0, "ece": None, "bins": []}
    per_bin: dict[int, list[tuple[float, bool]]] = {}
    for prob, correct in pairs:
        index = min(bins - 1, max(0, int(prob * bins)))
        per_bin.setdefault(index, []).append((prob, correct))
    total = len(pairs)
    ece = 0.0
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        bucket = per_bin.get(index)
        if not bucket:
            continue
        avg_prob = sum(p for p, _ in bucket) / len(bucket)
        acc = sum(1 for _, c in bucket if c) / len(bucket)
        ece += len(bucket) / total * abs(avg_prob - acc)
        rows.append(
            {
                "bin": f"[{index / bins:.1f}, {(index + 1) / bins:.1f})",
                "n": len(bucket),
                "avg_confidence": round(avg_prob, 4),
                "accuracy": round(acc, 4),
                "gap": round(avg_prob - acc, 4),
            }
        )
    return {"n": total, "ece": round(ece, 4), "bins": rows}


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Rank-based AUC with mid-rank tie handling; None when degenerate."""
    pairs = [(float(score), bool(label)) for score, label in zip(scores, labels, strict=True)]
    pos = sum(1 for _, flag in pairs if flag)
    neg = len(pairs) - pos
    if not pos or not neg:
        return None
    order = sorted(pairs, key=lambda p: p[0])
    sum_pos_ranks = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if order[k][1]:
                sum_pos_ranks += mid
        i = j + 1
    return round((sum_pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg), 4)


def discrimination_stats(records: Sequence[EvalRecord]) -> dict[str, Any]:
    """Does the process reward track correctness across whole runs?"""
    scored = [(r.card.confidence, r.card.correct) for r in records if r.card.confidence is not None]
    on_correct = [c for c, flag in scored if flag]
    on_wrong = [c for c, flag in scored if not flag]
    by_status: dict[str, list[bool]] = {}
    for record in records:
        by_status.setdefault(str(record.card.verify_status), []).append(record.card.correct)
    return {
        "n": len(records),
        "auc_confidence": auc([c for c, _ in scored], [flag for _, flag in scored]) if scored else None,
        "mean_conf_correct": round(sum(on_correct) / len(on_correct), 4) if on_correct else None,
        "mean_conf_wrong": round(sum(on_wrong) / len(on_wrong), 4) if on_wrong else None,
        "by_verify_status": {
            status: {"n": len(vals), "accuracy": round(sum(vals) / len(vals), 4)}
            for status, vals in sorted(by_status.items())
            if vals
        },
    }


def rerank_table(records: Sequence[EvalRecord]) -> dict[str, dict[str, float]]:
    """Accuracy under each answer-selection policy, replayed from recorded candidates."""
    out: dict[str, dict[str, float]] = {}
    for mode in ("first", "vote", "prm_mean", "prm_max", "oracle"):
        usable = [r for r in records if r.candidates]
        if not usable:
            continue
        hits = sum(1 for r in usable if answers_match(select_answer(r.candidates, mode), r.card.gold)[0])
        out[mode] = {"n": len(usable), "accuracy": round(hits / len(usable), 4)}
    return out


@dataclass
class _StepTally:
    """One population of steps (reference or flawed) counted by the verifier."""

    steps: int = 0
    rejected: int = 0
    scores: list[float] = field(default_factory=list)
    labels: list[bool] = field(default_factory=list)
    chain_scores: list[float] = field(default_factory=list)
    variants: int = 0
    flagged: int = 0


def verifier_discrimination(problems: Sequence[Problem]) -> dict[str, Any]:
    """Step-level check: reference steps should pass, flawed steps should fail.

    Runs the symbolic verifier over the bundled datasets only — no model calls,
    so this is a stable property of the verifier that the README can quote.
    """
    verifier = SymbolicVerifier()
    counts = {"pass": 0, "fail": 0, "unknown": 0, "not_checked": 0}
    reference = _StepTally()
    flawed = _StepTally()
    for problem in problems:
        if problem.steps:
            statuses = verifier.verify_path([Step(index=i, text=t) for i, t in enumerate(problem.steps)])
            reference.chain_scores.append(sum(r.score for r in statuses) / len(statuses))
            for result in statuses:
                counts[str(result.status)] = counts.get(str(result.status), 0) + 1
                reference.steps += 1
                if result.status == VerifyStatus.FAIL:
                    reference.rejected += 1
                reference.scores.append(result.score)
                reference.labels.append(True)
        for variant in problem.flawed:
            statuses = verifier.verify_path([Step(index=i, text=t) for i, t in enumerate(variant)])
            flawed.chain_scores.append(sum(r.score for r in statuses) / len(statuses))
            flawed.variants += 1
            if any(r.status == VerifyStatus.FAIL for r in statuses):
                flawed.flagged += 1
            for result in statuses:
                counts[str(result.status)] = counts.get(str(result.status), 0) + 1
                flawed.steps += 1
                if result.status == VerifyStatus.FAIL:
                    flawed.rejected += 1
                flawed.scores.append(result.score)
                flawed.labels.append(False)
    total = reference.steps + flawed.steps
    decided = counts["pass"] + counts["fail"]
    tp = flawed.rejected
    fn = flawed.steps - tp
    fp = reference.rejected
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    scores = reference.scores + flawed.scores
    labels = reference.labels + flawed.labels
    return {
        "steps": total,
        "reference_steps": reference.steps,
        "flawed_steps": flawed.steps,
        "flawed_variants": flawed.variants,
        "statuses": counts,
        "mean_step_score": {
            "reference": round(sum(reference.scores) / max(1, len(reference.scores)), 4),
            "flawed": round(sum(flawed.scores) / max(1, len(flawed.scores)), 4),
        },
        # chain-level AUROC ranks whole reference chains above whole flawed
        # chains by mean step score; step-level AUROC additionally depends on
        # how many steps of each flawed chain were corrupted, so both are kept
        "chain_auc": auc(
            flawed.chain_scores + reference.chain_scores,
            [False] * len(flawed.chain_scores) + [True] * len(reference.chain_scores),
        ),
        "decidable_pct": round(decided / total * 100.0, 2) if total else 0.0,
        "variant_detection_pct": (round(flawed.flagged / flawed.variants * 100.0, 2) if flawed.variants else 0.0),
        "reference_false_flag_pct": (round(fp / reference.steps * 100.0, 2) if reference.steps else 0.0),
        "flawed_step_rejection_pct": round(tp / flawed.steps * 100.0, 2) if flawed.steps else 0.0,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": f1,
        "step_auc": auc(scores, labels),
    }


def summarise(records: Sequence[EvalRecord]) -> dict[str, Any]:
    clean = [r for r in records if not r.error]
    errored = len(records) - len(clean)
    metrics = [r.card.metrics for r in clean]
    total_calls = sum(m.llm_calls for m in metrics)
    total_tokens = sum(m.total_tokens for m in metrics)
    correct_records = [r for r in clean if r.card.correct]
    return {
        "strategy": records[0].card.strategy if records else "",
        "n": len(records),
        "errors": errored,
        "accuracy": round(sum(1 for r in clean if r.card.correct) / len(clean), 4) if clean else None,
        "oracle_any_pct": round(sum(1 for r in clean if r.oracle_any) / len(clean) * 100.0, 2) if clean else None,
        "oracle_prm_pct": round(sum(1 for r in clean if r.oracle_prm_correct) / len(clean) * 100.0, 2) if clean else None,
        "mean_steps": round(sum(r.card.steps for r in clean) / len(clean), 2) if clean else None,
        "mean_branch_score": (
            round(sum(r.card.confidence for r in clean if r.card.confidence is not None)
                  / max(1, sum(1 for r in clean if r.card.confidence is not None)), 4)
            if clean
            else None
        ),
        "llm_calls": total_calls,
        "mean_llm_calls": round(total_calls / len(clean), 2) if clean else None,
        "cached_calls": sum(m.cached_calls for m in metrics),
        "cache_hit_pct": round(sum(m.cached_calls for m in metrics) / total_calls * 100.0, 2) if total_calls else 0.0,
        "total_tokens": total_tokens,
        "mean_tokens": round(total_tokens / len(clean), 1) if clean else None,
        "tokens_per_correct": round(total_tokens / len(correct_records), 1) if correct_records else None,
        "nodes_created": sum(m.nodes_created for m in metrics),
        "nodes_pruned": sum(m.nodes_pruned for m in metrics),
        "hard_failures": sum(m.hard_failures for m in metrics),
        "verifications": sum(m.verifications for m in metrics),
        "wall_time_s": round(sum(m.wall_time_s for m in metrics), 2),
        "mean_latency_ms": round(sum(m.latency_ms for m in metrics) / len(clean), 1) if clean else None,
        "calibration": expected_calibration_error(clean),
        "discrimination": discrimination_stats(clean),
        "rerank": rerank_table(clean),
    }


def group(
    records: Sequence[EvalRecord],
    key: str,
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[EvalRecord]] = {}
    for record in records:
        value = getattr(record.card, key) or "(unset)"
        buckets.setdefault(str(value), []).append(record)
    return {name: summarise(items) for name, items in sorted(buckets.items())}


def split_steps_text(text: str) -> list[str]:
    """Exposed for tests that build synthetic records without a dataset."""
    return [s.text for s in split_steps(text)]


__all__ = [
    "Candidate",
    "EvalRecord",
    "SelectionMode",
    "auc",
    "discrimination_stats",
    "expected_calibration_error",
    "group",
    "record_from_output",
    "rerank_table",
    "select_answer",
    "split_steps_text",
    "summarise",
    "verifier_discrimination",
]
