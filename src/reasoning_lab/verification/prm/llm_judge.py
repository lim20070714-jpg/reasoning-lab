"""LLM-as-judge process reward model (optional, API-backed).

Included because it is the common baseline in PRM papers, but it is never the
default: judge scores are neither reproducible across runs nor free, whereas
the symbolic verifier is both.
"""

from __future__ import annotations

import re

from ...llm.base import LLMClient
from ...prompts import templates
from ...schema import Problem, Step, VerifyStatus
from .base import Aggregator, ProcessRewardModel, StepScore

VERDICT_RE = re.compile(r"verdict\s*[:：]\s*(valid|invalid|uncertain|sound|unsound)", re.I)
CONFIDENCE_RE = re.compile(r"confidence\s*[:：]\s*([01]?\.?\d+)", re.I)

STATUS_MAP = {
    "valid": VerifyStatus.PASS,
    "sound": VerifyStatus.PASS,
    "invalid": VerifyStatus.FAIL,
    "unsound": VerifyStatus.FAIL,
    "uncertain": VerifyStatus.UNKNOWN,
}


class LLMJudgePRM(ProcessRewardModel):
    name = "judge"

    def __init__(self, problem: Problem, llm: LLMClient, *, aggregator: Aggregator = "geomean"):
        self.problem = problem
        self.llm = llm
        self.aggregator = aggregator

    async def score_step(self, problem_id: str, prefix: list[Step], step: Step) -> StepScore:
        prompt = templates.prm_judge(self.problem, prefix, step)
        completion = await self.llm.generate(prompt, tag="prm_judge")
        verdict = VERDICT_RE.search(completion.text or "")
        confidence = CONFIDENCE_RE.search(completion.text or "")
        status = STATUS_MAP.get(verdict.group(1).lower(), VerifyStatus.UNKNOWN) if verdict else VerifyStatus.UNKNOWN
        value = float(confidence.group(1)) if confidence else 0.5
        value = max(0.0, min(1.0, value))
        score = {VerifyStatus.PASS: value, VerifyStatus.FAIL: 1.0 - value}.get(status, 0.5 + 0.3 * (value - 0.5))
        return StepScore(
            score=round(max(0.001, min(0.999, score)), 5),
            status=status,
            detail=(completion.text or "").strip()[:160],
            source=self.name,
            checks=[],
        )
