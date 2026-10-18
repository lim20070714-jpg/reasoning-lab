"""Calibrated wrapper around the symbolic step verifier.

Raw verifier verdicts are mapped to step probabilities with an explicit
calibration table, then threaded along a path so shared tree prefixes reuse
one verifier state instead of replaying the whole chain per candidate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ...schema import CheckResult, Step, Verification, VerifyStatus
from ..symbolic import INITIAL_STATE, SymbolicVerifier, VerifierState
from .base import Aggregator, ProcessRewardModel, StepScore


@dataclass(frozen=True)
class Calibration:
    """Step-correctness priors per verifier outcome.

    These are the only hand-set numbers in the scoring path; the learned head
    replaces them with values fitted from labels, and the report documents the
    sensitivity of the ranking to this table.
    """

    pass_p: float = 0.95
    fail_p: float = 0.05
    setup_p: float = 0.85
    unparseable_p: float = 0.75
    no_relation_p: float = 0.80
    unknown_p: float = 0.78

    def check_probability(self, check: CheckResult) -> float:
        if check.status == VerifyStatus.PASS:
            return self.pass_p
        if check.status == VerifyStatus.FAIL:
            return self.fail_p
        return {
            "symbolic setup": self.setup_p,
            "unparseable": self.unparseable_p,
            "nothing to check": self.no_relation_p,
        }.get(check.detail, self.unknown_p)

    def verification_probability(self, verification: Verification) -> float:
        if not verification.checks:
            return self.no_relation_p
        if any(c.status == VerifyStatus.FAIL for c in verification.checks):
            return self.fail_p
        if all(c.status == VerifyStatus.PASS for c in verification.checks):
            return self.pass_p
        return max(self.check_probability(c) for c in verification.checks)

    @staticmethod
    def clamp(value: float) -> float:
        return max(0.001, min(0.999, value))


class SymbolicPRM(ProcessRewardModel):
    name = "symbolic"

    def __init__(
        self,
        verifier: SymbolicVerifier | None = None,
        calibration: Calibration | None = None,
        *,
        aggregator: Aggregator = "geomean",
    ):
        self.verifier = verifier or SymbolicVerifier()
        self.calibration = calibration or Calibration()
        self.aggregator = aggregator
        self._states: dict[str, VerifierState] = {}
        self.verify_calls = 0

    def _prefix_key(self, prefix: tuple[Step, ...]) -> str:
        digest = hashlib.sha256("\u241f".join(s.text for s in prefix).encode("utf-8")).hexdigest()
        return digest[:24]

    def state_for(self, prefix: list[Step]) -> VerifierState:
        """Replay the verifier over the prefix, memoising every visited depth."""
        if not prefix:
            return INITIAL_STATE
        key = self._prefix_key(tuple(prefix))
        cached = self._states.get(key)
        if cached is not None:
            return cached
        state = INITIAL_STATE
        for depth in range(1, len(prefix) + 1):
            sub = tuple(prefix[:depth])
            memo = self._states.get(self._prefix_key(sub))
            if memo is None:
                _, state = self.verifier.verify_step(prefix[depth - 1], state)
                self._states[self._prefix_key(sub)] = state
            else:
                state = memo
        self._states[key] = state
        return state

    async def score_step(self, problem_id: str, prefix: list[Step], step: Step) -> StepScore:
        state = self.state_for(prefix)
        verification, updated = self.verifier.verify_step(step, state)
        self.verify_calls += 1
        self._states[self._prefix_key(tuple([*prefix, step]))] = updated
        score = self.calibration.clamp(self.calibration.verification_probability(verification))
        return StepScore(
            score=round(score, 5),
            status=verification.status,
            detail=verification.notes or (verification.checks[0].detail if verification.checks else ""),
            source=self.name,
            checks=[c.model_dump(mode="json") for c in verification.checks],
        )

    def clear_cache(self) -> None:
        self._states.clear()
