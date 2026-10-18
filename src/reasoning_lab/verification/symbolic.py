"""Path-aware symbolic step verifier.

A step is checked in the context of everything before it: numeric identities
are decided outright, single-symbol assignments are bound, and symbolic
relations are accumulated so that later steps become decidable. This is what
makes an unsound chain fail *at the step that introduces the error*, which is
the supervision signal the process reward model is trained on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from ..schema import CheckResult, Step, Verification, VerifyStatus
from . import math_text as mt


@dataclass(frozen=True)
class VerifierState:
    bindings: dict[str, sp.Basic] = field(default_factory=dict)
    equations: tuple[sp.Basic, ...] = ()
    unknown_symbols: frozenset[str] = frozenset()

    def with_bindings(self, extra: dict[str, sp.Basic]) -> VerifierState:
        if not extra:
            return self
        return VerifierState(
            bindings={**self.bindings, **extra},
            equations=self.equations,
            unknown_symbols=frozenset(set(self.unknown_symbols) - set(extra)),
        )

    def with_equation(self, expr: sp.Basic, symbols: set[str]) -> VerifierState:
        return VerifierState(
            bindings=self.bindings,
            equations=(*self.equations, expr),
            unknown_symbols=frozenset(set(self.unknown_symbols) | symbols),
        )


INITIAL_STATE = VerifierState()


class SymbolicVerifier:
    """Rule-based process verifier; the default PRM backend."""

    name = "symbolic"

    def verify_step(self, step: Step, state: VerifierState = INITIAL_STATE) -> tuple[Verification, VerifierState]:
        text = step.text or ""
        equations = mt.extract_equations(text)
        if not equations:
            numeric = self._prose_numbers(text)
            if numeric is not None:
                return numeric, state
            return (
                Verification(
                    status=VerifyStatus.UNKNOWN,
                    score=0.5,
                    checks=[CheckResult(kind="no_relation", status=VerifyStatus.UNKNOWN, detail="nothing to check")],
                    verifier=self.name,
                ),
                state,
            )

        checks: list[CheckResult] = []
        current = state
        for equation in equations:
            for lhs, rhs in mt.split_equation(equation):
                result, detail, new_bindings = mt.check_pair(lhs, rhs, current.bindings)
                weight = 1.0 if result in (VerifyStatus.PASS, VerifyStatus.FAIL) else 0.6
                checks.append(
                    CheckResult(
                        kind="equality",
                        expression=_trim(equation),
                        status=result,
                        detail=detail,
                        weight=weight,
                    )
                )
                if result == VerifyStatus.FAIL:
                    continue
                if new_bindings:
                    current = current.with_bindings(new_bindings)
                    continue
                expr = _difference(lhs, rhs, current.bindings)
                if expr is not None:
                    free = {str(s) for s in expr.free_symbols}
                    if free:
                        current = current.with_equation(expr, free)
                        resolved = mt.solve_for(current.bindings, list(current.equations), set(current.unknown_symbols))
                        if resolved:
                            current = current.with_bindings(resolved)

        verification = Verification.from_checks(checks, verifier=self.name)
        if verification.status == VerifyStatus.FAIL:
            verification.notes = "hard failure detected by symbolic checker"
        return verification, current

    def verify_path(self, steps: list[Step]) -> list[Verification]:
        results: list[Verification] = []
        state = INITIAL_STATE
        for step in steps:
            verification, state = self.verify_step(step, state)
            results.append(verification)
        return results

    @staticmethod
    def _prose_numbers(text: str) -> Verification | None:
        """Catch claims like ``3*4=12`` written without any delimiters."""
        for lhs, rhs in mt.split_equation(text if "=" in text else ""):
            left, right = mt.parse_math(lhs), mt.parse_math(rhs)
            if left is None or right is None or left.free_symbols or right.free_symbols:
                continue
            lv, rv = mt.numeric_value(left), mt.numeric_value(right)
            if lv is None or rv is None:
                continue
            status = VerifyStatus.PASS if mt.is_close(lv, rv) else VerifyStatus.FAIL
            return Verification.from_checks(
                [CheckResult(kind="inline_equality", expression=_trim(text), status=status, detail=f"{lv:g} vs {rv:g}")],
                verifier="symbolic",
            )
        return None


def _difference(lhs: str, rhs: str, bindings: dict[str, sp.Basic]) -> sp.Basic | None:
    left, right = mt.parse_math(mt._apply_bindings(lhs, bindings)), mt.parse_math(mt._apply_bindings(rhs, bindings))
    if left is None or right is None:
        return None
    try:
        return sp.simplify(left - right)
    except (TypeError, ValueError, AttributeError):
        return None


def _trim(text: str, limit: int = 120) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"
