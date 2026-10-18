"""Feature vector for the learned PRM head.

Deliberately small and interpretable: the shallow head reweights symbolic
verifier evidence instead of replacing it, which is what makes the model
auditable and trainable on a CPU.
"""

from __future__ import annotations

import re

from ...schema import Step, VerifyStatus
from .. import math_text as mt
from .base import StepScore

FEATURE_NAMES: tuple[str, ...] = (
    "symbolic_score",
    "status_pass",
    "status_fail",
    "status_unknown",
    "status_not_checked",
    "n_equations",
    "n_pass",
    "n_fail",
    "n_unknown",
    "has_final_answer",
    "is_first",
    "depth_norm",
    "prefix_norm",
    "char_len_norm",
    "digit_density",
    "operator_density",
    "latex_density",
    "unit_word",
    "cumulative_fail",
)

UNIT_WORD_RE = re.compile(
    r"\b(m|kg|s|seconds?|hours?|m/s|km/h|newton|joule|watt|volt|ampere|ohm|hertz|mol|kelvin)\b", re.I
)
OPERATOR_RE = re.compile(r"[+\-*/^=<>]")
LATEX_RE = re.compile(r"\\[a-zA-Z]+")


def step_features(prefix: list[Step], step: Step, score: StepScore | None = None) -> list[float]:
    """Build the feature row for ``step`` given the ``prefix`` before it."""
    status = score.status if score else VerifyStatus.NOT_CHECKED
    checks = score.checks if score else []
    text = step.text or ""
    equations = mt.extract_equations(text)
    n_pass = sum(1 for c in checks if c.get("status") == VerifyStatus.PASS)
    n_fail = sum(1 for c in checks if c.get("status") == VerifyStatus.FAIL)
    n_unknown = sum(1 for c in checks if c.get("status") == VerifyStatus.UNKNOWN)
    depth = len(prefix)
    digits = sum(ch.isdigit() for ch in text)
    return [
        score.score if score else 0.5,
        1.0 if status == VerifyStatus.PASS else 0.0,
        1.0 if status == VerifyStatus.FAIL else 0.0,
        1.0 if status == VerifyStatus.UNKNOWN else 0.0,
        1.0 if status == VerifyStatus.NOT_CHECKED else 0.0,
        float(len(equations)),
        float(n_pass),
        float(n_fail),
        float(n_unknown),
        1.0 if step.answer else 0.0,
        1.0 if depth == 0 else 0.0,
        depth / 20.0,
        len(prefix) / 20.0,
        min(1.0, len(text) / 300.0),
        digits / max(1.0, len(text)),
        len(OPERATOR_RE.findall(text)) / max(1.0, len(text)),
        len(LATEX_RE.findall(text)) / max(1.0, len(text)),
        1.0 if UNIT_WORD_RE.search(text) else 0.0,
        min(3.0, float(n_fail + sum(1 for s in prefix if "=" not in s.text and not s.text.strip()))),
    ]
