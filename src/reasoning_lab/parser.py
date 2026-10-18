"""Turning free-form model output into structured steps and answers."""

from __future__ import annotations

import re

from .schema import Problem, Step

STEP_MARKER_RE = re.compile(
    r"^\s*(?:step\s*)?\d+\s*[\.\)]\s+|^\s*\(\d+\)\s+|^\s*[-*•·]\s+|^\s*\d+\s*[、．]\s*",
    re.I,
)
FINAL_ANSWER_RE = re.compile(r"(?:final\s*answer|最终答案|答案|答)\s*[:：]\s*(.+?)\s*$", re.I | re.M)
ANSWER_IS_RE = re.compile(r"(?:the\s+)?(?:answer|result)\s+is\s*(?:\s*(?:=|equals)\s*)?\$?([^.\n]+)", re.I)
BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
LAST_ASSIGN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^=.\n]+?)\s*[.!]?\s*$")
TRAILING_VALUE_RE = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*([A-Za-z°%/]{1,9})?\s*[.。]?\s*$")
BULLET_PREFIX_RE = re.compile(r"^\s*(?:step\s*)?\d+\s*[\.\)]\s*|^\s*\(\d+\)\s*|^\s*[-*•·]\s*|\s*\d+\s*[、．]\s*", re.I)
SENTENCE_RE = re.compile(r"(?<=[.!?。；;])\s+")


def clean_step_text(line: str) -> str:
    return BULLET_PREFIX_RE.sub("", line, count=1).strip()


def split_steps(text: str) -> list[Step]:
    """Segment a chain of thought into atomic steps.

    Prefers explicit numbering; falls back to paragraphs, then to sentence
    groups, so downstream verification always has something to check.
    """
    if not text or not text.strip():
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    steps: list[str] = []
    current: list[str] = []
    for line in lines:
        if STEP_MARKER_RE.match(line):
            if current:
                steps.append(" ".join(current))
            current = [clean_step_text(line)]
        elif current:
            current.append(line)
        else:
            current = [line]
    if current:
        steps.append(" ".join(current))

    if len(steps) < 2:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paragraphs) > 1:
            steps = paragraphs
        else:
            sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
            steps = [" ".join(sentences[i : i + 2]) for i in range(0, len(sentences), 2)]

    final = extract_final_answer(text)
    out: list[Step] = []
    for i, raw in enumerate(steps):
        if len(steps) > 1 and re.search(r"final\s*answer|最终答案", raw, re.I):
            raw = FINAL_ANSWER_RE.sub("", raw).strip(" .\n")
            if not raw:
                continue
        step = Step(index=i, text=raw)
        if final and i == len(steps) - 1:
            step.answer = final
        out.append(step)
    if out and final:
        out[-1].answer = final
    return out


def explicit_final_answer(text: str) -> str | None:
    """Only a stated final answer — used to decide when a step ends the search.

    Kept separate from :func:`extract_final_answer` because a mid-chain step
    ending in a number must not be mistaken for the solution.
    """
    if not text:
        return None
    for match in reversed(list(FINAL_ANSWER_RE.finditer(text))):
        value = match.group(1).strip().strip("$.").strip()
        if value:
            return value
    boxed = BOXED_RE.findall(text)
    return boxed[-1].strip() if boxed else None


def extract_final_answer(text: str) -> str | None:
    if not text:
        return None
    for match in reversed(list(FINAL_ANSWER_RE.finditer(text))):
        value = match.group(1).strip().strip("$.").strip()
        if value:
            return value
    boxed = BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    for match in reversed(list(ANSWER_IS_RE.finditer(text))):
        value = match.group(1).strip().strip("$.").strip()
        if value:
            return value
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    # models sometimes forget the required format but still state "x = 12"
    tail = LAST_ASSIGN_RE.search(lines[-1])
    if tail:
        return tail.group(2).strip().strip("$.").strip()
    numeric = TRAILING_VALUE_RE.search(lines[-1])
    if numeric:
        value, unit = numeric.group(1), numeric.group(2)
        return f"{value} {unit}".strip() if unit else value
    return None


def render_steps(steps: list[Step]) -> str:
    return "\n".join(f"{i + 1}. {s.text}" for i, s in enumerate(steps))


def problem_text(problem: Problem) -> str:
    return problem.question.strip()
