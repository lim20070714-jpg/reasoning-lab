"""Prompt templates for every strategy.

The ``<question>`` / ``<prior_steps>`` tags are part of the contract: the
offline MockLLM parses them to replay scripted chains, and the tree-search
prompts reuse them so interventions can be replayed verbatim.
"""

from __future__ import annotations

from ..llm.base import Prompt
from ..schema import Message, Problem, Step

ANSWER_INSTRUCTION = (
    "Give the complete reasoning, then end with exactly one line in this format:\n"
    "Final Answer: <the answer only, no sentences>"
)

SYSTEM = (
    "You are a careful mathematician and physicist. Reason precisely, keep every "
    "arithmetic statement explicit and self-contained, and never skip a calculation."
)

FEW_SHOT = (
    "Example:\n"
    "Question: <question>What is 3*(4+5)?</question>\n"
    "1. Compute the parenthesised term: 4+5=9.\n"
    "2. Multiply: 3*9=27.\n"
    "Final Answer: 27"
)


def _question_block(problem: Problem) -> str:
    return f"<question>{problem.question.strip()}</question>"


def _prior_block(steps: list[Step]) -> str:
    if not steps:
        return "<prior_steps>(nothing written yet)</prior_steps>"
    body = "\n".join(f"{s.index + 1 if s.index else i + 1}. {s.text}" for i, s in enumerate(steps))
    return f"<prior_steps>{body}</prior_steps>"


def _prompt(*blocks: str, tag: str) -> Prompt:
    return Prompt(messages=[Message(role="system", content=SYSTEM), Message(role="user", content="\n\n".join(blocks))], tag=tag)


def cot(problem: Problem, *, few_shot: bool = False, extra: str = "") -> Prompt:
    blocks = [f"Problem: {_question_block(problem)}", "Solve the problem step by step."]
    if few_shot:
        blocks.insert(1, FEW_SHOT)
    if extra:
        blocks.append(extra)
    blocks.append(ANSWER_INSTRUCTION)
    return _prompt(*blocks, tag="cot")


STEP_TERMINATION = (
    "If this step completes the solution, append `Final Answer: <the answer only>` "
    "at the end of that same line; otherwise do not mention the final answer."
)


def chain_step(problem: Problem, steps: list[Step], *, hint: str = "") -> Prompt:
    """Emit exactly one next step. Used by ToT and MCTS expansion."""
    blocks = [
        f"Problem: {_question_block(problem)}",
        _prior_block(steps),
        "Write ONLY the next single step of the solution. Do not restate earlier steps. "
        "If the step is a calculation, show the arithmetic explicitly.",
        STEP_TERMINATION,
    ]
    if hint:
        blocks.append(f"Direction for this step: {hint}")
    return _prompt(*blocks, tag="step")


def chain_step_candidates(problem: Problem, steps: list[Step], k: int) -> Prompt:
    """Ask for ``k`` distinct candidate next steps, numbered one per line."""
    blocks = [
        f"Problem: {_question_block(problem)}",
        _prior_block(steps),
        f"Propose exactly {k} distinct candidate next steps for this solution. "
        "Number them 1 to " + str(k) + ", one per line, each self-contained and showing its arithmetic. "
        "Cover different approaches rather than paraphrasing the same idea.",
        STEP_TERMINATION.replace("this step completes", "a candidate completes"),
    ]
    return _prompt(*blocks, tag="step_candidates")


def self_critique(problem: Problem, steps: list[Step]) -> Prompt:
    blocks = [
        f"Problem: {_question_block(problem)}",
        _prior_block(steps),
        "Audit the reasoning above. List every step that is wrong, unjustified, or unverifiable, "
        "then give the corrected final answer.",
        ANSWER_INSTRUCTION,
    ]
    return _prompt(*blocks, tag="critique")


def prm_judge(problem: Problem, steps: list[Step], candidate: Step) -> Prompt:
    """LLM-as-judge process reward prompt; the symbolic verifier is the default."""
    context = "\n".join(f"{i + 1}. {s.text}" for i, s in enumerate(steps)) or "(start of solution)"
    blocks = [
        f"Problem: {_question_block(problem)}",
        f"Steps so far:\n{context}",
        f"Candidate next step:\n{candidate.text}",
        "Decide whether the candidate step is mathematically valid given the steps so far. "
        "Reply in exactly two lines:\n"
        "Verdict: valid | invalid | uncertain\n"
        "Confidence: <number in 0..1>",
    ]
    return _prompt(*blocks, tag="prm_judge")


def value_estimate(problem: Problem, steps: list[Step]) -> Prompt:
    blocks = [
        f"Problem: {_question_block(problem)}",
        _prior_block(steps),
        "Estimate how likely this partial solution is to reach the correct final answer. "
        "Reply with one line: Value: <number in 0..1>",
    ]
    return _prompt(*blocks, tag="value")
