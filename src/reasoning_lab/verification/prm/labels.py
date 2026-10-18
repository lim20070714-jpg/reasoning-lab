"""Step-level supervision for the PRM head.

Three label sources, in increasing annotation cost:

``symbolic``   — verifier verdicts only; free, but silent on unverifiable steps.
``outcome``    — attribute the chain's correctness to the first failing step.
``rollout``    — Math-Shepherd/PRIME style: a step is good if continuing from it
                 usually reaches the gold answer. Uses no human step labels.

``pseudo_gold`` implements the TTRL idea: majority vote over sampled chains
supplies an outcome reward without any gold answer at all.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from ...llm.base import LLMClient
from ...parser import extract_final_answer, split_steps
from ...prompts import templates
from ...schema import GenerationParams, Problem, ReasoningTree, Step, VerifyStatus
from ..answer_check import answers_match
from .base import ProcessRewardModel, StepScore
from .features import step_features
from .learned import LabeledExample

STATUS_TO_LABEL = {VerifyStatus.PASS: 1.0, VerifyStatus.FAIL: 0.0}


async def score_chain(problem: Problem, steps: Sequence[Step], prm: ProcessRewardModel) -> list[StepScore]:
    return await prm.score_path(problem.id, list(steps))


def symbolic_examples(problem: Problem, steps: Sequence[Step], scores: Sequence[StepScore]) -> list[LabeledExample]:
    out: list[LabeledExample] = []
    for i, (step, score) in enumerate(zip(steps, scores, strict=True)):
        label = STATUS_TO_LABEL.get(score.status)
        if label is None:
            continue
        out.append(
            LabeledExample(
                problem_id=problem.id,
                depth=i,
                features=step_features(list(steps[:i]), step, score),
                label=label,
                source="symbolic",
            )
        )
    return out


def outcome_examples(
    problem: Problem, steps: Sequence[Step], scores: Sequence[StepScore], *, correct: bool
) -> list[LabeledExample]:
    """Back-attribute a chain-level result to the earliest suspect step."""
    first_bad = next((i for i, s in enumerate(scores) if s.status == VerifyStatus.FAIL), None)
    out: list[LabeledExample] = []
    for i, step in enumerate(steps):
        if correct:
            label = 1.0
        elif first_bad is None:
            label = 0.0 if i == len(steps) - 1 else 0.5
        else:
            label = 1.0 if i < first_bad else 0.0
        if label == 0.5:
            continue
        out.append(
            LabeledExample(
                problem_id=problem.id,
                depth=i,
                features=step_features(list(steps[:i]), step, scores[i] if i < len(scores) else None),
                label=label,
                source="outcome",
            )
        )
    return out


async def rollout_examples(
    problem: Problem,
    steps: Sequence[Step],
    llm: LLMClient,
    prm: ProcessRewardModel,
    *,
    k: int = 4,
    max_extra_steps: int = 6,
    temperature: float = 0.7,
    min_prefix: int = 2,
) -> list[LabeledExample]:
    """Label step *i* by the empirical success rate of continuations from it."""
    out: list[LabeledExample] = []
    params = GenerationParams(temperature=temperature, max_tokens=512)
    for depth in range(min_prefix, len(steps) + 1):
        prefix = list(steps[:depth])
        step = prefix[-1]
        base = await prm.score_step(problem.id, prefix[:-1], step)
        successes = 0
        for sample in range(k):
            chain = list(prefix)
            clone = params.model_copy(update={"seed": (params.seed or 0) + sample})
            for _ in range(max_extra_steps):
                prompt = templates.chain_step(problem, chain)
                completion = await llm.generate(prompt, clone, tag="rollout")
                added = split_steps(completion.text)
                if not added:
                    break
                chain.append(added[0])
                if extract_final_answer(completion.text):
                    break
            answer = extract_final_answer("\n".join(s.text for s in chain))
            matched, _ = answers_match(answer, problem.answer)
            successes += int(matched)
        out.append(
            LabeledExample(
                problem_id=problem.id,
                depth=depth - 1,
                features=step_features(prefix[:-1], step, base),
                label=successes / float(k),
                source="rollout",
            )
        )
    return out


def majority_answers(tree: ReasoningTree) -> list[tuple[str, int]]:
    return Counter(n.final_answer for n in tree.nodes_with_answers() if n.final_answer).most_common()


def pseudo_gold(tree: ReasoningTree, *, min_votes: int = 2) -> str | None:
    """TTRL-style label: the modal final answer across independent samples."""
    votes = majority_answers(tree)
    if not votes or votes[0][1] < min_votes:
        return None
    return votes[0][0]
