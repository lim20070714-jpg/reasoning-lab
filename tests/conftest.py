from __future__ import annotations

import pytest

from reasoning_lab.config import LLMSettings
from reasoning_lab.llm.mock import MockLLM
from reasoning_lab.schema import Problem
from reasoning_lab.verification.prm.symbolic import SymbolicPRM

PROBLEM = Problem(
    id="t-apples",
    question="A shop packs 3 boxes with 4 apples each. Mia buys all boxes and eats 5 apples. How many apples are left?",
    answer="7",
    subject="math",
    level="arithmetic",
    steps=[
        "Total apples packed: 3*4=12.",
        "Mia eats 5, so 12-5=7.",
        "Therefore the number left is x = 7.",
    ],
    flawed=[
        ["Total apples packed: 3*4=13.", "Mia eats 5, so 13-5=8.", "Final Answer: 8"],
    ],
)


@pytest.fixture
def problem() -> Problem:
    return PROBLEM


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM(
        [
            {
                "match": "Mia buys all boxes",
                "steps": list(PROBLEM.steps),
                "answer": PROBLEM.answer,
                "flawed": list(PROBLEM.flawed),
            }
        ],
        seed=1,
    )


@pytest.fixture
def prm() -> SymbolicPRM:
    return SymbolicPRM()


@pytest.fixture
def mock_settings() -> LLMSettings:
    return LLMSettings(provider="mock", base_url="mock://", model="mock-reasoning")
