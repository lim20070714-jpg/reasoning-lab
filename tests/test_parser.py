from __future__ import annotations

from reasoning_lab.parser import extract_final_answer, render_steps, split_steps

NUMBERED = """1. Total packed: 3*4=12.
2. Mia eats 5, so 12-5=7.
3. Final Answer: 7"""

DASHED = """- Compute the perimeter: 2*(3+4)
- = 2*7 = 14
Final Answer: 14"""

PARAGRAPH = """First we note the box count is 3 and each holds 4 apples.

Multiplying gives 3*4=12 apples in total.

Subtracting the 5 eaten leaves 12-5=7."""


def test_split_numbered_steps():
    steps = split_steps(NUMBERED)
    assert [s.text for s in steps] == ["Total packed: 3*4=12.", "Mia eats 5, so 12-5=7."]
    assert steps[-1].answer == "7"


def test_split_dash_steps_keeps_wrapped_line_together():
    steps = split_steps(DASHED)
    assert len(steps) == 2
    assert "2*7 = 14" in steps[1].text


def test_split_paragraphs_when_unnumbered():
    assert len(split_steps(PARAGRAPH)) == 3


def test_split_sentences_as_last_resort():
    steps = split_steps("The mass is 2 kg. The acceleration is 3 m/s^2. So force is 6 N.")
    assert len(steps) >= 2


def test_final_answer_variants():
    assert extract_final_answer("blah\nFinal Answer: 42") == "42"
    assert extract_final_answer(r"so the value is \boxed{3\pi}") == r"3\pi"
    assert extract_final_answer("Working... x = 12") == "12"
    assert extract_final_answer("the total cost is 18 dollars") == "18 dollars"
    assert extract_final_answer("I am not sure how to start.") is None
    assert extract_final_answer("") is None


def test_render_steps_is_numbered_and_lossless():
    steps = split_steps(NUMBERED)
    rendered = render_steps(steps)
    for step in steps:
        assert step.text in rendered
    assert rendered.startswith("1. ")
