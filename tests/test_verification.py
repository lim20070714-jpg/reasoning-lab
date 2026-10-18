from __future__ import annotations

import sympy as sp

from reasoning_lab.schema import Step, VerifyStatus
from reasoning_lab.verification import math_text as mt
from reasoning_lab.verification.answer_check import answers_match, normalize_answer
from reasoning_lab.verification.symbolic import SymbolicVerifier


def step(text: str, index: int = 0) -> Step:
    return Step(index=index, text=text)


# -- text -> SymPy ----------------------------------------------------------
def test_parses_latex_fraction():
    assert mt.parse_math(r"\frac{3}{4}") == sp.Rational(3, 4)


def test_parses_sqrt_and_power():
    assert mt.parse_math(r"\sqrt{16} + 2^3") == 12


def test_rejects_prose():
    assert mt.parse_math("therefore the total apples") is None


def test_implicit_multiplication_and_unicode():
    assert mt.parse_math("2x") == 2 * sp.Symbol("x")
    assert mt.parse_math("5 × 3") == 15


def test_extract_equations_from_mixed_prose():
    text = "We know the area is \\(2x + 3 = 11\\), so x = 4 and the total is 12 apples."
    equations = mt.extract_equations(text)
    assert "2x+3=11" in [e.replace(" ", "") for e in equations]
    assert not any("area is" in e for e in equations)


def test_split_chained_equality():
    pairs = mt.split_equation("3*4 = 12 = 6*2")
    assert pairs == [("3*4", "12"), ("12", "6*2")]


# -- equality checking ------------------------------------------------------
def test_numeric_identity_passes():
    assert mt.check_pair("3*4", "12")[0] == VerifyStatus.PASS


def test_numeric_error_fails():
    status, detail, _ = mt.check_pair("3*4", "13")
    assert status == VerifyStatus.FAIL
    assert "12" in detail


def test_symbolic_setup_is_unknown_not_wrong():
    assert mt.check_pair("2x+3", "11")[0] == VerifyStatus.UNKNOWN


def test_identity_in_symbols_passes():
    assert mt.check_pair("(x+1)^2", "x^2+2x+1")[0] == VerifyStatus.PASS


def test_binding_then_substitution():
    status, _, bindings = mt.check_pair("x", "12")
    assert bindings and status == VerifyStatus.PASS
    after = mt.check_pair("x+3", "15", bindings)
    assert after[0] == VerifyStatus.PASS
    assert mt.check_pair("x+3", "16", bindings)[0] == VerifyStatus.FAIL


# -- path-aware verifier ----------------------------------------------------
def test_clean_chain_all_decided_pass():
    verifier = SymbolicVerifier()
    results = verifier.verify_path(
        [step("Total packed: 3*4=12."), step("Eaten: 12-5=7."), step("Left x = 7.")]
    )
    assert [r.status for r in results] == [VerifyStatus.PASS, VerifyStatus.PASS, VerifyStatus.PASS]
    assert all(r.score > 0.7 for r in results)


def test_error_is_caught_at_the_step_that_introduces_it():
    verifier = SymbolicVerifier()
    results = verifier.verify_path([step("Total packed: 3*4=13."), step("Then 13-5=8.")])
    assert results[0].status == VerifyStatus.FAIL
    assert results[0].score < 0.2


def test_contradiction_with_earlier_binding():
    verifier = SymbolicVerifier()
    results = verifier.verify_path([step("Let x = 7."), step("So x = 9.")])
    assert results[0].status == VerifyStatus.PASS
    assert results[1].status == VerifyStatus.FAIL


def test_prose_only_step_is_unknown():
    verification, _ = SymbolicVerifier().verify_step(step("First, understand what the question asks."))
    assert verification.status == VerifyStatus.UNKNOWN
    assert verification.score == 0.5


# -- answer matching --------------------------------------------------------
def test_normalize_strips_latex_and_answer_prefix():
    assert normalize_answer(r"Answer: $17$") == "17"
    assert normalize_answer(r"\boxed{2\sqrt{3}}") == r"2\sqrt{3}"


def test_numeric_formats_match():
    assert answers_match("0.5", r"\frac{1}{2}")[0]
    assert answers_match("1,000", "1000")[0]
    assert not answers_match("0.5", "0.6")[0]


def test_units_and_mcq_tolerant():
    assert answers_match("12 V", "12")[0] is False or answers_match("12 V", "12V")[0]
    assert answers_match("B", "B. 12 V") == (True, "mcq")
    assert answers_match("(C) 5 m/s", "C") == (True, "mcq")


def test_symbolic_answers_match():
    assert answers_match("x^2+2x+1", "(x+1)^2")[0]


def test_empty_answer_never_matches():
    assert answers_match(None, "7") == (False, "empty")
    assert answers_match("", "7") == (False, "empty")
