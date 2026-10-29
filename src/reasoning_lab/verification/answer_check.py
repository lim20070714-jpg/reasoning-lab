"""Answer equivalence used for scoring: string, numeric, symbolic and MCQ."""

from __future__ import annotations

import re

import sympy as sp

from . import math_text as mt

MCQ_RE = re.compile(r"^\s*\(?([A-E])\)?\s*[.、:．]?\s*")
LATEX_WRAPPER_RE = re.compile(r"\\(?:text|mathrm|textbf|boxed|operatorname|bm)\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
UNIT_RE = re.compile(
    r"\s*(?:%|degrees?|(?:m/s|km/h|m\^?[23]|cm\^?[23]|mm|km|kg|mol|rad/s|rad|Hz|hours?|minutes?|seconds?|"
    r"ohm|degree|cm|g|s|ms|min|h|m|N|J|W|V|A|K|L|°C)\b)\s*$"
)
THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")


def _strip_math_delimiters(text: str) -> str:
    """Drop ``$…$`` / ``$$…$$`` / ``\\(…\\)`` / ``\\[…\\]`` wrappers around an answer."""
    out = re.sub(r"^\s*(?:\$\$|\\\(|\\\[)\s*", "", text)
    out = re.sub(r"\s*(?:\$\$|\\\)|\\\])\s*[.,;]?$", "", out)
    return out


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    s = text.strip()
    s = re.sub(r"^\s*(?:the\s+)?(?:final\s+)?answer\s*[:=：]?\s*", "", s, flags=re.I)
    s = LATEX_WRAPPER_RE.sub(r"\1", s)
    s = s.replace("\\!", "").replace("\\,", "").replace("~", " ")
    s = _strip_math_delimiters(s)
    s = s.strip().strip("$").strip()
    s = mt.normalize_unicode(s)
    s = THOUSANDS_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip().rstrip(".。").strip()
    return s


def _mcq(text: str) -> str | None:
    match = MCQ_RE.match(text)
    return match.group(1) if match else None


def _as_number(text: str) -> float | None:
    expr = mt.parse_math(text)
    if expr is None or expr.free_symbols:
        return None
    return mt.numeric_value(expr)


def _strip_units(text: str) -> str:
    return UNIT_RE.sub("", text).strip()


def answers_match(predicted: str | None, gold: str | None) -> tuple[bool, str]:
    """Return ``(matched, method)``; the method is recorded so scorecards stay auditable."""
    pred, ref = normalize_answer(predicted), normalize_answer(gold)
    if not pred or not ref:
        return (False, "empty")
    if pred.casefold() == ref.casefold():
        return (True, "exact")

    pm, rm = _mcq(pred), _mcq(ref)
    if pm and rm:
        return (pm == rm, "mcq")

    pnum, rnum = _as_number(_strip_units(pred)), _as_number(_strip_units(ref))
    if pnum is not None and rnum is not None:
        return (mt.is_close(pnum, rnum), "numeric")

    pe, re_ = mt.parse_math(pred), mt.parse_math(ref)
    if pe is not None and re_ is not None:
        try:
            if not (pe.free_symbols or re_.free_symbols):
                if sp.simplify(pe - re_) == 0:
                    return (True, "symbolic")
                value = mt.numeric_value(pe - re_)
                if value is not None:
                    return (mt.is_close(value, 0.0), "symbolic")
            elif pe.free_symbols == re_.free_symbols and mt.is_identity(pe, re_) is True:
                return (True, "symbolic")
        except (TypeError, ValueError, AttributeError):
            pass

    pset, rset = _set_of(pred), _set_of(ref)
    if pset and rset:
        return (pset == rset, "set")

    if _strip_units(pred) and _strip_units(pred).casefold() == _strip_units(ref).casefold():
        return (True, "unit-insensitive")
    return (False, "no-match")


def _set_of(text: str) -> frozenset[str] | None:
    inner = re.match(r"^\{(.*)\}$", text.strip())
    if inner:
        parts = [p.strip() for p in inner.group(1).split(",")]
    elif "," in text and len(text.split(",")) <= 6 and all(len(p.strip()) < 24 for p in text.split(",")):
        parts = [p.strip() for p in text.split(",")]
    else:
        return None
    if len(parts) < 2 or any(not p for p in parts):
        return None
    return frozenset(normalize_answer(p) for p in parts)
