"""LaTeX/plain-text math converted to SymPy objects, and per-equation checking.

The verifier is intentionally conservative: a statement that cannot be parsed
or that introduces a fresh symbolic relation is reported as UNKNOWN rather than
counted as an error. Reasoning models legitimately *set up* equations, and
treating setup as a mistake would poison the process reward signal.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
import warnings
from functools import lru_cache

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    function_exponentiation,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from ..schema import VerifyStatus

TRANSFORMS = (
    *standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)

DOLLAR_RE = re.compile(r"\$([^$]+)\$")
LATEX_INLINE_RE = re.compile(r"\\\((.+?)\\\)", re.S)
LATEX_DISPLAY_RE = re.compile(r"\\\[(.+?)\\\]", re.S)
ENV_RE = re.compile(r"\\begin\{(?:equation|align|gather)[^}]*\}(.*?)\\end\{(?:equation|align|gather)[^}]*\}", re.S)
BARE_MATH_RE = re.compile(
    r"(?:^|[\s,;:(])([A-Za-z0-9\.\+\-\*/\^\(\)\[\]{}\\,=<>\s]*=[A-Za-z0-9\.\+\-\*/\^\(\)\[\]{}\\,=<>\s]*[A-Za-z0-9])"
)
KNOWN_WORDS = {
    "sqrt", "root", "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan",
    "log", "ln", "lg", "exp", "abs", "max", "min", "mod", "gcd", "lcm", "sum", "prod",
    "int", "lim", "floor", "ceil", "sign", "Im", "Re", "area", "speed", "force", "mass",
    "time", "volt", "ohm", "mol", "joule", "watt", "newton", "hertz", "kelvin", "radius",
}
GREEK_NAMES = [
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "upsilon",
    "phi", "varphi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega",
]
LATEX_GREEK = {name: name for name in GREEK_NAMES}
# a bare theta after latex_to_source is math, not a prose word
KNOWN_WORDS |= {name.lower() for name in GREEK_NAMES}
PROSE_WORD_RE = re.compile(r"^[A-Za-z]{2,}$")
STOPWORDS = {
    "so", "if", "it", "we", "is", "in", "on", "by", "as", "at", "be", "an", "to", "or", "no", "of",
    "and", "then", "thus", "get", "gives", "here", "when", "with", "that", "this", "use", "set",
    "call", "let", "hence", "since", "because", "therefore", "which", "where", "substitute",
    "simplify", "solve", "compute", "divide", "multiply", "add", "subtract", "equals", "equal",
    "total", "result", "answer", "value", "find", "first", "next", "finally", "about", "around",
}


FRAC_RE = re.compile(r"\\(?:d|t)?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
SQRT_RE = re.compile(r"\\(?:d|t)?sqrt(?:\[[^\]]*\])?\s*\{([^{}]*)\}")
TEXT_RE = re.compile(r"\\(?:text|mathrm|mathbf|operatorname|bm)\s*\{([^{}]*)\}")
UNITS_RE = re.compile(
    r"\\(?:left|right|,|;|!|:|quad|qquad|displaystyle|limits|cdot|times|div|pm|mp|circ|degree)"
)
SUBSUP_RE = re.compile(r"[_^]\{([^{}]*)\}")
NUM_EQ_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")

UNICODE_MAP = {
    "×": "*",
    "÷": "/",
    "−": "-",
    "–": "-",
    "—": "-",
    "＝": "=",
    "≈": "=",
    "≤": "<=",
    "≥": ">=",
    "√": "sqrt",
    "π": "pi",
    "∞": "oo",
    "²": "^2",
    "³": "^3",
    "±": "+-",
    "∑": "sum",
    "∫": "int",
    "°": "",
    "％": "%",
    "，": ",",
}

MAX_FREE_SYMBOLS = 6
ABS_TOL = 1e-7
REL_TOL = 1e-6
IDENTITY_SAMPLES = (2, 3, 5, 7)


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for src, dst in UNICODE_MAP.items():
        text = text.replace(src, dst)
    return text


def latex_to_source(raw: str) -> str:
    """Best-effort LaTeX -> Python/SymPy source. Not a general TeX engine."""
    s = normalize_unicode(raw)
    s = s.replace("\\%", "%").replace("\\$", "")
    s = TEXT_RE.sub(r"\1", s)
    for _ in range(4):
        s2 = FRAC_RE.sub(r"((\1)/(\2))", s)
        if s2 == s:
            break
        s = s2
    s = SQRT_RE.sub(r"sqrt(\1)", s)
    s = UNITS_RE.sub(" ", s)
    s = s.replace("\\_", "_").replace("\\{", "(").replace("\\}", ")")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    for _ in range(4):
        s2 = SUBSUP_RE.sub(lambda m: ("**" if m.group(0).startswith("^") else "_") + f"({m.group(1)})", s)
        if s2 == s:
            break
        s = s2
    s = re.sub(r"\\([a-zA-Z]+)", lambda m: f" {LATEX_GREEK.get(m.group(1), '')} ", s)
    s = s.replace("{", "(").replace("}", ")").replace("|", "")
    s = re.sub(r"\s*\\\s*", " ", s)
    s = s.replace("**_", "**")
    return re.sub(r"[,]\s*(?=\d)", "", s).strip()


@lru_cache(maxsize=8192)
def parse_math(raw: str) -> sp.Expr | None:
    """Parse one side of an equation, or the whole expression."""
    src = latex_to_source(raw)
    if not src:
        return None
    src = src.strip().strip(";").strip()
    if src.endswith("="):
        src = src[:-1]
    if "=" in src or re.search(r"[<>]", src):
        return None
    if not looks_like_math(src):
        return None
    if "%" in src:
        src = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", src)
    try:
        with warnings.catch_warnings():
            # compiling a malformed candidate raises SyntaxWarning, which is
            # already handled by the UNKNOWN verdict below
            warnings.simplefilter("ignore", SyntaxWarning)
            expr = parse_expr(src, transformations=TRANSFORMS, evaluate=True)
    except Exception:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                expr = sp.sympify(src)
        except Exception:
            return None
    if expr is None or isinstance(expr, (bool, sp.core.relational.Relational)):
        return None
    if not isinstance(expr, sp.Basic):
        return None
    if expr.has(sp.EulerGamma) or (expr.free_symbols and len(expr.free_symbols) > MAX_FREE_SYMBOLS):
        return None
    return expr


def _droppable(token: str, others: list[str]) -> bool:
    bare = token.strip("(),;:")
    if not PROSE_WORD_RE.match(bare):
        return False
    lowered = bare.lower()
    if lowered in KNOWN_WORDS:
        return False
    if lowered in STOPWORDS:
        return True
    return len(lowered) >= 3 and not any(lowered in other.lower() for other in others)


def strip_prose_side(side: str) -> str:
    """Trim connective words from both ends of one side of an equation."""
    parts = side.strip().split()
    while parts and _droppable(parts[0], parts[1:]):
        parts.pop(0)
    while parts and _droppable(parts[-1], parts[:-1]):
        parts.pop()
    return " ".join(parts).strip()


def looks_like_math(source: str) -> bool:
    """Reject spans that are really sentences.

    Without this guard SymPy's implicit multiplication silently turns prose into
    a product of single-letter symbols, and the verifier confidently "checks"
    nonsense instead of reporting that it could not parse the step.
    """
    # \times, \cdot, \frac are operators, not words; count only letters that
    # are not part of a TeX control sequence
    stripped = re.sub(r"\\[a-zA-Z]+", " ", source)
    words = [w for w in re.findall(r"[A-Za-z]{2,}", stripped)]
    unknown = [w for w in words if w.lower() not in KNOWN_WORDS]
    return not any(len(w) >= 5 for w in unknown) and len(unknown) < 2


def split_equation(text: str) -> list[tuple[str, str]]:
    """Split a (possibly chained) equality into pairwise sides."""
    parts = [strip_prose_side(p) for p in re.split(r"=", text) if p.strip()]
    if len(parts) < 2:
        return []
    if len(parts) > 4:
        parts = parts[:4]
    return [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]


def extract_equations(text: str) -> list[str]:
    """Pull candidate equations out of a step, preferring delimited math."""
    chunks: list[str] = []
    for regex in (ENV_RE, LATEX_DISPLAY_RE, LATEX_INLINE_RE, DOLLAR_RE):
        for match in regex.finditer(text):
            chunks.append(match.group(1))
    remainder = text
    for regex in (ENV_RE, LATEX_DISPLAY_RE, LATEX_INLINE_RE, DOLLAR_RE):
        remainder = regex.sub(" ", remainder)
    for match in BARE_MATH_RE.finditer(remainder):
        span = (match.group(1) or "").strip(" ,;:.")
        if span.count("=") >= 1 and 2 < len(span) <= 120:
            chunks.append(span)
    out: list[str] = []
    for chunk in chunks:
        # a comma followed by a word is prose ("12 apples, so x=4"), not a
        # thousands separator or a coordinate list
        for line in re.split(r"[;\n]|,\s+(?=[A-Za-z])", chunk):
            line = line.strip().strip(".").strip()
            if line.count("=") < 1 or len(line) <= 2:
                continue
            sides = [strip_prose_side(side) for side in re.split(r"=", line)]
            sides = [side.strip(" \t.") for side in sides]
            if len(sides) < 2 or any(not side for side in sides):
                continue
            if not all(looks_like_math(side) for side in sides):
                continue
            equation = "=".join(sides)
            if equation not in out:
                out.append(equation)
    return out


def extract_expressions(text: str) -> list[str]:
    """Non-equation math fragments (useful for feature extraction)."""
    found: list[str] = []
    for match in DOLLAR_RE.finditer(text):
        value = match.group(1).strip()
        if value and "=" not in value and len(value) > 1:
            found.append(value)
    return found


def is_close(a: complex | float, b: complex | float) -> bool:
    diff = abs(complex(a) - complex(b))
    scale = max(abs(complex(a)), abs(complex(b)), 1.0)
    return diff <= max(ABS_TOL, REL_TOL * scale)


def numeric_value(expr: sp.Basic) -> float | None:
    try:
        value = complex(sp.N(expr, 20))
    except (TypeError, ValueError, AttributeError):
        return None
    if value.real != value.real or abs(value.imag) > 1e-9:
        return None
    return value.real


def check_pair(lhs: str, rhs: str, bindings: dict[str, sp.Basic] | None = None) -> tuple[VerifyStatus, str, dict]:
    """Check one equality under known bindings.

    Returns the verdict plus any new single-symbol bindings derived from it,
    which is how later steps become checkable.
    """
    bindings = bindings or {}
    left = parse_math(_apply_bindings(lhs, bindings))
    right = parse_math(_apply_bindings(rhs, bindings))
    if left is None or right is None:
        return VerifyStatus.UNKNOWN, "unparseable", {}

    free = {str(s) for s in (left.free_symbols | right.free_symbols)}
    if not free:
        lv, rv = numeric_value(left), numeric_value(right)
        if lv is None or rv is None:
            return VerifyStatus.UNKNOWN, "non-numeric", {}
        if is_close(lv, rv):
            return VerifyStatus.PASS, f"{lhs} == {rhs}", {}
        return VerifyStatus.FAIL, f"{lv:g} != {rv:g}", {}

    # x = <fully determined expression> -> bind for later steps
    single_assign = re.match(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*$", lhs.strip())
    if single_assign and not right.free_symbols and len(free) == 1:
        name = single_assign.group(1)
        return VerifyStatus.PASS, f"bound {name}={right}", {name: right}

    # all symbols known -> substitute then compare numerically
    if free <= set(bindings):
        diff = sp.simplify(left - right)
        value = numeric_value(diff)
        if value is None:
            return VerifyStatus.UNKNOWN, "unevaluable", {}
        return (VerifyStatus.PASS, "", {}) if is_close(value, 0.0) else (
            VerifyStatus.FAIL,
            f"contradicts earlier values (residual {value:g})",
            {},
        )

    # an algebraic transformation that holds for every value is a valid step
    if is_identity(left, right) is True:
        return VerifyStatus.PASS, "identity", {}

    # a fresh relation over unknown symbols: legitimate setup, not an error
    return VerifyStatus.UNKNOWN, "symbolic setup", {}


def is_identity(left: sp.Basic, right: sp.Basic) -> bool | None:
    """Decide whether two sides agree for all values of their symbols."""
    symbols = sorted(left.free_symbols | right.free_symbols, key=str)
    if not symbols or len(symbols) > 3:
        return None
    try:
        diff = sp.expand(left - right)
    except (TypeError, ValueError, AttributeError):
        return None
    if diff == 0:
        return True
    if not diff.free_symbols:
        value = numeric_value(diff)
        return None if value is None else is_close(value, 0.0)
    for combo in itertools.product(IDENTITY_SAMPLES, repeat=len(symbols)):
        try:
            probe = complex(sp.N(diff.subs(dict(zip(symbols, combo, strict=True))), 20))
        except (TypeError, ValueError, AttributeError, ZeroDivisionError):
            return None
        if abs(probe) > 1e-8:
            return False
    return True


def _apply_bindings(text: str, bindings: dict[str, sp.Basic]) -> str:
    if not bindings:
        return text
    out = text
    for name, value in bindings.items():
        out = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", f"({sp.nsimplify(value)})", out)
    return out


def solve_for(bindings: dict[str, sp.Basic], equations: list[sp.Basic], symbols: set[str]) -> dict[str, sp.Basic]:
    """Resolve remaining unknowns once the accumulated equations determine them."""
    if not symbols or len(symbols) > 4 or len(equations) > 6:
        return {}
    syms = [sp.Symbol(name) for name in sorted(symbols)]
    try:
        solutions = sp.solve(equations, syms, dict=True)
    except (TypeError, ValueError, AttributeError, sp.SympifyError, NotImplementedError):
        # sympy raises NotImplementedError for transcendental systems it cannot
        # invert (gamma / zeta equations); unresolved is UNKNOWN, not an error
        return {}
    if not solutions:
        return {}
    # c^2 = 169 has two roots, and binding only the first one makes every later
    # step about c read as a contradiction, so keep a value only when the whole
    # solution set agrees on it
    out: dict[str, sp.Basic] = {}
    for sym in syms:
        values = [sol[sym] for sol in solutions if sym in sol and not sol[sym].free_symbols]
        if len(values) != len(solutions):
            continue
        if all(_same_value(values[0], value) for value in values[1:]):
            out[str(sym)] = values[0]
    return out


def _same_value(a: sp.Basic, b: sp.Basic) -> bool:
    av, bv = numeric_value(a), numeric_value(b)
    if av is not None and bv is not None:
        return is_close(av, bv)
    try:
        return sp.simplify(a - b) == 0
    except (TypeError, ValueError, AttributeError, sp.SympifyError):
        return False
