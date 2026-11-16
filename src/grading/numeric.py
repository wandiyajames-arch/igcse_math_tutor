"""Numeric, angle and bearing grading (pure Python, exact arithmetic).

Student values are exact ``Fraction``s from the typed literal; the precision
("place", nearest 10^q) is read from the text. Spec values are certified
enclosures (see ``enclosure``); a decision is taken only when it is the same at
both ends of the enclosure, otherwise the enclosure is refined and finally the
grader abstains ``EQUIVALENCE_UNCONFIRMED``.

Check order (first failure wins): parse → structure → unit → literal shape →
bearing format → required form → simplest form → value.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from .enclosure import Interval, NotNumeric, Uncertain, contains_pi, evaluate
from .parser import (Div, Group, Mul, Neg, Number, Pos, Pow, Quantity, Ratio, Relation,
                     Sequence, Tuple, parse)
from .verdict import Reason, Status, Verdict, abstain, correct, incorrect

PRECISIONS = (40, 80, 160)


# ------------------------------------------------------------ student literal
@dataclass(frozen=True)
class Literal:
    value: Fraction
    kind: str                      # integer | decimal | fraction | standard_form
    places: Optional[tuple[int, ...]]  # possible precision readings; None = exact (fraction)
    eff_place: Optional[int]       # place ignoring superfluous trailing zeros after the point
    int_digits: int = 0            # digits before the point (bearings)
    text: str = ""
    num: int = 0
    den: int = 1
    mantissa_ok: bool = True
    has_decimal_point: bool = False
    zero_after_point_only: bool = False


def _number_literal(num: Number, negative: bool) -> Literal:
    text = num.text
    v = num.value * (-1 if negative else 1)
    if "." in text:
        ip, fp = text.split(".")
        stripped = fp.rstrip("0")
        place = -len(fp)
        eff = -len(stripped) if stripped else 0
        return Literal(v, "decimal", (place,), eff, len(ip), text, has_decimal_point=True,
                       zero_after_point_only=(stripped == ""))
    tz = len(text) - len(text.rstrip("0")) if num.value != 0 else 0
    places = tuple(range(0, tz + 1))
    return Literal(v, "integer", places, 0, len(text), text)


def student_literal(node) -> Literal | Verdict:
    """Read a plain numeric literal (integer, decimal, fraction, standard form)."""
    negative = False
    while isinstance(node, Group):
        node = node.inner
    if isinstance(node, (Neg, Pos)):
        negative = isinstance(node, Neg)
        node = node.operand
        while isinstance(node, Group):
            node = node.inner
    if isinstance(node, Number):
        return _number_literal(node, negative)
    if isinstance(node, Div):
        a, b = node.left, node.right
        sign = -1 if negative else 1
        if isinstance(a, Group) and isinstance(a.inner, Neg) and isinstance(a.inner.operand, Number):
            sign, a = -sign, a.inner.operand
        if isinstance(b, Group) and isinstance(b.inner, Neg) and isinstance(b.inner.operand, Number):
            return abstain(Reason.NOT_AUTO_GRADABLE, "sign in a denominator")
        if (isinstance(a, Number) and isinstance(b, Number) and "." not in a.text and "." not in b.text):
            if b.value == 0:
                return abstain(Reason.NOT_AUTO_GRADABLE, "division by zero")
            v = sign * a.value / b.value
            return Literal(v, "fraction", None, None, 0, f"{a.text}/{b.text}",
                           num=int(a.value), den=int(b.value))
    if (isinstance(node, Mul) and node.style == "explicit" and isinstance(node.left, Number)
            and isinstance(node.right, Pow) and isinstance(node.right.base, Number)
            and node.right.base.text == "10"):
        e = node.right.exp
        esign = 1
        if isinstance(e, Neg):
            esign, e = -1, e.operand
        if isinstance(e, Number) and "." not in e.text:
            exp = esign * int(e.value)
            m = node.left
            dec = len(m.text.split(".")[1]) if "." in m.text else 0
            v = m.value * Fraction(10) ** exp * (-1 if negative else 1)
            ok = 1 <= m.value < 10
            return Literal(v, "standard_form", (exp - dec,), exp - dec, 0, "", mantissa_ok=ok)
    return abstain(Reason.NOT_AUTO_GRADABLE, "not a plain number (symbolic or arithmetic answer)")


# ----------------------------------------------------------------- rounding
def round_half_up(x: Fraction, q: int) -> Fraction:
    """Round to a multiple of 10^q, ties away from zero (half-up on the magnitude)."""
    unit = Fraction(10) ** q
    m = abs(x) / unit
    n = m.numerator // m.denominator
    if m - n >= Fraction(1, 2):
        n += 1
    return (n if x >= 0 else -n) * unit


def truncate(x: Fraction, q: int) -> Fraction:
    unit = Fraction(10) ** q
    m = abs(x) / unit
    n = m.numerator // m.denominator
    return (n if x >= 0 else -n) * unit


def leading_place(x: Fraction) -> int:
    if x == 0:
        raise Uncertain("zero has no leading place")
    a = abs(x)
    p = len(str(a.numerator)) - len(str(a.denominator))
    while Fraction(10) ** p > a:
        p -= 1
    while Fraction(10) ** (p + 1) <= a:
        p += 1
    return p


def _same(f, e: Interval):
    a, b = f(e.lo), f(e.hi)
    if a != b:
        raise Uncertain("decision differs across the enclosure")
    return a


def _matches(f, e: Interval, v: Fraction) -> bool:
    """Certified test of v == f(E) for a monotone step function f."""
    a, b = f(e.lo), f(e.hi)
    if a == b:
        return v == a
    if v < min(a, b) or v > max(a, b):
        return False
    raise Uncertain("membership differs across the enclosure")


def _required_place(spec_acc, e: Interval) -> int:
    if spec_acc.kind == "dp":
        return -spec_acc.digits
    if e.contains_zero():
        raise Uncertain("enclosure around zero")
    lp = _same(leading_place, e)
    return lp - (spec_acc.digits - 1)


def _decide(lit: Literal, q: Optional[int], e: Interval, acc, allow_more_precise: bool) -> Verdict:
    v = lit.value
    if not e.is_point and e.lo <= v <= e.hi:
        raise Uncertain("answer inside the enclosure (could be exactly equal)")
    rnd = lambda p: (lambda x: round_half_up(x, p))
    trn = lambda p: (lambda x: truncate(x, p))
    if acc.kind == "exact":
        if e.is_point and v == e.lo:
            return correct()
        if q is not None and (_matches(rnd(q), e, v) or _matches(trn(q), e, v)):
            return incorrect(Reason.ROUNDED_EXACT_VALUE, "exact value required")
        return incorrect(Reason.WRONG_VALUE)
    try:
        Q = _required_place(acc, e)
    except Uncertain:
        if e.is_point and v == e.lo:
            return correct("exactly equal")
        raise
    if not allow_more_precise and lit.eff_place is not None and lit.eff_place < Q:
        return incorrect(Reason.WRONG_FORM, "more figures than required")
    if e.is_point and v == e.lo:
        return correct("exactly equal")
    if q is None:  # fraction: exact value, judged at the required accuracy
        return correct() if _matches(rnd(Q), e, round_half_up(v, Q)) else incorrect(Reason.WRONG_VALUE)
    if q <= Q:
        if _matches(rnd(Q), e, round_half_up(v, Q)):
            return correct()
        if _matches(trn(q), e, v) and not _matches(rnd(q), e, v):
            return incorrect(Reason.ROUNDING_ERROR, "truncated instead of rounded")
        return incorrect(Reason.WRONG_VALUE)
    # fewer figures than required
    if _matches(rnd(q), e, v) or _matches(trn(q), e, v):
        return incorrect(Reason.INSUFFICIENT_ACCURACY)
    return incorrect(Reason.WRONG_VALUE)


def _decide_readings(lit: Literal, e: Interval, acc, allow_more_precise: bool) -> Verdict:
    if lit.places is None:
        return _decide(lit, None, e, acc, allow_more_precise)
    verdicts = [_decide(lit, q, e, acc, allow_more_precise) for q in lit.places]
    if len(set(verdicts)) == 1:
        return verdicts[0]
    if all(v.status is Status.INCORRECT for v in verdicts):
        return verdicts[0]  # finest reading (interpretation 7)
    return abstain(Reason.AMBIGUOUS_SIGNIFICANT_FIGURES, "integer trailing zeros (policy rule 4)")


def _value_verdict(lit: Literal, spec_node, acc, allow_more_precise: bool) -> Verdict:
    modes = ("calculator", "3.142") if contains_pi(spec_node) else ("calculator",)
    results = []
    for mode in modes:
        result = None
        for p in PRECISIONS:
            try:
                e = evaluate(spec_node, p, mode)
                result = _decide_readings(lit, e, acc, allow_more_precise)
                break
            except Uncertain:
                continue
        results.append(result or abstain(Reason.EQUIVALENCE_UNCONFIRMED, "could not certify"))
    for r in results:
        if r.status is Status.CORRECT:
            return r
    return results[0]


# ---------------------------------------------------------------- graders
def _unit_check(unit_given: Optional[str], spec_unit: Optional[str]) -> Optional[Verdict]:
    if unit_given is None:
        return None
    if spec_unit is None:
        return abstain(Reason.UNEXPECTED_UNIT, "unit given where none expected (policy rule 5)")
    if unit_given != spec_unit:
        return incorrect(Reason.WRONG_UNIT)
    return None


def _structure_check(node) -> Optional[Verdict]:
    if isinstance(node, Sequence):
        return abstain(Reason.MULTIPLE_ANSWERS, "several answers in one field (policy rule 3)")
    if isinstance(node, (Tuple, Ratio, Relation)):
        return incorrect(Reason.WRONG_FORM, "tuple, ratio or relation for a number")
    return None


def _split_quantity(node):
    if isinstance(node, Quantity):
        return node.value, (node.unit or node.currency), node
    return node, None, node


def grade_numeric_node(spec, node, kind: str = "numeric") -> Verdict:
    """Grade an already-parsed answer against a Numeric/Angle/Bearing spec."""
    v = _structure_check(node)
    if v:
        return v
    inner, unit, _ = _split_quantity(node)
    if isinstance(inner, Quantity):
        return incorrect(Reason.WRONG_FORM)
    spec_unit = getattr(spec, "unit", None)
    if kind in ("angle", "bearing"):
        spec_unit = "°"
    v = _unit_check(unit, spec_unit)
    if v:
        return v
    lit = student_literal(inner)
    if isinstance(lit, Verdict):
        return lit
    if kind == "bearing":
        if lit.kind in ("fraction", "standard_form"):
            return incorrect(Reason.WRONG_FORM, "bearing as a fraction or in standard form")
        if lit.value >= 0 and lit.int_digits != 3:
            return abstain(Reason.BEARING_NOT_THREE_FIGURE, "bearings use three figures (policy rule 7)")
    req = getattr(spec, "required_form", "any")
    if req == "integer" and not (lit.kind == "integer" or (lit.kind == "decimal" and lit.zero_after_point_only)):
        return incorrect(Reason.WRONG_FORM, "integer required")
    if req == "decimal" and lit.kind != "decimal":
        return incorrect(Reason.WRONG_FORM, "decimal required")
    if req == "fraction" and lit.kind != "fraction":
        return incorrect(Reason.WRONG_FORM, "fraction required")
    if req == "standard_form" and not (lit.kind == "standard_form" and lit.mantissa_ok):
        return incorrect(Reason.WRONG_FORM, "standard form required")
    if getattr(spec, "simplest_form", True) and lit.kind == "fraction":
        from math import gcd
        if lit.den == 1 or gcd(lit.num, lit.den) != 1:
            return incorrect(Reason.NOT_SIMPLIFIED, "fraction not in simplest form")
    spec_node = parse(spec.value).node
    spec_node = spec_node.value if isinstance(spec_node, Quantity) else spec_node
    if getattr(spec, "mark", "oe") == "cao":
        spec_lit = student_literal(spec_node)
        if isinstance(spec_lit, Literal) and spec_lit.kind != lit.kind:
            return incorrect(Reason.WRONG_FORM, "cao: different form")
    for alt in getattr(spec, "alternatives", ()):
        alt_node = parse(alt).node
        alt_lit = student_literal(alt_node.value if isinstance(alt_node, Quantity) else alt_node)
        if isinstance(alt_lit, Literal) and alt_lit.value == lit.value:
            return correct("matches a listed alternative")
    rng = getattr(spec, "accept_range", None)
    if rng:
        lo, hi = Fraction(rng[0]), Fraction(rng[1])
        return correct("inside the accepted range") if lo <= lit.value <= hi else incorrect(Reason.WRONG_VALUE)
    acc = spec.accuracy
    try:
        return _value_verdict(lit, spec_node, acc, getattr(spec, "allow_more_precise", True))
    except NotNumeric as e:
        return abstain(Reason.NOT_AUTO_GRADABLE, f"spec value not numeric: {e}")


def grade_numeric(spec, text: str) -> Verdict:
    kind = getattr(spec, "type", "numeric")
    res = parse(text)
    if not res.ok:
        return res.verdict
    try:
        return grade_numeric_node(spec, res.node, kind)
    except Exception as e:  # internal-error guard
        return abstain(Reason.WORKER_ERROR, f"internal error: {type(e).__name__}: {e}")
