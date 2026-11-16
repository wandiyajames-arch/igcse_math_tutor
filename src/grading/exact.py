"""Exact forms: in terms of π, surd form, fraction in simplest form (R4, R10, R11).

Pure Python, no floats, no SymPy. The answer's AST is checked for form and
simplest form, then reduced to a canonical linear combination with Fraction
coefficients — a + bπ, a + Σ bₘ√m (m square-free) or a rational — and
compared by exact equality (π is transcendental and roots of distinct
square-free integers are linearly independent over ℚ).

Check order (first failure wins): 0 spec self-check → 1 parse → 2 structure →
3 unit → 4 supported structure (abstain) → 5 form (WRONG_FORM) →
6 simplest form (NOT_SIMPLIFIED) → 7 value. NOT_SIMPLIFIED is decided before
the value, so it does not imply the value was right.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd, isqrt
from typing import Optional

from .parser import (Add, Div, Group, Mul, Neg, Number, Pi, Pos, Pow, Quantity, Ratio,
                     Relation, Sequence, Sqrt, Sub, Tuple, Variable, parse)
from .verdict import Reason, Verdict, abstain, correct, incorrect


def square_free_part(n: int) -> tuple[int, int]:
    """Return (k, m) with n = k²·m and m square-free (exact integer trial division)."""
    if n <= 0:
        raise ValueError("positive integers only")
    k, m, f = 1, 1, 2
    while f * f * f <= n:
        if n % f == 0:
            e = 0
            while n % f == 0:
                n //= f
                e += 1
            k *= f ** (e // 2)
            m *= f ** (e % 2)
        f += 1
    r = isqrt(n)
    if r * r == n and n > 1:
        k *= r
    else:
        m *= n
    return k, m


@dataclass
class Analysis:
    abstain: list = field(default_factory=list)
    wrong_form: list = field(default_factory=list)
    not_simplified: list = field(default_factory=list)
    canon: dict = field(default_factory=dict)
    has_pi: bool = False
    has_root: bool = False
    has_decimal: bool = False
    term_bases: list = field(default_factory=list)


def _is_sum(n) -> bool:
    return isinstance(n, (Add, Sub))


def _leftmost_signed(n) -> bool:
    while isinstance(n, (Mul, Div, Group)):
        n = n.inner if isinstance(n, Group) else n.left
    return isinstance(n, (Neg, Pos)) or (isinstance(n, Group) and isinstance(n.inner, (Neg, Pos)))


def _terms(node, a: Analysis):
    """Split the top-level sum into (sign, term) pairs; flag signs after an operator."""
    out = []

    def rec(n, sign):
        if isinstance(n, Add):
            rec(n.left, sign)
            if _leftmost_signed(n.right):
                a.not_simplified.append("sign straight after an operator")
            out.append((sign, n.right))
        elif isinstance(n, Sub):
            rec(n.left, sign)
            if _leftmost_signed(n.right):
                a.not_simplified.append("sign straight after an operator")
            out.append((-sign, n.right))
        else:
            out.append((sign, n))
    rec(node, 1)
    return out


class _Term:
    def __init__(self, a: Analysis):
        self.a = a
        self.num = 1
        self.den = 1
        self.sign = 1
        self.pi = 0
        self.roots: list[int] = []
        self.numeric_factors = 0

    def factor(self, n, in_den: bool, leftmost: bool, parent: str):
        a = self.a
        if isinstance(n, Mul):
            self.factor(n.left, in_den, leftmost, "mul")
            if isinstance(n.right, (Neg, Pos)) or (isinstance(n.right, Group) and isinstance(n.right.inner, (Neg, Pos))):
                a.not_simplified.append("sign inside a product")
            self.factor(n.right, in_den, False, "mul")
        elif isinstance(n, Div):
            self.factor(n.left, in_den, leftmost, "divleft")
            self._denominator(n.right, in_den)
        elif isinstance(n, Group):
            inner = n.inner
            if _is_sum(inner):
                a.abstain.append("bracketed sum inside a term")
            elif isinstance(inner, (Neg, Pos)):
                if leftmost and not in_den:
                    if isinstance(inner, Neg):
                        self.sign = -self.sign
                    if isinstance(inner.operand, (Neg, Pos)):
                        a.not_simplified.append("repeated signs")
                    self.factor(inner.operand, in_den, True, parent)
                else:
                    a.not_simplified.append("sign inside a term")
                    if isinstance(inner, Neg):
                        self.sign = -self.sign
                    self.factor(inner.operand, in_den, False, parent)
            elif isinstance(inner, Div) and isinstance(inner.left, Number) and isinstance(inner.right, Number):
                self.factor(inner, in_den, leftmost, "group")
            else:
                self.factor(inner, in_den, leftmost, parent)
        elif isinstance(n, (Neg, Pos)):
            if leftmost and not in_den:
                if isinstance(n, Neg):
                    self.sign = -self.sign
                if isinstance(n.operand, (Neg, Pos)) or (isinstance(n.operand, Group) and isinstance(n.operand.inner, (Neg, Pos))):
                    a.not_simplified.append("repeated signs")
                self.factor(n.operand, in_den, True, parent)
            else:
                a.not_simplified.append("sign inside a term")
                if isinstance(n, Neg):
                    self.sign = -self.sign
                self.factor(n.operand, in_den, False, parent)
        elif isinstance(n, Number):
            if "." in n.text:
                a.has_decimal = True
                a.wrong_form.append("decimal in an exact answer")
            v = n.value
            if in_den:
                self.den *= v
            else:
                self.num *= v
                if parent == "mul":
                    self.numeric_factors += 1
                    if v == 1:
                        a.not_simplified.append("redundant factor 1")
                    if v == 0:
                        a.not_simplified.append("zero coefficient")
        elif isinstance(n, Pi):
            if in_den:
                a.abstain.append("π in a denominator")
            self.pi += 1
            a.has_pi = True
            if self.pi > 1:
                a.abstain.append("π written twice in one term")
        elif isinstance(n, Sqrt):
            a.has_root = True
            if in_den:
                a.not_simplified.append("root in a denominator")
            arg = n.arg
            while isinstance(arg, Group):
                arg = arg.inner
            if _is_sum(arg):
                a.abstain.append("bracketed sum inside a root")
                return
            if any(isinstance(x, Pi) for x in _iter(arg)):
                a.abstain.append("π inside a root")
                return
            if not (isinstance(arg, Number) and "." not in arg.text):
                a.wrong_form.append("radicand must be an integer literal")
                self.roots.append(1)
                return
            r = int(arg.value)
            if r in (0, 1):
                a.not_simplified.append("√0 or √1")
                if r == 0:
                    self.num = 0
                self.roots.append(1)
                return
            k, m = square_free_part(r)
            if k != 1:
                a.not_simplified.append("perfect-square radicand" if m == 1 else "radicand not square-free")
            self.num *= k
            self.roots.append(m)
            if len(self.roots) > 1:
                a.not_simplified.append("two roots in one term")
        elif isinstance(n, Pow):
            a.abstain.append("powers are not supported in exact answers")
        elif isinstance(n, Variable):
            a.abstain.append("variable in an exact answer")
        else:
            a.abstain.append(f"unsupported {type(n).__name__}")

    def _denominator(self, d, in_den: bool):
        a = self.a
        dd = d
        while isinstance(dd, Group):
            dd = dd.inner
        if isinstance(dd, (Neg, Pos)):
            a.not_simplified.append("sign in a denominator")
            if isinstance(dd, Neg):
                self.sign = -self.sign
            dd = dd.operand
        if _is_sum(dd):
            a.abstain.append("bracketed sum in a denominator")
            return
        if isinstance(dd, Number) and dd.value == 0:
            a.abstain.append("zero denominator")
            return
        if isinstance(dd, Number) and dd.value == 1:
            a.not_simplified.append("division by 1")
        self.factor(dd, not in_den, False, "den")


def _iter(n):
    from .parser import walk
    return walk(n)


def analyse(node) -> Analysis:
    a = Analysis()
    while isinstance(node, Group) and not isinstance(node.inner, (Neg, Pos)):
        node = node.inner  # redundant brackets around the whole answer
    terms = _terms(node, a)
    lone_zero = len(terms) == 1 and isinstance(terms[0][1], Number) and terms[0][1].value == 0
    bases = []
    for sign, t in terms:
        if isinstance(t, Group) and _is_sum(t.inner):
            a.abstain.append("bracketed sum inside a term")
            continue
        term = _Term(a)
        term.factor(t, False, True, "top")
        if term.numeric_factors > 1:
            a.not_simplified.append("coefficient written as a product")
        if term.den == 0:
            a.abstain.append("zero denominator")
            continue
        n_int, d_int = term.num, term.den
        if (isinstance(n_int, Fraction) and n_int.denominator != 1) or (isinstance(d_int, Fraction) and d_int.denominator != 1):
            pass  # decimals already flagged
        else:
            ni, di = int(n_int), int(d_int)
            if di != 1 and ni != 0 and gcd(abs(ni), di) != 1:
                a.not_simplified.append("fraction not in lowest terms")
        coef = Fraction(n_int) / Fraction(d_int) * sign * term.sign
        if term.roots and len(term.roots) == 1:
            basis = ("sqrt", term.roots[0]) if term.roots[0] != 1 else ("const",)
        elif term.roots:
            m = 1
            for r in term.roots:
                m *= r
            k2, m2 = square_free_part(m) if m > 0 else (1, 1)
            coef *= k2
            basis = ("sqrt", m2) if m2 != 1 else ("const",)
        elif term.pi:
            basis = ("pi",)
        else:
            basis = ("const",)
        if coef == 0 and not lone_zero:
            a.not_simplified.append("zero term")
        bases.append(basis)
        a.canon[basis] = a.canon.get(basis, Fraction(0)) + coef
    if len(bases) != len(set(bases)):
        a.not_simplified.append("like terms not collected")
    if a.has_pi and a.has_root:
        a.abstain.append("π together with a root")
    a.canon = {k: v for k, v in a.canon.items() if v != 0}
    a.term_bases = bases
    return a


def _form_issue(a: Analysis, form: str) -> Optional[str]:
    if form == "in_terms_of_pi" and not a.has_pi:
        return "answer must be in terms of π"
    if form == "surd" and not a.has_root:
        return "answer must be in surd form"
    if form == "surd" and a.has_pi:
        return "π in a surd answer"
    if form == "fraction_simplest" and (a.has_pi or a.has_root):
        return "a fraction with no π or roots is required"
    return None


def _check(a: Analysis, form: str) -> Optional[Verdict]:
    if a.abstain:
        return abstain(Reason.NOT_AUTO_GRADABLE, a.abstain[0])
    if a.wrong_form:
        return incorrect(Reason.WRONG_FORM, a.wrong_form[0])
    f = _form_issue(a, form)
    if f:
        return incorrect(Reason.WRONG_FORM, f)
    if a.not_simplified:
        return incorrect(Reason.NOT_SIMPLIFIED, a.not_simplified[0])
    return None


def check_exact_spec(spec) -> Optional[Verdict]:
    """Spec self-check: value and every alternative must be in the spec's own form."""
    for text in (spec.value, *getattr(spec, "alternatives", ())):
        res = parse(text)
        node = res.node.value if isinstance(res.node, Quantity) else res.node
        if not res.ok or isinstance(node, (Sequence, Tuple, Ratio, Relation)):
            return abstain(Reason.NOT_AUTO_GRADABLE, "invalid spec")
        if _check(analyse(node), spec.required_form) is not None:
            return abstain(Reason.NOT_AUTO_GRADABLE, f"spec value {text!r} is not in its own form")
    return None


def grade_exact_node(spec, node) -> Verdict:
    if isinstance(node, Sequence):
        return abstain(Reason.MULTIPLE_ANSWERS, "several answers in one field (policy rule 3)")
    if isinstance(node, (Tuple, Ratio, Relation)):
        return incorrect(Reason.WRONG_FORM, "tuple, ratio or relation")
    if isinstance(node, Quantity):
        unit = node.unit or node.currency
        spec_unit = getattr(spec, "unit", None)
        if spec_unit is None:
            return abstain(Reason.UNEXPECTED_UNIT, "unit given where none expected (policy rule 5)")
        if unit != spec_unit:
            return incorrect(Reason.WRONG_UNIT)
        node = node.value
    a = analyse(node)
    v = _check(a, spec.required_form)
    if v:
        return v
    spec_canon = analyse(_spec_node(spec.value)).canon
    if a.canon == spec_canon:
        return correct()
    for alt in getattr(spec, "alternatives", ()):
        if a.canon == analyse(_spec_node(alt)).canon:
            return correct("matches a listed alternative")
    return incorrect(Reason.WRONG_VALUE)


def _spec_node(text):
    n = parse(text).node
    return n.value if isinstance(n, Quantity) else n


def grade_exact(spec, text: str) -> Verdict:
    try:
        bad = check_exact_spec(spec)
        if bad:
            return bad
        res = parse(text)
        if not res.ok:
            return res.verdict
        return grade_exact_node(spec, res.node)
    except Exception as e:  # internal-error guard
        return abstain(Reason.WORKER_ERROR, f"internal error: {type(e).__name__}: {e}")
