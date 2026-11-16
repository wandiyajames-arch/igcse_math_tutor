"""Certified rational enclosures for spec values (integer arithmetic only).

π uses Machin's formula with explicit error bounds; square roots use
``math.isqrt``. No floats, no ``evalf``, no SymPy.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt

from .parser import (Add, Div, Group, Mul, Neg, Number, Pi, Pos, Pow, Sqrt, Sub)

PI_3142 = Fraction(3142, 1000)


class Uncertain(Exception):
    """A decision could not be certified at this precision."""


class NotNumeric(Exception):
    """The expression is not a closed numeric expression."""


@dataclass(frozen=True)
class Interval:
    lo: Fraction
    hi: Fraction

    @staticmethod
    def point(x: Fraction) -> "Interval":
        return Interval(Fraction(x), Fraction(x))

    @property
    def is_point(self) -> bool:
        return self.lo == self.hi

    def contains_zero(self) -> bool:
        return self.lo <= 0 <= self.hi

    def __add__(self, o): return Interval(self.lo + o.lo, self.hi + o.hi)
    def __sub__(self, o): return Interval(self.lo - o.hi, self.hi - o.lo)
    def __neg__(self): return Interval(-self.hi, -self.lo)

    def __mul__(self, o):
        c = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return Interval(min(c), max(c))

    def __truediv__(self, o):
        if o.contains_zero():
            if o.is_point:
                raise NotNumeric("division by zero")
            raise Uncertain("divisor enclosure contains zero")
        return self * Interval(1 / o.hi, 1 / o.lo) if o.lo > 0 else self * Interval(1 / o.hi, 1 / o.lo)

    def pow(self, n: int):
        if n == 0:
            return Interval.point(Fraction(1))
        if n < 0:
            return Interval.point(Fraction(1)) / self.pow(-n)
        if n % 2 == 0 and self.contains_zero():
            m = max(abs(self.lo), abs(self.hi)) ** n
            return Interval(Fraction(0), m)
        a, b = self.lo ** n, self.hi ** n
        return Interval(min(a, b), max(a, b))


_PI_CACHE: dict[int, Interval] = {}


def _arctan_inv_scaled(x: int, scale: int) -> tuple[int, int]:
    """Return (A, terms) with A ≈ arctan(1/x)·scale; each term truncated (error ≤ 1 per term)."""
    total = 0
    power = scale // x
    k = 0
    x2 = x * x
    while power:
        term = power // (2 * k + 1)
        total += -term if k % 2 else term
        power //= x2
        k += 1
    return total, k


def pi_enclosure(precision: int) -> Interval:
    """Certified enclosure of π with width ≤ 10^-precision."""
    if precision in _PI_CACHE:
        return _PI_CACHE[precision]
    guard = 10
    scale = 10 ** (precision + guard)
    a, n1 = _arctan_inv_scaled(5, scale)
    b, n2 = _arctan_inv_scaled(239, scale)
    approx = 16 * a - 4 * b
    err = 16 * (2 * n1 + 2) + 4 * (2 * n2 + 2) + 4  # truncation errors (floor of powers and terms)
    enc = Interval(Fraction(approx - err, scale), Fraction(approx + err, scale))
    _PI_CACHE[precision] = enc
    return enc


def sqrt_enclosure(r: Fraction, precision: int) -> Interval:
    """Certified enclosure of √r (r ≥ 0) with width ≤ 10^-precision; exact for perfect squares."""
    if r < 0:
        raise NotNumeric("square root of a negative number")
    p, q = r.numerator, r.denominator
    s = 10 ** precision
    n = p * q * s * s
    k = isqrt(n)
    if k * k == n:
        return Interval.point(Fraction(k, q * s))
    return Interval(Fraction(k, q * s), Fraction(k + 1, q * s))


def evaluate(node, precision: int, pi_value: str = "calculator") -> Interval:
    """Evaluate a closed numeric AST to a certified enclosure."""
    if isinstance(node, Number):
        return Interval.point(node.value)
    if isinstance(node, Pi):
        return Interval.point(PI_3142) if pi_value == "3.142" else pi_enclosure(precision + 5)
    if isinstance(node, (Group, Pos)):
        return evaluate(node.inner if isinstance(node, Group) else node.operand, precision, pi_value)
    if isinstance(node, Neg):
        return -evaluate(node.operand, precision, pi_value)
    if isinstance(node, Add):
        return evaluate(node.left, precision, pi_value) + evaluate(node.right, precision, pi_value)
    if isinstance(node, Sub):
        return evaluate(node.left, precision, pi_value) - evaluate(node.right, precision, pi_value)
    if isinstance(node, Mul):
        return evaluate(node.left, precision, pi_value) * evaluate(node.right, precision, pi_value)
    if isinstance(node, Div):
        return evaluate(node.left, precision, pi_value) / evaluate(node.right, precision, pi_value)
    if isinstance(node, Sqrt):
        a = evaluate(node.arg, precision + 5, pi_value)
        if a.hi < 0:
            raise NotNumeric("square root of a negative number")
        if a.lo < 0:
            raise Uncertain("radicand enclosure crosses zero")
        lo = sqrt_enclosure(a.lo, precision + 5).lo
        hi = sqrt_enclosure(a.hi, precision + 5).hi
        return Interval(lo, hi)
    if isinstance(node, Pow):
        e = evaluate(node.exp, precision, pi_value)
        if not e.is_point or e.lo.denominator != 1:
            raise NotNumeric("only integer powers are supported")
        return evaluate(node.base, precision + 5, pi_value).pow(int(e.lo))
    raise NotNumeric(f"not numeric: {type(node).__name__}")


def contains_pi(node) -> bool:
    from .parser import walk
    return any(isinstance(n, Pi) for n in walk(node))
