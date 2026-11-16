"""AST → SymPy converter. The only module in ``src`` that imports SymPy.

Every constructor uses ``evaluate=False`` so SymPy does not tidy what the
student wrote; numbers are exact rationals from the literal text (never float).
Algebra in the expression grader starts from ``.doit()``.
"""
from __future__ import annotations

from functools import lru_cache

import sympy as sp

from .parser import (Add, Div, Group, Mul, Neg, Number, Pi, Pos, Pow, Sqrt, Sub, Variable)


class ConversionError(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@lru_cache(maxsize=None)
def symbol(name: str) -> sp.Symbol:
    return sp.Symbol(name)


def to_sympy(node):
    if isinstance(node, Number):
        f = node.value
        return sp.Rational(f.numerator, f.denominator)
    if isinstance(node, Variable):
        return symbol(node.name)
    if isinstance(node, Pi):
        return sp.pi
    if isinstance(node, (Group, Pos)):
        return to_sympy(node.inner if isinstance(node, Group) else node.operand)
    if isinstance(node, Neg):
        return sp.Mul(-1, to_sympy(node.operand), evaluate=False)
    if isinstance(node, Add):
        return sp.Add(to_sympy(node.left), to_sympy(node.right), evaluate=False)
    if isinstance(node, Sub):
        return sp.Add(to_sympy(node.left), sp.Mul(-1, to_sympy(node.right), evaluate=False), evaluate=False)
    if isinstance(node, Mul):
        return sp.Mul(to_sympy(node.left), to_sympy(node.right), evaluate=False)
    if isinstance(node, Div):
        den = node.right
        while isinstance(den, Group):
            den = den.inner
        if isinstance(den, Number) and den.value == 0:
            raise ConversionError("not_auto_gradable", "division by zero")
        return sp.Mul(to_sympy(node.left), sp.Pow(to_sympy(node.right), -1, evaluate=False), evaluate=False)
    if isinstance(node, Pow):
        base = node.base
        while isinstance(base, Group):
            base = base.inner
        exp = to_sympy(node.exp)
        if isinstance(base, Number) and base.value == 0 and exp.is_number and exp.is_negative:
            raise ConversionError("not_auto_gradable", "division by zero")
        return sp.Pow(to_sympy(node.base), exp, evaluate=False)
    if isinstance(node, Sqrt):
        arg = node.arg
        while isinstance(arg, Group):
            arg = arg.inner
        if isinstance(arg, Number) is False and isinstance(arg, Neg) and isinstance(arg.operand, Number):
            raise ConversionError("not_auto_gradable", "square root of a negative number")
        return sp.Pow(to_sympy(node.arg), sp.Rational(1, 2), evaluate=False)
    raise ConversionError("unparseable", "not an expression")
