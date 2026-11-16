"""Expression equivalence and AST form checks (worker side; SymPy via ``symbolic``).

Check order (first failure wins): 0 spec self-check → 1 parse → 2 structure →
3 unit → 4 supported structure → 5 form (NOT_EXPANDED / NOT_FULLY_FACTORISED,
then NOT_SIMPLIFIED) → 6 cao → 7 value.

Value evidence (step 7): equality only when expand, cancel or simplify of the
difference gives exactly 0. WRONG_VALUE needs an exact rational counterexample
at a fixed probe point, or a numeric one together with expand and cancel not
giving 0 (two-witness rule). Fewer than 3 valid probe points abstain. Numeric
agreement without proof abstains EQUIVALENCE_UNCONFIRMED; simplify can only
confirm, never reject.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

import sympy as sp

from .parser import (Add, Div, Group, Mul, Neg, Number, Pi, Pos, Pow, Quantity, Ratio,
                     Relation, Sequence, Sqrt, Sub, Tuple, Variable, parse, walk)
from .symbolic import ConversionError, symbol, to_sympy
from .verdict import Reason, Verdict, abstain, correct, incorrect

PROBES = (Fraction(3, 7), Fraction(-5, 11), Fraction(13, 17), Fraction(-19, 23), Fraction(29, 31))
MAX_DEGREE = 60
MAX_TERMS = 1000


class _Stop(Exception):
    def __init__(self, verdict: Verdict):
        self.verdict = verdict


# ------------------------------------------------------------ AST helpers
def _strip(n):
    while isinstance(n, Group):
        n = n.inner
    return n


def _is_sum(n) -> bool:
    return isinstance(_strip(n), (Add, Sub))


def _has(node, cls) -> bool:
    return any(isinstance(x, cls) for x in walk(node))


def _is_numeric(n) -> bool:
    return not _has(n, (Variable,)) and not _has(n, Pi) and not _has(n, Sqrt)


def additive_terms(n):
    """Signed terms of a sum, reading through redundant brackets around sums."""
    n = _strip(n)
    if isinstance(n, Add):
        return additive_terms(n.left) + [(1, n.right)]
    if isinstance(n, Sub):
        return additive_terms(n.left) + [(-1, n.right)]
    return [(1, n)]


def _factors(n, den=False):
    """Flatten a product/quotient into (factor, in_denominator) pairs."""
    return [(f, d) for f, d, _ in _factors_roles(n, den)]


def _factors_roles(n, den=False, role="mul"):
    """Like ``_factors`` but also says whether a factor is a written numerator."""
    n2 = n
    if isinstance(n2, Group) and not _is_sum(n2.inner):
        n2 = n2.inner
    if isinstance(n2, (Neg, Pos)):
        return _factors_roles(n2.operand, den, role)
    if isinstance(n2, Mul):
        return _factors_roles(n2.left, den, "mul") + _factors_roles(n2.right, den, "mul")
    if isinstance(n2, Div):
        return _factors_roles(n2.left, den, "num") + _factors_roles(n2.right, not den, "den")
    return [(n2, den, role)]


def like_term_key(term):
    """Key of a monomial term (variables and integer powers); None if not a monomial."""
    powers: dict[str, Fraction] = {}
    for f, den in _factors(term):
        f = _strip(f)
        sgn = -1 if den else 1
        if isinstance(f, Number):
            continue
        if isinstance(f, Variable):
            powers[f.name] = powers.get(f.name, 0) + sgn
        elif isinstance(f, Pow) and isinstance(_strip(f.base), Variable) and isinstance(_strip(f.exp), Number):
            powers[_strip(f.base).name] = powers.get(_strip(f.base).name, 0) + sgn * _strip(f.exp).value
        elif isinstance(f, Pi):
            powers["π"] = powers.get("π", 0) + sgn
        else:
            return None
    return tuple(sorted((k, v) for k, v in powers.items() if v != 0))


def _sums(node):
    """Every maximal sum in the tree (including inside brackets)."""
    out = []

    def rec(n, inside_sum):
        if isinstance(n, (Add, Sub)):
            if not inside_sum:
                out.append(n)
            rec(n.left, True)
            rec(n.right, False)
            return
        if isinstance(n, Group):
            rec(n.inner, False)
            return
        for v in getattr(n, "__dict__", {}).values():
            if hasattr(v, "__dataclass_fields__"):
                rec(v, False)
    rec(node, False)
    return out


def _leftmost_signed(n) -> bool:
    while isinstance(n, (Mul, Div, Group)):
        n = n.inner if isinstance(n, Group) else n.left
    return isinstance(n, (Neg, Pos))


def products_of_sums(node) -> bool:
    for n in walk(node):
        if isinstance(n, Mul) and (_is_sum(n.left) or _is_sum(n.right)):
            return True
    return False


def powers_of_sums(node) -> bool:
    for n in walk(node):
        if isinstance(n, Pow) and _is_sum(n.base):
            e = _strip(n.exp)
            if isinstance(e, Number) and e.value.denominator == 1 and e.value >= 2:
                return True
    return False


def quotients_of_sums(node) -> bool:
    for n in walk(node):
        if isinstance(n, Div) and (_is_sum(n.left) or _is_sum(n.right)):
            return True
    return False


def _has_brackets_of_sum(node) -> bool:
    return products_of_sums(node) or powers_of_sums(node) or quotients_of_sums(node) or (
        isinstance(node, Neg) and _is_sum(node.operand))


# --------------------------------------------------------- supported checks
def _supported(node) -> Optional[Verdict]:
    for n in walk(node):
        if isinstance(n, Sqrt) and _has(n.arg, Variable):
            return abstain(Reason.NOT_AUTO_GRADABLE, "root of an expression with a variable")
        if isinstance(n, Pow):
            if _has(n.exp, Variable) or _has(n.exp, Pi):
                return abstain(Reason.NOT_AUTO_GRADABLE, "variable or π in an exponent")
            e = _strip(n.exp)
            ev = None
            if isinstance(e, Number):
                ev = e.value
            elif isinstance(e, Neg) and isinstance(_strip(e.operand), Number):
                ev = -_strip(e.operand).value
            if (ev is None or ev.denominator != 1) and _has(n.base, Variable):
                return abstain(Reason.NOT_AUTO_GRADABLE, "fractional power of a variable expression")
    deg, terms = _size(node)
    if deg > MAX_DEGREE or terms > MAX_TERMS:
        return abstain(Reason.TOO_COMPLEX, "expression too large to expand safely")
    return None


def _size(n) -> tuple[int, int]:
    """Upper bounds on (degree, number of terms after expansion)."""
    n = _strip(n)
    if isinstance(n, (Number, Pi, Sqrt)):
        return 0, 1
    if isinstance(n, Variable):
        return 1, 1
    if isinstance(n, (Neg, Pos)):
        return _size(n.operand)
    if isinstance(n, (Add, Sub)):
        a, b = _size(n.left), _size(n.right)
        return max(a[0], b[0]), a[1] + b[1]
    if isinstance(n, Mul):
        a, b = _size(n.left), _size(n.right)
        return a[0] + b[0], min(a[1] * b[1], 10 ** 9)
    if isinstance(n, Div):
        a, b = _size(n.left), _size(n.right)
        return a[0] + b[0], min(a[1] * b[1], 10 ** 9)
    if isinstance(n, Pow):
        a = _size(n.base)
        e = _strip(n.exp)
        k = int(abs(e.value)) if isinstance(e, Number) else 1
        from math import comb
        return a[0] * k, min(comb(a[1] + k - 1, k) if a[1] > 1 else 1, 10 ** 9)
    return 0, 1


# ------------------------------------------------------------------ forms
def _trivial(node) -> Optional[str]:
    for n in walk(node):
        if isinstance(n, (Add, Sub)) and _leftmost_signed(n.right):
            return "sign straight after an operator"
        if isinstance(n, (Neg, Pos)) and (isinstance(_strip(n.operand), (Neg, Pos)) or _leftmost_signed(n.operand)):
            return "repeated signs"
        if isinstance(n, Mul) and _leftmost_signed(n.right):
            return "sign inside a product"
        if isinstance(n, Pow):
            e = _strip(n.exp)
            if isinstance(e, Number) and e.value in (0, 1):
                return "exponent 0 or 1"
            if isinstance(_strip(n.base), Pow):
                return "power of a power"
            if _is_numeric(n.base) and _is_numeric(n.exp) and not _has(n.base, Pi):
                return "numeric power not evaluated"
            if isinstance(n.base, Group) and isinstance(n.base.inner, Neg):
                return "power of a negative term"
        if isinstance(n, Div):
            d = _strip(n.right)
            if isinstance(d, Number) and d.value == 1:
                return "denominator 1"
        if isinstance(n, Group) and _is_sum(n.inner) and _is_numeric(n.inner):
            return "numeric arithmetic not done"
    for s in _sums(node):
        keys = []
        for sign, t in additive_terms(s):
            tt = _strip(t)
            if isinstance(tt, Number) and tt.value == 0:
                return "zero term"
            k = like_term_key(t)
            if k is not None:
                keys.append(k)
        if len(keys) != len(set(keys)):
            return "like terms not collected"
    # per-term checks
    for s in _sums(node) + [node]:
        for sign, t in additive_terms(s):
            facs = _factors(t)
            nums = [f for f, den in facs if isinstance(_strip(f), Number) and not den]
            if len(nums) > 1:
                return "numeric arithmetic not done"
            for f, den, role in _factors_roles(t):
                f2 = _strip(f)
                if isinstance(f2, Number) and role == "mul" and f2.value in (0, 1) and len(facs) > 1:
                    return "factor 0 or 1"
            seen = set()
            for f, den in facs:
                f2 = _strip(f)
                v = None
                if isinstance(f2, Variable):
                    v = f2.name
                elif isinstance(f2, Pow) and isinstance(_strip(f2.base), Variable):
                    v = _strip(f2.base).name
                if v is not None:
                    if v in seen:
                        return "same variable in two factors"
                    seen.add(v)
    return None


def _fraction_common_factor(node) -> bool:
    for n in walk(node):
        if isinstance(n, Div):
            num = to_sympy(n.left).doit()
            den = to_sympy(n.right).doit()
            g = sp.gcd(sp.expand(num), sp.expand(den))
            if g not in (1, -1) and g != 0:
                return True
    return False


def _factorised_issue(node) -> Optional[Verdict]:
    top = _strip(node)
    if isinstance(top, (Add, Sub)):
        return incorrect(Reason.NOT_FULLY_FACTORISED, "not factorised")
    for f, den in _factors(node):
        f2 = _strip(f)
        base = f2
        if isinstance(f2, Pow):
            base = _strip(f2.base)
        if isinstance(base, (Add, Sub)):
            if _has(base, Pi) or _has(base, Sqrt):
                return abstain(Reason.NOT_AUTO_GRADABLE, "surd or π inside a factor")
            expr = sp.expand(to_sympy(base).doit())
            syms = sorted(expr.free_symbols, key=str)
            if not syms:
                continue
            poly = sp.Poly(expr, *syms)
            coeffs = poly.coeffs()
            if any(not c.is_Integer for c in coeffs):
                return abstain(Reason.NOT_AUTO_GRADABLE, "non-integer coefficients inside a factor")
            content = 0
            from math import gcd
            for c in coeffs:
                content = gcd(content, int(c))
            if content != 1:
                return incorrect(Reason.NOT_FULLY_FACTORISED, "common factor left inside a bracket")
            _, fl = sp.factor_list(expr)
            if len(fl) > 1 or (fl and fl[0][1] > 1):
                return incorrect(Reason.NOT_FULLY_FACTORISED, "a bracket can be factorised further")
    return None


def form_check(node, form: str, spec_node) -> Optional[Verdict]:
    if form == "any":
        return None
    if form == "expanded":
        if products_of_sums(node) or powers_of_sums(node) or (isinstance(_strip(node), Neg) and _is_sum(_strip(node).operand)):
            return incorrect(Reason.NOT_EXPANDED, "a product or power of a sum")
        if quotients_of_sums(node) and not quotients_of_sums(spec_node):
            return abstain(Reason.NOT_AUTO_GRADABLE, "quotient involving a sum")
    if form == "factorised_fully":
        v = _factorised_issue(node)
        if v:
            return v
    if form == "simplified" and _has_brackets_of_sum(node) and not _has_brackets_of_sum(spec_node):
        return abstain(Reason.NOT_AUTO_GRADABLE, "brackets where the expected answer has none")
    t = _trivial(node)
    if t:
        return incorrect(Reason.NOT_SIMPLIFIED, t)
    if _fraction_common_factor(node):
        return incorrect(Reason.NOT_SIMPLIFIED, "fraction with a common factor")
    return None


# -------------------------------------------------------------------- cao
def structure_key(n):
    n = _strip(n)
    if isinstance(n, Number):
        return ("n", n.text.rstrip("0").rstrip(".") if "." in n.text else n.text)
    if isinstance(n, Variable):
        return ("v", n.name)
    if isinstance(n, Pi):
        return ("pi",)
    if isinstance(n, Sqrt):
        return ("sqrt", structure_key(n.arg))
    if isinstance(n, (Add, Sub)):
        items = []
        for sign, t in additive_terms(n):
            items.append((sign, structure_key(t)))
        return ("+", tuple(sorted(items)))
    if isinstance(n, Neg):
        return ("neg", structure_key(n.operand))
    if isinstance(n, Pos):
        return structure_key(n.operand)
    if isinstance(n, (Mul, Div)):
        items = sorted((den, structure_key(f)) for f, den in _factors(n))
        return ("*", tuple(items))
    if isinstance(n, Pow):
        return ("^", structure_key(n.base), structure_key(n.exp))
    return ("?", repr(n))


# ------------------------------------------------------------------ value
def _point(variables, i):
    return {symbol(v): sp.Rational(PROBES[(i + j) % 5].numerator, PROBES[(i + j) % 5].denominator) + j
            for j, v in enumerate(variables)}


def _poles(node, subs) -> bool:
    for n in walk(node):
        if isinstance(n, Div):
            d = to_sympy(n.right).doit().subs(subs)
            if d == 0 or d.has(sp.zoo, sp.nan):
                return True
        if isinstance(n, Pow):
            e = to_sympy(n.exp).doit()
            if e.is_number and e.is_negative and to_sympy(n.base).doit().subs(subs) == 0:
                return True
    return False


def value_check(student_node, spec_node, variables) -> Verdict:
    s_expr = to_sympy(student_node).doit()
    e_expr = to_sympy(spec_node).doit()
    for x in (s_expr, e_expr):
        if x.has(sp.zoo, sp.nan, sp.oo):
            return abstain(Reason.NOT_AUTO_GRADABLE, "undefined value (zero denominator)")
    d = s_expr - e_expr
    allvars = sorted(set(variables) | {str(s) for s in d.free_symbols})
    valid = 0
    numeric_witness = False
    for i in range(5):
        subs = _point(allvars, i)
        if _poles(student_node, subs) or _poles(spec_node, subs):
            continue
        val = d.subs(subs)
        if val.has(sp.zoo, sp.nan, sp.oo):
            continue
        valid += 1
        if val.is_Rational:
            if val != 0:
                return incorrect(Reason.WRONG_VALUE, "exact counterexample at a probe point")
        else:
            approx = sp.N(val, 60)
            if abs(approx) > sp.Rational(1, 10 ** 40):
                numeric_witness = True
    if sp.expand(d) == 0:
        return correct()
    try:
        if sp.cancel(sp.together(d)) == 0:
            return correct()
    except Exception:
        pass
    if numeric_witness:
        return incorrect(Reason.WRONG_VALUE, "numeric counterexample and no symbolic proof (two witnesses)")
    if valid < 3:
        return abstain(Reason.NOT_AUTO_GRADABLE, "fewer than 3 valid probe points")
    if sp.simplify(d) == 0:
        return correct()
    return abstain(Reason.EQUIVALENCE_UNCONFIRMED, "numeric agreement without symbolic proof")


# ----------------------------------------------------------------- grader
def _spec_check(spec) -> Optional[Verdict]:
    nodes = []
    for text in (spec.value, *spec.alternatives):
        res = parse(text, spec.variables)
        if not res.ok or isinstance(res.node, (Sequence, Tuple, Ratio, Relation, Quantity)):
            return abstain(Reason.NOT_AUTO_GRADABLE, "invalid spec")
        n = res.node
        if _supported(n) is not None:
            return abstain(Reason.NOT_AUTO_GRADABLE, "spec uses unsupported structure")
        if form_check(n, spec.required_form, n) is not None:
            return abstain(Reason.NOT_AUTO_GRADABLE, f"spec value {text!r} is not in its own form")
        nodes.append(n)
    for alt in nodes[1:]:
        if value_check(alt, nodes[0], spec.variables).reason is not Reason.CORRECT:
            return abstain(Reason.NOT_AUTO_GRADABLE, "alternative not equivalent to the value")
    return None


def grade_expression_node(spec, node) -> Verdict:
    if isinstance(node, Sequence):
        return abstain(Reason.MULTIPLE_ANSWERS, "several answers in one field (policy rule 3)")
    if isinstance(node, (Tuple, Ratio, Relation)):
        return incorrect(Reason.WRONG_FORM, "tuple, ratio or relation")
    if isinstance(node, Quantity):
        return abstain(Reason.UNEXPECTED_UNIT, "unit on an algebraic answer (policy rule 5)")
    v = _supported(node)
    if v:
        return v
    spec_node = parse(spec.value, spec.variables).node
    try:
        to_sympy(node)
    except ConversionError as e:
        return abstain(Reason.NOT_AUTO_GRADABLE, e.detail)
    v = form_check(node, spec.required_form, spec_node)
    if v:
        return v
    if spec.mark == "cao":
        key = structure_key(node)
        keys = [structure_key(spec_node)] + [structure_key(parse(a, spec.variables).node) for a in spec.alternatives]
        if key in keys:
            return correct("structural match")
        vv = value_check(node, spec_node, spec.variables)
        return incorrect(Reason.WRONG_FORM, "cao: equal value in another form") if vv.reason is Reason.CORRECT else vv
    verdict = value_check(node, spec_node, spec.variables)
    if verdict.reason is not Reason.CORRECT:
        for alt in spec.alternatives:
            if value_check(node, parse(alt, spec.variables).node, spec.variables).reason is Reason.CORRECT:
                return correct("matches a listed alternative")
    return verdict


def grade_expression(spec, text: str) -> Verdict:
    try:
        bad = _spec_check(spec)
        if bad:
            return bad
        res = parse(text, spec.variables)
        if not res.ok:
            return res.verdict
        return grade_expression_node(spec, res.node)
    except ConversionError as e:
        return abstain(Reason.NOT_AUTO_GRADABLE, e.detail)
    except RecursionError:
        return abstain(Reason.TOO_COMPLEX, "too deep")
    except MemoryError:
        return abstain(Reason.RESOURCE_LIMIT, "memory limit")
