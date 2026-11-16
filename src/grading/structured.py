"""Solutions, coordinates, vectors and ratios (pure Python, no SymPy).

Every component goes through the numeric or exact node grader, so accuracy,
units, simplest form and component alternatives behave as for single answers.
Solutions are matched with a maximum bipartite matching (not greedy).

Check order: 0 spec self-check → 1 parse → 2 structure → 3 unit →
4 components → 5 aggregation.
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Optional

from .exact import check_exact_spec, grade_exact_node
from .numeric import Literal, grade_numeric_node, student_literal
from .parser import (Group, Number, Quantity, Ratio, Relation, Sequence, Tuple, Variable, parse)
from .verdict import Reason, Status, Verdict, abstain, correct, incorrect


def _grade_component(spec, node) -> Verdict:
    if spec.type == "exact":
        return grade_exact_node(spec, node)
    return grade_numeric_node(spec, node, "numeric")


def _component_spec_check(components) -> Optional[Verdict]:
    for c in components:
        if c.type == "exact":
            bad = check_exact_spec(c)
            if bad:
                return bad
    return None


def _max_matching(ok: list[list[bool]]) -> list[int]:
    """Kuhn's algorithm. Returns match_of_spec[j] = item index or -1."""
    n_items = len(ok)
    n_spec = len(ok[0]) if ok else 0
    match = [-1] * n_spec

    def try_item(i, seen):
        for j in range(n_spec):
            if ok[i][j] and not seen[j]:
                seen[j] = True
                if match[j] == -1 or try_item(match[j], seen):
                    match[j] = i
                    return True
        return False
    for i in range(n_items):
        try_item(i, [False] * n_spec)
    return match


# ------------------------------------------------------------------ solutions
def _solution_item(node, var: str):
    """Return the value node of `x = v`, `v = x` or a bare `v`; or a Verdict."""
    if isinstance(node, Relation):
        if node.op != "=":
            return incorrect(Reason.WRONG_FORM, "inequality given for a solution")
        left, right = node.left, node.right
        if isinstance(left, Variable) and left.name == var:
            return right
        if isinstance(right, Variable) and right.name == var:
            return left
        return incorrect(Reason.WRONG_FORM, "equation not solved for the variable")
    if isinstance(node, (Tuple, Ratio)):
        return incorrect(Reason.WRONG_FORM, "pair or ratio given for a solution")
    if isinstance(node, Quantity):
        return abstain(Reason.UNEXPECTED_UNIT, "unit on a solution (policy rule 5)")
    return node


def grade_solutions_node(spec, node) -> Verdict:
    items = list(node.items) if isinstance(node, Sequence) else [node]
    values = []
    for it in items:
        v = _solution_item(it, spec.variable)
        if isinstance(v, Verdict):
            return v
        values.append(v)
    sols = spec.solutions
    grid = [[_grade_component(s, v) for s in sols] for v in values]
    ok = [[g.status is Status.CORRECT for g in row] for row in grid]
    # merge repeated items correct for exactly the same solutions (D3)
    keep, seen_sets = [], []
    for i, row in enumerate(ok):
        key = tuple(row)
        if any(row) and key in seen_sets:
            continue
        seen_sets.append(key)
        keep.append(i)
    grid = [grid[i] for i in keep]
    ok = [ok[i] for i in keep]
    match = _max_matching(ok)
    matched_items = {i for i in match if i != -1}
    unmatched_items = [i for i in range(len(grid)) if i not in matched_items]
    unmatched_sols = [j for j in range(len(sols)) if match[j] == -1]
    if not unmatched_items and not unmatched_sols:
        return correct()
    # abstentions among the unresolved parts win (conservative)
    for i in unmatched_items:
        for g in grid[i]:
            if g.status is Status.ABSTAIN:
                return g
    for j in unmatched_sols:
        for i in range(len(grid)):
            if grid[i][j].status is Status.ABSTAIN:
                return grid[i][j]
    if unmatched_sols and any(sum(r) > 1 for r in ok):
        return abstain(Reason.NOT_AUTO_GRADABLE, "one answer fits several solutions (overlapping spec)")
    if unmatched_sols and not unmatched_items:
        return incorrect(Reason.MISSING_SOLUTION)
    if unmatched_items and not unmatched_sols:
        for i in unmatched_items:
            reasons = {g.reason for g in grid[i] if g.status is Status.INCORRECT}
            if reasons and Reason.WRONG_VALUE not in reasons:
                return incorrect(sorted(reasons, key=lambda r: r.value)[0], "restated solution fails")
        return incorrect(Reason.EXTRA_SOLUTION)
    shared = None
    for i in unmatched_items:
        for j in unmatched_sols:
            r = grid[i][j].reason
            shared = r if shared in (None, r) else Reason.WRONG_VALUE
    return incorrect(shared or Reason.WRONG_VALUE)


# --------------------------------------------------------- coordinates/vector
def grade_pair_node(spec, node) -> Verdict:
    if isinstance(node, Sequence):
        return abstain(Reason.MULTIPLE_ANSWERS, "several answers or an unbracketed pair (policy rule 3)")
    if isinstance(node, Quantity):
        return abstain(Reason.UNEXPECTED_UNIT, "unit on a pair (policy rule 5)")
    if not isinstance(node, Tuple) or len(node.items) != 2:
        return incorrect(Reason.WRONG_FORM, "a bracketed pair (x, y) is required")
    comps = spec.components
    g = [_grade_component(c, n) for c, n in zip(comps, node.items)]
    if all(v.status is Status.CORRECT for v in g):
        return correct()
    swapped = [_grade_component(c, n) for c, n in zip(comps, reversed(node.items))]
    if all(v.status is Status.CORRECT for v in swapped):
        return incorrect(Reason.WRONG_VALUE, "components in the wrong order")
    for v in g:
        if v.status is Status.INCORRECT:
            return v
    for v in g:
        if v.status is Status.ABSTAIN:
            return v
    return incorrect(Reason.WRONG_VALUE)  # pragma: no cover


# ----------------------------------------------------------------------- ratio
def grade_ratio_node(spec, node) -> Verdict:
    if isinstance(node, Sequence):
        return abstain(Reason.MULTIPLE_ANSWERS, "several answers in one field (policy rule 3)")
    if isinstance(node, Quantity):
        return abstain(Reason.UNEXPECTED_UNIT, "unit on a ratio (policy rule 5)")
    if not isinstance(node, Ratio):
        return incorrect(Reason.WRONG_FORM, "a ratio a:b is required")
    if len(node.parts) != len(spec.parts):
        return incorrect(Reason.WRONG_FORM, "wrong number of parts")
    vals = []
    for p in node.parts:
        if isinstance(p, Quantity):
            return abstain(Reason.UNEXPECTED_UNIT, "unit on a ratio part (policy rule 5)")
        lit = student_literal(p)
        if not isinstance(lit, Literal):
            return abstain(Reason.NOT_AUTO_GRADABLE, "ratio part is not a plain number")
        vals.append(lit.value)
    if any(v <= 0 for v in vals):
        return incorrect(Reason.WRONG_VALUE, "ratio parts must be positive")
    target = [Fraction(p) for p in spec.parts]

    def proportional(a, b):
        return all(x * b[0] == y * a[0] for x, y in zip(a, b))
    if not proportional(vals, target):
        from itertools import permutations
        if any(proportional(list(p), target) for p in permutations(vals)):
            return incorrect(Reason.WRONG_VALUE, "parts in a different order")
        return incorrect(Reason.WRONG_VALUE)
    if any(v.denominator != 1 for v in vals):
        return incorrect(Reason.NOT_SIMPLIFIED, "non-integer parts")
    g = 0
    for v in vals:
        g = gcd(g, int(v))
    if g != 1:
        return incorrect(Reason.NOT_SIMPLIFIED, f"common factor {g}")
    return correct()


# ---------------------------------------------------------------- entry point
def grade_structured(spec, text: str) -> Verdict:
    try:
        if spec.type in ("solutions", "coordinates", "vector"):
            comps = spec.solutions if spec.type == "solutions" else spec.components
            bad = _component_spec_check(comps)
            if bad:
                return bad
        variables = (spec.variable,) if spec.type == "solutions" else ()
        res = parse(text, variables)
        if not res.ok:
            return res.verdict
        if spec.type == "solutions":
            return grade_solutions_node(spec, res.node)
        if spec.type in ("coordinates", "vector"):
            return grade_pair_node(spec, res.node)
        return grade_ratio_node(spec, res.node)
    except Exception as e:  # internal-error guard
        return abstain(Reason.WORKER_ERROR, f"internal error: {type(e).__name__}: {e}")
