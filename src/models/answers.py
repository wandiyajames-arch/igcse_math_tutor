"""Typed answer specifications (discriminated union on ``type``).

Pure Pydantic + the pure-Python parser: importing this module loads no SymPy.

Policy rules (grading policy, not Cambridge syllabus rules):
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from src.grading.parser import UNITS, Number, Neg, Quantity, parse

POLICY_RULES: tuple[str, ...] = (
    "1. cao accepts reordered terms, while preserving required-form constraints — policy, confirmed by user",
    "2. Half-up rounding — policy, confirmed by user",
    "3. Abstain on two answers in one field — policy, confirmed by user",
    "4. Integer trailing zeros where significant figures cannot be determined: abstain — policy, confirmed by user",
    "5. Abstain on unexpected units (a unit given where the spec has none) — policy, confirmed by user",
    "6. oe covers value equivalence only; simplest form is still required unless the spec says otherwise — policy, confirmed by user",
    "7. Bearing '25' for '025': abstain with an explanatory reason — policy, confirmed by user",
)

ALLOWED_UNITS = frozenset(UNITS) | {u + p for u in ("mm", "cm", "m", "km") for p in ("²", "³")} | {"%", "°", "$", "£", "€"}

_FROZEN = ConfigDict(frozen=True, extra="forbid")


def _check_value_text(text: str, variables=()) -> None:
    res = parse(text, variables)
    if not res.ok:
        raise ValueError(f"value {text!r} does not parse: {res.verdict.reason.value} ({res.verdict.detail})")


class Accuracy(BaseModel):
    """Required accuracy: significant figures (1–10), decimal places (0–10; 0 d.p. =
    nearest whole number) or exact (no digits)."""
    model_config = _FROZEN
    kind: Literal["sf", "dp", "exact"] = "sf"
    digits: Optional[int] = 3

    @field_validator("digits", mode="before")
    @classmethod
    def _no_bool(cls, v):
        if isinstance(v, bool):
            raise ValueError("digits must be an integer, not a boolean")
        return v

    @model_validator(mode="after")
    def _range(self):
        if self.kind == "exact":
            if self.digits is not None:
                raise ValueError("exact accuracy takes no digits")
        elif self.kind == "sf":
            if self.digits is None or not 1 <= self.digits <= 10:
                raise ValueError("s.f. digits must be 1–10")
        elif self.digits is None or not 0 <= self.digits <= 10:
            raise ValueError("d.p. digits must be 0–10")
        return self


class _Base(BaseModel):
    model_config = _FROZEN
    notes: str = ""


class NumericSpec(_Base):
    """Single numeric answer. Defaults: 3 s.f. (R1), oe, simplest form (R3), half-up
    rounding (policy 2), more precise answers accepted (P1-D2)."""
    type: Literal["numeric"] = "numeric"
    value: str
    accuracy: Accuracy = Accuracy()
    mark: Literal["oe", "cao", "awrt"] = "oe"
    required_form: Literal["any", "integer", "decimal", "fraction", "standard_form"] = "any"
    simplest_form: bool = True
    rounding: Literal["half_up"] = "half_up"
    allow_more_precise: bool = True
    unit: Optional[str] = None
    accept_range: Optional[tuple[str, str]] = None
    alternatives: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate(self):
        _check_value_text(self.value)
        for a in self.alternatives:
            _check_value_text(a)
        if self.unit is not None and self.unit not in ALLOWED_UNITS:
            raise ValueError(f"unknown unit {self.unit!r}")
        if self.accept_range is not None:
            if self.accuracy.kind == "exact":
                raise ValueError("a range cannot be used with exact accuracy")
            lo, hi = (_range_endpoint(x) for x in self.accept_range)
            if lo > hi:
                raise ValueError("range low > high")
            v = _literal_value(self.value)
            if v is not None and not lo <= v <= hi:
                raise ValueError("literal value outside its range")
        return self


class AngleSpec(_Base):
    """Angle in degrees, fixed 1 d.p. (R6), half-up rounding."""
    type: Literal["angle"] = "angle"
    value: str
    accept_range: Optional[tuple[str, str]] = None
    alternatives: tuple[str, ...] = ()

    @property
    def accuracy(self) -> Accuracy:
        return Accuracy(kind="dp", digits=1)

    @model_validator(mode="after")
    def _validate(self):
        _check_value_text(self.value)
        return self


class BearingSpec(_Base):
    """Three-figure bearing 000–360 (R7), fixed 1 d.p.; '25' for '025' abstains (policy 7)."""
    type: Literal["bearing"] = "bearing"
    value: str
    accept_range: Optional[tuple[str, str]] = None
    alternatives: tuple[str, ...] = ()

    @property
    def accuracy(self) -> Accuracy:
        return Accuracy(kind="dp", digits=1)

    @model_validator(mode="after")
    def _validate(self):
        _check_value_text(self.value)
        v = _literal_value(self.value)
        if v is not None and not 0 <= v <= 360:
            raise ValueError("bearing must be 000–360")
        return self


class ExactSpec(_Base):
    """Exact value in terms of π, in surd form, or a fraction in simplest form (R4, R10, R11)."""
    type: Literal["exact"] = "exact"
    value: str
    required_form: Literal["in_terms_of_pi", "surd", "fraction_simplest"]
    mark: Literal["oe", "cao"] = "oe"
    unit: Optional[str] = None
    alternatives: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate(self):
        _check_value_text(self.value)
        if self.unit is not None and self.unit not in ALLOWED_UNITS:
            raise ValueError(f"unknown unit {self.unit!r}")
        return self


class ExpressionSpec(_Base):
    """Algebraic expression; required form simplified (default), expanded,
    factorised_fully (over ℚ, P1-D7) or any."""
    type: Literal["expression"] = "expression"
    value: str
    variables: tuple[str, ...]
    required_form: Literal["simplified", "expanded", "factorised_fully", "any"] = "simplified"
    mark: Literal["oe", "cao"] = "oe"
    alternatives: tuple[str, ...] = ()
    allow_unit_variable_clash: bool = False

    @model_validator(mode="after")
    def _validate(self):
        _check_variables(self.variables, self.allow_unit_variable_clash)
        _check_value_text(self.value, self.variables)
        for a in self.alternatives:
            _check_value_text(a, self.variables)
        return self


Component = Annotated[Union[NumericSpec, ExactSpec], Field(discriminator="type")]


class SolutionsSpec(_Base):
    """All solutions of an equation; order free, all required."""
    type: Literal["solutions"] = "solutions"
    variable: str
    solutions: tuple[Component, ...]
    allow_unit_variable_clash: bool = False

    @model_validator(mode="after")
    def _validate(self):
        _check_variables((self.variable,), self.allow_unit_variable_clash)
        if not self.solutions:
            raise ValueError("at least one solution")
        if any(c.unit for c in self.solutions):
            raise ValueError("components carry no unit")
        if len({c.value for c in self.solutions}) != len(self.solutions):
            raise ValueError("solutions must be distinct")
        return self


class CoordinatesSpec(_Base):
    """Ordered pair (x, y)."""
    type: Literal["coordinates"] = "coordinates"
    components: tuple[Component, Component]

    @model_validator(mode="after")
    def _validate(self):
        if any(c.unit for c in self.components):
            raise ValueError("components carry no unit")
        return self


class VectorSpec(CoordinatesSpec):
    """2-D column vector, typed as (x, y)."""
    type: Literal["vector"] = "vector"  # type: ignore[assignment]


class RatioSpec(_Base):
    """Ratio of positive integers in simplest form (R10)."""
    type: Literal["ratio"] = "ratio"
    parts: tuple[int, ...]

    @model_validator(mode="after")
    def _validate(self):
        if len(self.parts) < 2:
            raise ValueError("a ratio needs at least two parts")
        if any(isinstance(p, bool) or p <= 0 for p in self.parts):
            raise ValueError("ratio parts must be positive integers")
        g = 0
        for p in self.parts:
            g = gcd(g, p)
        if g != 1:
            raise ValueError("ratio spec must be in simplest form")
        return self


AnswerSpec = Annotated[
    Union[NumericSpec, AngleSpec, BearingSpec, ExactSpec, ExpressionSpec,
          SolutionsSpec, CoordinatesSpec, VectorSpec, RatioSpec],
    Field(discriminator="type"),
]
ANSWER_SPEC_ADAPTER = TypeAdapter(AnswerSpec)


def parse_answer_spec(data) -> "AnswerSpec":
    if isinstance(data, (str, bytes)):
        return ANSWER_SPEC_ADAPTER.validate_json(data)
    return ANSWER_SPEC_ADAPTER.validate_python(data)


# ------------------------------------------------------------------ helpers
def _check_variables(variables, allow_clash: bool) -> None:
    if not variables:
        raise ValueError("declare at least one variable")
    for v in variables:
        if not (isinstance(v, str) and len(v) == 1 and v.isascii() and v.isalpha()):
            raise ValueError(f"variable {v!r} must be a single ASCII letter")
    if len(set(variables)) != len(variables):
        raise ValueError("variables must be distinct")
    if not allow_clash:
        clashes = [u for u in UNITS if all(c in variables for c in u)]
        if {"p", "i"} <= set(variables):
            clashes.append("pi")
        if clashes:
            raise ValueError(f"variables clash with unit words {clashes}; set allow_unit_variable_clash=True")


def _range_endpoint(text: str) -> Fraction:
    import re
    if not isinstance(text, str) or not re.fullmatch(r"-?\d+(\.\d+)?", text):
        raise ValueError(f"range endpoint {text!r} must be an integer or decimal literal")
    return Fraction(text)


def _literal_value(text: str) -> Optional[Fraction]:
    node = parse(text).node
    if isinstance(node, Quantity):
        node = node.value
    if isinstance(node, Number):
        return node.value
    if isinstance(node, Neg) and isinstance(node.operand, Number):
        return -node.operand.value
    return None
