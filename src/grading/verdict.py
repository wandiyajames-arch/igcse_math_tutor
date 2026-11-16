"""Verdict vocabulary for the deterministic grader.

Every grader returns a ``Verdict``: one of three statuses and exactly one
reason. Abstain reasons mean "no decision"; the grader never marks an answer
incorrect unless it is certain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    ABSTAIN = "abstain"


class Reason(str, Enum):
    CORRECT = "correct"
    # incorrect (11)
    WRONG_VALUE = "wrong_value"
    INSUFFICIENT_ACCURACY = "insufficient_accuracy"
    ROUNDING_ERROR = "rounding_error"
    ROUNDED_EXACT_VALUE = "rounded_exact_value"
    NOT_SIMPLIFIED = "not_simplified"
    NOT_EXPANDED = "not_expanded"
    NOT_FULLY_FACTORISED = "not_fully_factorised"
    WRONG_FORM = "wrong_form"
    WRONG_UNIT = "wrong_unit"
    MISSING_SOLUTION = "missing_solution"
    EXTRA_SOLUTION = "extra_solution"
    # abstain (15)
    EMPTY_INPUT = "empty_input"
    UNPARSEABLE = "unparseable"
    AMBIGUOUS_INPUT = "ambiguous_input"
    MULTIPLE_ANSWERS = "multiple_answers"
    AMBIGUOUS_SIGNIFICANT_FIGURES = "ambiguous_significant_figures"
    UNEXPECTED_UNIT = "unexpected_unit"
    BEARING_NOT_THREE_FIGURE = "bearing_not_three_figure"
    TOO_LONG = "too_long"
    TOO_COMPLEX = "too_complex"
    RESOURCE_LIMIT = "resource_limit"
    TIMEOUT = "timeout"
    WORKER_ERROR = "worker_error"
    BUSY = "busy"
    EQUIVALENCE_UNCONFIRMED = "equivalence_unconfirmed"
    NOT_AUTO_GRADABLE = "not_auto_gradable"


INCORRECT_REASONS = frozenset({
    Reason.WRONG_VALUE, Reason.INSUFFICIENT_ACCURACY, Reason.ROUNDING_ERROR,
    Reason.ROUNDED_EXACT_VALUE, Reason.NOT_SIMPLIFIED, Reason.NOT_EXPANDED,
    Reason.NOT_FULLY_FACTORISED, Reason.WRONG_FORM, Reason.WRONG_UNIT,
    Reason.MISSING_SOLUTION, Reason.EXTRA_SOLUTION,
})

REASON_STATUS: dict[Reason, Status] = {
    r: (Status.CORRECT if r is Reason.CORRECT
        else Status.INCORRECT if r in INCORRECT_REASONS
        else Status.ABSTAIN)
    for r in Reason
}

# Six feedback categories used for human-label agreement (RQ2).
REASON_CATEGORY: dict[Reason, str] = {
    Reason.WRONG_VALUE: "wrong_value",
    Reason.INSUFFICIENT_ACCURACY: "accuracy",
    Reason.ROUNDING_ERROR: "accuracy",
    Reason.ROUNDED_EXACT_VALUE: "accuracy",
    Reason.NOT_SIMPLIFIED: "form",
    Reason.NOT_EXPANDED: "form",
    Reason.NOT_FULLY_FACTORISED: "form",
    Reason.WRONG_FORM: "form",
    Reason.WRONG_UNIT: "units_notation",
    Reason.MISSING_SOLUTION: "incomplete_extra",
    Reason.EXTRA_SOLUTION: "incomplete_extra",
}


@dataclass(frozen=True)
class Verdict:
    status: Status
    reason: Reason
    detail: str = field(default="", compare=False)  # for logs only

    def __post_init__(self) -> None:
        if not isinstance(self.reason, Reason) or not isinstance(self.status, Status):
            raise TypeError("status and reason must be enum members")
        if REASON_STATUS[self.reason] is not self.status:
            raise ValueError(f"reason {self.reason.value} does not belong to status {self.status.value}")

    def to_dict(self) -> dict:
        return {"status": self.status.value, "reason": self.reason.value, "detail": self.detail}

    @classmethod
    def from_dict(cls, d: dict) -> "Verdict":
        return cls(Status(d["status"]), Reason(d["reason"]), d.get("detail", ""))


def correct(detail: str = "") -> Verdict:
    return Verdict(Status.CORRECT, Reason.CORRECT, detail)


def incorrect(reason: Reason, detail: str = "") -> Verdict:
    return Verdict(Status.INCORRECT, reason, detail)


def abstain(reason: Reason, detail: str = "") -> Verdict:
    return Verdict(Status.ABSTAIN, reason, detail)
