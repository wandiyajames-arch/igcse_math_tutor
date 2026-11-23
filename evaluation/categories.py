"""Reason codes → statuses and the 6 rater categories (RQ2). Evaluation-side only.

Both grader versions use these 27 reason codes; keeping the mapping here means
``src/`` (frozen) is never edited.
"""
CORRECT = {"correct"}
INCORRECT = {"wrong_value", "insufficient_accuracy", "rounding_error", "rounded_exact_value", "not_simplified",
             "not_expanded", "not_fully_factorised", "wrong_form", "wrong_unit", "missing_solution",
             "extra_solution"}

REASON_CATEGORY = {
    "wrong_value": "value",
    "insufficient_accuracy": "accuracy",
    "rounding_error": "accuracy",
    "rounded_exact_value": "accuracy",
    "not_simplified": "form",
    "not_expanded": "form",
    "not_fully_factorised": "form",
    "wrong_form": "form",
    "wrong_unit": "units",
    "missing_solution": "solution_set",
    "extra_solution": "solution_set",
}


def status_of(reason: str) -> str:
    return "correct" if reason in CORRECT else "incorrect" if reason in INCORRECT else "abstain"
