"""Checks of the evaluation maths against hand-computed values."""
import numpy as np
import pytest

from evaluation.data import cohen_kappa
from evaluation.judges import aggregate, decide, parse_vote
from evaluation.leakage import candidates, wilson
from evaluation.metrics import (SystemArrays, aurc, cascade, mcnemar_exact, paired_test, summarise)
from evaluation.systems import to_latex


def S(name, decided, says, conf):
    n = len(decided)
    return SystemArrays(name, np.array(decided, bool), np.array(says, bool),
                        np.where(np.array(decided, bool), np.array(conf, float), -np.inf), [""] * n,
                        np.full(n, 1.0))


GOLD = np.array([True, True, False, False])  # 2 correct, 2 incorrect answers


def test_rule_system_metrics_by_hand():
    s = S("r", [1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0])
    m = summarise(s, GOLD)
    assert m["coverage"] == 0.75
    assert m["selective_risk"] == pytest.approx(2 / 3)       # items 2 and 3 wrong
    assert m["false_accept_rate"] == pytest.approx(1 / 2)    # item 3 accepted, of 2 incorrect
    assert m["false_reject_rate"] == pytest.approx(1 / 2)    # item 2 rejected, of 2 correct


def test_matched_coverage_uses_expected_ties():
    j = S("j", [1, 1, 1, 1], [1, 1, 1, 1], [1.0, 0.6, 0.6, 0.6])
    m = summarise(j, GOLD, m_target=2)
    # first: item 1 (right); then 1 of 3 tied items, 2 of which are wrong -> expected errors 2/3
    assert m["coverage"] == 0.5
    assert m["selective_risk"] == pytest.approx((2 / 3) / 2)


def test_bootstrap_weights_act_like_duplicates():
    s = S("r", [1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0])
    w = np.array([2, 1, 0, 1], float)
    m = summarise(s, GOLD, w)
    assert m["coverage"] == pytest.approx(3 / 4) and m["selective_risk"] == pytest.approx(1 / 3)


def test_cascade_prefers_ours_and_fills_abstentions():
    ours = S("o", [1, 0, 1, 0], [1, 0, 0, 0], [1, 0, 1, 0])
    judge = S("j", [1, 1, 1, 1], [0, 1, 1, 1], [0.6, 1, 1, 0.8])
    c = cascade(ours, judge)
    assert c.decided.all() and list(c.says_correct) == [True, True, False, True]
    assert summarise(c, GOLD, m_target=2)["selective_risk"] == 0  # ours' two decisions rank first


def test_aurc_of_flat_system_equals_its_risk():
    s = S("r", [1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0])
    assert aurc(s, GOLD) == pytest.approx(2 / 3)


def test_mcnemar_exact_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert mcnemar_exact(3, 3) == 1.0


def test_paired_test_counts():
    ours = S("o", [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1])
    other = S("x", [1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 1, 1])
    p = paired_test(ours, other, GOLD, 4)
    assert (p["ours_right_other_wrong"], p["ours_wrong_other_right"]) == (2, 0)


def test_kappa_by_hand():
    a = ["c", "c", "i", "i"]
    assert cohen_kappa(a, a) == 1.0
    assert cohen_kappa(a, ["c", "i", "c", "i"]) == pytest.approx(0.0)


def test_votes_and_aggregation():
    assert parse_vote("examiner", "INCORRECT.") == "incorrect"
    assert parse_vote("examiner", "<think>x</think> Correct") == "correct"
    assert parse_vote("compass", "A") == "correct" and parse_vote("compass", "C") == "invalid"
    assert aggregate(["correct"] * 3 + ["incorrect"] * 2) == ("correct", 0.6)
    assert aggregate(["correct", "incorrect", "invalid", "invalid", "invalid"])[0] == "abstain"
    assert decide(["correct"] * 4 + ["incorrect"], 0.8) == "correct"
    assert decide(["correct"] * 3 + ["incorrect"] * 2, 0.8) == "abstain"


def test_math_verify_input_conversion():
    assert to_latex("3√2") == r"$3\sqrt{2}$"
    assert to_latex("100-4π") == r"$100-4\pi $"
    assert to_latex("√(x+1)") == r"$\sqrt{x+1}$"


def test_leak_candidates_and_wilson():
    c = candidates("Try collecting like terms: the answer is 7a - 6b.")
    assert any("7a - 6b" in x or "7a" in x for x in c)
    lo, hi = wilson(5, 50)
    assert 0.04 < lo < 0.05 and 0.21 < hi < 0.22
