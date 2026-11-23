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
    p = paired_test(ours, other, GOLD, np.ones(4, bool))
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


# ---------------------------------------------------------------- additions
from evaluation.data import gwet_ac1
from evaluation.grader_adapter import grader_fingerprint, spec_examples, stratum, validate_spec
from evaluation.metrics import clopper_pearson, error_counts, rule_of_three
from evaluation.metrics import (at_mask, intended_vs_gold, matched_threshold, operating_point,
                                reason_confusion)


def test_gwet_ac1_by_hand():
    a = ["c"] * 18 + ["i"] * 2
    b = ["c"] * 18 + ["c", "i"]
    # po = 0.95; pi_c = 37/40, pi_i = 3/40; pe = 2 * (37/40)(3/40) / 1 = 0.13875
    assert gwet_ac1(a, b) == pytest.approx((0.95 - 0.13875) / (1 - 0.13875))
    assert gwet_ac1(a, a) == 1.0


def test_exact_binomial_bounds():
    lo, hi = clopper_pearson(0, 100)
    assert lo == 0 and hi == pytest.approx(0.0362, abs=1e-4)          # 1 - 0.025 ** (1/100)
    lo, hi = clopper_pearson(5, 50)
    assert lo == pytest.approx(0.0333, abs=1e-3) and hi == pytest.approx(0.2181, abs=1e-3)
    assert rule_of_three(100) == 0.03


def test_error_counts_report_rule_of_three_only_when_zero():
    s = S("r", [1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0])
    k = error_counts(s, GOLD)
    assert (k["false_accepts"], k["false_rejects"]) == (0, 1)
    assert k["fa_rule_of_three"] == pytest.approx(1.5) and k["fr_rule_of_three"] is None


def test_strata():
    assert stratum({"type": "numeric", "value": "2/3", "accuracy": {"kind": "exact"}}) == "value_only"
    assert stratum({"type": "numeric", "value": "12.3"}) == "form_sensitive"            # 3 s.f. default
    assert stratum({"type": "numeric", "value": "4", "accuracy": {"kind": "exact"}, "unit": "cm"}) == "form_sensitive"
    assert stratum({"type": "expression", "value": "x+1", "variables": ["x"]}) == "form_sensitive"


def test_spec_examples_are_valid_for_the_grader_in_src():
    for _, s in spec_examples():
        validate_spec(__import__("json").loads(s))


def test_grader_fingerprint_is_stable():
    assert grader_fingerprint() == grader_fingerprint()


# ------------------------------------------------------- protocol v1.1
def test_matched_threshold_is_closest_without_falling_below():
    # confidences 1.0, 0.8, 0.8, 0.6: coverages by threshold are 25%, 75%, 100%
    j = S("j", [1, 1, 1, 1], [1, 1, 0, 1], [1.0, 0.8, 0.8, 0.6])
    assert matched_threshold(j, 0.5) == (0.8, True)        # 75% is the first level >= 50%
    assert matched_threshold(j, 0.25) == (1.0, True)
    op = operating_point(j, GOLD, 0.5)
    assert op["coverage"] == 0.75 and op["false_accept_rate"] == 0.0   # item 4 (wrong) not decided
    r = S("r", [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0])
    assert matched_threshold(r, 0.9)[1] is False           # a single-point system that cannot reach it


def test_abstention_on_correct_answers():
    s = S("r", [1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 1, 1])
    m = at_mask(s.decided, s, GOLD)
    assert m["abstain_on_correct"] == 0.5                  # item 2 is correct and not decided
    assert summarise(s, GOLD)["abstain_on_correct"] == pytest.approx(0.5)


def test_vote_tie_break_never_crosses_a_vote_level():
    import math
    lab, c = aggregate(["correct"] * 4 + ["incorrect"], [-0.1, -0.1, -0.1, -0.1, -5.0])
    assert lab == "correct" and c == pytest.approx(0.8 + 0.01 * math.exp(-0.1))
    _, c3 = aggregate(["correct"] * 3 + ["incorrect"] * 2, [0.0] * 5)    # best possible tie-break at 3/5
    assert c3 < 0.8                                                  # still below any 4/5 answer
    assert aggregate(["correct"] * 3 + ["incorrect"] * 2) == ("correct", 0.6)  # no log-probs: plain share


def test_reason_confusion_matrix_and_kappa():
    ours = S("o", [1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 1, 1])
    ours.reason[:] = ["wrong_value", "rounding_error", "not_simplified", "correct"]
    rc = reason_confusion(ours, ["incorrect", "incorrect", "incorrect", "correct"], ["value", "form", "form", ""])
    assert rc["n"] == 3 and rc["matrix"]["value"]["value"] == 1
    assert rc["matrix"]["form"]["accuracy"] == 1 and rc["matrix"]["form"]["form"] == 1


def test_intended_vs_gold():
    from evaluation.data import Response
    rs = [Response("a", "i", "1", "ms_form", intended_label="correct"),
          Response("b", "i", "2", "examiner_report", "x", intended_label="incorrect", intended_category="value"),
          Response("c", "i", "3", "generic", "sign_error", intended_label="incorrect", intended_category="value")]
    gold = {"a": {"label": "correct"}, "b": {"label": "correct"}, "c": {"label": "incorrect"}}
    r = intended_vs_gold(rs, gold)
    assert (r["n"], r["differ"]) == (3, 1) and r["examples"][0]["response_id"] == "b"


def test_stratified_sample_by_paper():
    from evaluation.author import stratified_sample
    papers = {"p1": [f"a{i}" for i in range(60)], "p2": [f"b{i}" for i in range(30)], "p3": [f"c{i}" for i in range(30)]}
    s = stratified_sample(papers, 100, 2027)
    assert len(s) == 100 and sum(x.startswith("a") for x in s) == 50
    assert s == stratified_sample(papers, 100, 2027)                # fixed seed
    assert len(stratified_sample({"p": ["x", "y"]}, 100)) == 2      # under the cap: everything


def test_unsupported_part_always_abstains_and_validates():
    from evaluation.grader_adapter import GraderHandle
    spec = {"type": "unsupported", "value": "x > 3"}
    validate_spec(spec)
    with GraderHandle() as g:
        assert g.grade(spec, "x > 3") == ("abstain", "spec_unsupported")


def test_string_match_normalises_whitespace_only():
    from evaluation.systems import normalise_ws
    assert normalise_ws("  3 x  + 1 ") == normalise_ws("3 x + 1")
    assert normalise_ws("3x+1") != normalise_ws("3 x + 1")


def test_validation_of_answer_records():
    from evaluation.data import Dataset, Item, Response, validate
    it = Item("i", "test", "0580/22", "June 2025", "1", "Number", "numeric", "q", "7",
              {"type": "numeric", "value": "7", "accuracy": {"kind": "exact"}}, "", "value_only")
    good = Response("r1", "i", "7", "ms_form", "", "seven", "author", "correct")
    bad = [Response("r2", "i", "8", "generic", "made_up", "eight", "author", "incorrect", "value"),
           Response("r3", "i", "7", "alt_form", "", "seven again", "author", "incorrect", "value"),
           Response("r4", "i", "", "examiner_report", "ER p.3", "desc", "classmate", "incorrect", "value")]
    errs = validate(Dataset({"i": it}, [good] + bad))
    assert not [e for e in errs if e.startswith("r1")]
    assert any("generic" in e for e in errs if e.startswith("r2"))
    assert any("implies" in e for e in errs if e.startswith("r3"))
    assert any("classmate typing" in e for e in errs if e.startswith("r4"))


def test_leakage_manual_sample_is_per_condition():
    from evaluation.leakage import _mark_manual_sample
    rows = [{"auto_leak": 0, "condition": c, "message": 0} for c in ("controller", "instructed") for _ in range(30)]
    rows += [{"auto_leak": 1, "condition": "instructed", "message": 1}]
    _mark_manual_sample(rows)
    for c in ("controller", "instructed"):
        assert sum(r["check_by_hand"] for r in rows if r["condition"] == c and r["message"] == 0) == 20
    assert rows[-1]["check_by_hand"] == 1
