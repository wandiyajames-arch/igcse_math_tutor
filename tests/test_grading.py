"""Grader tests: vocabulary, parser, labelled cases, properties, security, isolation."""
import json
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from src.grading.enclosure import pi_enclosure, sqrt_enclosure
from src.grading.exact import square_free_part
from src.grading.grader import grade
from src.grading.numeric import round_half_up, truncate
from src.grading.parser import (MAX_CHARS, Mul, Pow, Sub, parse)
from src.grading.verdict import (INCORRECT_REASONS, REASON_CATEGORY, REASON_STATUS, Reason, Status,
                                 Verdict, abstain, correct, incorrect)
from src.models.answers import POLICY_RULES, parse_answer_spec

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests/grading_cases/cases.json").read_text())


# ------------------------------------------------------------------ verdict
def test_vocabulary_sizes():
    assert len(Reason) == 27
    assert sum(s is Status.INCORRECT for s in REASON_STATUS.values()) == 11
    assert sum(s is Status.ABSTAIN for s in REASON_STATUS.values()) == 15
    assert set(REASON_CATEGORY) == INCORRECT_REASONS


def test_mismatched_status_rejected():
    with pytest.raises(ValueError):
        Verdict(Status.CORRECT, Reason.WRONG_VALUE)
    assert incorrect(Reason.NOT_SIMPLIFIED).status is Status.INCORRECT
    assert abstain(Reason.TIMEOUT).status is Status.ABSTAIN
    assert Verdict.from_dict(correct("x").to_dict()) == correct()


def test_policy_rules_labelled():
    assert len(POLICY_RULES) == 7
    assert all(r.endswith("policy, confirmed by user") for r in POLICY_RULES)


# ------------------------------------------------------------------- parser
@pytest.mark.parametrize("text", ["1/2x", "2/3x", "√2x", "2^3x", "x^2y", "1,5", "1,500", "17 900",
                                  "2 1/3", "x2", "(x+1)2", "1/2(x+1)", "x/2y", "√xy", "-x^2y", "2^3 x"])
def test_ambiguous_inputs_abstain(text):
    r = parse(text, ("x", "y"))
    assert not r.ok and r.verdict.reason is Reason.AMBIGUOUS_INPUT


@pytest.mark.parametrize("text", ["2x", "2(x+1)", "xy", "√(2x)", "x(x+1)", "x²y", "2√3", "2π", "3x^2",
                                  "(2^3)^2", "(x+1)(x-1)", "4.5×10^3", "5 cm²", "x = 3 or x = -2"])
def test_unambiguous_inputs_parse(text):
    assert parse(text, ("x", "y")).ok


def test_unary_minus_below_power_and_ast_keeps_terms():
    n = parse("-x^2", ("x",)).node
    assert type(n).__name__ == "Neg" and isinstance(n.operand, Pow)
    n = parse("2a+3b+5a-9b", ("a", "b")).node
    assert isinstance(n, Sub)  # four terms kept, not tidied


@pytest.mark.parametrize("text,reason", [
    ("x" * (MAX_CHARS + 1), Reason.TOO_LONG), ("", Reason.EMPTY_INPUT), ("   ", Reason.EMPTY_INPUT),
    ("2^3^2", Reason.TOO_COMPLEX), ("2^21", Reason.TOO_COMPLEX), ("x^(2^3)", Reason.TOO_COMPLEX),
    ("1234567890123456", Reason.TOO_COMPLEX), ("(" * 11 + "1" + ")" * 11, Reason.TOO_COMPLEX),
    ("x<y<z", Reason.TOO_COMPLEX),
])
def test_limits(text, reason):
    r = parse(text, ("x", "y", "z"))
    assert not r.ok and r.verdict.reason is reason


def test_limit_boundaries_accepted():
    assert parse("2^20").ok and parse("123456789012345").ok
    assert parse("(" * 10 + "1" + ")" * 10).ok


SECURITY = ["__import__('os').system('rm -rf /')", "eval('1')", "import os", "os.system('ls')",
            "exec('x')", "lambda: 1", "open('/etc/passwd')", "1;rm -rf /", "$(ls)", "`ls`",
            "x​+1", "\x00", "1\n2", "<script>", "{1}", "[1]", "sympy.N(1)", "1 if 1 else 2",
            "x.__class__", "'a'*10**9", "9**9**9", "2^999999", "%s", "\\", "x\x1b[2J"]


@pytest.mark.parametrize("text", SECURITY)
def test_security_inputs_abstain(text, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = parse(text, ("x",))
    assert not r.ok and r.verdict.status is Status.ABSTAIN
    assert list(tmp_path.iterdir()) == []


def test_fuzz_parser_never_raises():
    rng = random.Random(1234)
    alphabet = "0123456789xy+-*/^()=<>,.;:√π²³ %°$ab"
    for _ in range(3000):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        r = parse(s, ("x", "y"))
        assert r.ok or r.verdict.reason is not Reason.WORKER_ERROR


# ------------------------------------------------------------ arithmetic
def test_pi_enclosure_certified():
    import sympy
    ref = Fraction(str(sympy.pi.evalf(220)))  # error < 10^-200 (test-only reference)
    tol = Fraction(1, 10 ** 200)
    for p in (40, 80, 160):
        e = pi_enclosure(p)
        assert e.lo < ref - tol and ref + tol < e.hi and e.hi - e.lo <= Fraction(1, 10 ** p)


def test_sqrt_enclosure_certified():
    rng = random.Random(7)
    for _ in range(300):
        r = Fraction(rng.randint(1, 10 ** 6), rng.randint(1, 1000))
        e = sqrt_enclosure(r, 40)
        assert e.lo ** 2 <= r <= e.hi ** 2 and e.hi - e.lo <= Fraction(1, 10 ** 40)
    assert sqrt_enclosure(Fraction(16), 40).is_point


def test_rounding_half_up_and_truncate():
    assert round_half_up(Fraction("12.5"), 0) == 13 and round_half_up(Fraction("-12.5"), 0) == -13
    assert round_half_up(Fraction("2.225"), -2) == Fraction("2.23")
    assert truncate(Fraction("2.669"), -2) == Fraction("2.66") and truncate(Fraction("-2.669"), -2) == Fraction("-2.66")


def test_square_free_part_against_brute_force():
    def brute(n):
        k = 1
        for f in range(2, int(n ** 0.5) + 1):
            while n % (f * f) == 0:
                n //= f * f
                k *= f
        return k, n
    for n in range(1, 5000):
        assert square_free_part(n) == brute(n)
    assert square_free_part(20402) == (101, 2)
    assert square_free_part(31622743 ** 2) == (31622743, 1)


# --------------------------------------------------------- labelled cases
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_labelled_case(case):
    v = grade(case["spec"], case["answer"])
    assert v.reason.value == case["expect"], (case["basis"], v.detail)


def test_case_file_is_well_formed():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))
    for c in CASES:
        parse_answer_spec(c["spec"])
        assert c["expect"] in {r.value for r in Reason} and c["basis"]


# ---------------------------------------------------------- properties
def test_property_numeric_correct_rounding_is_correct_and_perturbation_is_not():
    rng = random.Random(42)
    for _ in range(300):
        v = Fraction(rng.randint(1, 10 ** 6), rng.randint(1, 997))
        spec = {"type": "numeric", "value": f"{v.numerator}/{v.denominator}"}
        from src.grading.numeric import leading_place
        q = leading_place(v) - 2
        r = round_half_up(v, q)
        text = _decimal_text(r, q)
        if "." not in text and text.endswith("0"):
            continue  # integer trailing zeros abstain by policy rule 4
        assert grade(spec, text).status is Status.CORRECT, text
        wrong = _decimal_text(r + 2 * Fraction(10) ** q, q)
        assert grade(spec, wrong).status is not Status.CORRECT, wrong


def _decimal_text(x: Fraction, q: int) -> str:
    if q >= 0:
        return str(int(x))
    s = f"{abs(x.numerator) * 10 ** (-q) // x.denominator:0{-q + 1}d}"
    out = s[:q] + "." + s[q:]
    return ("-" if x < 0 else "") + out


def test_property_expression_equivalent_forms():
    rng = random.Random(3)
    for _ in range(40):
        a, b, c = rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9)
        spec = {"type": "expression", "value": f"{a}x+{b}", "variables": ["x"], "required_form": "any"}
        assert grade(spec, f"{b}+{a}x").status is Status.CORRECT
        assert grade(spec, f"{a}x+{b + c}").reason is Reason.WRONG_VALUE


def test_value_is_never_compared_after_a_form_failure():
    spec = {"type": "exact", "value": "3√2", "required_form": "surd"}
    assert grade(spec, "√20").reason is Reason.NOT_SIMPLIFIED  # value also wrong, form decided first


def test_grade_never_raises_on_odd_input():
    for spec in ({"type": "numeric", "value": "2"}, {"type": "nonsense"}, "not json"):
        for ans in (None, 3, "", "x" * 500):
            assert isinstance(grade(spec, ans), Verdict)


# ----------------------------------------------------------- isolation
def test_importing_the_application_side_loads_no_sympy():
    code = ("import sys; import src.grading, src.grading.parser, src.grading.numeric, src.grading.exact, "
            "src.grading.structured, src.models.answers; print('sympy' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr


def test_only_symbolic_and_expression_import_sympy():
    for p in (ROOT / "src/grading").glob("*.py"):
        if p.name in ("symbolic.py", "expression.py"):
            continue
        assert "import sympy" not in p.read_text(), p.name


def test_numeric_agreement_without_proof_abstains(monkeypatch):
    import src.grading.expression as ex
    monkeypatch.setattr(ex.sp, "simplify", lambda d: d)  # proof unavailable
    spec = {"type": "expression", "value": "x+√2+√3", "variables": ["x"], "required_form": "any"}
    v = grade(spec, "x+√(5+2√6)")  # equal, but expand/cancel cannot show it
    assert v.reason is Reason.EQUIVALENCE_UNCONFIRMED


def test_simplify_can_confirm_but_never_reject():
    spec = {"type": "expression", "value": "x+√2+√3", "variables": ["x"], "required_form": "any"}
    assert grade(spec, "x+√(5+2√6)").status is Status.CORRECT
    assert grade(spec, "x+√(5+2√7)").reason is Reason.WRONG_VALUE
