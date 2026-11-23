"""One interface to the grader under test, whichever version is in ``src/``.

The harness runs unchanged on two code bases:

* ``angelah`` – the original grader (``src.grading.grader.Grader``), the one the paper evaluates;
* ``rebuild`` – the stop-gap rebuild (``src.grading.grader.grade_safe``).

Everything the harness needs from the grader goes through this file, so
``src/`` is never edited (the grader stays frozen). The two versions share the
27 reason codes; their answer-spec JSON differs in a few keys, handled here.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def flavour() -> str:
    import src.grading.grader as g
    return "angelah" if hasattr(g, "Grader") else "rebuild"


# ------------------------------------------------------------- grading
class GraderHandle:
    """A sandboxed grader process for a whole run. ``grade`` never raises."""

    def __init__(self, timeout: float = 2.0):
        self.flavour = flavour()
        t0 = time.perf_counter()
        from src.grading.sandbox import SandboxClient
        if self.flavour == "angelah":
            from src.grading.grader import Grader
            self._client = SandboxClient(job_timeout=timeout)
            self._grader = Grader(self._client)
        else:
            from src.grading.grader import grade_safe
            self._client = SandboxClient()
            self._grade_safe, self._timeout = grade_safe, timeout
        self.startup_ms = (time.perf_counter() - t0) * 1000

    def grade(self, spec: dict, text: str) -> tuple[str, str]:
        """(status, reason) – status is correct | incorrect | abstain."""
        if is_unsupported(spec):  # protocol §3: included part the grader cannot express -> always abstain
            return "abstain", "spec_unsupported"
        try:
            if self.flavour == "angelah":
                v = self._grader.grade(spec, text)
            else:
                v = self._grade_safe(spec, text, client=self._client, timeout=self._timeout)
            return v.status.value, v.reason.value
        except Exception as e:  # never expected; an error is an abstention, never a verdict
            return "abstain", f"harness_error:{type(e).__name__}"

    def peak_rss_mb(self):
        pid = getattr(self._client, "pid", None)
        if pid is None:
            proc = getattr(self._client, "_proc", None)
            pid = getattr(proc, "pid", None)
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmHWM"):
                    return round(int(line.split()[1]) / 1024, 1)
        except Exception:
            return None

    def info(self) -> dict:
        ready = getattr(self._client, "ready_info", None)
        return {"grader": self.flavour, "startup_ms": round(self.startup_ms, 1),
                "worker_ready": ready if isinstance(ready, (dict, str, type(None))) else str(ready)}

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


UNSUPPORTED = "unsupported"


def is_unsupported(spec: dict) -> bool:
    """An included part whose answer the grader's spec format cannot express (e.g. an inequality)."""
    return (spec or {}).get("type") == UNSUPPORTED


def validate_spec(spec: dict) -> None:
    """Raise if the spec is not valid for the grader in ``src/``."""
    if is_unsupported(spec):
        return
    from src.models.answers import parse_answer_spec
    parse_answer_spec(spec)


# ------------------------------------------------------- spec helpers
def _components(spec: dict) -> list:
    return spec.get("values") or spec.get("solutions") or spec.get("components") or []


def mark_scheme_answer(spec: dict) -> str:
    """A plain answer string built from a spec (used for the pilot only)."""
    t = spec["type"]
    if t == UNSUPPORTED:
        return spec.get("value", "")
    if t == "ratio":
        return ":".join(str(p) for p in spec["parts"])
    if t == "solutions":
        return " or ".join(f"{spec['variable']} = {c['value']}" for c in _components(spec))
    if t in ("coordinates", "vector"):
        return "(" + ", ".join(c["value"] for c in _components(spec)) + ")"
    return spec["value"]


def item_tag(item) -> str:
    """The value-only / form-sensitive tag set from the mark scheme (protocol §3); falls back to the spec."""
    return getattr(item, "tag", "") or stratum(item.spec)


def stratum(spec: dict) -> str:
    """``value_only`` if any value-equal answer earns the mark; else ``form_sensitive``.

    Form-sensitive items need more than the right value: a rounding/accuracy rule,
    a required form (fraction, standard form, surd, π, simplified/expanded/factorised),
    a unit, cao, a three-figure bearing or a ratio in simplest form. Math-Verify checks
    value only, so its errors on form-sensitive items are reported separately.
    """
    t = spec.get("type")
    if t == UNSUPPORTED:
        return "form_sensitive"
    if t in ("exact", "expression", "bearing", "ratio"):
        return "form_sensitive"
    if spec.get("equivalence") == "cao" or spec.get("unit"):
        return "form_sensitive"
    if t in ("solutions", "coordinates", "vector"):
        return "form_sensitive" if any(stratum(c) == "form_sensitive" for c in _components(spec)) else "value_only"
    acc = spec.get("accuracy") or {}
    if spec.get("required_form", "any") not in ("any", None):
        return "form_sensitive"
    if t == "numeric" and acc.get("kind", "sf") != "exact":
        return "form_sensitive"  # 3 s.f. default or a stated accuracy
    if t == "angle":
        return "form_sensitive"  # 1 d.p. default
    return "value_only"


SPEC_EXAMPLES = {
    "angelah": [
        ("numeric, 3 s.f. default", '{"type":"numeric","value":"25π","unit":"cm²"}'),
        ("numeric, 1 d.p.", '{"type":"numeric","value":"12.345","accuracy":{"kind":"dp","digits":1}}'),
        ("numeric, exact fraction", '{"type":"numeric","value":"2/3","accuracy":{"kind":"exact"},"required_form":"fraction"}'),
        ("numeric, range", '{"type":"numeric","value":"2.4","accept_range":["2.3","2.5"]}'),
        ("numeric, standard form", '{"type":"numeric","value":"4500","accuracy":{"kind":"exact"},"required_form":"standard_form"}'),
        ("numeric, cao", '{"type":"numeric","value":"7","accuracy":{"kind":"exact"},"equivalence":"cao"}'),
        ("angle (1 d.p.)", '{"type":"angle","value":"36.87"}'),
        ("bearing", '{"type":"bearing","value":"65"}'),
        ("exact, in terms of π", '{"type":"exact","value":"100-4π","form":"in_terms_of_pi"}'),
        ("exact, surd", '{"type":"exact","value":"3√2","form":"surd"}'),
        ("exact, fraction", '{"type":"exact","value":"-5/6","form":"fraction_simplest"}'),
        ("expression, simplified", '{"type":"expression","value":"7a-6b","variables":["a","b"]}'),
        ("expression, expanded", '{"type":"expression","value":"x^2+5x+6","variables":["x"],"required_form":"expanded"}'),
        ("expression, factorised", '{"type":"expression","value":"(x+2)(x+3)","variables":["x"],"required_form":"factorised_fully"}'),
        ("solutions", '{"type":"solutions","variable":"x","values":[{"type":"exact","value":"3","form":"fraction_simplest"},{"type":"exact","value":"-2","form":"fraction_simplest"}]}'),
        ("coordinates", '{"type":"coordinates","components":[{"type":"exact","value":"3","form":"fraction_simplest"},{"type":"exact","value":"-2","form":"fraction_simplest"}]}'),
        ("vector", '{"type":"vector","components":[{"type":"exact","value":"4","form":"fraction_simplest"},{"type":"exact","value":"-1","form":"fraction_simplest"}]}'),
        ("ratio", '{"type":"ratio","parts":[2,3,4]}'),
        ("grader cannot express (e.g. inequality) - grader abstains", '{"type":"unsupported","value":"x > 3"}'),
    ],
    "rebuild": [
        ("numeric, 3 s.f. default", '{"type":"numeric","value":"25π","unit":"cm²"}'),
        ("numeric, 1 d.p.", '{"type":"numeric","value":"12.345","accuracy":{"kind":"dp","digits":1}}'),
        ("numeric, exact", '{"type":"numeric","value":"2/3","accuracy":{"kind":"exact","digits":null}}'),
        ("numeric, range", '{"type":"numeric","value":"2.4","accept_range":["2.3","2.5"]}'),
        ("numeric, standard form", '{"type":"numeric","value":"4500","required_form":"standard_form"}'),
        ("angle (1 d.p.)", '{"type":"angle","value":"36.87"}'),
        ("bearing", '{"type":"bearing","value":"65"}'),
        ("exact, in terms of π", '{"type":"exact","value":"100-4π","required_form":"in_terms_of_pi"}'),
        ("exact, surd", '{"type":"exact","value":"3√2","required_form":"surd"}'),
        ("exact, fraction", '{"type":"exact","value":"-5/6","required_form":"fraction_simplest"}'),
        ("expression, simplified", '{"type":"expression","value":"7a-6b","variables":["a","b"]}'),
        ("expression, expanded", '{"type":"expression","value":"x^2+5x+6","variables":["x"],"required_form":"expanded"}'),
        ("expression, factorised", '{"type":"expression","value":"(x+2)(x+3)","variables":["x"],"required_form":"factorised_fully"}'),
        ("solutions", '{"type":"solutions","variable":"x","solutions":[{"type":"numeric","value":"3"},{"type":"numeric","value":"-2"}]}'),
        ("coordinates", '{"type":"coordinates","components":[{"type":"numeric","value":"3"},{"type":"numeric","value":"-2"}]}'),
        ("vector", '{"type":"vector","components":[{"type":"numeric","value":"4"},{"type":"numeric","value":"-1"}]}'),
        ("ratio", '{"type":"ratio","parts":[2,3,4]}'),
        ("grader cannot express (e.g. inequality) - grader abstains", '{"type":"unsupported","value":"x > 3"}'),
    ],
}


def spec_examples():
    return SPEC_EXAMPLES[flavour()]


# ------------------------------------------------------- regression cases
def regression_cases() -> list[dict]:
    """The developers' own labelled cases, normalised to {id, spec, answer, expect, basis}.

    For Angelah's grader only the 325 approved cases are used.
    """
    cases_dir = ROOT / "tests" / "grading_cases"
    if (cases_dir / "approved.json").exists():
        approved = {a["id"] for a in json.loads((cases_dir / "approved.json").read_text())["approved_cases"]}
        out = []
        for name in ("numeric", "exact", "expression", "structured", "markscheme_derived"):
            p = cases_dir / f"{name}.json"
            if not p.exists():
                continue
            for c in json.loads(p.read_text())["cases"]:
                if c["id"] in approved:
                    out.append({"id": c["id"], "spec": c["spec"], "answer": c["input"],
                                "expect": c["expected"]["reason"], "basis": c.get("basis", "")})
        return out
    return [{"id": c["id"], "spec": c["spec"], "answer": c["answer"], "expect": c["expect"],
             "basis": c.get("basis", "")} for c in json.loads((cases_dir / "cases.json").read_text())]


# ------------------------------------------------------- freeze identity
GRADER_FILES = ("src/grading", "src/models/answers.py")


def grader_fingerprint() -> dict:
    """SHA-256 over every grader source file (sorted paths + contents)."""
    h = hashlib.sha256()
    files = []
    for g in GRADER_FILES:
        p = ROOT / g
        files += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    for f in files:
        h.update(str(f.relative_to(ROOT)).encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return {"flavour": flavour(), "files": len(files), "sha256": h.hexdigest()}


# ---------------------------------------------------------- tutor (RQ4)
def tutor_prompt(question: str, student_message: str):
    """(system, user) for the tutor before the reveal, as the app builds it.

    With Angelah's controller this is her approved pre-reveal prompt and user-turn
    template: the answer is never in the context. Otherwise a plain equivalent.
    """
    try:
        from src.tutoring.llm_tutor import PRE_REVEAL_PROMPT, PreRevealContext, render_user_turn
        ctx = PreRevealContext(stem=question, hints=(), gradable=True, feedback=(), student_message=student_message)
        return PRE_REVEAL_PROMPT, render_user_turn(ctx), "controller"
    except Exception:
        system = ("You are a mathematics tutor helping a student with one question. Do not give the final "
                  "answer or a full worked solution. Keep the reply under 150 words.")
        return system, f"Question:\n{question}\n\nStudent's message:\n{student_message}", "plain"


def output_guard(spec: dict, text: str):
    """Angelah's exact-text output guard (P2-T8) if present: True = reply would be blocked."""
    try:
        from src.grading.redaction import contains_distinctive, protected_texts
    except Exception:
        return None
    try:
        return bool(contains_distinctive(text, protected_texts(spec)))
    except Exception:
        return True  # the app fails closed
