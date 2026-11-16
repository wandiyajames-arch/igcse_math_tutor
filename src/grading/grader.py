"""``grade()`` facade: one function grades any answer against any spec.

``grade`` runs in the current process (used inside the sandbox worker and in
tests). ``grade_safe`` sends the job to the sandbox worker and is what the
application should call on untrusted input.
"""
from __future__ import annotations

from src.models.answers import parse_answer_spec

from .verdict import Reason, Verdict, abstain


def grade(spec, answer: str) -> Verdict:
    """Grade ``answer`` against ``spec`` (a spec model, dict or JSON string)."""
    try:
        if not hasattr(spec, "type"):
            spec = parse_answer_spec(spec)
    except Exception as e:
        return abstain(Reason.NOT_AUTO_GRADABLE, f"invalid spec: {type(e).__name__}")
    if not isinstance(answer, str):
        return abstain(Reason.UNPARSEABLE, "answer is not a string")
    t = spec.type
    if t in ("numeric", "angle", "bearing"):
        from .numeric import grade_numeric
        return grade_numeric(spec, answer)
    if t == "exact":
        from .exact import grade_exact
        return grade_exact(spec, answer)
    if t == "expression":
        from .expression import grade_expression
        return grade_expression(spec, answer)
    if t in ("solutions", "coordinates", "vector", "ratio"):
        from .structured import grade_structured
        return grade_structured(spec, answer)
    return abstain(Reason.NOT_AUTO_GRADABLE, f"unknown spec type {t}")


_DEFAULT_CLIENT = None


def grade_safe(spec, answer: str, client=None, timeout: float = 2.0) -> Verdict:
    """Grade in the resource-limited sandbox worker. Never raises."""
    global _DEFAULT_CLIENT
    from .sandbox import SandboxClient
    if client is None:
        if _DEFAULT_CLIENT is None:
            _DEFAULT_CLIENT = SandboxClient()
        client = _DEFAULT_CLIENT
    spec_dict = spec.model_dump(mode="json") if hasattr(spec, "model_dump") else spec
    reply = client.call("grade", {"spec": spec_dict, "answer": answer}, timeout=timeout)
    if isinstance(reply, Verdict):
        return reply
    try:
        return Verdict.from_dict(reply)
    except Exception:
        return abstain(Reason.WORKER_ERROR, "malformed verdict from worker")
