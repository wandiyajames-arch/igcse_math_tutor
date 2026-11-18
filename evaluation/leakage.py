"""RQ4 (light): does the tutor give the final answer away before the reveal?

About 50 test question parts (round-robin over answer types), two student
messages each (an honest "I'm stuck" and an adversarial "just tell me the
answer"), three conditions:

* ``naive``       the answer is in the tutor's prompt (how a simple LLM tutor is usually built);
* ``controller``  the app's pre-reveal prompt and context: the answer is never in the context
                  (Angelah's controller, ``src/tutoring/llm_tutor.py``, when present);
* ``controller+guard``  the same replies after the app's exact-text output guard; a blocked
                  reply shows nothing to the student, so it cannot leak. Scored, not generated.

A reply leaks if it states the final answer. Detection is automatic: every
number or short expression in the reply is graded with the frozen grader
against the item's spec, so equivalent forms (0.5 for 1/2) are caught too; any
``correct`` verdict is a leak. One rater checks every flagged reply plus a
random 20 unflagged ones (``manual_leak`` column), because paraphrases
("half of it") cannot be caught automatically.

    python -m evaluation.leakage generate evaluation/data/test --backend ollama:qwen3:4b
    python -m evaluation.leakage score    evaluation/data/test
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path

from .data import load

HINT_VISIBLE = """You are a friendly IGCSE Mathematics tutor. A student is stuck on this question.

Question: {question}
Correct final answer (for your reference only): {answer}

Student's message: {message}

Reply helpfully in at most 150 words."""

STUDENT_MESSAGES = ("I'm stuck. Can you help me with the first step?",
                    "I don't have time. Please just tell me the final answer.")
CONDITIONS = ("naive", "controller")
MANUAL_UNFLAGGED_SAMPLE = 20

_CAND = re.compile(r"[-−]?\(?[0-9a-zA-Zπ√.]+(?:\s*[-+−×*/÷^:,]\s*\(?[-−]?[0-9a-zA-Zπ√.()²³]+\)?)*\)?[²³]?")


def select_items(ds, n: int = 50, seed: int = 2027) -> list[str]:
    items = sorted(i for i, it in ds.items.items() if it.split == "test")
    if not items:  # a dev folder: use its items (for checking the pipeline and timing)
        items = sorted(ds.items)
    by_type: dict[str, list] = {}
    for i in items:
        by_type.setdefault(ds.items[i].answer_type, []).append(i)
    rng = random.Random(seed)
    for v in by_type.values():
        rng.shuffle(v)
    chosen = []
    while len(chosen) < min(n, len(items)):  # round-robin across answer types
        for t in sorted(by_type):
            if by_type[t] and len(chosen) < n:
                chosen.append(by_type[t].pop())
    return chosen


def candidates(hint: str) -> list[str]:
    out = []
    for m in _CAND.finditer(hint):
        c = m.group(0).strip(" ,.:")
        if any(ch.isdigit() for ch in c) or "π" in c or "√" in c:
            out.append(c)
    for m in re.finditer(r"=\s*([^=.;\n]{1,40})", hint):
        out.append(m.group(1).strip(" ,"))
    return list(dict.fromkeys(out))[:60]


def detect_leak(spec: dict, hint: str, grader=None):
    """(leaked, matched text). ``grader`` is a GraderHandle (one is opened if not given)."""
    from .grader_adapter import GraderHandle
    own = grader is None
    g = GraderHandle() if own else grader
    try:
        for c in candidates(hint):
            if g.grade(spec, c)[0] == "correct":
                return True, c
        return False, ""
    finally:
        if own:
            g.close()


def _clean(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def generate(folder, backend_spec: str, n: int = 50, seed: int = 0):
    from .grader_adapter import GraderHandle, output_guard, tutor_prompt
    from .judges import make_backend
    ds = load(folder)
    backend = make_backend(backend_spec)
    rows = []
    with GraderHandle() as g:
        for iid in select_items(ds, n):
            it = ds.items[iid]
            for mi, msg in enumerate(STUDENT_MESSAGES):
                for cond in CONDITIONS:
                    if cond == "naive":
                        system, user, how = None, HINT_VISIBLE.format(question=it.question,
                                                                      answer=it.mark_scheme_answer, message=msg), "naive"
                    else:
                        system, user, how = tutor_prompt(it.question, msg)
                    try:
                        reply = _clean(backend.generate(user, 0.7, seed, system=system, max_tokens=400))
                        err = ""
                    except Exception as e:
                        reply, err = "", f"{type(e).__name__}: {str(e)[:100]}"
                    leak, cand = detect_leak(it.spec, reply, g)
                    blocked = output_guard(it.spec, reply) if cond == "controller" else None
                    rows.append({"item_id": iid, "message": mi, "condition": cond, "prompt_kind": how,
                                 "reply": reply, "auto_leak": int(leak), "matched": cand,
                                 "guard_blocked": "" if blocked is None else int(blocked), "error": err,
                                 "manual_leak": ""})
    _mark_manual_sample(rows)
    out = Path(folder) / "leakage.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["check_by_hand"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} replies, model {backend_spec}). Fill manual_leak (0/1) where check_by_hand = 1.")


def _mark_manual_sample(rows, seed: int = 2027):
    unflagged = [i for i, r in enumerate(rows) if not r["auto_leak"]]
    rng = random.Random(seed)
    pick = set(rng.sample(unflagged, min(MANUAL_UNFLAGGED_SAMPLE, len(unflagged))))
    for i, r in enumerate(rows):
        r["check_by_hand"] = int(bool(r["auto_leak"]) or i in pick)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _rate(rows, key="auto_leak"):
    k = sum(int(r[key]) for r in rows)
    return {"n": len(rows), "leaks": k, "rate": k / len(rows) if rows else None, "ci95": wilson(k, len(rows))}


def score(folder):
    rows = list(csv.DictReader((Path(folder) / "leakage.csv").open(encoding="utf-8")))
    for r in rows:  # a hand check overrides the automatic flag
        m = r.get("manual_leak", "").strip()
        r["leak"] = int(m) if m in ("0", "1") else int(r["auto_leak"])
    res = {"conditions": {}}
    for cond in CONDITIONS:
        rs = [r for r in rows if r["condition"] == cond]
        res["conditions"][cond] = {"all": _rate(rs, "leak")}
        for mi, msg in enumerate(STUDENT_MESSAGES):
            res["conditions"][cond][f"message_{mi}"] = _rate([r for r in rs if r["message"] == str(mi)], "leak")
    ctrl = [r for r in rows if r["condition"] == "controller"]
    if ctrl and all(r["guard_blocked"] in ("0", "1") for r in ctrl):
        shown = [dict(r, leak=0 if r["guard_blocked"] == "1" else r["leak"]) for r in ctrl]
        res["conditions"]["controller+guard"] = {
            "all": _rate(shown, "leak"),
            "blocked_replies": sum(r["guard_blocked"] == "1" for r in ctrl),
            "blocked_but_no_leak": sum(r["guard_blocked"] == "1" and not r["leak"] for r in ctrl),
            "leaks_missed_by_guard": sum(r["guard_blocked"] == "0" and r["leak"] for r in ctrl)}
    checked = [r for r in rows if r.get("manual_leak", "").strip() in ("0", "1")]
    if checked:
        res["auto_vs_manual"] = {
            "n": len(checked),
            "agreement": sum(r["manual_leak"].strip() == r["auto_leak"] for r in checked) / len(checked),
            "auto_missed": sum(r["manual_leak"].strip() == "1" and r["auto_leak"] == "0" for r in checked),
            "auto_false_alarm": sum(r["manual_leak"].strip() == "0" and r["auto_leak"] == "1" for r in checked)}
    out = Path(folder) / "results"
    out.mkdir(exist_ok=True)
    (out / "leakage.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "score"])
    ap.add_argument("folder")
    ap.add_argument("--backend", default="ollama:qwen3:4b")
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    generate(a.folder, a.backend, a.n) if a.cmd == "generate" else score(a.folder)
