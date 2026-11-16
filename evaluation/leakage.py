"""RQ3 (light): does the tutor's first hint reveal the final answer?

For 50 test question parts the local LLM writes a first hint twice:
  visible  – the prompt includes the mark-scheme answer (a naive tutor);
  withheld – the prompt has only the question (our design before reveal).
A hint leaks if it states the final answer. Detection is automatic: every
number or short expression in the hint is graded with the frozen grader
against the item's spec; any ``correct`` verdict counts as a leak. One rater
then checks all hints by hand (``manual_leak`` column) and we report the
agreement.

    python -m evaluation.leakage generate evaluation/data/test --backend ollama:llama3.2:1b
    python -m evaluation.leakage score evaluation/data/test
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

Give ONE short first hint (at most two sentences) that helps the student take the next step."""

HINT_WITHHELD = """You are a friendly IGCSE Mathematics tutor. A student is stuck on this question.

Question: {question}

Give ONE short first hint (at most two sentences) that helps the student take the next step."""

_CAND = re.compile(r"[-−]?\(?[0-9a-zA-Zπ√.]+(?:\s*[-+−×*/÷^:,]\s*\(?[-−]?[0-9a-zA-Zπ√.()²³]+\)?)*\)?[²³]?")


def select_items(ds, n: int = 50, seed: int = 2027) -> list[str]:
    items = sorted(i for i, it in ds.items.items() if it.split == "test")
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
    return list(dict.fromkeys(out))


def detect_leak(spec: dict, hint: str):
    from src.grading.grader import grade
    for c in candidates(hint):
        if grade(spec, c).status.value == "correct":
            return True, c
    return False, ""


def generate(folder, backend_spec: str, n: int = 50):
    from .judges import make_backend
    ds = load(folder)
    backend = make_backend(backend_spec)
    rows = []
    for iid in select_items(ds, n):
        it = ds.items[iid]
        for cond, tmpl in (("visible", HINT_VISIBLE), ("withheld", HINT_WITHHELD)):
            hint = backend.generate(tmpl.format(question=it.question, answer=it.mark_scheme_answer), 0.7, 0)
            hint = re.sub(r"<think>.*?</think>", "", hint, flags=re.S).strip()
            leak, cand = detect_leak(it.spec, hint)
            rows.append({"item_id": iid, "condition": cond, "hint": hint, "auto_leak": int(leak),
                         "matched": cand, "manual_leak": ""})
    out = Path(folder) / "leakage.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} hints, model {backend_spec})")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def score(folder):
    rows = list(csv.DictReader((Path(folder) / "leakage.csv").open(encoding="utf-8")))
    res = {}
    for cond in ("visible", "withheld"):
        rs = [r for r in rows if r["condition"] == cond]
        k = sum(int(r["auto_leak"]) for r in rs)
        res[cond] = {"n": len(rs), "auto_leaks": k, "auto_rate": k / len(rs) if rs else None,
                     "auto_ci95": wilson(k, len(rs))}
        man = [r for r in rs if r["manual_leak"].strip() in ("0", "1")]
        if man:
            km = sum(int(r["manual_leak"]) for r in man)
            res[cond].update({"manual_n": len(man), "manual_rate": km / len(man), "manual_ci95": wilson(km, len(man))})
    checked = [r for r in rows if r["manual_leak"].strip() in ("0", "1")]
    if checked:
        res["auto_vs_manual_agreement"] = sum(r["manual_leak"].strip() == r["auto_leak"] for r in checked) / len(checked)
    out = Path(folder) / "results"
    out.mkdir(exist_ok=True)
    (out / "leakage.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "score"])
    ap.add_argument("folder")
    ap.add_argument("--backend", default="ollama:llama3.2:1b")
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    generate(a.folder, a.backend, a.n) if a.cmd == "generate" else score(a.folder)
