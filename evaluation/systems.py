"""Run every system on a dataset and save one JSONL file per system.

    python -m evaluation.systems evaluation/data/test --system ours
    python -m evaluation.systems evaluation/data/test --system math_verify
    python -m evaluation.systems evaluation/data/test --system string_match
    python -m evaluation.systems evaluation/data/test --system qwen3_4b --backend ollama:qwen3:4b
    python -m evaluation.systems evaluation/data/test --system qwen25_7b --backend ollama:qwen2.5:7b
    python -m evaluation.systems evaluation/data/test --system compass_3b --backend hf:opencompass/CompassVerifier-3B --kind compass

Each output line: response_id, decision (rule systems) or votes (LLM judges),
reason, latency_ms. Runs resume: responses already in the file are skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from .data import load

RUNS = "runs"


def _done(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(l)["response_id"] for l in path.read_text().splitlines() if l.strip()}


# ----------------------------------------------------------------- ours
def run_ours(ds, out: Path, timeout: float = 2.0):
    from .grader_adapter import GraderHandle, grader_fingerprint
    done = _done(out)
    first_ms = None
    with GraderHandle(timeout) as g, out.open("a") as f:
        for r in ds.responses:
            if r.response_id in done:
                continue
            it = ds.item_of(r)
            t0 = time.perf_counter()
            status, reason = g.grade(it.spec, r.answer)
            ms = (time.perf_counter() - t0) * 1000
            if first_ms is None:
                first_ms = ms  # includes starting the sandboxed worker: the cold start
            f.write(json.dumps({"response_id": r.response_id, "decision": status,
                                "reason": reason, "latency_ms": round(ms, 3)}) + "\n")
        meta = {"worker_peak_rss_mb": g.peak_rss_mb(), **g.info(), "cold_start_ms": first_ms,
                "grader_fingerprint": grader_fingerprint()}
    return meta


# ---------------------------------------------------------- Math-Verify
def to_latex(text: str) -> str:
    """Typed answer -> LaTeX that Math-Verify can parse (wrapped in $...$)."""
    s = text.strip()
    s = s.replace("−", "-").replace("×", r" \times ").replace("·", r" \cdot ").replace("÷", "/")
    s = s.replace("π", r"\pi ").replace("²", "^2").replace("³", "^3")
    s = re.sub(r"√\(([^()]*)\)", r"\\sqrt{\1}", s)
    s = re.sub(r"√(\d+|[a-zA-Z])", r"\\sqrt{\1}", s)
    s = s.replace("√", r"\sqrt")
    return f"${s}$"


def run_math_verify(ds, out: Path):
    from math_verify import parse, verify
    done = _done(out)
    with out.open("a") as f:
        for r in ds.responses:
            if r.response_id in done:
                continue
            it = ds.item_of(r)
            t0 = time.perf_counter()
            gold = parse(to_latex(it.mark_scheme_answer), parsing_timeout=5)
            ans = parse(to_latex(r.answer), parsing_timeout=5)
            if not gold or not ans:
                decision, reason = "abstain", "unparseable"
            else:
                decision = "correct" if verify(gold, ans, timeout_seconds=5) else "incorrect"
                reason = ""
            ms = (time.perf_counter() - t0) * 1000
            f.write(json.dumps({"response_id": r.response_id, "decision": decision, "reason": reason,
                                "latency_ms": round(ms, 3)}) + "\n")
    return {}


# --------------------------------------------------------- string match
def normalise_ws(s: str) -> str:
    return " ".join((s or "").split())


def run_string_match(ds, out: Path):
    """Minor row (protocol §2): exact match with the mark-scheme answer after whitespace normalisation.
    It never abstains."""
    done = _done(out)
    with out.open("a") as f:
        for r in ds.responses:
            if r.response_id in done:
                continue
            t0 = time.perf_counter()
            same = normalise_ws(r.answer) == normalise_ws(ds.item_of(r).mark_scheme_answer)
            ms = (time.perf_counter() - t0) * 1000
            f.write(json.dumps({"response_id": r.response_id, "decision": "correct" if same else "incorrect",
                                "reason": "", "latency_ms": round(ms, 4)}) + "\n")
    return {}


# ----------------------------------------------------------- LLM judges
def run_judge(ds, out: Path, backend_spec: str, kind: str = "examiner", k: int = 5,
              temperature: float = 0.7, limit: int | None = None):
    from .judges import build_prompt, make_backend, sample
    backend = make_backend(backend_spec)
    done = _done(out)
    todo = sum(r.response_id not in done for r in ds.responses)
    if limit is not None:
        todo = min(todo, limit)
    n, t_start = 0, time.perf_counter()
    with out.open("a") as f:
        for r in ds.responses:
            if r.response_id in done:
                continue
            if limit is not None and n >= limit:
                break
            it = ds.item_of(r)
            prompt = build_prompt(kind, it.question, it.mark_scheme_answer, it.mark_scheme_notes, r.answer)
            if backend_spec == "mock":  # test-only: let the mock know the gold label
                prompt += f"\n[[gold={ds.gold.get(r.response_id, {}).get('label', 'incorrect')}]]"
            try:
                votes, raw, ms, lps = sample(backend, prompt, kind, k=k, temperature=temperature)
                rec = {"response_id": r.response_id, "votes": votes, "logprobs": lps, "raw": raw,
                       "latency_ms": round(ms, 1)}
            except Exception as e:
                rec = {"response_id": r.response_id, "votes": ["invalid"] * k, "raw": [], "error": str(e)[:200],
                       "latency_ms": None}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
            if n % 10 == 0 or n == todo:
                per = (time.perf_counter() - t_start) / n
                print(f"  {n}/{todo} responses, {per:.1f} s each, about {per * (todo - n) / 60:.0f} min left",
                      flush=True)
    return {"backend": backend_spec, "kind": kind, "k": k, "temperature": temperature,
            "seeds": list(range(k)), "model": backend.info()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--system", required=True)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--kind", default="examiner", choices=["examiner", "compass"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    ds = load(a.folder)
    runs = Path(a.folder) / RUNS
    runs.mkdir(exist_ok=True)
    out = runs / f"{a.system}.jsonl"
    if a.system == "ours":
        meta = run_ours(ds, out)
    elif a.system == "math_verify":
        meta = run_math_verify(ds, out)
    elif a.system == "string_match":
        meta = run_string_match(ds, out)
    else:
        if not a.backend:
            ap.error("--backend is required for LLM judges")
        meta = run_judge(ds, out, a.backend, a.kind, a.k, limit=a.limit)
    (runs / f"{a.system}.meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
