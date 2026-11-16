"""Run every system on a dataset and save one JSONL file per system.

    python -m evaluation.systems evaluation/data/test --system ours
    python -m evaluation.systems evaluation/data/test --system math_verify
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
    from src.grading.grader import grade_safe
    from src.grading.sandbox import SandboxClient
    client = SandboxClient()
    done = _done(out)
    with out.open("a") as f:
        for r in ds.responses:
            if r.response_id in done:
                continue
            it = ds.item_of(r)
            t0 = time.perf_counter()
            v = grade_safe(it.spec, r.answer, client=client, timeout=timeout)
            ms = (time.perf_counter() - t0) * 1000
            f.write(json.dumps({"response_id": r.response_id, "decision": v.status.value,
                                "reason": v.reason.value, "latency_ms": round(ms, 3)}) + "\n")
    peak = _peak_rss_mb(client)
    client.close()
    return {"worker_peak_rss_mb": peak, "cold_start": client.ready_info}


def _peak_rss_mb(client):
    try:
        for line in Path(f"/proc/{client._proc.pid}/status").read_text().splitlines():
            if line.startswith("VmHWM"):
                return round(int(line.split()[1]) / 1024, 1)
    except Exception:
        return None


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


# ----------------------------------------------------------- LLM judges
def run_judge(ds, out: Path, backend_spec: str, kind: str = "examiner", k: int = 5,
              temperature: float = 0.7, limit: int | None = None):
    from .judges import build_prompt, make_backend, sample
    backend = make_backend(backend_spec)
    done = _done(out)
    n = 0
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
                votes, raw, ms = sample(backend, prompt, kind, k=k, temperature=temperature)
                rec = {"response_id": r.response_id, "votes": votes, "raw": raw, "latency_ms": round(ms, 1)}
            except Exception as e:
                rec = {"response_id": r.response_id, "votes": ["invalid"] * k, "raw": [], "error": str(e)[:200],
                       "latency_ms": None}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
    return {"backend": backend_spec, "kind": kind, "k": k, "temperature": temperature}


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
    else:
        if not a.backend:
            ap.error("--backend is required for LLM judges")
        meta = run_judge(ds, out, a.backend, a.kind, a.k, limit=a.limit)
    (runs / f"{a.system}.meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
