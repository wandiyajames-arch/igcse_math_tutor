"""Pre-registration: freeze the specs, the answers, the grader and the prompts by hash.

Two stages, matching the protocol (specs are written from the mark scheme
before any candidate answer exists):

    python -m evaluation.freeze specs   evaluation/data/test   # after the specs, before answers
    python -m evaluation.freeze answers evaluation/data/test   # after the answers, before any run
    python -m evaluation.freeze verify  evaluation/data/test   # any time; the report runs it too

``specs`` writes FREEZE_specs.json (items + grader + prompts); ``answers``
writes FREEZE.json (adds responses and excluded parts). A stage is never
overwritten. Put both files in the protocol/appendix: anyone can re-check that
nothing changed after the freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .grader_adapter import ROOT, grader_fingerprint

STAGE_FILES = {"specs": ("items.jsonl",), "answers": ("items.jsonl", "responses.jsonl", "excluded.csv")}
STAGE_OUT = {"specs": "FREEZE_specs.json", "answers": "FREEZE.json"}


def _sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def prompts_fingerprint() -> str:
    from . import judges, leakage
    h = hashlib.sha256()
    for text in (judges.CV_PROMPT, judges.EXAMINER_PROMPT, leakage.HINT_VISIBLE, leakage.STUDENT_MESSAGES[0],
                 leakage.STUDENT_MESSAGES[1]):
        h.update(text.encode())
        h.update(b"\0")
    return h.hexdigest()


def _git_commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def snapshot(folder: Path, stage: str) -> dict:
    return {"stage": stage, "files": {f: _sha(folder / f) for f in STAGE_FILES[stage]},
            "grader": grader_fingerprint(), "prompts_sha256": prompts_fingerprint()}


def create(folder, stage: str) -> Path:
    folder = Path(folder)
    out = folder / STAGE_OUT[stage]
    if out.exists():
        raise SystemExit(f"{out} already exists: a freeze is never overwritten")
    if stage == "answers" and not (folder / STAGE_OUT["specs"]).exists():
        print("warning: no FREEZE_specs.json – specs were not frozen before the answers were written")
    if stage == "answers" and (folder / "runs").exists() and any((folder / "runs").glob("*.jsonl")):
        raise SystemExit("runs/ already has results: freeze the answers before running any system")
    snap = snapshot(folder, stage)
    snap.update({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": _git_commit(),
                 "python": sys.version.split()[0]})
    out.write_text(json.dumps(snap, indent=1) + "\n")
    print(f"wrote {out}")
    return out


def verify(folder, quiet: bool = False) -> dict:
    folder = Path(folder)
    problems, checked = [], []
    for stage, name in STAGE_OUT.items():
        p = folder / name
        if not p.exists():
            continue
        checked.append(name)
        saved = json.loads(p.read_text())
        now = snapshot(folder, stage)
        for f, h in saved["files"].items():
            if now["files"].get(f) != h:
                problems.append(f"{f} changed since {name}")
        if now["grader"]["sha256"] != saved["grader"]["sha256"]:
            problems.append(f"grader source changed since {name}")
        if now["prompts_sha256"] != saved["prompts_sha256"]:
            problems.append(f"judge/tutor prompts changed since {name}")
    if "FREEZE.json" not in checked:
        problems.append("answers not frozen yet (no FREEZE.json)")
    res = {"ok": not problems, "checked": checked, "problems": problems}
    if not quiet:
        print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["specs", "answers", "verify"])
    ap.add_argument("folder")
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        sys.exit(0 if verify(a.folder)["ok"] else 1)
    create(a.folder, a.cmd)


if __name__ == "__main__":
    main()
