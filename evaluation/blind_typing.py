"""Blind answer typing by classmates who have never seen the grader (protocol §4, Change 1).

    python -m evaluation.blind_typing assign  evaluation/data/test --typists 4
    (send each typing_sheet_<n>.xlsx to one classmate; they fill the typed_answer column)
    python -m evaluation.blind_typing collect evaluation/data/test

``assign`` draws about one answer in five at random (seed 2027) and gives each
classmate a sheet with the question and the answer's plain description only:
no answer text, no ids that reveal how it was built, nothing about the grader.
It needs ``typing_guide.txt`` in the folder: the short guide the authors wrote
*without looking at the grader* ("type it as a student would on a phone or
keyboard"); it is copied into every sheet. ``collect`` writes
``typed_by_classmates.csv``; from then on those responses carry the classmate's
typing (``data.load``), and the report gives their results separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font

SHARE, SEED = 0.2, 2027


def _responses(folder: Path) -> list[dict]:
    return [json.loads(l) for l in (folder / "responses.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def assign(folder, typists: int = 4, share: float = SHARE, seed: int = SEED):
    folder = Path(folder)
    if (folder / "typing_assignment.csv").exists():
        raise SystemExit("typing_assignment.csv already exists: the draw is made once")
    guide_p = folder / "typing_guide.txt"
    if not guide_p.exists():
        raise SystemExit("write typing_guide.txt first (without looking at the grader), then run assign")
    if not 3 <= typists <= 5:
        print("note: the protocol says 3 to 5 classmates")
    guide = guide_p.read_text(encoding="utf-8").strip().splitlines()
    resp = sorted(_responses(folder), key=lambda r: r["response_id"])
    items = {json.loads(l)["item_id"]: json.loads(l) for l in (folder / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    rng = random.Random(seed)
    chosen = rng.sample(resp, round(share * len(resp)))
    rng.shuffle(chosen)
    rows = []
    for k, r in enumerate(chosen):
        rows.append({"response_id": r["response_id"], "typist": f"typist{k % typists + 1}", "row": f"T{k + 1:03d}"})
    with (folder / "typing_assignment.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["response_id", "typist", "row"])
        w.writeheader()
        w.writerows(rows)
    by_id = {r["response_id"]: r for r in resp}
    for t in sorted({x["typist"] for x in rows}):
        wb = Workbook()
        g = wb.active
        g.title = "Guide"
        for line in ["How to type the answers"] + guide:
            g.append([line])
        g["A1"].font = Font(bold=True)
        g.column_dimensions["A"].width = 110
        ws = wb.create_sheet("Answers")
        ws.append(["row", "question", "answer to type (description)", "typed_answer"])
        for c in ws[1]:
            c.font = Font(bold=True)
        for x in rows:
            if x["typist"] != t:
                continue
            r = by_id[x["response_id"]]
            ws.append([x["row"], items[r["item_id"]]["question"], r["description"], ""])
        for col, w in zip("ABCD", (8, 70, 50, 30)):
            ws.column_dimensions[col].width = w
        for row in ws.iter_rows(min_row=2):
            for c in row[:3]:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        wb.save(folder / f"typing_sheet_{t}.xlsx")
    print(f"{len(rows)} of {len(resp)} answers drawn for {typists} classmates -> typing_sheet_typist*.xlsx")
    return rows


def collect(folder):
    folder = Path(folder)
    rows = list(csv.DictReader((folder / "typing_assignment.csv").open(encoding="utf-8-sig")))
    by_row = {x["row"]: x for x in rows}
    typed, missing = [], []
    for t in sorted({x["typist"] for x in rows}):
        p = folder / f"typing_sheet_{t}.xlsx"
        got = {}
        if p.exists():
            for r in list(load_workbook(p, read_only=True)["Answers"].iter_rows(values_only=True))[1:]:
                if r and r[0] in by_row and r[3] not in (None, ""):
                    got[r[0]] = str(r[3])
        for x in rows:
            if x["typist"] == t:
                if x["row"] in got:
                    typed.append({"response_id": x["response_id"], "answer": got[x["row"]], "typist": t})
                else:
                    missing.append(f"{t} {x['row']}")
    with (folder / "typed_by_classmates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["response_id", "answer", "typist"])
        w.writeheader()
        w.writerows(typed)
    print(f"{len(typed)} typed answers collected; {len(missing)} missing" + (": " + ", ".join(missing[:10]) if missing else ""))
    return typed, missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["assign", "collect"])
    ap.add_argument("folder")
    ap.add_argument("--typists", type=int, default=4)
    a = ap.parse_args()
    assign(a.folder, a.typists) if a.cmd == "assign" else collect(a.folder)
