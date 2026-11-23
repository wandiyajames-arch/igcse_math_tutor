"""Author the test set in Excel, then convert it to the harness format.

    python -m evaluation.author template evaluation/data/test/authoring.xlsx
    python -m evaluation.author sample   evaluation/data/test/authoring.xlsx evaluation/data/test   # if > 100 parts
    python -m evaluation.author build    evaluation/data/test/authoring.xlsx evaluation/data/test

``sample`` applies the size cap of protocol §3 (at most 100 parts, stratified
by paper, seed 2027) and writes selected_items.txt; ``build`` then keeps only
those parts. ``build`` writes items.jsonl, responses.jsonl and excluded.csv and
lists every problem (invalid spec, missing citation, intended label that
does not match the source, ...) so nothing broken reaches the raters.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from .data import (CATEGORIES, EXCLUSION_REASONS, GENERIC_ERRORS, SOURCES, TAGS, Dataset, Item, Response,
                   validate)
from .grader_adapter import GraderHandle, is_unsupported, spec_examples

ITEM_COLS = ["item_id", "split", "paper", "session", "question_no", "topic", "tag", "question",
             "mark_scheme_answer", "mark_scheme_notes", "spec_json"]
RESP_COLS = ["response_id", "item_id", "description", "answer", "source", "source_note",
             "intended_label", "intended_category"]
EXCL_COLS = ["part_id", "split", "paper", "question_no", "reason", "note"]
TOPICS = ["Number", "Algebra and graphs", "Coordinate geometry", "Geometry", "Mensuration",
          "Trigonometry", "Transformations and vectors", "Probability", "Statistics"]
CAP, SEED = 100, 2027


def _header(ws, cols):
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="305496")
    ws.freeze_panes = "A2"


def _dropdown(ws, values, rng):
    dv = DataValidation(type="list", formula1='"' + ",".join(values) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(rng)


def template(path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Items"
    _header(ws, ITEM_COLS)
    ws.append(["0580-s25-22-q05", "test", "0580/22", "June 2025", "5", "Algebra and graphs", "form_sensitive",
               "[Type the question text here]", "[mark-scheme answer]", "[e.g. oe, awrt]",
               '[spec JSON — copy a pattern from the "Spec examples" sheet]'])
    for col, w in zip("ABCDEFGHIJK", (18, 8, 10, 12, 10, 22, 16, 60, 22, 18, 70)):
        ws.column_dimensions[col].width = w
    _dropdown(ws, ("dev", "test"), "B2:B1000")
    _dropdown(ws, TOPICS, "F2:F1000")
    _dropdown(ws, TAGS, "G2:G1000")

    rs = wb.create_sheet("Responses")
    _header(rs, RESP_COLS)
    rs.append(["0580-s25-22-q05-r1", "0580-s25-22-q05", "[plain words: the answer in the mark scheme's form]",
               "[answer exactly as typed]", "ms_form", "", "correct", ""])
    rs.append(["0580-s25-22-q05-r2", "0580-s25-22-q05", "[plain words: the same value in another accepted form]",
               "[answer]", "alt_form", "", "correct", ""])
    rs.append(["0580-s25-22-q05-r3", "0580-s25-22-q05", "[plain words: the error the report describes]",
               "[answer]", "examiner_report", "[the report sentence this error comes from]", "incorrect", "value"])
    rs.append(["0580-s25-22-q05-r4", "0580-s25-22-q05", "[plain words: the generic error applied here]",
               "[answer]", "generic", "premature_rounding", "incorrect", "accuracy"])
    for col, w in zip("ABCDEFGH", (22, 18, 45, 24, 16, 50, 14, 16)):
        rs.column_dimensions[col].width = w
    _dropdown(rs, SOURCES, "E2:E5000")
    _dropdown(rs, ("correct", "incorrect"), "G2:G5000")
    _dropdown(rs, CATEGORIES, "H2:H5000")

    ex = wb.create_sheet("Spec examples")
    _header(ex, ["answer type", "spec_json pattern"])
    for row in spec_examples():
        ex.append(list(row))
    ex.append([])
    ex.append(["generic errors (source_note when source = generic)", ", ".join(GENERIC_ERRORS)])
    ex.column_dimensions["A"].width = 34
    ex.column_dimensions["B"].width = 120

    xs = wb.create_sheet("Excluded")
    _header(xs, EXCL_COLS)
    xs.append(["[0580-s25-42-q03c]", "test", "0580/42", "3(c)", "drawing_or_graph", "[draw the graph]"])
    for col, w in zip("ABCDEF", (22, 8, 10, 10, 24, 50)):
        xs.column_dimensions[col].width = w
    _dropdown(xs, EXCLUSION_REASONS, "E2:E2000")
    wb.save(path)
    return Path(path)


def _cell(v):
    return "" if v is None else str(v).strip()


def _is_example(d, key):
    return not d.get(key) or str(d[key]).startswith("[")


def _read_items(wb):
    rows, problems = [], []
    for n, r in enumerate(list(wb["Items"].iter_rows(values_only=True))[1:], 2):
        d = dict(zip(ITEM_COLS, r))
        if _is_example(d, "item_id") or _cell(d.get("question")).startswith("["):
            continue
        rows.append((n, d))
    return rows, problems


def stratified_sample(papers: dict, cap: int = CAP, seed: int = SEED) -> list:
    """Protocol §3: at most ``cap`` parts, allocated to papers in proportion to their size."""
    total = sum(len(v) for v in papers.values())
    if total <= cap:
        return sorted(i for v in papers.values() for i in v)
    quota = {p: cap * len(v) / total for p, v in papers.items()}
    take = {p: int(q) for p, q in quota.items()}
    for p in sorted(quota, key=lambda p: (-(quota[p] - take[p]), p))[:cap - sum(take.values())]:
        take[p] += 1  # largest remainders
    rng = random.Random(seed)
    out = []
    for p in sorted(papers):
        out += rng.sample(sorted(papers[p]), take[p])
    return sorted(out)


def sample(xlsx, folder, cap: int = CAP, seed: int = SEED):
    wb = load_workbook(xlsx, read_only=True)
    rows, _ = _read_items(wb)
    papers = defaultdict(list)
    for _, d in rows:
        if _cell(d["split"]) == "test":
            papers[_cell(d["paper"])].append(_cell(d["item_id"]))
    chosen = stratified_sample(papers, cap, seed)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "selected_items.txt").write_text("\n".join(chosen) + "\n")
    print(f"{len(chosen)} of {sum(len(v) for v in papers.values())} eligible test parts selected "
          f"(cap {cap}, seed {seed}) -> {folder / 'selected_items.txt'}")
    return chosen


def build(xlsx, folder):
    wb = load_workbook(xlsx, read_only=True)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sel_p = folder / "selected_items.txt"
    selected = set(sel_p.read_text().split()) if sel_p.exists() else None
    rows, problems = _read_items(wb)
    items = {}
    for n, d in rows:
        iid = _cell(d["item_id"])
        if selected is not None and _cell(d["split"]) == "test" and iid not in selected:
            continue
        try:
            spec = json.loads(d["spec_json"])
        except Exception:
            problems.append(f"Items row {n} ({iid}): spec_json is not valid JSON")
            continue
        items[iid] = Item(iid, _cell(d["split"]), _cell(d["paper"]), _cell(d["session"]), _cell(d["question_no"]),
                          _cell(d["topic"]), spec.get("type", ""), _cell(d["question"]),
                          _cell(d["mark_scheme_answer"]), spec, _cell(d["mark_scheme_notes"]), _cell(d["tag"]))
        if not items[iid].tag:
            problems.append(f"{iid}: tag (value_only / form_sensitive) is required, set from the mark scheme")
    responses = []
    for n, r in enumerate(list(wb["Responses"].iter_rows(values_only=True))[1:], 2):
        d = dict(zip(RESP_COLS, r))
        if _is_example(d, "response_id") or _cell(d.get("description")).startswith("["):
            continue
        if _cell(d["item_id"]) not in items and selected is not None:
            continue  # its part was not sampled
        responses.append(Response(_cell(d["response_id"]), _cell(d["item_id"]), _cell(d["answer"]),
                                  _cell(d["source"]), _cell(d["source_note"]), _cell(d["description"]),
                                  "author", _cell(d["intended_label"]).lower(),
                                  _cell(d["intended_category"]).lower()))
    excluded = []
    if "Excluded" in wb.sheetnames:
        for r in list(wb["Excluded"].iter_rows(values_only=True))[1:]:
            d = {k: _cell(v) for k, v in zip(EXCL_COLS, r)}
            if d["part_id"] and not d["part_id"].startswith("["):
                excluded.append(d)
    (folder / "items.jsonl").write_text("\n".join(json.dumps(i.__dict__, ensure_ascii=False) for i in items.values()) + "\n")
    (folder / "responses.jsonl").write_text("\n".join(json.dumps(r.__dict__, ensure_ascii=False) for r in responses) + "\n")
    with (folder / "excluded.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXCL_COLS)
        w.writeheader()
        w.writerows(excluded)
    from .data import load
    ds = load(folder)  # applies any classmate typing already collected
    problems += validate(ds)
    problems += self_check(items)
    per_item = defaultdict(int)
    for r in ds.responses:
        per_item[r.item_id] += 1
    for iid in items:
        if per_item[iid] and per_item[iid] != 5:
            problems.append(f"{iid}: has {per_item[iid]} answers; the protocol asks for 5 per part")
    print(f"{len(items)} items, {len(responses)} responses, {len(excluded)} excluded parts -> {folder}")
    for p in problems:
        print("PROBLEM:", p)
    return problems


def self_check(items: dict) -> list[str]:
    """Spec check from the mark scheme only: the grader must accept the mark-scheme answer.

    Only the mark-scheme answer is graded here, never a candidate answer, so the
    spec author learns nothing about how the grader treats the test responses.
    Parts with an ``unsupported`` spec are skipped (the grader always abstains on them).
    """
    out = []
    valid = {i: it for i, it in items.items() if it.spec.get("type") and not is_unsupported(it.spec)}
    if not valid:
        return out
    with GraderHandle() as g:
        for iid, it in valid.items():
            status, reason = g.grade(it.spec, it.mark_scheme_answer)
            if status != "correct":
                out.append(f"{iid}: the grader does not accept the mark-scheme answer "
                           f"{it.mark_scheme_answer!r} ({status}/{reason}) – check the spec")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["template", "sample", "build"])
    ap.add_argument("xlsx")
    ap.add_argument("folder", nargs="?")
    a = ap.parse_args()
    if a.cmd == "template":
        print(template(a.xlsx))
    elif a.cmd == "sample":
        sample(a.xlsx, a.folder)
    else:
        build(a.xlsx, a.folder)
