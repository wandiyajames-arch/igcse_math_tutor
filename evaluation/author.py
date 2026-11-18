"""Author the test set in Excel, then convert it to the harness format.

    python -m evaluation.author template evaluation/data/test/authoring.xlsx
    (fill in the Items and Responses sheets)
    python -m evaluation.author build evaluation/data/test/authoring.xlsx evaluation/data/test

``build`` writes items.jsonl and responses.jsonl and lists every problem
(invalid spec, unknown item, missing examiner-report citation) so nothing
broken reaches the raters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from .data import EXCLUSION_REASONS, SOURCES, Dataset, Item, Response, validate
from .grader_adapter import GraderHandle, spec_examples

ITEM_COLS = ["item_id", "split", "paper", "session", "question_no", "topic", "question",
             "mark_scheme_answer", "mark_scheme_notes", "spec_json"]
RESP_COLS = ["response_id", "item_id", "answer", "source", "source_note"]
EXCL_COLS = ["part_id", "split", "paper", "question_no", "reason", "note"]
TOPICS = ["Number", "Algebra and graphs", "Coordinate geometry", "Geometry", "Mensuration",
          "Trigonometry", "Transformations and vectors", "Probability", "Statistics"]



def _header(ws, cols):
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="305496")
    ws.freeze_panes = "A2"


def template(path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Items"
    _header(ws, ITEM_COLS)
    ws.append(["0580-s25-22-q05", "test", "0580/22", "June 2025", "5", "Algebra and graphs",
               "[Type the question text here]", "[mark-scheme answer]", "[e.g. oe, awrt]",
               '[spec JSON — copy a pattern from the "Spec examples" sheet]'])
    for col, w in zip("ABCDEFGHIJ", (18, 8, 10, 12, 10, 22, 60, 22, 18, 70)):
        ws.column_dimensions[col].width = w
    dv = DataValidation(type="list", formula1='"dev,test"', allow_blank=True)
    dv2 = DataValidation(type="list", formula1='"' + ",".join(TOPICS) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    ws.add_data_validation(dv2)
    dv.add("B2:B1000")
    dv2.add("F2:F1000")
    rs = wb.create_sheet("Responses")
    _header(rs, RESP_COLS)
    rs.append(["0580-s25-22-q05-r1", "0580-s25-22-q05", "[candidate answer exactly as typed]", "correct_form", ""])
    rs.append(["0580-s25-22-q05-r2", "0580-s25-22-q05", "[typical error]", "examiner_report",
               "[quote or page of the Principal Examiner Report describing this error]"])
    for col, w in zip("ABCDE", (22, 18, 30, 18, 70)):
        rs.column_dimensions[col].width = w
    dv3 = DataValidation(type="list", formula1='"' + ",".join(SOURCES) + '"', allow_blank=True)
    rs.add_data_validation(dv3)
    dv3.add("D2:D5000")
    ex = wb.create_sheet("Spec examples")
    _header(ex, ["answer type", "spec_json pattern"])
    for row in spec_examples():
        ex.append(list(row))
    ex.column_dimensions["A"].width = 28
    ex.column_dimensions["B"].width = 120
    xs = wb.create_sheet("Excluded")
    _header(xs, EXCL_COLS)
    xs.append(["0580-s25-42-q03c", "test", "0580/42", "3(c)", "graph_or_diagram", "draw the graph"])
    for col, w in zip("ABCDEF", (22, 8, 10, 10, 22, 50)):
        xs.column_dimensions[col].width = w
    dv4 = DataValidation(type="list", formula1='"' + ",".join(EXCLUSION_REASONS) + '"', allow_blank=True)
    xs.add_data_validation(dv4)
    dv4.add("E2:E2000")
    wb.save(path)
    return Path(path)


def build(xlsx, folder):
    wb = load_workbook(xlsx, read_only=True)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    items, problems = {}, []
    rows = list(wb["Items"].iter_rows(values_only=True))
    for n, r in enumerate(rows[1:], 2):
        d = dict(zip(ITEM_COLS, r))
        if not d.get("item_id") or str(d["item_id"]).startswith("[") or str(d.get("question") or "").startswith("["):
            continue
        try:
            spec = json.loads(d["spec_json"])
        except Exception:
            problems.append(f"Items row {n} ({d['item_id']}): spec_json is not valid JSON")
            continue
        it = Item(str(d["item_id"]).strip(), str(d["split"] or "").strip(), str(d["paper"] or ""),
                  str(d["session"] or ""), str(d["question_no"] or ""), str(d["topic"] or ""),
                  spec.get("type", ""), str(d["question"] or ""), str(d["mark_scheme_answer"] or ""),
                  spec, str(d["mark_scheme_notes"] or ""))
        items[it.item_id] = it
    responses = []
    for n, r in enumerate(list(wb["Responses"].iter_rows(values_only=True))[1:], 2):
        d = dict(zip(RESP_COLS, r))
        if not d.get("response_id") or str(d.get("answer") or "").startswith("["):
            continue
        responses.append(Response(str(d["response_id"]).strip(), str(d["item_id"]).strip(),
                                  str(d["answer"]), str(d["source"] or "").strip(), str(d["source_note"] or "")))
    excluded = []
    if "Excluded" in wb.sheetnames:
        for r in list(wb["Excluded"].iter_rows(values_only=True))[1:]:
            d = {k: ("" if v is None else str(v).strip()) for k, v in zip(EXCL_COLS, r)}
            if d["part_id"] and d["note"] != "draw the graph":
                excluded.append(d)
    problems += validate(Dataset(items, responses, excluded=excluded))
    problems += self_check(items)
    import csv
    with (folder / "excluded.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXCL_COLS)
        w.writeheader()
        w.writerows(excluded)
    (folder / "items.jsonl").write_text("\n".join(json.dumps(i.__dict__, ensure_ascii=False) for i in items.values()) + "\n")
    (folder / "responses.jsonl").write_text("\n".join(json.dumps(r.__dict__, ensure_ascii=False) for r in responses) + "\n")
    print(f"{len(items)} items, {len(responses)} responses, {len(excluded)} excluded parts -> {folder}")
    for p in problems:
        print("PROBLEM:", p)
    return problems


def self_check(items: dict) -> list[str]:
    """Spec check from the mark scheme only: the grader must accept the mark-scheme answer.

    Only the mark-scheme answer is graded here, never a candidate answer, so the
    spec author learns nothing about how the grader treats the test responses.
    A failure usually means the spec or the mark-scheme answer string is wrong.
    """
    out = []
    valid = {i: it for i, it in items.items() if it.spec.get("type")}
    if not valid:
        return out
    with GraderHandle() as g:
        for iid, it in valid.items():
            status, reason = g.grade(it.spec, it.mark_scheme_answer)
            if status != "correct":
                out.append(f"{iid}: the grader does not accept the mark-scheme answer "
                           f"{it.mark_scheme_answer!r} ({status}/{reason}) – check the spec or write the "
                           f"answer as a candidate would type it")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["template", "build"])
    ap.add_argument("xlsx")
    ap.add_argument("folder", nargs="?")
    a = ap.parse_args()
    print(template(a.xlsx)) if a.cmd == "template" else build(a.xlsx, a.folder)
