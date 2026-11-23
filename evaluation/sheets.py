"""Labelling sheets for raters (Excel, with dropdowns).

Raters see the question, the mark-scheme answer and notes, and the candidate
answer, in a shuffled order. They never see the answer's source or any
system's verdict.

    python -m evaluation.sheets evaluation/data/test --rater rater1
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from .data import CATEGORIES, LABELS, load

GUIDE = [
    "Labelling guide (read before starting)",
    "For each row decide whether the final answer would earn the final-answer mark under the mark scheme.",
    "label: correct | incorrect | undecidable (= Cannot decide; add a one-line note in comment).",
    "Method marks are ignored: would this answer, written on the answer line, earn the final-answer mark?",
    "category (only when label = incorrect):",
    "  value – the value is wrong.",
    "  accuracy – right value, wrong precision or rounding (s.f., d.p., rounded an exact value).",
    "  form – right value, not in the required form (not simplified, not expanded, not fully factorised, wrong form).",
    "  units – a unit is wrong or missing where one is required.",
    "  solution_set – a solution is missing, or an extra one is given.",
    "  other – anything else; explain in the comment column.",
    "Work alone. Do not discuss items with the other rater until both sheets are finished.",
]


def make_sheet(folder, rater: str, out=None, seed: int = 2027) -> Path:
    ds = load(folder)
    rows = list(ds.responses)
    random.Random(f"{seed}-{rater}").shuffle(rows)
    wb = Workbook()
    g = wb.active
    g.title = "Guide"
    for i, line in enumerate(GUIDE, 1):
        g.cell(i, 1, line).font = Font(bold=(i == 1))
    g.column_dimensions["A"].width = 120
    ws = wb.create_sheet("Labels")
    head = ["response_id", "question", "mark_scheme_answer", "mark_scheme_notes", "candidate_answer",
            "label", "category", "comment"]
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="305496")
    for r in rows:
        it = ds.item_of(r)
        ws.append([r.response_id, it.question, it.mark_scheme_answer, it.mark_scheme_notes, r.answer, "", "", ""])
    n = len(rows) + 1
    dv1 = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True)
    dv2 = DataValidation(type="list", formula1='"' + ",".join(CATEGORIES) + '"', allow_blank=True)
    ws.add_data_validation(dv1)
    ws.add_data_validation(dv2)
    dv1.add(f"F2:F{n}")
    dv2.add(f"G2:G{n}")
    for col, w in zip("ABCDEFGH", (24, 60, 22, 26, 22, 14, 18, 30)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "B2"
    out = Path(out or Path(folder) / f"labelling_{rater}.xlsx")
    wb.save(out)
    return out


def sheet_to_csv(xlsx, csv_out) -> Path:
    """Export a completed sheet to labels_<rater>.csv for ``data.read_labels``."""
    import csv
    from openpyxl import load_workbook
    ws = load_workbook(xlsx, read_only=True)["Labels"]
    rows = list(ws.iter_rows(values_only=True))
    head = list(rows[0])
    with Path(csv_out).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["response_id", "label", "category", "comment"])
        for r in rows[1:]:
            d = dict(zip(head, r))
            w.writerow([d["response_id"], d["label"] or "", d["category"] or "", d["comment"] or ""])
    return Path(csv_out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--rater", required=True)
    a = ap.parse_args()
    print(make_sheet(a.folder, a.rater))
