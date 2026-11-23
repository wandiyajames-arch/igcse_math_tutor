"""Build a PIPELINE-TEST-ONLY dataset from the grader's regression cases.

These cases were written by the developers while building the grader, so
any numbers computed on them are meaningless for the paper. The folder exists
only to check that every script runs end to end.

    python -m evaluation.pilot evaluation/data/pilot_PIPELINE_TEST_ONLY
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

from .categories import REASON_CATEGORY, status_of
from .grader_adapter import mark_scheme_answer, regression_cases

ROOT = Path(__file__).resolve().parents[1]


def build(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cases = regression_cases()
    items, responses, gold = [], [], []
    for c in cases:
        if not str(c["answer"]).strip():
            continue  # grader tests for empty input; a real answer is never empty
        iid = f"pilot-{c['id']}"
        items.append({"item_id": iid, "split": "dev", "paper": f"pilot/{len(items) % 4 + 1}", "session": "pipeline test",
                      "question_no": c["id"], "topic": "pilot", "answer_type": c["spec"]["type"],
                      "question": f"(Pipeline test) Give the answer. Marking basis: {c['basis']}",
                      "mark_scheme_answer": mark_scheme_answer(c["spec"]), "spec": c["spec"], "mark_scheme_notes": ""})
        rid = f"{iid}-r1"
        responses.append({"response_id": rid, "item_id": iid, "answer": c["answer"],
                          "source": "ms_form" if c["expect"] == "correct" else "examiner_report",
                          "source_note": "pipeline test", "description": "pipeline test",
                          "typed_by": "classmate" if len(responses) % 5 == 0 else "author",
                          "intended_label": "correct" if status_of(c["expect"]) == "correct" else "incorrect",
                          "intended_category": REASON_CATEGORY.get(c["expect"], "other")
                          if status_of(c["expect"]) != "correct" else ""})
        r = c["expect"]
        st = status_of(r)
        if st == "correct" or r == "equivalence_unconfirmed":
            gold.append((rid, "correct", ""))
        elif st == "incorrect":
            gold.append((rid, "incorrect", REASON_CATEGORY[r]))
        else:
            gold.append((rid, "undecidable", ""))
    (folder / "items.jsonl").write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n")
    (folder / "responses.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in responses) + "\n")
    with (folder / "excluded.csv").open("w", newline="") as f:  # pretend 1 in 5 answer lines are out of scope
        w = csv.writer(f)
        w.writerow(["part_id", "split", "paper", "question_no", "reason", "note"])
        for k in range(len(items) // 4):
            w.writerow([f"pilot-excluded-{k}", "dev", "pilot", str(k), "drawing_or_graph", "pipeline test"])
    rng = random.Random(1)
    for rater in ("rater1", "rater2"):  # simulated raters with 5% disagreement
        with (folder / f"labels_{rater}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["response_id", "label", "category", "comment"])
            for rid, lab, cat in gold:
                if rater == "rater2" and rng.random() < 0.05:
                    lab = "incorrect" if lab == "correct" else "correct"
                    cat = "value" if lab == "incorrect" else ""
                w.writerow([rid, lab, cat, ""])
    with (folder / "gold.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["response_id", "label", "category"])
        w.writerows(gold)
    (folder / "README.txt").write_text("PIPELINE TEST ONLY. Built from developer-written regression cases.\n"
                                       "Do not report any number from this folder.\n")
    print(f"wrote {folder}")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "evaluation/data/pilot_PIPELINE_TEST_ONLY")
