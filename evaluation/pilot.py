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

from src.grading.verdict import REASON_CATEGORY, REASON_STATUS, Reason, Status

ROOT = Path(__file__).resolve().parents[1]


def _ms_answer(spec: dict) -> str:
    t = spec["type"]
    if t == "ratio":
        return ":".join(str(p) for p in spec["parts"])
    if t == "solutions":
        return " or ".join(f"{spec['variable']} = {c['value']}" for c in spec["solutions"])
    if t in ("coordinates", "vector"):
        return "(" + ", ".join(c["value"] for c in spec["components"]) + ")"
    return spec["value"]


def build(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cases = json.loads((ROOT / "tests/grading_cases/cases.json").read_text())
    items, responses, gold = [], [], []
    for c in cases:
        iid = f"pilot-{c['id']}"
        items.append({"item_id": iid, "split": "dev", "paper": "pilot", "session": "pipeline test",
                      "question_no": c["id"], "topic": "pilot", "answer_type": c["spec"]["type"],
                      "question": f"(Pipeline test) Give the answer. Marking basis: {c['basis']}",
                      "mark_scheme_answer": _ms_answer(c["spec"]), "spec": c["spec"], "mark_scheme_notes": ""})
        rid = f"{iid}-r1"
        responses.append({"response_id": rid, "item_id": iid, "answer": c["answer"],
                          "source": "correct_form" if c["expect"] == "correct" else "examiner_report",
                          "source_note": "pipeline test"})
        r = Reason(c["expect"])
        st = REASON_STATUS[r]
        if st is Status.CORRECT or r is Reason.EQUIVALENCE_UNCONFIRMED:
            gold.append((rid, "correct", ""))
        elif st is Status.INCORRECT:
            gold.append((rid, "incorrect", REASON_CATEGORY[r]))
        else:
            gold.append((rid, "undecidable", ""))
    (folder / "items.jsonl").write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n")
    (folder / "responses.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in responses) + "\n")
    rng = random.Random(1)
    for rater in ("rater1", "rater2"):  # simulated raters with 5% disagreement
        with (folder / f"labels_{rater}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["response_id", "label", "category", "comment"])
            for rid, lab, cat in gold:
                if rater == "rater2" and rng.random() < 0.05:
                    lab = "incorrect" if lab == "correct" else "correct"
                    cat = "wrong_value" if lab == "incorrect" else ""
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
