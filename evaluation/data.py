"""Test-set format, loading, rater agreement and adjudication.

Files (one folder per split, e.g. ``evaluation/data/test/``):

* ``items.jsonl``      one question part per line (see ``Item``)
* ``responses.jsonl``  one candidate answer per line (see ``Response``)
* ``labels_<rater>.csv`` one file per rater: response_id, label, category, comment
* ``gold.csv``         adjudicated labels (written by ``adjudicate``)

Gold labels: correct | incorrect | undecidable.
Reason categories (incorrect only): wrong_value, accuracy, form,
incomplete_extra, units_notation, other.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

LABELS = ("correct", "incorrect", "undecidable")
CATEGORIES = ("wrong_value", "accuracy", "form", "incomplete_extra", "units_notation", "other")
SOURCES = ("correct_form", "examiner_report", "real_student")


@dataclass
class Item:
    item_id: str                 # e.g. 0580-s25-42-q07b
    split: str                   # dev | test
    paper: str                   # e.g. 0580/42
    session: str                 # e.g. June 2025
    question_no: str             # e.g. 7(b)
    topic: str                   # syllabus topic (1-9 name)
    answer_type: str             # spec type: numeric, exact, expression, ...
    question: str                # question text as shown to judges
    mark_scheme_answer: str      # primary mark-scheme answer (given to every judge)
    spec: dict                   # AnswerSpec for our grader
    mark_scheme_notes: str = ""  # e.g. "oe", "awrt 78.5", accepted alternatives


@dataclass
class Response:
    response_id: str             # e.g. 0580-s25-42-q07b-r3
    item_id: str
    answer: str                  # candidate answer exactly as typed
    source: str                  # correct_form | examiner_report | real_student
    source_note: str = ""        # e.g. examiner-report sentence the error comes from


@dataclass
class Dataset:
    items: dict[str, Item]
    responses: list[Response]
    gold: dict[str, dict] = field(default_factory=dict)  # response_id -> {label, category}

    def item_of(self, r: Response) -> Item:
        return self.items[r.item_id]

    def gradable(self) -> list[Response]:
        """Responses with a correct/incorrect gold label (undecidable excluded)."""
        return [r for r in self.responses if self.gold.get(r.response_id, {}).get("label") in ("correct", "incorrect")]


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(ds: Dataset) -> list[str]:
    """Return a list of problems (empty = OK)."""
    from src.models.answers import parse_answer_spec
    errs = []
    for it in ds.items.values():
        try:
            parse_answer_spec(it.spec)
        except Exception as e:
            errs.append(f"{it.item_id}: invalid spec ({type(e).__name__}: {str(e)[:120]})")
        if it.split not in ("dev", "test"):
            errs.append(f"{it.item_id}: split must be dev or test")
    seen = set()
    for r in ds.responses:
        if r.response_id in seen:
            errs.append(f"{r.response_id}: duplicate response_id")
        seen.add(r.response_id)
        if r.item_id not in ds.items:
            errs.append(f"{r.response_id}: unknown item {r.item_id}")
        if r.source not in SOURCES:
            errs.append(f"{r.response_id}: source must be one of {SOURCES}")
        if r.source == "examiner_report" and not r.source_note:
            errs.append(f"{r.response_id}: examiner_report responses must cite the report (source_note)")
    for rid, g in ds.gold.items():
        if g["label"] not in LABELS:
            errs.append(f"{rid}: bad gold label {g['label']!r}")
        if g["label"] == "incorrect" and g.get("category") not in CATEGORIES:
            errs.append(f"{rid}: incorrect gold label needs a category")
    return errs


def load(folder) -> Dataset:
    folder = Path(folder)
    items = {d["item_id"]: Item(**d) for d in _read_jsonl(folder / "items.jsonl")}
    responses = [Response(**d) for d in _read_jsonl(folder / "responses.jsonl")]
    gold = {}
    if (folder / "gold.csv").exists():
        for row in csv.DictReader((folder / "gold.csv").open(encoding="utf-8")):
            gold[row["response_id"]] = {"label": row["label"].strip(), "category": (row.get("category") or "").strip()}
    return Dataset(items, responses, gold)


# ------------------------------------------------------------- raters
def read_labels(path) -> dict[str, dict]:
    out = {}
    for row in csv.DictReader(Path(path).open(encoding="utf-8-sig")):
        rid = row["response_id"].strip()
        out[rid] = {"label": (row.get("label") or "").strip().lower(),
                    "category": (row.get("category") or "").strip().lower(),
                    "comment": (row.get("comment") or "").strip()}
    return out


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two raters over the same items."""
    assert len(a) == len(b) and a
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement(r1: dict, r2: dict) -> dict:
    ids = sorted(set(r1) & set(r2))
    lab1 = [r1[i]["label"] for i in ids]
    lab2 = [r2[i]["label"] for i in ids]
    both_inc = [i for i in ids if r1[i]["label"] == r2[i]["label"] == "incorrect"]
    out = {"n": len(ids), "label_agreement": sum(x == y for x, y in zip(lab1, lab2)) / max(len(ids), 1),
           "label_kappa": cohen_kappa(lab1, lab2) if ids else None,
           "n_category": len(both_inc)}
    if both_inc:
        c1 = [r1[i]["category"] for i in both_inc]
        c2 = [r2[i]["category"] for i in both_inc]
        out["category_agreement"] = sum(x == y for x, y in zip(c1, c2)) / len(both_inc)
        out["category_kappa"] = cohen_kappa(c1, c2)
    return out


def disagreements(r1: dict, r2: dict) -> list[str]:
    return [i for i in sorted(set(r1) & set(r2))
            if r1[i]["label"] != r2[i]["label"]
            or (r1[i]["label"] == "incorrect" and r1[i]["category"] != r2[i]["category"])]


def adjudicate(r1: dict, r2: dict, decisions: Optional[dict] = None) -> dict[str, dict]:
    """Gold = agreed label; disagreements must be settled in ``decisions``."""
    decisions = decisions or {}
    gold, missing = {}, []
    for i in sorted(set(r1) & set(r2)):
        if i in decisions:
            gold[i] = decisions[i]
        elif i in disagreements(r1, r2):
            missing.append(i)
        else:
            gold[i] = {"label": r1[i]["label"], "category": r1[i]["category"] if r1[i]["label"] == "incorrect" else ""}
    if missing:
        raise ValueError(f"{len(missing)} disagreements need a decision, e.g. {missing[:5]}")
    return gold


def write_gold(gold: dict, path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["response_id", "label", "category"])
        for rid in sorted(gold):
            w.writerow([rid, gold[rid]["label"], gold[rid].get("category", "")])
