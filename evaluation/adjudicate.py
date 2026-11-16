"""Compare the two raters, list disagreements, and write gold.csv.

    python -m evaluation.adjudicate evaluation/data/test
        -> prints κ and writes disagreements.csv (fill decided_label / decided_category)
    python -m evaluation.adjudicate evaluation/data/test --decisions evaluation/data/test/disagreements.csv
        -> writes gold.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .data import adjudicate, agreement, disagreements, read_labels, write_gold


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--decisions", default=None)
    a = ap.parse_args(argv)
    folder = Path(a.folder)
    files = sorted(folder.glob("labels_*.csv"))
    if len(files) != 2:
        raise SystemExit(f"need exactly two labels_<rater>.csv files, found {[f.name for f in files]}")
    r1, r2 = read_labels(files[0]), read_labels(files[1])
    print(json.dumps(agreement(r1, r2), indent=1))
    dis = disagreements(r1, r2)
    if a.decisions is None:
        with (folder / "disagreements.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["response_id", f"{files[0].stem}_label", f"{files[0].stem}_category",
                        f"{files[1].stem}_label", f"{files[1].stem}_category", "decided_label", "decided_category"])
            for i in dis:
                w.writerow([i, r1[i]["label"], r1[i]["category"], r2[i]["label"], r2[i]["category"], "", ""])
        print(f"{len(dis)} disagreements -> {folder / 'disagreements.csv'}")
        if not dis:
            write_gold(adjudicate(r1, r2), folder / "gold.csv")
            print(f"no disagreements: wrote {folder / 'gold.csv'}")
        return
    decisions = {}
    for row in csv.DictReader(Path(a.decisions).open(encoding="utf-8")):
        if row.get("decided_label"):
            decisions[row["response_id"]] = {"label": row["decided_label"].strip(),
                                             "category": (row.get("decided_category") or "").strip()}
    write_gold(adjudicate(r1, r2, decisions), folder / "gold.csv")
    print(f"wrote {folder / 'gold.csv'}")


if __name__ == "__main__":
    main()
