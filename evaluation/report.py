"""Compute every result in the protocol and write the paper tables and figure.

    python -m evaluation.report evaluation/data/test --headline-judge qwen3_4b

Writes to ``<folder>/results/``: results.json, results.md, tables.tex and
risk_coverage.png. Systems are read from ``<folder>/runs/*.jsonl``; a cascade
row is added for every LLM judge (ours first, the judge only on abstentions).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .data import agreement, load, read_labels
from .grader_adapter import is_unsupported, item_tag
from .metrics import (_load_run, arrays_from_run, at_mask, aurc, cascade, ci, cluster_bootstrap, error_counts,
                      intended_vs_gold, operating_point, paired_test, reason_agreement, reason_confusion,
                      risk_at_coverages, risk_coverage_curve, summarise)

NICE = {"ours": "Ours (deterministic)", "math_verify": "Math-Verify", "string_match": "String match", "qwen3_4b": "Qwen3-4B judge",
        "qwen25_7b": "Qwen2.5-7B judge", "compass_3b": "CompassVerifier-3B", "compass_7b": "CompassVerifier-7B"}


def nice(name: str) -> str:
    if name.startswith("cascade+"):
        return f"Cascade: ours → {nice(name[8:])}"
    return NICE.get(name, name)


RULE = ("ours", "math_verify", "string_match")


def build(folder, B: int = 10000, headline_judge: str | None = None, seed: int = 2027):
    folder = Path(folder)
    ds = load(folder)
    resp = ds.gradable()
    if not resp:
        raise SystemExit("no gold labels: run adjudication first (gold.csv)")
    ids = [r.response_id for r in resp]
    gold_correct = np.array([ds.gold[i]["label"] == "correct" for i in ids])
    gold_labels = [ds.gold[i]["label"] for i in ids]
    gold_cats = [ds.gold[i].get("category", "") for i in ids]
    item_ids = sorted({r.item_id for r in resp})
    item_index = np.array([item_ids.index(r.item_id) for r in resp])
    runs_dir = folder / "runs"
    systems = {}
    for p in sorted(runs_dir.glob("*.jsonl")):
        systems[p.stem] = arrays_from_run(p.stem, _load_run(p), ids)
    if "ours" not in systems:
        raise SystemExit("runs/ours.jsonl is required")
    ours = systems["ours"]
    judges = [k for k in systems if k not in RULE]
    for j in judges:
        c = cascade(ours, systems[j])
        systems[c.name] = c
    rules = [r for r in ("math_verify", "string_match") if r in systems]
    order = ["ours"] + [f"cascade+{j}" for j in judges] + rules + judges

    c_star = float(ours.decided.mean())
    res = {"n_responses": len(ids), "n_items": len(item_ids), "n_undecidable": sum(
        1 for r in ds.responses if ds.gold.get(r.response_id, {}).get("label") == "undecidable"),
        "our_coverage": c_star, "bootstrap_B": B, "systems": {}}

    # point estimates: each system at the threshold closest to our coverage without falling below it
    ours_op = operating_point(ours, gold_correct, c_star)
    for name in order:
        s = systems[name]
        op = operating_point(s, gold_correct, c_star)
        res["systems"][name] = {"label": nice(name), "max": summarise(s, gold_correct),
                                "matched": {k: v for k, v in op.items() if k != "mask"},
                                "aurc_to_our_coverage": aurc(s, gold_correct, up_to=c_star),
                                "aurc_full": aurc(s, gold_correct),
                                "counts": error_counts(s, gold_correct),
                                "risk_at_coverage": risk_at_coverages(s, gold_correct)}
        if name != "ours":
            res["systems"][name]["mcnemar_vs_ours"] = paired_test(ours, s, gold_correct, op["mask"])

    # every judge again at the headline cascade's coverage
    res["at_cascade_coverage"] = None
    if headline_judge and f"cascade+{headline_judge}" in systems:
        cc = float(systems[f"cascade+{headline_judge}"].decided.mean())
        res["at_cascade_coverage"] = {"coverage": cc, "systems": {}}
        for name in [f"cascade+{headline_judge}"] + judges:
            op = operating_point(systems[name], gold_correct, cc)
            res["at_cascade_coverage"]["systems"][name] = {k: v for k, v in op.items() if k != "mask"}

    # clustered bootstrap (question parts); the threshold is re-chosen in every resample
    keys = ("m_cov", "m_risk", "m_fa", "m_fr", "m_aoc", "diff_fa", "diff_risk", "max_cov", "max_risk", "aurc")
    boot = {n: {k: [] for k in keys} for n in order}
    ra = []
    for b, w in enumerate(cluster_bootstrap(item_index, B, seed)):
        c_b = float((w * ours.decided).sum() / w.sum())
        base = at_mask(ours.decided, ours, gold_correct, w)
        for name in order:
            s = systems[name]
            op = base if name == "ours" else operating_point(s, gold_correct, c_b, w)
            mx = summarise(s, gold_correct, w)
            d = boot[name]
            d["m_cov"].append(op["coverage"]); d["m_risk"].append(op["selective_risk"])
            d["m_fa"].append(op["false_accept_rate"]); d["m_fr"].append(op["false_reject_rate"])
            d["m_aoc"].append(op["abstain_on_correct"])
            d["diff_fa"].append(base["false_accept_rate"] - op["false_accept_rate"])
            d["diff_risk"].append(base["selective_risk"] - op["selective_risk"])
            d["max_cov"].append(mx["coverage"]); d["max_risk"].append(mx["selective_risk"])
            if b < min(B, 1000):
                d["aurc"].append(aurc(s, gold_correct, w, up_to=c_b))
        ra.append(reason_agreement(ours, gold_labels, gold_cats, w)[0])
    for name in order:
        res["systems"][name]["ci"] = {k: ci(v) for k, v in boot[name].items()}

    # RQ2
    val, n_ra = reason_agreement(ours, gold_labels, gold_cats)
    res["reason_agreement"] = {"value": val, "n": n_ra, "ci": ci(ra)}
    res["reason_confusion"] = reason_confusion(ours, gold_labels, gold_cats)

    # breakdowns: confirmatory (protocol §6) and exploratory
    uses_generic = any(r.source == "generic" for r in resp)
    groupings = [("by_stratum", "confirmatory", lambda r: item_tag(ds.item_of(r))),
                 ("by_paper", "confirmatory", lambda r: ds.item_of(r).paper or "?"),
                 ("by_typist", "confirmatory", lambda r: r.typed_by),
                 ("by_source", "confirmatory" if uses_generic else "exploratory", lambda r: r.source),
                 ("by_answer_type", "exploratory", lambda r: ds.item_of(r).answer_type),
                 ("by_topic", "exploratory", lambda r: ds.item_of(r).topic or "?")]
    res["breakdown_status"] = {}
    for key, status, getter in groupings:
        res[key] = {}
        res["breakdown_status"][key] = status
        for g in sorted({getter(r) for r in resp}):
            mask = np.array([getter(r) == g for r in resp], float)
            res[key][g] = {"n": int(mask.sum())}
            for name in order:
                s = summarise(systems[name], gold_correct, mask)
                res[key][g][name] = {k: s[k] for k in ("coverage", "selective_risk", "false_accept_rate",
                                                         "false_reject_rate", "abstain_on_correct")}

    # scope: coverage over included parts and over all answer-line parts
    n_incl = len(ds.items)
    n_excl = len(ds.excluded)
    share = n_incl / (n_incl + n_excl) if n_incl + n_excl else float("nan")
    n_unsup = sum(1 for it in ds.items.values() if is_unsupported(it.spec))
    res["scope"] = {"included_parts": n_incl, "excluded_parts": n_excl, "unsupported_parts": n_unsup,
                    "included_share": share,
                    "excluded_by_reason": dict(Counter(e.get("reason", "") for e in ds.excluded)),
                    "coverage_over_included": c_star,
                    "coverage_over_all_answer_lines": c_star * share if share == share else None}

    # Addition: intended vs gold label
    res["intended_vs_gold"] = intended_vs_gold(ds.responses, ds.gold)

    # error analysis: every wrong decision of ours
    res["our_errors"] = []
    for i, r in enumerate(resp):
        if ours.decided[i] and ours.says_correct[i] != gold_correct[i]:
            it = ds.item_of(r)
            res["our_errors"].append({"response_id": r.response_id, "item_id": it.item_id, "answer": r.answer,
                                      "mark_scheme_answer": it.mark_scheme_answer, "gold": gold_labels[i],
                                      "ours": "correct" if ours.says_correct[i] else "incorrect",
                                      "reason": ours.reason[i], "kind": "false_accept" if ours.says_correct[i]
                                      else "false_reject", "source": r.source, "typed_by": r.typed_by})

    # freeze check
    try:
        from .freeze import verify
        res["freeze"] = verify(folder, quiet=True)
    except Exception as e:
        res["freeze"] = {"ok": None, "problems": [f"not checked: {type(e).__name__}"]}

    # abstentions; the cascade on the grader's abstained subset (protocol §2, §10.4)
    abst = ~ours.decided
    res["our_abstentions"] = dict(Counter(ours.reason[i] for i in np.where(abst)[0]))
    res["cascade_on_abstained"] = {}
    if abst.any():
        for j in judges + rules:
            s = summarise(systems[j], gold_correct, abst.astype(float))
            res["cascade_on_abstained"][j] = {k: s[k] for k in ("coverage", "selective_risk", "false_accept_rate",
                                                                 "false_reject_rate")}
            res["cascade_on_abstained"][j]["n"] = int(abst.sum())

    # RQ3 efficiency table
    res["efficiency"] = {}
    for name in ["ours"] + rules + judges:
        lat = systems[name].latency
        lat = lat[~np.isnan(lat)]
        meta_p = runs_dir / f"{name}.meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        model = meta.get("model") or {}
        res["efficiency"][name] = {"median_ms": float(np.median(lat)) if len(lat) else None,
                                   "p95_ms": float(np.percentile(lat, 95)) if len(lat) else None,
                                   "cold_start_ms": meta.get("cold_start_ms"),
                                   "peak_mem_mb": meta.get("worker_peak_rss_mb"),
                                   "model_size_mb": (model.get("size_bytes") or 0) / 2 ** 20 or None,
                                   "digest": model.get("digest") or model.get("revision_sha"),
                                   "seeds": meta.get("seeds")}

    rater_files = sorted(folder.glob("labels_*.csv"))
    if len(rater_files) >= 2:
        res["rater_agreement"] = agreement(read_labels(rater_files[0]), read_labels(rater_files[1]))

    res["headline_cascade"] = f"cascade+{headline_judge}" if headline_judge else None
    lk = folder / "results" / "leakage.json"
    res["leakage"] = json.loads(lk.read_text()) if lk.exists() else None
    curves = {name: risk_coverage_curve(systems[name], gold_correct) for name in order}
    return res, curves, order


# ---------------------------------------------------------------- writers
def _pct(x):
    return "–" if x is None or x != x else f"{100 * x:.1f}"


def _pct_ci(x, c):
    return f"{_pct(x)} [{_pct(c[0])}, {_pct(c[1])}]"


def write_markdown(res, order, out: Path):
    fz = res.get("freeze", {})
    sc = res["scope"]
    c = res["our_coverage"]
    L = [f"# Results ({res['n_responses']} responses, {res['n_items']} question parts; "
         f"{res['n_undecidable']} 'Cannot decide' left out)", "",
         "Freeze check: " + ("**passed** (data, specs, grader and prompts match the published FREEZE files)"
                             if fz.get("ok") else "**NOT PASSED** – " + "; ".join(fz.get("problems", [])[:3])), "",
         f"Scope: {sc['included_parts']} included parts ({sc['unsupported_parts']} the grader cannot express) and "
         f"{sc['excluded_parts']} excluded answer lines. Our coverage: **{_pct(sc['coverage_over_included'])}%** of "
         f"answers on included parts, about {_pct(sc['coverage_over_all_answer_lines'])}% over all answer-line parts.",
         "", f"Brackets: 95% bootstrap CIs clustered by question part (B = {res['bootstrap_B']}).", "",
         f"## Table 1 – Matched coverage (primary outcome: false-accept rate)", "",
         f"Each judge runs at the threshold whose coverage is closest to our grader's ({_pct(c)}%) without falling "
         "below it; the coverage it actually reached is shown.", "",
         "| System | Coverage reached % | False accept % | FA diff (ours − system) | False reject % | "
         "Abstain on correct % | Selective risk % | McNemar p |",
         "|---|---|---|---|---|---|---|---|"]
    for n in order:
        s = res["systems"][n]
        m, ci_ = s["matched"], s["ci"]
        mc = s.get("mcnemar_vs_ours", {})
        flag = "" if m["reached_target"] else " †"
        head = "**" if n == res.get("headline_cascade") else ""
        ours_fa = res["systems"]["ours"]["matched"]["false_accept_rate"]
        diff = "" if n == "ours" else _pct_ci(ours_fa - m["false_accept_rate"], ci_["diff_fa"])
        p_txt = "" if not mc else "{:.3g}".format(mc["p_value"])
        L.append(f"| {head}{s['label']}{head}{flag} | {_pct_ci(m['coverage'], ci_['m_cov'])} | "
                 f"{_pct_ci(m['false_accept_rate'], ci_['m_fa'])} | {diff} | {_pct_ci(m['false_reject_rate'], ci_['m_fr'])} | "
                 f"{_pct_ci(m['abstain_on_correct'], ci_['m_aoc'])} | {_pct_ci(m['selective_risk'], ci_['m_risk'])} | {p_txt} |")
    L += ["", "† cannot reach our coverage; shown at its maximum. The primary test is the clustered bootstrap CI "
          "of the false-accept difference; exact McNemar (answers treated as independent) is secondary.", ""]
    acc = res.get("at_cascade_coverage")
    if acc:
        L += [f"## Judges at the headline cascade's coverage ({_pct(acc['coverage'])}%)", "",
              "| System | Coverage reached % | False accept % | False reject % | Selective risk % |", "|---|---|---|---|---|"]
        for n, m in acc["systems"].items():
            L.append(f"| {nice(n)} | {_pct(m['coverage'])} | {_pct(m['false_accept_rate'])} | "
                     f"{_pct(m['false_reject_rate'])} | {_pct(m['selective_risk'])} |")
        L.append("")
    L += ["## Error counts with exact 95% CIs (Clopper–Pearson, not clustered)", "",
          "| System | False accepts | FA exact CI % | False rejects | FR exact CI % |", "|---|---|---|---|---|"]
    for n in order:
        k = res["systems"][n]["counts"]
        fa_ci = f"{_pct(k['fa_exact_ci'][0])}–{_pct(k['fa_exact_ci'][1])}"
        fr_ci = f"{_pct(k['fr_exact_ci'][0])}–{_pct(k['fr_exact_ci'][1])}"
        if k["fa_rule_of_three"] is not None:
            fa_ci += f" (rule of three ≤ {_pct(k['fa_rule_of_three'])})"
        L.append(f"| {res['systems'][n]['label']} | {k['false_accepts']}/{k['n_incorrect']} | {fa_ci} | "
                 f"{k['false_rejects']}/{k['n_correct']} | {fr_ci} |")
    L += ["", "## Table 2 – Each system at its own maximum coverage", "",
          "| System | Coverage % | Selective risk % | False accept % | False reject % | AURC to our coverage % |",
          "|---|---|---|---|---|---|"]
    for n in order:
        s = res["systems"][n]
        f, ci_ = s["max"], s["ci"]
        L.append(f"| {s['label']} | {_pct_ci(f['coverage'], ci_['max_cov'])} | {_pct_ci(f['selective_risk'], ci_['max_risk'])} | "
                 f"{_pct(f['false_accept_rate'])} | {_pct(f['false_reject_rate'])} | {_pct_ci(s['aurc_to_our_coverage'], ci_['aurc'])} |")
    L += ["", "## The cascade on the answers our grader abstained on", "",
          "The cascade trades the tutor's design rule (an LLM never judges correctness) for coverage; this is its "
          "false-accept rate on that subset.", "",
          "| Judge used | n | Coverage % | False accept % | False reject % | Selective risk % |", "|---|---|---|---|---|---|"]
    for j, d in res["cascade_on_abstained"].items():
        L.append(f"| {nice(j)} | {d['n']} | {_pct(d['coverage'])} | {_pct(d['false_accept_rate'])} | "
                 f"{_pct(d['false_reject_rate'])} | {_pct(d['selective_risk'])} |")
    ra, rc = res["reason_agreement"], res["reason_confusion"]
    L += ["", f"## RQ2 – Reason agreement: {_pct_ci(ra['value'], ra['ci'])}% (n = {ra['n']:.0f}); "
          f"Cohen's κ = {rc['kappa']:.2f}", "",
          "Rows: gold category; columns: grader category.", "",
          "| | " + " | ".join(rc["categories"]) + " |", "|---|" + "---|" * len(rc["categories"])]
    for g in rc["categories"]:
        L.append(f"| {g} | " + " | ".join(str(rc["matrix"][g][x]) for x in rc["categories"]) + " |")
    L.append("")
    if "rater_agreement" in res:
        a = res["rater_agreement"]
        L += [f"Rater agreement (before adjudication): raw {_pct(a['label_agreement'])}%, label κ = {a['label_kappa']:.2f}, "
              f"Gwet's AC1 = {a['label_ac1']:.2f} (n = {a['n']}); category κ = "
              f"{a.get('category_kappa', float('nan')):.2f}, AC1 = {a.get('category_ac1', float('nan')):.2f} "
              f"(n = {a['n_category']}).", ""]
    ig = res["intended_vs_gold"]
    L += [f"## Intended vs gold label: {ig['differ']} of {ig['n']} answers ({_pct(ig['share'])}%) differ", "",
          "| Intended → gold | n |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(ig["table"].items())] + [""]
    titles = {"by_stratum": "Value-only vs form-sensitive parts", "by_paper": "By paper",
              "by_typist": "Author-typed vs classmate-typed answers", "by_source": "By answer source",
              "by_answer_type": "By answer type", "by_topic": "By syllabus topic"}
    for key, title in titles.items():
        status = res["breakdown_status"][key]
        L += [f"## {title} ({status}; coverage % / false accept % / false reject %)", "",
              "| Group | n | " + " | ".join(res["systems"][n]["label"] for n in order) + " |", "|---|---|" + "---|" * len(order)]
        for g, d in res[key].items():
            L.append(f"| {g} | {d['n']} | " + " | ".join(
                f"{_pct(d[n]['coverage'])} / {_pct(d[n]['false_accept_rate'])} / {_pct(d[n]['false_reject_rate'])}"
                for n in order) + " |")
        L.append("")
    lv = list(res["systems"][order[0]]["risk_at_coverage"])
    L += ["## Selective risk % at fixed coverage (exploratory)", "",
          "| System | " + " | ".join(f"{float(x) * 100:.0f}%" for x in lv) + " |", "|---|" + "---|" * len(lv)]
    for n in order:
        r = res["systems"][n]["risk_at_coverage"]
        L.append(f"| {res['systems'][n]['label']} | " + " | ".join(_pct(r[x]) for x in lv) + " |")
    L += ["", "## Our abstentions by reason", "", "| Reason | n |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(res["our_abstentions"].items(), key=lambda x: -x[1])]
    L += ["", "## RQ3 – Efficiency (same laptop)", "",
          "| System | Median ms/answer | p95 ms | Cold start ms | Peak memory MB | Model size MB | Model digest |",
          "|---|---|---|---|---|---|---|"]
    fm = lambda x: "–" if x is None else f"{x:.2f}" if x < 100 else f"{x:.0f}"
    for n, d in res["efficiency"].items():
        dg = (d.get("digest") or "–")[:19]
        L.append(f"| {nice(n)} | {fm(d['median_ms'])} | {fm(d['p95_ms'])} | {fm(d['cold_start_ms'])} | "
                 f"{fm(d['peak_mem_mb'])} | {fm(d['model_size_mb'])} | {dg} |")
    L += ["", f"## Our wrong decisions ({len(res['our_errors'])}) – see our_errors.csv", ""]
    lk = res.get("leakage")
    if lk:
        L += ["## RQ4 – Answer leakage in the first hint (main study)", "",
              "| Condition | Leaks | Rate % [95% Wilson CI] | Found only by the human check |", "|---|---|---|---|"]
        for cond, d in lk["conditions"].items():
            a = d["all"]
            L.append(f"| {cond} | {a['leaks']}/{a['n']} | {_pct_ci(a['rate'], a['ci95'])} | {d.get('leaks_only_human_found', '–')} |")
        if lk.get("exploratory"):
            L += ["", "Exploratory:", "", "| Condition | Leaks | Rate % [95% Wilson CI] |", "|---|---|---|"]
            for cond, d in lk["exploratory"].items():
                a = d["all"]
                L.append(f"| {cond} | {a['leaks']}/{a['n']} | {_pct_ci(a['rate'], a['ci95'])} |")
        if "auto_vs_manual" in lk:
            m = lk["auto_vs_manual"]
            L.append(f"\nAutomatic vs hand check: agreement {_pct(m['agreement'])}% on {m['n']} hints "
                     f"({m['auto_missed']} missed, {m['auto_false_alarm']} false alarms).")
    out.write_text("\n".join(L) + "\n")


def write_latex(res, order, out: Path):
    rows = []
    for n in order:
        s = res["systems"][n]
        m, c = s["matched"], s["ci"]
        p = s.get("mcnemar_vs_ours", {}).get("p_value")
        label = s["label"].replace("→", "$\\to$") + ("$^\\dagger$" if not m["reached_target"] else "")
        p_txt = "--" if p is None else "{:.3g}".format(p)
        rows.append(f"{label} & {_pct(m['coverage'])} & {_pct(m['false_accept_rate'])} "
                    f"\\scriptsize[{_pct(c['m_fa'][0])}, {_pct(c['m_fa'][1])}] & {_pct(m['false_reject_rate'])} & "
                    f"{_pct(m['abstain_on_correct'])} & {_pct(m['selective_risk'])} & {_pct(s['aurc_to_our_coverage'])} & {p_txt} \\\\")
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{Grading at matched coverage (\\%): each judge at the threshold "
           "closest to the grader's coverage without falling below it. 95\\% CIs clustered by question part. "
           "$^\\dagger$ cannot reach the grader's coverage.}\n"
           "\\label{tab:main}\n\\small\n\\begin{tabular}{lccccccc}\n\\hline\n"
           "System & Cov. & FA & FR & Abst.\\ corr. & Sel.\\ risk & AURC & McNemar $p$ \\\\\n\\hline\n" + "\n".join(rows) +
           "\n\\hline\n\\end{tabular}\n\\end{table}\n")
    out.write_text(tex)


def write_figure(curves, res, order, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    for n in order:
        pts = curves[n]
        if not pts:
            continue
        x, y = zip(*pts)
        if n in RULE:
            s = res["systems"][n]["max"]
            ax.plot([s["coverage"]], [s["selective_risk"]], "o", label=res["systems"][n]["label"], ms=7)
        else:
            ax.plot(x, y, label=res["systems"][n]["label"], lw=1.6)
    ax.axvline(res["our_coverage"], color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Coverage (share of answers decided)")
    ax.set_ylabel("Selective risk (share of decisions wrong)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--headline-judge", default=None, help="judge chosen on the dev set for the headline cascade")
    a = ap.parse_args(argv)
    res, curves, order = build(a.folder, a.B, a.headline_judge)
    out = Path(a.folder) / "results"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    write_markdown(res, order, out / "results.md")
    write_latex(res, order, out / "tables.tex")
    write_figure(curves, res, order, out / "risk_coverage.png")
    import csv
    with (out / "our_errors.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["response_id", "item_id", "answer", "mark_scheme_answer", "gold", "ours", "reason", "kind", "source",
                "typed_by"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(res["our_errors"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
