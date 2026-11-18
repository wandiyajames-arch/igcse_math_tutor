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
from .grader_adapter import stratum
from .metrics import (_load_run, arrays_from_run, aurc, cascade, ci, cluster_bootstrap, error_counts,
                      paired_test, reason_agreement, risk_at_coverages, risk_coverage_curve, summarise)

NICE = {"ours": "Ours (deterministic)", "math_verify": "Math-Verify", "qwen3_4b": "Qwen3-4B judge",
        "qwen25_7b": "Qwen2.5-7B judge", "compass_3b": "CompassVerifier-3B", "compass_7b": "CompassVerifier-7B"}


def nice(name: str) -> str:
    if name.startswith("cascade+"):
        return f"Cascade: ours → {nice(name[8:])}"
    return NICE.get(name, name)


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
    judges = [k for k, v in systems.items() if k not in ("ours", "math_verify")]
    for j in judges:
        c = cascade(ours, systems[j])
        systems[c.name] = c
    order = ["ours"] + [f"cascade+{j}" for j in judges] + (["math_verify"] if "math_verify" in systems else []) + judges

    m_star = float(ours.decided.sum())
    c_star = m_star / len(ids)
    res = {"n_responses": len(ids), "n_items": len(item_ids), "n_undecidable": sum(
        1 for r in ds.responses if ds.gold.get(r.response_id, {}).get("label") == "undecidable"),
        "our_coverage": c_star, "bootstrap_B": B, "systems": {}}

    # point estimates
    for name in order:
        s = systems[name]
        full = summarise(s, gold_correct)
        matched = summarise(s, gold_correct, m_target=m_star)
        res["systems"][name] = {"label": nice(name), "max": full, "matched": matched,
                                "aurc_to_our_coverage": aurc(s, gold_correct, up_to=c_star),
                                "aurc_full": aurc(s, gold_correct),
                                "counts": error_counts(s, gold_correct),
                                "risk_at_coverage": risk_at_coverages(s, gold_correct)}
        if name != "ours":
            res["systems"][name]["mcnemar_vs_ours"] = paired_test(ours, s, gold_correct, int(m_star))

    # clustered bootstrap
    boot = {n: {"max_risk": [], "max_cov": [], "max_fa": [], "max_fr": [], "m_risk": [], "m_fa": [], "m_fr": [],
                "diff_risk": [], "aurc": []} for n in order}
    ra = []
    for b, w in enumerate(cluster_bootstrap(item_index, B, seed)):
        m_b = float((w * ours.decided).sum())
        base = summarise(ours, gold_correct, w)
        for name in order:
            s = systems[name]
            f = base if name == "ours" else summarise(s, gold_correct, w)
            mt = summarise(s, gold_correct, w, m_target=m_b)
            d = boot[name]
            d["max_risk"].append(f["selective_risk"]); d["max_cov"].append(f["coverage"])
            d["max_fa"].append(f["false_accept_rate"]); d["max_fr"].append(f["false_reject_rate"])
            d["m_risk"].append(mt["selective_risk"]); d["m_fa"].append(mt["false_accept_rate"])
            d["m_fr"].append(mt["false_reject_rate"])
            d["diff_risk"].append(base["selective_risk"] - mt["selective_risk"])
            if b < min(B, 1000):
                d["aurc"].append(aurc(s, gold_correct, w, up_to=c_star))
        ra.append(reason_agreement(ours, gold_labels, gold_cats, w)[0])
    for name in order:
        d = boot[name]
        res["systems"][name]["ci"] = {k: ci(v) for k, v in d.items()}

    # RQ2 reason agreement
    val, n_ra = reason_agreement(ours, gold_labels, gold_cats)
    res["reason_agreement"] = {"value": val, "n": n_ra, "ci": ci(ra)}

    # breakdowns
    res["by_stratum"], res["by_answer_type"], res["by_source"], res["by_topic"] = {}, {}, {}, {}
    for key, getter in (("by_stratum", lambda r: stratum(ds.item_of(r).spec)),
                        ("by_answer_type", lambda r: ds.item_of(r).answer_type),
                        ("by_source", lambda r: r.source),
                        ("by_topic", lambda r: ds.item_of(r).topic or "?")):
        groups = sorted({getter(r) for r in resp})
        for g in groups:
            mask = np.array([getter(r) == g for r in resp], float)
            res[key][g] = {"n": int(mask.sum())}
            for name in order:
                s = summarise(systems[name], gold_correct, mask)
                res[key][g][name] = {"coverage": s["coverage"], "selective_risk": s["selective_risk"],
                                     "false_accept_rate": s["false_accept_rate"],
                                     "false_reject_rate": s["false_reject_rate"]}

    # scope: two coverage denominators
    n_parts_spec = len({it_id for it_id, it in ds.items.items()})
    n_parts_excl = len(ds.excluded)
    share = n_parts_spec / (n_parts_spec + n_parts_excl) if n_parts_spec + n_parts_excl else float("nan")
    res["scope"] = {"specifiable_parts": n_parts_spec, "excluded_parts": n_parts_excl,
                    "specifiable_share": share,
                    "excluded_by_reason": dict(Counter(e.get("reason", "") for e in ds.excluded)),
                    "coverage_over_specifiable": c_star,
                    "coverage_over_all_answer_lines": c_star * share if share == share else None}

    # error analysis: every wrong decision of ours, for the paper's qualitative section
    res["our_errors"] = []
    for i, r in enumerate(resp):
        if ours.decided[i] and ours.says_correct[i] != gold_correct[i]:
            it = ds.item_of(r)
            res["our_errors"].append({"response_id": r.response_id, "item_id": it.item_id, "answer": r.answer,
                                      "mark_scheme_answer": it.mark_scheme_answer, "gold": gold_labels[i],
                                      "ours": "correct" if ours.says_correct[i] else "incorrect",
                                      "reason": ours.reason[i], "kind": "false_accept" if ours.says_correct[i]
                                      else "false_reject", "source": r.source})

    # freeze check
    try:
        from .freeze import verify
        res["freeze"] = verify(folder, quiet=True)
    except Exception as e:
        res["freeze"] = {"ok": None, "problems": [f"not checked: {type(e).__name__}"]}

    # abstentions
    abst = ~ours.decided
    res["our_abstentions"] = dict(Counter(ours.reason[i] for i in np.where(abst)[0]))
    res["judges_on_our_abstentions"] = {}
    if abst.any():
        for j in judges + (["math_verify"] if "math_verify" in systems else []):
            s = summarise(systems[j], gold_correct, abst.astype(float))
            res["judges_on_our_abstentions"][j] = {"coverage": s["coverage"], "selective_risk": s["selective_risk"],
                                                   "false_accept_rate": s["false_accept_rate"]}

    # efficiency
    res["efficiency"] = {}
    for name in ["ours", "math_verify"] + judges:
        if name in systems:
            lat = systems[name].latency
            lat = lat[~np.isnan(lat)]
            meta_p = runs_dir / f"{name}.meta.json"
            meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
            res["efficiency"][name] = {"median_ms": float(np.median(lat)) if len(lat) else None,
                                       "p95_ms": float(np.percentile(lat, 95)) if len(lat) else None,
                                       "peak_mem_mb": meta.get("worker_peak_rss_mb")}

    # raters
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
    L = [f"# Results ({res['n_responses']} responses, {res['n_items']} question parts; "
         f"{res['n_undecidable']} undecidable excluded)", "",
         "Freeze check: " + ("**passed** (data, specs, grader and prompts match FREEZE.json)" if fz.get("ok")
                             else "**NOT PASSED** – " + "; ".join(fz.get("problems", [])[:3])), "",
         f"Scope: {sc['specifiable_parts']} of {sc['specifiable_parts'] + sc['excluded_parts']} answer-line parts "
         f"could be specified ({_pct(sc['specifiable_share'])}%). Our coverage is {_pct(sc['coverage_over_specifiable'])}% "
         f"of responses on specifiable parts and about {_pct(sc['coverage_over_all_answer_lines'])}% of all answer lines.", "",
         f"Our grader's coverage: **{_pct(res['our_coverage'])}%**. Brackets: 95% bootstrap CIs "
         f"clustered by question part (B = {res['bootstrap_B']}).", "",
         "## Table 1 – Matched coverage (every system decides as many responses as ours)", "",
         "| System | Coverage % | Selective risk % | False accept % | False reject % | Risk diff (ours − system) | McNemar p |",
         "|---|---|---|---|---|---|---|"]
    for n in order:
        s = res["systems"][n]
        m, c = s["matched"], s["ci"]
        mc = s.get("mcnemar_vs_ours", {})
        flag = "" if m["reached_target"] else " †"
        head = "**" if n == res.get("headline_cascade") else ""
        diff = res["systems"]["ours"]["max"]["selective_risk"] - m["selective_risk"]
        diff_txt = "" if n == "ours" else _pct_ci(diff, c["diff_risk"])
        p_txt = "" if not mc else "{:.3g}".format(mc["p_value"])
        L.append(f"| {head}{s['label']}{head}{flag} | {_pct(m['coverage'])} | {_pct_ci(m['selective_risk'], c['m_risk'])} | "
                 f"{_pct_ci(m['false_accept_rate'], c['m_fa'])} | {_pct_ci(m['false_reject_rate'], c['m_fr'])} | "
                 f"{diff_txt} | {p_txt} |")
    L += ["", "† cannot reach our coverage; shown at its maximum coverage. Primary test: the paired, clustered "
          "bootstrap CI of the risk difference; McNemar (which treats responses as independent) is secondary.", "",
          "## Error counts with exact 95% CIs (Clopper–Pearson, not clustered)", "",
          "| System | False accepts | FA exact CI % | False rejects | FR exact CI % |", "|---|---|---|---|---|"]
    for n in order:
        k = res["systems"][n]["counts"]
        fa_ci = f"{_pct(k['fa_exact_ci'][0])}–{_pct(k['fa_exact_ci'][1])}"
        fr_ci = f"{_pct(k['fr_exact_ci'][0])}–{_pct(k['fr_exact_ci'][1])}"
        if k["fa_rule_of_three"] is not None:
            fa_ci += f" (rule of three ≤ {_pct(k['fa_rule_of_three'])})"
        L.append(f"| {res['systems'][n]['label']} | {k['false_accepts']}/{k['n_incorrect']} | {fa_ci} | "
                 f"{k['false_rejects']}/{k['n_correct']} | {fr_ci} |")
    levels = list(res["systems"][order[0]]["risk_at_coverage"])
    L += ["", "## Selective risk % at fixed coverage", "",
          "| System | " + " | ".join(f"{float(c) * 100:.0f}%" for c in levels) + " |", "|---|" + "---|" * len(levels)]
    for n in order:
        rc = res["systems"][n]["risk_at_coverage"]
        L.append(f"| {res['systems'][n]['label']} | " + " | ".join(_pct(rc[c]) for c in levels) + " |")
    L += ["",
          "## Table 2 – Each system at its own maximum coverage", "",
          "| System | Coverage % | Selective risk % | False accept % | False reject % | AURC to our coverage % |",
          "|---|---|---|---|---|---|"]
    for n in order:
        s = res["systems"][n]
        f, c = s["max"], s["ci"]
        L.append(f"| {s['label']} | {_pct_ci(f['coverage'], c['max_cov'])} | {_pct_ci(f['selective_risk'], c['max_risk'])} | "
                 f"{_pct_ci(f['false_accept_rate'], c['max_fa'])} | {_pct_ci(f['false_reject_rate'], c['max_fr'])} | "
                 f"{_pct_ci(s['aurc_to_our_coverage'], c['aurc'])} |")
    ra = res["reason_agreement"]
    L += ["", f"## RQ2 – Reason agreement: {_pct_ci(ra['value'], ra['ci'])}% (n = {ra['n']:.0f})", ""]
    if "rater_agreement" in res:
        a = res["rater_agreement"]
        L += [f"Rater agreement: raw {_pct(a['label_agreement'])}%, label κ = {a['label_kappa']:.2f}, "
              f"Gwet's AC1 = {a['label_ac1']:.2f} (n = {a['n']}); category κ = "
              f"{a.get('category_kappa', float('nan')):.2f}, AC1 = {a.get('category_ac1', float('nan')):.2f} "
              f"(n = {a['n_category']}).", ""]
    for key, title in (("by_stratum", "Value-only vs form-sensitive items"), ("by_answer_type", "By answer type"),
                       ("by_source", "By response source"), ("by_topic", "By syllabus topic")):
        L += [f"## {title} (coverage % / selective risk %)", "", "| Group | n | " + " | ".join(res["systems"][n]["label"] for n in order) + " |",
              "|---|---|" + "---|" * len(order)]
        for g, d in res[key].items():
            L.append(f"| {g} | {d['n']} | " + " | ".join(f"{_pct(d[n]['coverage'])} / {_pct(d[n]['selective_risk'])}" for n in order) + " |")
        L.append("")
    L += ["## Our abstentions by reason", "", "| Reason | n |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(res["our_abstentions"].items(), key=lambda x: -x[1])]
    L += ["", "## Other systems on the responses our grader abstained on", "",
          "| System | Coverage % | Selective risk % | False accept % |", "|---|---|---|---|"]
    for j, d in res["judges_on_our_abstentions"].items():
        L.append(f"| {nice(j)} | {_pct(d['coverage'])} | {_pct(d['selective_risk'])} | {_pct(d['false_accept_rate'])} |")
    L += ["", "## Efficiency", "", "| System | Median ms | p95 ms | Peak memory MB |", "|---|---|---|---|"]
    for n, d in res["efficiency"].items():
        f = lambda x: "–" if x is None else f"{x:.2f}" if x < 100 else f"{x:.0f}"
        L.append(f"| {nice(n)} | {f(d['median_ms'])} | {f(d['p95_ms'])} | {f(d['peak_mem_mb'])} |")
    L += ["", f"## Our wrong decisions ({len(res['our_errors'])}) – see our_errors.csv", ""]
    lk = res.get("leakage")
    if lk:
        L += ["## RQ4 – Answer leakage before the reveal (share of tutor replies that state the answer)", "",
              "| Condition | Leaks | Rate % [95% Wilson CI] |", "|---|---|---|"]
        for cond, d in lk["conditions"].items():
            a = d["all"]
            L.append(f"| {cond} | {a['leaks']}/{a['n']} | {_pct_ci(a['rate'], a['ci95'])} |")
        if "auto_vs_manual" in lk:
            m = lk["auto_vs_manual"]
            L.append(f"\nAutomatic vs hand check: agreement {_pct(m['agreement'])}% on {m['n']} replies "
                     f"({m['auto_missed']} missed, {m['auto_false_alarm']} false alarms).")
    out.write_text("\n".join(L) + "\n")


def write_latex(res, order, out: Path):
    rows = []
    for n in order:
        s = res["systems"][n]
        m, c = s["matched"], s["ci"]
        p = s.get("mcnemar_vs_ours", {}).get("p_value")
        label = s["label"].replace("→", "$\\to$")
        p_txt = "--" if p is None else "{:.3g}".format(p)
        rows.append(f"{label} & {_pct(m['coverage'])} & {_pct(m['selective_risk'])} "
                    f"\\scriptsize[{_pct(c['m_risk'][0])}, {_pct(c['m_risk'][1])}] & {_pct(m['false_accept_rate'])} & "
                    f"{_pct(m['false_reject_rate'])} & {_pct(s['aurc_to_our_coverage'])} & {p_txt} \\\\")
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{Grading quality at matched coverage (\\%), 95\\% CIs clustered by question part.}\n"
           "\\label{tab:main}\n\\small\n\\begin{tabular}{lcccccc}\n\\hline\n"
           "System & Cov. & Sel. risk & FA & FR & AURC & McNemar $p$ \\\\\n\\hline\n" + "\n".join(rows) +
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
        if n in ("ours", "math_verify"):
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
        cols = ["response_id", "item_id", "answer", "mark_scheme_answer", "gold", "ours", "reason", "kind", "source"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(res["our_errors"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
