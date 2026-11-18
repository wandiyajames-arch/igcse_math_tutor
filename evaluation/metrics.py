"""Selective-prediction metrics, matched coverage, clustered bootstrap, McNemar.

Every system is turned into per-response arrays: a decision (correct /
incorrect / none) and a confidence. Rule systems have one confidence level;
LLM judges have the self-consistency share (0.6, 0.8, 1.0 for k = 5); the
cascade ranks our grader's decisions above the judge's.

Ties in confidence are handled by expectation (as if broken uniformly at
random), so matched-coverage numbers do not depend on an arbitrary order.
Only McNemar's test, which needs concrete pairs, uses a fixed tie-break seed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Optional

import numpy as np

from .categories import REASON_CATEGORY

from .judges import aggregate


@dataclass
class SystemArrays:
    name: str
    decided: np.ndarray   # bool (at the system's most permissive operating point)
    says_correct: np.ndarray  # bool, meaningful where decided
    conf: np.ndarray      # float, higher = more confident; -inf where not decided
    reason: list          # our reasons (or "")
    latency: np.ndarray   # ms, nan where unknown

    @property
    def is_curve(self) -> bool:
        return len(set(self.conf[self.decided].tolist())) > 1


def _load_run(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["response_id"]] = d
    return out


def arrays_from_run(name: str, run: dict, ids: list[str], min_conf: float = 0.0) -> SystemArrays:
    n = len(ids)
    decided = np.zeros(n, bool)
    says = np.zeros(n, bool)
    conf = np.full(n, -np.inf)
    reason = [""] * n
    lat = np.full(n, np.nan)
    for i, rid in enumerate(ids):
        d = run.get(rid)
        if d is None:
            raise KeyError(f"{name}: no result for {rid}")
        if d.get("latency_ms") is not None:
            lat[i] = d["latency_ms"]
        if "votes" in d:
            label, c = aggregate(d["votes"])
            if label != "abstain" and c >= min_conf:
                decided[i], says[i], conf[i] = True, label == "correct", c
        else:
            reason[i] = d.get("reason", "")
            if d["decision"] in ("correct", "incorrect"):
                decided[i], says[i], conf[i] = True, d["decision"] == "correct", 1.0
    return SystemArrays(name, decided, says, conf, reason, lat)


def cascade(ours: SystemArrays, judge: SystemArrays) -> SystemArrays:
    use_ours = ours.decided
    decided = ours.decided | judge.decided
    says = np.where(use_ours, ours.says_correct, judge.says_correct)
    conf = np.where(use_ours, 2.0, judge.conf)  # our decisions rank first
    lat = ours.latency + np.where(use_ours, 0.0, np.nan_to_num(judge.latency))
    return SystemArrays(f"cascade+{judge.name}", decided, says, conf, ours.reason, lat)


# ------------------------------------------------------------ core maths
def _groups(sysa: SystemArrays):
    levels = sorted(set(sysa.conf[sysa.decided].tolist()), reverse=True)
    g = np.full(len(sysa.conf), -1)
    for k, lv in enumerate(levels):
        g[(sysa.conf == lv) & sysa.decided] = k
    return g, len(levels)


def summarise(sysa: SystemArrays, gold_correct: np.ndarray, w: Optional[np.ndarray] = None,
              m_target: Optional[float] = None) -> dict:
    """Expected metrics when the system decides its m_target most confident responses
    (all it can decide if m_target is None). ``w`` are bootstrap weights."""
    n = len(gold_correct)
    w = np.ones(n) if w is None else w
    err = sysa.decided & (sysa.says_correct != gold_correct)
    fa = sysa.decided & sysa.says_correct & ~gold_correct
    fr = sysa.decided & ~sysa.says_correct & gold_correct
    g, ng = _groups(sysa)
    sel = g >= 0
    cnt = np.bincount(g[sel], weights=w[sel], minlength=ng)
    e = np.bincount(g[sel], weights=(w * err)[sel], minlength=ng)
    a = np.bincount(g[sel], weights=(w * fa)[sel], minlength=ng)
    r = np.bincount(g[sel], weights=(w * fr)[sel], minlength=ng)
    total = w.sum()
    n_inc = (w * ~gold_correct).sum()
    n_cor = (w * gold_correct).sum()
    m_max = cnt.sum()
    m = m_max if m_target is None else min(m_target, m_max)
    took = E = A = R = 0.0
    for k in range(ng):
        if took >= m - 1e-12:
            break
        take = min(cnt[k], m - took)
        frac = take / cnt[k] if cnt[k] else 0.0
        E += frac * e[k]
        A += frac * a[k]
        R += frac * r[k]
        took += take
    return {"coverage": took / total if total else 0.0,
            "selective_risk": E / took if took else float("nan"),
            "false_accept_rate": A / n_inc if n_inc else float("nan"),
            "false_reject_rate": R / n_cor if n_cor else float("nan"),
            "decided": took, "reached_target": m_target is None or m_max >= m_target - 1e-9}


def risk_coverage_curve(sysa: SystemArrays, gold_correct: np.ndarray, points: int = 200):
    """(coverage, expected risk) from 1 response up to the maximum coverage."""
    n = len(gold_correct)
    m_max = int(sysa.decided.sum())
    out = []
    for m in np.unique(np.linspace(1, m_max, min(points, m_max)).round().astype(int)) if m_max else []:
        s = summarise(sysa, gold_correct, m_target=float(m))
        out.append((m / n, s["selective_risk"]))
    return out


def aurc(sysa: SystemArrays, gold_correct: np.ndarray, w=None, up_to: Optional[float] = None) -> float:
    """Mean expected risk over coverage 1/N … up_to·N (lower is better)."""
    w = np.ones(len(gold_correct)) if w is None else w
    total = w.sum()
    m_max = float((w * sysa.decided).sum())
    m_end = m_max if up_to is None else min(m_max, up_to * total)
    if m_end < 1:
        return float("nan")
    ms = np.linspace(1, m_end, 50)
    return float(np.mean([summarise(sysa, gold_correct, w, m)["selective_risk"] for m in ms]))


# ------------------------------------------------------------- inference
def cluster_bootstrap(item_index: np.ndarray, B: int = 10000, seed: int = 2027):
    """Yield response weights for B resamples of question parts (items) with replacement."""
    rng = np.random.default_rng(seed)
    n_items = item_index.max() + 1
    for _ in range(B):
        draw = rng.integers(0, n_items, n_items)
        counts = np.bincount(draw, minlength=n_items)
        yield counts[item_index].astype(float)


def ci(values, level=0.95):
    v = np.asarray([x for x in values if x == x])
    if not len(v):
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(v, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def concrete_topm(sysa: SystemArrays, m: int, seed: int = 0) -> np.ndarray:
    """A concrete set of the m most confident decisions (ties broken by a fixed seed)."""
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(sysa.conf))
    order = np.lexsort((jitter, -np.where(sysa.decided, sysa.conf, -np.inf)))
    chosen = np.zeros(len(sysa.conf), bool)
    chosen[order[:m]] = True
    return chosen & sysa.decided


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial) p-value."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def paired_test(ours: SystemArrays, other: SystemArrays, gold_correct: np.ndarray, m: int, seed: int = 0):
    o = ours.decided
    s = concrete_topm(other, m, seed)
    both = o & s
    ok_o = ours.says_correct == gold_correct
    ok_s = other.says_correct == gold_correct
    b = int((both & ok_o & ~ok_s).sum())
    c = int((both & ~ok_o & ok_s).sum())
    return {"n_both": int(both.sum()), "ours_right_other_wrong": b, "ours_wrong_other_right": c,
            "p_value": mcnemar_exact(b, c)}


def reason_agreement(ours: SystemArrays, gold_labels: list[str], gold_cats: list[str], w=None):
    w = np.ones(len(gold_labels)) if w is None else w
    num = den = 0.0
    for i, (lab, cat) in enumerate(zip(gold_labels, gold_cats)):
        if lab == "incorrect" and ours.decided[i] and not ours.says_correct[i]:
            mine = REASON_CATEGORY.get(ours.reason[i])
            if mine is None:
                continue
            den += w[i]
            num += w[i] * (mine == cat)
    return num / den if den else float("nan"), den


# ------------------------------------------------------- rare-event bounds
def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Bin(n, p)."""
    from math import exp, lgamma, log
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    s = 0.0
    for i in range(0, k + 1):
        s += exp(lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * log(p) + (n - i) * log(1 - p))
    return min(1.0, s)


def clopper_pearson(k: int, n: int, level: float = 0.95):
    """Exact two-sided binomial CI for k events in n trials (no clustering)."""
    if n == 0:
        return (float("nan"), float("nan"))
    a = (1 - level) / 2

    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: 1 - _binom_cdf(k - 1, n, p) < a)
    upper = 1.0 if k == n else solve(lambda p: _binom_cdf(k, n, p) > a)
    return lower, upper


def rule_of_three(n: int) -> float:
    """Approximate 95% upper bound on a rate when 0 events were seen in n trials."""
    return 3.0 / n if n else float("nan")


def error_counts(sysa: SystemArrays, gold_correct: np.ndarray) -> dict:
    """Raw counts behind the false-accept and false-reject rates (whole decided set)."""
    fa = int((sysa.decided & sysa.says_correct & ~gold_correct).sum())
    fr = int((sysa.decided & ~sysa.says_correct & gold_correct).sum())
    n_inc, n_cor = int((~gold_correct).sum()), int(gold_correct.sum())
    return {"false_accepts": fa, "n_incorrect": n_inc, "false_rejects": fr, "n_correct": n_cor,
            "fa_exact_ci": clopper_pearson(fa, n_inc), "fr_exact_ci": clopper_pearson(fr, n_cor),
            "fa_rule_of_three": rule_of_three(n_inc) if fa == 0 else None,
            "fr_rule_of_three": rule_of_three(n_cor) if fr == 0 else None}


def risk_at_coverages(sysa: SystemArrays, gold_correct: np.ndarray, levels=(0.5, 0.7, 0.8, 0.9, 0.95)) -> dict:
    """Expected selective risk when the system decides a fixed share of all responses."""
    n = len(gold_correct)
    out = {}
    for c in levels:
        s = summarise(sysa, gold_correct, m_target=c * n)
        out[f"{c:.2f}"] = s["selective_risk"] if s["reached_target"] else None
    return out
