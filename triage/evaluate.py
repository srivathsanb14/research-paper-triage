"""Evaluation: predictions vs. hand labels (EVALUATED).

Honest-by-default:
* The learned model is evaluated with k-fold cross-validation — a paper's own
  hand label (and any feedback on it) is never used to score it.
* Baselines (keyword-only, semantic-only, untrained prior) are reported
  alongside, so we can tell whether learning from feedback actually helps.
* Threshold-free ranking metrics (Spearman ρ, NDCG@10) are reported next to
  label metrics, because label metrics depend on the human-set cutoffs.
* Suggested cutoffs are found by grid search on the same labels, so they are
  optimistic; the UI says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .ranking import Cutoffs, labels_for
from .relevance import BLEND_GRID

GAIN = {"READ": 2.0, "SKIM": 1.0, "SKIP": 0.0}


def label_metrics(truth: list[str], pred: list[str]) -> dict:
    labs = list(config.LABELS)
    idx = {l: i for i, l in enumerate(labs)}
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(truth, pred):
        cm[idx[t], idx[p]] += 1
    per_class = {}
    f1s = []
    for l, i in idx.items():
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[l] = {"precision": prec, "recall": rec, "f1": f1, "support": int(cm[i, :].sum())}
        if cm[i, :].sum():  # macro over classes present in the truth
            f1s.append(f1)
    n = max(1, len(truth))
    severe = int(cm[idx["READ"], idx["SKIP"]] + cm[idx["SKIP"], idx["READ"]])
    return {
        "accuracy": float(np.trace(cm) / n),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "within_one": float(1 - severe / n),
        "severe_errors": severe,
        "missed_reads": int(cm[idx["READ"], idx["SKIP"]]),
        "per_class": per_class,
        "confusion": cm.tolist(),
    }


def spearman(scores: np.ndarray, truth: list[str]) -> float:
    from scipy.stats import spearmanr

    y = [config.LABEL_VALUE[t] for t in truth]
    if len(set(y)) < 2 or np.ptp(scores) == 0:
        return float("nan")
    return float(spearmanr(scores, y).statistic)


def ndcg_at_k(scores: np.ndarray, truth: list[str], k: int = 10) -> float:
    gains = np.array([GAIN[t] for t in truth])
    order = np.argsort(-scores, kind="stable")[:k]
    disc = 1 / np.log2(np.arange(2, len(order) + 2))
    dcg = float((((2**gains[order]) - 1) * disc).sum())
    ideal = np.sort(gains)[::-1][:k]
    idcg = float((((2**ideal) - 1) * disc[: len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else float("nan")


def best_cutoffs(scores: np.ndarray, truth: list[str], step: float = 0.02) -> tuple[Cutoffs, dict]:
    """Grid-search (skim, read) cutoffs maximising macro-F1 (tie-break: fewer severe errors, accuracy).

    Vectorised: all (skim, read) pairs are scored at once via cumulative counts.
    """
    scores = np.asarray(scores, dtype=float)
    lo, hi = float(scores.min()), float(scores.max())
    grid = np.unique(np.round(np.concatenate([np.arange(0, 1 + step, step), np.quantile(scores, np.linspace(0, 1, 41))]), 4))
    grid = grid[(grid >= lo - step) & (grid <= hi + step)]
    if len(grid) == 0:
        grid = np.array([lo])
    y = np.array([{"READ": 0, "SKIM": 1, "SKIP": 2}[t] for t in truth])
    # ge[c, g] = #papers of class c with score >= grid[g]
    ge = np.stack([(scores[y == c][:, None] >= grid[None, :]).sum(axis=0) for c in range(3)])
    tot = np.array([(y == c).sum() for c in range(3)])
    S, R = np.meshgrid(np.arange(len(grid)), np.arange(len(grid)), indexing="ij")
    valid = R >= S
    # Predicted READ: score>=r; SKIM: s<=score<r; SKIP: score<s.
    pr = ge[:, R]  # (3, G, G) count of each true class predicted READ
    ps = ge[:, S] - ge[:, R]
    pk = tot[:, None, None] - ge[:, S]
    preds = [pr, ps, pk]
    f1s, present = [], tot > 0
    for c in range(3):
        tp = preds[c][c]
        pred_n = preds[c].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.where(pred_n > 0, tp / pred_n, 0.0)
            rec = tp / tot[c] if tot[c] else np.zeros_like(prec)
            f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        if present[c]:
            f1s.append(f1)
    macro = np.mean(f1s, axis=0)
    severe = pk[0] + pr[2]
    acc = (pr[0] + ps[1] + pk[2]) / max(1, len(y))
    m = np.where(valid, np.round(macro, 6), -1.0).ravel()
    # np.lexsort sorts by the last key first; take the maximum.
    order = np.lexsort((acc.ravel(), -severe.ravel(), m))
    si, ri = np.unravel_index(order[-1], macro.shape)
    best = Cutoffs(read=float(grid[ri]), skim=float(grid[si]))
    return best, label_metrics(truth, labels_for(scores, best))


def average_precision(scores: np.ndarray, truth: list[str], good: set[str]) -> float:
    is_good = np.array([t in good for t in truth], dtype=float)
    if is_good.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores), kind="stable")
    hits = is_good[order]
    prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((prec * hits).sum() / hits.sum())


def bootstrap_ci(stat, n: int, b: int = 300, seed: int = 0, level: float = 0.9) -> tuple[float, float] | None:
    """Percentile bootstrap over papers. `stat(idx)` computes the statistic on a resample."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        v = stat(rng.integers(0, n, n))
        if v is not None and not np.isnan(v):
            vals.append(v)
    if len(vals) < b // 3:
        return None
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def select_blend(prior_cv, learned_cv, truth, good: set[str], margin: float = 0.01) -> tuple[float, dict[float, float]]:
    """Pick how much the learned model should count, by cross-validated average precision.

    Learning is only switched on if it beats the prior by `margin`; ties go to the smaller weight.
    """
    if learned_cv is None:
        return 0.0, {}
    scores = {w: average_precision((1 - w) * prior_cv + w * learned_cv, truth, good) for w in BLEND_GRID}
    if any(np.isnan(v) for v in scores.values()):
        return 0.0, scores
    base = scores[0.0]
    best_w = 0.0
    for w in BLEND_GRID[1:]:
        if scores[w] > max(base + margin, scores[best_w] + 1e-9):
            best_w = w
    return best_w, scores


# ------------------------------------------------------------------------
# How quickly does the system surface good papers?
#
# Two views, both computed on hand-labelled papers with cross-validated scores:
#
# * Discovery curve — walk down the ranked list; after reviewing k papers, what
#   share of all the good papers have you found? Compared with a random order
#   (the diagonal) and a perfect ranking. Summarised as "papers to review to
#   find the first / half / 80% of the good papers".
# * Learning curve — replay your labels and feedback in the order you gave
#   them; after n signals, how good is the top of the ranking? Shows how much
#   feedback the model needs before its recommendations are useful.
# ------------------------------------------------------------------------


@dataclass
class DiscoveryCurve:
    name: str
    found: np.ndarray  # found[k-1] = share of good papers within the top k
    first_good: int | None  # papers reviewed until the first good one
    to_half: int | None
    to_80: int | None
    hit_rate_at_10: float  # good papers in the top 10 / min(10, #good)
    hit_rate_ci: tuple[float, float] | None = None
    to_half_ci: tuple[float, float] | None = None
    to_80_ci: tuple[float, float] | None = None


def _reviews_to(found: np.ndarray, share: float) -> int | None:
    idx = np.nonzero(found >= share - 1e-9)[0]
    return int(idx[0]) + 1 if len(idx) else None


def discovery_curve(name: str, scores: np.ndarray, truth: list[str], good: set[str], with_ci: bool = False) -> DiscoveryCurve:
    is_good = np.array([t in good for t in truth], dtype=float)
    n_good = is_good.sum()
    order = np.argsort(-np.asarray(scores), kind="stable")
    found = np.cumsum(is_good[order]) / max(n_good, 1)
    hits10 = is_good[order][:10].sum() / max(1.0, min(10.0, n_good))
    first = np.nonzero(is_good[order])[0]
    curve = DiscoveryCurve(
        name=name,
        found=found,
        first_good=int(first[0]) + 1 if len(first) else None,
        to_half=_reviews_to(found, 0.5),
        to_80=_reviews_to(found, 0.8),
        hit_rate_at_10=float(hits10),
    )
    if with_ci:
        sc, tr = np.asarray(scores), np.array(truth)

        def sub(i):
            t = list(tr[i])
            return discovery_curve("", sc[i], t, good) if any(x in good for x in t) else None

        n = len(truth)
        curve.hit_rate_ci = bootstrap_ci(lambda i: (c := sub(i)) and c.hit_rate_at_10, n)
        curve.to_half_ci = bootstrap_ci(lambda i: (c := sub(i)) and (c.to_half or np.nan), n)
        curve.to_80_ci = bootstrap_ci(lambda i: (c := sub(i)) and (c.to_80 or np.nan), n)
    return curve
