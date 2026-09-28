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

from dataclasses import dataclass, field

import numpy as np

from . import config
from .models import Paper
from .ranking import Cutoffs, labels_for
from .relevance import BLEND_GRID, FEATURES, RelevanceModel, build_examples

GAIN = {"READ": 2.0, "SKIM": 1.0, "SKIP": 0.0}
MIN_LABELS = 6


@dataclass
class SystemReport:
    name: str
    scores: np.ndarray
    spearman: float
    ndcg10: float
    at_current: dict | None = None  # label metrics at the user's cutoffs
    best_cutoffs: Cutoffs | None = None
    at_best: dict | None = None
    spearman_ci: tuple[float, float] | None = None  # 90% bootstrap interval
    ndcg10_ci: tuple[float, float] | None = None


@dataclass
class EvalReport:
    n_labels: int
    label_counts: dict[str, int]
    systems: list[SystemReport] = field(default_factory=list)
    paper_ids: list[str] = field(default_factory=list)
    truth: list[str] = field(default_factory=list)
    cv_predictions: list[str] = field(default_factory=list)
    folds: int = 0
    message: str = ""
    blend: dict = field(default_factory=dict)  # {"weight", "scores": {w: AP}, "metric"}

    @property
    def ok(self) -> bool:
        return bool(self.systems)


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


def _folds(truth: list[str], k: int, seed: int = 0) -> list[np.ndarray]:
    """Stratified-ish k folds (round-robin within each class after shuffling)."""
    rng = np.random.default_rng(seed)
    fold_of = np.zeros(len(truth), dtype=int)
    for lab in config.LABELS:
        idx = np.array([i for i, t in enumerate(truth) if t == lab])
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold_of[i] = j % k
    return [np.where(fold_of == f)[0] for f in range(k)]


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


def cv_components(model_factory, ids, truth, papers, V, labels, feedback, papers_by_id, vecs_by_id, k, seeds=None):
    """Cross-validated (prior, learned) scores: each fold's model never sees that fold's labels or feedback."""
    prior_cv, learned_cv, any_learned = np.zeros(len(ids)), np.zeros(len(ids)), False
    for test_idx in _folds(truth, k):
        if len(test_idx) == 0:
            continue
        test_ids = {ids[i] for i in test_idx}
        train_labels = {pid: lab for pid, lab in labels.items() if pid not in test_ids}
        m = model_factory().fit(
            build_examples(train_labels, feedback, exclude=test_ids, seeds=seeds), papers_by_id, vecs_by_id
        )
        pr, le, _ = m.components([papers[i] for i in test_idx], V[test_idx])
        prior_cv[test_idx] = pr
        learned_cv[test_idx] = pr if le is None else le
        any_learned |= le is not None
    return prior_cv, (learned_cv if any_learned else None)


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


def evaluate(
    model_factory,
    papers_by_id: dict[str, Paper],
    vecs_by_id: dict[str, np.ndarray],
    labels: dict[str, str],
    feedback: list[dict],
    cutoffs: Cutoffs,
    k: int = 5,
    seeds: list[str] | None = None,
    good: set[str] | None = None,
) -> EvalReport:
    """`model_factory()` must return a fresh, untrained RelevanceModel for the profile."""
    ids = [pid for pid in labels if pid in papers_by_id and pid in vecs_by_id]
    truth = [labels[i] for i in ids]
    counts = {l: truth.count(l) for l in config.LABELS}
    rep = EvalReport(n_labels=len(ids), label_counts=counts, paper_ids=ids, truth=truth)
    if len(ids) < MIN_LABELS or sum(1 for c in counts.values() if c) < 2:
        rep.message = (
            f"Hand-label at least {MIN_LABELS} papers with at least two different labels "
            f"to evaluate (have {len(ids)})."
        )
        return rep

    papers = [papers_by_id[i] for i in ids]
    V = np.vstack([vecs_by_id[i] for i in ids])

    # Untrained model → prior + individual feature baselines (no label leakage possible).
    base = model_factory()
    F0 = base.features(papers, V)
    prior = base.prior(F0)

    # Cross-validated learned model; the blend weight is chosen on these CV scores.
    k = max(2, min(k, len(ids) // 2))
    prior_cv, learned_cv = cv_components(model_factory, ids, truth, papers, V, labels, feedback, papers_by_id, vecs_by_id, k, seeds)
    good = good or {"READ"}
    w, blend_scores = select_blend(prior_cv, learned_cv, truth, good)
    cv_scores = prior_cv if learned_cv is None else (1 - w) * prior_cv + w * learned_cv
    rep.folds = k
    rep.blend = {"weight": w, "scores": blend_scores, "metric": "average precision", "good": sorted(good)}

    systems = [
        ("Personalised model (cross-validated)", cv_scores, True),
        ("Profile only, no learning", prior, True),
        ("Baseline: semantic similarity", F0[:, FEATURES.index("semantic")], False),
        ("Baseline: keyword match", F0[:, FEATURES.index("keyword_lex")], False),
    ]
    for name, sc, comparable in systems:
        bc, bm = best_cutoffs(sc, truth)
        sc_arr, truth_arr = np.asarray(sc), np.array(truth)
        rep.systems.append(
            SystemReport(
                name=name,
                scores=sc,
                spearman=spearman(sc, truth),
                ndcg10=ndcg_at_k(sc, truth),
                at_current=label_metrics(truth, labels_for(sc, cutoffs)) if comparable else None,
                best_cutoffs=bc,
                at_best=bm,
                spearman_ci=bootstrap_ci(lambda i: spearman(sc_arr[i], list(truth_arr[i])), len(ids)),
                ndcg10_ci=bootstrap_ci(lambda i: ndcg_at_k(sc_arr[i], list(truth_arr[i])), len(ids)),
            )
        )
    rep.cv_predictions = labels_for(cv_scores, cutoffs)
    return rep


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

GOOD_SETS = {"read": {"READ"}, "read_skim": {"READ", "SKIM"}}


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


@dataclass
class SpeedReport:
    good_labels: set[str]
    n_papers: int
    n_good: int
    curves: list[DiscoveryCurve] = field(default_factory=list)
    random: np.ndarray | None = None
    ideal: np.ndarray | None = None
    learning: list[dict] = field(default_factory=list)  # {"signals", "hit_rate_at_10", "lo", "hi", "ndcg10"}
    message: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.curves)


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


def speed_report(
    model_factory,
    papers_by_id: dict[str, Paper],
    vecs_by_id: dict[str, np.ndarray],
    label_events: list[dict],
    feedback: list[dict],
    good: set[str],
    eval_report: EvalReport | None = None,
    k: int = 5,
    steps: int = 10,
    seeds: list[str] | None = None,
    blend_weight: float | None = None,
) -> SpeedReport:
    """Discovery curves (model vs. profile-only vs. random vs. ideal) and a learning curve."""
    labels = {e["paper_id"]: e["label"] for e in label_events}
    ids = [pid for pid in labels if pid in papers_by_id and pid in vecs_by_id]
    truth = [labels[i] for i in ids]
    n_good = sum(t in good for t in truth)
    rep = SpeedReport(good_labels=good, n_papers=len(ids), n_good=n_good)
    if len(ids) < MIN_LABELS or n_good == 0 or n_good == len(ids):
        rep.message = (
            f"Needs at least {MIN_LABELS} labelled papers including both good and not-good ones "
            f"(have {len(ids)}, of which {n_good} good)."
        )
        return rep

    papers = [papers_by_id[i] for i in ids]
    V = np.vstack([vecs_by_id[i] for i in ids])
    n = len(ids)

    # Discovery curves. Reuse the evaluation's CV scores when available.
    if eval_report is not None and eval_report.ok and eval_report.paper_ids == ids:
        model_scores = eval_report.systems[0].scores
        prior_scores = eval_report.systems[1].scores
    else:
        base = model_factory()
        prior_scores = base.prior(base.features(papers, V))
        pr_cv, le_cv = cv_components(
            model_factory, ids, truth, papers, V, labels, feedback, papers_by_id, vecs_by_id, max(2, min(k, n // 2)), seeds
        )
        w = blend_weight or 0.0
        model_scores = pr_cv if le_cv is None else (1 - w) * pr_cv + w * le_cv
    rep.curves = [
        discovery_curve("Personalised model", model_scores, truth, good, with_ci=True),
        discovery_curve("Profile only", prior_scores, truth, good),
    ]
    ks = np.arange(1, n + 1)
    rep.random = ks / n
    rep.ideal = np.minimum(ks, n_good) / n_good

    # Learning curve: chronological replay of labels + explicit feedback.
    events = [(e["created_at"], "label", e) for e in label_events if e["paper_id"] in labels]
    events += [(f["created_at"], "feedback", f) for f in feedback if f["action"] in ("useful", "not_useful", "correct", "open", "dismiss")]
    events.sort(key=lambda t: t[0])
    folds = [f for f in _folds(truth, max(2, min(k, n // 2))) if len(f)]
    fold_train = []
    for test_idx in folds:
        test_ids = {ids[i] for i in test_idx}
        fold_train.append((test_idx, [ev for ev in events if ev[2]["paper_id"] not in test_ids]))
    max_train = max(len(tr) for _, tr in fold_train)
    grid = np.unique(np.linspace(0, max_train, min(steps, max_train + 1)).round().astype(int))
    for m in grid:
        scores = np.zeros(n)
        for test_idx, train in fold_train:
            prefix = train[:m]
            lab = {ev[2]["paper_id"]: ev[2]["label"] for ev in prefix if ev[1] == "label"}
            fb = [ev[2] for ev in prefix if ev[1] == "feedback"]
            model = model_factory()
            model.weight_override = blend_weight
            model.fit(build_examples(lab, fb, seeds=seeds), papers_by_id, vecs_by_id)
            s, _, _ = model.score([papers[i] for i in test_idx], V[test_idx])
            scores[test_idx] = s
        curve = discovery_curve("", scores, truth, good, with_ci=True)
        lo, hi = curve.hit_rate_ci or (curve.hit_rate_at_10, curve.hit_rate_at_10)
        rep.learning.append(
            {"signals": int(m), "hit_rate_at_10": curve.hit_rate_at_10, "lo": lo, "hi": hi, "ndcg10": ndcg_at_k(scores, truth)}
        )
    return rep
