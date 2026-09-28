"""Paper relevance model (TRAINED).

In:  interest profile + paper vectors + feedback
Out: relevance / preference score in [0, 1]

Two layers:

1. **Prior** – a transparent, hand-weighted combination of interpretable
   features (semantic similarity, focus similarity, keyword matches, recency,
   avoided-topic penalty, similarity to papers the user liked/disliked). Works
   with zero labels, so the system is useful on day one.
2. **Learned preference model** – a ridge regression over the same features,
   trained on hand labels, seed papers and explicit/implicit feedback. How much
   it counts is *validated per profile*: the pipeline cross-validates a few
   blend weights on the hand labels and keeps learning only if it beats the
   prior (see `evaluate.select_blend`). Without enough labels to validate, a
   conservative weight n / (n + K), capped at 30%, is used.

The feedback-affinity feature is computed leave-self-out during training so a
labelled paper doesn't "see" its own label (target leakage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

from . import config
from .embeddings import Embedder
from .models import Paper
from .preprocess import phrase_in_text
from .profile import ProfileVectors

FEATURES = ["semantic", "focus", "keyword_sem", "keyword_lex", "recency", "avoid", "feedback"]

PRIOR_WEIGHTS = {
    "semantic": 0.40,
    "focus": 0.15,
    "keyword_sem": 0.15,
    "keyword_lex": 0.25,
    "recency": 0.05,
    "avoid": -0.40,
    "feedback": 0.25,
}

BLEND_K = 30  # unvalidated learned weight = n / (n + BLEND_K) …
MAX_UNVALIDATED_WEIGHT = 0.3  # … capped low until cross-validation says learning helps
BLEND_GRID = (0.0, 0.15, 0.3, 0.5, 0.7)
MIN_TRAIN_EXAMPLES = 8

# Feedback → (target, sample weight). Hand labels weigh 1.0.
FEEDBACK_TARGETS = {
    "useful": (1.0, 1.0),
    "not_useful": (0.0, 1.0),
    "open": (0.75, 0.3),
    "dismiss": (0.15, 0.5),
}
CORRECTION_WEIGHT = 1.5
SEED_WEIGHT = 2.0  # papers the user named as essential


@dataclass
class Example:
    paper_id: str
    target: float  # 0..1 (SKIP=0, SKIM=.5, READ=1)
    weight: float
    source: str  # "label" | "feedback" | "seed"


def build_examples(
    labels: dict[str, str],
    feedback: list[dict],
    exclude: set[str] | None = None,
    use_labels: bool = True,
    seeds: list[str] | None = None,
) -> list[Example]:
    """One training example per paper. Priority: seed > hand label > correction > explicit > implicit."""
    exclude = exclude or set()
    out: dict[str, Example] = {}
    for pid in seeds or []:
        if pid not in exclude:
            out[pid] = Example(pid, 1.0, SEED_WEIGHT, "seed")
    if use_labels:
        for pid, lab in labels.items():
            if pid not in exclude and pid not in out:
                out[pid] = Example(pid, config.LABEL_VALUE[lab], 1.0, "label")
    explicit: dict[str, Example] = {}
    implicit: dict[str, Example] = {}
    for f in feedback:  # chronological, later events override earlier ones
        pid, action = f["paper_id"], f["action"]
        if pid in exclude or pid in out:
            continue
        if action == "correct" and f.get("value") in config.LABEL_VALUE:
            explicit[pid] = Example(pid, config.LABEL_VALUE[f["value"]], CORRECTION_WEIGHT, "feedback")
        elif action in ("useful", "not_useful"):
            t, w = FEEDBACK_TARGETS[action]
            explicit[pid] = Example(pid, t, w, "feedback")
        elif action in ("open", "dismiss"):
            t, w = FEEDBACK_TARGETS[action]
            prev = implicit.get(pid)
            # "opened then dismissed" → dismissed; "dismissed then opened" → opened
            implicit[pid] = Example(pid, t, w if prev is None else max(w, prev.weight), "feedback")
    for pid, ex in implicit.items():
        explicit.setdefault(pid, ex)
    out.update({k: v for k, v in explicit.items() if k not in out})
    return list(out.values())


def _recency(p: Paper, today: date | None = None) -> float:
    today = today or date.today()
    try:
        d = datetime.strptime(p.published[:10], "%Y-%m-%d").date() if p.published else date(int(p.year), 7, 1)
    except (TypeError, ValueError):
        return 0.5
    age = max(0, (today - d).days)
    return float(np.exp(-age / 365.0))


def _keyword_lex(p: Paper, keywords: list[str]) -> tuple[float, list[str], list[str]]:
    """Coverage of the user's keywords; title hits count more. Full credit at 3 keyword hits."""
    if not keywords:
        return 0.0, [], []
    body = f"{p.abstract} {p.full_text[:5000]}"
    in_title = [k for k in keywords if phrase_in_text(k, p.title)]
    in_body = [k for k in keywords if k not in in_title and phrase_in_text(k, body)]
    raw = 1.0 * len(in_title) + 0.6 * len(in_body)
    return float(min(1.0, raw / min(len(keywords), 3))), in_title, in_body


def _topk_mean(sims: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean of the top-k similarities per row (robust 'nearest liked papers')."""
    if sims.shape[1] == 0:
        return np.zeros(sims.shape[0])
    k = min(k, sims.shape[1])
    part = -np.partition(-sims, k - 1, axis=1)[:, :k]
    return part.mean(axis=1)


@dataclass
class RelevanceModel:
    embedder: Embedder
    profile_vecs: ProfileVectors
    ridge_coef: np.ndarray | None = None
    ridge_intercept: float = 0.0
    learned_weight: float = 0.0
    n_train: int = 0
    pos_vecs: np.ndarray | None = None
    neg_vecs: np.ndarray | None = None
    pos_ids: list[str] = field(default_factory=list)
    neg_ids: list[str] = field(default_factory=list)
    train_info: dict = field(default_factory=dict)
    weight_override: float | None = None  # validated blend weight, when available

    # ----------------------------------------------------------- features
    def features(self, papers: list[Paper], vecs: np.ndarray) -> np.ndarray:
        """(n, len(FEATURES)) matrix, every column roughly in [0, 1] (feedback in [-1, 1])."""
        n = len(papers)
        if n == 0:
            return np.zeros((0, len(FEATURES)))
        pv, cal = self.profile_vecs, self.embedder.calibrate
        F = np.zeros((n, len(FEATURES)))
        sem = cal(vecs @ pv.main)
        F[:, 0] = sem
        F[:, 1] = cal(vecs @ pv.focus, "short") if pv.focus is not None else sem
        F[:, 2] = cal((vecs @ pv.keywords.T).max(axis=1), "short") if len(pv.keywords) else sem
        F[:, 3] = [_keyword_lex(p, pv.keyword_names)[0] for p in papers]
        F[:, 4] = [_recency(p) for p in papers]
        if len(pv.avoid):
            avoid_sem = cal((vecs @ pv.avoid.T).max(axis=1), "short")
            avoid_lex = np.array(
                [1.0 if any(phrase_in_text(a, f"{p.title} {p.abstract}") for a in pv.avoid_names) else 0.0 for p in papers]
            )
            # Only penalise when the avoided topic is clearly present.
            F[:, 5] = np.maximum(np.clip((avoid_sem - 0.5) * 2, 0, 1), avoid_lex * 0.8)
        F[:, 6] = self._feedback_affinity(papers, vecs)
        return F

    def _feedback_affinity(self, papers: list[Paper], vecs: np.ndarray) -> np.ndarray:
        cal = self.embedder.calibrate
        out = np.zeros(len(papers))
        for vecs_ref, ids_ref, sign in ((self.pos_vecs, self.pos_ids, 1.0), (self.neg_vecs, self.neg_ids, -1.0)):
            if vecs_ref is None or len(vecs_ref) == 0:
                continue
            sims = vecs @ vecs_ref.T
            # Leave-self-out: a paper must not match its own feedback.
            idx = {pid: j for j, pid in enumerate(ids_ref)}
            for i, p in enumerate(papers):
                j = idx.get(p.id)
                if j is not None:
                    sims[i, j] = -1.0
            out += sign * cal(_topk_mean(sims))
        return out

    # ------------------------------------------------------------- scoring
    @staticmethod
    def prior(F: np.ndarray) -> np.ndarray:
        w = np.array([PRIOR_WEIGHTS[f] for f in FEATURES])
        return np.clip(F @ w, 0.0, 1.0)

    def components(self, papers: list[Paper], vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """(prior score, learned score or None if untrained, feature matrix)."""
        F = self.features(papers, vecs)
        prior = self.prior(F)
        if self.ridge_coef is None:
            return prior, None, F
        return prior, np.clip(F @ self.ridge_coef + self.ridge_intercept, 0.0, 1.0), F

    def score(self, papers: list[Paper], vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (final score, prior score, feature matrix)."""
        prior, learned, F = self.components(papers, vecs)
        if learned is None or self.learned_weight == 0:
            return prior, prior, F
        final = (1 - self.learned_weight) * prior + self.learned_weight * learned
        return final, prior, F

    # ------------------------------------------------------------ training
    def fit(self, examples: list[Example], papers_by_id: dict[str, Paper], vecs_by_id: dict[str, np.ndarray]) -> "RelevanceModel":
        from sklearn.linear_model import Ridge

        ex = [e for e in examples if e.paper_id in papers_by_id and e.paper_id in vecs_by_id]
        # Feedback-affinity anchors: clearly positive / negative examples.
        pos = [e for e in ex if e.target >= 0.75]
        neg = [e for e in ex if e.target <= 0.25]
        self.pos_ids = [e.paper_id for e in pos]
        self.neg_ids = [e.paper_id for e in neg]
        self.pos_vecs = np.vstack([vecs_by_id[i] for i in self.pos_ids]) if pos else None
        self.neg_vecs = np.vstack([vecs_by_id[i] for i in self.neg_ids]) if neg else None
        self.n_train = len(ex)
        self.ridge_coef, self.ridge_intercept, self.learned_weight = None, 0.0, 0.0
        targets = np.array([e.target for e in ex])
        self.train_info = {"n_examples": len(ex), "n_pos": len(pos), "n_neg": len(neg), "learned": False}
        if len(ex) < MIN_TRAIN_EXAMPLES or np.ptp(targets) == 0:
            self.train_info["reason"] = (
                f"need ≥{MIN_TRAIN_EXAMPLES} examples with at least two different labels to train "
                f"(have {len(ex)}); using the prior only"
            )
            return self
        P = [papers_by_id[e.paper_id] for e in ex]
        V = np.vstack([vecs_by_id[e.paper_id] for e in ex])
        F = self.features(P, V)
        # Fit the residual-free target directly; alpha keeps weights sane with few labels.
        ridge = Ridge(alpha=1.0).fit(F, targets, sample_weight=np.array([e.weight for e in ex]))
        self.ridge_coef, self.ridge_intercept = ridge.coef_, float(ridge.intercept_)
        if self.weight_override is not None:
            self.learned_weight = float(self.weight_override)
        else:
            self.learned_weight = min(MAX_UNVALIDATED_WEIGHT, len(ex) / (len(ex) + BLEND_K))
        self.train_info.update(
            learned=True,
            learned_weight=round(self.learned_weight, 3),
            weight_validated=self.weight_override is not None,
            coefficients={f: round(float(c), 3) for f, c in zip(FEATURES, ridge.coef_)},
            intercept=round(self.ridge_intercept, 3),
        )
        return self


def keyword_matches(p: Paper, keywords: list[str]) -> tuple[list[str], list[str]]:
    _, t, b = _keyword_lex(p, keywords)
    return t, b
