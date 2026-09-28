"""Ranking + triage (TUNED).

In: scores → Out: ranked list + READ / SKIM / SKIP label.

Cutoffs are set by human judgment (sliders in the UI) and can be tuned against
hand labels on the Evaluation page. The user's time budget optionally caps how
many papers get READ; the overflow is demoted to SKIM (with a note), so the
top of the list is always something the user can actually finish.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .models import Paper, TriageResult
from .relevance import FEATURES


@dataclass
class Cutoffs:
    read: float = config.DEFAULT_READ_CUTOFF
    skim: float = config.DEFAULT_SKIM_CUTOFF

    def __post_init__(self) -> None:
        self.read = float(np.clip(self.read, 0.0, 1.0))
        self.skim = float(np.clip(self.skim, 0.0, 1.0))
        if self.skim > self.read:
            self.skim, self.read = self.read, self.skim


def label_for(score: float, cutoffs: Cutoffs) -> str:
    if score >= cutoffs.read:
        return "READ"
    if score >= cutoffs.skim:
        return "SKIM"
    return "SKIP"


def labels_for(scores: np.ndarray, cutoffs: Cutoffs) -> list[str]:
    return [label_for(float(s), cutoffs) for s in scores]


def read_budget(hours_per_week: float) -> int:
    """How many READ papers fit the user's weekly time (≥1)."""
    return max(1, int((max(0.0, hours_per_week) * 60) // config.MINUTES_PER_READ))


def triage(
    papers: list[Paper],
    scores: np.ndarray,
    prior: np.ndarray,
    F: np.ndarray,
    cutoffs: Cutoffs,
    max_read: int | None = None,
    user_labels: dict[str, str] | None = None,
) -> list[TriageResult]:
    """Rank papers by score and assign labels.

    `user_labels` (corrections/hand labels) override the model's label for that
    paper: the user already told us what it is.
    """
    user_labels = user_labels or {}
    order = np.argsort(-scores, kind="stable")
    out: list[TriageResult] = []
    n_read = 0
    for rank, i in enumerate(order, start=1):
        p = papers[i]
        s = float(scores[i])
        label = label_for(s, cutoffs)
        note = ""
        if p.id in user_labels:
            if user_labels[p.id] != label:
                note = f"model said {label}; your label applied"
            label = user_labels[p.id]
        elif label == "READ" and max_read is not None:
            if n_read >= max_read:
                label, note = "SKIM", "demoted from READ: over your weekly reading budget"
        if label == "READ":
            n_read += 1
        out.append(
            TriageResult(
                paper=p,
                score=s,
                label=label,
                rank=rank,
                prior_score=float(prior[i]),
                features={f: float(F[i, j]) for j, f in enumerate(FEATURES)},
                note=note,
            )
        )
    return out


def summarize(results: list[TriageResult]) -> dict[str, int]:
    counts = {lab: 0 for lab in config.LABELS}
    for r in results:
        counts[r.label] += 1
    return counts


def estimated_minutes(results: list[TriageResult]) -> int:
    c = summarize(results)
    return c["READ"] * config.MINUTES_PER_READ + c["SKIM"] * config.MINUTES_PER_SKIM


# Cosine similarity above which two papers count as near-duplicates, per backend.
SIMILAR_THRESHOLD = {"sbert": 0.72, "tfidf": 0.55}  # sbert: ≥0.72 on the sample = same narrow subtopic


def group_similar(results: list[TriageResult], vecs_by_id: dict[str, np.ndarray], threshold: float) -> list[TriageResult]:
    """Group near-identical papers under the highest-ranked one (in place).

    Walking down the ranking, a paper joins the most similar existing lead if the
    cosine similarity is ≥ threshold, otherwise it becomes a lead itself. Scores
    and labels are untouched; the UI uses this to show one card per cluster.
    """
    lead_ids: list[str] = []
    lead_vecs: list[np.ndarray] = []
    by_id = {r.paper.id: r for r in results}
    for r in results:
        r.group_lead, r.similar = "", []
    for r in results:
        v = vecs_by_id.get(r.paper.id)
        if v is None:
            continue
        if lead_vecs:
            sims = np.vstack(lead_vecs) @ v
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                r.group_lead = lead_ids[j]
                by_id[lead_ids[j]].similar.append(r.paper.id)
                continue
        lead_ids.append(r.paper.id)
        lead_vecs.append(v)
    return results
