"""End-to-end orchestration used by the UI, CLI scripts and tests."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from . import config, fulltext, seeds, sources
from .embeddings import Embedder, get_embedder
from .evaluate import (
    GOOD_SETS,
    MIN_LABELS,
    EvalReport,
    SpeedReport,
    _folds,
    cv_components,
    evaluate,
    select_blend,
    speed_report,
)
from .models import InterestProfile, Paper, TriageResult
from .profile import ProfileVectors, build_profile_vectors, profile_texts
from .ranking import SIMILAR_THRESHOLD, Cutoffs, group_similar, read_budget, triage
from .relevance import RelevanceModel, build_examples
from .store import Store

log = logging.getLogger(__name__)

AUTO_FETCH_HOURS = 20  # refresh a profile's feed at most about once a day
BORDERLINE_READ = 0.10  # |score − read cutoff| within this → fetch full text
BORDERLINE_SKIM = 0.06
MAX_FULL_TEXT_PER_RUN = 12


@dataclass
class TriageRun:
    results: list[TriageResult]
    model: RelevanceModel
    papers: list[Paper]
    vecs_by_id: dict[str, np.ndarray]
    embedder_name: str
    warnings: list[str] = field(default_factory=list)
    blend: dict = field(default_factory=dict)

    @property
    def by_id(self) -> dict[str, TriageResult]:
        return {r.paper.id: r for r in self.results}


@dataclass
class _Context:
    """Everything scoring needs for one profile: papers, vectors, profile vectors."""

    papers_by_id: dict[str, Paper]
    vecs: dict[str, np.ndarray]
    emb: Embedder
    pv: ProfileVectors
    seeds: list[str]
    labels: dict[str, str]
    feedback: list[dict]

    def model(self, weight: float | None = None) -> RelevanceModel:
        return RelevanceModel(embedder=self.emb, profile_vecs=self.pv, weight_override=weight)


class TriageEngine:
    def __init__(self, store: Store | None = None, backend: str | None = None):
        self.store = store or Store()
        self.backend = backend

    # ------------------------------------------------------------- data
    def fetch_into_pool(
        self,
        profile: InterestProfile,
        source: str,
        categories: list[str] | None = None,
        max_results: int = 100,
        days_back: int | None = None,
        extra_query: list[str] | None = None,
        progress=None,
    ) -> tuple[int, int, list[str]]:
        """Fetch papers → store → add to the profile's pool. Every attempt is logged.

        Returns (n_fetched, n_new_in_pool, warnings).
        """
        if profile.id is None:
            raise ValueError("save the profile before fetching")
        query = list(extra_query or []) or list(profile.keywords)
        if not query and source not in ("sample", "arxiv_new"):
            raise sources.SourceError("Add keywords to your profile (they are used as the search query).")
        try:
            papers, warnings = sources.fetch(source, query, categories, max_results, days_back, progress)
        except Exception as e:
            self.store.log_fetch(profile.id, source, False, message=str(e))
            raise
        if papers:
            self.store.upsert_papers(papers)
        added = self.store.add_to_pool(profile.id, [p.id for p in papers])
        self.store.log_fetch(profile.id, source, bool(papers), len(papers), added, "; ".join(warnings))
        return len(papers), added, warnings

    def refresh_if_due(self, profile: InterestProfile, force: bool = False, progress=None) -> dict | None:
        """Daily auto-fetch of new arXiv listings for the profile's categories (+ full text for borderline).

        Returns a summary dict when a fetch happened, else None.
        """
        if profile.id is None or not self.store.get_setting(profile.id, "auto_fetch", True):
            return None
        if self.store.get_setting(profile.id, "demo", False):
            return None  # the demo profile is a fixed snapshot
        cats = self.store.get_setting(profile.id, "categories", None)
        if not cats:
            return None
        last = self.store.last_successful_fetch(profile.id)
        if not force and last:
            age = datetime.now(timezone.utc) - datetime.fromisoformat(last)
            if age < timedelta(hours=AUTO_FETCH_HOURS):
                return None
        try:
            got, added, warns = self.fetch_into_pool(profile, "arxiv_new", cats, progress=progress)
        except (sources.SourceError, ValueError) as e:
            return {"ok": False, "message": str(e)}
        enriched = self.enrich_borderline(profile) if added else 0
        return {"ok": True, "fetched": got, "added": added, "warnings": warns, "full_text": enriched}

    def pool(self, profile: InterestProfile) -> list[Paper]:
        ids = self.store.pool_ids(profile.id)
        by_id = {p.id: p for p in self.store.get_papers(ids)}
        return [by_id[i] for i in ids if i in by_id]

    # ------------------------------------------------------------ seeds
    def add_seed_papers(self, profile: InterestProfile, text: str = "", bibtex: str = "") -> tuple[int, list[str]]:
        papers, warnings = seeds.resolve(text, bibtex)
        if papers:
            self.store.upsert_papers(papers)
            n = self.store.add_seeds(profile.id, [p.id for p in papers])
        else:
            n = 0
        return n, warnings

    def seed_papers(self, profile: InterestProfile) -> list[Paper]:
        ids = self.store.seed_ids(profile.id)
        by_id = {p.id: p for p in self.store.get_papers(ids)}
        return [by_id[i] for i in ids if i in by_id]

    # ------------------------------------------------------- vectors
    def _context(self, profile: InterestProfile, include_labelled: bool = False) -> _Context:
        pool = self.pool(profile)
        seed_papers = self.seed_papers(profile)
        labels = self.store.get_labels(profile.id)
        seed_ids = {p.id for p in seed_papers}
        papers = pool + [p for p in seed_papers if p.id not in {q.id for q in pool}]
        if include_labelled:  # labelled papers may have left the pool
            have = {p.id for p in papers}
            papers += self.store.get_papers([pid for pid in labels if pid not in have])
        emb = get_embedder(self.backend)
        if emb.needs_fit:
            emb.fit([p.text for p in papers] + profile_texts(profile))
        V = emb.encode_papers(papers, self.store)
        vecs = {p.id: V[i] for i, p in enumerate(papers)}
        seed_vecs = np.vstack([vecs[i] for i in seed_ids]) if seed_ids else None
        pv = build_profile_vectors(profile, emb, seed_vecs)
        return _Context(
            {p.id: p for p in papers},
            vecs,
            emb,
            pv,
            [p.id for p in seed_papers],
            {k: v for k, v in labels.items() if k not in seed_ids},
            self.store.get_feedback(profile.id),
        )

    # ---------------------------------------------------- blend weight
    def blend(self, profile: InterestProfile, ctx: _Context | None = None) -> dict:
        """How much the learned model should count, validated on this profile's labels (cached)."""
        good = self.store.get_setting(profile.id, "good_mode", "read")
        key = "|".join([
            self.store.feedback_version(profile.id), str(len(self.store.pool_ids(profile.id))), good,
            profile.fingerprint(), str(self.store.get_setting(profile.id, "content_rev", "")),
        ])
        cached = self.store.get_setting(profile.id, "blend", None)
        if cached and cached.get("key") == key:
            return cached
        ctx = ctx or self._context(profile, include_labelled=True)
        ids = [pid for pid in ctx.labels if pid in ctx.papers_by_id]
        truth = [ctx.labels[i] for i in ids]
        good_set = GOOD_SETS[good]
        n_good = sum(t in good_set for t in truth)
        result = {"key": key, "weight": None, "scores": {}, "validated": False,
                  "reason": f"needs ≥{MIN_LABELS} hand labels including good and not-good papers"}
        if len(ids) >= MIN_LABELS and 0 < n_good < len(ids):
            papers = [ctx.papers_by_id[i] for i in ids]
            V = np.vstack([ctx.vecs[i] for i in ids])
            k = max(2, min(5, len(ids) // 2))
            prior_cv, learned_cv = cv_components(
                ctx.model, ids, truth, papers, V, ctx.labels, ctx.feedback, ctx.papers_by_id, ctx.vecs, k, ctx.seeds
            )
            w, scores = select_blend(prior_cv, learned_cv, truth, good_set)
            result.update(weight=w, scores={str(k_): v for k_, v in scores.items()}, validated=bool(scores), reason="")
            if not scores:
                result["reason"] = "not enough varied feedback to train a learned model yet"
        self.store.set_setting(profile.id, "blend", result)
        return result

    # ------------------------------------------------------------ run
    def run(
        self,
        profile: InterestProfile,
        cutoffs: Cutoffs | None = None,
        respect_budget: bool = True,
        learn: bool = True,
        group: bool | None = None,
    ) -> TriageRun:
        cutoffs = cutoffs or self.cutoffs(profile)
        if not self.store.pool_ids(profile.id):
            raise ValueError("No papers yet — fetch some papers first.")
        ctx = self._context(profile, include_labelled=True)
        blend = self.blend(profile, ctx) if learn else {}
        model = ctx.model(blend.get("weight"))
        if learn:
            model.fit(build_examples(ctx.labels, ctx.feedback, seeds=ctx.seeds), ctx.papers_by_id, ctx.vecs)
        seed_set = set(ctx.seeds)
        pool_ids = set(self.store.pool_ids(profile.id))
        papers = [p for pid, p in ctx.papers_by_id.items() if pid in pool_ids and pid not in seed_set]
        if not papers:
            raise ValueError("No papers yet — fetch some papers first.")
        V = np.vstack([ctx.vecs[p.id] for p in papers])
        scores, prior, F = model.score(papers, V)
        results = triage(
            papers,
            scores,
            prior,
            F,
            cutoffs,
            max_read=read_budget(profile.hours_per_week) if respect_budget else None,
            user_labels=self.user_labels(profile, ctx.labels, ctx.feedback),
        )
        if group if group is not None else self.store.get_setting(profile.id, "group_similar", True):
            backend = "sbert" if ctx.emb.name.startswith("sbert") else "tfidf"
            group_similar(results, ctx.vecs, SIMILAR_THRESHOLD[backend])
        return TriageRun(results, model, papers, ctx.vecs, ctx.emb.name, blend=blend)

    @staticmethod
    def user_labels(profile: InterestProfile, labels: dict[str, str], feedback: list[dict]) -> dict[str, str]:
        """Latest explicit correction per paper, falling back to hand labels."""
        out = dict(labels)
        for f in feedback:
            if f["action"] == "correct" and f.get("value") in config.LABELS:
                out[f["paper_id"]] = f["value"]
        return out

    # ------------------------------------------------------- full text
    def borderline(self, profile: InterestProfile, run: TriageRun | None = None) -> list[Paper]:
        """Papers near a cutoff that haven't had full text fetched yet (arXiv only), most borderline first."""
        run = run or self.run(profile, respect_budget=False, group=False)
        cut = self.cutoffs(profile)
        cands = []
        for r in run.results:
            p = r.paper
            if p.full_text_status or not p.id.startswith("arxiv:"):
                continue
            d = min(abs(r.score - cut.read) / BORDERLINE_READ, abs(r.score - cut.skim) / BORDERLINE_SKIM)
            if d <= 1.0:
                cands.append((d, p))
        return [p for _, p in sorted(cands, key=lambda t: t[0])]

    def enrich_borderline(self, profile: InterestProfile, limit: int = MAX_FULL_TEXT_PER_RUN) -> int:
        """Fetch intro + conclusion for up to `limit` borderline papers. Returns how many got full text."""
        if not self.store.get_setting(profile.id, "full_text", True):
            return 0
        todo = self.borderline(profile)[:limit]
        if not todo:
            return 0
        done = fulltext.enrich(todo)
        self.store.upsert_papers(done)
        self.store.set_setting(profile.id, "content_rev", datetime.now(timezone.utc).isoformat())
        return sum(1 for p in done if p.full_text)

    # ------------------------------------------------------ evaluation
    def evaluate(self, profile: InterestProfile, cutoffs: Cutoffs | None = None, k: int = 5) -> EvalReport:
        return self.evaluate_all(profile, cutoffs, k=k)[0]

    def evaluate_all(
        self, profile: InterestProfile, cutoffs: Cutoffs | None = None, good: str = "read", k: int = 5
    ) -> tuple[EvalReport, SpeedReport]:
        """Label-agreement report plus the 'how fast are good papers found' report (shared vectors)."""
        cutoffs = cutoffs or self.cutoffs(profile)
        ctx = self._context(profile, include_labelled=True)
        good_set = GOOD_SETS[good]
        rep = evaluate(ctx.model, ctx.papers_by_id, ctx.vecs, ctx.labels, ctx.feedback, cutoffs, k=k,
                       seeds=ctx.seeds, good=good_set)
        seed_set = set(ctx.seeds)
        label_events = [e for e in self.store.get_label_events(profile.id) if e["paper_id"] not in seed_set]
        speed = speed_report(
            ctx.model, ctx.papers_by_id, ctx.vecs, label_events, ctx.feedback, good_set, rep, k=k,
            seeds=ctx.seeds, blend_weight=rep.blend.get("weight"),
        )
        return rep, speed

    # -------------------------------------------------------- settings
    def cutoffs(self, profile: InterestProfile) -> Cutoffs:
        c = self.store.get_setting(profile.id, "cutoffs", None) if profile.id is not None else None
        return Cutoffs(**c) if c else Cutoffs()

    def save_cutoffs(self, profile: InterestProfile, c: Cutoffs) -> None:
        self.store.set_setting(profile.id, "cutoffs", {"read": c.read, "skim": c.skim})

    def new_since(self, profile: InterestProfile, since: str | None) -> set[str]:
        """Pool papers added after `since` (ISO timestamp); empty set if `since` is None."""
        if not since:
            return set()
        return {pid for pid, t in self.store.pool_added(profile.id).items() if t > since}
