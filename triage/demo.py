"""Demo profile with simulated labels and feedback.

Everything created here is **synthetic** and marked as such (the profile is
flagged `demo` in settings, labels use labeler="simulated"). It exists so the
Evaluation tab, learning curve and personalised ranking have something to show
before a real user has labelled anything.

The simulated researcher is a transparent keyword rule set that is independent
of the ranking model (it never looks at embeddings or scores), plus a little
seeded noise so the labels are imperfect, like real subjective labels.
"""

from __future__ import annotations

import random
import re
from datetime import datetime, timedelta, timezone

from .models import InterestProfile, Paper
from .pipeline import TriageEngine

DEMO_NAME = "Demo — RAG research"
DEMO_PROFILE = dict(
    name=DEMO_NAME,
    description=(
        "I build retrieval-augmented generation (RAG) systems for question answering over scientific "
        "documents, and study how LLM agents use search tools and cite evidence."
    ),
    keywords=["retrieval augmented generation", "question answering", "LLM agents", "information retrieval", "reranking"],
    focus="evaluating retrieval quality and faithfulness in RAG pipelines",
    avoid=["robotics", "medical imaging"],
    hours_per_week=3.0,
)

_CORE = re.compile(
    r"retrieval[- ]augmented|\brag\b|rerank|passage retrieval|dense retriev|question answering|\bqa\b|"
    r"fact[- ]check|citation|grounded generation|hallucinat|search agent|agentic search|web agent",
    re.I,
)
_RELATED = re.compile(
    r"retriev|search|large language model|\bllms?\b|language model|agent|benchmark|embedding|"
    r"recommend|knowledge graph|summari[sz]|faithful|evidence|document",
    re.I,
)
_OFF = re.compile(r"robot|medical imag|segmentation|locomotion|quadrotor|autopilot|mri\b", re.I)


def simulated_label(p: Paper) -> str:
    """What the simulated RAG researcher would say about this paper."""
    title, text = p.title, f"{p.title} {p.abstract}"
    if _OFF.search(text):
        return "SKIP"
    core_hits = len(_CORE.findall(text))
    if _CORE.search(title) or core_hits >= 3:
        return "READ"
    if core_hits >= 1 or len(_RELATED.findall(text)) >= 3:
        return "SKIM"
    return "SKIP"


def _noisy(label: str, rng: random.Random, rate: float = 0.12) -> str:
    """Occasionally shift a label one step, like a tired human labeller."""
    if rng.random() >= rate:
        return label
    return {"READ": "SKIM", "SKIP": "SKIM", "SKIM": rng.choice(["READ", "SKIP"])}[label]


def seed_demo(engine: TriageEngine, n_labels: int = 60, n_feedback: int = 45, seed: int = 7) -> InterestProfile:
    """Create (or reset) the demo profile with sample papers, simulated labels and feedback."""
    store = engine.store
    old = store.get_profile(DEMO_NAME)
    if old is not None:
        store.delete_profile(old.id)
    prof = store.save_profile(InterestProfile(**DEMO_PROFILE))
    engine.fetch_into_pool(prof, "sample")
    store.set_setting(prof.id, "demo", True)

    rng = random.Random(seed)
    papers = engine.pool(prof)
    truth = {p.id: simulated_label(p) for p in papers}
    by_label = {lab: [p for p in papers if truth[p.id] == lab] for lab in ("READ", "SKIM", "SKIP")}
    for group in by_label.values():
        rng.shuffle(group)

    # Three essential papers the researcher named up front (seed papers).
    seed_papers, by_label["READ"] = by_label["READ"][:3], by_label["READ"][3:]
    store.add_seeds(prof.id, [p.id for p in seed_papers])

    # Hand-label set: all READ candidates (they're rare) plus a mix of SKIM/SKIP.
    n_read = min(len(by_label["READ"]), max(8, n_labels // 4))
    n_skim = min(len(by_label["SKIM"]), (n_labels - n_read) // 2)
    labelled = by_label["READ"][:n_read] + by_label["SKIM"][:n_skim] + by_label["SKIP"][: n_labels - n_read - n_skim]
    rng.shuffle(labelled)

    # Feedback on other papers, as if browsing the reading list over two weeks.
    rest = [p for p in papers if p.id not in {q.id for q in labelled}]
    browse = (
        by_label["READ"][n_read:]
        + [p for p in rest if truth[p.id] == "SKIM"][: n_feedback // 2]
        + [p for p in rest if truth[p.id] == "SKIP"][: n_feedback // 3]
    )[:n_feedback]
    rng.shuffle(browse)

    # Interleave labelling sessions and browsing sessions over the past 14 days.
    events: list[tuple[str, Paper]] = [("label", p) for p in labelled] + [("feedback", p) for p in browse]
    rng.shuffle(events)
    start = datetime.now(timezone.utc) - timedelta(days=14)
    step = timedelta(days=14) / max(1, len(events))
    for i, (kind, p) in enumerate(events):
        ts = (start + i * step + timedelta(seconds=rng.randint(0, 600))).isoformat(timespec="seconds")
        lab = _noisy(truth[p.id], rng)
        if kind == "label":
            store.set_label(prof.id, p.id, lab, labeler="simulated", created_at=ts)
            continue
        store.log_feedback(prof.id, p.id, "open", created_at=ts)
        if lab == "READ":
            store.log_feedback(prof.id, p.id, "useful", created_at=ts)
        elif lab == "SKIP":
            action = "dismiss" if rng.random() < 0.4 else "not_useful"
            store.log_feedback(prof.id, p.id, action, created_at=ts)
        elif rng.random() < 0.5:
            store.log_feedback(prof.id, p.id, "correct", "SKIM", created_at=ts)
    # Papers arrived over several days: most three days ago, a batch today.
    # The simulated last visit was yesterday, so "New" shows today's batch.
    now = datetime.now(timezone.utc)
    touched = {p.id for p in labelled} | {p.id for p in browse}
    fresh = [p.id for p in papers if p.id not in touched]
    rng.shuffle(fresh)
    today = set(fresh[:45])
    store.set_pool_added(prof.id, {
        p.id: (now - (timedelta(hours=2) if p.id in today else timedelta(days=3))).isoformat(timespec="seconds")
        for p in papers
    })
    store.set_setting(prof.id, "last_visit", (now - timedelta(days=1)).isoformat(timespec="seconds"))

    # A couple of explanation reviews so that panel isn't empty either.
    for p in labelled[:4]:
        store.log_feedback(prof.id, p.id, "explanation_ok", "template")
    return prof


def is_demo(engine: TriageEngine, profile: InterestProfile) -> bool:
    return bool(profile.id is not None and engine.store.get_setting(profile.id, "demo", False))
