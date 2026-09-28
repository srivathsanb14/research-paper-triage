import numpy as np
import pytest

from triage.embeddings import TfidfEmbedder
from triage.evaluate import best_cutoffs, label_metrics, ndcg_at_k
from triage.explain import explain, template_reason, verify_llm_reason
from triage.models import InterestProfile
from triage.pipeline import TriageEngine
from triage.profile import build_profile_vectors, parse_list
from triage.ranking import Cutoffs, label_for, read_budget, triage
from triage.relevance import RelevanceModel, build_examples


@pytest.fixture
def engine(store):
    return TriageEngine(store, backend="tfidf")


def _setup(engine, store, corpus, profile):
    store.upsert_papers(corpus)
    prof = store.save_profile(profile)
    store.add_to_pool(prof.id, [p.id for p in corpus])
    return prof


def test_parse_list():
    assert parse_list("a, b; c\nA\n\n d ") == ["a", "b", "c", "d"]


def test_empty_profile_rejected(store, corpus, engine):
    prof = store.save_profile(InterestProfile(name="empty"))
    store.add_to_pool(prof.id, [p.id for p in corpus])
    store.upsert_papers(corpus)
    with pytest.raises(ValueError):
        engine.run(prof)


def test_no_papers(store, engine, rag_profile):
    prof = store.save_profile(rag_profile)
    with pytest.raises(ValueError, match="No papers"):
        engine.run(prof)


def test_ranking_puts_relevant_first(store, engine, corpus, rag_profile):
    prof = _setup(engine, store, corpus, rag_profile)
    run = engine.run(prof, Cutoffs(read=0.6, skim=0.35), respect_budget=False)
    top5 = {r.paper.title for r in run.results[:5]}
    rag_titles = {p.title for p in corpus[:5]}
    assert len(top5 & rag_titles) >= 4
    bottom = [r.paper.title for r in run.results[-6:]]
    assert any("Robot" in t or "Quadrotor" in t for t in bottom)
    # scores sorted, ranks contiguous, features present
    scores = [r.score for r in run.results]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in run.results] == list(range(1, len(corpus) + 1))
    assert all(0 <= r.score <= 1 for r in run.results)
    # avoided topic is penalised
    robot = next(r for r in run.results if "Legged Robot" in r.paper.title)
    assert robot.features["avoid"] > 0 and robot.label == "SKIP"


def test_time_budget_caps_read(corpus):
    from triage.relevance import FEATURES

    scores = np.linspace(1, 0.5, len(corpus))
    F = np.zeros((len(corpus), len(FEATURES)))
    res = triage(corpus, scores, scores, F, Cutoffs(read=0.6, skim=0.3), max_read=2)
    assert sum(r.label == "READ" for r in res) == 2
    assert any("budget" in r.note for r in res)
    res = triage(corpus, scores, scores, F, Cutoffs(read=0.6, skim=0.3), user_labels={corpus[0].id: "SKIP"})
    assert res[0].label == "SKIP" and "your label" in res[0].note


def test_cutoffs_and_budget():
    c = Cutoffs(read=0.3, skim=0.6)  # swapped automatically
    assert (c.read, c.skim) == (0.6, 0.3)
    assert label_for(0.6, c) == "READ" and label_for(0.59, c) == "SKIM" and label_for(0.1, c) == "SKIP"
    assert read_budget(0) == 1 and read_budget(3) == 6


def test_build_examples_priority():
    labels = {"a": "READ"}
    fb = [
        {"paper_id": "a", "action": "not_useful"},  # hand label wins
        {"paper_id": "b", "action": "open"},
        {"paper_id": "b", "action": "not_useful"},  # explicit beats implicit
        {"paper_id": "c", "action": "useful"},
        {"paper_id": "c", "action": "correct", "value": "SKIM"},  # latest explicit wins
        {"paper_id": "d", "action": "dismiss"},
        {"paper_id": "e", "action": "explanation_ok"},  # ignored
    ]
    ex = {e.paper_id: e for e in build_examples(labels, fb)}
    assert ex["a"].target == 1.0 and ex["a"].source == "label"
    assert ex["b"].target == 0.0
    assert ex["c"].target == 0.5
    assert ex["d"].target < 0.3
    assert "e" not in ex
    assert "a" not in {e.paper_id for e in build_examples(labels, fb, exclude={"a"})}


def test_feedback_changes_ranking(store, engine, corpus, rag_profile):
    prof = _setup(engine, store, corpus, rag_profile)
    before = {r.paper.id: r.score for r in engine.run(prof, respect_budget=False).results}
    # User keeps saying they like the LM-ish middle papers and dislike some RAG ones.
    for p in corpus[5:8]:
        store.log_feedback(prof.id, p.id, "useful")
    for p in corpus[8:14]:
        store.log_feedback(prof.id, p.id, "not_useful")
    for p in corpus[:2]:
        store.set_label(prof.id, p.id, "READ")
    run = engine.run(prof, respect_budget=False)
    assert run.model.train_info["learned"]
    after = {r.paper.id: r.score for r in run.results}
    assert any(abs(after[k] - before[k]) > 1e-3 for k in after)


def test_feedback_affinity_leave_self_out(corpus, rag_profile):
    emb = TfidfEmbedder().fit([p.text for p in corpus])
    m = RelevanceModel(emb, build_profile_vectors(rag_profile, emb))
    V = emb.encode_papers(corpus)
    vecs = {p.id: V[i] for i, p in enumerate(corpus)}
    # Only one positive: its own affinity must not come from itself.
    m.fit(build_examples({}, [{"paper_id": corpus[10].id, "action": "useful"}]), {p.id: p for p in corpus}, vecs)
    F = m.features([corpus[10]], V[10:11])
    assert F[0, -1] == pytest.approx(0.0)


def test_evaluate_cv(store, engine, corpus, rag_profile):
    prof = _setup(engine, store, corpus, rag_profile)
    rep = engine.evaluate(prof)
    assert not rep.ok and "at least" in rep.message
    for p in corpus[:5]:
        store.set_label(prof.id, p.id, "READ")
    for p in corpus[5:8]:
        store.set_label(prof.id, p.id, "SKIM")
    for p in corpus[8:]:
        store.set_label(prof.id, p.id, "SKIP")
    rep = engine.evaluate(prof)
    assert rep.ok and rep.n_labels == len(corpus)
    names = [s.name for s in rep.systems]
    assert names[0].startswith("Personalised model")
    full = rep.systems[0]
    assert full.spearman > 0.5
    assert 0 <= full.at_current["macro_f1"] <= 1
    assert sum(map(sum, full.at_current["confusion"])) == len(corpus)
    assert full.at_best["macro_f1"] >= full.at_current["macro_f1"] - 1e-9


def test_metrics():
    t = ["READ", "SKIM", "SKIP", "SKIP"]
    m = label_metrics(t, t)
    assert m["accuracy"] == 1 and m["macro_f1"] == 1 and m["severe_errors"] == 0
    m = label_metrics(t, ["SKIP", "SKIM", "READ", "SKIP"])
    assert m["severe_errors"] == 2 and m["missed_reads"] == 1
    assert ndcg_at_k(np.array([0.9, 0.5, 0.1]), ["READ", "SKIM", "SKIP"]) == pytest.approx(1.0)
    assert ndcg_at_k(np.array([0.1, 0.5, 0.9]), ["READ", "SKIM", "SKIP"]) < 0.7


def test_best_cutoffs_finds_perfect_split():
    scores = np.array([0.9, 0.85, 0.6, 0.55, 0.2, 0.1])
    truth = ["READ", "READ", "SKIM", "SKIM", "SKIP", "SKIP"]
    c, m = best_cutoffs(scores, truth)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert 0.6 < c.read <= 0.85 and 0.2 < c.skim <= 0.55


def test_template_reason_is_grounded(corpus, rag_profile):
    from triage.models import TriageResult

    r = TriageResult(paper=corpus[0], score=0.8, label="READ", rank=1, features={"focus": 0.7, "feedback": 0})
    ex = explain(r, rag_profile)
    assert ex["source"] == "template" and ex["grounded"]
    assert "question answering" in ex["reason"] or "retrieval augmented generation" in ex["reason"]
    # Every quoted term in the reason appears in the paper or profile.
    ok, why = verify_llm_reason(ex["reason"], ["retrieval"], corpus[0], rag_profile)
    assert ok, why
    ev = {"keywords_in_title": [], "keywords_in_abstract": [], "shared_terms": [], "topic_terms": ["quadrotor"], "avoid_hits": [], "best_sentence": ""}
    assert template_reason("SKIP", ev) == "Little overlap with your interests."
    assert template_reason("SKIP", dict(ev, avoid_hits=["robot"])) == "Covers “robot”, which you excluded."
    read = template_reason("READ", dict(ev, keywords_in_title=["dense retrieval"]), {"focus": 0.8}, focus="x")
    assert read == "“dense retrieval” in the title; close to your current focus."
    assert len(read) < 90  # one short line


def test_llm_verifier_rejects_invented_overlap(corpus, rag_profile):
    p = corpus[0]
    assert not verify_llm_reason("Uses graph neural networks.", ["graph neural networks"], p, rag_profile)[0]
    assert not verify_llm_reason("Great paper", [], p, rag_profile)[0]
    assert not verify_llm_reason('It covers "protein folding" and dense retrieval', ["dense retrieval"], p, rag_profile)[0]
    assert verify_llm_reason("Directly about your RAG work: a “reranker” over dense retrieval.", ["dense retrieval", "reranker"], p, rag_profile)[0]


def test_llm_failure_falls_back(monkeypatch, corpus, rag_profile, store):
    import triage.explain as ex
    from triage.models import TriageResult

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(ex, "llm_reason", boom)
    r = TriageResult(paper=corpus[0], score=0.8, label="READ", rank=1, features={})
    out = ex.explain(r, rag_profile, use_llm=True, store=store)
    assert out["source"] == "template" and "LLM error" in out["warning"]

    monkeypatch.setattr(ex, "llm_reason", lambda *a, **k: {"reason": "Mentions quantum chemistry.", "cited_terms": ["quantum chemistry"]})
    out = ex.explain(r, rag_profile, use_llm=True, store=store)
    assert out["source"] == "template" and "rejected" in out["warning"]

    monkeypatch.setattr(ex, "llm_reason", lambda *a, **k: {"reason": "Builds a RAG pipeline with dense retrieval.", "cited_terms": ["dense retrieval"]})
    r2 = TriageResult(paper=corpus[0], score=0.8, label="SKIM", rank=1, features={})
    out = ex.explain(r2, rag_profile, use_llm=True, store=store)
    assert out["source"] == "llm"
    # cached: a failing LLM now still returns the cached answer
    monkeypatch.setattr(ex, "llm_reason", boom)
    assert ex.explain(r2, rag_profile, use_llm=True, store=store)["source"] == "llm"


def test_store_roundtrip(store, corpus, rag_profile):
    store.upsert_papers(corpus)
    prof = store.save_profile(rag_profile)
    assert store.get_profile("rag").keywords == rag_profile.keywords
    assert store.add_to_pool(prof.id, [corpus[0].id, corpus[1].id]) == 2
    assert store.add_to_pool(prof.id, [corpus[0].id]) == 0
    v0 = store.feedback_version(prof.id)
    store.log_feedback(prof.id, corpus[0].id, "dismiss")
    assert store.dismissed_ids(prof.id) == {corpus[0].id}
    store.log_feedback(prof.id, corpus[0].id, "undismiss")
    assert store.dismissed_ids(prof.id) == set()
    assert store.feedback_version(prof.id) != v0
    with pytest.raises(ValueError):
        store.log_feedback(prof.id, corpus[0].id, "correct", "MAYBE")
    with pytest.raises(ValueError):
        store.set_label(prof.id, corpus[0].id, "maybe")
    store.set_setting(prof.id, "cutoffs", {"read": 0.7, "skim": 0.3})
    assert store.get_setting(prof.id, "cutoffs")["read"] == 0.7
    store.put_embeddings({"x": np.ones(3)}, "m")
    assert store.get_embeddings(["x", "y"], "m")["x"].shape == (3,)
    store.delete_profile(prof.id)
    assert store.get_profile("rag") is None and store.pool_ids(prof.id) == []


def test_discovery_curve_math():
    from triage.evaluate import discovery_curve

    c = discovery_curve("m", np.array([0.9, 0.8, 0.7, 0.1]), ["READ", "SKIP", "READ", "SKIP"], {"READ"})
    assert c.found.tolist() == [0.5, 0.5, 1.0, 1.0]
    assert (c.first_good, c.to_half, c.to_80) == (1, 1, 3)
    assert c.hit_rate_at_10 == 1.0


def test_speed_report(store, engine, corpus, rag_profile):
    prof = _setup(engine, store, corpus, rag_profile)
    for p in corpus[:5]:
        store.set_label(prof.id, p.id, "READ")
    for p in corpus[5:8]:
        store.set_label(prof.id, p.id, "SKIM")
    for p in corpus[8:]:
        store.set_label(prof.id, p.id, "SKIP")
    store.log_feedback(prof.id, corpus[0].id, "useful")
    rep, speed = engine.evaluate_all(prof, good="read")
    assert rep.ok and speed.ok and speed.n_good == 5
    model = speed.curves[0]
    assert model.found[-1] == pytest.approx(1.0)
    # Better than random: finds half the good papers in fewer reviews than a random order would on average.
    assert model.to_half <= np.searchsorted(speed.random, 0.5) + 1
    assert speed.learning[0]["signals"] == 0 and speed.learning[-1]["signals"] > 0
    assert all(0 <= p["hit_rate_at_10"] <= 1 for p in speed.learning)
    _, loose = engine.evaluate_all(prof, good="read_skim")
    assert loose.n_good == 8


def test_seed_demo_is_marked_and_reproducible(store):
    from collections import Counter

    from triage.demo import DEMO_NAME, is_demo, seed_demo

    engine = TriageEngine(store, backend="tfidf")
    prof = seed_demo(engine)
    assert is_demo(engine, prof) and prof.name == DEMO_NAME
    labels = store.get_labels(prof.id)
    counts = Counter(labels.values())
    assert len(labels) == 60 and all(counts[l] > 0 for l in ("READ", "SKIM", "SKIP"))
    assert len(store.seed_ids(prof.id)) == 3 and not set(store.seed_ids(prof.id)) & set(labels)
    fb = store.get_feedback(prof.id)
    assert {"useful", "open"} <= {f["action"] for f in fb}
    events = store.get_label_events(prof.id)
    assert [e["created_at"] for e in events] == sorted(e["created_at"] for e in events)
    # Reset replaces the profile rather than duplicating it, with identical simulated labels.
    again = seed_demo(engine)
    assert [p.name for p in store.list_profiles()] == [DEMO_NAME]
    assert store.get_labels(again.id) == labels


def test_blend_selection_prefers_prior_when_learning_does_not_help():
    from triage.evaluate import select_blend

    truth = ["READ", "SKIP", "READ", "SKIP", "SKIM", "SKIP"]
    prior = np.array([0.9, 0.1, 0.8, 0.2, 0.5, 0.3])
    learned_bad = prior[::-1].copy()
    w, scores = select_blend(prior, learned_bad, truth, {"READ"})
    assert w == 0.0 and scores[0.0] == pytest.approx(1.0)
    learned_good = np.array([0.9, 0.0, 0.95, 0.0, 0.4, 0.0])
    prior_meh = np.array([0.5, 0.6, 0.55, 0.2, 0.5, 0.3])
    w, _ = select_blend(prior_meh, learned_good, truth, {"READ"})
    assert w > 0


def test_run_uses_validated_blend_and_groups(store, corpus, rag_profile):
    engine = TriageEngine(store, backend="tfidf")
    prof = _setup(engine, store, corpus, rag_profile)
    for p in corpus[:5]:
        store.set_label(prof.id, p.id, "READ")
    for p in corpus[8:]:
        store.set_label(prof.id, p.id, "SKIP")
    run = engine.run(prof)
    assert run.blend["validated"] and run.blend["weight"] in (0.0, 0.15, 0.3, 0.5, 0.7)
    assert run.model.learned_weight == run.blend["weight"]
    # cached: same key → same object content, no recomputation needed
    assert engine.blend(prof)["key"] == run.blend["key"]
    # every grouped paper points at a lead that exists and lists it back
    by_id = run.by_id
    for r in run.results:
        if r.group_lead:
            assert r.paper.id in by_id[r.group_lead].similar


def test_seed_papers_shape_ranking(store, corpus, rag_profile, monkeypatch):
    import triage.seeds as seeds_mod
    from triage.models import Paper

    engine = TriageEngine(store, backend="tfidf")
    prof = _setup(engine, store, corpus, InterestProfile(name="robots", keywords=["control"]))
    seed = Paper(id="arxiv:9999.00001", title="Quadruped locomotion with model predictive control",
                 abstract="Legged robots walk over rough terrain using model predictive control and reinforcement learning.")
    monkeypatch.setattr(seeds_mod, "resolve", lambda text="", bibtex="": ([seed], []))
    n, warns = engine.add_seed_papers(prof, "9999.00001")
    assert n == 1 and store.seed_ids(prof.id) == [seed.id]
    run = engine.run(prof, respect_budget=False)
    assert seed.id not in run.by_id  # seeds are not re-recommended
    top3 = [r.paper.title for r in run.results[:3]]
    assert any("Robot" in t or "Quadrotor" in t for t in top3)


def test_exports_and_digest(corpus):
    from triage import export
    from triage.models import TriageResult

    bib = export.to_bibtex(corpus[:2])
    assert bib.count("@misc{") == 2 and "eprint = {2609.00000}" in bib and "archiveprefix = {arXiv}" in bib
    ris = export.to_ris(corpus[:1])
    assert ris.startswith("TY  - GEN") and "ER  - " in ris
    res = [TriageResult(paper=p, score=0.9 - i / 20, label="READ" if i < 2 else "SKIM", rank=i + 1) for i, p in enumerate(corpus[:4])]
    md = export.digest_markdown("Me", res, {corpus[0].id: "Because."}, new_ids={p.id for p in corpus[:3]})
    assert "## Read" in md and "Because." in md and "3 new papers" in md


def test_new_since_and_fetch_log(store, corpus, rag_profile):
    engine = TriageEngine(store, backend="tfidf")
    prof = _setup(engine, store, corpus, rag_profile)
    assert engine.new_since(prof, None) == set()
    assert engine.new_since(prof, "2000-01-01T00:00:00+00:00") == {p.id for p in corpus}
    assert engine.new_since(prof, "2999-01-01T00:00:00+00:00") == set()
    store.log_fetch(prof.id, "arxiv_new", True, 5, 3)
    store.log_fetch(prof.id, "semantic_scholar", False, message="429")
    assert store.last_successful_fetch(prof.id) and store.fetch_history(prof.id)[0]["ok"] == 0
    # auto-fetch is a no-op without categories
    assert engine.refresh_if_due(prof) is None


def test_bootstrap_ci_brackets_estimate():
    from triage.evaluate import bootstrap_ci

    x = np.random.default_rng(1).normal(size=200)
    lo, hi = bootstrap_ci(lambda i: float(x[i].mean()), len(x))
    assert lo < x.mean() < hi and hi - lo < 0.5
