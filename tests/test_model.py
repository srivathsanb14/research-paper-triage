import numpy as np
import pytest

from triage.embeddings import TfidfEmbedder
from triage.evaluate import best_cutoffs, label_metrics, ndcg_at_k
from triage.explain import extract_evidence, template_reason
from triage.profile import build_profile_vectors, parse_list
from triage.ranking import Cutoffs, label_for, read_budget, triage
from triage.relevance import RelevanceModel, build_examples


def test_parse_list():
    assert parse_list("a, b; c\nA\n\n d ") == ["a", "b", "c", "d"]


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


def test_feedback_affinity_leave_self_out(corpus, rag_profile):
    emb = TfidfEmbedder().fit([p.text for p in corpus])
    m = RelevanceModel(emb, build_profile_vectors(rag_profile, emb))
    V = emb.encode_papers(corpus)
    vecs = {p.id: V[i] for i, p in enumerate(corpus)}
    # Only one positive: its own affinity must not come from itself.
    m.fit(build_examples({}, [{"paper_id": corpus[10].id, "action": "useful"}]), {p.id: p for p in corpus}, vecs)
    F = m.features([corpus[10]], V[10:11])
    assert F[0, -1] == pytest.approx(0.0)


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


def test_discovery_curve_math():
    from triage.evaluate import discovery_curve

    c = discovery_curve("m", np.array([0.9, 0.8, 0.7, 0.1]), ["READ", "SKIP", "READ", "SKIP"], {"READ"})
    assert c.found.tolist() == [0.5, 0.5, 1.0, 1.0]
    assert (c.first_good, c.to_half, c.to_80) == (1, 1, 3)
    assert c.hit_rate_at_10 == 1.0


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


def test_bootstrap_ci_brackets_estimate():
    from triage.evaluate import bootstrap_ci

    x = np.random.default_rng(1).normal(size=200)
    lo, hi = bootstrap_ci(lambda i: float(x[i].mean()), len(x))
    assert lo < x.mean() < hi and hi - lo < 0.5


def test_template_reason_is_grounded(corpus, rag_profile):
    ev = extract_evidence(corpus[0], rag_profile)
    reason = template_reason("READ", ev, {"focus": 0.7, "feedback": 0}, rag_profile.focus)
    assert "question answering" in reason or "retrieval augmented generation" in reason
    # Every quoted term in the reason appears in the paper text.
    text = f"{corpus[0].title} {corpus[0].abstract}".lower().replace("-", " ")
    for term in __import__("re").findall(r"“([^”]+)”", reason):
        assert term.lower() in text
    ev = {"keywords_in_title": [], "keywords_in_abstract": [], "shared_terms": [], "topic_terms": ["quadrotor"], "avoid_hits": [], "best_sentence": ""}
    assert template_reason("SKIP", ev) == "Little overlap with your interests."
    assert template_reason("SKIP", dict(ev, avoid_hits=["robot"])) == "Covers “robot”, which you excluded."
    read = template_reason("READ", dict(ev, keywords_in_title=["dense retrieval"]), {"focus": 0.8}, focus="x")
    assert read == "“dense retrieval” in the title; close to your current focus."
    assert len(read) < 90  # one short line
