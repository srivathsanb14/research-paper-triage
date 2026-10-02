"""The browser engine (web/js) must agree with the Python reference (triage/).

Runs tests/js/parity.mjs under Node with identical inputs; skipped without Node.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from triage import quality
from triage.explain import document_frequencies, extract_evidence, template_reason
from triage.models import InterestProfile, Paper
from triage.preprocess import phrase_in_text
from triage.relevance import FEATURES

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")

PHRASES = [
    ("retrieval augmented generation", "We study retrieval-augmented generations for QA."),
    ("retrieval augmented generation", "A RAG pipeline for science."),
    ("LLM agents", "LLM-based agents that browse the web"),
    ("question answering", "questions about the answer key"),
    ("graph neural networks", "Message passing on graphs with neural nets"),
    ("climate", "Climatic change in coastal regions"),
]


def papers() -> list[Paper]:
    return [
        Paper("doi:1", "A Survey of Retrieval-Augmented Generation", "We review retrieval augmented generation. Code is available at github.com/x/y. "
              "We evaluate on five benchmarks with ablations. " * 3, venue="ACM Computing Surveys", venue_type="journal",
              work_type="review", venue_core=True, version="publishedVersion", oa_status="gold", fwci=3.2, references_count=120),
        Paper("arxiv:2", "Groundbreaking unprecedented agents", "A short abstract about agents.", source="arxiv"),
        Paper("doi:3", "Household savings and nudges", "A randomized controlled trial with n = 12,000 participants. Data are publicly available on Zenodo. "
              "We preregistered hypotheses about savings behaviour among households. " * 2, venue="J. Econ", venue_type="journal",
              references_count=5, oa_status="closed"),
        Paper("doi:4", "Medieval manuscripts in the digital age", "Archives and manuscripts " * 40, venue_type="repository", work_type="article"),
    ]


def test_browser_engine_matches_python():
    ps = papers()
    profile = InterestProfile(name="t", description="I evaluate retrieval augmented generation and household savings",
                              keywords=["retrieval augmented generation", "savings", "agents"], focus="faithfulness of answers")
    labels = ["READ", "SKIP", "SKIM", "SKIP"]
    rng = np.random.default_rng(0)
    feats = rng.uniform(0, 1, (len(ps), len(FEATURES))).round(3)
    X = rng.uniform(0, 1, (30, len(FEATURES)))
    y = (X @ rng.normal(size=len(FEATURES)) + rng.normal(scale=0.1, size=30)).clip(0, 1)
    w = rng.choice([0.3, 1.0, 1.5, 2.0], size=30)
    job = {
        "papers": [p.to_dict() for p in ps], "profile": {**profile.__dict__}, "labels": labels, "features": feats.tolist(),
        "phrases": PHRASES, "ridge": {"X": X.tolist(), "y": y.tolist(), "w": w.tolist()},
    }
    res = subprocess.run(["node", str(ROOT / "tests/js/parity.mjs")], input=json.dumps(job), capture_output=True, text=True, check=True, cwd=ROOT)
    js = json.loads(res.stdout)

    assert js["phrases"] == [phrase_in_text(a, b) for a, b in PHRASES]

    for p, s in zip(ps, js["signals"]):
        py = quality.signals(p)
        assert {k: s[k] for k in py.__dict__} == py.__dict__, p.id

    df, n = document_frequencies(ps)
    fdict = [dict(zip(FEATURES, row)) for row in feats]
    py_reasons = [template_reason(lab, extract_evidence(p, profile, df, n), f, profile.focus) for p, lab, f in zip(ps, labels, fdict)]
    assert js["reasons"] == py_reasons

    ref = Ridge(alpha=1.0).fit(X, y, sample_weight=w)
    np.testing.assert_allclose(js["ridge"]["coef"], ref.coef_, atol=1e-9)
    assert js["ridge"]["intercept"] == pytest.approx(ref.intercept_, abs=1e-9)
