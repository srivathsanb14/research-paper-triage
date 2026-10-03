"""The browser (web/js) must agree with independent references: Python worth-it signals (triage/quality.py,
which builds the dataset) and scikit-learn's ridge regression.

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
from triage.models import Paper

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")

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


def test_browser_matches_references():
    ps = papers()
    rng = np.random.default_rng(0)
    n_features = 7  # the ranker's seven features
    X = rng.uniform(0, 1, (30, n_features))
    y = (X @ rng.normal(size=n_features) + rng.normal(scale=0.1, size=30)).clip(0, 1)
    w = rng.choice([0.3, 1.0, 1.5, 2.0], size=30)
    job = {"papers": [p.to_dict() for p in ps], "ridge": {"X": X.tolist(), "y": y.tolist(), "w": w.tolist()}}
    res = subprocess.run(["node", str(ROOT / "tests/js/parity.mjs")], input=json.dumps(job), capture_output=True, text=True, check=True, cwd=ROOT)
    js = json.loads(res.stdout)

    for p, s in zip(ps, js["signals"]):
        py = quality.signals(p)
        assert {k: s[k] for k in py.__dict__} == py.__dict__, p.id

    ref = Ridge(alpha=1.0).fit(X, y, sample_weight=w)
    np.testing.assert_allclose(js["ridge"]["coef"], ref.coef_, atol=1e-9)
    assert js["ridge"]["intercept"] == pytest.approx(ref.intercept_, abs=1e-9)
