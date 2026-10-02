"""Paper and profile vectors for the Python reference engine.

TF-IDF (1–2 grams) ⊕ LSA, fitted on the current pool and profile text. The browser app uses
MiniLM sentence embeddings or TF-IDF (`web/js/engine.js`). Raw cosine similarity sits on a
different scale per embedder, so ``calibrate()`` maps cosine → [0, 1] with a fixed range, which
keeps the human-set READ/SKIM cutoffs meaningful.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .models import Paper


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x / n if n > 0 else x
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class Embedder:
    name = "base"
    # (unrelated, strongly related) cosine for calibration. "doc" = profile/paper
    # vs. paper; "short" = a keyword or one-line focus vs. paper (lower ceiling).
    cos_range = (0.0, 1.0)
    short_cos_range = (0.0, 1.0)

    def fit(self, texts: list[str]) -> "Embedder":
        return self

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def calibrate(self, cos: np.ndarray | float, kind: str = "doc") -> np.ndarray | float:
        lo, hi = self.short_cos_range if kind == "short" else self.cos_range
        return np.clip((np.asarray(cos) - lo) / (hi - lo), 0.0, 1.0)

    def encode_papers(self, papers: list["Paper"]) -> np.ndarray:
        if not papers:
            return np.zeros((0, 1), dtype=np.float32)
        return self.encode([p.text for p in papers])


class TfidfEmbedder(Embedder):
    """TF-IDF ⊕ LSA. cos(a,b) = ½·cos_tfidf + ½·cos_lsa."""

    name = "tfidf"
    cos_range = (0.02, 0.30)
    short_cos_range = (0.01, 0.25)

    def __init__(self, lsa_dims: int = 100):
        self.lsa_dims = lsa_dims
        self.vec = None
        self.svd = None

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [t for t in texts if t and t.strip()] or ["empty"]
        self.vec = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            min_df=1,
            max_df=0.9 if len(texts) > 10 else 1.0,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b",
        )
        X = self.vec.fit_transform(texts)
        k = min(self.lsa_dims, X.shape[0] - 1, X.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(X) if k >= 2 else None
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.vec is None:
            self.fit(texts)
        X = self.vec.transform(texts)
        tf = _l2(X.toarray())
        if self.svd is None:
            return tf
        lsa = _l2(self.svd.transform(X))
        return np.hstack([tf * np.sqrt(0.5), lsa * np.sqrt(0.5)]).astype(np.float32)
