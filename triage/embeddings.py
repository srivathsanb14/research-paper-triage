"""Paper / profile vectors.

Two interchangeable backends:

* ``sbert``  – sentence-transformers (all-MiniLM-L6-v2 by default). Best quality.
               Vectors are cached in SQLite, so each paper is encoded once.
* ``tfidf``  – TF-IDF (1–2 grams) ⊕ LSA, fitted on the current pool + profile
               text. Zero heavy deps; used automatically when torch is missing.

Raw cosine similarity lives on different scales per backend, so each backend
also exposes ``calibrate()`` which maps cosine → [0, 1] using a fixed range.
That keeps the human-set READ/SKIM cutoffs meaningful across backends.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from . import config

if TYPE_CHECKING:
    from .models import Paper
    from .store import Store

log = logging.getLogger(__name__)


FULL_TEXT_ABSTRACT_WEIGHT = 0.65


def _chunks(text: str, size: int = 900, limit: int = 8) -> list[str]:
    """~900-character chunks (fits a MiniLM 256-token window)."""
    words, out, cur = text.split(), [], []
    for w in words:
        cur.append(w)
        if sum(len(x) + 1 for x in cur) >= size:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out[:limit] or [text[:size]]


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
    needs_fit = False

    def fit(self, texts: list[str]) -> "Embedder":
        return self

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def calibrate(self, cos: np.ndarray | float, kind: str = "doc") -> np.ndarray | float:
        lo, hi = self.short_cos_range if kind == "short" else self.cos_range
        return np.clip((np.asarray(cos) - lo) / (hi - lo), 0.0, 1.0)

    def encode_papers(self, papers: list["Paper"], store: "Store | None" = None) -> np.ndarray:
        if not papers:
            return np.zeros((0, 1), dtype=np.float32)
        return self._with_full_text(papers, self.encode([p.text for p in papers]))

    def _with_full_text(self, papers: list["Paper"], V: np.ndarray) -> np.ndarray:
        """Blend in intro/conclusion text where it has been fetched (abstract stays dominant)."""
        idx = [i for i, p in enumerate(papers) if p.full_text]
        if not idx:
            return V
        V = V.copy()
        for i in idx:
            chunks = _chunks(papers[i].full_text)
            ft = _l2(self.encode(chunks).mean(axis=0))
            V[i] = _l2(FULL_TEXT_ABSTRACT_WEIGHT * V[i] + (1 - FULL_TEXT_ABSTRACT_WEIGHT) * ft)
        return V


class SbertEmbedder(Embedder):
    name = "sbert"
    cos_range = (0.15, 0.50)
    short_cos_range = (0.12, 0.45)

    def __init__(self, model_name: str = config.SBERT_MODEL):
        from sentence_transformers import SentenceTransformer  # heavy import, deferred

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.name = f"sbert:{model_name}"
        self._lock = threading.Lock()  # torch models are not re-entrant across threads

    def _dim(self) -> int:
        # Renamed in newer sentence-transformers; support both.
        fn = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return int(fn())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim()), dtype=np.float32)
        with self._lock:
            v = self.model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        return _l2(v)

    def encode_papers(self, papers, store=None):
        if not papers:
            return np.zeros((0, self._dim()), dtype=np.float32)
        out = np.zeros((len(papers), self._dim()), dtype=np.float32)
        # Papers with full text are cached under a separate key so enrichment invalidates correctly.
        for with_ft in (False, True):
            idx = [i for i, p in enumerate(papers) if bool(p.full_text) == with_ft]
            if not idx:
                continue
            key = self.name + ("+ft" if with_ft else "")
            group = [papers[i] for i in idx]
            cached = store.get_embeddings([p.id for p in group], key) if store else {}
            missing = [p for p in group if p.id not in cached]
            if missing:
                new = self.encode([p.text for p in missing])
                if with_ft:
                    new = self._with_full_text(missing, new)
                fresh = {p.id: v for p, v in zip(missing, new)}
                if store:
                    store.put_embeddings(fresh, key)
                cached.update(fresh)
            for i, p in zip(idx, group):
                out[i] = cached[p.id]
        return out


class TfidfEmbedder(Embedder):
    """TF-IDF ⊕ LSA. cos(a,b) = ½·cos_tfidf + ½·cos_lsa."""

    name = "tfidf"
    cos_range = (0.02, 0.30)
    short_cos_range = (0.01, 0.25)
    needs_fit = True

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


_cache: dict[str, Embedder] = {}
_cache_lock = threading.Lock()


def sbert_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except Exception:
        return False


def get_embedder(backend: str | None = None) -> Embedder:
    """Return a (shared, for sbert) embedder. TF-IDF instances are fresh because they are fitted per pool."""
    backend = (backend or config.EMBEDDING_BACKEND).lower()
    if backend == "auto":
        backend = "sbert" if sbert_available() else "tfidf"
    if backend == "sbert":
        with _cache_lock:
            if "sbert" not in _cache:
                try:
                    _cache["sbert"] = SbertEmbedder()
                except Exception as e:  # model download failure, torch issues, …
                    log.warning("sentence-transformers unavailable (%s); falling back to TF-IDF", e)
                    return TfidfEmbedder()
            return _cache["sbert"]
    if backend == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"unknown embedding backend {backend!r}")
