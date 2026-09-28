"""User-interest representation (CONFIGURED).

In: user context (project description, keywords, current focus, avoided topics)
Out: interest profile vectors in the same space as paper vectors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .embeddings import Embedder, _l2
from .models import InterestProfile

# Relative weights of the user-context fields in the main profile vector.
W_DESCRIPTION = 1.0
W_KEYWORDS = 1.0
W_FOCUS = 1.5  # "current focus" matters most for this week's triage
W_SEEDS = 1.5  # centroid of the user's seed papers (when given)


def parse_list(text: str) -> list[str]:
    """'a, b; c\\nd' -> ['a','b','c','d'] (deduped, order kept)."""
    items = [s.strip() for s in re.split(r"[,;\n]", text or "")]
    seen: set[str] = set()
    out = []
    for s in items:
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


@dataclass
class ProfileVectors:
    main: np.ndarray
    focus: np.ndarray | None
    keywords: np.ndarray  # (k, d), may be empty
    avoid: np.ndarray  # (a, d), may be empty
    keyword_names: list[str] = field(default_factory=list)
    avoid_names: list[str] = field(default_factory=list)


def profile_texts(profile: InterestProfile) -> list[str]:
    """All free text in the profile (used to fit the TF-IDF vocabulary)."""
    return [t for t in [profile.description, profile.focus, *profile.keywords, *profile.avoid] if t and t.strip()]


def build_profile_vectors(profile: InterestProfile, embedder: Embedder, seed_vecs: np.ndarray | None = None) -> ProfileVectors:
    if profile.is_empty() and (seed_vecs is None or not len(seed_vecs)):
        raise ValueError("The interest profile is empty — add a project description, keywords, or a focus.")
    texts: list[str] = []
    slots: dict[str, slice] = {}

    def add(name: str, items: list[str]) -> None:
        start = len(texts)
        texts.extend(items)
        slots[name] = slice(start, len(texts))

    add("desc", [profile.description] if profile.description.strip() else [])
    # (seed papers are embedded by the caller, in the paper space, and passed as seed_vecs)
    add("focus", [profile.focus] if profile.focus.strip() else [])
    add("kw", list(profile.keywords))
    add("avoid", list(profile.avoid))
    if texts:
        V = embedder.encode(texts)
    else:  # seeds only
        V = np.zeros((0, np.asarray(seed_vecs).shape[1]), dtype=np.float32)
    dim = V.shape[1]

    def rows(name: str) -> np.ndarray:
        return V[slots[name]] if slots[name].stop > slots[name].start else np.zeros((0, dim), dtype=np.float32)

    desc, focus, kw, avoid = rows("desc"), rows("focus"), rows("kw"), rows("avoid")
    parts, weights = [], []
    if len(desc):
        parts.append(desc[0]); weights.append(W_DESCRIPTION)
    if len(kw):
        parts.append(_l2(kw.mean(axis=0))); weights.append(W_KEYWORDS)
    if len(focus):
        parts.append(focus[0]); weights.append(W_FOCUS)
    if seed_vecs is not None and len(seed_vecs):
        parts.append(_l2(np.asarray(seed_vecs).mean(axis=0))); weights.append(W_SEEDS)
    main = _l2(np.average(np.vstack(parts), axis=0, weights=weights))
    return ProfileVectors(
        main=main,
        focus=focus[0] if len(focus) else None,
        keywords=kw,
        avoid=avoid,
        keyword_names=list(profile.keywords),
        avoid_names=list(profile.avoid),
    )
