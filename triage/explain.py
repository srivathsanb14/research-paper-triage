"""Explanation generator.

In: paper + interests + label → Out: a 1-line reason that only states verified evidence
(keyword hits in the title or abstract, shared distinctive terms, the abstract sentence
closest to the profile), so it cannot invent overlap. The browser app mirrors this in
`web/js/engine.js`; `tests/test_web_parity.py` keeps the two in step.
"""

from __future__ import annotations

import math
from collections import Counter

from .models import InterestProfile, Paper
from .preprocess import phrase_in_text, split_sentences, stem_tokens, tokenize
from .relevance import keyword_matches


# ---------------------------------------------------------------- evidence


def document_frequencies(papers: list[Paper]) -> tuple[Counter, int]:
    df: Counter = Counter()
    for p in papers:
        df.update(set(stem_tokens(f"{p.title} {p.abstract}")))
    return df, max(1, len(papers))


def extract_evidence(
    paper: Paper,
    profile: InterestProfile,
    df: Counter | None = None,
    n_docs: int = 1,
) -> dict:
    kw_title, kw_body = keyword_matches(paper, profile.keywords)
    paper_text = f"{paper.title} {paper.abstract}"
    avoid_hits = [a for a in profile.avoid if phrase_in_text(a, paper_text)]

    # Distinctive profile words that also occur in the paper (stem match), ranked by IDF.
    profile_words = {}
    for w in tokenize(f"{profile.description} {profile.focus}"):
        profile_words.setdefault(stem_tokens(w)[0] if stem_tokens(w) else w, w)
    paper_stems = set(stem_tokens(paper_text))
    kw_stems = {s for k in profile.keywords for s in stem_tokens(k)}
    shared = []
    for stem, word in profile_words.items():
        if stem in paper_stems and stem not in kw_stems:
            idf = math.log((1 + n_docs) / (1 + (df or {}).get(stem, 0))) + 1
            shared.append((idf, word))
    shared_terms = [w for _, w in sorted(shared, reverse=True)[:4]]

    # Paper's own distinctive topic words (what it is *about*), for SKIM/SKIP reasons.
    title_terms = []
    for w in tokenize(paper.title):
        s = stem_tokens(w)
        if s and s[0] not in kw_stems and w not in title_terms:
            idf = math.log((1 + n_docs) / (1 + (df or {}).get(s[0], 0))) + 1
            title_terms.append((idf, w))
    topic_terms = [w for _, w in sorted(title_terms, reverse=True)[:3]]

    # Abstract sentence with most profile overlap.
    want = set(stem_tokens(" ".join([profile.description, profile.focus, *profile.keywords])))
    best, best_score = "", 0.0
    for s in split_sentences(paper.abstract):
        toks = stem_tokens(s)
        if toks:
            sc = len(want.intersection(toks)) / math.sqrt(len(toks))
            if sc > best_score:
                best, best_score = s, sc

    return {
        "keywords_in_title": kw_title,
        "keywords_in_abstract": kw_body,
        "shared_terms": shared_terms,
        "topic_terms": topic_terms,
        "avoid_hits": avoid_hits,
        "best_sentence": best,
    }


def _q(items: list[str], n: int = 2) -> str:
    items = items[:n]
    return ", ".join(f"“{i}”" for i in items)


def template_reason(label: str, ev: dict, features: dict[str, float] | None = None, focus: str = "") -> str:
    """One short, grounded sentence (at most two clauses). The label chip already says READ/SKIM/SKIP."""
    f = features or {}
    kt, ka, shared = ev["keywords_in_title"], ev["keywords_in_abstract"], ev["shared_terms"]
    if ev["avoid_hits"]:
        return f"Covers {_q(ev['avoid_hits'], 1)}, which you excluded."
    bits: list[str] = []
    if kt:
        bits.append(f"{_q(kt)} in the title")
    elif ka:
        bits.append(f"mentions {_q(ka)}")
    elif shared and label != "SKIP":
        bits.append(f"shares {_q(shared)} with your project")
    if focus and f.get("focus", 0) >= 0.6 and label != "SKIP":
        bits.append("close to your current focus")
    if f.get("feedback", 0) >= 0.3:
        bits.append("like papers you found relevant")
    elif f.get("feedback", 0) <= -0.3:
        bits.append("like papers you found not relevant")
    bits = bits[:2]
    if not bits:
        bits = [{
            "READ": "strong match to your project description",
            "SKIM": "related to your project, but not central",
            "SKIP": "little overlap with your interests",
        }[label]]
    elif label == "SKIP" and not any(b.startswith("like papers") for b in bits):
        bits = ["little overlap beyond " + bits[0].replace("mentions ", "").replace(" in the title", "")]
    reason = "; ".join(bits) + "."
    return reason[0].upper() + reason[1:]
