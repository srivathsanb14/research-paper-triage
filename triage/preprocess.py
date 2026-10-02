"""Preprocessing: clean text → extract fields → (embeddings live in embeddings.py).

Paper metadata from APIs is noisy (LaTeX, HTML entities, hard line-wraps,
missing venues, duplicate records across sources). Everything here is pure and
defensive: it never raises on odd input.
"""

from __future__ import annotations

import html
import re
import unicodedata


from .models import Paper

_WS = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]+>")
_LATEX_CMD_WITH_ARG = re.compile(r"\\(?:textbf|textit|emph|mathrm|mathbf|mathcal|text|texttt|url)\{([^{}]*)\}")
_LATEX_ACCENT = re.compile(r"\\[\'\"`^~=.uvHc]\{?([A-Za-z])\}?")
_LATEX_SPECIAL = {r"{\l}": "ł", r"\l ": "ł", r"{\L}": "Ł", r"{\o}": "ø", r"{\O}": "Ø", r"{\ss}": "ß", r"{\ae}": "æ", r"{\i}": "i"}
_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?")
_MATH_DELIM = re.compile(r"\$+")
_BRACES = re.compile(r"[{}]")
_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?$", re.I)
_TOKEN = re.compile(r"[a-z][a-z0-9]+")  # hyphens split: "retrieval-augmented" → 2 tokens

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had has
    have having he her here hers herself him himself his how i if in into is it its itself just me more
    most my myself no nor not now of off on once only or other our ours ourselves out over own same she
    should so some such than that the their theirs them themselves then there these they this those
    through to too under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves via using use used based new novel approach approaches
    method methods paper propose proposed show shows results result present presents study work
    towards toward study studies however thus therefore while well within without across among many
    several various different one two three first second large small high low et al e g i e""".split()
)


def clean_text(text: str | None) -> str:
    """Strip HTML/LaTeX noise, normalise unicode and whitespace."""
    if not text:
        return ""
    t = html.unescape(str(text))
    t = _TAG.sub(" ", t)
    for k, v in _LATEX_SPECIAL.items():
        t = t.replace(k, v)
    t = _LATEX_ACCENT.sub(r"\1", t)
    t = _LATEX_CMD_WITH_ARG.sub(r"\1", t)
    t = _LATEX_CMD.sub(" ", t)
    t = _MATH_DELIM.sub("", t)
    t = _BRACES.sub("", t)
    t = unicodedata.normalize("NFKC", t)
    return _WS.sub(" ", t).strip()


def title_key(title: str) -> str:
    """Aggressive normalisation for cross-source de-duplication."""
    t = clean_text(title).lower()
    return re.sub(r"[^a-z0-9]+", "", t)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in STOPWORDS]


def _stem(word: str) -> str:
    """Tiny suffix stripper so 'models' matches 'model', 'learning' ≈ 'learn'."""
    for suf in ("ings", "ing", "ies", "es", "s", "ed"):
        if len(word) > len(suf) + 3 and word.endswith(suf):
            return word[: -len(suf)] + ("y" if suf == "ies" else "")
    return word


def stem_tokens(text: str) -> list[str]:
    return [_stem(t) for t in tokenize(text)]


def _all_stems(text: str) -> list[str]:
    """Stems of *all* tokens (stopwords kept) so word positions are preserved."""
    return [_stem(t) for t in _TOKEN.findall((text or "").lower())]


def phrase_in_text(phrase: str, text: str, max_gap: int = 1) -> bool:
    """True if the phrase's content words occur in `text` in order, each within
    `max_gap` intervening words of the previous one (stem match).

    "retrieval augmented generation" matches "retrieval-augmented generations";
    "LLM agents" matches "LLM-based agents"; but "question answering" does not
    match "questions about the answer key". Multi-word phrases (≥3 words) also
    match their acronym as a standalone token ("RAG").
    """
    p = stem_tokens(phrase)
    if not p:
        return False
    toks = _all_stems(text)
    if len(p) == 1:
        found = p[0] in toks
    else:
        found = False
        for i, t in enumerate(toks):
            if t != p[0]:
                continue
            pos, ok = i, True
            for w in p[1:]:
                window = toks[pos + 1 : pos + 2 + max_gap]
                if w not in window:
                    ok = False
                    break
                pos = pos + 1 + window.index(w)
            if ok:
                found = True
                break
    if found:
        return True
    words = _TOKEN.findall(phrase.lower())
    if len(words) >= 3:
        acronym = "".join(w[0] for w in words)
        return acronym in set(_TOKEN.findall((text or "").lower()))
    return False


def clean_paper(p: Paper) -> Paper:
    """Return a cleaned copy of a paper. Robust to missing fields."""
    authors = [clean_text(a) for a in (p.authors or []) if clean_text(a)]
    # Deduplicate authors preserving order.
    seen: set[str] = set()
    authors = [a for a in authors if not (a.lower() in seen or seen.add(a.lower()))]
    year = p.year
    if not year and p.published[:4].isdigit():
        year = int(p.published[:4])
    venue = clean_text(p.venue)
    if not venue and p.source == "arxiv":
        venue = "arXiv preprint"
    return Paper(
        id=p.id,
        title=clean_text(p.title) or "(untitled)",
        abstract=clean_text(p.abstract),
        authors=authors,
        venue=venue,
        year=year,
        published=(p.published or "")[:10],
        url=p.url or "",
        pdf_url=p.pdf_url or "",
        categories=list(dict.fromkeys(p.categories or [])),
        source=p.source,
        citation_count=p.citation_count,
        work_type=p.work_type,
        venue_type=p.venue_type,
        venue_core=p.venue_core,
        version=p.version,
        oa_status=p.oa_status,
        fwci=p.fwci,
        references_count=p.references_count,
        full_text=clean_text(p.full_text)[:20000],
        full_text_status=p.full_text_status,
    )


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text or "")
    return [s.strip() for s in parts if len(s.strip()) > 20]
