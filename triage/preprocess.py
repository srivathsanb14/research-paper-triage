"""Cleaning paper records from OpenAlex: LaTeX, HTML entities, hard line-wraps, missing fields.

Everything here is pure and defensive: it never raises on odd input.
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
