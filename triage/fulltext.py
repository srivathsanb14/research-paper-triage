"""Full text for borderline papers.

Abstracts often undersell (or oversell) how relevant a paper is. For papers whose
score sits near a Read/Skim cutoff, we fetch the introduction and conclusion —
from arXiv's HTML rendering when available (clean section structure), otherwise
from the PDF — and fold them into the paper's vector and keyword matching.
"""

from __future__ import annotations

import io
import logging
import re
import time
from html.parser import HTMLParser

from . import config
from .models import Paper
from .preprocess import clean_text
from .sources import SourceError, _get_with_retry

log = logging.getLogger(__name__)

MAX_SECTION_CHARS = 3500
INTRO = re.compile(r"introduction|background|motivation", re.I)
OUTRO = re.compile(r"conclusion|discussion|summary|limitations", re.I)


class _SectionParser(HTMLParser):
    """Collects text per top-level section of an arXiv (LaTeXML) HTML paper."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[tuple[str, list[str]]] = []
        self._in_heading = False
        self._heading: list[str] = []
        self._skip = 0  # inside <math>, <script>, <style>, <figure>, <table>

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class", "") or ""
        if tag in ("math", "script", "style", "figure", "table", "nav", "footer"):
            self._skip += 1
        elif tag == "h2" and "ltx_title_section" in cls:
            self._in_heading, self._heading = True, []

    def handle_endtag(self, tag):
        if tag in ("math", "script", "style", "figure", "table", "nav", "footer"):
            self._skip = max(0, self._skip - 1)
        elif tag == "h2" and self._in_heading:
            self._in_heading = False
            self.sections.append((" ".join(self._heading), []))

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_heading:
            self._heading.append(data)
        elif self.sections:
            self.sections[-1][1].append(data)


def extract_from_html(html_text: str) -> str:
    p = _SectionParser()
    p.feed(html_text)
    picked = []
    for pattern in (INTRO, OUTRO):
        for heading, chunks in p.sections:
            if pattern.search(heading):
                picked.append(clean_text(" ".join(chunks))[:MAX_SECTION_CHARS])
                break
    return " ".join(x for x in picked if x)


def extract_from_pdf(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [pg.extract_text() or "" for pg in reader.pages[: min(len(reader.pages), 12)]]
    text = clean_text(" ".join(pages))
    out = []
    m = re.search(r"\b1\.?\s+Introduction\b|\bIntroduction\b", text)
    if m:
        out.append(text[m.start() : m.start() + MAX_SECTION_CHARS])
    m = re.search(r"\b\d?\.?\s*(Conclusions?|Discussion)\b(?!.*\b(Conclusions?)\b)", text)
    if m:
        out.append(text[m.start() : m.start() + MAX_SECTION_CHARS])
    return " ".join(out) or text[:MAX_SECTION_CHARS]


def fetch_full_text(paper: Paper) -> str:
    """Introduction + conclusion text, or "" if unavailable. Only arXiv papers are supported."""
    if not paper.id.startswith("arxiv:"):
        return ""
    aid = paper.id.split(":", 1)[1]
    try:
        r = _get_with_retry(f"https://arxiv.org/html/{aid}", {}, retries=2)
        text = extract_from_html(r.text)
        if len(text) > 500:
            return text
    except SourceError:
        pass  # no HTML rendering for this paper — try the PDF
    try:
        r = _get_with_retry(paper.pdf_url or f"https://arxiv.org/pdf/{aid}", {}, retries=2)
        return extract_from_pdf(r.content)
    except (SourceError, Exception) as e:  # pypdf raises assorted errors on odd PDFs
        log.info("no full text for %s: %s", paper.id, e)
        return ""


def enrich(papers: list[Paper], delay: float = 1.0) -> list[Paper]:
    """Fetch full text for each paper (rate-limited). Marks attempts so they aren't retried."""
    out = []
    for i, p in enumerate(papers):
        if i:
            time.sleep(delay)
        text = fetch_full_text(p)
        p.full_text = text
        p.full_text_status = "ok" if text else "unavailable"
        out.append(p)
    return out
