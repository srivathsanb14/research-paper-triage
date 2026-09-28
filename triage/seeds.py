"""Seed papers: papers the user already considers essential.

They fix the cold start — a handful of real papers describes a research
interest far better than keywords. Input can be arXiv IDs / URLs, DOIs, or a
BibTeX export (Zotero, Google Scholar, Overleaf). Seeds become strong positive
examples and pull the interest profile towards them; they are not shown in the
reading list (you already know them).
"""

from __future__ import annotations

import re
import time

import requests

from . import config
from .models import Paper
from .preprocess import clean_paper, clean_text
from .sources import OAI_URL, SourceError, _get_with_retry, parse_arxiv_oai

S2_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"

ARXIV_NEW = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(v\d+)?(?![\d])")
ARXIV_OLD = re.compile(r"\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?\b")
DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>{}]+[^\s\"'<>{}.,;])", re.I)
CROSSREF_URL = "https://api.crossref.org/works/{doi}"


def parse_identifiers(text: str) -> tuple[list[str], list[str]]:
    """Pull arXiv IDs and DOIs out of free text (one per line, URLs, citations…)."""
    arxiv, dois = [], []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        d = DOI.search(line)
        # arXiv DOIs (10.48550/arXiv.2401.01234) are really arXiv IDs.
        if d and "arxiv" in d.group(1).lower():
            m = re.search(r"arxiv\.(\d{4}\.\d{4,5})", d.group(1), re.I)
            if m:
                arxiv.append(m.group(1))
                continue
        if d:
            dois.append(d.group(1))
            continue
        m = ARXIV_NEW.search(line) or ARXIV_OLD.search(line)
        if m:
            arxiv.append(m.group(1))
    return list(dict.fromkeys(arxiv)), list(dict.fromkeys(dois))


def parse_bibtex(text: str) -> list[dict[str, str]]:
    """Minimal BibTeX reader (no dependency): returns one dict of fields per entry."""
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]*)\s*,", text or ""):
        if m.group(1).lower() in ("comment", "string", "preamble"):
            continue
        i, depth, fields_start = m.end(), 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body = text[fields_start : i - 1]
        fields: dict[str, str] = {}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            j = fm.end()
            if j >= len(body):
                break
            if body[j] == "{":
                d, k = 1, j + 1
                while k < len(body) and d:
                    d += {"{": 1, "}": -1}.get(body[k], 0)
                    k += 1
                val = body[j + 1 : k - 1]
            elif body[j] == '"':
                k = body.find('"', j + 1)
                val = body[j + 1 : k if k > 0 else len(body)]
            else:
                val = re.match(r"[^,\n]*", body[j:]).group(0)
            fields.setdefault(fm.group(1).lower(), val.strip())
        entries.append(fields)
    return entries


def _paper_from_bib(f: dict[str, str]) -> Paper | None:
    title = clean_text(f.get("title", ""))
    if not title:
        return None
    authors = [clean_text(a) for a in re.split(r"\s+and\s+", f.get("author", "")) if a.strip()]
    # "Last, First" → "First Last"
    authors = [" ".join(reversed([x.strip() for x in a.split(",", 1)])) if "," in a else a for a in authors]
    year = f.get("year", "")
    key = re.sub(r"[^a-z0-9]+", "", title.lower())[:40]
    return Paper(
        id=f"bib:{key}",
        title=title,
        abstract=f.get("abstract", ""),
        authors=authors,
        venue=clean_text(f.get("journal") or f.get("booktitle") or ""),
        year=int(year) if year[:4].isdigit() else None,
        url=f.get("url", ""),
        source="bibtex",
    )


def fetch_arxiv_records(ids: list[str], session: requests.Session | None = None) -> list[Paper]:
    """Look up arXiv papers by id via OAI-PMH GetRecord (works where the search API is blocked)."""
    out = []
    for i, aid in enumerate(ids):
        if i:
            time.sleep(1)
        r = _get_with_retry(
            OAI_URL,
            {"verb": "GetRecord", "identifier": f"oai:arXiv.org:{aid}", "metadataPrefix": "arXiv"},
            retries=3,
            session=session,
        )
        papers, _ = parse_arxiv_oai(r.content)  # bytes: OAI omits the charset header
        out += papers
    return out


def fetch_doi(doi: str, session: requests.Session | None = None) -> Paper | None:
    """Look up a DOI on Crossref (abstracts are included when the publisher deposits them)."""
    r = _get_with_retry(CROSSREF_URL.format(doi=doi), {}, retries=3, session=session)
    msg = r.json().get("message", {})
    title = clean_text(" ".join(msg.get("title") or []))
    if not title:
        return None
    authors = [" ".join(x for x in (a.get("given"), a.get("family")) if x) for a in msg.get("author", [])]
    year = (msg.get("issued", {}).get("date-parts") or [[None]])[0][0]
    return Paper(
        id=f"doi:{doi.lower()}",
        title=title,
        abstract=clean_text(msg.get("abstract", "")),
        authors=[a for a in authors if a],
        venue=clean_text(" ".join(msg.get("container-title") or [])),
        year=year,
        url=f"https://doi.org/{doi}",
        source="crossref",
    )


def _enrich_from_s2(p: Paper, doi: str) -> Paper | None:
    """Crossref often lacks abstracts; Semantic Scholar (or the arXiv version) usually has one."""
    headers = {"x-api-key": config.S2_API_KEY} if config.S2_API_KEY else {}
    try:
        r = _get_with_retry(S2_PAPER_URL.format(doi=doi), {"fields": "abstract,externalIds"}, headers=headers, retries=2)
        data = r.json()
    except (SourceError, ValueError):
        return None
    arxiv_id = (data.get("externalIds") or {}).get("ArXiv")
    if arxiv_id:
        try:
            got = fetch_arxiv_records([arxiv_id])
            if got:
                return got[0]
        except SourceError:
            pass
    if data.get("abstract"):
        p.abstract = data["abstract"]
        return p
    return None


def resolve(text: str = "", bibtex: str = "") -> tuple[list[Paper], list[str]]:
    """Turn user input into Paper records. Returns (papers, warnings)."""
    papers: list[Paper] = []
    warnings: list[str] = []
    arxiv, dois = parse_identifiers(text)
    for f in parse_bibtex(bibtex):
        eprint = f.get("eprint") or ""
        m = ARXIV_NEW.search(eprint) or ARXIV_NEW.search(f.get("url", "")) or ARXIV_NEW.search(f.get("doi", ""))
        if m and m.group(1) not in arxiv:
            arxiv.append(m.group(1))
        elif f.get("doi") and f["doi"] not in dois and not f.get("abstract"):
            dois.append(f["doi"])
        else:
            p = _paper_from_bib(f)
            if p:
                papers.append(p)
    if arxiv:
        try:
            got = fetch_arxiv_records(arxiv)
            papers += got
            missing = set(arxiv) - {p.id.split(":", 1)[1] for p in got}
            if missing:
                warnings.append(f"arXiv ids not found: {', '.join(sorted(missing))}")
        except SourceError as e:
            warnings.append(f"arXiv lookup failed: {e}")
    for doi in dois:
        try:
            p = fetch_doi(doi)
            if p and len(p.abstract) < 40:
                p = _enrich_from_s2(p, doi) or p
            if p:
                papers.append(p)
            else:
                warnings.append(f"DOI not found: {doi}")
        except (SourceError, ValueError) as e:
            warnings.append(f"DOI {doi}: {e}")
        time.sleep(0.5)
    cleaned = [clean_paper(p) for p in papers]
    thin = [p.title for p in cleaned if len(p.abstract) < 40]
    if thin:
        warnings.append(
            f"{len(thin)} paper(s) have no abstract, so only their titles inform your profile: "
            + "; ".join(t[:60] for t in thin[:3])
        )
    return cleaned, warnings
