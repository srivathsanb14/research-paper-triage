"""External paper sources: arXiv API, Semantic Scholar Graph API, bundled sample set.

Both APIs are rate limited and occasionally flaky, so every call goes through
`_get_with_retry` (exponential backoff honouring Retry-After). Failures raise
`SourceError` with a human-readable message that the UI can display.
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Callable

import requests

from . import config
from .models import Paper
from .preprocess import clean_paper, dedupe, is_usable, normalize_arxiv_id

log = logging.getLogger(__name__)

ARXIV_URL = "https://export.arxiv.org/api/query"
ARXIV_RSS_URL = "https://rss.arxiv.org/atom/{cats}"
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,authors,venue,year,url,externalIds,citationCount,openAccessPdf,publicationDate"
ATOM = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "dc": "http://purl.org/dc/elements/1.1/",
}

# Common arXiv categories offered in the UI.
ARXIV_CATEGORIES = {
    "cs.AI": "Artificial Intelligence",
    "cs.CL": "Computation & Language (NLP)",
    "cs.CV": "Computer Vision",
    "cs.LG": "Machine Learning",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.RO": "Robotics",
    "cs.CR": "Cryptography & Security",
    "cs.SE": "Software Engineering",
    "cs.DB": "Databases",
    "cs.DC": "Distributed Computing",
    "cs.CY": "Computers & Society",
    "stat.ML": "Statistics — ML",
    "q-bio.QM": "Quantitative Methods (bio)",
    "eess.AS": "Audio & Speech",
}


class SourceError(RuntimeError):
    """A paper source failed in a way the user should be told about."""


_last_arxiv_call = 0.0


def _get_with_retry(
    url: str,
    params: dict,
    headers: dict | None = None,
    retries: int = 4,
    backoff: float = 2.0,
    session: requests.Session | None = None,
) -> requests.Response:
    s = session or requests
    hdrs = {"User-Agent": config.USER_AGENT, "Accept-Encoding": "gzip", **(headers or {})}
    last_err: str = ""
    for attempt in range(retries):
        try:
            r = s.get(url, params=params, headers=hdrs, timeout=config.HTTP_TIMEOUT_S)
        except requests.RequestException as e:
            last_err = f"network error: {e.__class__.__name__}"
        else:
            if r.status_code == 200:
                return r
            # arXiv signals throttling with 406/429 (and flakes with 5xx) — all transient.
            if r.status_code in (406, 429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}"
                ra = r.headers.get("Retry-After")
                if ra and ra.isdigit():
                    time.sleep(min(int(ra), 30))
                    continue
            else:
                raise SourceError(f"{url} returned HTTP {r.status_code}: {r.text[:200]}")
        if attempt < retries - 1:
            time.sleep(backoff * (2**attempt))
    raise SourceError(f"{url} failed after {retries} attempts ({last_err})")


# --------------------------------------------------------------------------- arXiv


def build_arxiv_query(keywords: list[str], categories: list[str] | None = None) -> str:
    """(all:"kw1" OR all:"kw2") AND (cat:cs.LG OR cat:cs.CL)."""
    terms = []
    for kw in keywords:
        kw = kw.strip().replace('"', "")
        if not kw:
            continue
        terms.append(f'all:"{kw}"' if " " in kw else f"all:{kw}")
    q = " OR ".join(terms)
    if categories:
        cats = " OR ".join(f"cat:{c}" for c in categories)
        q = f"({q}) AND ({cats})" if q else f"({cats})"
    return q


def parse_arxiv_feed(xml_text: str) -> list[Paper]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise SourceError(f"arXiv returned malformed XML: {e}") from e
    papers: list[Paper] = []
    for entry in root.findall("a:entry", ATOM):
        raw_id = entry.findtext("a:id", default="", namespaces=ATOM)
        if "api/errors" in raw_id:  # arXiv reports query errors as a pseudo-entry
            raise SourceError("arXiv rejected the query: " + (entry.findtext("a:summary", "", ATOM) or "").strip())
        aid = normalize_arxiv_id(raw_id)
        pdf = ""
        for link in entry.findall("a:link", ATOM):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf = link.get("href", "")
        journal = entry.findtext("arxiv:journal_ref", default="", namespaces=ATOM)
        papers.append(
            Paper(
                id=f"arxiv:{aid}",
                title=entry.findtext("a:title", default="", namespaces=ATOM),
                abstract=entry.findtext("a:summary", default="", namespaces=ATOM),
                authors=[a.findtext("a:name", default="", namespaces=ATOM) for a in entry.findall("a:author", ATOM)],
                venue=journal,
                published=entry.findtext("a:published", default="", namespaces=ATOM),
                url=f"https://arxiv.org/abs/{aid}",
                pdf_url=pdf or f"https://arxiv.org/pdf/{aid}",
                categories=[c.get("term", "") for c in entry.findall("a:category", ATOM) if c.get("term")],
                source="arxiv",
            )
        )
    return papers


def search_arxiv(
    keywords: list[str],
    categories: list[str] | None = None,
    max_results: int = 100,
    days_back: int | None = None,
    sort_by: str = "submittedDate",
    session: requests.Session | None = None,
) -> list[Paper]:
    global _last_arxiv_call
    query = build_arxiv_query(keywords, categories)
    if not query:
        raise SourceError("Add at least one keyword or category to search arXiv.")
    if days_back:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days_back)
        query = f"({query}) AND submittedDate:[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"
    papers: list[Paper] = []
    page = min(max_results, 100)
    for start_idx in range(0, max_results, page):
        # arXiv asks clients to wait ~3 s between requests.
        wait = 3.0 - (time.time() - _last_arxiv_call)
        if wait > 0:
            time.sleep(wait)
        try:
            r = _get_with_retry(
                ARXIV_URL,
                {
                    "search_query": query,
                    "start": start_idx,
                    "max_results": min(page, max_results - start_idx),
                    "sortBy": sort_by,
                    "sortOrder": "descending",
                },
                retries=3,
                backoff=4.0,
                session=session,
            )
        except SourceError as e:
            if "406" in str(e):
                raise SourceError(
                    "the arXiv search API is refusing requests from this client right now (HTTP 406). "
                    "Use 'arXiv new listings' (RSS) or Semantic Scholar instead, or retry later."
                ) from e
            raise
        _last_arxiv_call = time.time()
        batch = parse_arxiv_feed(r.text)
        papers.extend(batch)
        if len(batch) < page:
            break
    return _finalize(papers)


# ------------------------------------------------------------ arXiv new listings


def parse_arxiv_rss(xml_text: str, announce_types: tuple[str, ...] = ("new", "cross")) -> list[Paper]:
    """Parse rss.arxiv.org Atom feeds (today's announcements per category)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise SourceError(f"arXiv RSS returned malformed XML: {e}") from e
    papers: list[Paper] = []
    for entry in root.findall("a:entry", ATOM):
        atype = (entry.findtext("arxiv:announce_type", default="new", namespaces=ATOM) or "new").strip()
        if announce_types and atype not in announce_types:
            continue
        aid = normalize_arxiv_id(entry.findtext("a:id", default="", namespaces=ATOM).replace("oai:arXiv.org:", ""))
        summary = entry.findtext("a:summary", default="", namespaces=ATOM) or ""
        # "arXiv:2609.30287v1 Announce Type: new \nAbstract: …"
        if "Abstract:" in summary:
            summary = summary.split("Abstract:", 1)[1]
        creators = entry.findtext("dc:creator", default="", namespaces=ATOM) or ""
        authors = [a.strip() for a in creators.replace(" and ", ", ").split(",") if a.strip()]
        papers.append(
            Paper(
                id=f"arxiv:{aid}",
                title=entry.findtext("a:title", default="", namespaces=ATOM),
                abstract=summary,
                authors=authors,
                published=(entry.findtext("a:published", default="", namespaces=ATOM) or "")[:10],
                url=f"https://arxiv.org/abs/{aid}",
                pdf_url=f"https://arxiv.org/pdf/{aid}",
                categories=[c.get("term", "") for c in entry.findall("a:category", ATOM) if c.get("term")],
                source="arxiv",
            )
        )
    return papers


def fetch_arxiv_new(
    categories: list[str],
    include_replacements: bool = False,
    session: requests.Session | None = None,
) -> list[Paper]:
    """Today's new (and cross-listed) arXiv papers in the given categories."""
    if not categories:
        raise SourceError("Pick at least one arXiv category for new listings.")
    types = ("new", "cross", "replace", "replace-cross") if include_replacements else ("new", "cross")
    r = _get_with_retry(ARXIV_RSS_URL.format(cats="+".join(categories)), {}, retries=3, session=session)
    papers = parse_arxiv_rss(r.text, types)
    if not papers:
        raise SourceError(
            "arXiv has no new announcements in these categories right now "
            "(feeds are empty on weekends/holidays)."
        )
    return _finalize(papers)


# ------------------------------------------------------------ arXiv OAI-PMH (bulk)

OAI_URL = "https://oaipmh.arxiv.org/oai"
OAI_NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "ax": "http://arxiv.org/OAI/arXiv/"}


def parse_arxiv_oai(xml_text: str | bytes) -> tuple[list[Paper], str | None]:
    """Parse one ListRecords page (metadataPrefix=arXiv). Returns (papers, resumption token)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise SourceError(f"arXiv OAI returned malformed XML: {e}") from e
    papers = []
    for rec in root.iter(f"{{{OAI_NS['oai']}}}record"):
        md = rec.find("oai:metadata/ax:arXiv", OAI_NS)
        if md is None:
            continue
        aid = (md.findtext("ax:id", default="", namespaces=OAI_NS) or "").strip()
        authors = []
        for a in md.findall("ax:authors/ax:author", OAI_NS):
            name = " ".join(
                x for x in (a.findtext("ax:forenames", "", OAI_NS), a.findtext("ax:keyname", "", OAI_NS)) if x
            )
            if name:
                authors.append(name)
        created = md.findtext("ax:created", default="", namespaces=OAI_NS) or ""
        papers.append(
            Paper(
                id=f"arxiv:{aid}",
                title=md.findtext("ax:title", default="", namespaces=OAI_NS),
                abstract=md.findtext("ax:abstract", default="", namespaces=OAI_NS),
                authors=authors,
                venue=md.findtext("ax:journal-ref", default="", namespaces=OAI_NS) or "",
                published=created[:10],
                url=f"https://arxiv.org/abs/{aid}",
                pdf_url=f"https://arxiv.org/pdf/{aid}",
                categories=(md.findtext("ax:categories", default="", namespaces=OAI_NS) or "").split(),
                source="arxiv",
            )
        )
    token_el = root.find(".//oai:resumptionToken", OAI_NS)
    token = token_el.text.strip() if token_el is not None and token_el.text else None
    return papers, token


def fetch_arxiv_oai(
    since: str,
    categories: list[str] | None = None,
    oai_set: str = "cs",
    max_pages: int = 3,
    created_since: str | None = None,
    session: requests.Session | None = None,
) -> list[Paper]:
    """Bulk-harvest recent arXiv metadata (≈1,300 records per page) and filter by category.

    OAI also returns old papers whose metadata was updated; `created_since`
    (YYYY-MM-DD) keeps only papers first submitted on or after that date.
    """
    params: dict = {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": oai_set, "from": since}
    out: list[Paper] = []
    for page in range(max_pages):
        r = _get_with_retry(OAI_URL, params, retries=4, backoff=5.0, session=session)
        papers, token = parse_arxiv_oai(r.content)  # bytes: OAI omits the charset header
        out += papers
        if not token:
            break
        params = {"verb": "ListRecords", "resumptionToken": token}
        time.sleep(3)
    if categories:
        wanted = set(categories)
        out = [p for p in out if wanted.intersection(p.categories)]
    if created_since:
        out = [p for p in out if p.published >= created_since]
    return _finalize(out)


# --------------------------------------------------------------- Semantic Scholar


def parse_s2_item(item: dict) -> Paper | None:
    if not item or not item.get("title"):
        return None
    ext = item.get("externalIds") or {}
    arxiv = ext.get("ArXiv")
    pid = f"arxiv:{arxiv}" if arxiv else f"s2:{item.get('paperId')}"
    oa = item.get("openAccessPdf") or {}
    return Paper(
        id=pid,
        title=item.get("title") or "",
        abstract=item.get("abstract") or "",
        authors=[a.get("name", "") for a in (item.get("authors") or [])],
        venue=item.get("venue") or "",
        year=item.get("year"),
        published=item.get("publicationDate") or "",
        url=f"https://arxiv.org/abs/{arxiv}" if arxiv else (item.get("url") or ""),
        pdf_url=oa.get("url") or "",
        source="semantic_scholar",
        citation_count=item.get("citationCount"),
    )


def search_semantic_scholar(
    keywords: list[str],
    max_results: int = 100,
    year_from: int | None = None,
    session: requests.Session | None = None,
) -> list[Paper]:
    query = " ".join(k.strip() for k in keywords if k.strip())
    if not query:
        raise SourceError("Add at least one keyword to search Semantic Scholar.")
    headers = {"x-api-key": config.S2_API_KEY} if config.S2_API_KEY else {}
    papers: list[Paper] = []
    limit = min(max_results, 100)
    for offset in range(0, max_results, limit):
        params = {"query": query, "limit": min(limit, max_results - offset), "offset": offset, "fields": S2_FIELDS}
        if year_from:
            params["year"] = f"{year_from}-"
        try:
            r = _get_with_retry(S2_SEARCH_URL, params, headers=headers, retries=5, backoff=3.0, session=session)
        except SourceError as e:
            if "429" in str(e) and not config.S2_API_KEY:
                raise SourceError(
                    "Semantic Scholar is rate-limiting anonymous requests. Try again in a minute, or set "
                    "S2_API_KEY (free key: semanticscholar.org/product/api)."
                ) from e
            raise
        try:
            data = r.json()
        except json.JSONDecodeError as e:
            raise SourceError("Semantic Scholar returned invalid JSON") from e
        items = data.get("data") or []
        papers.extend(p for p in (parse_s2_item(i) for i in items) if p)
        if len(items) < limit or offset + limit >= (data.get("total") or 0):
            break
        if not config.S2_API_KEY:
            time.sleep(1.5)  # unauthenticated pool is shared & strict
    return _finalize(papers)


# ------------------------------------------------------------------------ sample


def load_sample_papers() -> list[Paper]:
    """Bundled offline dataset (real arXiv metadata snapshot) for demos and tests."""
    path = config.SAMPLE_PAPERS_PATH
    if not path.exists():
        raise SourceError(f"Sample dataset not found at {path}. Run scripts/build_sample_data.py.")
    with open(path) as f:
        raw = json.load(f)
    return _finalize([Paper.from_dict(d) for d in raw])


def _finalize(papers: list[Paper]) -> list[Paper]:
    return [p for p in dedupe(clean_paper(p) for p in papers) if is_usable(p)]


def fetch(
    source: str,
    keywords: list[str],
    categories: list[str] | None = None,
    max_results: int = 100,
    days_back: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[Paper], list[str]]:
    """Fetch from one or more sources. Returns (papers, warnings).

    A failing source produces a warning instead of aborting, so e.g. a
    Semantic Scholar 429 does not block arXiv results.
    """
    say = progress or (lambda _m: None)
    papers: list[Paper] = []
    warnings: list[str] = []
    wanted = ["arxiv_new", "semantic_scholar"] if source == "both" else [source]
    for src in wanted:
        try:
            if src == "arxiv_new":
                say("Fetching today's arXiv listings…")
                papers += fetch_arxiv_new(categories or [])
            elif src == "arxiv":
                say("Querying arXiv…")
                papers += search_arxiv(keywords, categories, max_results, days_back)
            elif src == "semantic_scholar":
                say("Querying Semantic Scholar…")
                year_from = datetime.now().year - max(1, (days_back or 365) // 365) if days_back else None
                papers += search_semantic_scholar(keywords, max_results, year_from)
            elif src == "sample":
                say("Loading bundled sample papers…")
                papers += load_sample_papers()
            else:
                warnings.append(f"Unknown source '{src}'")
        except SourceError as e:
            log.warning("source %s failed: %s", src, e)
            warnings.append(f"{src}: {e}")
    return dedupe(papers), warnings
