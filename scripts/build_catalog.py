"""Collect a small, balanced OpenAlex catalog for publication as static files.

Run on GitHub Actions, never in a visitor's browser. OPENALEX_API_KEY is optional
and remains in the job environment. Failed refreshes leave the last catalog intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from itertools import zip_longest
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage.catalog import AREAS, CATALOG_DIR, balanced_papers  # noqa: E402
from triage.models import Paper  # noqa: E402
from triage.preprocess import title_key  # noqa: E402
from triage.transfer import parse_papers  # noqa: E402

SELECT = ("id,doi,title,type,abstract_inverted_index,authorships,primary_location,best_oa_location,open_access,"
          "publication_date,publication_year,primary_topic,cited_by_count,fwci,referenced_works_count")


def abstract_text(index: dict | None) -> str:
    if not isinstance(index, dict):
        return ""
    words = {}
    for word, positions in index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if type(position) is int and 0 <= position < 10000:
                words[position] = word
    return " ".join(words[k] for k in sorted(words))


def paper_from_work(work: dict) -> Paper | None:
    abstract = abstract_text(work.get("abstract_inverted_index"))
    if len(abstract.split()) < 30 or not work.get("title"):
        return None
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    oa = work.get("best_oa_location") or {}
    topic = work.get("primary_topic") or {}
    field = topic.get("field") or {}
    doi = (work.get("doi") or "").strip().lower()
    row = {
        "id": "doi:" + doi.removeprefix("https://doi.org/") if doi else "openalex:" + work["id"].rsplit("/", 1)[-1],
        "title": work["title"], "abstract": abstract,
        "authors": [a["author"]["display_name"] for a in work.get("authorships", [])
                    if a.get("author", {}).get("display_name")],
        "venue": source.get("display_name") or "",
        "published": work.get("publication_date") or "",
        "year": work.get("publication_year"), "source": "openalex",
        "url": doi or location.get("landing_page_url") or work["id"],
        "pdf_url": oa.get("pdf_url") or "",
        "categories": [name for name in [field.get("display_name"), topic.get("display_name")] if name],
        "citation_count": work.get("cited_by_count"),
        "work_type": work.get("type") or "",
        "venue_type": source.get("type") or "",
        "venue_core": source.get("is_core") is True,
        "version": location.get("version") or "",
        "oa_status": (work.get("open_access") or {}).get("oa_status") or "",
        "fwci": work["fwci"] if isinstance(work.get("fwci"), (int, float)) else None,
        "references_count": work.get("referenced_works_count"),
    }
    try:
        paper = parse_papers([row])[0]
        return paper if len(paper.abstract.split()) >= 30 else None
    except ValueError:
        return None


def _fetch_sorted(session, filters: str, sort: str, count: int, seen: set[str]) -> list[Paper]:
    papers = []
    # Bound API use even if some records have missing or unusable abstracts.
    for page in range(1, math.ceil(count / 100) + 3):
        params = {"filter": filters, "sort": sort, "per_page": 100, "page": page, "select": SELECT}
        if os.environ.get("OPENALEX_API_KEY"):
            params["api_key"] = os.environ["OPENALEX_API_KEY"]
        for attempt in range(3):
            response = session.get("https://api.openalex.org/works", params=params, timeout=30)
            if response.status_code == 200:
                break
            if response.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
                # Never print a request URL, which may contain the owner's key.
                raise RuntimeError(f"OpenAlex returned HTTP {response.status_code}; keeping the previous catalog.")
            delay = response.headers.get("Retry-After", "3")
            if delay.isdigit() and int(delay) > 30:
                raise RuntimeError("OpenAlex quota is exhausted; keeping the previous catalog.")
            time.sleep(min(int(delay), 10) if delay.isdigit() else 3)
        works = response.json()["results"]
        for work in works:
            paper = paper_from_work(work)
            if paper and paper.id not in seen:
                seen.add(paper.id)
                papers.append(paper)
                if len(papers) >= count:
                    return papers
        if len(works) < 100:
            break
        time.sleep(0.2)
    return papers


def fetch_field(session, field: int, count: int, since: str, until: str) -> list[Paper]:
    """Half the newest papers, half the most-cited ones from the same window.

    Newest-first alone fills the feed with whatever was uploaded last (repository
    dumps, theses); the cited half shows what the field is already engaging with.
    Ids of 1000 and above are OpenAlex subfields, smaller ones are fields.
    """
    level = "subfield" if field >= 1000 else "field"
    filters = (f"primary_topic.{level}.id:{field},from_publication_date:{since},to_publication_date:{until},"
               "has_abstract:true,is_retracted:false,language:en,type:article|review|preprint")
    seen: set[str] = set()
    newest = _fetch_sorted(session, filters, "publication_date:desc", math.ceil(count / 2), seen)
    cited = _fetch_sorted(session, filters, "cited_by_count:desc", count - len(newest), seen)
    return [p for pair in zip_longest(newest, cited) for p in pair if p is not None]


def build(output: Path = CATALOG_DIR, per_area: int = 300, days: int = 180) -> None:
    if not 30 <= per_area <= 1000 or not 1 <= days <= 730:
        raise ValueError("Use 30–1,000 papers per area and 1–730 days.")
    today = date.today()
    since = (today - timedelta(days=days)).isoformat()
    session = requests.Session()
    session.headers["User-Agent"] = "PaperTriageCatalog/1.0 (public research metadata)"
    info = {"version": 1, "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "OpenAlex", "source_url": "https://openalex.org", "license": "CC0",
            "since": since, "until": today.isoformat(), "language": "en", "areas": []}
    # Stage every field before replacing the manifest, so failed calls cannot
    # publish an empty or partially refreshed catalog.
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary)
        taken: set[str] = set()  # areas overlap ("ai" is part of Computer Science): first area keeps a paper
        for key, (label, fields) in AREAS.items():
            groups = [fetch_field(session, field, math.ceil(per_area / len(fields)), since, today.isoformat()) for field in fields]
            groups = [[p for p in g if p.id not in taken and title_key(p.title) not in taken] for g in groups]
            papers = balanced_papers(groups, per_area)
            taken.update(p.id for p in papers)
            taken.update(title_key(p.title) for p in papers)
            if len(papers) < min(30, per_area):
                raise RuntimeError(f"Insufficient papers for {label}; keeping the previous catalog.")
            raw = (json.dumps([p.to_dict() for p in papers], ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            digest = hashlib.sha256(raw).hexdigest()
            filename = f"{key}-{digest}.json"
            (stage / filename).write_bytes(raw)
            info["areas"].append({"id": key, "label": label, "file": filename, "sha256": digest,
                                  "count": len(papers), "fields": fields})
            print(f"{label}: {len(papers)} papers", flush=True)
        for entry in info["areas"]:
            (output / entry["file"]).write_bytes((stage / entry["file"]).read_bytes())
        manifest = output / "manifest.json"
        pending = output / "manifest.tmp"
        pending.write_text(json.dumps(info, indent=2) + "\n")
        pending.replace(manifest)
        keep = {entry["file"] for entry in info["areas"]} | {"manifest.json"}
        for old in output.glob("*.json"):
            if old.name not in keep:
                old.unlink()
    print(f"Published {sum(a['count'] for a in info['areas'])} records across {len(AREAS)} areas.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CATALOG_DIR)
    parser.add_argument("--per-area", type=int, default=300)
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()
    try:
        build(args.output, args.per_area, args.days)
    except (RuntimeError, requests.RequestException, ValueError, KeyError) as exc:
        # Requests exceptions may embed credential-bearing URLs.
        print(str(exc) if isinstance(exc, RuntimeError) else "Catalog refresh failed; previous files are unchanged.", file=sys.stderr)
        sys.exit(1)
