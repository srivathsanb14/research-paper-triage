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
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage.catalog import AREAS, CATALOG_DIR, balanced_papers  # noqa: E402
from triage.models import Paper  # noqa: E402
from triage.transfer import parse_papers  # noqa: E402

SELECT = "id,doi,title,abstract_inverted_index,authorships,primary_location,best_oa_location,publication_date,publication_year,primary_topic,cited_by_count"


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
    oa = work.get("best_oa_location") or {}
    topic = work.get("primary_topic") or {}
    field = topic.get("field") or {}
    doi = (work.get("doi") or "").strip().lower()
    row = {
        "id": "doi:" + doi.removeprefix("https://doi.org/") if doi else "openalex:" + work["id"].rsplit("/", 1)[-1],
        "title": work["title"], "abstract": abstract,
        "authors": [a["author"]["display_name"] for a in work.get("authorships", [])
                    if a.get("author", {}).get("display_name")],
        "venue": (location.get("source") or {}).get("display_name") or "",
        "published": work.get("publication_date") or "",
        "year": work.get("publication_year"), "source": "openalex",
        "url": doi or location.get("landing_page_url") or work["id"],
        "pdf_url": oa.get("pdf_url") or "",
        "categories": [name for name in [field.get("display_name"), topic.get("display_name")] if name],
        "citation_count": work.get("cited_by_count"),
    }
    try:
        return parse_papers([row])[0]
    except ValueError:
        return None


def fetch_field(session, field: int, count: int, since: str, until: str) -> list[Paper]:
    papers, seen = [], set()
    # Bound API use even if some records have missing or unusable abstracts.
    for page in range(1, math.ceil(count / 100) + 3):
        params = {"filter": f"primary_topic.field.id:{field},from_publication_date:{since},to_publication_date:{until},has_abstract:true,is_retracted:false,language:en,type:article|review|preprint",
                  "sort": "publication_date:desc", "per_page": 100, "page": page, "select": SELECT}
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
        for key, (label, fields) in AREAS.items():
            groups = [fetch_field(session, field, math.ceil(per_area / len(fields)), since, today.isoformat()) for field in fields]
            papers = balanced_papers(groups, per_area)
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
