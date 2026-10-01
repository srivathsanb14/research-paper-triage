"""Read the site's public paper catalog; personalization never leaves the browser."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from itertools import zip_longest
from pathlib import Path
from urllib.parse import urljoin

from . import config
from .models import Paper
from .preprocess import title_key
from .transfer import load_json, parse_papers

CATALOG_DIR = config.DATA_DIR / "catalog"
# All 26 OpenAlex fields appear once. IDs follow OpenAlex's field taxonomy.
AREAS = {
    "computing": ("Computing & engineering", [17, 21, 22, 25]),
    "physical": ("Maths & physical sciences", [15, 16, 26, 31]),
    "life": ("Biology & medicine", [13, 24, 27, 28, 29, 30, 34, 35, 36]),
    "environment": ("Earth & environment", [11, 19, 23]),
    "society": ("Society, psychology & economics", [14, 18, 20, 32, 33]),
    "humanities": ("Arts & humanities", [12]),
}


def manifest() -> dict:
    try:
        obj = json.loads((CATALOG_DIR / "manifest.json").read_text())
        if obj.get("version") != 1 or not isinstance(obj.get("areas"), list):
            raise ValueError("Invalid catalog manifest")
        for area in obj["areas"]:
            if (area.get("id") not in AREAS
                    or not re.fullmatch(r"[a-z]+-[0-9a-f]{64}\.json", area.get("file", ""))
                    or not re.fullmatch(r"[0-9a-f]{64}", area.get("sha256", ""))):
                raise ValueError("Invalid catalog entry")
        return obj
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("The paper catalog is unavailable. Please try again later.") from exc


def read_shard(area: dict) -> list[Paper]:
    try:
        if sys.platform == "emscripten":
            # Browser worker XHR: same-origin, public files only; no API credentials.
            from pyodide.http import open_url
            base = os.environ["TRIAGE_CATALOG_URL"]
            raw = open_url(urljoin(base, area["file"])).getvalue().encode("utf-8")
        else:
            raw = (CATALOG_DIR / area["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != area["sha256"]:
            raise ValueError("Catalog checksum mismatch")
        return parse_papers(load_json(raw))
    except Exception as exc:
        raise ValueError("Couldn’t load the paper catalog. Check your connection and reload the page.") from exc


def balanced_papers(groups: list[list[Paper]], limit: int) -> list[Paper]:
    """Interleave disciplines so a large field cannot consume the entire batch."""
    if not 1 <= limit <= 1000:
        raise ValueError("Choose between 1 and 1,000 papers.")
    papers, ids, titles = [], set(), set()
    for row in zip_longest(*groups):
        for p in row:
            if p is None:
                continue
            title = title_key(p.title)
            if p.id in ids or (title and title in titles):
                continue
            ids.add(p.id)
            titles.add(title)
            papers.append(p)
            if len(papers) == limit:
                return papers
    return papers


def load_selection(selected: list[str], limit: int = 600) -> list[Paper]:
    if not selected or any(key not in AREAS for key in selected):
        raise ValueError("Choose at least one research field.")
    entries = {entry["id"]: entry for entry in manifest()["areas"]}
    if any(key not in entries for key in selected):
        raise ValueError("Some selected fields are unavailable. Please reload the page.")
    papers = balanced_papers([read_shard(entries[key]) for key in dict.fromkeys(selected)], limit)
    if not papers:
        raise ValueError("No papers are available in these fields yet.")
    return papers
