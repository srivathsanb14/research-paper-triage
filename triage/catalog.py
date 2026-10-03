"""Read and validate the public paper catalog in data/catalog/ (OpenAlex snapshot, CC0 metadata)."""

from __future__ import annotations

import hashlib
import json
import re
from itertools import zip_longest

from . import config
from .models import Paper
from .preprocess import title_key
from .transfer import load_json, parse_papers

CATALOG_DIR = config.DATA_DIR / "catalog"
# All 26 OpenAlex fields (2-digit ids) appear once. "ai" additionally selects two
# subfields (4-digit ids) of Computer Science, so ML researchers get a dense feed.
AREAS = {
    "ai": ("AI & machine learning", [1702, 1707]),
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
    """One area's papers, checked against the manifest's checksum."""
    try:
        raw = (CATALOG_DIR / area["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != area["sha256"]:
            raise ValueError("Catalog checksum mismatch")
        return parse_papers(load_json(raw))
    except Exception as exc:
        raise ValueError("Couldn’t load the paper catalog: a shard is missing or damaged.") from exc


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
