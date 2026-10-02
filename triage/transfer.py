"""Parsing and validating portable paper collections (JSON) for the catalog and tests."""

from __future__ import annotations

import json
import math
from urllib.parse import urlsplit

from .models import Paper
from .preprocess import clean_paper

MAX_BYTES = 10 * 1024 * 1024
MAX_PAPERS = 2000


def load_json(data: bytes | str):
    if len(data) > MAX_BYTES:
        raise ValueError("Choose a JSON file smaller than 10 MB.")
    try:
        return json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("This file is not valid JSON.") from exc


def _text(value, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text.")
    return value


def _strings(value, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{name} must be a list of text values.")
    return value


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def parse_papers(rows) -> list[Paper]:
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_PAPERS:
        raise ValueError(f"A collection must contain 1–{MAX_PAPERS} papers.")
    papers, ids = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each paper must be a JSON object.")
        values = {}
        for key in Paper.__dataclass_fields__:
            if key not in row:
                continue
            value = row[key]
            if key in ("authors", "categories"):
                value = _strings(value, key)
            elif key in ("year", "citation_count", "references_count"):
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError(f"{key} must be a non-negative integer or null.")
            elif key == "fwci":
                if value is not None:
                    value = _number(value, key)
            elif key == "venue_core":
                if type(value) is not bool:
                    raise ValueError("venue_core must be true or false.")
            else:
                value = _text(value, key)
            values[key] = value
        if not values.get("id", "").strip() or not values.get("title", "").strip():
            raise ValueError("Every paper needs a non-empty id and title.")
        if values["id"] in ids:
            raise ValueError(f"Duplicate paper id: {values['id']}")
        ids.add(values["id"])
        for key in ("url", "pdf_url"):
            url = values.get(key, "")
            if url:
                parsed = urlsplit(url)
                if parsed.scheme not in ("http", "https") or not parsed.netloc:
                    raise ValueError(f"{key} must be an http or https URL.")
        papers.append(clean_paper(Paper(**values)))
    return papers
