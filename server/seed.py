"""Starter label sets: the shared label dataset (data/labels/) offered as ready-made profiles.

data/labels/profiles.json describes each research profile and data/labels/labels.jsonl holds one
row per (profile, paper). Both are read from disk, not from the users database.
"""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache

from . import db

LABEL_DIR = db.ROOT / "data" / "labels"


@lru_cache(maxsize=1)
def load() -> dict[str, dict]:
    """{slug: {"slug", "name", "origin", "profile": {...}, "labels": [{paper_id, label, labeled_at}]}}."""
    profiles = {p["slug"]: p for p in json.loads((LABEL_DIR / "profiles.json").read_text(encoding="utf-8"))["profiles"]}
    labels: dict[str, list[dict]] = defaultdict(list)
    for line in (LABEL_DIR / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            labels[r["profile"]].append({"paper_id": r["paper_id"], "label": r["label"], "labeled_at": r["labeled_at"], "origin": r["origin"]})
    out = {}
    for slug, rows in labels.items():
        p = profiles[slug]
        origins = {r["origin"] for r in rows}
        out[slug] = {
            "slug": slug, "name": p["name"], "origin": origins.pop() if len(origins) == 1 else "mixed",
            "profile": {k: p.get(k, "" if k in ("description", "focus") else []) for k in ("name", "description", "keywords", "focus", "avoid", "fields")},
            "labels": rows,
        }
    return out


def summaries() -> list[dict]:
    return sorted(({"slug": s["slug"], "name": s["name"], "origin": s["origin"], "n": len(s["labels"])} for s in load().values()),
                  key=lambda s: (s["origin"] != "manual", s["name"]))
