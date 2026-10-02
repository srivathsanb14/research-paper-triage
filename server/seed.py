"""Load the repository's labelled data (data/labels/) into the database as starter label sets.

Human-verified labels and rule-generated (synthetic) labels stay separate sets, so a user can
tell which is which. Re-running replaces the sets; users' own labels are never touched.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

from . import db

LABELS = ("READ", "SKIM", "SKIP")
LABEL_DIR = db.ROOT / "data" / "labels"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "set"


def read_rows():
    for kind, folder in (("human", "manual"), ("synthetic", "synthetic")):
        for path in sorted((LABEL_DIR / folder).glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("label") in LABELS and row.get("paper_id"):
                    yield kind, row


def profile_defaults() -> dict[str, dict]:
    path = LABEL_DIR / "proposals" / "profiles.json"
    if not path.exists():
        return {}
    return {p["name"]: p for p in json.loads(path.read_text(encoding="utf-8")).get("profiles", [])}


def load() -> list[dict]:
    """Rebuild seed_sets / seed_labels. Returns the sets created."""
    defaults = profile_defaults()
    sets: dict[str, dict] = {}
    labels: dict[str, dict[str, tuple]] = defaultdict(dict)  # latest label per paper wins
    for kind, row in read_rows():
        name = row.get("profile_name") or "Imported labels"
        slug = f"{kind}-{slugify(name)}"
        base = defaults.get(name, {})
        sets.setdefault(slug, {
            "slug": slug, "kind": kind,
            "name": name,
            "profile": {
                "name": name,
                "description": row.get("profile_description") or base.get("description", ""),
                "keywords": row.get("profile_keywords") or base.get("keywords", []),
                "focus": row.get("profile_focus") or base.get("focus", ""),
                "avoid": base.get("avoid", []),
                "fields": base.get("fields", []),
            },
        })
        stamp = row.get("labeled_at", "")
        prev = labels[slug].get(row["paper_id"])
        if prev is None or stamp >= prev[3]:
            labels[slug][row["paper_id"]] = (row["label"], row.get("labeler", ""), stamp, stamp)
    with db.session() as conn:
        conn.execute("DELETE FROM seed_labels")
        conn.execute("DELETE FROM seed_sets")
        for slug, s in sets.items():
            conn.execute("INSERT INTO seed_sets VALUES (?,?,?,?,?)",
                         (slug, s["name"], s["kind"], db.dumps(s["profile"]), len(labels[slug])))
            conn.executemany("INSERT INTO seed_labels VALUES (?,?,?,?,?)",
                             [(slug, pid, lab, who, at) for pid, (lab, who, at, _) in labels[slug].items()])
    return list(sets.values())
