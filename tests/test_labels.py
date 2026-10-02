"""The shared label dataset: data/labels/profiles.json + labels.jsonl, joined to data/catalog/ by paper id."""

import json
from collections import Counter

from triage import catalog
from triage.config import LABELS

from server.db import ROOT

LABEL_DIR = ROOT / "data" / "labels"
ORIGINS = {"manual", "synthetic"}


def rows():
    return [json.loads(line) for line in (LABEL_DIR / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def test_every_label_points_at_a_catalog_paper_and_a_known_profile():
    ids = {p.id for a in catalog.manifest()["areas"] for p in catalog.read_shard(a)}
    profiles = {p["slug"] for p in json.loads((LABEL_DIR / "profiles.json").read_text(encoding="utf-8"))["profiles"]}
    data = rows()
    assert len(data) > 500
    assert all(r["paper_id"] in ids and r["profile"] in profiles and r["label"] in LABELS for r in data)
    assert {r["origin"] for r in data} <= ORIGINS


def test_labels_hold_only_label_data():
    """Paper details live in the catalog and no label records who made it."""
    assert all(set(r) == {"profile", "paper_id", "label", "labeled_at", "origin"} for r in rows())


def test_one_label_per_profile_and_paper():
    counts = Counter((r["profile"], r["paper_id"]) for r in rows())
    assert max(counts.values()) == 1


def test_every_profile_has_all_three_labels():
    by = {}
    for r in rows():
        by.setdefault(r["profile"], set()).add(r["label"])
    assert all(v == set(LABELS) for v in by.values()), by
