"""Portable paper collections and profile backups for the browser app."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from urllib.parse import urlsplit

from . import config
from .models import InterestProfile, Paper
from .preprocess import clean_paper
from .store import FEEDBACK_ACTIONS, Store

MAX_BYTES = 10 * 1024 * 1024
MAX_PAPERS = 2000
SETTINGS = ("cutoffs", "budget", "auto_update", "group_similar", "good_mode", "demo")


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
            elif key in ("year", "citation_count"):
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError(f"{key} must be a non-negative integer or null.")
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


def export_profile(store: Store, profile: InterestProfile) -> str:
    pool = store.pool_ids(profile.id)
    seeds = store.seed_ids(profile.id)
    labels = store.get_labels(profile.id)
    feedback = store.get_feedback(profile.id)
    paper_ids = set(pool + seeds) | set(labels) | {event["paper_id"] for event in feedback}
    return json.dumps({
        "format": "paper-triage-profile",
        "version": 1,
        "profile": asdict(profile),
        "papers": [p.to_dict() for p in store.get_papers(sorted(paper_ids))],
        "pool": pool,
        "seeds": seeds,
        "labels": labels,
        "label_times": {event["paper_id"]: event["created_at"] for event in store.get_label_events(profile.id)},
        "feedback": feedback,
        "settings": {key: store.get_setting(profile.id, key) for key in SETTINGS},
    }, ensure_ascii=False, indent=2, allow_nan=False)


def restore_profile(store: Store, data: bytes | str) -> InterestProfile:
    """Validate before writing, then restore into a new, uniquely named profile."""
    obj = load_json(data)
    if not isinstance(obj, dict) or obj.get("format") != "paper-triage-profile" or obj.get("version") != 1:
        raise ValueError("Choose a Paper Triage profile backup (version 1).")
    raw = obj.get("profile")
    if not isinstance(raw, dict):
        raise ValueError("The backup is missing its research profile.")
    fields = {key: _text(raw.get(key, ""), key) for key in ("name", "description", "focus")}
    fields.update({key: _strings(raw.get(key, []), key) for key in ("keywords", "avoid")})
    hours = _number(raw.get("hours_per_week", 3), "Reading time")
    if not 0.5 <= hours <= 40 or not fields["name"].strip():
        raise ValueError("The profile needs a name and reading time between 0.5 and 40 hours.")
    profile = InterestProfile(**fields, hours_per_week=hours)
    if profile.is_empty():
        raise ValueError("The profile needs research interests.")
    papers = parse_papers(obj.get("papers")) if obj.get("papers") != [] else []
    ids = {p.id for p in papers}
    pool = _strings(obj.get("pool"), "Paper list")
    seeds = _strings(obj.get("seeds"), "Seed papers")
    labels = obj.get("labels")
    label_times = obj.get("label_times", {})
    feedback = obj.get("feedback")
    settings = obj.get("settings")
    if not set(pool + seeds) <= ids:
        raise ValueError("The backup references missing papers.")
    if not isinstance(labels, dict) or any(pid not in ids or lab not in config.LABELS for pid, lab in labels.items()):
        raise ValueError("The backup contains invalid labels.")
    if not isinstance(label_times, dict) or not set(label_times) <= set(labels):
        raise ValueError("The backup contains invalid label timestamps.")
    for timestamp in label_times.values():
        try:
            datetime.fromisoformat(_text(timestamp, "Label timestamp"))
        except ValueError as exc:
            raise ValueError("Label timestamps must use ISO date format.") from exc
    if not isinstance(feedback, list) or not isinstance(settings, dict):
        raise ValueError("The backup contains invalid feedback or settings.")
    for event in feedback:
        if (not isinstance(event, dict) or not isinstance(event.get("paper_id"), str)
                or not isinstance(event.get("action"), str)
                or event["paper_id"] not in ids or event["action"] not in FEEDBACK_ACTIONS):
            raise ValueError("The backup contains an invalid feedback event.")
        if event["action"] == "correct" and event.get("value") not in config.LABELS:
            raise ValueError("A corrected label must be READ, SKIM or SKIP.")
        for key in ("value", "predicted_label", "created_at"):
            if event.get(key) is not None:
                _text(event[key], key)
        if event.get("score") is not None:
            _number(event["score"], "Feedback score")
        if event.get("created_at"):
            try:
                datetime.fromisoformat(event["created_at"])
            except ValueError as exc:
                raise ValueError("Feedback timestamps must use ISO date format.") from exc
    for key, value in settings.items():
        if key not in SETTINGS or value is None:
            continue
        if key == "cutoffs":
            if not isinstance(value, dict) or set(value) != {"read", "skim"}:
                raise ValueError("Invalid triage cutoffs.")
            read = _number(value["read"], "Read cutoff")
            skim = _number(value["skim"], "Skim cutoff")
            if not 0 <= skim <= read <= 1:
                raise ValueError("Cutoffs must satisfy 0 ≤ Skim ≤ Read ≤ 1.")
        elif key == "good_mode":
            if value not in ("read", "read_skim"):
                raise ValueError("Unknown evaluation mode.")
        elif type(value) is not bool:
            raise ValueError(f"{key} must be true or false.")
    name, suffix = profile.name, 1
    while store.get_profile(profile.name) is not None:
        profile.name = f"{name} (imported {suffix})"
        suffix += 1
    store.save_profile(profile)
    store.upsert_papers(papers)
    store.add_to_pool(profile.id, pool)
    store.add_seeds(profile.id, seeds)
    for pid, label in labels.items():
        store.set_label(profile.id, pid, label, created_at=label_times.get(pid))
    for event in feedback:
        store.log_feedback(profile.id, event["paper_id"], event["action"], event.get("value"),
                           event.get("predicted_label"), event.get("score"), event.get("created_at"))
    for key in SETTINGS:
        if settings.get(key) is not None:
            store.set_setting(profile.id, key, settings[key])
    return profile
