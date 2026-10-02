"""SQLite storage: users, sessions, per-user app state, per-user labels, seed label sets."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("TRIAGE_DB", ROOT / "data" / "users.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    pw_hash TEXT NOT NULL,
    created TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS user_state (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    state TEXT NOT NULL,
    updated TEXT NOT NULL
);
-- One row per hand label, kept in step with the state blob so labels can be queried and exported.
CREATE TABLE IF NOT EXISTS labels (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    profile_id TEXT NOT NULL,
    profile_name TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('READ','SKIM','SKIP')),
    labeler TEXT NOT NULL DEFAULT '',
    labeled_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (user_id, profile_id, paper_id)
);
CREATE TABLE IF NOT EXISTS seed_sets (
    slug TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    profile TEXT NOT NULL,
    n INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS seed_labels (
    slug TEXT NOT NULL REFERENCES seed_sets(slug) ON DELETE CASCADE,
    paper_id TEXT NOT NULL,
    label TEXT NOT NULL,
    labeler TEXT NOT NULL DEFAULT '',
    labeled_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (slug, paper_id)
);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def session():
    conn = connect()
    try:
        with conn:  # commit or roll back
            yield conn
    finally:
        conn.close()


def init() -> None:
    with session() as conn:
        conn.executescript(SCHEMA)


def sync_labels(conn: sqlite3.Connection, user_id: int, state: dict) -> int:
    """Rewrite the user's label rows from the state blob (the blob is the source of truth)."""
    rows = []
    for pid, prof in (state.get("profiles") or {}).items():
        if not isinstance(prof, dict):
            continue
        for paper_id, lab in (prof.get("labels") or {}).items():
            label = lab.get("label") if isinstance(lab, dict) else lab
            if label not in ("READ", "SKIM", "SKIP"):
                continue
            meta = lab if isinstance(lab, dict) else {}
            rows.append((user_id, str(pid), str(prof.get("name", ""))[:200], str(paper_id), label,
                         str(meta.get("labeler", ""))[:80], str(meta.get("at", ""))[:40]))
    conn.execute("DELETE FROM labels WHERE user_id = ?", (user_id,))
    conn.executemany("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?)", rows)
    return len(rows)


def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
