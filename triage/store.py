"""SQLite persistence: papers, embeddings cache, profiles, hand labels, feedback log,
explanation cache, per-profile settings.

A fresh connection is opened per operation (Streamlit reruns scripts on many
threads); WAL mode keeps concurrent reads cheap.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import config
from .models import InterestProfile, Paper

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    paper_id TEXT NOT NULL,
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (paper_id, model)
);
CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS profile_papers (
    profile_id INTEGER NOT NULL,
    paper_id TEXT NOT NULL,
    added_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, paper_id)
);
CREATE TABLE IF NOT EXISTS labels (
    profile_id INTEGER NOT NULL,
    paper_id TEXT NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('READ','SKIM','SKIP')),
    labeler TEXT NOT NULL DEFAULT 'team',
    created_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, paper_id)
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL,
    paper_id TEXT NOT NULL,
    action TEXT NOT NULL,
    value TEXT,
    predicted_label TEXT,
    score REAL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_profile ON feedback(profile_id, paper_id);
CREATE TABLE IF NOT EXISTS explanations (
    cache_key TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS seeds (
    profile_id INTEGER NOT NULL,
    paper_id TEXT NOT NULL,
    added_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, paper_id)
);
CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL,
    source TEXT NOT NULL,
    ok INTEGER NOT NULL,
    n_papers INTEGER NOT NULL DEFAULT 0,
    n_new INTEGER NOT NULL DEFAULT 0,
    message TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    profile_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (profile_id, key)
);
"""

# Feedback actions the UI can log. "correct" carries the corrected label in `value`.
FEEDBACK_ACTIONS = {"open", "dismiss", "useful", "not_useful", "correct", "undismiss", "explanation_ok", "explanation_bad"}

_init_lock = threading.Lock()
_initialized: set[str] = set()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path | None = None):
        self.path = str(path or config.DB_PATH)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._mem_conn: sqlite3.Connection | None = None
        with _init_lock:
            if self.path not in _initialized or self.path == ":memory:":
                with self._conn() as c:
                    c.executescript(SCHEMA)
                _initialized.add(self.path)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        if self.path == ":memory:":  # tests: keep one shared connection
            if self._mem_conn is None:
                self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._mem_conn.row_factory = sqlite3.Row
            yield self._mem_conn
            self._mem_conn.commit()
            return
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            # A single database file is safer to sync to IndexedDB in the browser.
            conn.execute("PRAGMA journal_mode=DELETE" if config.BROWSER_MODE else "PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ papers
    def upsert_papers(self, papers: list[Paper]) -> int:
        now = _now()
        with self._conn() as c:
            c.executemany(
                "INSERT INTO papers(id, data, fetched_at) VALUES (?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET data=excluded.data",
                [(p.id, json.dumps(p.to_dict()), now) for p in papers],
            )
        return len(papers)

    def get_papers(self, ids: list[str] | None = None) -> list[Paper]:
        with self._conn() as c:
            if ids is None:
                rows = c.execute("SELECT data FROM papers").fetchall()
            else:
                rows = []
                for chunk in _chunks(ids, 500):
                    q = f"SELECT data FROM papers WHERE id IN ({','.join('?' * len(chunk))})"
                    rows += c.execute(q, chunk).fetchall()
        return [Paper.from_dict(json.loads(r["data"])) for r in rows]

    def get_paper(self, paper_id: str) -> Paper | None:
        got = self.get_papers([paper_id])
        return got[0] if got else None

    # -------------------------------------------------------------- embeddings
    def get_embeddings(self, ids: list[str], model: str) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with self._conn() as c:
            for chunk in _chunks(ids, 500):
                q = f"SELECT paper_id, vector FROM embeddings WHERE model=? AND paper_id IN ({','.join('?' * len(chunk))})"
                for r in c.execute(q, [model, *chunk]):
                    out[r["paper_id"]] = np.frombuffer(r["vector"], dtype=np.float32)
        return out

    def put_embeddings(self, vecs: dict[str, np.ndarray], model: str) -> None:
        with self._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO embeddings(paper_id, model, vector) VALUES (?,?,?)",
                [(pid, model, np.asarray(v, dtype=np.float32).tobytes()) for pid, v in vecs.items()],
            )

    # ---------------------------------------------------------------- profiles
    def save_profile(self, profile: InterestProfile) -> InterestProfile:
        data = {k: v for k, v in profile.__dict__.items() if k != "id"}
        with self._conn() as c:
            c.execute(
                "INSERT INTO profiles(name, data, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at",
                (profile.name, json.dumps(data), _now()),
            )
            row = c.execute("SELECT id FROM profiles WHERE name=?", (profile.name,)).fetchone()
        profile.id = int(row["id"])
        return profile

    def list_profiles(self) -> list[InterestProfile]:
        with self._conn() as c:
            rows = c.execute("SELECT id, data FROM profiles ORDER BY updated_at DESC").fetchall()
        return [self._row_to_profile(r) for r in rows]

    def get_profile(self, name: str) -> InterestProfile | None:
        with self._conn() as c:
            r = c.execute("SELECT id, data FROM profiles WHERE name=?", (name,)).fetchone()
        return self._row_to_profile(r) if r else None

    def delete_profile(self, profile_id: int) -> None:
        with self._conn() as c:
            for table in ("profile_papers", "labels", "feedback", "settings", "seeds", "fetch_log"):
                c.execute(f"DELETE FROM {table} WHERE profile_id=?", (profile_id,))
            c.execute("DELETE FROM profiles WHERE id=?", (profile_id,))

    @staticmethod
    def _row_to_profile(r: sqlite3.Row) -> InterestProfile:
        d = json.loads(r["data"])
        known = InterestProfile.__dataclass_fields__
        p = InterestProfile(**{k: v for k, v in d.items() if k in known and k != "id"})
        p.id = int(r["id"])
        return p

    # ------------------------------------------------------------ paper pools
    def add_to_pool(self, profile_id: int, paper_ids: list[str]) -> int:
        with self._conn() as c:
            before = c.execute("SELECT COUNT(*) FROM profile_papers WHERE profile_id=?", (profile_id,)).fetchone()[0]
            c.executemany(
                "INSERT OR IGNORE INTO profile_papers(profile_id, paper_id, added_at) VALUES (?,?,?)",
                [(profile_id, pid, _now()) for pid in paper_ids],
            )
            after = c.execute("SELECT COUNT(*) FROM profile_papers WHERE profile_id=?", (profile_id,)).fetchone()[0]
        return after - before

    def pool_ids(self, profile_id: int) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT paper_id FROM profile_papers WHERE profile_id=? ORDER BY added_at, paper_id", (profile_id,)
            ).fetchall()
        return [r["paper_id"] for r in rows]

    def pool_added(self, profile_id: int) -> dict[str, str]:
        """paper_id → when it entered this profile's pool (ISO timestamp)."""
        with self._conn() as c:
            rows = c.execute("SELECT paper_id, added_at FROM profile_papers WHERE profile_id=?", (profile_id,)).fetchall()
        return {r["paper_id"]: r["added_at"] for r in rows}

    def set_pool_added(self, profile_id: int, added: dict[str, str]) -> None:
        """Backdate pool entries (used by the demo to simulate papers arriving over several days)."""
        with self._conn() as c:
            c.executemany(
                "UPDATE profile_papers SET added_at=? WHERE profile_id=? AND paper_id=?",
                [(t, profile_id, pid) for pid, t in added.items()],
            )

    def clear_pool(self, profile_id: int) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM profile_papers WHERE profile_id=?", (profile_id,))

    # ------------------------------------------------------------------ labels
    def set_label(
        self, profile_id: int, paper_id: str, label: str, labeler: str = "team", created_at: str | None = None
    ) -> None:
        if label not in config.LABELS:
            raise ValueError(f"label must be one of {config.LABELS}")
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO labels(profile_id, paper_id, label, labeler, created_at) VALUES (?,?,?,?,?)",
                (profile_id, paper_id, label, labeler, created_at or _now()),
            )

    def remove_label(self, profile_id: int, paper_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM labels WHERE profile_id=? AND paper_id=?", (profile_id, paper_id))

    def get_label_events(self, profile_id: int) -> list[dict[str, Any]]:
        """Labels with their timestamps, oldest first (for learning curves)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT paper_id, label, created_at FROM labels WHERE profile_id=? ORDER BY created_at, rowid", (profile_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_labels(self, profile_id: int) -> dict[str, str]:
        with self._conn() as c:
            rows = c.execute("SELECT paper_id, label FROM labels WHERE profile_id=?", (profile_id,)).fetchall()
        return {r["paper_id"]: r["label"] for r in rows}

    # ---------------------------------------------------------------- feedback
    def log_feedback(
        self,
        profile_id: int,
        paper_id: str,
        action: str,
        value: str | None = None,
        predicted_label: str | None = None,
        score: float | None = None,
        created_at: str | None = None,
    ) -> None:
        if action not in FEEDBACK_ACTIONS:
            raise ValueError(f"unknown feedback action {action!r}")
        if action == "correct" and value not in config.LABELS:
            raise ValueError("a correction must carry a READ/SKIM/SKIP value")
        with self._conn() as c:
            c.execute(
                "INSERT INTO feedback(profile_id, paper_id, action, value, predicted_label, score, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (profile_id, paper_id, action, value, predicted_label, score, created_at or _now()),
            )

    def get_feedback(self, profile_id: int) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM feedback WHERE profile_id=? ORDER BY id", (profile_id,)).fetchall()
        return [dict(r) for r in rows]

    def dismissed_ids(self, profile_id: int) -> set[str]:
        """Papers whose most recent dismiss/undismiss event is a dismiss."""
        state: dict[str, bool] = {}
        for f in self.get_feedback(profile_id):
            if f["action"] == "dismiss":
                state[f["paper_id"]] = True
            elif f["action"] == "undismiss":
                state[f["paper_id"]] = False
        return {pid for pid, d in state.items() if d}

    def feedback_version(self, profile_id: int) -> str:
        """Token that changes whenever labels/feedback change (cache key)."""
        with self._conn() as c:
            f = c.execute("SELECT COALESCE(MAX(id),0), COUNT(*) FROM feedback WHERE profile_id=?", (profile_id,)).fetchone()
            l = c.execute(
                "SELECT COUNT(*), COALESCE(MAX(created_at),'') FROM labels WHERE profile_id=?", (profile_id,)
            ).fetchone()
            sd = c.execute("SELECT COUNT(*), COALESCE(MAX(added_at),'') FROM seeds WHERE profile_id=?", (profile_id,)).fetchone()
        return f"{f[0]}-{f[1]}-{l[0]}-{l[1]}-{sd[0]}-{sd[1]}"

    # ------------------------------------------------------------------- seeds
    def add_seeds(self, profile_id: int, paper_ids: list[str]) -> int:
        with self._conn() as c:
            before = c.execute("SELECT COUNT(*) FROM seeds WHERE profile_id=?", (profile_id,)).fetchone()[0]
            c.executemany(
                "INSERT OR IGNORE INTO seeds(profile_id, paper_id, added_at) VALUES (?,?,?)",
                [(profile_id, pid, _now()) for pid in paper_ids],
            )
            after = c.execute("SELECT COUNT(*) FROM seeds WHERE profile_id=?", (profile_id,)).fetchone()[0]
        return after - before

    def seed_ids(self, profile_id: int) -> list[str]:
        with self._conn() as c:
            rows = c.execute("SELECT paper_id FROM seeds WHERE profile_id=? ORDER BY added_at", (profile_id,)).fetchall()
        return [r["paper_id"] for r in rows]

    def remove_seed(self, profile_id: int, paper_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM seeds WHERE profile_id=? AND paper_id=?", (profile_id, paper_id))

    # ---------------------------------------------------------------- fetches
    def log_fetch(self, profile_id: int, source: str, ok: bool, n_papers: int = 0, n_new: int = 0, message: str = "") -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO fetch_log(profile_id, source, ok, n_papers, n_new, message, created_at) VALUES (?,?,?,?,?,?,?)",
                (profile_id, source, int(ok), n_papers, n_new, message[:500], _now()),
            )

    def fetch_history(self, profile_id: int, limit: int = 20) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM fetch_log WHERE profile_id=? ORDER BY id DESC LIMIT ?", (profile_id, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    def last_successful_fetch(self, profile_id: int) -> str | None:
        with self._conn() as c:
            r = c.execute(
                "SELECT MAX(created_at) FROM fetch_log WHERE profile_id=? AND ok=1", (profile_id,)
            ).fetchone()
        return r[0] if r and r[0] else None

    # ------------------------------------------------------------ explanations
    def get_explanation(self, key: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT data FROM explanations WHERE cache_key=?", (key,)).fetchone()
        return json.loads(r["data"]) if r else None

    def put_explanation(self, key: str, data: dict) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO explanations(cache_key, data, created_at) VALUES (?,?,?)",
                (key, json.dumps(data), _now()),
            )

    # ---------------------------------------------------------------- settings
    def get_setting(self, profile_id: int, key: str, default: Any = None) -> Any:
        with self._conn() as c:
            r = c.execute("SELECT value FROM settings WHERE profile_id=? AND key=?", (profile_id, key)).fetchone()
        return json.loads(r["value"]) if r else default

    def set_setting(self, profile_id: int, key: str, value: Any) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO settings(profile_id, key, value) VALUES (?,?,?)",
                (profile_id, key, json.dumps(value)),
            )


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]
