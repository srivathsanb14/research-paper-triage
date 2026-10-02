"""Create the demo accounts and give each two profiles with labels from the shared label dataset.

    python -m server.demo_users            # create what is missing; never overwrites a user's saved work
    python -m server.demo_users --reset    # replace the demo users' saved state again

sri (materials discovery), ishaan (robotics) and chris (AI and design) all use password "demo",
which is shorter than registration allows, so the accounts are created directly. Demo only: don't
use this on a public host. Each profile also has a few papers saved to its reading list. Each user gets their own synthetic (rule-based mockup) profile plus one
of the manual profiles, and lands on the manual one; both come from data/labels/ (labels.jsonl, origin "synthetic" or "manual").
"""

import argparse
from datetime import datetime, timezone

from . import db, seed
from .app import hash_password

PASSWORD = "demo"
# user → (their synthetic profile, the manual profile added to their login)
USERS = {
    "sri": ("materials-discovery", "cancer-immunotherapy"),
    "ishaan": ("robotics", "rag-llm-evaluation"),
    "chris": ("ai-and-design", "climate-adaptation"),
}


SAVED_PER_PROFILE = 5


def profile_state(slug: str, now: str) -> dict:
    found = seed.load()[slug]
    # A few papers already on the reading list: Read-labelled first, then Skim, by id (deterministic).
    ranked = sorted((r["label"] != "READ", r["paper_id"]) for r in found["labels"] if r["label"] in ("READ", "SKIM"))
    saved = [pid for _, pid in ranked[:SAVED_PER_PROFILE]]
    return {
        "id": f"demo-{slug}", **found["profile"], "hours": 3, "cutoffs": {"read": 0.62, "skim": 0.4}, "budget": True,
        "group": True, "seeds": [], "starter": [],
        "feedback": [{"pid": pid, "action": "save", "value": None, "at": now, "predicted": None, "score": None} for pid in saved], "extra": [], "seen": [], "lastVisit": None, "created": now,
        "labels": {r["paper_id"]: {"label": r["label"], "at": r["labeled_at"]} for r in found["labels"]},
    }


def demo_state(user: str, slugs: tuple[str, ...]) -> dict:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    profiles = {f"demo-{slug}": profile_state(slug, now) for slug in slugs}
    return {"version": 1, "active": f"demo-{slugs[1]}", "owner": user, "settings": {"semantic": True}, "profiles": profiles}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    reset = parser.parse_args().reset
    db.init()
    with db.session() as conn:
        for user, slugs in USERS.items():
            row = conn.execute("SELECT id FROM users WHERE username = ?", (user,)).fetchone()
            if row:
                uid = row["id"]
                print(f"{user}: account exists")
            else:
                uid = conn.execute("INSERT INTO users (username, pw_hash) VALUES (?,?)", (user, hash_password(PASSWORD))).lastrowid
                print(f"{user}: account created")
            if conn.execute("SELECT 1 FROM user_state WHERE user_id = ?", (uid,)).fetchone() and not reset:
                print(f"{user}: keeps saved state (use --reset to replace)")
                continue
            state = demo_state(user, slugs)
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            conn.execute("INSERT INTO user_state VALUES (?,?,?) ON CONFLICT(user_id) DO UPDATE SET state=excluded.state, updated=excluded.updated",
                         (uid, db.dumps(state), stamp))
            n = db.sync_labels(conn, uid, state)
            names = " + ".join(p["name"] for p in state["profiles"].values())
            print(f"{user}: {names} ({n} labels)")


if __name__ == "__main__":
    main()
