"""Create the demo accounts and give each a profile with labels from the shared label dataset.

    python -m server.demo_users            # create what is missing; never overwrites a user's saved work
    python -m server.demo_users --reset    # replace the demo users' saved state again

sri (materials discovery), ishaan (robotics) and chris (AI and design) all use password "demo",
which is shorter than registration allows, so the accounts are created directly. Demo only: don't
use this on a public host. Each profile and its labels come from data/labels/ (labels.jsonl rows
with origin "rule", produced by scripts/simulate_labels.py).
"""

import argparse
from datetime import datetime, timezone

from . import db, seed
from .app import hash_password

PASSWORD = "demo"
USERS = {"sri": "materials-discovery", "ishaan": "robotics", "chris": "ai-and-design"}  # user → profile slug


def demo_state(user: str, slug: str) -> dict:
    found = seed.load()[slug]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    pid = f"demo-{slug}"
    profile = {
        "id": pid, **found["profile"], "hours": 3, "cutoffs": {"read": 0.62, "skim": 0.4}, "budget": True,
        "group": True, "seeds": [], "starter": [], "feedback": [], "extra": [], "seen": [], "lastVisit": None, "created": now,
        "labels": {r["paper_id"]: {"label": r["label"], "at": r["labeled_at"]} for r in found["labels"]},
    }
    return {"version": 1, "active": pid, "owner": user, "settings": {"semantic": True}, "profiles": {pid: profile}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    reset = parser.parse_args().reset
    db.init()
    with db.session() as conn:
        for user, slug in USERS.items():
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
            state = demo_state(user, slug)
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            conn.execute("INSERT INTO user_state VALUES (?,?,?) ON CONFLICT(user_id) DO UPDATE SET state=excluded.state, updated=excluded.updated",
                         (uid, db.dumps(state), stamp))
            n = db.sync_labels(conn, uid, state)
            print(f"{user}: profile “{state['profiles'][next(iter(state['profiles']))]['name']}” with {n} labels")


if __name__ == "__main__":
    main()
