"""Create the demo accounts and give each a profile with simulated labels.

    python -m server.demo_users            # create what is missing; never overwrites a user's saved work
    python -m server.demo_users --reset    # replace the demo users' saved state with the mock data again

sri (materials discovery), ishaan (robotics) and chris (AI and design) all use password "demo",
which is shorter than registration allows, so the accounts are created directly. Demo only: don't
use this on a public host. Profiles and labels come from data/labels/synthetic/<user>.jsonl
(scripts/simulate_users.py); they are rule-generated, and the labeler field says so.
"""

import argparse
import json
from datetime import datetime, timezone

from . import db, seed
from .app import hash_password

PASSWORD = "demo"
USERS = {"sri": "materials-discovery", "ishaan": "robotics", "chris": "ai-and-design"}


def demo_state(user: str, slug: str) -> dict:
    path = db.ROOT / "data" / "labels" / "synthetic" / f"{user}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    first = rows[0]
    persona = seed.profile_defaults().get(first["profile_name"], {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    pid = f"demo-{slug}"
    profile = {
        "id": pid, "name": first["profile_name"], "description": first["profile_description"],
        "keywords": first["profile_keywords"], "focus": first["profile_focus"], "avoid": persona.get("avoid", []),
        "hours": 3, "fields": persona.get("fields", []), "cutoffs": {"read": 0.62, "skim": 0.4}, "budget": True,
        "group": True, "seeds": [], "starter": [], "feedback": [], "extra": [], "seen": [], "lastVisit": None, "created": now,
        "labels": {r["paper_id"]: {"label": r["label"], "at": r["labeled_at"], "labeler": "simulated"} for r in rows},
    }
    return {"version": 1, "active": pid, "owner": user, "settings": {"semantic": True, "labeler": ""}, "profiles": {pid: profile}}


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
