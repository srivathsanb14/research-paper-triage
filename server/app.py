"""FastAPI app: accounts, per-user state sync, starter label sets, and the built static site."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db, seed

COOKIE = "pt_session"
SESSION_DAYS = 30
MAX_STATE_BYTES = 8 * 1024 * 1024
USERNAME = re.compile(r"^[A-Za-z0-9_.@-]{3,40}$")
SITE = db.ROOT / "_site"

# Login throttle: at most 8 failures per 10 minutes per (client, username).
_failures: dict[str, deque] = defaultdict(deque)


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, digest_hex = stored.split("$")
        expected = hash_password(password, bytes.fromhex(salt_hex)).split("$")[2]
    except ValueError:
        return False
    return hmac.compare_digest(expected, digest_hex)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Credentials(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=256)


class StateBody(BaseModel):
    state: dict
    base: str | None = None  # `updated` stamp the client last saw; a mismatch means another device wrote since


def create_app() -> FastAPI:
    db.init()
    seed.load()
    app = FastAPI(title="Paper Triage accounts", docs_url=None, redoc_url=None)

    def start_session(conn, response: Response, user_id: int) -> None:
        token = secrets.token_urlsafe(32)
        expires = time.time() + SESSION_DAYS * 86400
        conn.execute("DELETE FROM sessions WHERE expires < ?", (time.time(),))
        conn.execute("INSERT INTO sessions VALUES (?,?,?)", (token_hash(token), user_id, expires))
        response.set_cookie(COOKIE, token, max_age=SESSION_DAYS * 86400, httponly=True, samesite="lax",
                            secure=os.environ.get("TRIAGE_SECURE_COOKIES") == "1", path="/")

    def current_user(request: Request) -> dict | None:
        token = request.cookies.get(COOKIE)
        if not token:
            return None
        with db.session() as conn:
            row = conn.execute(
                "SELECT u.id, u.username FROM sessions s JOIN users u ON u.id = s.user_id "
                "WHERE s.token_hash = ? AND s.expires > ?", (token_hash(token), time.time())).fetchone()
        return dict(row) if row else None

    def require_user(user: dict | None = Depends(current_user)) -> dict:
        if not user:
            raise HTTPException(401, "Sign in first.")
        return user

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # Cross-site forms can't send JSON without a CORS preflight, which we never grant.
        if request.url.path.startswith("/api/") and request.method in ("POST", "PUT", "DELETE"):
            if request.method != "DELETE" and "application/json" not in request.headers.get("content-type", ""):
                return PlainTextResponse("Expected application/json.", status_code=415)
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    @app.get("/api/me")
    def me(user: dict | None = Depends(current_user)):
        return {"backend": True, "user": {"username": user["username"]} if user else None}

    @app.post("/api/register")
    def register(body: Credentials, response: Response):
        username = body.username.strip()
        if not USERNAME.match(username):
            raise HTTPException(422, "Username must be 3–40 letters, digits or . _ @ -")
        if len(body.password) < 8:
            raise HTTPException(422, "Password must be at least 8 characters.")
        with db.session() as conn:
            if conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
                raise HTTPException(409, "That username is taken.")
            uid = conn.execute("INSERT INTO users (username, pw_hash) VALUES (?,?)",
                               (username, hash_password(body.password))).lastrowid
            start_session(conn, response, uid)
        return {"user": {"username": username}}

    @app.post("/api/login")
    def login(body: Credentials, request: Request, response: Response):
        username = body.username.strip()
        key = f"{request.client.host if request.client else '?'}|{username.lower()}"
        recent = _failures[key]
        while recent and recent[0] < time.time() - 600:
            recent.popleft()
        if len(recent) >= 8:
            raise HTTPException(429, "Too many attempts. Try again in a few minutes.")
        with db.session() as conn:
            row = conn.execute("SELECT id, username, pw_hash FROM users WHERE username = ?", (username,)).fetchone()
            ok = verify_password(body.password, row["pw_hash"] if row else hash_password("x"))
            if not (row and ok):
                recent.append(time.time())
                raise HTTPException(401, "Wrong username or password.")
            start_session(conn, response, row["id"])
        _failures.pop(key, None)
        return {"user": {"username": row["username"]}}

    @app.post("/api/logout")
    def logout(request: Request, response: Response):
        token = request.cookies.get(COOKIE)
        if token:
            with db.session() as conn:
                conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash(token),))
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    @app.get("/api/state")
    def get_state(user: dict = Depends(require_user)):
        with db.session() as conn:
            row = conn.execute("SELECT state, updated FROM user_state WHERE user_id = ?", (user["id"],)).fetchone()
        return {"state": json.loads(row["state"]) if row else None, "updated": row["updated"] if row else None}

    @app.put("/api/state")
    def put_state(body: StateBody, user: dict = Depends(require_user)):
        state = body.state
        if state.get("version") != 1 or not isinstance(state.get("profiles"), dict):
            raise HTTPException(422, "Unrecognised state format.")
        blob = db.dumps(state)
        if len(blob.encode()) > MAX_STATE_BYTES:
            raise HTTPException(413, "State is too large.")
        stamp = now_iso()
        with db.session() as conn:
            row = conn.execute("SELECT updated FROM user_state WHERE user_id = ?", (user["id"],)).fetchone()
            if row and body.base is not None and body.base != row["updated"]:
                raise HTTPException(409, "Your account changed on another device. Reload to get the latest.")
            conn.execute("INSERT INTO user_state VALUES (?,?,?) ON CONFLICT(user_id) DO UPDATE SET state=excluded.state, updated=excluded.updated",
                         (user["id"], blob, stamp))
            n = db.sync_labels(conn, user["id"], state)
        return {"updated": stamp, "labels": n}

    @app.get("/api/labels")
    def export_labels(user: dict = Depends(require_user)):
        with db.session() as conn:
            rows = conn.execute("SELECT profile_id, profile_name, paper_id, label, labeler, labeled_at FROM labels "
                                "WHERE user_id = ? ORDER BY profile_name, labeled_at", (user["id"],)).fetchall()
        text = "".join(db.dumps(dict(r)) + "\n" for r in rows)
        return PlainTextResponse(text, media_type="application/jsonl",
                                 headers={"Content-Disposition": 'attachment; filename="my-labels.jsonl"'})

    @app.get("/api/seed-sets")
    def seed_sets(user: dict = Depends(require_user)):
        with db.session() as conn:
            rows = conn.execute("SELECT slug, name, kind, n FROM seed_sets ORDER BY kind, name").fetchall()
        return {"sets": [dict(r) for r in rows]}

    @app.get("/api/seed-sets/{slug}")
    def seed_set(slug: str, user: dict = Depends(require_user)):
        with db.session() as conn:
            head = conn.execute("SELECT slug, name, kind, profile FROM seed_sets WHERE slug = ?", (slug,)).fetchone()
            if not head:
                raise HTTPException(404, "No such label set.")
            rows = conn.execute("SELECT paper_id, label, labeler, labeled_at FROM seed_labels WHERE slug = ?", (slug,)).fetchall()
        return {"slug": head["slug"], "name": head["name"], "kind": head["kind"], "profile": json.loads(head["profile"]),
                "labels": [dict(r) for r in rows]}

    if SITE.is_dir():  # the built static site, so one process serves app and API
        app.mount("/", StaticFiles(directory=SITE, html=True), name="site")
    return app


app = create_app() if os.environ.get("TRIAGE_NO_AUTOLOAD") != "1" else None
