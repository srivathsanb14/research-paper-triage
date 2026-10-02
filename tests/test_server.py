"""Accounts backend: auth, per-user state + label sync, seed label sets."""

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from server import app as server_app  # noqa: E402
from server import db  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "users.db")
    return TestClient(server_app.create_app())


def state_with(labels):
    return {"version": 1, "active": "p1", "settings": {}, "profiles": {"p1": {"name": "Mine", "labels": labels, "feedback": []}}}


def test_register_login_logout(client):
    assert client.get("/api/me").json() == {"backend": True, "user": None}
    assert client.get("/api/state").status_code == 401
    assert client.post("/api/register", json={"username": "ab", "password": "longenough"}).status_code == 422
    assert client.post("/api/register", json={"username": "ada", "password": "short"}).status_code == 422
    assert client.post("/api/register", json={"username": "ada", "password": "correct horse"}).status_code == 200
    assert client.get("/api/me").json()["user"] == {"username": "ada"}
    assert client.post("/api/logout", json={}).status_code == 200
    assert client.get("/api/me").json()["user"] is None
    assert client.post("/api/register", json={"username": "ADA", "password": "another one!"}).status_code == 409
    assert client.post("/api/login", json={"username": "ada", "password": "wrong password"}).status_code == 401
    assert client.post("/api/login", json={"username": "ada", "password": "correct horse"}).status_code == 200
    assert client.get("/api/me").json()["user"]["username"] == "ada"


def test_login_is_throttled(client):
    client.post("/api/register", json={"username": "bob", "password": "correct horse"})
    client.post("/api/logout", json={})
    server_app._failures.clear()
    codes = [client.post("/api/login", json={"username": "bob", "password": "nope nope"}).status_code for _ in range(10)]
    assert codes[:8] == [401] * 8 and codes[8:] == [429, 429]
    server_app._failures.clear()


def test_json_only_for_writes(client):
    assert client.post("/api/login", content="username=a&password=b", headers={"content-type": "application/x-www-form-urlencoded"}).status_code == 415


def test_state_roundtrip_and_label_table(client):
    client.post("/api/register", json={"username": "ada", "password": "correct horse"})
    assert client.get("/api/state").json() == {"state": None, "updated": None}
    labels = {"doi:10.1/a": {"label": "READ", "at": "2026-10-01T00:00:00Z"}, "doi:10.1/b": {"label": "SKIP"}, "doi:10.1/c": {"label": "bogus"}}
    first = client.put("/api/state", json={"state": state_with(labels)}).json()
    assert first["labels"] == 2  # the invalid label is ignored
    got = client.get("/api/state").json()
    assert got["state"]["profiles"]["p1"]["labels"]["doi:10.1/a"]["label"] == "READ" and got["updated"] == first["updated"]
    rows = [json.loads(line) for line in client.get("/api/labels").text.splitlines()]
    assert {r["paper_id"] for r in rows} == {"doi:10.1/a", "doi:10.1/b"} and rows[0]["profile_name"] == "Mine"
    # removing a label removes its row
    client.put("/api/state", json={"state": state_with({"doi:10.1/a": {"label": "READ"}}), "base": first["updated"]})
    assert len(client.get("/api/labels").text.splitlines()) == 1
    assert client.put("/api/state", json={"state": {"version": 2}}).status_code == 422


def test_stale_write_is_rejected_and_users_are_isolated(client):
    client.post("/api/register", json={"username": "ada", "password": "correct horse"})
    first = client.put("/api/state", json={"state": state_with({})}).json()["updated"]
    assert client.put("/api/state", json={"state": state_with({}), "base": "2000-01-01T00:00:00Z"}).status_code == 409
    assert client.put("/api/state", json={"state": state_with({}), "base": first}).status_code == 200
    client.post("/api/logout", json={})
    client.post("/api/register", json={"username": "bob", "password": "correct horse"})
    assert client.get("/api/state").json()["state"] is None
    assert client.get("/api/labels").text == ""


def test_seed_sets_come_from_the_label_dataset(client):
    client.post("/api/register", json={"username": "ada", "password": "correct horse"})
    sets = {s["slug"]: s for s in client.get("/api/seed-sets").json()["sets"]}
    assert sets["rag-llm-evaluation"]["origin"] == "manual" and sets["robotics"]["origin"] == "synthetic"
    full = client.get("/api/seed-sets/rag-llm-evaluation").json()
    assert full["profile"]["keywords"] and len(full["labels"]) == sets["rag-llm-evaluation"]["n"]
    assert {x["label"] for x in full["labels"]} <= {"READ", "SKIM", "SKIP"}
    assert all("labeler" not in x for x in full["labels"])
    assert client.get("/api/seed-sets/nope").status_code == 404
    client.post("/api/logout", json={})
    assert client.get("/api/seed-sets").status_code == 401


def test_users_database_holds_only_user_data(tmp_path, monkeypatch):
    import sqlite3

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "users.db")
    server_app.create_app()
    tables = {r[0] for r in sqlite3.connect(db.DB_PATH).execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"users", "sessions", "user_state", "labels"}
    assert "labeler" not in {r[1] for r in sqlite3.connect(db.DB_PATH).execute("PRAGMA table_info(labels)")}


def test_demo_users_have_a_synthetic_and_a_manual_profile(tmp_path, monkeypatch):
    from server import demo_users, seed

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "users.db")
    monkeypatch.setattr("sys.argv", ["demo_users"])
    demo_users.main()
    demo_users.main()  # idempotent
    client = TestClient(server_app.create_app())
    for user, (own, manual) in demo_users.USERS.items():
        assert client.post("/api/login", json={"username": user, "password": "demo"}).status_code == 200
        state = client.get("/api/state").json()["state"]
        assert set(state["profiles"]) == {f"demo-{own}", f"demo-{manual}"} and state["active"] == f"demo-{own}"
        assert seed.load()[own]["origin"] == "synthetic" and seed.load()[manual]["origin"] == "manual"
        for prof in state["profiles"].values():
            assert {x["label"] for x in prof["labels"].values()} == {"READ", "SKIM", "SKIP"}
            assert all("labeler" not in x for x in prof["labels"].values())
        total = sum(len(p["labels"]) for p in state["profiles"].values())
        assert len(client.get("/api/labels").text.splitlines()) == total
