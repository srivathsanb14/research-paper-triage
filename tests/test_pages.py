import json

import pytest

from scripts.build_pages import build
from triage import config
from triage.models import Paper
from triage.store import Store
from triage.transfer import export_profile, load_json, parse_papers, restore_profile


def test_pages_artifact_contains_only_public_app_files(tmp_path):
    out = tmp_path / "site"
    build(out)
    assert {p.name for p in out.iterdir()} == {"index.html", "styles.css", "js", "catalog", "quality-rules.json", ".nojekyll"}
    assert {p.name for p in (out / "js").iterdir()} == {p.name for p in (config.ROOT / "web" / "js").glob("*.js")}
    manifest = json.loads((out / "catalog" / "manifest.json").read_text())
    embeddings = json.loads((out / "catalog" / "embeddings.json").read_text())
    expected = {"manifest.json", "embeddings.json", *(a["file"] for a in manifest["areas"]), *(e["file"] for e in embeddings["areas"].values())}
    assert {p.name for p in (out / "catalog").iterdir()} == expected | {"proposals.json"}
    # Every published vector file belongs to a published paper shard of the same size.
    shards = {a["id"]: a for a in manifest["areas"]}
    for area, e in embeddings["areas"].items():
        assert e["papers_sha256"] == shards[area]["sha256"]
        assert (out / "catalog" / e["file"]).stat().st_size == e["count"] * (4 + embeddings["dim"])
    assert not any(p.suffix in (".db", ".py", ".toml") for p in out.rglob("*"))


def test_hugging_face_space_build_has_front_matter(tmp_path):
    build(tmp_path / "space", hf_space=True)
    readme = (tmp_path / "space" / "README.md").read_text()
    assert readme.startswith("---\n") and "sdk: static" in readme and "app_file: index.html" in readme


@pytest.mark.parametrize("row", [
    {"id": "a"},
    {"id": "a", "title": "Title", "authors": "A. Author"},
    {"id": "a", "title": "Title", "year": "2026"},
    {"id": "a", "title": "Title", "url": "javascript:alert(1)"},
    {"id": "a", "title": "Title", "pdf_url": "file:///etc/passwd"},
])
def test_import_rejects_invalid_records(row):
    with pytest.raises(ValueError):
        parse_papers([row])


def test_paper_collection_roundtrip(corpus):
    parsed = parse_papers(load_json(json.dumps([p.to_dict() for p in corpus]).encode()))
    assert [p.id for p in parsed] == [p.id for p in corpus]
    assert parsed[0].title == corpus[0].title
    with pytest.raises(ValueError, match="Duplicate"):
        parse_papers([corpus[0].to_dict(), corpus[0].to_dict()])


def test_profile_backup_preserves_personalization(store, corpus, rag_profile):
    store.save_profile(rag_profile)
    store.upsert_papers(corpus)
    store.add_to_pool(rag_profile.id, [p.id for p in corpus])
    store.add_seeds(rag_profile.id, [corpus[0].id])
    store.set_label(rag_profile.id, corpus[1].id, "READ", created_at="2026-09-01T12:00:00+00:00")
    store.log_feedback(rag_profile.id, corpus[2].id, "not_useful", score=0.5)
    store.set_setting(rag_profile.id, "cutoffs", {"read": 0.7, "skim": 0.3})
    store.set_setting(rag_profile.id, "budget", False)
    restored = restore_profile(store, export_profile(store, rag_profile))
    assert restored.id != rag_profile.id
    assert restored.name == "rag (imported 1)"
    assert restored.fingerprint() == rag_profile.fingerprint()
    assert store.pool_ids(restored.id) == store.pool_ids(rag_profile.id)
    assert store.seed_ids(restored.id) == [corpus[0].id]
    assert store.get_labels(restored.id) == {corpus[1].id: "READ"}
    assert store.get_label_events(restored.id)[0]["created_at"] == "2026-09-01T12:00:00+00:00"
    assert store.get_feedback(restored.id)[0]["action"] == "not_useful"
    assert store.get_setting(restored.id, "cutoffs") == {"read": 0.7, "skim": 0.3}
    assert store.get_setting(restored.id, "budget") is False


def test_backup_retains_labels_when_pool_was_cleared(store, corpus, rag_profile):
    store.save_profile(rag_profile)
    store.upsert_papers(corpus)
    store.set_label(rag_profile.id, corpus[0].id, "READ")
    restored = restore_profile(store, export_profile(store, rag_profile))
    assert store.pool_ids(restored.id) == []
    assert store.get_labels(restored.id) == {corpus[0].id: "READ"}


@pytest.mark.parametrize("bad_event", [
    {"paper_id": "missing", "action": "useful"},
    {"paper_id": [], "action": "useful"},
    {"paper_id": "a", "action": []},
    {"paper_id": "a", "action": "correct", "value": "UNKNOWN"},
    {"paper_id": "a", "action": "useful", "created_at": "not a date"},
])
def test_invalid_backup_does_not_write_partial_profile(store, rag_profile, bad_event):
    store.save_profile(rag_profile)
    store.upsert_papers([Paper(id="a", title="A paper")])
    store.add_to_pool(rag_profile.id, ["a"])
    backup = json.loads(export_profile(store, rag_profile))
    backup["feedback"] = [bad_event]
    with pytest.raises(ValueError):
        restore_profile(store, json.dumps(backup))
    assert len(store.list_profiles()) == 1


def test_browser_database_survives_connection_restart(tmp_path, monkeypatch, rag_profile):
    monkeypatch.setattr(config, "BROWSER_MODE", True)
    path = tmp_path / "browser.db"
    original = Store(path)
    original.save_profile(rag_profile)
    original.log_feedback(rag_profile.id, "a", "useful")
    with original._conn() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    reopened = Store(path)
    assert reopened.get_profile("rag").id == rag_profile.id
    assert reopened.get_feedback(rag_profile.id)[0]["action"] == "useful"
