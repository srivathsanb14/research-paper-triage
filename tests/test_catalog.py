import io
import sys
from types import SimpleNamespace

import pytest

from scripts import build_catalog
from triage import catalog
from triage.models import Paper
from triage.transfer import export_profile, restore_profile


def test_areas_cover_every_openalex_field_once():
    fields = [field for _, fields in catalog.AREAS.values() for field in fields]
    assert sorted(fields) == list(range(11, 37))


def test_balanced_selection_deduplicates_without_starving_small_fields():
    groups = [[Paper("a", "First"), Paper("b", "Second"), Paper("c", "Third")],
              [Paper("d", "Small field"), Paper("other", "First")]]
    assert [p.id for p in catalog.balanced_papers(groups, 4)] == ["a", "d", "b", "c"]


def test_openalex_conversion_reconstructs_abstract_and_stable_doi():
    text = "Climate and agriculture " + " ".join(f"word{i}" for i in range(35))
    index = {}
    for i, word in enumerate(text.split()):
        index.setdefault(word, []).append(i)
    work = {"id": "https://openalex.org/W123", "doi": "https://doi.org/10.123/ABC", "title": "Climate research",
            "abstract_inverted_index": index, "primary_topic": {"field": {"display_name": "Earth Sciences"}}}
    paper = build_catalog.paper_from_work(work)
    assert paper.id == "doi:10.123/abc"
    assert paper.abstract == text
    assert paper.categories == ["Earth Sciences"]
    assert paper.source == "openalex"
    assert build_catalog.paper_from_work({**work, "abstract_inverted_index": None}) is None


def test_failed_refresh_keeps_previous_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text('{"previous": true}')
    monkeypatch.setattr(build_catalog, "fetch_field", lambda *args: (_ for _ in ()).throw(RuntimeError("quota")))
    with pytest.raises(RuntimeError, match="quota"):
        build_catalog.build(tmp_path)
    assert path.read_text() == '{"previous": true}'


def test_published_catalog_has_real_papers_and_verified_shards():
    info = catalog.manifest()
    assert {entry["id"] for entry in info["areas"]} == set(catalog.AREAS)
    for entry in info["areas"]:
        papers = catalog.read_shard(entry)
        assert len(papers) == entry["count"] >= 30
        assert all(p.source == "openalex" and len(p.abstract.split()) >= 30 for p in papers)
        assert all(info["since"] <= p.published <= info["until"] for p in papers)
    selected = catalog.load_selection(list(catalog.AREAS), 300)
    assert len(selected) == 300
    assert len({p.id for p in selected}) == 300


def test_browser_loads_only_selected_same_site_shards(monkeypatch):
    requests = []
    original = catalog.CATALOG_DIR
    def open_url(url):
        requests.append(url)
        return io.StringIO((original / url.rsplit("/", 1)[-1]).read_text())
    monkeypatch.setitem(sys.modules, "pyodide.http", SimpleNamespace(open_url=open_url))
    monkeypatch.setattr(catalog.sys, "platform", "emscripten")
    monkeypatch.setenv("TRIAGE_CATALOG_URL", "https://example.test/repository/catalog/")
    assert len(catalog.load_selection(["humanities"], 100)) == 100
    assert len(requests) == 1
    assert requests[0].startswith("https://example.test/repository/catalog/humanities-")


def test_damaged_download_is_rejected(tmp_path, monkeypatch):
    entry = catalog.manifest()["areas"][0]
    (tmp_path / entry["file"]).write_text("[]")
    monkeypatch.setattr(catalog, "CATALOG_DIR", tmp_path)
    with pytest.raises(ValueError, match="reload"):
        catalog.read_shard(entry)


@pytest.mark.parametrize("selection", [[], ["unknown"], ["../secrets"]])
def test_bad_field_selection_is_rejected(selection):
    with pytest.raises(ValueError):
        catalog.load_selection(selection)


def test_catalog_preferences_survive_backup(store, rag_profile, corpus):
    store.save_profile(rag_profile)
    store.upsert_papers(corpus)
    store.add_to_pool(rag_profile.id, [p.id for p in corpus])
    store.set_setting(rag_profile.id, "catalog_fields", ["humanities", "life"])
    restored = restore_profile(store, export_profile(store, rag_profile))
    assert store.get_setting(restored.id, "catalog_fields") == ["humanities", "life"]


def test_new_browser_visitor_can_start_without_account_or_import(tmp_path, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from triage import config
    from triage.store import Store

    monkeypatch.setattr(config, "BROWSER_MODE", True)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "visitor.db")
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "tfidf")
    st.cache_resource.clear()
    st.cache_data.clear()
    app = AppTest.from_file(config.ROOT / "app.py", default_timeout=45)
    app.query_params["view"] = "ALL"
    try:
        app.run()
        assert not app.exception
        assert Store(config.DB_PATH).list_profiles() == []
        next(w for w in app.text_area if w.label == "What are you interested in?").set_value("Environmental history and society")
        next(w for w in app.multiselect if w.label == "Research fields").set_value(["humanities", "environment"])
        next(w for w in app.select_slider if w.label == "Papers to load").set_value(100)
        next(b for b in app.button if b.label == "Find papers").click().run()
        assert not app.exception, [e.message for e in app.exception]
        saved = Store(config.DB_PATH)
        profile = saved.get_profile("My reading list")
        assert len(saved.pool_ids(profile.id)) == 100
        assert saved.get_labels(profile.id) == {}
        assert saved.get_feedback(profile.id) == []
        assert saved.get_setting(profile.id, "catalog_fields") == ["humanities", "environment"]
    finally:
        st.cache_resource.clear()
        st.cache_data.clear()
