import pytest

from scripts import build_catalog
from triage import catalog
from triage.models import Paper


def test_areas_cover_every_openalex_field_once():
    fields = [field for _, fields in catalog.AREAS.values() for field in fields if field < 1000]
    assert sorted(fields) == list(range(11, 37))
    assert catalog.AREAS["ai"][1] == [1702, 1707]  # AI and computer vision subfields


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
        assert entry["fields"] == catalog.AREAS[entry["id"]][1]
        papers = catalog.read_shard(entry)
        assert len(papers) == entry["count"] >= 30
        assert all(p.source == "openalex" and len(p.abstract.split()) >= 30 for p in papers)
        assert all(info["since"] <= p.published <= info["until"] for p in papers)
    ids = [p.id for entry in info["areas"] for p in catalog.read_shard(entry)]
    assert len(ids) == len(set(ids)), "no paper appears in two areas"


def test_damaged_download_is_rejected(tmp_path, monkeypatch):
    entry = catalog.manifest()["areas"][0]
    (tmp_path / entry["file"]).write_text("[]")
    monkeypatch.setattr(catalog, "CATALOG_DIR", tmp_path)
    with pytest.raises(ValueError, match="damaged"):
        catalog.read_shard(entry)
