import json
from pathlib import Path

import pytest

from scripts.build_pages import build
from triage import config
from triage.transfer import load_json, parse_papers


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


def test_stylesheet_braces_are_balanced():
    """An unclosed rule or @media block silently swallows every rule after it."""
    css = (Path(__file__).resolve().parent.parent / "web" / "styles.css").read_text(encoding="utf-8")
    assert css.count("{") == css.count("}")
