from triage import quality
from triage.models import Paper

LONG = " ".join(["word"] * 120)


def test_peer_review_comes_from_venue_and_version():
    assert quality.peer_reviewed(Paper("a", "T", venue_type="journal", version="publishedVersion")) is True
    assert quality.peer_reviewed(Paper("a", "T", venue_type="repository")) is False
    assert quality.peer_reviewed(Paper("a", "T", source="arxiv")) is False
    assert quality.peer_reviewed(Paper("a", "T", venue_type="journal", work_type="preprint")) is False
    assert quality.peer_reviewed(Paper("a", "T")) is None


def test_abstract_cues_for_code_data_and_study_design():
    s = quality.signals(Paper("a", "Trial", f"Code is available at github.com/a/b. Data are publicly available. "
                                         f"A randomized controlled trial with n = 4,200 participants. {LONG}"))
    assert s.code and s.data
    assert {"Randomized trial", "Large sample"} <= set(s.evidence)
    plain = quality.signals(Paper("b", "Notes on poetry", LONG))
    assert not (plain.code or plain.data or plain.evidence)


def test_cautions_and_levels():
    thin = quality.signals(Paper("a", "Groundbreaking", "An unprecedented, groundbreaking breakthrough.", source="arxiv"))
    assert thin.thin_abstract and thin.promotional and thin.level == "limited"
    strong = quality.signals(Paper("b", "A survey of X", f"We release code on github.com/x. {LONG}", venue_type="journal",
                                   venue_core=True, oa_status="gold", fwci=2.0, work_type="review"))
    assert strong.review and strong.level == "strong"
    # Zero references usually means missing metadata, so it is not flagged.
    assert not quality.signals(Paper("c", "T", LONG, references_count=0)).few_references
    assert quality.signals(Paper("c", "T", LONG, references_count=3)).few_references


def test_published_catalog_carries_signal_metadata():
    from triage import catalog
    papers = [p for a in catalog.manifest()["areas"] for p in catalog.read_shard(a)]
    assert sum(p.venue_type == "journal" for p in papers) > len(papers) / 4
    assert any(p.fwci is not None for p in papers)
    levels = {quality.signals(p).level for p in papers}
    assert levels == {"strong", "moderate", "limited"}
