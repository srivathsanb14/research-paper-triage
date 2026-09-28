import json

import pytest
import requests

from triage import sources
from triage.models import Paper
from triage.preprocess import clean_paper, clean_text, dedupe, is_usable, normalize_arxiv_id, phrase_in_text

ARXIV_XML = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.01234v2</id>
    <published>2024-01-03T18:00:00Z</published>
    <title>A \\textbf{Retrieval}-Augmented
      Generation Study</title>
    <summary>  We study $x^2$ retrieval &amp; generation   in depth, with a long enough abstract.</summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <arxiv:journal_ref>ACL 2024</arxiv:journal_ref>
    <link href="http://arxiv.org/pdf/2401.01234v2" rel="related" type="application/pdf" title="pdf"/>
    <category term="cs.CL"/><category term="cs.IR"/>
  </entry>
</feed>"""

RSS_XML = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>oai:arXiv.org:2609.30287v1</id>
    <title>Sparse Probing of BERT</title>
    <summary>arXiv:2609.30287v1 Announce Type: new
Abstract: We study which neurons in a frozen encoder support detection of generated text.</summary>
    <category term="cs.CL"/>
    <published>2026-09-28T00:00:00-04:00</published>
    <arxiv:announce_type>new</arxiv:announce_type>
    <dc:creator>Pawe{\\l} Blicharz, Mi{\\l}osz Grunwald</dc:creator>
  </entry>
  <entry>
    <id>oai:arXiv.org:2501.00001v3</id>
    <title>Old paper replaced</title>
    <summary>arXiv:2501.00001v3 Announce Type: replace
Abstract: A replaced paper with a sufficiently long abstract for testing purposes.</summary>
    <arxiv:announce_type>replace</arxiv:announce_type>
    <dc:creator>X Y</dc:creator>
  </entry>
</feed>"""


def test_clean_text_handles_latex_html_whitespace():
    assert clean_text("A \\textbf{bold}\n  claim &amp; $x$") == "A bold claim & x"
    assert clean_text(None) == ""
    assert clean_text("Pawe{\\l} and Andr\\'e") == "Paweł and Andre"


def test_normalize_arxiv_id():
    assert normalize_arxiv_id("http://arxiv.org/abs/2401.01234v2") == "2401.01234"
    assert normalize_arxiv_id("hep-th/9901001v1") == "hep-th/9901001"


def test_phrase_matching_hyphen_plural_acronym():
    assert phrase_in_text("retrieval augmented generation", "retrieval-augmented generations help")
    assert phrase_in_text("retrieval augmented generation", "Our RAG system")
    assert phrase_in_text("LLM agents", "LLM-based agent")
    assert not phrase_in_text("question answering", "questions about the answer key")
    assert not phrase_in_text("", "anything")


def test_parse_arxiv_feed():
    [p] = sources.parse_arxiv_feed(ARXIV_XML)
    assert p.id == "arxiv:2401.01234"
    assert p.pdf_url.endswith("2401.01234v2")
    p = clean_paper(p)
    assert p.title == "A Retrieval-Augmented Generation Study"
    assert p.abstract.startswith("We study x^2 retrieval & generation in depth")
    assert p.authors == ["Ada Lovelace", "Alan Turing"]
    assert p.venue == "ACL 2024" and p.year == 2024 and p.categories == ["cs.CL", "cs.IR"]


def test_parse_arxiv_error_entry():
    xml = ARXIV_XML.replace("http://arxiv.org/abs/2401.01234v2", "http://arxiv.org/api/errors#bad")
    with pytest.raises(sources.SourceError):
        sources.parse_arxiv_feed(xml)
    with pytest.raises(sources.SourceError):
        sources.parse_arxiv_feed("<not xml")


def test_parse_arxiv_rss_filters_announce_type():
    papers = sources.parse_arxiv_rss(RSS_XML)
    assert [p.id for p in papers] == ["arxiv:2609.30287"]
    p = clean_paper(papers[0])
    assert p.abstract.startswith("We study which neurons")
    assert p.authors == ["Paweł Blicharz", "Miłosz Grunwald"]
    both = sources.parse_arxiv_rss(RSS_XML, ("new", "replace"))
    assert len(both) == 2


def test_build_arxiv_query():
    q = sources.build_arxiv_query(["graph neural network", "GNN", " "], ["cs.LG"])
    assert q == '(all:"graph neural network" OR all:GNN) AND (cat:cs.LG)'
    assert sources.build_arxiv_query([], None) == ""


def test_parse_s2_item():
    item = {
        "paperId": "abc",
        "externalIds": {"ArXiv": "2401.00001"},
        "title": "T",
        "abstract": "x" * 50,
        "authors": [{"name": "N"}],
        "venue": "NeurIPS",
        "year": 2024,
        "citationCount": 7,
        "openAccessPdf": {"url": "http://pdf"},
    }
    p = sources.parse_s2_item(item)
    assert p.id == "arxiv:2401.00001" and p.url == "https://arxiv.org/abs/2401.00001" and p.citation_count == 7
    assert sources.parse_s2_item({"title": None}) is None
    item2 = dict(item, externalIds=None, openAccessPdf=None)
    assert sources.parse_s2_item(item2).id == "s2:abc"


def test_dedupe_merges_across_sources():
    a = Paper(id="s2:1", title="Same Title!", abstract="short abstract here but long enough to be usable ok", venue="EMNLP", citation_count=3, source="semantic_scholar")
    b = Paper(id="arxiv:1", title="Same title", abstract="a much longer abstract " * 5, venue="arXiv preprint", url="u", source="arxiv")
    [m] = dedupe([a, b])
    assert m.id == "arxiv:1" and m.venue == "EMNLP" and m.citation_count == 3


def test_is_usable():
    assert not is_usable(Paper(id="x", title="T", abstract="short"))
    assert is_usable(Paper(id="x", title="T", abstract="long " * 20))


class FakeResp:
    def __init__(self, status, text="", headers=None):
        self.status_code, self.text, self.headers = status, text, headers or {}

    def json(self):
        return json.loads(self.text)


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get(self, *a, **k):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_retry_then_success(monkeypatch):
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    s = FakeSession([FakeResp(429), requests.ConnectionError("boom"), FakeResp(200, "ok")])
    r = sources._get_with_retry("u", {}, session=s)
    assert r.text == "ok" and s.calls == 3


def test_retry_gives_up_and_non_retryable(monkeypatch):
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    with pytest.raises(sources.SourceError, match="after 2 attempts"):
        sources._get_with_retry("u", {}, retries=2, session=FakeSession([FakeResp(503), FakeResp(503)]))
    with pytest.raises(sources.SourceError, match="HTTP 400"):
        sources._get_with_retry("u", {}, session=FakeSession([FakeResp(400, "bad")]))


def test_semantic_scholar_search(monkeypatch):
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    body = json.dumps({"total": 1, "data": [{"paperId": "p1", "title": "Graph nets", "abstract": "a" * 60, "authors": []}]})
    papers = sources.search_semantic_scholar(["graph"], 10, session=FakeSession([FakeResp(200, body)]))
    assert [p.id for p in papers] == ["s2:p1"]


def test_fetch_collects_warnings(monkeypatch):
    def boom(*a, **k):
        raise sources.SourceError("down")

    monkeypatch.setattr(sources, "fetch_arxiv_new", boom)
    monkeypatch.setattr(sources, "search_semantic_scholar", lambda *a, **k: [Paper(id="s2:1", title="T", abstract="x" * 50)])
    papers, warns = sources.fetch("both", ["k"], ["cs.CL"])
    assert len(papers) == 1 and warns == ["arxiv_new: down"]


def test_sample_dataset_loads():
    papers = sources.load_sample_papers()
    assert len(papers) > 100
    assert all(p.title and p.abstract for p in papers)
    assert len({p.id for p in papers}) == len(papers)


OAI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
<record><header><identifier>oai:arXiv.org:2609.00001</identifier></header><metadata>
<arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>2609.00001</id><created>2026-09-20</created>
<authors><author><keyname>Lovelace</keyname><forenames>Ada</forenames></author></authors>
<title>Dense Retrieval at Scale</title><categories>cs.IR cs.CL</categories>
<abstract>We scale dense retrieval to billions of passages with a new index structure.</abstract></arXiv>
</metadata></record>
<resumptionToken>abc123</resumptionToken></ListRecords></OAI-PMH>"""


def test_parse_arxiv_oai():
    papers, token = sources.parse_arxiv_oai(OAI_XML)
    assert token == "abc123"
    [p] = papers
    assert p.id == "arxiv:2609.00001" and p.authors == ["Ada Lovelace"] and p.categories == ["cs.IR", "cs.CL"]
    assert p.published == "2026-09-20"


def test_seed_identifier_and_bibtex_parsing():
    from triage.seeds import parse_bibtex, parse_identifiers

    arxiv, dois = parse_identifiers(
        "https://arxiv.org/abs/2005.11401v4\n10.18653/v1/2020.emnlp-main.550\ndoi:10.48550/arXiv.2104.08663\nhep-th/9901001"
    )
    assert arxiv == ["2005.11401", "2104.08663", "hep-th/9901001"] and dois == ["10.18653/v1/2020.emnlp-main.550"]
    bib = parse_bibtex('@article{k, title={A {Nested} Title}, author = "Doe, Jane and Roe, R.", year = 2021}\n@comment{x}')
    assert bib == [{"title": "A {Nested} Title", "author": "Doe, Jane and Roe, R.", "year": "2021"}]


def test_full_text_html_extraction():
    from triage.fulltext import extract_from_html

    html_doc = (
        "<html><h2 class='ltx_title ltx_title_section'>1 Introduction</h2><p>We study retrieval.</p>"
        "<math>x^2</math><h2 class='ltx_title ltx_title_section'>2 Method</h2><p>Details.</p>"
        "<h2 class='ltx_title ltx_title_section'>5 Conclusion</h2><p>It works.</p></html>"
    )
    text = extract_from_html(html_doc)
    assert "We study retrieval." in text and "It works." in text and "Details" not in text and "x^2" not in text
