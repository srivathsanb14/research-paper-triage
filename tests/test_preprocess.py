from triage.models import Paper
from triage.preprocess import clean_text, dedupe, is_usable, normalize_arxiv_id, phrase_in_text


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


def test_dedupe_merges_duplicates():
    a = Paper(id="s2:1", title="Same Title!", abstract="short abstract here but long enough to be usable ok", venue="EMNLP", citation_count=3, source="semantic_scholar")
    b = Paper(id="arxiv:1", title="Same title", abstract="a much longer abstract " * 5, venue="arXiv preprint", url="u", source="arxiv")
    [m] = dedupe([a, b])
    assert m.id == "arxiv:1" and m.venue == "EMNLP" and m.citation_count == 3


def test_is_usable():
    assert not is_usable(Paper(id="x", title="T", abstract="short"))
    assert is_usable(Paper(id="x", title="T", abstract="long " * 20))
