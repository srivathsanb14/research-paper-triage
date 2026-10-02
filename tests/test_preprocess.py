from triage.preprocess import clean_text, phrase_in_text


def test_clean_text_handles_latex_html_whitespace():
    assert clean_text("A \\textbf{bold}\n  claim &amp; $x$") == "A bold claim & x"
    assert clean_text(None) == ""
    assert clean_text("Pawe{\\l} and Andr\\'e") == "Paweł and Andre"


def test_phrase_matching_hyphen_plural_acronym():
    assert phrase_in_text("retrieval augmented generation", "retrieval-augmented generations help")
    assert phrase_in_text("retrieval augmented generation", "Our RAG system")
    assert phrase_in_text("LLM agents", "LLM-based agent")
    assert not phrase_in_text("question answering", "questions about the answer key")
    assert not phrase_in_text("", "anything")
