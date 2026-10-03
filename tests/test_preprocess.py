from triage.preprocess import clean_text


def test_clean_text_handles_latex_html_whitespace():
    assert clean_text("A \\textbf{bold}\n  claim &amp; $x$") == "A bold claim & x"
    assert clean_text(None) == ""
    assert clean_text("Pawe{\\l} and Andr\\'e") == "Paweł and Andre"
