"""Explanation generator (CONFIGURED).

In: paper + interests + label → Out: a 1-line reason.

Known risk from the design: an LLM "may invent overlap" between the paper and
the user's interests. Mitigations here:

1. Evidence is extracted deterministically first (keyword hits in title/abstract,
   shared distinctive terms, the abstract sentence closest to the profile).
2. The template explainer only ever states that evidence → grounded by construction.
3. The LLM explainer (optional, Claude) receives the evidence, must return the
   exact terms it relies on, and every cited term is verified against the paper
   text. Any unverifiable claim → the LLM output is rejected and the template
   reason is shown instead, flagged in the UI.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from . import config
from .models import InterestProfile, Paper, TriageResult
from .preprocess import phrase_in_text, split_sentences, stem_tokens, tokenize
from .relevance import keyword_matches

if TYPE_CHECKING:
    from .store import Store

log = logging.getLogger(__name__)

MAX_REASON_CHARS = 240


# ---------------------------------------------------------------- evidence


def document_frequencies(papers: list[Paper]) -> tuple[Counter, int]:
    df: Counter = Counter()
    for p in papers:
        df.update(set(stem_tokens(f"{p.title} {p.abstract}")))
    return df, max(1, len(papers))


def extract_evidence(
    paper: Paper,
    profile: InterestProfile,
    df: Counter | None = None,
    n_docs: int = 1,
) -> dict:
    kw_title, kw_body = keyword_matches(paper, profile.keywords)
    paper_text = f"{paper.title} {paper.abstract}"
    avoid_hits = [a for a in profile.avoid if phrase_in_text(a, paper_text)]

    # Distinctive profile words that also occur in the paper (stem match), ranked by IDF.
    profile_words = {}
    for w in tokenize(f"{profile.description} {profile.focus}"):
        profile_words.setdefault(stem_tokens(w)[0] if stem_tokens(w) else w, w)
    paper_stems = set(stem_tokens(paper_text))
    kw_stems = {s for k in profile.keywords for s in stem_tokens(k)}
    shared = []
    for stem, word in profile_words.items():
        if stem in paper_stems and stem not in kw_stems:
            idf = math.log((1 + n_docs) / (1 + (df or {}).get(stem, 0))) + 1
            shared.append((idf, word))
    shared_terms = [w for _, w in sorted(shared, reverse=True)[:4]]

    # Paper's own distinctive topic words (what it is *about*), for SKIM/SKIP reasons.
    title_terms = []
    for w in tokenize(paper.title):
        s = stem_tokens(w)
        if s and s[0] not in kw_stems and w not in title_terms:
            idf = math.log((1 + n_docs) / (1 + (df or {}).get(s[0], 0))) + 1
            title_terms.append((idf, w))
    topic_terms = [w for _, w in sorted(title_terms, reverse=True)[:3]]

    # Abstract sentence with most profile overlap.
    want = set(stem_tokens(" ".join([profile.description, profile.focus, *profile.keywords])))
    best, best_score = "", 0.0
    for s in split_sentences(paper.abstract):
        toks = stem_tokens(s)
        if toks:
            sc = len(want.intersection(toks)) / math.sqrt(len(toks))
            if sc > best_score:
                best, best_score = s, sc

    return {
        "keywords_in_title": kw_title,
        "keywords_in_abstract": kw_body,
        "shared_terms": shared_terms,
        "topic_terms": topic_terms,
        "avoid_hits": avoid_hits,
        "best_sentence": best,
    }


def _q(items: list[str], n: int = 2) -> str:
    items = items[:n]
    return ", ".join(f"“{i}”" for i in items)


def template_reason(label: str, ev: dict, features: dict[str, float] | None = None, focus: str = "") -> str:
    """One short, grounded sentence (at most two clauses). The label chip already says READ/SKIM/SKIP."""
    f = features or {}
    kt, ka, shared = ev["keywords_in_title"], ev["keywords_in_abstract"], ev["shared_terms"]
    if ev["avoid_hits"]:
        return f"Covers {_q(ev['avoid_hits'], 1)}, which you excluded."
    bits: list[str] = []
    if kt:
        bits.append(f"{_q(kt)} in the title")
    elif ka:
        bits.append(f"mentions {_q(ka)}")
    elif shared and label != "SKIP":
        bits.append(f"shares {_q(shared)} with your project")
    if focus and f.get("focus", 0) >= 0.6 and label != "SKIP":
        bits.append("close to your current focus")
    if f.get("feedback", 0) >= 0.3:
        bits.append("like papers you found relevant")
    elif f.get("feedback", 0) <= -0.3:
        bits.append("like papers you found not relevant")
    bits = bits[:2]
    if not bits:
        bits = [{
            "READ": "strong match to your project description",
            "SKIM": "related to your project, but not central",
            "SKIP": "little overlap with your interests",
        }[label]]
    elif label == "SKIP" and not any(b.startswith("like papers") for b in bits):
        bits = ["little overlap beyond " + bits[0].replace("mentions ", "").replace(" in the title", "")]
    reason = "; ".join(bits) + "."
    return reason[0].upper() + reason[1:]


# --------------------------------------------------------------------- LLM


SYSTEM_PROMPT = """You write one-line explanations for a research-paper triage tool.
A researcher gave their interests; a ranking model labelled a paper READ, SKIM or SKIP.
Explain the label in one sentence (max 30 words) addressed to the researcher ("you/your").

Rules:
- Only claim overlap that is supported by the paper's title/abstract. Never invent methods, datasets or results.
- Prefer concrete terms that literally appear in the paper text.
- In cited_terms, list every paper term your reason relies on, copied verbatim from the title or abstract (1-4 terms).
- For SKIP, say briefly why it is off-target; for SKIM, say what is relevant and what is not."""


def _llm_client():
    import anthropic

    return anthropic.Anthropic(timeout=config.LLM_TIMEOUT_S, max_retries=2)


def llm_available() -> bool:
    """True when the SDK is installed and a credential is configured."""
    import os

    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    from pathlib import Path

    # `ant auth login` profiles are picked up by the SDK without any env var.
    return bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or (Path.home() / ".config" / "anthropic").is_dir()
    )


def verify_llm_reason(reason: str, cited: list[str], paper: Paper, profile: InterestProfile) -> tuple[bool, str]:
    """Check the LLM didn't invent overlap. Returns (ok, why_not)."""
    if not reason or not reason.strip():
        return False, "empty reason"
    text = f"{paper.title} {paper.abstract} {paper.full_text[:5000]}"
    if not cited:
        return False, "no cited terms"
    missing = [t for t in cited if not phrase_in_text(t, text)]
    if missing:
        return False, f"cited terms not in paper: {missing}"
    # Any quoted phrase in the reason must appear in the paper or the profile.
    profile_text = " ".join([profile.description, profile.focus, *profile.keywords, *profile.avoid])
    for quoted in re.findall(r"[\"“]([^\"”]{3,60})[\"”]", reason):
        if not (phrase_in_text(quoted, text) or phrase_in_text(quoted, profile_text)):
            return False, f"quoted phrase not grounded: {quoted!r}"
    return True, ""


def llm_reason(paper: Paper, profile: InterestProfile, label: str, score: float, ev: dict) -> dict:
    """Call Claude for a one-line reason. Raises on API failure (caller falls back)."""
    import anthropic
    from pydantic import BaseModel

    class Reason(BaseModel):
        reason: str
        cited_terms: list[str]

    user = (
        f"RESEARCHER INTERESTS\nProject: {profile.description or '(none)'}\n"
        f"Keywords: {', '.join(profile.keywords) or '(none)'}\n"
        f"Current focus: {profile.focus or '(none)'}\n"
        f"Avoid: {', '.join(profile.avoid) or '(none)'}\n\n"
        f"PAPER\nTitle: {paper.title}\nAbstract: {paper.abstract[:3000]}\n\n"
        f"MODEL OUTPUT\nLabel: {label} (relevance score {score:.2f} on 0-1)\n"
        f"Verified evidence: keywords in title={ev['keywords_in_title']}, "
        f"keywords in abstract={ev['keywords_in_abstract']}, shared terms={ev['shared_terms']}, "
        f"avoided topics present={ev['avoid_hits']}"
    )
    client = _llm_client()
    resp = client.messages.parse(
        model=config.LLM_MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": config.LLM_EFFORT},
        output_format=Reason,
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("model declined")
    parsed = resp.parsed_output
    if parsed is None:
        raise RuntimeError(f"no structured output (stop_reason={resp.stop_reason})")
    reason = " ".join(parsed.reason.split())
    if len(reason) > MAX_REASON_CHARS:
        reason = reason[: MAX_REASON_CHARS - 1].rsplit(" ", 1)[0] + "…"
    return {"reason": reason, "cited_terms": [t.strip() for t in parsed.cited_terms if t.strip()]}


# ---------------------------------------------------------------- facade


def explanation_key(paper_id: str, profile: InterestProfile, label: str, mode: str) -> str:
    raw = f"{mode}|{config.LLM_MODEL if mode == 'llm' else ''}|{profile.fingerprint()}|{paper_id}|{label}"
    return hashlib.sha1(raw.encode()).hexdigest()


def explain(
    result: TriageResult,
    profile: InterestProfile,
    df: Counter | None = None,
    n_docs: int = 1,
    use_llm: bool = False,
    store: "Store | None" = None,
) -> dict:
    """Return {'reason', 'source': 'template'|'llm', 'grounded': bool, 'evidence', 'warning'}."""
    ev = extract_evidence(result.paper, profile, df, n_docs)
    template = template_reason(result.label, ev, result.features, profile.focus)
    base = {"reason": template, "source": "template", "grounded": True, "evidence": ev, "warning": ""}
    if not use_llm:
        return base

    key = explanation_key(result.paper.id, profile, result.label, "llm")
    if store and (cached := store.get_explanation(key)):
        return {**cached, "evidence": ev}
    try:
        out = llm_reason(result.paper, profile, result.label, result.score, ev)
    except Exception as e:  # network, auth, rate limit, refusal, parse → degrade gracefully
        import anthropic

        if isinstance(e, anthropic.AuthenticationError):
            msg = "LLM auth failed — check ANTHROPIC_API_KEY"
        elif isinstance(e, anthropic.RateLimitError):
            msg = "LLM rate-limited"
        elif isinstance(e, anthropic.APIConnectionError):
            msg = "LLM unreachable"
        else:
            msg = f"LLM error: {e.__class__.__name__}"
        log.warning("explanation LLM failed for %s: %s", result.paper.id, e)
        return {**base, "warning": msg}  # transient: not cached
    ok, why = verify_llm_reason(out["reason"], out["cited_terms"], result.paper, profile)
    if ok:
        data = {"reason": out["reason"], "source": "llm", "grounded": True, "warning": "", "cited_terms": out["cited_terms"]}
    else:
        data = {
            "reason": template,
            "source": "template",
            "grounded": True,
            "warning": f"LLM reason rejected ({why})",
            "rejected_llm_reason": out["reason"],
        }
    if store:
        store.put_explanation(key, data)
    return {**data, "evidence": ev}


def explain_many(
    results: list[TriageResult],
    profile: InterestProfile,
    pool: list[Paper],
    use_llm: bool = False,
    store: "Store | None" = None,
) -> dict[str, dict]:
    df, n = document_frequencies(pool)
    if not use_llm:
        return {r.paper.id: explain(r, profile, df, n) for r in results}
    with ThreadPoolExecutor(max_workers=config.LLM_MAX_WORKERS) as ex:
        futs = {r.paper.id: ex.submit(explain, r, profile, df, n, True, store) for r in results}
        return {pid: f.result() for pid, f in futs.items()}
