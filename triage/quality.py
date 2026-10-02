"""Worth-it signals: transparent cues about how much a paper can be trusted and reused.

Relevance answers "is this about my research?". When papers are cheap to produce,
researchers also need "is this worth my time?". These signals never change the
relevance score; the UI shows them next to it and lets people filter or sort.

Every signal is a rule a reader can check: publication metadata from OpenAlex
(venue type, peer-review version, open access, field-weighted citations) and
phrases in the abstract (released code or data, study designs). They are cues,
not a verdict. A missing cue usually means the abstract does not mention it,
and fields differ: humanities papers rarely release code.

The rules live in quality_rules.json, shared with the browser (web/js/quality.js).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .models import Paper

RULES_PATH = Path(__file__).with_name("quality_rules.json")


@lru_cache(maxsize=1)
def rules() -> dict:
    raw = json.loads(RULES_PATH.read_text())
    compile_all = lambda pats: [re.compile(p, re.I) for p in pats]  # noqa: E731
    return {
        **raw,
        "code_re": compile_all(raw["code"]),
        "data_re": compile_all(raw["data"]),
        "evidence_re": {name: re.compile(p, re.I) for name, p in raw["evidence"].items()},
        "hype_re": compile_all(raw["hype"]),
        "review_re": re.compile(raw["review_title"], re.I),
    }


@dataclass
class Signals:
    peer_reviewed: bool | None  # None = unknown venue
    established_venue: bool
    review: bool
    code: bool
    data: bool
    evidence: list[str] = field(default_factory=list)
    open_access: bool = False
    impact: bool = False
    thin_abstract: bool = False
    promotional: bool = False
    few_references: bool = False
    points: int = 0
    level: str = "limited"  # "strong" | "moderate" | "limited"


def peer_reviewed(p: Paper) -> bool | None:
    if p.work_type == "preprint" or p.venue_type == "repository" or p.source == "arxiv" or p.version == "submittedVersion":
        return False
    if p.venue_type in rules()["peer_reviewed_venues"]:
        return True
    return None


def signals(p: Paper) -> Signals:
    r = rules()
    text = f"{p.title}. {p.abstract}"
    evidence = [name for name, rx in r["evidence_re"].items() if rx.search(text)]
    words = len(p.abstract.split())
    s = Signals(
        peer_reviewed=peer_reviewed(p),
        established_venue=bool(p.venue_core),
        review=p.work_type == "review" or bool(r["review_re"].search(p.title)),
        code=any(rx.search(text) for rx in r["code_re"]),
        data=any(rx.search(text) for rx in r["data_re"]),
        evidence=evidence,
        open_access=bool(p.pdf_url) or p.oa_status not in ("", "closed"),
        impact=p.fwci is not None and p.fwci >= r["impact_fwci"],
        thin_abstract=words < r["thin_abstract_words"],
        promotional=sum(bool(rx.search(text)) for rx in r["hype_re"]) >= 2,
        # 0 references usually means missing metadata, not an empty bibliography.
        few_references=p.references_count is not None and 0 < p.references_count < r["few_references"],
    )
    pts = r["points"]
    s.points = (
        pts["peer_reviewed"] * bool(s.peer_reviewed)
        + pts["established_venue"] * s.established_venue
        + pts["code"] * s.code
        + pts["data"] * s.data
        + min(pts["evidence_max"], pts["evidence_each"] * len(evidence))
        + pts["open_access"] * s.open_access
        + pts["impact"] * s.impact
        + pts["thin_abstract"] * s.thin_abstract
        + pts["promotional"] * s.promotional
        + pts["few_references"] * s.few_references
    )
    lv = r["levels"]
    s.level = "strong" if s.points >= lv["strong"] else "moderate" if s.points >= lv["moderate"] else "limited"
    return s
