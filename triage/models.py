"""Core data records shared across the pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Paper:
    """A cleaned research-paper record (the 'Research paper database' box)."""

    id: str  # namespaced: "arxiv:2401.01234" or "s2:<paperId>"
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    year: int | None = None
    published: str = ""  # ISO date (YYYY-MM-DD) when known
    url: str = ""
    pdf_url: str = ""
    categories: list[str] = field(default_factory=list)
    source: str = ""  # "arxiv" | "semantic_scholar" | "sample"
    citation_count: int | None = None
    # Publication metadata behind the "worth it" signals (see triage/quality.py).
    work_type: str = ""  # "article" | "review" | "preprint" | …
    venue_type: str = ""  # "journal" | "conference" | "repository" | …
    venue_core: bool = False  # venue is in the CWTS core list (established, indexed)
    version: str = ""  # "publishedVersion" | "acceptedVersion" | "submittedVersion"
    oa_status: str = ""  # "gold" | "green" | "hybrid" | "bronze" | "diamond" | "closed"
    fwci: float | None = None  # field-weighted citation impact
    references_count: int | None = None
    full_text: str = ""  # introduction + conclusion, fetched for borderline papers
    full_text_status: str = ""  # "" (not tried) | "ok" | "unavailable"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Paper":
        known = {f for f in cls.__dataclass_fields__}  # tolerate extra/missing keys
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def text(self) -> str:
        """Text used for embeddings: title twice (it's the densest signal) + abstract."""
        return f"{self.title}. {self.title}. {self.abstract}".strip()

    @property
    def author_str(self) -> str:
        if not self.authors:
            return "Unknown authors"
        if len(self.authors) <= 4:
            return ", ".join(self.authors)
        return f"{', '.join(self.authors[:3])}, … +{len(self.authors) - 3} more"


@dataclass
class InterestProfile:
    """User context typed into the app (the 'User context' box)."""

    name: str = "default"
    description: str = ""  # free-text project description
    keywords: list[str] = field(default_factory=list)
    focus: str = ""  # current focus (weighted higher)
    avoid: list[str] = field(default_factory=list)  # topics to down-rank
    hours_per_week: float = 3.0  # time available for reading
    id: int | None = None

    def is_empty(self) -> bool:
        return not (self.description.strip() or self.keywords or self.focus.strip())

    def fingerprint(self) -> str:
        """Stable hash of the fields that affect scoring (cache key)."""
        payload = json.dumps(
            {
                "d": self.description.strip(),
                "k": sorted(k.lower() for k in self.keywords),
                "f": self.focus.strip(),
                "a": sorted(a.lower() for a in self.avoid),
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:16]


@dataclass
class TriageResult:
    paper: Paper
    score: float
    label: str
    rank: int
    features: dict[str, float] = field(default_factory=dict)
    prior_score: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    note: str = ""  # e.g. "demoted to SKIM by time budget"
    group_lead: str = ""  # id of a higher-ranked, near-identical paper ("" = this is a lead)
    similar: list[str] = field(default_factory=list)  # ids grouped under this lead
