"""Core data records shared across the pipeline."""

from __future__ import annotations

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

    @property
    def text(self) -> str:
        """Text used for embeddings: title twice (it's the densest signal) + abstract."""
        return f"{self.title}. {self.title}. {self.abstract}".strip()
