"""Synthetic Read/Skim/Skip labels for pipeline tests, never for claims about quality.

    python scripts/simulate_labels.py   # → data/labels/synthetic/demo_rag.jsonl

Applies the rule-based "simulated RAG researcher" defined below (keyword
rules, independent of the ranking model) to the AI and computing papers in the
catalog. Rows are marked label_source=synthetic-rule and live apart from manual labels.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage import catalog  # noqa: E402
from triage.models import Paper  # noqa: E402

DEMO_PROFILE = dict(
    description=(
        "I build retrieval-augmented generation (RAG) systems for question answering over scientific "
        "documents, and study how LLM agents use search tools and cite evidence."
    ),
    keywords=["retrieval augmented generation", "question answering", "LLM agents", "information retrieval", "reranking"],
    focus="evaluating retrieval quality and faithfulness in RAG pipelines",
)

_CORE = re.compile(
    r"retrieval[- ]augmented|\brag\b|rerank|passage retrieval|dense retriev|question answering|\bqa\b|"
    r"fact[- ]check|citation|grounded generation|hallucinat|search agent|agentic search|web agent",
    re.I,
)
_RELATED = re.compile(
    r"retriev|search|large language model|\bllms?\b|language model|agent|benchmark|embedding|"
    r"recommend|knowledge graph|summari[sz]|faithful|evidence|document",
    re.I,
)
_OFF = re.compile(r"robot|medical imag|segmentation|locomotion|quadrotor|autopilot|mri\b", re.I)


def simulated_label(p: Paper) -> str:
    """What the simulated RAG researcher would say about this paper."""
    title, text = p.title, f"{p.title} {p.abstract}"
    if _OFF.search(text):
        return "SKIP"
    core_hits = len(_CORE.findall(text))
    if _CORE.search(title) or core_hits >= 3:
        return "READ"
    if core_hits >= 1 or len(_RELATED.findall(text)) >= 3:
        return "SKIM"
    return "SKIP"


OUT = Path(__file__).resolve().parent.parent / "data/labels/synthetic/demo_rag.jsonl"


def main() -> None:
    rows = []
    for area in catalog.manifest()["areas"]:
        if area["id"] not in ("ai", "computing"):
            continue
        for p in catalog.read_shard(area):
            rows.append({
                "paper_id": p.id, "title": p.title, "label": simulated_label(p), "labeler": "rule:demo_rag",
                "labeled_at": catalog.manifest()["updated_at"], "label_source": "synthetic-rule",
                "profile_name": "Demo: RAG research (simulated)", "profile_description": DEMO_PROFILE["description"],
                "profile_keywords": DEMO_PROFILE["keywords"], "profile_focus": DEMO_PROFILE["focus"],
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} synthetic labels to {OUT}")


if __name__ == "__main__":
    main()
