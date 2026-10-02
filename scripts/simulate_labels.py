"""Synthetic Read/Skim/Skip labels for pipeline tests, never for claims about quality.

    python scripts/simulate_labels.py   # → data/labels/synthetic/demo_rag.jsonl

Applies the rule-based "simulated RAG researcher" from triage/demo.py (keyword
rules, independent of the ranking model) to the AI and computing papers in the
catalog. Rows are marked label_source=synthetic-rule and live apart from manual labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage import catalog  # noqa: E402
from triage.demo import DEMO_PROFILE, simulated_label  # noqa: E402

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
