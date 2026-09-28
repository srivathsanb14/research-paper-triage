"""Snapshot a diverse set of current arXiv papers into data/sample_papers.json.

The bundled sample lets the app run offline (demo, tests, flaky network).
Categories are deliberately mixed so any interest profile sees READ, SKIM and SKIP.
Uses arXiv's daily-listing feeds (rss.arxiv.org), sampling up to N per category,
plus recent cs.IR / cs.CL papers from arXiv's OAI-PMH endpoint so the demo
profile (retrieval / RAG research) has a realistic number of relevant papers.

    python scripts/build_sample_data.py [--per-category 30] [--oai-extra 160] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage import config  # noqa: E402
from triage.preprocess import dedupe  # noqa: E402
from datetime import date, timedelta  # noqa: E402

from triage.sources import SourceError, fetch_arxiv_new, fetch_arxiv_oai  # noqa: E402

CATEGORIES = ["cs.CL", "cs.IR", "cs.LG", "cs.CV", "cs.RO", "cs.HC", "cs.CR", "cs.DL", "cs.SE", "q-bio.QM", "eess.AS", "stat.ML"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oai-extra", type=int, default=160, help="extra recent cs.IR/cs.CL papers via OAI-PMH")
    ap.add_argument("--oai-days", type=int, default=21)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    papers, seen = [], set()
    for cat in CATEGORIES:
        try:
            batch = [p for p in fetch_arxiv_new([cat], include_replacements=True) if p.id not in seen]
        except SourceError as e:
            print(f"  ! {cat}: {e}")
            continue
        # Prefer papers whose primary category is this one, for diversity.
        primary = [p for p in batch if p.categories and p.categories[0] == cat] or batch
        pick = rng.sample(primary, min(args.per_category, len(primary)))
        for p in pick:
            p.source = "sample"
            seen.add(p.id)
        print(f"  {len(pick):3d} / {len(batch):3d}  {cat}")
        papers += pick
        time.sleep(3)
    if args.oai_extra:
        since = (date.today() - timedelta(days=args.oai_days)).isoformat()
        try:
            recent = fetch_arxiv_oai(since, ["cs.IR", "cs.CL"], max_pages=3, created_since=since)
        except SourceError as e:
            print(f"  ! OAI: {e}")
            recent = []
        recent = [p for p in recent if p.id not in seen]
        # Prefer cs.IR (retrieval) papers, then cs.CL.
        ir = [p for p in recent if "cs.IR" in p.categories]
        cl = [p for p in recent if "cs.IR" not in p.categories]
        pick = rng.sample(ir, min(len(ir), args.oai_extra // 2))
        pick += rng.sample(cl, min(len(cl), args.oai_extra - len(pick)))
        for p in pick:
            p.source = "sample"
        print(f"  {len(pick):3d} / {len(recent):3d}  OAI cs.IR+cs.CL since {since}")
        papers += pick
    papers = dedupe(papers)
    config.SAMPLE_PAPERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SAMPLE_PAPERS_PATH, "w") as f:
        json.dump([p.to_dict() for p in papers], f, indent=1, ensure_ascii=False)
    print(f"Wrote {len(papers)} papers to {config.SAMPLE_PAPERS_PATH}")


if __name__ == "__main__":
    main()
