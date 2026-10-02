"""Build the static website (GitHub Pages or a Hugging Face static Space).

    python scripts/build_pages.py                 # → _site/
    python scripts/build_pages.py --hf-space      # also writes the Space README front matter
    python -m http.server 8000 --directory _site

Copies only an explicit list of public files: the browser app (web/), the
shared quality rules and the verified catalog with its embeddings. Local
databases, secrets and caches never leave the machine. No dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "_site"
WEB = ROOT / "web"
CATALOG = ROOT / "data" / "catalog"
SHARD = re.compile(r"[a-z]+-[0-9a-f]{64}\.(json|bin)")

SPACE_README = """---
title: Paper Triage
emoji: 📚
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Read, Skim or Skip for research papers, with trust signals
tags:
  - research
  - recommender
  - sentence-transformers
  - transformers.js
models:
  - Xenova/all-MiniLM-L6-v2
---

# Paper Triage

Describe your research once. Paper Triage sorts recent papers into Read, Skim and Skip,
gives a one-line reason, and shows worth-it signals (peer review, released code or
data, study-design cues). Your ratings train a small model in your browser.

Source, model cards and dataset: see the GitHub repository linked in the app.
"""


def _verified(name: str, expected: str) -> bytes:
    if not SHARD.fullmatch(name):
        raise ValueError(f"Invalid catalog filename: {name}")
    raw = (CATALOG / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Catalog checksum mismatch: {name}")
    return raw


def build(output: Path = OUTPUT, hf_space: bool = False) -> None:
    if output.exists():
        shutil.rmtree(output)
    (output / "js").mkdir(parents=True)
    (output / "catalog").mkdir()
    for name in ("index.html", "styles.css"):
        shutil.copyfile(WEB / name, output / name)
    for path in sorted((WEB / "js").glob("*.js")):
        shutil.copyfile(path, output / "js" / path.name)
    shutil.copyfile(ROOT / "triage" / "quality_rules.json", output / "quality-rules.json")
    (output / ".nojekyll").touch()

    manifest = json.loads((CATALOG / "manifest.json").read_text())
    for area in manifest["areas"]:
        (output / "catalog" / area["file"]).write_bytes(_verified(area["file"], area["sha256"]))
    shutil.copyfile(CATALOG / "manifest.json", output / "catalog" / "manifest.json")

    emb_path = CATALOG / "embeddings.json"
    n_vec = 0
    if emb_path.exists():
        emb = json.loads(emb_path.read_text())
        shards = {a["id"]: a["sha256"] for a in manifest["areas"]}
        # Only publish vectors that belong to the papers being published.
        emb["areas"] = {k: v for k, v in emb["areas"].items() if shards.get(k) == v["papers_sha256"]}
        for entry in emb["areas"].values():
            (output / "catalog" / entry["file"]).write_bytes(_verified(entry["file"], entry["sha256"]))
            n_vec += entry["count"]
        (output / "catalog" / "embeddings.json").write_text(json.dumps(emb, indent=2) + "\n")
    n_prop = _proposals(output / "catalog" / "proposals.json", {p["id"] for a in manifest["areas"] for p in json.loads((CATALOG / a["file"]).read_text())})
    if hf_space:
        (output / "README.md").write_text(SPACE_README)
    n = sum(a["count"] for a in manifest["areas"])
    print(f"Built {output}: {n} papers, {n_vec} embeddings, {n_prop} label suggestions (no local databases or secrets).")


def _proposals(dest: Path, catalog_ids: set[str]) -> int:
    """Model-proposed labels for the Label tab's review mode (humans accept or correct each one)."""
    folder = ROOT / "data" / "labels" / "proposals"
    profiles_file = folder.parent / "profiles.json"
    meta = json.loads(profiles_file.read_text()) if profiles_file.exists() else {"profiles": []}
    sets = []
    for prof in meta["profiles"]:
        items = []
        for path in sorted(folder.glob(f"{prof['slug']}*.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("paper_id") in catalog_ids and r.get("proposed_label") in ("READ", "SKIM", "SKIP"):
                        items.append({"paper_id": r["paper_id"], "proposed_label": r["proposed_label"], "reason": str(r.get("reason", ""))[:300]})
        if items:
            sets.append({**prof, "items": items})
    dest.write_text(json.dumps({"proposer": meta.get("proposer", ""), "sets": sets}, ensure_ascii=False) + "\n")
    return sum(len(s["items"]) for s in sets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--hf-space", action="store_true", help="add Hugging Face Space front matter (README.md)")
    args = parser.parse_args()
    build(args.output, args.hf_space)
