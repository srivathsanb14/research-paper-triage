"""Synthetic (rule-based) Read/Skim/Skip labels for four demo profiles, merged into the label dataset.

    python scripts/simulate_labels.py   # rewrites the synthetic rows of data/labels/labels.jsonl

Each profile (data/labels/profiles.json) gets a transparent keyword rule that never looks at the
ranking model, plus a little seeded noise so labels are imperfect like a person's. Rows are
tagged origin="synthetic"; rows with any other origin (the manual labels) are left untouched.
Three of the rules also encode a hidden interest the written profile omits, so the Insights
learning curve has something to learn. These labels support demos and tests, not claims about
ranking quality.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage import catalog  # noqa: E402
from triage.models import Paper  # noqa: E402

LABELS_DIR = Path(__file__).resolve().parent.parent / "data" / "labels"
CAPS = {"READ": 45, "SKIM": 50, "SKIP": 65}
# Noise only between neighbouring labels: a stray Read among 1,600 Skips would swamp the real ones.
FLIPS = {"READ": [("SKIM", 0.08)], "SKIM": [("READ", 0.05), ("SKIP", 0.10)], "SKIP": [("SKIM", 0.01)]}


def rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.I)


# --- the original simulated RAG researcher: labels every AI and computing paper -------------------
_CORE = rx(r"retrieval[- ]augmented|\brag\b|rerank|passage retrieval|dense retriev|question answering|\bqa\b|"
           r"fact[- ]check|citation|grounded generation|hallucinat|search agent|agentic search|web agent")
_RELATED = rx(r"retriev|search|large language model|\bllms?\b|language model|agent|benchmark|embedding|"
              r"recommend|knowledge graph|summari[sz]|faithful|evidence|document")
_OFF = rx(r"robot|medical imag|segmentation|locomotion|quadrotor|autopilot|mri\b")


def rag_label(p: Paper) -> str:
    title, text = p.title, f"{p.title} {p.abstract}"
    if _OFF.search(text):
        return "SKIP"
    core_hits = len(_CORE.findall(text))
    if _CORE.search(title) or core_hits >= 3:
        return "READ"
    if core_hits >= 1 or len(_RELATED.findall(text)) >= 3:
        return "SKIM"
    return "SKIP"


# --- personas: core topic, related topics, off-topic, and a hidden interest ----------------------
PERSONAS = {
    "materials-discovery": dict(
        core=rx(r"materials? (discovery|design|informatics)|crystal structure|interatomic potential|perovskite|solid[- ]state electrolyte|"
                r"battery (material|cathode|anode|electrolyte)|catalyst (design|discovery)|high-throughput (screening|computation)|"
                r"density functional|self-driving lab|autonomous (lab|experiment)|inverse design|metal[- ]organic framework"),
        related=rx(r"material|crystal|alloy|catalys|battery|electrolyte|semiconductor|polymer|nanomaterial|molecular dynamics|"
                   r"first[- ]principles|graph neural network|active learning|bayesian optimi[sz]ation|synthesis|band gap"),
        off=rx(r"social media|election|clinical trial|tweet|political|marketing"),
        hidden=rx(r"solar cell|photovoltaic|thermoelectric|photocatal|graphene|two-dimensional"),
    ),
    "robotics": dict(
        core=rx(r"robot|manipulat(or|ion)|locomotion|legged|quadruped|humanoid|grasp(ing)?\b|sim-to-real|motion planning|"
                r"visuomotor|dexterous|slam\b|\buav\b|drone"),
        related=rx(r"reinforcement learning|control policy|imitation learning|trajectory|autonomous (driving|vehicle|navigation)|"
                   r"embodied|sensor|tactile|simulation|planning|vision-language-action|world model|point cloud|lidar"),
        off=rx(r"medical imag|clinical|genom|protein|tweet|political|marketing"),
        hidden=rx(r"tactile|soft robot|exoskeleton|teleoperat|multi-robot|swarm|path planning|trajectory"),
    ),
    "ai-and-design": dict(
        core=rx(r"generative design|co-?creat(ive|ion)|creative (tool|support|process|practice|ai)|design (tool|process|practice|space|system)s?\b|"
                r"human-ai (collaboration|co-)|user interface (generation|design)|designers?\b|ai[- ]assisted design|generative (ai|model)s? for design"),
        related=rx(r"\bdesign|creativ|human-computer|\bhci\b|user (study|experience)|usability|visuali[sz]ation|image generation|"
                   r"interactive|artist|aesthetic|multimodal|layout|sketch|architecture|prototyp|diffusion model|text-to-image|\bgui\b"),
        off=rx(r"bioinformatics|clinical|genom|protein|lidar|quadrotor|pharmaceutic|drug|quality by design|health ?care|patients?\b"),
        hidden=rx(r"urban design|architectur|interior|heritage|typograph|visuali[sz]ation|infographic|fashion|game design"),
    ),
}


def persona_label(rules: dict, title: str, text: str) -> str:
    if rules["hidden"].search(title):
        return "READ"
    if rules["hidden"].search(text):
        return "SKIM"
    if rules["off"].search(text):
        return "SKIP"
    hits = len(rules["core"].findall(text))
    if rules["core"].search(title) or hits >= 3 or (hits >= 1 and len(rules["related"].findall(text)) >= 3):
        return "READ"
    if hits >= 1 or len(rules["related"].findall(text)) >= 4:
        return "SKIM"
    return "SKIP"


def persona_rows(slug: str, papers: list[Paper], stamp: str) -> list[dict]:
    rng = random.Random(f"simulate:{slug}")
    rules = PERSONAS[slug]
    by_label: dict[str, list[Paper]] = {"READ": [], "SKIM": [], "SKIP": []}
    for p in papers:
        lab = persona_label(rules, p.title, f"{p.title} {p.abstract}")
        for other, prob in FLIPS[lab]:
            if rng.random() < prob:
                lab = other
                break
        by_label[lab].append(p)
    picked = []
    for lab, cap in CAPS.items():
        pool = by_label[lab]
        rng.shuffle(pool)
        picked += [(p, lab) for p in pool[:cap]]
    return [{"profile": slug, "paper_id": p.id, "label": lab, "labeled_at": stamp, "origin": "synthetic"} for p, lab in picked]


def main() -> None:
    info = catalog.manifest()
    papers = [p for area in info["areas"] for p in catalog.read_shard(area)]
    stamp = info["updated_at"]
    kept = [r for r in (json.loads(x) for x in (LABELS_DIR / "labels.jsonl").read_text(encoding="utf-8").splitlines() if x.strip())
            if r["origin"] != "synthetic"]
    new = []
    for area in info["areas"]:
        if area["id"] in ("ai", "computing"):
            new += [{"profile": "rag-research-demo", "paper_id": p.id, "label": rag_label(p), "labeled_at": stamp, "origin": "synthetic"}
                    for p in catalog.read_shard(area)]
    for slug in PERSONAS:
        new += persona_rows(slug, papers, stamp)
    rows = sorted(kept + new, key=lambda r: (r["profile"], r["paper_id"]))
    (LABELS_DIR / "labels.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"Kept {len(kept)} manual labels, wrote {len(new)} synthetic labels → {LABELS_DIR / 'labels.jsonl'}")


if __name__ == "__main__":
    main()
