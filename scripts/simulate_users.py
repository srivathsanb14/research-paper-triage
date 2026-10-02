"""Simulated researchers for the demo accounts: profiles plus rule-based Read/Skim/Skip labels.

    python scripts/simulate_users.py   # → data/labels/synthetic/{sri,ishaan,chris}.jsonl

Each persona is a transparent keyword rule set (including a "hidden interest" the written profile omits) that never looks at the ranking model, plus a
little seeded noise so labels are imperfect like real ones. Rows are marked
label_source=synthetic-rule. They are mock data for demos and tests, never evidence of quality.
Each file holds a stratified sample of the catalog (all strong matches up to a cap, plus
random Skim and Skip papers), not the whole catalog, so cross-validation stays fast.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage import catalog  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data/labels/synthetic"
CAPS = {"READ": 45, "SKIM": 50, "SKIP": 65}
# Imperfect like a real person, but only between neighbouring labels: a stray Read among 1,600 Skips would swamp the real ones.
FLIPS = {"READ": [("SKIM", 0.08)], "SKIM": [("READ", 0.05), ("SKIP", 0.10)], "SKIP": [("SKIM", 0.01)]}


def rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.I)


PERSONAS = {
    "sri": dict(
        name="Materials discovery",
        description="I use machine learning and high-throughput computation to discover new functional materials, such as battery electrolytes, catalysts and perovskites, and I want to close the loop with automated experiments.",
        keywords=["materials discovery", "machine learning interatomic potentials", "high-throughput screening", "crystal structure prediction", "battery materials"],
        focus="ML-guided discovery of battery and catalyst materials",
        avoid=["social media"],
        fields=["physical", "computing", "ai"],
        core=rx(r"materials? (discovery|design|informatics)|crystal structure|interatomic potential|perovskite|solid[- ]state electrolyte|"
                r"battery (material|cathode|anode|electrolyte)|catalyst (design|discovery)|high-throughput (screening|computation)|"
                r"density functional|self-driving lab|autonomous (lab|experiment)|inverse design|metal[- ]organic framework"),
        related=rx(r"material|crystal|alloy|catalys|battery|electrolyte|semiconductor|polymer|nanomaterial|molecular dynamics|"
                   r"first[- ]principles|graph neural network|active learning|bayesian optimi[sz]ation|synthesis|band gap"),
        off=rx(r"social media|election|clinical trial|tweet|political|marketing"),
        hidden=rx(r"solar cell|photovoltaic|thermoelectric|photocatal|graphene|two-dimensional"),
    ),
    "ishaan": dict(
        name="Robotics",
        description="I work on robot learning: manipulation, legged locomotion and sim-to-real transfer, and I care about policies that stay reliable outside the lab.",
        keywords=["robot manipulation", "legged locomotion", "sim-to-real transfer", "reinforcement learning for control", "motion planning"],
        focus="robust sim-to-real policies for manipulation and locomotion",
        avoid=["medical imaging"],
        fields=["ai", "computing"],
        core=rx(r"robot|manipulat(or|ion)|locomotion|legged|quadruped|humanoid|grasp(ing)?\b|sim-to-real|motion planning|"
                r"visuomotor|dexterous|slam\b|\buav\b|drone"),
        related=rx(r"reinforcement learning|control policy|imitation learning|trajectory|autonomous (driving|vehicle|navigation)|"
                   r"embodied|sensor|tactile|simulation|planning|vision-language-action|world model|point cloud|lidar"),
        off=rx(r"medical imag|clinical|genom|protein|tweet|political|marketing"),
        hidden=rx(r"tactile|soft robot|exoskeleton|teleoperat|multi-robot|swarm|path planning|trajectory"),
    ),
    "chris": dict(
        name="AI and design",
        description="I study how generative AI changes design practice: co-creative tools, interface and prototype generation, and how designers and models collaborate.",
        keywords=["generative design", "human-AI co-creation", "creative tools", "interface generation", "design process"],
        focus="designer-model collaboration in creative tools",
        avoid=["bioinformatics"],
        fields=["ai", "computing", "humanities", "society"],
        core=rx(r"generative design|co-?creat(ive|ion)|creative (tool|support|process|practice|ai)|design (tool|process|practice|space|system)s?\b|"
                r"human-ai (collaboration|co-)|user interface (generation|design)|designers?\b|ai[- ]assisted design|generative (ai|model)s? for design"),
        related=rx(r"\bdesign|creativ|human-computer|\bhci\b|user (study|experience)|usability|visuali[sz]ation|image generation|"
                   r"interactive|artist|aesthetic|multimodal|layout|sketch|architecture|prototyp|diffusion model|text-to-image|\bgui\b"),
        off=rx(r"bioinformatics|clinical|genom|protein|lidar|quadrotor|pharmaceutic|drug|quality by design|health ?care|patients?\b"),
        hidden=rx(r"urban design|architectur|interior|heritage|typograph|visuali[sz]ation|infographic|fashion|game design"),
    ),
}


def label_for(persona: dict, title: str, text: str) -> str:
    # Hidden interests: topics this person reads that their written profile never mentions. Only
    # learning from labels can pick these up, which is what the Insights learning curve shows.
    if persona["hidden"].search(title):
        return "READ"
    if persona["hidden"].search(text):
        return "SKIM"
    if persona["off"].search(text):
        return "SKIP"
    hits = len(persona["core"].findall(text))
    if persona["core"].search(title) or hits >= 3 or (hits >= 1 and len(persona["related"].findall(text)) >= 3):
        return "READ"
    if hits >= 1 or len(persona["related"].findall(text)) >= 4:
        return "SKIM"
    return "SKIP"


def main() -> None:
    papers = [p for area in catalog.manifest()["areas"] for p in catalog.read_shard(area)]
    stamp = catalog.manifest()["updated_at"]
    for user, persona in PERSONAS.items():
        rng = random.Random(f"simulate:{user}")
        by_label: dict[str, list] = {"READ": [], "SKIM": [], "SKIP": []}
        for p in papers:
            lab = label_for(persona, p.title, f"{p.title} {p.abstract}")
            for other, prob in FLIPS[lab]:
                if rng.random() < prob:
                    lab = other
                    break
            by_label[lab].append(p)
        rows = []
        for lab, cap in CAPS.items():
            pool = by_label[lab]
            rng.shuffle(pool)
            rows += [(p, lab) for p in pool[:cap]]
        rng.shuffle(rows)
        path = OUT / f"{user}.jsonl"
        path.write_text("".join(json.dumps({
            "paper_id": p.id, "title": p.title, "label": lab, "labeler": f"rule:{user}", "labeled_at": stamp,
            "label_source": "synthetic-rule", "profile_name": persona["name"], "profile_description": persona["description"],
            "profile_keywords": persona["keywords"], "profile_focus": persona["focus"],
        }, ensure_ascii=False) + "\n" for p, lab in rows), encoding="utf-8")
        counts = {k: sum(1 for _, l in rows if l == k) for k in CAPS}
        print(f"{user}: {len(rows)} labels {counts} (matches before sampling: { {k: len(v) for k, v in by_label.items()} }) → {path.name}")


if __name__ == "__main__":
    main()
