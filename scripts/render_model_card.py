"""Fill the evaluation numbers in the Hugging Face model cards from docs/evaluation/report.json.

    python scripts/render_model_card.py

The cards keep their prose; only the blocks between `<!-- eval:NAME -->` and `<!-- /eval:NAME -->`
are rewritten, so the published numbers always match the latest `node scripts/evaluate_web.mjs` run.
scripts/publish_hf.sh runs this before uploading.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = json.loads((ROOT / "docs/evaluation/report.json").read_text(encoding="utf-8"))

METHODS = [("Personalised", "Personalised"), ("Profile only", "Profile only"), ("Semantic similarity only", "Semantic similarity only"),
           ("Keyword matches only", "Keyword matches only")]


def pick(origin: str, space: str, good: str = "read") -> list[dict]:
    return [r for r in REPORT if r["origin"] == origin and r["space"] == space and r["good"] == good and r["ok"]]


def method_rows(r: dict, label: str | None) -> list[str]:
    n_good = r["counts"]["READ"]
    rows = []
    for i, (key, shown) in enumerate(METHODS):
        s = next(x for x in r["systems"] if x["name"].startswith(key))
        name = f"Personalised (learning {round((r['blend']['weight'] or 0) * 100)}%)" if key == "Personalised" and label is None else shown
        first = f"{r['profile']} | {r['n']} ({r['counts']['READ']}/{r['counts']['SKIM']}/{r['counts']['SKIP']}) |" if i == 0 and label else " |  |" if label else ""
        cells = f"{name} | {s['ndcg10']:.2f} | {s['ap']:.2f} | {s['hit10'] * 100:.0f}% | {s['to80'] if s['to80'] is not None else '—'}"
        rows.append(f"| {first} {cells} |" if label else f"| {cells} |")
    rows.append((f"|  |  | Random order | — | {r['randomAp']:.2f} | {n_good / r['n'] * 100:.0f}% | {r['random80']} |" if label
                 else f"| Random order | — | {r['randomAp']:.2f} | {n_good / r['n'] * 100:.0f}% | {r['random80']} |"))
    return rows


def manual_block() -> str:
    head = ["| Profile | Labels (Read/Skim/Skip) | Method | NDCG@10 | AP | Good in top 10 | Papers to find 80% of good |", "|---|---|---|---|---|---|---|"]
    rows = [line for r in pick("manual", "MiniLM") for line in method_rows(r, "p")]
    return "\n".join(head + rows)


def synthetic_block() -> str:
    r = pick("synthetic", "MiniLM")[0] if len(pick("synthetic", "MiniLM")) == 1 else next(x for x in pick("synthetic", "MiniLM") if "RAG" in x["profile"])
    head = ["| Method (MiniLM space) | NDCG@10 | AP | Good in top 10 | Papers to find 80% of good |", "|---|---|---|---|---|"]
    return "\n".join(head + method_rows(r, None))


def compare_sentence() -> str:
    parts = []
    for m, t in zip(pick("manual", "MiniLM"), pick("manual", "TF-IDF")):
        ap = lambda r: next(x for x in r["systems"] if x["name"].startswith("Personalised"))["ap"]  # noqa: E731
        parts.append(f"{m['profile']} {ap(m):.2f} vs {ap(t):.2f}")
    return "MiniLM vs TF-IDF (personalised ranker, AP for Read): " + "; ".join(parts) + "."


def weight_sentence() -> str:
    weights = {round((r["blend"]["weight"] or 0) * 100) for r in pick("manual", "MiniLM")}
    if weights == {0}:
        return ("The cross-validated blend chose a learning weight of **0%** for all manual profiles: training on these labels did not beat "
                "the profile score, so the safeguard kept learning off.")
    return "The cross-validated blend chose learning weights of " + ", ".join(f"{w}%" for w in sorted(weights)) + " across the manual profiles."


def short_compare() -> str:
    parts = []
    for m, t in zip(pick("manual", "MiniLM"), pick("manual", "TF-IDF")):
        ap = lambda r: next(x for x in r["systems"] if x["name"].startswith("Personalised"))["ap"]  # noqa: E731
        parts.append(f"{ap(m):.2f} vs {ap(t):.2f}")
    names = ", ".join(m["profile"] for m in pick("manual", "MiniLM"))
    return f"On the manual labels (good = Read), the full ranker reaches AP {', '.join(p.split(' vs ')[0] for p in parts)} with MiniLM vs {', '.join(p.split(' vs ')[1] for p in parts)} with TF-IDF ({names})."


BLOCKS = {
    "manual": manual_block, "synthetic": synthetic_block, "compare": lambda: compare_sentence() + "\n\n" + weight_sentence(),
    "minilm-vs-tfidf": short_compare,
}


def render(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for name, fn in BLOCKS.items():
        pattern = re.compile(rf"(<!-- eval:{name} -->\n).*?(\n<!-- /eval:{name} -->)", re.S)
        if pattern.search(text):
            text = pattern.sub(lambda m: m.group(1) + fn() + m.group(2), text)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    for card in ("hf/model-ranker/README.md", "hf/model-embeddings/README.md"):
        render(ROOT / card)
        print("rendered", card)
