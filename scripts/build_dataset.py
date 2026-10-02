"""Package the paper catalog and Read/Skim/Skip labels as a Hugging Face dataset.

    python scripts/simulate_labels.py          # optional: synthetic labels (kept separate)
    python scripts/build_dataset.py            # → _dataset/  (+ docs/EDA.md)
    hf upload <user>/paper-triage-labels _dataset . --repo-type=dataset

Inputs
  data/catalog/                 the public OpenAlex snapshot the app ranks (CC0 metadata)
  data/labels/manual/*.jsonl|csv  exports from the app's Label tab (human labels)
  data/labels/synthetic/*.jsonl   rule-generated labels (never mixed with manual ones)

Outputs
  _dataset/papers.jsonl           one row per paper, with the worth-it signals
  _dataset/labels_manual.jsonl    validated human labels, de-duplicated per labeller
  _dataset/labels_synthetic.jsonl synthetic labels, clearly marked
  _dataset/README.md              dataset card with EDA
  docs/EDA.md                     the same exploratory analysis for the repository
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage import catalog, quality  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LABELS = ("READ", "SKIM", "SKIP")
OUT = ROOT / "_dataset"


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_papers() -> list[dict]:
    rows, seen = [], set()
    for area in catalog.manifest()["areas"]:
        for p in catalog.read_shard(area):
            if p.id in seen:
                continue
            seen.add(p.id)
            s = quality.signals(p)
            d = p.to_dict()
            for k in ("full_text", "full_text_status"):
                d.pop(k)
            d.update(area=area["id"], area_label=area["label"], abstract_words=len(p.abstract.split()),
                     signals={k: v for k, v in asdict(s).items() if k not in ("points", "level")},
                     signal_points=s.points, signal_level=s.level)
            rows.append(d)
    return rows


def load_labels(folder: Path, papers: dict[str, dict], source: str) -> tuple[list[dict], list[str]]:
    out, problems, seen = [], [], {}
    for path in sorted(folder.glob("*.jsonl")) + sorted(folder.glob("*.csv")):
        for i, r in enumerate(read_rows(path), 1):
            pid, label = r.get("paper_id", ""), r.get("label", "")
            if label not in LABELS:
                problems.append(f"{path.name}:{i}: unknown label {label!r}")
                continue
            if pid not in papers and not r.get("title"):
                problems.append(f"{path.name}:{i}: paper {pid!r} is not in the catalog and the row has no title")
                continue
            keywords = r.get("profile_keywords", [])
            if isinstance(keywords, str):
                keywords = [k for k in keywords.split("; ") if k]
            proposed = r.get("proposed_label") or ""
            if proposed and proposed not in LABELS:
                problems.append(f"{path.name}:{i}: unknown proposed label {proposed!r}")
                continue
            row = {
                "paper_id": pid, "label": label, "labeler": r.get("labeler") or "unknown", "labeled_at": r.get("labeled_at", ""),
                # A human checked a model's suggestion and kept or changed it: recorded as such, never as plain manual.
                "label_source": "human-verified" if source == "manual" and proposed else source,
                "proposed_label": proposed, "changed_by_human": bool(proposed) and proposed != label, "proposal_set": r.get("proposal_set", ""), "profile_name": r.get("profile_name", ""), "profile_description": r.get("profile_description", ""),
                "profile_keywords": keywords, "profile_focus": r.get("profile_focus", ""), "in_catalog": pid in papers,
                "title": r.get("title") or papers.get(pid, {}).get("title", ""), "source_file": path.name,
            }
            # One label per (labeller, profile, paper): the latest wins.
            key = (row["labeler"], row["profile_name"], pid)
            if key in seen:
                out[seen[key]] = row
            else:
                seen[key] = len(out)
                out.append(row)
    return out, problems


def kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n < 2:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in LABELS) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def agreement(labels: list[dict]) -> list[str]:
    by = {}
    for r in labels:
        by.setdefault((r["profile_name"], r["paper_id"]), {})[r["labeler"]] = r["label"]
    pairs: dict[tuple[str, str], tuple[list, list]] = {}
    for votes in by.values():
        names = sorted(votes)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = pairs.setdefault((names[i], names[j]), ([], []))
                a.append(votes[names[i]])
                b.append(votes[names[j]])
    lines = []
    for (x, y), (a, b) in sorted(pairs.items()):
        k = kappa(a, b)
        agree = sum(p == q for p, q in zip(a, b)) / len(a)
        lines.append(f"| {x} vs {y} | {len(a)} | {agree:.0%} | {'—' if k is None else f'{k:.2f}'} |")
    return lines


def pct(n: int, d: int) -> str:
    return f"{n / d:.0%}" if d else "—"


def eda(papers: list[dict], manual: list[dict], synthetic: list[dict]) -> str:
    n = len(papers)
    areas = Counter(p["area_label"] for p in papers)
    venue = Counter(p["venue_type"] or "unknown" for p in papers)
    wtype = Counter(p["work_type"] or "unknown" for p in papers)
    words = sorted(p["abstract_words"] for p in papers)
    q = lambda f: words[min(len(words) - 1, int(f * len(words)))]  # noqa: E731
    levels = Counter(p["signal_level"] for p in papers)
    sig = lambda k: sum(1 for p in papers if p["signals"][k] is True)  # noqa: E731
    ev = Counter(e for p in papers for e in p["signals"]["evidence"])
    dates = sorted(p["published"] for p in papers if p["published"])
    out = [
        "## Exploratory data analysis", "",
        f"**Papers:** {n:,} · published {dates[0]} to {dates[-1]} · source OpenAlex (CC0) · English abstracts of ≥ 30 words.", "",
        "### Papers per area", "", "| Area | Papers | Share |", "|---|---|---|",
        *[f"| {a} | {c} | {pct(c, n)} |" for a, c in areas.most_common()], "",
        "### Publication type", "", "| Venue type | Papers |  | Work type | Papers |", "|---|---|---|---|---|",
        *[f"| {v} | {c} |  | {w} | {d} |" for (v, c), (w, d) in zip(venue.most_common(6) + [("", "")] * 6, wtype.most_common(6) + [("", "")] * 6) if v or w], "",
        f"### Abstract length (words)\n\nmin {words[0]} · 25% {q(.25)} · median {q(.5)} · 75% {q(.75)} · max {words[-1]}. "
        f"{sum(w < 100 for w in words)} abstracts ({pct(sum(w < 100 for w in words), n)}) are under 100 words and get the “Short abstract” caution.", "",
        "### Worth-it signals", "", "| Signal | Papers | Share |", "|---|---|---|",
        *[f"| {label} | {c} | {pct(c, n)} |" for label, c in [
            ("Peer-reviewed venue", sig("peer_reviewed")), ("Established (core) venue", sig("established_venue")), ("Review / survey", sig("review")),
            ("Code released", sig("code")), ("Data released", sig("data")), ("Any study-design cue", sum(1 for p in papers if p["signals"]["evidence"])),
            ("Open access", sig("open_access")), ("Cited above field average (FWCI ≥ 1.5)", sig("impact")),
            ("Caution: short abstract", sig("thin_abstract")), ("Caution: promotional wording", sig("promotional")), ("Caution: few references", sig("few_references"))]], "",
        f"Signal level: strong {levels['strong']} · moderate {levels['moderate']} · limited {levels['limited']}. "
        "Study-design cues: " + ", ".join(f"{k} {v}" for k, v in ev.most_common()) + ".", "",
        "Signals vary by field: code and data releases are rare in the humanities, so their papers skew towards “limited”. "
        "Read the level as “how much the abstract and metadata let you check”, not as a quality grade.", "",
        "### Signal level by area", "", "| Area | Strong | Moderate | Limited |", "|---|---|---|---|",
    ]
    for a in areas:
        c = Counter(p["signal_level"] for p in papers if p["area_label"] == a)
        t = sum(c.values())
        out.append(f"| {a} | {pct(c['strong'], t)} | {pct(c['moderate'], t)} | {pct(c['limited'], t)} |")
    for name, rows in (("Manual labels", manual), ("Synthetic labels", synthetic)):
        out += ["", f"### {name}", ""]
        if not rows:
            out.append("None yet. Label papers in the app (Label tab), export them, and put the files in "
                       f"`data/labels/{'manual' if name.startswith('Manual') else 'synthetic'}/`.")
            continue
        c = Counter(r["label"] for r in rows)
        profiles = Counter(r["profile_name"] for r in rows)
        labelers = Counter(r["labeler"] for r in rows)
        out += [f"{len(rows):,} labels · Read {c['READ']} ({pct(c['READ'], len(rows))}) · Skim {c['SKIM']} ({pct(c['SKIM'], len(rows))}) · "
                f"Skip {c['SKIP']} ({pct(c['SKIP'], len(rows))})", "",
                "| Profile | Labels | Read | Skim | Skip |", "|---|---|---|---|---|"]
        for prof, k in profiles.most_common():
            cc = Counter(r["label"] for r in rows if r["profile_name"] == prof)
            out.append(f"| {prof} | {k} | {cc['READ']} | {cc['SKIM']} | {cc['SKIP']} |")
        out += ["", "Labellers: " + ", ".join(f"{k} ({v})" for k, v in labelers.most_common()) + "."]
        src = Counter(r["label_source"] for r in rows)
        out += ["", "Provenance: " + ", ".join(f"{k} {v}" for k, v in src.most_common()) + "."]
        verified = [r for r in rows if r.get("proposed_label")]
        if verified:
            changed = sum(r["changed_by_human"] for r in verified)
            k = kappa([r["proposed_label"] for r in verified], [r["label"] for r in verified])
            batches = Counter(r["labeled_at"] for r in verified)
            if batches and batches.most_common(1)[0][1] == len(verified) and len(verified) > 1:
                out += ["", f"All {len(verified)} reviewed labels share one timestamp ({verified[0]['labeled_at']}): the labeller reviewed the "
                            "suggestions offline and the results were written to the file in a single batch, so per-label review times are not available."]
            out += ["", f"Model-proposed labels checked by a person: {len(verified)}. The person changed {changed} "
                        f"({pct(changed, len(verified))}); agreement between proposal and final label: Cohen’s κ "
                        f"{'—' if k is None else f'{k:.2f}'}.", "",
                    "| Proposed → final | Read | Skim | Skip |", "|---|---|---|---|"]
            for a in LABELS:
                cc = Counter(r["label"] for r in verified if r["proposed_label"] == a)
                out.append(f"| {a} | {cc['READ']} | {cc['SKIM']} | {cc['SKIP']} |")
        ag = agreement(rows)
        if ag:
            out += ["", "Agreement on papers labelled by two people for the same profile:", "",
                    "| Pair | Shared papers | Agreement | Cohen’s κ |", "|---|---|---|---|", *ag]
        by_level = {}
        papers_by_id = {p["id"]: p for p in papers}
        for r in rows:
            lvl = papers_by_id.get(r["paper_id"], {}).get("signal_level")
            if lvl:
                by_level.setdefault(lvl, Counter())[r["label"]] += 1
        if by_level:
            out += ["", "Label mix by signal level (do people prefer papers with stronger signals?):", "",
                    "| Signal level | Labels | Read | Skim | Skip |", "|---|---|---|---|---|"]
            for lvl in ("strong", "moderate", "limited"):
                cc = by_level.get(lvl, Counter())
                t = sum(cc.values())
                out.append(f"| {lvl} | {t} | {pct(cc['READ'], t)} | {pct(cc['SKIM'], t)} | {pct(cc['SKIP'], t)} |")
    return "\n".join(out) + "\n"


CARD = """---
license: cc-by-4.0
language:
  - en
pretty_name: Paper Triage — research papers with Read/Skim/Skip relevance labels
size_categories:
  - 1K<n<10K
task_categories:
  - text-classification
  - text-retrieval
tags:
  - research-papers
  - recommendation
  - relevance
  - openalex
configs:
  - config_name: papers
    data_files: papers.jsonl
  - config_name: labels_manual
    data_files: labels_manual.jsonl
  - config_name: labels_synthetic
    data_files: labels_synthetic.jsonl
---

# Paper Triage dataset

Recent research papers across all 26 OpenAlex fields, with transparent "worth-it" signals and
human **Read / Skim / Skip** judgements made against written research profiles. It is the
data behind [Paper Triage](https://github.com/srivathsanb14/research-paper-triage), a tool that
helps researchers decide which papers deserve their time.

## Files

| File | Rows | What it is |
|---|---|---|
| `papers.jsonl` | {n_papers} | Paper metadata (title, abstract, authors, venue, dates, links, OpenAlex type, venue type, open-access status, FWCI) plus computed signals |
| `labels_manual.jsonl` | {n_manual} | **Human** labels from the app's Label tab: paper, label, labeller, time, and the research profile it was judged against. `label_source` is `manual` (labelled from scratch) or `human-verified` (an AI model proposed a label from title and abstract and a person accepted or changed it; `proposed_label` and `changed_by_human` record which) |
| `labels_synthetic.jsonl` | {n_synth} | **Synthetic** labels from a keyword rule (`triage/demo.py`). For pipeline tests only, never mixed with manual labels |

## Collection

* **Papers** — `scripts/build_catalog.py` queries the OpenAlex API (CC0) for each field and two AI
  subfields: half the newest and half the most-cited works of the past {days} days, English, with an
  abstract of ≥ 30 words, not retracted, of type article / review / preprint. Areas are balanced and
  de-duplicated by id and normalised title. Snapshot: {since} → {until}.
* **Signals** — rules in `triage/quality_rules.json` (venue type and version, CWTS core list, open-access
  status, FWCI ≥ 1.5, abstract phrases for released code/data and study designs, cautions for short
  abstracts, promotional wording and very short reference lists).
* **Labels** — labellers write a research profile (description, keywords, focus) and judge papers in
  the app with the ranker's prediction hidden. Some labels were made in a review mode: an AI model
  (Claude) proposed a label and reason from the title and abstract only, without the ranker's scores,
  and a person accepted or corrected each one. These rows are `human-verified`; models trained or
  evaluated on them may share the proposer's biases, so their override rate is reported below. Sampling is balanced across five bands of the
  profile score so the set is not all Skip. Read = worth reading in full; Skim = useful but
  peripheral; Skip = low relevance. Labels are subjective by design: they belong to a profile.

## Intended use and limits

Evaluating and training personalised paper rankers; studying how relevance and trust signals
interact. Not a measure of paper quality: signals are cues from metadata and abstract wording,
skewed by field (code and data are rarely mentioned in the humanities). Metadata comes from
OpenAlex and can be wrong or incomplete. Abstracts are publisher text redistributed by OpenAlex;
check publisher terms before reusing abstracts beyond research. Labels reflect a few labellers'
interests and are not a general relevance ground truth.

## Ethics

No personal data about labellers beyond a self-chosen labeller name. Author names are public
bibliographic metadata. Synthetic labels are kept in a separate file and marked
`label_source: synthetic-rule`.

{eda}
"""


def main() -> None:
    papers = load_papers()
    by_id = {p["id"]: p for p in papers}
    manual, p1 = load_labels(ROOT / "data/labels/manual", by_id, "manual")
    synthetic, p2 = load_labels(ROOT / "data/labels/synthetic", by_id, "synthetic-rule")
    for msg in p1 + p2:
        print("skipped", msg, file=sys.stderr)
    OUT.mkdir(exist_ok=True)
    for name, rows in (("papers", papers), ("labels_manual", manual), ("labels_synthetic", synthetic)):
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    info = catalog.manifest()
    report = eda(papers, manual, synthetic)
    days = (date.fromisoformat(info["until"]) - date.fromisoformat(info["since"])).days
    (OUT / "README.md").write_text(CARD.format(n_papers=f"{len(papers):,}", n_manual=f"{len(manual):,}", n_synth=f"{len(synthetic):,}",
                                               since=info["since"], until=info["until"], days=days, eda=report), encoding="utf-8")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/EDA.md").write_text(f"# Dataset EDA\n\nGenerated by `python scripts/build_dataset.py` on {date.today()}.\n\n" + report, encoding="utf-8")
    print(f"Wrote {OUT}: {len(papers)} papers, {len(manual)} manual labels, {len(synthetic)} synthetic labels.")
    if len(manual) < 500:
        print(f"Note: {len(manual)} manual labels; the course rubric asks for at least 500.", file=sys.stderr)


if __name__ == "__main__":
    main()
