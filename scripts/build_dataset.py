"""Package the paper catalog and the Read/Skim/Skip labels as a Hugging Face dataset.

    python scripts/build_dataset.py            # → _dataset/  (+ docs/EDA.md)
    hf upload <user>/paper-triage-dataset _dataset . --repo-type=dataset

Inputs (papers and labels are kept apart; labels point at papers by id)
  data/catalog/            the public OpenAlex snapshot the app ranks (CC0 metadata)
  data/labels/profiles.json  the research profiles labels are judged against
  data/labels/labels.jsonl   one row per (profile, paper): label, time, origin

Outputs
  _dataset/papers.jsonl    one row per paper, with the worth-it signals
  _dataset/profiles.json   the research profiles
  _dataset/labels.jsonl    validated labels (origin: manual or synthetic)
  _dataset/README.md       dataset card with EDA
  docs/EDA.md              the same exploratory analysis for the repository
"""

from __future__ import annotations

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
ORIGINS = ("manual", "synthetic")
OUT = ROOT / "_dataset"


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


def load_labels(papers: dict[str, dict], profiles: dict[str, dict]) -> tuple[list[dict], list[str]]:
    out, problems, seen = [], [], set()
    path = ROOT / "data/labels/labels.jsonl"
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        where = f"{path.name}:{i}"
        if r.get("label") not in LABELS:
            problems.append(f"{where}: unknown label {r.get('label')!r}")
        elif r.get("origin") not in ORIGINS:
            problems.append(f"{where}: unknown origin {r.get('origin')!r}")
        elif r.get("paper_id") not in papers:
            problems.append(f"{where}: paper {r.get('paper_id')!r} is not in the catalog")
        elif r.get("profile") not in profiles:
            problems.append(f"{where}: unknown profile {r.get('profile')!r}")
        elif (r["profile"], r["paper_id"]) in seen:
            problems.append(f"{where}: duplicate label for {r['profile']} / {r['paper_id']}")
        else:
            seen.add((r["profile"], r["paper_id"]))
            out.append({k: r[k] for k in ("profile", "paper_id", "label", "labeled_at", "origin")})
    return out, problems


def pct(n: int, d: int) -> str:
    return f"{n / d:.0%}" if d else "—"


def eda(papers: list[dict], labels: list[dict], profiles: dict[str, dict]) -> str:
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
    out += ["", "### Labels", ""]
    c = Counter(r["label"] for r in labels)
    out += [f"{len(labels):,} labels over {len({r['paper_id'] for r in labels}):,} papers and {len({r['profile'] for r in labels})} research profiles · "
            f"Read {c['READ']} ({pct(c['READ'], len(labels))}) · Skim {c['SKIM']} ({pct(c['SKIM'], len(labels))}) · "
            f"Skip {c['SKIP']} ({pct(c['SKIP'], len(labels))})", "",
            "| Profile | Origin | Labels | Read | Skim | Skip |", "|---|---|---|---|---|---|"]
    for slug, k in Counter(r["profile"] for r in labels).most_common():
        cc = Counter(r["label"] for r in labels if r["profile"] == slug)
        origin = "/".join(sorted({r["origin"] for r in labels if r["profile"] == slug}))
        out.append(f"| {profiles[slug]['name']} | {origin} | {k} | {cc['READ']} | {cc['SKIM']} | {cc['SKIP']} |")
    origins = Counter(r["origin"] for r in labels)
    out += ["", "Origin: " + ", ".join(f"{k} {v}" for k, v in origins.most_common()) + ". "
            "`manual` = a person judged the paper for the profile (in review mode a model's suggestion was shown first and the person "
            "accepted or changed it); `synthetic` = a rule-based mockup generated by a transparent keyword rule (`scripts/simulate_labels.py`) "
            "for the demo accounts and pipeline tests.", ""]
    by_level: dict[str, Counter] = {}
    papers_by_id = {p["id"]: p for p in papers}
    for r in labels:
        lvl = papers_by_id.get(r["paper_id"], {}).get("signal_level")
        if lvl:
            by_level.setdefault(lvl, Counter())[r["label"]] += 1
    out += ["Label mix by signal level (do people prefer papers with stronger signals?):", "",
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
  - config_name: labels
    data_files: labels.jsonl
  - config_name: profiles
    data_files: profiles.json
---

# Paper Triage dataset

Recent research papers across all 26 OpenAlex fields, with transparent "worth-it" signals and
human **Read / Skim / Skip** judgements made against written research profiles. It is the
data behind [Paper Triage](https://github.com/srivathsanb14/research-paper-triage), a tool that
helps researchers decide which papers deserve their time.

## Files

Papers and labels are separate tables; a label points at a paper by `paper_id` and at a research profile by `profile`.

| File | Rows | What it is |
|---|---|---|
| `papers.jsonl` | {n_papers} | Paper metadata (title, abstract, authors, venue, dates, links, OpenAlex type, venue type, open-access status, FWCI) plus computed signals |
| `profiles.json` | {n_profiles} | The research profiles labels are judged against: description, keywords, focus, excluded topics, fields |
| `labels.jsonl` | {n_labels} | One row per (profile, paper): `label` (READ / SKIM / SKIP), `labeled_at`, and `origin`: `manual` (a person judged the paper) or `synthetic` (a rule-based mockup from a transparent keyword rule, for the demo accounts and pipeline tests; never evidence of quality) |

## Collection

* **Papers** — `scripts/build_catalog.py` queries the OpenAlex API (CC0) for each field and two AI
  subfields: half the newest and half the most-cited works of the past {days} days, English, with an
  abstract of ≥ 30 words, not retracted, of type article / review / preprint. Areas are balanced and
  de-duplicated by id and normalised title. Snapshot: {since} → {until}.
* **Signals** — rules in `triage/quality_rules.json` (venue type and version, CWTS core list, open-access
  status, FWCI ≥ 1.5, abstract phrases for released code/data and study designs, cautions for short
  abstracts, promotional wording and very short reference lists).
* **Labels** — each research profile has a description, keywords and a focus. Labels are Read (worth reading
  in full), Skim (useful but peripheral) or Skip (low relevance), judged on the title and abstract. The manual labels were made in a
  review mode: an AI model (Claude) proposed a label and reason from the title and abstract only, without the
  ranker's scores, and a person accepted or corrected each one (models trained or evaluated on
  these may share the proposer's biases). Candidate papers are sampled across five bands of the profile score so
  the set is not all Skip. Labels are subjective by design: they belong to a profile.

## Intended use and limits

Evaluating and training personalised paper rankers; studying how relevance and trust signals
interact. Not a measure of paper quality: signals are cues from metadata and abstract wording,
skewed by field (code and data are rarely mentioned in the humanities). Metadata comes from
OpenAlex and can be wrong or incomplete. Abstracts are publisher text redistributed by OpenAlex;
check publisher terms before reusing abstracts beyond research. Labels reflect a few research profiles'
interests and are not a general relevance ground truth.

## Ethics

Labels carry no information about who made them. Author names are public bibliographic
metadata. Synthetic labels are marked `origin: synthetic`.

{eda}
"""


def main() -> None:
    papers = load_papers()
    by_id = {p["id"]: p for p in papers}
    meta = json.loads((ROOT / "data/labels/profiles.json").read_text(encoding="utf-8"))
    profiles = {p["slug"]: p for p in meta["profiles"]}
    labels, problems = load_labels(by_id, profiles)
    for msg in problems:
        print("skipped", msg, file=sys.stderr)
    OUT.mkdir(exist_ok=True)
    for name, rows in (("papers", papers), ("labels", labels)):
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (OUT / "profiles.json").write_text(json.dumps(list(profiles.values()), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    info = catalog.manifest()
    report = eda(papers, labels, profiles)
    days = (date.fromisoformat(info["until"]) - date.fromisoformat(info["since"])).days
    (OUT / "README.md").write_text(CARD.format(n_papers=f"{len(papers):,}", n_labels=f"{len(labels):,}", n_profiles=len(profiles),
                                               since=info["since"], until=info["until"], days=days, eda=report), encoding="utf-8")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/EDA.md").write_text(f"# Dataset EDA\n\nGenerated by `python scripts/build_dataset.py` on {date.today()}.\n\n" + report, encoding="utf-8")
    print(f"Wrote {OUT}: {len(papers)} papers, {len(profiles)} profiles, {len(labels)} labels.")
    if len(labels) < 500:
        print(f"Note: {len(labels)} labels; the course rubric asks for at least 500.", file=sys.stderr)


if __name__ == "__main__":
    main()
