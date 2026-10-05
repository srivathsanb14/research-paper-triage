# Paper Triage

Too many papers, too little time. Describe your research and every recent paper is sorted into
**Read / Skim / Skip** with a one-line reason and separate "worth-it" signals (peer review, released
code or data, study design, citation impact, cautions). It learns from your ratings and labels.
Runs in the browser; no API key. **Load more** pulls fresh papers live from OpenAlex, Europe PMC and Crossref (Settings → Sources).

**Live:** [Hugging Face Space](https://huggingface.co/spaces/ishaanamahajan/paper-triage) · [dataset](https://huggingface.co/datasets/ishaanamahajan/paper-triage-dataset) · model cards: [ranker](https://huggingface.co/ishaanamahajan/paper-triage-ranker), [MiniLM](https://huggingface.co/ishaanamahajan/paper-triage-minilm-embeddings) · GitHub Pages `https://srivathsanb14.github.io/research-paper-triage/`

## Requirements

Course paperwork (technical submission text, check-in figure) lives in [`course/`](course/); what each script does is in [`scripts/README.md`](scripts/README.md).

| # | Requirement | How it is met |
|---|---|---|
| 1 | **Functional and useful** | Triage for a real reading backlog. Measured with 5-fold cross-validation on our own labels (NDCG@10, average precision, papers to screen for 80% of the good ones, time saved) against profile-only, semantic-only, keyword-only and random baselines: [report](docs/evaluation/report.md), the Insights page, and a learning curve of quality vs. number of labels. |
| 2 | **≥ 500 manual samples** | **500 manual** labels (a person judged each paper for a research profile) plus **984 synthetic** rule-based mockups, in one table, [`data/labels/labels.jsonl`](data/labels/labels.jsonl), over a 1,859-paper catalog of real papers (OpenAlex, CC0). The `origin` column (`manual` / `synthetic`) separates them, and only the manual ones count towards the 500. Card and EDA: [docs/EDA.md](docs/EDA.md). |
| 3 | **≥ 2 model types** | **Off-the-shelf:** all-MiniLM-L6-v2 embeddings ([card](hf/model-embeddings/README.md)). **Trained from scratch:** a per-user ridge ranker over seven readable features, blended with a hand-set profile score by a cross-validated weight ([card](hf/model-ranker/README.md)). |
| 4 | **Public GUI on Hugging Face Spaces** | Static Space built by `scripts/build_pages.py --hf-space`, published with `scripts/publish_hf.sh`. |

## Use it

New here? Read [how it works](docs/HOW_IT_WORKS.md): one flowing guide from papers to scores, labels, the blend, cross-validation and the metrics, with the exact math at each step.

1. Pick your fields and optionally describe your research, then **Show papers**.
2. Rate at least 5 of the first 20 (more papers: **Load more**, de-duplicated across sources) as 👍 or 👎 (or hand-label in **Label**). The rest are sorted into Read / Skim / Skip and re-sorted with every rating.
3. **Insights** shows how well the ranking finds what you would pick, and how fast it learns you.
4. **Saved** exports BibTeX, RIS or CSV. **Settings** has cutoffs and backups.

## How it works

```
 description ─┐                    ┌─ profile score (hand-set weights)
 papers ──────┼─ MiniLM ─ 7 features ─┤
 ratings ─────┘                    └─ ridge regression (trained per user)
                                       final = (1−w)·profile + w·learned, w by cross-validation
 metadata + abstract ─ rules ─ worth-it signals (shown beside the score, never mixed in)
```

* **Relevance ≠ quality.** Signals are checkable cues, not a verdict, and they never change relevance.
* **Validated learning.** A few subjective labels can make a learned model worse, so its weight stays 0 unless it beats the profile alone in cross-validation.
* **Grounded reasons.** Explanations cite only words found in the paper.

Code: the ranker is `web/js/engine.js`, its evaluation is `web/js/evaluate.js`, and the worth-it rules are `triage/quality_rules.json` (shared by the browser and the Python dataset tools). A test checks the browser's ridge regression against scikit-learn and its worth-it signals against Python.

## Run

```bash
python3 scripts/build_pages.py                                  # builds _site/
python3 -m http.server 8000 --bind 127.0.0.1 --directory _site  # static site, browser-only
```

Optional accounts (sign-in, per-user profiles and labels in SQLite, starter label sets):

```bash
pip install -r requirements.txt
python -m server                  # http://localhost:8000, serves _site/ and /api
python -m server.demo_users       # demo accounts sri, ishaan, chris (password "demo", simulated labels)
```

Tests: `pytest -q` · `npm run test:js` · `npm run test:pages` (Playwright end to end; `npx playwright install chromium` first).

## Data

```
data/catalog/       PAPERS: 1,859 papers, signals and MiniLM vectors (read-only snapshot)
data/labels/        LABELS: profiles.json (research profiles) + labels.jsonl (profile, paper_id, label, labeled_at, origin)
data/users.db       USERS: accounts, sessions, each user's saved state and labels (SQLite; nothing about papers)
```

Papers, labels and users are separate: labels point at papers by id, and a user's labels are their own copy in `users.db`.

1. Label papers in **Label** (predictions hidden) and export **Labels JSONL**; its rows have the same shape as `data/labels/labels.jsonl`, so append them (origin `manual`).
2. `python scripts/build_dataset.py` validates labels against the catalog and writes `_dataset/` and `docs/EDA.md`.
3. `node scripts/evaluate_web.mjs` writes `docs/evaluation/report.md` (5-fold cross-validation per profile).
4. `python scripts/simulate_labels.py` regenerates the synthetic (rule-based) labels and leaves the manual ones untouched.

## Publish

* **Hugging Face:** `hf auth login`, then `scripts/publish_hf.sh <hf-user>` creates the Space, dataset and both model cards.
* **GitHub Pages:** Settings → Pages → Source: GitHub Actions. The workflow tests, refreshes the catalog daily and deploys.

## Limitations

* The catalog is a bounded recent sample (~270 papers per area); use *Load more* or import your own papers for depth.
* English abstracts only. Signals come from metadata and abstract wording and inherit OpenAlex errors.
* Without the optional server, data lives in one browser; use Settings → Download backup.

## AI assistance

Large parts of the code, tests and docs were written with Claude Code under the team's direction. Claude also proposed labels for the review sets in `data/labels/proposals/`; a person judged each one before it became a manual label.
<!-- Team: add your own reflection: what you asked the tools for, what they got wrong, what you rewrote. -->
