# Paper Triage

Too many papers, too little time. Describe your research and every recent paper is sorted into
**Read / Skim / Skip** with a one-line reason and separate "worth-it" signals (peer review, released
code or data, study design, citation impact, cautions). It learns from your ratings and labels.
Runs in the browser; no API key.

**Live:** Hugging Face Space (see [Publish](#publish)) · GitHub Pages `https://srivathsanb14.github.io/research-paper-triage/`

## Requirements

| # | Requirement | How it is met |
|---|---|---|
| 1 | **Functional and useful** | Triage for a real reading backlog. Measured with 5-fold cross-validation on our own labels (NDCG@10, average precision, papers to screen for 80% of the good ones, time saved) against profile-only, semantic-only, keyword-only and random baselines: [report](docs/evaluation/report.md), the Insights page, and a learning curve of quality vs. number of labels. |
| 2 | **≥ 500 manual samples** | 500 labels in [`data/labels/manual/`](data/labels/manual) over a 1,859-paper catalog (OpenAlex, CC0). Rule-generated labels, including the demo accounts' mock data, live apart in [`data/labels/synthetic/`](data/labels/synthetic) and are never counted. Card and EDA: [docs/EDA.md](docs/EDA.md). |
| 3 | **≥ 2 model types** | **Off-the-shelf:** all-MiniLM-L6-v2 embeddings ([card](hf/model-embeddings/README.md)). **Trained from scratch:** a per-user ridge ranker over seven readable features, blended with a hand-set profile score by a cross-validated weight ([card](hf/model-ranker/README.md)). |
| 4 | **Public GUI on Hugging Face Spaces** | Static Space built by `scripts/build_pages.py --hf-space`, published with `scripts/publish_hf.sh`. |

## Use it

1. Pick your fields and optionally describe your research, then **Show papers**.
2. Rate at least 5 of the first 20 as 👍 or 👎 (or hand-label in **Label**). The rest are sorted into Read / Skim / Skip and re-sorted with every rating.
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

Code: `web/js/engine.js` (browser), `triage/relevance.py` (Python reference, checked for parity), `web/js/evaluate.js`, `triage/quality_rules.json`.

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

## Data workflow

1. Label papers in **Label** (predictions hidden) and export **Labels JSONL** into `data/labels/manual/`.
2. `python scripts/build_dataset.py` validates and merges labels, computes agreement, and writes `_dataset/` and `docs/EDA.md`.
3. `node scripts/evaluate_web.mjs data/labels/manual/*.jsonl` writes `docs/evaluation/report.md`.
4. Synthetic labels: `scripts/simulate_labels.py`, `scripts/simulate_users.py`.

## Publish

* **Hugging Face:** `hf auth login`, then `scripts/publish_hf.sh <hf-user>` creates the Space, dataset and both model cards.
* **GitHub Pages:** Settings → Pages → Source: GitHub Actions. The workflow tests, refreshes the catalog daily and deploys.

## Limitations

* The catalog is a bounded recent sample (~270 papers per area); use *Load more* or import your own papers for depth.
* English abstracts only. Signals come from metadata and abstract wording and inherit OpenAlex errors.
* Without the optional server, data lives in one browser; use Settings → Download backup.
* The original Streamlit app is still in `app.py` (`streamlit run app.py`) as the server-side research edition.

## AI assistance

Large parts of the code, tests and docs were written with Claude Code under the team's direction. Claude also proposed labels for the review sets in `data/labels/proposals/`; people reviewed them before they became dataset rows (provenance is stored per row).
<!-- Team: add your own reflection: what you asked the tools for, what they got wrong, what you rewrote. -->
