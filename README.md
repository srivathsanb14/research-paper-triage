# Paper Triage

*"Which papers are actually worth my time?"*

Papers are cheaper to write than ever: preprint servers, paper mills and AI-assisted writing
mean more new papers every week than anyone can screen. Paper Triage helps a researcher spend
their limited reading time on the right ones:

* **Relevance, personalised.** Describe your research in a sentence. Every paper is sorted into
  **Read / Skim / Skip** with a one-line reason that only cites words actually in the paper.
* **Worth-it signals, separately.** Is it peer-reviewed? Does it release code or data? What study
  design does the abstract describe? Is it cited above its field's average? Any cautions (very
  short abstract, promotional wording)? Filter or sort by these; they never change relevance.
* **Learns your taste, honestly.** Thumbs up/down, corrections and hand labels train a small model
  in your browser. It only gets a say when cross-validation on your own labels shows it helps,
  and the Insights page shows by how much.
* **No setup, private.** A static website: no account, no server, no API key. Your interests,
  ratings and labels never leave the browser.

**Live:** GitHub Pages `https://srivathsanb14.github.io/research-paper-triage/` and a Hugging
Face static Space (see [Publish](#publish)).

## Try it

1. Open the site, pick your fields (and describe your research if you like, or click an example), then **Show papers**.
2. You get 20 papers. Mark at least **5** as 👍 Relevant or 👎 Not relevant (press again to undo).
   **Load more** below the list fetches fresh papers for your fields from OpenAlex.
3. After 5 ratings the algorithm sorts every paper you haven't rated into **Read / Skim / Skip**, with a
   one-line reason and trust signals. Each further rating re-sorts the lists; rated papers move to **Rated**.
   Read is capped by your weekly reading time, and adaptive cutoffs keep it filled with your best matches.
   Keyboard: `J`/`K` move, `U`/`N` rate, `S` save, `X` hide, `Enter` details, `?` help.
4. **Label** papers with predictions hidden, or review suggested labels, and open **Insights** to see how
   well the ranking finds what you would pick: good papers in the top 10, papers to screen for 80% of good
   ones, time saved, NDCG@10 / AP / Spearman against baselines.
5. **Saved** exports BibTeX / RIS / CSV for Zotero, Mendeley or EndNote. Settings has backups.

## How it works

```
 your description ─┐                       ┌─ profile score (hand-set weights, works on day one)
 papers ───────────┼─ MiniLM embeddings ─ 7 readable features ─┤
 your ratings ─────┘   (off-the-shelf)                         └─ ridge regression (trained from scratch, per person)
                                                                     final = (1−w)·profile + w·learned,  w by cross-validation
 metadata + abstract ─ rules ─ worth-it signals (shown beside the score, never mixed in)
```

| Part | What | Where |
|---|---|---|
| Catalog | ~1,900 recent papers across all 26 OpenAlex fields + AI/ML subfields; half newest, half most-cited in the past 180 days; refreshed daily by GitHub Actions | `scripts/build_catalog.py`, `data/catalog/` |
| Off-the-shelf model | all-MiniLM-L6-v2 (q8 ONNX via transformers.js). Catalog vectors precomputed; your text embedded in a Web Worker | `scripts/embed_catalog.mjs`, `web/js/embed-worker.js`, [card](hf/model-embeddings/README.md) |
| Model trained from scratch | 7-feature ridge ranker with a cross-validated blend, Read/Skim/Skip cutoffs and a weekly reading budget | `web/js/engine.js` (browser), `triage/relevance.py` (Python reference), [card](hf/model-ranker/README.md) |
| Worth-it signals | Transparent rules shared by Python and the browser | `triage/quality_rules.json`, `triage/quality.py`, `web/js/quality.js` |
| Explanations | Deterministic evidence (keyword hits, shared terms, best sentence); reasons state only that evidence | `web/js/engine.js` → `evidence`, `reason` |
| Evaluation | 5-fold CV against profile-only, learned-only, semantic-only, keyword-only and random | `web/js/evaluate.js`, `scripts/evaluate_web.mjs`, [report](docs/evaluation/report.md) |
| Dataset | Papers + signals, manual labels, synthetic labels (separate), card with EDA | `scripts/build_dataset.py`, [EDA](docs/EDA.md) |

Design decisions worth knowing:

* **Relevance ≠ quality.** Mixing "is it about my work?" with "is it trustworthy?" into one number
  hides both. Signals are cues you can check (each badge has a tooltip saying why it's there), not a
  verdict, and they are field-dependent (humanities papers rarely mention code).
* **Validated learning.** A handful of subjective labels can make a learned model worse than the
  profile. The learned weight is picked from {0, 0.15, 0.3, 0.5, 0.7} by cross-validated average
  precision and stays 0 unless it beats the profile alone.
* **Grounded explanations.** The original design flagged that an LLM "may invent overlap". Reasons
  are built only from verified evidence; the server edition's optional Claude explanations must cite
  terms that are then checked against the abstract.
* **Fast first visit.** The page is interactive in ~0.5 s with TF-IDF ranking; MiniLM (~23 MB, cached)
  loads in the background and the ranking switches over automatically.

## Run locally

Website (only Python 3.11+ needed to build and serve):

```bash
python3 scripts/build_pages.py
python3 -m http.server 8000 --bind 127.0.0.1 --directory _site    # open http://127.0.0.1:8000
```

Rebuild the catalog and its embeddings (network access to OpenAlex and the Hugging Face Hub):

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-core.txt
npm ci
.venv/bin/python scripts/build_catalog.py    # optional OPENALEX_API_KEY for a larger quota
node scripts/embed_catalog.mjs               # only changed shards are re-embedded
```

Tests:

```bash
.venv/bin/pip install -r requirements-core.txt 'anthropic>=1.0' 'pytest>=8.0'
.venv/bin/python -m pytest -q        # Python engine, catalog, signals, and JS↔Python parity (needs Node)
npm run test:js                      # browser engine unit tests
npx playwright install chromium
python3 scripts/build_pages.py && npm run test:pages   # end-to-end: onboarding → feedback → labels → insights → backup, desktop + mobile
```

## Data and evaluation workflow

1. Each labeller creates a profile for their own research, labels papers in **Label**
   (balanced sampling, predictions hidden), and exports **Labels JSONL**.
2. Put the files in `data/labels/manual/`. Rule-generated labels live in `data/labels/synthetic/`
   (`python scripts/simulate_labels.py`) and are never mixed with manual ones.
3. `python scripts/build_dataset.py` validates and merges labels (latest per labeller, profile and
   paper), computes inter-annotator agreement (Cohen's κ) and writes `_dataset/` plus `docs/EDA.md`.
4. `node scripts/evaluate_web.mjs data/labels/manual/*.jsonl` writes `docs/evaluation/report.md`,
   the numbers quoted in the model cards.

## Publish

* **GitHub Pages**: Settings → Pages → Source: GitHub Actions; merge to `main`. The workflow
  tests everything, refreshes the catalog daily (06:23 UTC), embeds new papers and deploys.
  Pages on a private repository requires a paid plan; public repositories can use GitHub Free.
* **Hugging Face**: `pip install -U huggingface_hub && hf auth login`, then
  `scripts/publish_hf.sh <your-hf-user>` creates/updates the Space (`sdk: static`), the dataset and
  both model cards. To keep the Space in sync automatically, add the repository secret `HF_TOKEN`
  and variable `HF_SPACE` (e.g. `user/paper-triage`); the workflow then mirrors every deploy.

Import format for your own papers (**Import papers (JSON)**): a JSON array of up to 2,000 objects with a
unique `id` and a `title`; `abstract`, `authors`, `venue`, `published`, `url` (http/https) and the other
fields of `triage.models.Paper` are optional.

## Server edition (Streamlit)

The original Python app is still here for research use: `pip install -r requirements.txt && streamlit run app.py`.
It adds live arXiv / Semantic Scholar fetching, DOI/BibTeX seed papers, full-text re-scoring of borderline
papers, sentence-transformers on the server, optional Claude-written explanations (verified against the
abstract) and Slack/email digests. Environment variables: `ANTHROPIC_API_KEY`, `TRIAGE_LLM_MODEL`,
`S2_API_KEY`, `TRIAGE_EMBEDDING_BACKEND` (`auto`/`sbert`/`tfidf`), `TRIAGE_DB_PATH`, `SLACK_WEBHOOK_URL`, SMTP
variables. Its profile backups can be restored in the website.

## Limitations

* The catalog is a bounded, recent sample (~270 papers per area), not complete coverage; use
  *Find more on OpenAlex* or import your own collection for depth.
* English abstracts only. Ranking quality depends on the profile text and the abstract.
* Signals come from metadata and abstract wording and inherit OpenAlex metadata errors.
* Data lives in one browser; use Settings → Download backup to move or keep it.

## AI assistance

Parts of this repository (the browser app, the JavaScript port of the engine, tests, build
scripts and documentation) were written with Claude Code (Anthropic) under the team's direction.
<!-- Team: describe here how you used AI tools and what you reviewed or changed yourselves. -->
