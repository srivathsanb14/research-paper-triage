# Personalized Research Paper Triage

*"Which papers are actually worth my time?"*

Describe your research once; every batch of new papers comes back ranked and sorted into
**Read / Skim / Skip**, each with a one-line reason. Your feedback retrains the ranking.

## Run

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Open http://localhost:8501. On first launch a **Demo — RAG research** profile is created: ~400 real
arXiv papers (bundled, offline) with **simulated** labels and feedback from a rule-based stand-in
researcher (`triage/demo.py`), so every view has data. The app marks it as a demo; reset it under
Settings. Create your own profile from the sidebar for real use.

The active profile, tab and reading-list filter are kept in the URL, so refreshing or sharing a link
returns you to the same view.

Optional environment variables:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Enables Claude-written explanations (Settings → Explanations). |
| `TRIAGE_LLM_MODEL` | Model for explanations (default `claude-opus-5`). |
| `S2_API_KEY` | Semantic Scholar key; anonymous requests are heavily rate-limited. |
| `TRIAGE_EMBEDDING_BACKEND` | `auto` (default), `sbert`, or `tfidf`. |
| `TRIAGE_DB_PATH` | SQLite location (default `data/triage.db`). |
| `SLACK_WEBHOOK_URL` / SMTP vars | Digest delivery (optional). |

Tests: `.venv/bin/python -m pytest`

## GitHub Pages (browser edition)

The Pages edition runs the Python ranking engine in the visitor's browser using
[Stlite](https://stlite.net/browser/) 1.9.2 and Pyodide. It needs no Python server,
account, API key, or paid hosting. The existing `streamlit run app.py` command still
runs the server edition.

| Feature | Browser edition |
|---|---|
| Ranking, feedback, labels, evaluation, reading budget | Available; TF-IDF + LSA and learned regression |
| Broad research catalog | 1,800 OpenAlex records across six areas, covering all 26 fields |
| Getting started | Choose fields, optionally describe your interests, then click **Find papers** |
| Sample papers | 416 bundled records, with simulated demo feedback |
| Your own papers | Import a JSON collection from **Add papers** |
| Seed papers | Select papers already in your collection |
| Profiles and feedback | Saved in this browser using IndexedDB; no sync between devices |
| Backups | **Settings → Download profile backup**; restore from the sidebar |
| BibTeX, RIS, CSV and Markdown exports | Available |
| Live API searches, DOI/BibTeX resolution, full-text downloads | Server edition only |
| MiniLM embeddings, Claude explanations, Slack/email delivery | Server edition only |
| Shared catalog refresh | Daily on GitHub Actions, even while visitors have the app closed |
| Personal live API fetching while closed | Server edition only |

The first load downloads the Python runtime and scientific packages from public
CDNs and can take a minute or two. An internet connection is needed to load those
assets. Research interests and feedback are processed locally. Clearing the
browser's site data deletes saved profiles; download a backup before doing so.
Only one tab at a time can write this app's saved data. Current browsers with
WebAssembly, IndexedDB and Web Locks are required.

### Shared paper catalog

Visitors need no installation, login, API key, or file import. New visitors see a
field picker; the RAG demo is an optional example. Existing profiles and their
feedback remain intact. **Add papers → Load field papers** adds a fresh selection
to an existing profile. Reload the website to pick up a newly published catalog.

The public catalog covers computing/engineering; maths/physical sciences;
biology/medicine; earth/environment; society/psychology/economics; and
arts/humanities. It samples up to 300 papers per area, balanced across the 26
[OpenAlex fields](https://help.openalex.org/data/fields/). Records must have usable
English abstracts and publication dates within the past 180 days. This is a
bounded selection, not complete coverage of every paper. Titles, abstracts,
authors, dates, categories and links are hosted; PDFs are linked, not mirrored.

Only the selected areas' files are downloaded from the same website. A load adds
100–1,000 papers, balanced across the selected areas. Ranking, learning, research
interests, labels and feedback stay in the browser. Catalog files contain public
metadata only; no visitor data is sent to OpenAlex or GitHub Actions.

The Pages workflow refreshes the catalog daily at 06:23 UTC and on **Run workflow**.
It caches the last validated snapshot and falls back to it if OpenAlex is down or
rate-limited; the checked-in snapshot also supports an initial deployment. The
app shows the catalog's actual update date. GitHub schedules start after the
workflow is merged into the default branch and can be delayed.

For the site owner, an optional repository Actions secret `OPENALEX_API_KEY` gives
the collector its own API allowance if GitHub's shared anonymous quota runs out.
Visitors never supply a key, and the key is never included in site files. No
separate backend service is required. API access:
[OpenAlex authentication](https://help.openalex.org/api/authentication/).

Maintainers can rebuild the snapshot with `.venv/bin/python scripts/build_catalog.py`.
The **Collect paper catalog** workflow also uploads a downloadable snapshot for
review. Failed collections leave the existing manifest unchanged.

### Preview locally

Only Python 3.11+ is needed to build and serve the static files:

```bash
python3.12 scripts/build_pages.py
python3.12 -m http.server 8000 --bind 127.0.0.1 --directory _site
```

Open http://127.0.0.1:8000. Do not open `index.html` directly as a `file://` URL.
The build copies an explicit list of application files and sample data into
`_site/app.zip`; local databases, credentials, caches and `.git` are excluded.
The app source and bundled sample data become publicly downloadable with the site.

### Publish

1. A repository admin or maintainer opens **Settings → Pages** and selects
   **GitHub Actions** as the publishing source.
2. Merge the Pages changes into `main`. The **GitHub Pages** workflow tests the
   Python code, builds `_site`, and deploys it. Pull requests run tests and build
   without publishing.
3. Open the URL shown by the workflow's deployment job. For this repository, the
   expected URL is `https://srivathsanb14.github.io/research-paper-triage/`.
   That URL is not live until the first successful deployment.

The repository is currently private. GitHub Pages on a private repository
requires an eligible paid GitHub plan; public repositories can use GitHub Free.
The workflow does not change repository visibility or account billing.
See [GitHub's Pages requirements](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)
and [publishing permissions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

### Import format

Upload a JSON array (up to 2,000 papers and 10 MB). Each paper needs a unique `id`
and `title`; abstracts substantially improve ranking. Optional fields follow
`triage.models.Paper`. URLs must use `https://` or `http://`.

```json
[
  {
    "id": "local:example-paper",
    "title": "Evaluating Retrieval-Augmented Generation",
    "abstract": "We evaluate retrieval quality and answer faithfulness.",
    "authors": ["A. Researcher"],
    "year": 2026
  }
]
```

Use **Settings → Download paper collection** in either edition to export this
format. A profile backup additionally contains interests, labels, feedback, seeds
and triage settings. Restoring creates a new profile instead of overwriting one.

### Verify the Pages edition

```bash
.venv/bin/pip install -r requirements-core.txt 'anthropic>=1.0' 'pytest>=8.0'
.venv/bin/python -m pytest -q
python3.12 scripts/build_pages.py
npm ci
npx playwright install chromium
npm run test:pages
```

The browser check serves the site under a repository-style URL prefix and verifies
startup, tab locking, feedback persistence across reloads, downloads, paper imports,
labeling, evaluation and mobile layout. If Chrome is already installed, use
`BROWSER_CHANNEL=chrome npm run test:pages` instead of installing Chromium.

## How it maps to the design

| Design box | Code |
|---|---|
| User context | `models.InterestProfile`, sidebar form |
| Research paper database (arXiv / Semantic Scholar) | `triage/sources.py` |
| Preprocessing: clean → fields → embeddings | `triage/preprocess.py`, `triage/embeddings.py` |
| Labels + feedback log | `triage/store.py` (SQLite) |
| User-interest representation — CONFIGURED | `triage/profile.py` |
| Paper relevance model — TRAINED | `triage/relevance.py` |
| Ranking + triage — TUNED | `triage/ranking.py` |
| Explanation generator — CONFIGURED | `triage/explain.py` |
| Evaluation vs. hand labels — EVALUATED | `triage/evaluate.py` |
| Streamlit interface | `app.py` |

**Relevance model.** A transparent prior (semantic similarity to the project, to the current focus and to
each keyword; exact keyword hits; recency; excluded-topic penalty; similarity to papers you rated) is
blended with a ridge regression trained on hand labels and feedback. The learned weight grows with data,
`n / (n + 30)`, capped at 70%, so a handful of subjective labels can't swamp the prior.

**Triage.** Cutoffs are set by human judgment (Settings) and can be tuned on the Evaluation tab. Read is
optionally capped by your weekly reading time; overflow is demoted to Skim.

**Explanations.** Evidence (keyword hits, shared terms) is extracted deterministically. Rule-based
explanations state only that evidence. Claude explanations must cite the paper terms they rely on; each
citation is verified against the abstract and unverifiable explanations are discarded (the design's
"may invent overlap" risk). The project team can also mark explanations accurate/inaccurate.

**Validated learning.** How much the learned model counts is chosen per profile by cross-validation
on your hand labels (0–70%, by average precision for good papers). Learning is switched off unless it
beats the profile-only ranking; Settings shows the decision and the numbers behind it. Without enough
labels to validate, learning is capped at 30%.

**Seed papers.** Paste arXiv IDs/links or DOIs, or upload a BibTeX export (Zotero, Google Scholar).
Seeds pull the interest profile towards them and act as strong positive examples from day one. arXiv
ids are resolved via OAI-PMH, DOIs via Crossref (falling back to Semantic Scholar / the arXiv version
for missing abstracts).

**Daily feed.** With arXiv categories set, the app fetches new listings automatically when opened
(at most once a day) and opens on **New** — papers worth a look that arrived since your last visit.
For fetching while the app is closed, schedule `scripts/daily.py` (see its docstring for cron).
Every fetch is logged (Settings → Fetch history) and the sidebar shows when the last one succeeded.

**Near-duplicates.** Papers that are near-identical (cosine ≥ 0.72 with sentence-transformers) are
grouped under the highest-ranked one, so the top of the list covers more ground.

**Full text for borderline papers.** After a fetch, up to 12 papers whose score is close to a cutoff
are re-scored using their introduction and conclusion (arXiv HTML, falling back to the PDF).

**Export and digest.** Read / Read + Skim lists as BibTeX or RIS (Zotero, Mendeley, EndNote), and a
Markdown digest. Set `SLACK_WEBHOOK_URL` or SMTP variables (`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`,
`SMTP_PASSWORD`, `DIGEST_FROM`, `DIGEST_TO`) to post or email it, from the app or from
`scripts/daily.py --digest slack|email`.

**Uncertainty.** Evaluation metrics carry 90% bootstrap ranges; the learning curve shows them as a band.

**How quickly good papers surface.** On the Evaluation tab:
*discovery curve* — walking down the ranked list, the share of good papers found after k papers,
against random order and a perfect ranking, summarised as papers to review for the first / half / 80%
of good papers and minutes of screening saved; *learning curve* — labels and ratings replayed in the
order given, showing how the top 10 improves with feedback (cross-validated). "Good" is Read by
default, or Read + Skim.

**Evaluation.** k-fold cross-validation (no paper scored by a model trained on its own label), against
profile-only, semantic-only and keyword-only baselines; Spearman ρ, NDCG@10, macro-F1, confusion
matrix, and suggested cutoffs.

## Data sources — known limitations

* **arXiv OAI-PMH** (oaipmh.arxiv.org) is used by `scripts/build_sample_data.py` to add recent
  cs.IR / cs.CL papers (`sources.fetch_arxiv_oai`).
* **arXiv new listings** (rss.arxiv.org) is the most reliable source: the day's announcements in the
  chosen categories. Feeds are empty on weekends and holidays.
* **arXiv keyword search** (export.arxiv.org) currently answers HTTP 406 to Python HTTP clients from
  some networks; the app reports this and suggests the other sources.
* **Semantic Scholar** throttles anonymous requests; set `S2_API_KEY` for reliable use.
* Regenerate the offline sample with `python scripts/build_sample_data.py`.
