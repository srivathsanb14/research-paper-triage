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
