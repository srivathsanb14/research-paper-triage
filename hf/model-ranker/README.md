---
license: mit
language:
  - en
library_name: transformers.js
pipeline_tag: text-ranking
tags:
  - recommender
  - research-papers
  - learning-to-rank
  - ridge-regression
  - personalization
base_model: Xenova/all-MiniLM-L6-v2
datasets:
  - <hf-user>/paper-triage-dataset
---

# Paper Triage relevance ranker (trained from scratch)

A small, interpretable model that scores how relevant a research paper is to *one*
researcher, then sorts papers into **Read / Skim / Skip**. It runs in the browser in
[Paper Triage](https://github.com/srivathsanb14/research-paper-triage) and is trained, per
person, on that person's own ratings and hand labels. No weights are shared between users:
this repository publishes the model definition (`config.json`), its training procedure, and
its evaluation.

## What it does

1. **Features.** Seven readable signals for each paper, each roughly in [0, 1]:
   similarity to the research description, to the current focus and to the closest keyword
   (cosine on [MiniLM embeddings](../model-embeddings/README.md), calibrated to [0, 1]),
   exact keyword coverage (title hits count more), recency, an excluded-topic penalty,
   and similarity to papers the person rated relevant minus not relevant (top-3 mean,
   computed leave-self-out so a paper never matches its own rating).
2. **Profile score (configured).** A fixed weighting of those features (`config.json → features[].prior_weight`).
   It works from the first visit, with zero labels.
3. **Learned score (trained from scratch).** A weighted ridge regression (α = 1, scikit-learn
   semantics) on the same features. Targets: Read 1, Skim 0.5, Skip 0. Examples and weights:
   papers named as essential (2.0), hand labels (1.0), corrections (1.5), relevant / not
   relevant / saved (1.0), opened (0.3) and hidden (0.5). Training starts at 5 ratings, the number the app asks for before it triages.
4. **Validated blend.** `final = (1 − w)·profile + w·learned`, with w ∈ {0, 0.15, 0.3, 0.5, 0.7}
   chosen by 5-fold cross-validated average precision on the person's hand labels. Learning
   gets a say only if it beats w = 0 by 0.01. With fewer than 6 hand labels w = min(0.3, n/(n+30)).
5. **Triage.** Read ≥ 0.62, Skim ≥ 0.40 (adjustable; the app suggests cutoffs that maximise
   macro-F1). Read is capped by the weekly reading time (30 min per paper); overflow moves to Skim.
   A person's own correction always overrides the model.

The model lives in `web/js/engine.js` and runs in the browser. `tests/test_web_parity.py` checks its ridge regression against scikit-learn
(coefficients agree to 1e-9).

## Why this design

* **Few, subjective labels.** A person rates tens of papers, not thousands. A 7-feature ridge
  model can't overfit much, trains in milliseconds in a browser, and its weights can be read.
* **Useful on day one.** The profile score needs no labels; learning is layered on top only
  when it measurably helps (the blend is validated, not assumed).
* **Explainable.** Every score decomposes into feature contributions, shown in the app's
  "Why this score" panel, and one-line reasons cite only terms found in the paper.
* Alternatives considered: fine-tuning a cross-encoder (needs far more labels and a GPU;
  can't run per-user in a browser), an LLM judge (costs money per paper, sends data to an API,
  "may invent overlap"), and pure embedding similarity (kept as a baseline below).

## Training data

Per user: their ratings and hand labels in the app. For the reported evaluation: the
[Paper Triage dataset](https://huggingface.co/datasets/<hf-user>/paper-triage-dataset), whose
`labels` table holds Read/Skim/Skip labels made against written research profiles (`profiles` table), joined to `papers` by paper id.

## Evaluation

`node scripts/evaluate_web.mjs` reproduces the numbers from `data/labels/`. Everything is
5-fold cross-validated: no paper is scored by a model that saw its label. Baselines: profile only,
learned only, semantic similarity only, keyword matches only, and random order.

**Manual labels** (500 papers, 3 research profiles; a person judged each paper, starting from a model's suggestion; see the dataset card). MiniLM space, “good” = Read, 5-fold cross-validated:

<!-- eval:manual -->
| Profile | Labels (Read/Skim/Skip) | Method | NDCG@10 | AP | Good in top 10 | Papers to find 80% of good |
|---|---|---|---|---|---|---|
| Cancer immunotherapy | 100 (3/15/82) | Personalised | 0.97 | 1.00 | 100% | 3 |
|  |  | Profile only | 0.97 | 1.00 | 100% | 3 |
|  |  | Semantic similarity only | 0.98 | 0.92 | 100% | 4 |
|  |  | Keyword matches only | 0.89 | 0.87 | 100% | 5 |
|  |  | Random order | — | 0.03 | 3% | 80 |
| Climate adaptation | 200 (3/24/173) | Personalised | 0.90 | 0.83 | 100% | 6 |
|  |  | Profile only | 0.90 | 0.83 | 100% | 6 |
|  |  | Semantic similarity only | 0.88 | 0.67 | 100% | 9 |
|  |  | Keyword matches only | 0.76 | 0.71 | 67% | 25 |
|  |  | Random order | — | 0.01 | 2% | 160 |
| RAG & LLM evaluation | 200 (6/33/161) | Personalised | 0.97 | 1.00 | 100% | 5 |
|  |  | Profile only | 0.97 | 1.00 | 100% | 5 |
|  |  | Semantic similarity only | 0.93 | 0.93 | 100% | 5 |
|  |  | Keyword matches only | 0.91 | 0.78 | 83% | 8 |
|  |  | Random order | — | 0.03 | 3% | 160 |
<!-- /eval:manual -->

<!-- eval:compare -->
MiniLM vs TF-IDF (personalised ranker, AP for Read): Cancer immunotherapy 1.00 vs 1.00; Climate adaptation 0.83 vs 0.78; RAG & LLM evaluation 1.00 vs 0.83.

The cross-validated blend chose a learning weight of **0%** for all manual profiles: training on these labels did not beat the profile score, so the safeguard kept learning off.
<!-- /eval:compare -->

Caveats: only 3–6 Read papers per profile, so AP for Read moves a lot with a single paper; labels began as suggestions from a language model reading the same title and abstract, which makes agreement with text-similarity rankers optimistic. Full tables (including good = Read or Skim and TF-IDF): `docs/evaluation/report.md`.

Pipeline check on **synthetic** labels (the "RAG research (rule-based demo)" profile: 529 AI/computing papers labelled by the keyword rule in
`scripts/simulate_labels.py`, "good" = Read). This only shows the pipeline works. The rule is keyword-based,
so keyword matching is expected to win:

<!-- eval:synthetic -->
| Method (MiniLM space) | NDCG@10 | AP | Good in top 10 | Papers to find 80% of good |
|---|---|---|---|---|
| Personalised (learning 70%) | 0.68 | 0.50 | 44% | 72 |
| Profile only | 0.68 | 0.49 | 44% | 78 |
| Semantic similarity only | 0.69 | 0.51 | 44% | 67 |
| Keyword matches only | 0.76 | 0.56 | 56% | 69 |
| Random order | — | 0.02 | 2% | 424 |
<!-- /eval:synthetic -->

## Intended use

Ranking a few thousand candidate papers for one researcher, as a reading aid. The final decision
stays with the person: the app shows reasons, score breakdowns and trust signals, and every
label can be overridden.

## Limitations

* Only as good as the profile text and the abstract. Papers without abstracts can't be ranked well.
* English only (MiniLM and the keyword rules).
* Few labels → noisy validation; differences of a few AP points are within noise.
* Recency is a weak prior (catalog papers are all from the past 180 days).
* The relevance score is not a quality score. Trust signals are shown separately and never
  change it.
