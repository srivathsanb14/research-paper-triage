# How Paper Triage works

Too many research papers come out for anyone to read them all. Paper Triage sorts them into
**Read**, **Skim** and **Skip**. It does this in three moves: it **guesses** how relevant each paper is
to you, you **tell it the truth** on a few papers, and it **adjusts its guesses** toward your answers,
but only when that really makes them better.

This guide follows one paper's journey, step by step. Each step starts in plain words and then gives the
exact math. All of it is implemented in [`web/js/engine.js`](../web/js/engine.js) (scoring, training,
validation) and [`web/js/evaluate.js`](../web/js/evaluate.js) (metrics and the learning curve). A test
checks the ridge regression against scikit-learn.

**In one line:** score = the app's guess; labels = your answers; learning = nudging the guess toward your
answers, but only when it really helps.

---

## 1. Where the papers come from

The app ships with about 1,900 recent papers from every field, fetched ahead of time from OpenAlex and
stored in `data/catalog/` (with a checksum the browser verifies). Each paper already carries a "meaning
fingerprint" computed in advance, so comparing meanings is instant. **Load more**, a search, or pasting a
DOI or arXiv link pulls fresh papers live from OpenAlex, Europe PMC and Crossref (none needs a key), removes
duplicates, and scores them the same way. Only search terms leave your device. The app works from titles,
abstracts and metadata, never PDFs.

If the optional server is running you sign in first (or choose "Continue without an account"). Your
profiles, ratings and labels are saved to your account, or just in your browser if you skip it.

## 2. What you give it

| Input | How you give it | What it is used for |
|---|---|---|
| **Profile text**: description, focus sentence, keywords, excluded topics | typed in | the seven checks below |
| **Ratings**: 👍 / 👎, Save, Open, Dismiss, corrections | clicks in the feed | training examples |
| **Hand labels**: Read / Skim / Skip with the prediction hidden | the Label tab | training **and** the answer key for testing |

You rate at least 5 of the first 20 papers and the app starts sorting. Hand labels matter most: they teach
the ranker, and they are what the app is tested against.

### Why two kinds of answers

They do different jobs, and the split is deliberate.

| | 👍 / 👎 rating (plus Save, Open, Dismiss) | Hand label: Read / Skim / Skip |
|---|---|---|
| **Where** | the feed, while you browse | the Label tab |
| **What you see** | the app's own Read/Skim/Skip guess | the prediction **hidden** |
| **Choices** | 2 (relevant or not) | 3 (Read, Skim or Skip) |
| **Effort** | one click | a real judgement |
| **Trains the ranker** | yes | yes |
| **Answer key for testing** | **no** | **yes** |

- **Ratings are cheap but biased.** You press them while seeing the app's guess, and the app decides which
  papers you even see, so your answers get nudged toward what it already thinks. Grading the app on them
  would partly grade it on its own choices. Asking for just 5 clicks is what lets it work immediately.
- **Hand labels are slower but clean.** With the prediction hidden they are an independent answer key, and
  three levels (Read, Skim, Skip) are fine enough to measure ranking quality.
- **Ratings also shape the score itself** through check 7, "looks like papers you liked", even before any
  learning happens.
- A person who only ever presses 👍/👎 gets a sorted list and learning, but no measured quality: Insights asks
  for at least 6 hand labels with at least two different answers.

## 3. Meaning as numbers (embeddings)

**In words.** A model turns each text into a list of 384 numbers so that texts with similar *meaning* land
close together. "Grounded generation" ends up near "retrieval-augmented generation" even though they share
no words.

**The math.** We use the off-the-shelf **all-MiniLM-L6-v2**, unchanged. If it hasn't loaded yet, a TF-IDF
fallback ranks in the meantime.

- Vectors have length 1, so the dot product of two vectors is their **cosine similarity** (higher = closer
  in meaning).
- Paper vectors are precomputed; only your own profile text is embedded in the browser.
- Your **profile vector** is a weighted sum of description (1.0), keywords (1.0), focus (1.5) and seed
  papers (1.5), re-scaled to length 1.
- **Calibration.** Raw cosines bunch up, so each is stretched onto 0–1:
  `calibrated = clip((cosine − lo) / (hi − lo), 0, 1)`, with `lo = 0.15`, `hi = 0.50` for paper-vs-profile
  (0.12 and 0.45 for short texts like one keyword).

## 4. Scoring: the app's guess

**In words.** Every paper is asked seven simple questions about your profile. Say your profile is "RAG
evaluation" and the paper is *"Measuring faithfulness in retrieval-augmented generation"*:

| Question | Example answer |
|---|---|
| Does its meaning match my description? | very close |
| Does it match my current focus? | close |
| Does it match one of my keywords by meaning? | yes |
| Do my keywords appear literally? | yes, in the title |
| Is it recent? | yes |
| Does it cover a topic I excluded? | no |
| Is it similar to papers I already liked? | yes |

**The math.** Each answer is a number from 0 to 1:

| # | Feature | How it's computed |
|---|---|---|
| 1 | **semantic** | calibrated similarity: paper vs. main profile vector |
| 2 | **focus** | calibrated similarity to the focus sentence (falls back to #1 if none) |
| 3 | **keyword_sem** | calibrated similarity to the best-matching single keyword |
| 4 | **keyword_lex** | `min(1, (title hits + 0.6 × abstract-only hits) / min(#keywords, 3))` |
| 5 | **recency** | `exp(−age_in_days / 365)`: 1 for today, about 0.37 after a year |
| 6 | **avoid** | `max(clip((similarity − 0.5) × 2), 0.8 if the term appears literally)`, so it fires only when clearly present |
| 7 | **feedback** | calibrated top-3 similarity to papers you liked, minus the same for papers you disliked, **leave-self-out** (a paper never matches its own rating) |

The seven numbers are combined two ways.

**Profile score: hand-set weights, no labels needed.** It works from your first visit, and everyone gets the
same weights:

`prior = clip(0.40·semantic + 0.15·focus + 0.15·keyword_sem + 0.25·keyword_lex + 0.05·recency − 0.40·avoid + 0.25·feedback, 0, 1)`

**Learned score: weights fitted to you.** A **weighted ridge regression** looks at your labelled and rated
papers and finds which of the seven checks actually predicted *your* answers. Maybe you ignore recency but
always care about exact keywords.

- **Targets:** Read = 1, Skim = 0.5, Skip = 0. Ratings: 👍 = 1, 👎 = 0, Save = 1, Open = 0.75, Dismiss = 0.15.
- **Example weights** (how much each counts): hand label 1, rating 1, correction 1.5, seed paper 2, Open about
  0.3, Dismiss 0.5. If a paper has several signals the priority is seed > hand label > correction > explicit
  rating > implicit click.
- **The fit** minimises `Σ wᵢ·(yᵢ − intercept − xᵢ·β)² + α·Σ βⱼ²` with `α = 1`. The penalty keeps the weights
  small, so a few noisy labels can't produce wild ones. The intercept is not penalised. We solve it with a
  short Gaussian-elimination routine and no library.
- **Prediction:** `learned = clip(intercept + Σ βⱼ·featureⱼ, 0, 1)`. It trains only with at least 5 examples
  that include two different answers.

## 5. Labels and the blend `w`

**In words.** Both ways of scoring have a say. The final score mixes them with a dial called **w**, the share
of trust given to learning:

`final = (1 − w) × profile + w × learned`

At `w = 0` only the profile counts; at `w = 0.7` learning dominates. The app picks `w` by testing on your own
labels. If the learned model doesn't beat the plain profile, **w stays at 0%** and learning gets no say. It
only counts when it earns it. (How the test works is in section 7.)

**The math.** `w` is recomputed every time the ranking refreshes (after any rating, label, profile edit or
cutoff change):

- **Fewer than 6 hand labels, or only one kind of label.** There is nothing to test against. If there are at
  least 5 ratings with two different answers, the ranker still trains, with a capped ramp
  `w = min(0.3, n / (n + 30))` (5 → 0.14, 10 → 0.25, capped at 0.30). Settings says "capped until your hand
  labels can validate it".
- **6 or more hand labels with at least two kinds.** `w` is **validated** by cross-validation (section 7)
  and is called "chosen by cross-validation on your hand labels".

## 6. From score to Read, Skim or Skip

**In words.** The final score is compared with two cutoffs. **Read** is also capped by the reading time you
set, and your own labels always override the app's guess. Each card shows a one-line reason built only from
words that really appear in the paper, plus separate **worth-it signals** that never change the score.

**The math.**

- `final ≥ 0.62` → **Read**; `final ≥ 0.40` → **Skim**; otherwise **Skip**. Cutoffs are adjustable in Settings.
- **Adaptive cutoffs.** If scores run low (for example a profile built from a few liked papers), cutoffs only
  ever move *down*, so Read isn't empty: Read becomes the top 6 papers (never below the Skim line) and Skim
  about the top 10% (never below 0.20).
- **Weekly cap.** `Read slots = floor(hours × 60 / 30)`, at 30 minutes per paper. Extra Read papers drop to
  Skim with a note.
- **Papers you have already decided** leave the lists and move to **Reviewed**.
- **Worth-it signals** are pattern rules on the abstract (released code or data, trial, meta-analysis,
  preregistered, large sample, hype wording) plus OpenAlex metadata (peer-reviewed, open access, citation
  level), in `triage/quality_rules.json`. They are cues, not a quality grade, and they never enter the
  formulas.

## 7. Cross-validation: checking it fairly

**In words.** If a model trains on your labels and we then grade it on those same papers, it looks great
because it saw the answers, like grading a student on questions they already studied. So the app **hides
papers, trains on the rest, and scores the hidden ones**, several times, so every paper gets an honest score.

**The math.** **5-fold cross-validation:**

1. Take your hand-labelled papers (say 200). Sort them by a hash of their id, so it is **deterministic**, and
   deal them into 5 piles, **stratified** so each pile has a similar mix of Read, Skim and Skip. With fewer
   than 5 labels, the number of piles drops to match.
2. For each pile: **hide it**, removing its labels *and any ratings on those papers*. Fit a fresh ridge model
   on the rest. Score the hidden papers, computing both `prior` and `learned`.
3. After 5 rounds, **every labelled paper has one out-of-fold score** from a model that never saw its label.

Every honest number in the app is measured on those pooled scores. The profile score uses no labels, so it is
honest by construction. Cross-validation is what makes the *learned* score honest.

**Choosing `w`.** With the pooled scores, try `w ∈ {0, 0.15, 0.3, 0.5, 0.7}`. For each, compute the average
precision (section 8) of `(1−w)·prior + w·learned`, where "good" means Read (or Read + Skim if you have no
Read labels). Start at `w = 0`. A larger `w` replaces the best so far only if it beats `AP(w=0) + 0.01`
**and** the current best. With only a few labels a learned model can chase noise and do **worse** than a
sensible default, so the 0.01 margin means learning counts only when it clearly helps.

In our data: the three manual profiles (RAG, Climate, Cancer) end up at `w = 0%`, because the hand-set profile
was already as good as it gets. The demo Materials profile, which has a hidden interest its description
omits, earns `w = 70%`.

## 8. The measurements

**In words.** The Insights page uses your labels as the answer key and reports how high your relevant papers
rank, compared with your profile alone and with a random order.

**What it is measured against.** Only your **hand labels** (Read / Skim / Skip), never your thumbs. The code
passes just the hand labels to the evaluation; thumbs travel in a separate list that is used for training but
never for grading. Example: you hand-labelled 100 papers (8 Read, 20 Skim, 72 Skip) and pressed 👍/👎 on about
30 others.

1. The answer key is those 100 papers and their 100 labels, nothing else.
2. Cross-validation (section 7) splits them into 5 piles of 20. For each pile it trains on the other 80 labels
   **plus all your thumbs** (except thumbs on papers in the hidden pile) and scores the hidden 20.
3. After 5 rounds each of the 100 papers has an honest score. The metrics below compare those scores with the
   labels: did the 8 Read papers rank near the top, was the order Read > Skim > Skip, and how often was the
   label right at the cutoffs?
4. Baselines and random order are scored against the same 100 labels, so the comparison is fair.

Thumbs help the model *learn*; they never decide whether it was *right*. The 500 manual labels and the demo
accounts' labels are all hand-label-style data, so every number in the report is measured against them.

**The math.** Everything is computed on the pooled out-of-fold scores. "Good" means Read, or Read + Skim if
chosen.

**Random order.** Shuffle the papers like a deck of cards and read down the pile: what pure luck gives you.
Good papers are spread evenly, so reaching 80% of them takes about 80% of the pile (**0.8 × n**: 160 of 200,
80 of 100). If a fraction `g` of papers are good, a random pick is good with probability `g`, so **random AP ≈ g**
(6 of 200 gives 0.03). A ranking that can't clearly beat random order is no better than guessing.

- **Average precision (AP).** Rank best-first and walk down. Each time you hit a good paper, record the share
  of papers so far that were good, then average those values. **1.00** means every good paper outranks every
  other paper; **≈ g** is random. *Example:* good papers at ranks 1, 2 and 4 give `(1/1 + 2/2 + 3/4) / 3 = 0.92`.
- **Papers to screen for 80%.** The rank at which the share of good papers found first reaches 80%. For RAG it
  is **5**, versus **160** at random.
- **NDCG@10.** Looks at the top 10. Gains are Read = 3, Skim = 1, Skip = 0, each discounted by
  `1 / log₂(position + 1)` and divided by the best possible total.
- **Spearman correlation.** Agreement between the score ordering and Read > Skim > Skip, using ranks with ties
  averaged.
- **Time saved.** `(random papers needed − our papers needed) × 2 minutes` per abstract. For RAG:
  `(160 − 5) × 2 ≈ 5.2 hours`. This is a label-based estimate, not a measured user result.
- **Label metrics at the current cutoffs.** A 3×3 confusion matrix, per-class F1, macro-F1, accuracy and a
  count of "severe" mistakes (Read predicted as Skip, or the reverse). **Best cutoffs** come from a grid
  search over 0.00–1.00 in steps of 0.02, maximising macro-F1 and breaking ties by fewer severe mistakes.
- **Baselines.** The same metrics are computed for profile only, learned only, semantic similarity only
  (feature 1), keyword matches only (feature 4), and random, to show what each ingredient adds.

## 9. Does it get better as I label? (the learning curve)

**In words.** The Insights learning curve answers: "if I label more papers, do the recommendations improve?"

**The math.** It repeats the whole pipeline at training sizes `k = 5, 10, 15, 20, 30, 45 …` (every 5 up to 20,
then about +50% a step, ending at the full pool). For each `k`:

1. Use the same stratified 5-fold split. For each fold, draw `k` labels from the training part, **stratified by
   label** so early points can learn at all, and repeat 3 times.
2. Build the recommender as the app would from only those `k` labels, including **its own choice of `w`**,
   decided using only those `k` labels.
3. Score the held-out fold, which it has never seen.
4. Report AP for the recommender, for the profile alone, and for random order. The band is the spread over the
   repeats.

It needs at least 20 labels with two kinds. "Clearly ahead" means at least 2 AP points above the profile
alone, staying ahead from that size onward. With fewer than 10 good papers or fewer than 60 labels the curve
is marked **rough**. On the Materials demo it rises from about 52% to 63% and passes the profile alone at about
45 labels. On the three manual profiles it stays flat, because `w` stays 0.

## 10. Honest limits

- With only **3–6 Read papers** per profile, one paper moves AP by several points, so small gaps between
  methods are noise.
- The 500 manual labels began as model suggestions that a person reviewed. That can flatter text-similarity
  rankers.
- The profile weights and keywords were set after seeing the data. That is human tuning, not a leak in the
  cross-validation, but it should be disclosed.
- Strong AP on a tiny number of Read papers shows the ranking is sensible on our data. It does not prove time
  saved for real users, which needs a trial.

## Cheat sheet

| Term | Meaning |
|---|---|
| **Feature** | one of seven 0–1 checks on a paper |
| **Profile score** | hand-weighted sum of the checks (no labels needed) |
| **Learned score** | ridge regression that re-weights the same checks from your labels |
| **`w`** | share of trust in learning; `final = (1−w)·profile + w·learned` |
| **Cross-validation** | hide papers, train on the rest, score the hidden ones, repeat; every paper gets an honest score |
| **Random order** | shuffled papers: 80% of the good ones takes about 80% of the pile, and AP ≈ the share of good papers |
| **AP** | quality of the ranking; 1.0 is perfect, random is about the base rate |
| **Papers to 80%** | how far down the list you must read to find 80% of the good papers |

See the [README](../README.md) for how to run it and where each piece of data lives, and
[`docs/evaluation/report.md`](evaluation/report.md) for the measured results.
