# How Paper Triage works

Part 1 explains the flow in plain words. [Part 2](#part-2-the-math-in-detail) gives every formula, the cross-validation, and the metrics.

## Part 1: the flow in plain words

Too many research papers come out for anyone to read them all. Paper Triage sorts them into
**Read**, **Skim** and **Skip**, so you spend your time on the right ones. It does this in three
moves: it **guesses** how relevant each paper is to you, you **tell it the truth** on a few papers,
and it **adjusts its guesses** toward your answers, but only when that really makes them better.

### 1. Where the papers come from

The app ships with a ready-made pile of about 1,900 recent papers from every field, collected from
OpenAlex. Each paper already carries a "meaning fingerprint" (a MiniLM embedding) computed in advance,
so comparing meanings is instant. **Load more** pulls fresh papers live from OpenAlex, Europe PMC and
Crossref, and removes duplicates.

If the optional server is running you sign in first (or choose "Continue without an account"). Your
profiles, ratings and labels are saved to your account, or just in your browser if you skip signing in.
Everything below runs in your browser either way.

### 2. Telling it what you care about

You pick your fields and, if you like, describe your research in a sentence and add keywords. That is
your **profile**. You can keep several, for example one per research topic.

Then you rate a few papers: the app shows 20 and you press 👍 or 👎 on at least 5. That is all it needs to
start sorting.

### 3. Scoring: the app's guess

Every paper gets a number from 0 to 1. Higher means "more relevant to you". Say your profile is
"RAG evaluation" and the paper is *"Measuring faithfulness in retrieval-augmented generation"*. It earns
points by answering a few simple questions:

| Question | Example answer |
|---|---|
| Does the paper's meaning match my description? | very close |
| Does it match my current focus? | close |
| Do my keywords appear in it? | yes, in the title |
| Is it recent? | yes |
| Does it mention a topic I excluded? | no |
| Is it similar to papers I already liked? | yes |

These are added up with fixed weights (meaning counts most, keywords next) into the **profile score**,
say 0.82. This part needs no training and works from your very first visit.

### 4. Labels: the truth you give it

You can tell the app the truth in two ways:

- **Quick ratings.** In the feed, press 👍 or 👎 on a paper.
- **Labels.** In the Label tab, mark a paper **Read**, **Skim** or **Skip**. The app's prediction is hidden
  while you do this, so it cannot sway you.

Both say the same thing: "for this profile, this paper is good or not". Papers you have already decided
on leave the Read/Skim/Skip lists and move to **Reviewed**.

### 5. Learning from your labels

Once you have given enough labels, the app trains a small model on them. It learns patterns such as "this
person's relevant papers usually have a close meaning match and a keyword in the title", and gives every
paper a second number, the **learned score**. The final score mixes the two:

> final score = (1 − w) × profile score + w × learned score

The app picks **w** by testing on your own labels. If the learned model does not beat the plain profile score,
**w stays at 0%** and your ranking simply uses the profile. Learning only gets a say when it earns it.

### 6. From score to Read, Skim or Skip

The final score is compared with two cutoffs, for example:

- 0.62 or higher: **Read**
- 0.40 to 0.62: **Skim**
- below 0.40: **Skip**

**Read** is also capped by the reading time you set (about 30 minutes per paper), and extra papers drop to
Skim. Your own labels always override the app's guess. Each card shows a one-line reason built only from
words that really appear in the paper, plus separate **trust signals** (peer-reviewed or not, code or data
released, study type, citation level, cautions). Those signals never change the relevance score.

### 7. Checking that it works

Your labels also work as the answer key. To test fairly, the app hides each paper's label, ranks the papers
without it, and compares the ranking with what you said. The **Insights** page shows the result: how high
your relevant papers rank compared with your profile alone and with a random order, and whether
recommendations improve as you label more. If your written profile already matches what you pick, learning
adds nothing, and the page says so instead of pretending.

### In one line

**Score = the app's guess. Labels = your answers. Learning = nudging the guess toward your answers, but only
when it really helps.**

---

## Part 2: the math in detail

Everything here is implemented in [`web/js/engine.js`](../web/js/engine.js) (scoring, training, validation) and [`web/js/evaluate.js`](../web/js/evaluate.js) (metrics and the learning curve). A test checks the ridge regression against scikit-learn.

The system answers two questions: **how do we score a paper?** (sections 2.1–2.4) and **how do we know the scoring is any good?** (sections 2.5–2.8).

---

### 2.0 The inputs

| Input | Where it comes from | Used for |
|---|---|---|
| **Profile text**: description, focus sentence, keywords, excluded topics | typed by the user | the seven checks |
| **Papers**: title + abstract + metadata | the catalog (pre-fetched OpenAlex) or live search | what gets scored |
| **Ratings**: 👍 / 👎, Save, Open, Dismiss, corrections | clicks in the feed | training examples |
| **Hand labels**: Read / Skim / Skip with the prediction hidden | the Label tab | training **and** the answer key for validation |

Hand labels are the important distinction. Ratings teach the ranker. Hand labels both teach it and are what we test against.

---

### 2.1 Turning text into numbers (embeddings)

An embedding model turns a piece of text into a list of 384 numbers, so that texts with similar **meaning** end up close together. We use **all-MiniLM-L6-v2**, off the shelf and unchanged. If it hasn't loaded yet, a TF-IDF fallback (word counts weighted by how rare each word is) ranks in the meantime.

- Every vector is scaled to length 1, so the dot product of two vectors is their **cosine similarity**, from −1 to 1. Higher means closer in meaning.
- Catalog paper vectors are precomputed at build time. Only your own profile text is embedded in the browser.
- **Profile vector.** Your description, keywords, focus and any seed papers are combined into one "main" vector as a weighted sum, then re-scaled to length 1. Weights: description 1.0, keywords 1.0, focus 1.5, seed papers 1.5.
- **Calibration.** Raw cosines bunch up in a narrow band, so each is stretched onto 0–1:

  `calibrated = clip( (cosine − lo) / (hi − lo), 0, 1 )`

  For MiniLM, `lo = 0.15` and `hi = 0.50` for paper-versus-profile comparisons. A cosine of 0.15 or less counts as 0, and 0.50 or more counts as 1. Short texts (a keyword or focus line) use 0.12 and 0.45.

---

### 2.2 The seven checks (features)

For each paper, seven numbers between 0 and 1 (the sixth is 0 unless an excluded topic is clearly present):

| # | Feature | How it's computed |
|---|---|---|
| 1 | **semantic** | calibrated similarity: paper vs. main profile vector |
| 2 | **focus** | calibrated similarity: paper vs. focus sentence (falls back to #1 if none) |
| 3 | **keyword_sem** | calibrated similarity to the **best-matching single keyword** |
| 4 | **keyword_lex** | exact phrase hits: `min(1, (title hits + 0.6 × abstract-only hits) / min(#keywords, 3))` |
| 5 | **recency** | `exp(−age_in_days / 365)`: 1 for today, about 0.37 after a year (0.5 if no date) |
| 6 | **avoid** | for excluded topics: `max( clip((similarity − 0.5) × 2), 0.8 if the term appears literally )`, so it fires only when clearly present |
| 7 | **feedback** | similarity to papers you liked minus similarity to papers you disliked |

Check 7 is `+calibrated(top-3 mean similarity to liked papers) − calibrated(top-3 mean similarity to disliked papers)`. It is **leave-self-out**: a paper never matches its own rating.

---

### 2.3 Two ways to combine the seven checks

#### Profile score (hand-set weights, needs no labels)

`prior = clip( 0.40·semantic + 0.15·focus + 0.15·keyword_sem + 0.25·keyword_lex + 0.05·recency − 0.40·avoid + 0.25·feedback , 0, 1 )`

We chose these weights by hand so the app works on the first visit. Everyone gets the same weights.

#### Learned score (weights fitted to *you*)

A **weighted ridge regression** looks at your labelled and rated papers and finds the weights that best predict your answers.

- **Targets:** Read = 1, Skim = 0.5, Skip = 0. Ratings map to similar targets: 👍 = 1, 👎 = 0, Save = 1, Open = 0.75, Dismiss = 0.15.
- **Example weights** (how much each example counts): hand label 1, rating 1, correction 1.5, seed paper 2, Open about 0.3, Dismiss 0.5. Priority when a paper has several signals: seed > hand label > correction > explicit rating > implicit click.
- **The fit.** Find coefficients `β` and an intercept that minimise

  `Σ wᵢ · (yᵢ − intercept − xᵢ·β)²  +  α · Σ βⱼ²`

  with `α = 1`. The second term (the "ridge" penalty) keeps the coefficients small, so a few noisy labels can't produce wild weights. The intercept is not penalised. We solve it with a short Gaussian-elimination routine, with no library, and check it against scikit-learn to 1e-9.
- **Prediction:** `learned = clip(intercept + Σ βⱼ·featureⱼ, 0, 1)`.
- It trains only with **at least 5 examples that include at least two different answers**. Otherwise there is nothing to learn from.

It learns *how much you care about each check*. Maybe you ignore recency but always care about exact keywords.

---

### 2.4 The blend `w`, and the final label

`final = (1 − w) × prior + w × learned`

`w` (from 0 to 1) is how much we trust learning. At 0, only the profile counts. At 0.7, learning dominates. It is recomputed whenever the ranking refreshes (see section 2.6).

**Turning the score into a label:**

- `final ≥ 0.62` → **Read**; `final ≥ 0.40` → **Skim**; otherwise **Skip**. Cutoffs are adjustable in Settings.
- **Adaptive cutoffs.** If scores run low (a profile built from a few liked papers), the cutoffs only ever move *down*, so Read isn't empty: Read becomes the top 6 papers (but never below the Skim line), and Skim about the top 10% (but never below 0.20).
- **Weekly cap.** `Read slots = floor(hours × 60 / 30)`, at 30 minutes per paper. Extra Read papers drop to Skim with a note.
- **Your own label wins.** If you hand-labelled or corrected a paper, your label overrides the model.

---

### 2.5 Cross-validation: scoring papers the model has never seen

**The problem.** If a model trains on your labels and we then grade it on those same papers, it looks great because it saw the answers. We want to know how it does on *new* papers.

**5-fold cross-validation** hides papers, trains on the rest, scores the hidden ones, and repeats:

1. Take all your hand-labelled papers (say 200). Sort them by a hash of their id (so it is **deterministic**) and deal them into 5 piles, **stratified** so each pile has a similar mix of Read, Skim and Skip. With fewer than 5 labels, the number of piles drops to match.
2. For each pile: **hide it**. Remove its labels *and any ratings on those papers*. Fit a fresh ridge model on the rest. Then score the hidden papers, computing both `prior` and `learned`.
3. After 5 rounds, **every labelled paper has exactly one out-of-fold score** from a model that never saw its label.

Every honest number in the app (AP, NDCG, papers-to-80%, time saved) is measured on those pooled out-of-fold scores. The profile score uses no labels, so it is honest by construction. Cross-validation is what makes the *learned* score honest.

---

### 2.6 How `w` is chosen in the app

The rule: **learning must earn its weight.**

Whenever the ranking refreshes (`rank()` calls `selectBlend()`):

**Fewer than 6 hand labels, or only one kind of label.** There is nothing to validate. If at least 5 ratings with two different answers exist, the ranker still trains, with a capped ramp:

`w = min(0.3, n / (n + 30))` (n = number of examples; 5 → 0.14, 10 → 0.25, capped at 0.30)

Settings shows this as "capped until your hand labels can validate it".

**6 or more hand labels with at least two kinds.** Cross-validation runs, and `w` is **validated**:

1. Compute the pooled out-of-fold `prior` and `learned` scores.
2. For each candidate `w ∈ {0, 0.15, 0.3, 0.5, 0.7}`, compute the average precision of `(1−w)·prior + w·learned`. "Good" means Read, or Read + Skim if you have no Read labels.
3. Start at `w = 0`. A larger `w` replaces the current best only if its AP beats `AP(w=0) + 0.01` **and** beats the current best.

This is the safeguard. With only a handful of labels, a learned model can chase noise and do **worse** than a sensible default. The 0.01 margin means learning counts only when it clearly helps.

**What happened in our data:**

- Three manual profiles (RAG, Climate, Cancer): `w = 0%`. The hand-set profile was already as good as it gets.
- Demo Materials profile (a hidden interest the description omits): `w = 70%`.

---

### 2.7 The metrics

All computed on the pooled out-of-fold scores. "Good" means Read, or Read + Skim if chosen.

#### Random order (the baseline)

Shuffle the papers like a deck of cards and read down the pile. That is **random order**: what you'd get from pure luck.

- Good papers are spread evenly through a shuffled pile, so reaching 80% of them takes about 80% of the pile: **0.8 × n** papers (160 of 200, 80 of 100).
- If a fraction `g` of papers are good, a random pick is good with probability `g`, so **random AP ≈ g** (6 of 200 → 0.03).

A ranking that can't clearly beat random order is no better than guessing.

#### Average precision (AP)

Rank papers best-first and walk down. Every time you hit a good paper, record the share of papers so far that were good. Average those values over all good papers.

- **1.00** means every good paper outranks every other paper. **≈ g** is random.
- *Example:* 3 good papers at ranks 1, 2 and 4 gives `(1/1 + 2/2 + 3/4) / 3 = 0.92`.

#### Papers to screen for 80% (and 50%, and first hit)

The rank at which the cumulative share of good papers found first reaches 80%. For RAG it is **5**, versus **160** for random order.

#### NDCG@10

Looks at the top 10. Gains are Read = 3, Skim = 1, Skip = 0 (`2^grade − 1`), each discounted by `1 / log₂(position + 1)`, then divided by the best possible total. 1.0 means the top 10 is in perfect order.

#### Spearman correlation

How well the score ordering agrees with the Read > Skim > Skip ordering across all papers, using ranks with ties averaged. It is the standard correlation of the two rank lists.

#### Time saved

`(random-order papers needed − our papers needed) × 2 minutes`, using 2 minutes to screen one abstract. For RAG: `(160 − 5) × 2 ≈ 5.2 hours`. This is a label-based estimate, not a measured user result.

#### Label metrics at the current cutoffs

A 3×3 confusion matrix (true label vs. predicted), per-class F1, macro-F1 (the average over the classes that exist), accuracy, and a count of "severe" mistakes (Read predicted as Skip, or Skip as Read). **Best cutoffs** are found by a grid search over 0.00–1.00 in steps of 0.02, maximising macro-F1 and breaking ties by fewer severe mistakes.

#### Baselines

The same metrics are computed for weaker rankers, to show what each ingredient adds:

| Baseline | Scores each paper by |
|---|---|
| Profile only | `prior` |
| Learned only | `learned` |
| Semantic only | feature 1 alone |
| Keyword only | feature 4 alone |
| Random | a shuffle |

---

### 2.8 The learning curve ("does it get better as I label?")

The Insights learning curve repeats the whole pipeline at several training sizes `k = 5, 10, 15, 20, 30, 45 …` (every 5 up to 20, then about +50% a step, ending at the full pool).

For each `k`:

1. Use the same stratified 5-fold split. For each fold, draw `k` labels from the training part, **stratified by label** so early points can learn at all, and repeat 3 times.
2. Build the recommender as the app would from only those `k` labels. That includes **its own `selectBlend()`**, so `w` is decided using only those `k` labels.
3. Score the held-out fold, which the model has never seen.
4. Report AP for the recommender, for the profile alone, and for random order. The shaded band is the spread over the repeats.

Reading it:

- It needs at least 20 labels with two kinds.
- "Clearly ahead" means at least 2 AP points above the profile alone, staying ahead from that size onward.
- With fewer than 10 good papers or fewer than 60 labels, the app labels the curve **rough**, because one paper moves it a lot.
- When there are too few Read labels, Read + Skim count as good.

On the Materials demo profile, the curve rises from about 52% to 63% and passes the profile alone at about 45 labels. On the three manual profiles it stays flat, because `w` stays 0.

---

### 2.9 What isn't math, on purpose

- **Worth-it signals** (peer review, code or data released, study design, hype cautions, citation impact) are pattern rules on the abstract plus OpenAlex metadata, in `triage/quality_rules.json`. They are shown beside the score and **never enter the formulas above**.
- **Reasons** ("'hallucination' in the title") are built only from words actually found in the paper, so they can't invent overlap.

---

### 2.10 Honest limits of the numbers

- With only **3–6 Read papers** per profile, one paper moves AP by several points, so small gaps between methods are noise.
- The 500 manual labels began as model suggestions that a person reviewed. That can flatter text-similarity rankers.
- We set the profile weights and keywords after seeing the data. That is human tuning, not a leak in the cross-validation, but it should be disclosed.
- Strong AP on a tiny number of Read papers shows the ranking is sensible on our data. It does not prove time saved for real users, which needs a trial (criterion C7).

---

### Cheat sheet

| Term | Meaning |
|---|---|
| **Feature** | one of seven 0–1 checks on a paper |
| **Profile score** | hand-weighted sum of the checks (no labels needed) |
| **Learned score** | ridge regression that re-weights the checks from your labels |
| **`w`** | share of trust in learning; the final score is `(1−w)·profile + w·learned` |
| **Cross-validation** | hide papers, train on the rest, score the hidden ones, repeat; every paper gets an honest score |
| **Random order** | shuffled papers: reaching 80% of the good ones takes about 80% of the pile, and AP ≈ the share of good papers |
| **AP** | quality of the ranking; 1.0 is perfect, random is about the base rate |
| **Papers to 80%** | how far down the list you must read to find 80% of the good papers |

---

See the [README](../README.md) for how to run it and where each piece of data lives, and
[`docs/evaluation/report.md`](evaluation/report.md) for the measured results.
