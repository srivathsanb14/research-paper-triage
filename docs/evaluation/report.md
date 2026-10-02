# Evaluation report

Generated 2026-10-02 by `node scripts/evaluate_web.mjs` from `data/labels/manual/IM-human-verified-500.jsonl`, `data/labels/synthetic/demo_rag.jsonl`.
Every number is 5-fold cross-validated: no paper is scored by a model trained on its own label. “Good” = Read (or Read + Skim where noted).

## Human labels (manual, and model-proposed labels checked by a person)

### RAG & LLM evaluation · MiniLM · good = Read

200 labels (Read 6, Skim 33, Skip 161), labellers: IM. 200 were model-proposed and checked by a person, who changed 0.

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.03; papers to screen for 80% of good ones in random order: 160.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.97 | 1.00 | 0.64 | 100% | 5 |
| Profile only (no learning) | 0.97 | 1.00 | 0.64 | 100% | 5 |
| Learned from feedback only | 0.99 | 0.93 | 0.61 | 100% | 6 |
| Semantic similarity only | 0.93 | 0.93 | 0.58 | 100% | 5 |
| Keyword matches only | 0.91 | 0.78 | 0.56 | 83% | 8 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.77, accuracy 0.87, 0 Read↔Skip errors.

### RAG & LLM evaluation · MiniLM · good = Read or Skim

200 labels (Read 6, Skim 33, Skip 161), labellers: IM. 200 were model-proposed and checked by a person, who changed 0.

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.20; papers to screen for 80% of good ones in random order: 160.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.97 | 0.78 | 0.64 | 90% | 48 |
| Profile only (no learning) | 0.97 | 0.78 | 0.64 | 90% | 48 |
| Learned from feedback only | 0.99 | 0.80 | 0.61 | 100% | 53 |
| Semantic similarity only | 0.93 | 0.68 | 0.58 | 80% | 65 |
| Keyword matches only | 0.91 | 0.75 | 0.56 | 100% | 70 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.77, accuracy 0.87, 0 Read↔Skip errors.

### RAG & LLM evaluation · TF-IDF · good = Read

200 labels (Read 6, Skim 33, Skip 161), labellers: IM. 200 were model-proposed and checked by a person, who changed 0.

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.03; papers to screen for 80% of good ones in random order: 160.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.97 | 0.83 | 0.62 | 100% | 7 |
| Profile only (no learning) | 0.97 | 0.83 | 0.62 | 100% | 7 |
| Learned from feedback only | 0.97 | 0.83 | 0.58 | 100% | 7 |
| Semantic similarity only | 0.91 | 0.80 | 0.60 | 83% | 10 |
| Keyword matches only | 0.84 | 0.73 | 0.57 | 67% | 11 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.61, accuracy 0.84, 0 Read↔Skip errors.

### RAG & LLM evaluation · TF-IDF · good = Read or Skim

200 labels (Read 6, Skim 33, Skip 161), labellers: IM. 200 were model-proposed and checked by a person, who changed 0.

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.20; papers to screen for 80% of good ones in random order: 160.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.97 | 0.82 | 0.62 | 100% | 51 |
| Profile only (no learning) | 0.97 | 0.82 | 0.62 | 100% | 51 |
| Learned from feedback only | 0.97 | 0.79 | 0.58 | 100% | 55 |
| Semantic similarity only | 0.91 | 0.77 | 0.60 | 100% | 53 |
| Keyword matches only | 0.84 | 0.75 | 0.57 | 100% | 66 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.61, accuracy 0.84, 0 Read↔Skip errors.

### Climate adaptation · MiniLM · good = Read

200 labels (Read 3, Skim 24, Skip 173), labellers: IM. 200 were model-proposed and checked by a person, who changed 0.

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.01; papers to screen for 80% of good ones in random order: 160.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.90 | 0.83 | 0.50 | 100% | 6 |
| Profile only (no learning) | 0.90 | 0.83 | 0.50 | 100% | 6 |
| Learned from feedback only | 0.78 | 0.64 | 0.47 | 67% | 12 |
| Semantic similarity only | 0.88 | 0.67 | 0.54 | 100% | 9 |
| Keyword matches only | 0.76 | 0.71 | 0.52 | 67% | 25 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.60, accuracy 0.88, 0 Read↔Skip errors.

### Climate adaptation · MiniLM · good = Read or Skim

200 labels (Read 3, Skim 24, Skip 173), labellers: IM. 200 were model-proposed and checked by a person, who changed 0.

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.14; papers to screen for 80% of good ones in random order: 160.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.90 | 0.62 | 0.50 | 80% | 58 |
| Profile only (no learning) | 0.90 | 0.62 | 0.50 | 80% | 58 |
| Learned from feedback only | 0.78 | 0.62 | 0.47 | 80% | 59 |
| Semantic similarity only | 0.88 | 0.64 | 0.54 | 90% | 54 |
| Keyword matches only | 0.76 | 0.57 | 0.52 | 70% | 52 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.60, accuracy 0.88, 0 Read↔Skip errors.

### Climate adaptation · TF-IDF · good = Read

200 labels (Read 3, Skim 24, Skip 173), labellers: IM. 200 were model-proposed and checked by a person, who changed 0.

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.01; papers to screen for 80% of good ones in random order: 160.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.92 | 0.78 | 0.45 | 100% | 9 |
| Profile only (no learning) | 0.92 | 0.78 | 0.45 | 100% | 9 |
| Learned from feedback only | 0.80 | 0.59 | 0.42 | 67% | 11 |
| Semantic similarity only | 0.85 | 0.67 | 0.55 | 100% | 9 |
| Keyword matches only | 0.76 | 0.71 | 0.53 | 67% | 22 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.65, accuracy 0.89, 1 Read↔Skip errors.

### Climate adaptation · TF-IDF · good = Read or Skim

200 labels (Read 3, Skim 24, Skip 173), labellers: IM. 200 were model-proposed and checked by a person, who changed 0.

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.14; papers to screen for 80% of good ones in random order: 160.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.92 | 0.64 | 0.45 | 90% | 69 |
| Profile only (no learning) | 0.92 | 0.64 | 0.45 | 90% | 69 |
| Learned from feedback only | 0.80 | 0.66 | 0.42 | 90% | 52 |
| Semantic similarity only | 0.85 | 0.69 | 0.55 | 80% | 40 |
| Keyword matches only | 0.76 | 0.58 | 0.53 | 70% | 41 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.65, accuracy 0.89, 1 Read↔Skip errors.

### Cancer immunotherapy · MiniLM · good = Read

100 labels (Read 3, Skim 15, Skip 82), labellers: IM. 100 were model-proposed and checked by a person, who changed 0.

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.03; papers to screen for 80% of good ones in random order: 80.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.97 | 1.00 | 0.74 | 100% | 3 |
| Profile only (no learning) | 0.97 | 1.00 | 0.74 | 100% | 3 |
| Learned from feedback only | 0.98 | 0.92 | 0.62 | 100% | 4 |
| Semantic similarity only | 0.98 | 0.92 | 0.67 | 100% | 4 |
| Keyword matches only | 0.89 | 0.87 | 0.59 | 100% | 5 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.79, accuracy 0.89, 0 Read↔Skip errors.

### Cancer immunotherapy · MiniLM · good = Read or Skim

100 labels (Read 3, Skim 15, Skip 82), labellers: IM. 100 were model-proposed and checked by a person, who changed 0.

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.18; papers to screen for 80% of good ones in random order: 80.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.97 | 0.90 | 0.74 | 90% | 20 |
| Profile only (no learning) | 0.97 | 0.90 | 0.74 | 90% | 20 |
| Learned from feedback only | 0.98 | 0.87 | 0.62 | 100% | 21 |
| Semantic similarity only | 0.98 | 0.86 | 0.67 | 100% | 19 |
| Keyword matches only | 0.89 | 0.72 | 0.59 | 80% | 36 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.79, accuracy 0.89, 0 Read↔Skip errors.

### Cancer immunotherapy · TF-IDF · good = Read

100 labels (Read 3, Skim 15, Skip 82), labellers: IM. 100 were model-proposed and checked by a person, who changed 0.

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.03; papers to screen for 80% of good ones in random order: 80.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.96 | 1.00 | 0.54 | 100% | 3 |
| Profile only (no learning) | 0.96 | 1.00 | 0.54 | 100% | 3 |
| Learned from feedback only | 0.90 | 1.00 | 0.47 | 100% | 3 |
| Semantic similarity only | 0.93 | 1.00 | 0.52 | 100% | 3 |
| Keyword matches only | 0.89 | 0.87 | 0.50 | 100% | 5 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.61, accuracy 0.85, 0 Read↔Skip errors.

### Cancer immunotherapy · TF-IDF · good = Read or Skim

100 labels (Read 3, Skim 15, Skip 82), labellers: IM. 100 were model-proposed and checked by a person, who changed 0.

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.18; papers to screen for 80% of good ones in random order: 80.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.96 | 0.73 | 0.54 | 90% | 35 |
| Profile only (no learning) | 0.96 | 0.73 | 0.54 | 90% | 35 |
| Learned from feedback only | 0.90 | 0.71 | 0.47 | 70% | 34 |
| Semantic similarity only | 0.93 | 0.72 | 0.52 | 80% | 35 |
| Keyword matches only | 0.89 | 0.66 | 0.50 | 80% | 37 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.61, accuracy 0.85, 0 Read↔Skip errors.

## Synthetic labels (pipeline check only, not evidence of quality)

### Demo: RAG research (simulated) · MiniLM · good = Read

529 labels (Read 9, Skim 98, Skip 422), labellers: rule:demo_rag.

Learning weight chosen by cross-validation: **70%**. Random-order AP: 0.02; papers to screen for 80% of good ones in random order: 424.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 70%) | 0.68 | 0.50 | 0.42 | 44% | 76 |
| Profile only (no learning) | 0.68 | 0.49 | 0.49 | 44% | 84 |
| Learned from feedback only | 0.68 | 0.51 | 0.42 | 44% | 72 |
| Semantic similarity only | 0.69 | 0.51 | 0.48 | 44% | 67 |
| Keyword matches only | 0.76 | 0.56 | 0.48 | 56% | 69 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.51, accuracy 0.81, 4 Read↔Skip errors.

### Demo: RAG research (simulated) · MiniLM · good = Read or Skim

529 labels (Read 9, Skim 98, Skip 422), labellers: rule:demo_rag.

Learning weight chosen by cross-validation: **70%**. Random-order AP: 0.20; papers to screen for 80% of good ones in random order: 424.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 70%) | 0.68 | 0.60 | 0.42 | 80% | 238 |
| Profile only (no learning) | 0.68 | 0.59 | 0.49 | 80% | 304 |
| Learned from feedback only | 0.68 | 0.59 | 0.42 | 80% | 238 |
| Semantic similarity only | 0.69 | 0.59 | 0.48 | 80% | 258 |
| Keyword matches only | 0.76 | 0.60 | 0.48 | 90% | 258 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.51, accuracy 0.81, 4 Read↔Skip errors.

### Demo: RAG research (simulated) · TF-IDF · good = Read

529 labels (Read 9, Skim 98, Skip 422), labellers: rule:demo_rag.

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.02; papers to screen for 80% of good ones in random order: 424.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.74 | 0.54 | 0.44 | 44% | 156 |
| Profile only (no learning) | 0.74 | 0.54 | 0.44 | 44% | 156 |
| Learned from feedback only | 0.74 | 0.53 | 0.36 | 44% | 149 |
| Semantic similarity only | 0.79 | 0.55 | 0.49 | 56% | 267 |
| Keyword matches only | 0.79 | 0.57 | 0.49 | 56% | 267 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.48, accuracy 0.81, 5 Read↔Skip errors.

### Demo: RAG research (simulated) · TF-IDF · good = Read or Skim

529 labels (Read 9, Skim 98, Skip 422), labellers: rule:demo_rag.

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.20; papers to screen for 80% of good ones in random order: 424.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.74 | 0.59 | 0.44 | 100% | 307 |
| Profile only (no learning) | 0.74 | 0.59 | 0.44 | 100% | 307 |
| Learned from feedback only | 0.74 | 0.57 | 0.36 | 100% | 301 |
| Semantic similarity only | 0.79 | 0.59 | 0.49 | 100% | 279 |
| Keyword matches only | 0.79 | 0.58 | 0.49 | 100% | 279 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.48, accuracy 0.81, 5 Read↔Skip errors.

