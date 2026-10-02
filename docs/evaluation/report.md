# Evaluation report

Generated 2026-10-02 by `node scripts/evaluate_web.mjs` from `data/labels/labels.jsonl` and `data/labels/profiles.json`.
Every number is 5-fold cross-validated: no paper is scored by a model trained on its own label. “Good” = Read (or Read + Skim where noted).
Origin: `reviewed` = a model proposed the label and a person accepted or changed it; `rule` = a transparent keyword rule (demo and pipeline check, not evidence of quality).

## Labels reviewed by a person

### Cancer immunotherapy · MiniLM · good = Read

100 labels (Read 3, Skim 15, Skip 82).

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

100 labels (Read 3, Skim 15, Skip 82).

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

100 labels (Read 3, Skim 15, Skip 82).

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

100 labels (Read 3, Skim 15, Skip 82).

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.18; papers to screen for 80% of good ones in random order: 80.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.96 | 0.73 | 0.54 | 90% | 35 |
| Profile only (no learning) | 0.96 | 0.73 | 0.54 | 90% | 35 |
| Learned from feedback only | 0.90 | 0.71 | 0.47 | 70% | 34 |
| Semantic similarity only | 0.93 | 0.72 | 0.52 | 80% | 35 |
| Keyword matches only | 0.89 | 0.66 | 0.50 | 80% | 37 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.61, accuracy 0.85, 0 Read↔Skip errors.

### Climate adaptation · MiniLM · good = Read

200 labels (Read 3, Skim 24, Skip 173).

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

200 labels (Read 3, Skim 24, Skip 173).

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

200 labels (Read 3, Skim 24, Skip 173).

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

200 labels (Read 3, Skim 24, Skip 173).

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.14; papers to screen for 80% of good ones in random order: 160.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.92 | 0.64 | 0.45 | 90% | 69 |
| Profile only (no learning) | 0.92 | 0.64 | 0.45 | 90% | 69 |
| Learned from feedback only | 0.80 | 0.66 | 0.42 | 90% | 52 |
| Semantic similarity only | 0.85 | 0.69 | 0.55 | 80% | 40 |
| Keyword matches only | 0.76 | 0.58 | 0.53 | 70% | 41 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.65, accuracy 0.89, 1 Read↔Skip errors.

### RAG & LLM evaluation · MiniLM · good = Read

200 labels (Read 6, Skim 33, Skip 161).

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

200 labels (Read 6, Skim 33, Skip 161).

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

200 labels (Read 6, Skim 33, Skip 161).

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

200 labels (Read 6, Skim 33, Skip 161).

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.20; papers to screen for 80% of good ones in random order: 160.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.97 | 0.82 | 0.62 | 100% | 51 |
| Profile only (no learning) | 0.97 | 0.82 | 0.62 | 100% | 51 |
| Learned from feedback only | 0.97 | 0.79 | 0.58 | 100% | 55 |
| Semantic similarity only | 0.91 | 0.77 | 0.60 | 100% | 53 |
| Keyword matches only | 0.84 | 0.75 | 0.57 | 100% | 66 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.61, accuracy 0.84, 0 Read↔Skip errors.

## Rule-based labels (pipeline check)

### AI and design · MiniLM · good = Read

160 labels (Read 45, Skim 50, Skip 65).

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.28; papers to screen for 80% of good ones in random order: 128.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.60 | 0.49 | 0.46 | 50% | 85 |
| Profile only (no learning) | 0.60 | 0.49 | 0.46 | 50% | 85 |
| Learned from feedback only | 0.67 | 0.51 | 0.48 | 50% | 82 |
| Semantic similarity only | 0.55 | 0.42 | 0.31 | 40% | 115 |
| Keyword matches only | 0.59 | 0.43 | 0.31 | 40% | 115 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.23, accuracy 0.41, 38 Read↔Skip errors.

### AI and design · MiniLM · good = Read or Skim

160 labels (Read 45, Skim 50, Skip 65).

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.59; papers to screen for 80% of good ones in random order: 128.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.60 | 0.79 | 0.46 | 80% | 106 |
| Profile only (no learning) | 0.60 | 0.79 | 0.46 | 80% | 106 |
| Learned from feedback only | 0.67 | 0.81 | 0.48 | 80% | 103 |
| Semantic similarity only | 0.55 | 0.76 | 0.31 | 80% | 118 |
| Keyword matches only | 0.59 | 0.77 | 0.31 | 80% | 118 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.23, accuracy 0.41, 38 Read↔Skip errors.

### AI and design · TF-IDF · good = Read

160 labels (Read 45, Skim 50, Skip 65).

Learning weight chosen by cross-validation: **50%**. Random-order AP: 0.28; papers to screen for 80% of good ones in random order: 128.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 50%) | 0.74 | 0.58 | 0.53 | 70% | 83 |
| Profile only (no learning) | 0.67 | 0.50 | 0.46 | 60% | 93 |
| Learned from feedback only | 0.86 | 0.58 | 0.49 | 80% | 83 |
| Semantic similarity only | 0.58 | 0.35 | 0.07 | 50% | 116 |
| Keyword matches only | 0.61 | 0.35 | 0.07 | 50% | 116 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.24, accuracy 0.41, 37 Read↔Skip errors.

### AI and design · TF-IDF · good = Read or Skim

160 labels (Read 45, Skim 50, Skip 65).

Learning weight chosen by cross-validation: **50%**. Random-order AP: 0.59; papers to screen for 80% of good ones in random order: 128.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 50%) | 0.74 | 0.83 | 0.53 | 90% | 98 |
| Profile only (no learning) | 0.67 | 0.78 | 0.46 | 80% | 102 |
| Learned from feedback only | 0.86 | 0.83 | 0.49 | 100% | 102 |
| Semantic similarity only | 0.58 | 0.66 | 0.07 | 70% | 121 |
| Keyword matches only | 0.61 | 0.67 | 0.07 | 70% | 121 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.24, accuracy 0.41, 37 Read↔Skip errors.

### Materials discovery · MiniLM · good = Read

160 labels (Read 45, Skim 50, Skip 65).

Learning weight chosen by cross-validation: **70%**. Random-order AP: 0.28; papers to screen for 80% of good ones in random order: 128.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 70%) | 0.89 | 0.63 | 0.61 | 80% | 69 |
| Profile only (no learning) | 0.64 | 0.52 | 0.59 | 40% | 69 |
| Learned from feedback only | 0.85 | 0.63 | 0.62 | 90% | 69 |
| Semantic similarity only | 0.61 | 0.45 | 0.49 | 40% | 79 |
| Keyword matches only | 0.61 | 0.45 | 0.49 | 40% | 79 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.52, accuracy 0.57, 5 Read↔Skip errors.

### Materials discovery · MiniLM · good = Read or Skim

160 labels (Read 45, Skim 50, Skip 65).

Learning weight chosen by cross-validation: **70%**. Random-order AP: 0.59; papers to screen for 80% of good ones in random order: 128.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 70%) | 0.89 | 0.88 | 0.61 | 100% | 86 |
| Profile only (no learning) | 0.64 | 0.83 | 0.59 | 70% | 88 |
| Learned from feedback only | 0.85 | 0.89 | 0.62 | 100% | 85 |
| Semantic similarity only | 0.61 | 0.81 | 0.49 | 70% | 90 |
| Keyword matches only | 0.61 | 0.81 | 0.49 | 70% | 90 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.52, accuracy 0.57, 5 Read↔Skip errors.

### Materials discovery · TF-IDF · good = Read

160 labels (Read 45, Skim 50, Skip 65).

Learning weight chosen by cross-validation: **50%**. Random-order AP: 0.28; papers to screen for 80% of good ones in random order: 128.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 50%) | 0.88 | 0.67 | 0.65 | 80% | 64 |
| Profile only (no learning) | 0.84 | 0.62 | 0.60 | 70% | 69 |
| Learned from feedback only | 0.77 | 0.65 | 0.65 | 70% | 61 |
| Semantic similarity only | 0.88 | 0.49 | 0.32 | 80% | 108 |
| Keyword matches only | 0.88 | 0.49 | 0.32 | 80% | 108 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.35, accuracy 0.48, 22 Read↔Skip errors.

### Materials discovery · TF-IDF · good = Read or Skim

160 labels (Read 45, Skim 50, Skip 65).

Learning weight chosen by cross-validation: **50%**. Random-order AP: 0.59; papers to screen for 80% of good ones in random order: 128.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 50%) | 0.88 | 0.91 | 0.65 | 100% | 89 |
| Profile only (no learning) | 0.84 | 0.88 | 0.60 | 100% | 94 |
| Learned from feedback only | 0.77 | 0.91 | 0.65 | 100% | 89 |
| Semantic similarity only | 0.88 | 0.78 | 0.32 | 100% | 114 |
| Keyword matches only | 0.88 | 0.78 | 0.32 | 100% | 114 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.35, accuracy 0.48, 22 Read↔Skip errors.

### RAG research (rule-based demo) · MiniLM · good = Read

529 labels (Read 9, Skim 98, Skip 422).

Learning weight chosen by cross-validation: **70%**. Random-order AP: 0.02; papers to screen for 80% of good ones in random order: 424.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 70%) | 0.68 | 0.50 | 0.43 | 44% | 72 |
| Profile only (no learning) | 0.68 | 0.49 | 0.50 | 44% | 78 |
| Learned from feedback only | 0.68 | 0.50 | 0.43 | 44% | 71 |
| Semantic similarity only | 0.69 | 0.51 | 0.47 | 44% | 67 |
| Keyword matches only | 0.76 | 0.56 | 0.47 | 56% | 69 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.51, accuracy 0.82, 3 Read↔Skip errors.

### RAG research (rule-based demo) · MiniLM · good = Read or Skim

529 labels (Read 9, Skim 98, Skip 422).

Learning weight chosen by cross-validation: **70%**. Random-order AP: 0.20; papers to screen for 80% of good ones in random order: 424.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 70%) | 0.68 | 0.62 | 0.43 | 80% | 234 |
| Profile only (no learning) | 0.68 | 0.60 | 0.50 | 80% | 299 |
| Learned from feedback only | 0.68 | 0.61 | 0.43 | 80% | 234 |
| Semantic similarity only | 0.69 | 0.59 | 0.47 | 80% | 258 |
| Keyword matches only | 0.76 | 0.60 | 0.47 | 90% | 258 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.51, accuracy 0.82, 3 Read↔Skip errors.

### RAG research (rule-based demo) · TF-IDF · good = Read

529 labels (Read 9, Skim 98, Skip 422).

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.02; papers to screen for 80% of good ones in random order: 424.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.74 | 0.55 | 0.46 | 44% | 148 |
| Profile only (no learning) | 0.74 | 0.55 | 0.46 | 44% | 148 |
| Learned from feedback only | 0.74 | 0.53 | 0.37 | 44% | 152 |
| Semantic similarity only | 0.79 | 0.55 | 0.49 | 56% | 267 |
| Keyword matches only | 0.79 | 0.57 | 0.49 | 56% | 267 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.48, accuracy 0.81, 5 Read↔Skip errors.

### RAG research (rule-based demo) · TF-IDF · good = Read or Skim

529 labels (Read 9, Skim 98, Skip 422).

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.20; papers to screen for 80% of good ones in random order: 424.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.74 | 0.62 | 0.46 | 100% | 300 |
| Profile only (no learning) | 0.74 | 0.62 | 0.46 | 100% | 300 |
| Learned from feedback only | 0.74 | 0.58 | 0.37 | 100% | 298 |
| Semantic similarity only | 0.79 | 0.59 | 0.49 | 100% | 279 |
| Keyword matches only | 0.79 | 0.58 | 0.49 | 100% | 279 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.48, accuracy 0.81, 5 Read↔Skip errors.

### Robotics · MiniLM · good = Read

135 labels (Read 20, Skim 50, Skip 65).

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.15; papers to screen for 80% of good ones in random order: 108.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.85 | 0.59 | 0.49 | 70% | 46 |
| Profile only (no learning) | 0.85 | 0.59 | 0.49 | 70% | 46 |
| Learned from feedback only | 0.79 | 0.57 | 0.48 | 60% | 50 |
| Semantic similarity only | 0.82 | 0.58 | 0.50 | 70% | 59 |
| Keyword matches only | 0.82 | 0.58 | 0.50 | 70% | 59 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.37, accuracy 0.53, 14 Read↔Skip errors.

### Robotics · MiniLM · good = Read or Skim

135 labels (Read 20, Skim 50, Skip 65).

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.52; papers to screen for 80% of good ones in random order: 108.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.85 | 0.79 | 0.49 | 100% | 90 |
| Profile only (no learning) | 0.85 | 0.79 | 0.49 | 100% | 90 |
| Learned from feedback only | 0.79 | 0.78 | 0.48 | 90% | 86 |
| Semantic similarity only | 0.82 | 0.78 | 0.50 | 90% | 97 |
| Keyword matches only | 0.82 | 0.78 | 0.50 | 90% | 97 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.37, accuracy 0.53, 14 Read↔Skip errors.

### Robotics · TF-IDF · good = Read

135 labels (Read 20, Skim 50, Skip 65).

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.15; papers to screen for 80% of good ones in random order: 108.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.69 | 0.54 | 0.55 | 40% | 35 |
| Profile only (no learning) | 0.69 | 0.54 | 0.55 | 40% | 35 |
| Learned from feedback only | 0.70 | 0.48 | 0.51 | 60% | 40 |
| Semantic similarity only | 0.71 | 0.43 | 0.34 | 50% | 74 |
| Keyword matches only | 0.71 | 0.43 | 0.34 | 50% | 74 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.22, accuracy 0.48, 19 Read↔Skip errors.

### Robotics · TF-IDF · good = Read or Skim

135 labels (Read 20, Skim 50, Skip 65).

Learning weight chosen by cross-validation: **0%**. Random-order AP: 0.52; papers to screen for 80% of good ones in random order: 108.

| Method | NDCG@10 | AP | Spearman ρ | Good in top 10 | Papers to 80% |
|---|---|---|---|---|---|
| Personalised (learning 0%) | 0.69 | 0.81 | 0.55 | 100% | 83 |
| Profile only (no learning) | 0.69 | 0.81 | 0.55 | 100% | 83 |
| Learned from feedback only | 0.70 | 0.80 | 0.51 | 100% | 90 |
| Semantic similarity only | 0.71 | 0.72 | 0.34 | 90% | 99 |
| Keyword matches only | 0.71 | 0.72 | 0.34 | 90% | 99 |

At the default cutoffs (Read ≥ 0.62, Skim ≥ 0.4): macro-F1 0.22, accuracy 0.48, 19 Read↔Skip errors.

