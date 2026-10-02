---
license: apache-2.0
language:
  - en
library_name: transformers.js
pipeline_tag: feature-extraction
base_model: sentence-transformers/all-MiniLM-L6-v2
tags:
  - sentence-similarity
  - research-papers
  - off-the-shelf
---

# MiniLM sentence embeddings in Paper Triage (off-the-shelf)

Paper Triage uses **[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)**
unchanged, through its ONNX export [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2)
(8-bit quantized, ~23 MB) and [transformers.js](https://github.com/huggingface/transformers.js) 3.8.1.
This card describes how we use it. The model itself is not modified or fine-tuned.

## Role in the system

* **Catalog side (build time).** `scripts/embed_catalog.mjs` embeds every catalog paper as
  `"title. abstract"` (first 2,000 characters), mean pooling, L2-normalised. Vectors are stored as
  int8 with a per-vector scale (`data/catalog/*.bin`, 388 bytes per paper) and checked against the
  paper shard's SHA-256.
* **Visitor side (in the browser).** A Web Worker loads the same model and quantization and embeds
  the visitor's description, focus, keywords and excluded topics, plus any papers they add.
  Paper and profile vectors therefore share one space.
* **Use.** Cosine similarities feed four of the ranker's seven features (research, focus, closest
  keyword, excluded topic). Similarities are mapped to [0, 1] with fixed ranges: cosine 0.15 → 0 and 0.50 → 1
  for documents; 0.12 → 0 and 0.45 → 1 for short phrases. Near-duplicates (cosine ≥ 0.72) are grouped.
* **Fallback.** Until the model has downloaded (or if it is disabled), a TF-IDF space is used with its
  own calibration, so the app is usable within a second of opening.

## Why this model

Small enough to download once and run on a laptop CPU or phone in a browser, strong on short
English text similarity, Apache-2.0 licensed, and widely used (good known behaviour). Larger
embedding models (e.g. 768-dim) would triple the catalog vectors and the first-visit download for
modest gains on short abstracts; scientific-domain models (SPECTER2) need custom pooling and are
not available as small quantized ONNX builds for transformers.js.

We checked calibration on the catalog: for realistic profiles (RAG, climate adaptation, CRISPR,
household finance, digital humanities) the best matches score cosine 0.41–0.57 and the median paper
≈ 0.0–0.05, so the 0.15–0.50 range spreads relevant papers across the Read/Skim cutoffs.

## Evaluation

The embedding model is evaluated as part of the ranker: the "Semantic similarity only" baseline
in [`docs/evaluation/report.md`](https://github.com/srivathsanb14/research-paper-triage/blob/main/docs/evaluation/report.md)
uses nothing but MiniLM cosine to the profile, and the report compares the full ranker in MiniLM vs
TF-IDF space on the same labels.

On the 500 manual labels (good = Read), the full ranker reaches AP 1.00, 0.83, 1.00 with MiniLM vs 0.83, 0.78, 1.00 with TF-IDF (RAG, climate and cancer profiles).

## Limitations

English only. Inputs are truncated by the tokenizer (MiniLM was trained on ≤ 256 tokens), so long
abstracts are represented mostly by their first part. General-domain training: very specialised
terminology may be embedded less precisely than by domain models. Inherits any biases of the
original model's training data (see the
[upstream card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)).
