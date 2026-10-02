// Writes hf/model-ranker/config.json from the engine's constants (the published
// model card must describe the code that actually runs).  node scripts/export_model_config.mjs
import { writeFile } from "node:fs/promises";
import { BLEND_GRID, DEFAULT_CUTOFFS, DenseSpace, FEATURES, FEATURE_NAMES, MIN_LABELS, MINUTES_PER_READ, PRIOR_WEIGHTS, SparseSpace } from "../web/js/engine.js";

const config = {
  model_type: "paper-triage-relevance-ranker",
  version: 1,
  features: FEATURES.map(f => ({ name: f, description: FEATURE_NAMES[f], prior_weight: PRIOR_WEIGHTS[f] })),
  prior: "clip(sum(prior_weight * feature), 0, 1)",
  learned: { type: "ridge_regression", alpha: 1.0, fit_intercept: true, min_examples: 8, target: { READ: 1, SKIM: 0.5, SKIP: 0 } },
  blend: { final: "(1 - w) * prior + w * learned", grid: BLEND_GRID, selection: "5-fold CV average precision on hand labels; w > 0 only if it beats w = 0 by 0.01", min_labels: MIN_LABELS, unvalidated: "min(0.3, n / (n + 30))" },
  calibration: { minilm: new DenseSpace().cal, tfidf: new SparseSpace([]).cal },
  near_duplicate_threshold: { minilm: new DenseSpace().groupThreshold, tfidf: new SparseSpace([]).groupThreshold },
  triage: { cutoffs: DEFAULT_CUTOFFS, minutes_per_read: MINUTES_PER_READ, read_budget: "floor(hours_per_week * 60 / minutes_per_read)" },
  embedding_model: "Xenova/all-MiniLM-L6-v2 (q8, transformers.js 3.8.1)",
};
await writeFile(new URL("../hf/model-ranker/config.json", import.meta.url), JSON.stringify(config, null, 2) + "\n");
console.log("Wrote hf/model-ranker/config.json");
