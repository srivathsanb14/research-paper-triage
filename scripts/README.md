# Scripts

Everything you can run from the repo root, grouped by what you are trying to do. Python scripts use the
virtualenv (`.venv/bin/python`); Node scripts need `npm ci` once.

## Run the app
| Script | What it does |
|---|---|
| `python scripts/build_pages.py` | Builds the static website into `_site/` (only public files). Add `--hf-space` for the Hugging Face front matter. |
| `python -m server` | Serves `_site/` plus sign-in and per-user storage. `python -m server.demo_users` creates the demo accounts. |

## Refresh the data
| Script | What it does |
|---|---|
| `python scripts/build_catalog.py` | Re-collects the paper catalog from OpenAlex into `data/catalog/` (needs network; optional `OPENALEX_API_KEY`). |
| `node scripts/embed_catalog.mjs` | Computes MiniLM vectors for new catalog papers (needs network once for the model). Run after `build_catalog.py`. |
| `node scripts/propose_candidates.mjs` | Picks papers to label for the review profiles in `data/labels/profiles.json`. |
| `python scripts/simulate_labels.py` | Regenerates the synthetic (rule-based) rows of `data/labels/labels.jsonl`; manual rows are untouched. |

## Measure and document
| Script | What it does |
|---|---|
| `node scripts/evaluate_web.mjs` | 5-fold cross-validated evaluation per profile, written to `docs/evaluation/report.{md,json}`. |
| `python scripts/build_dataset.py` | Validates the labels against the catalog and writes `_dataset/` and `docs/EDA.md`. |
| `node scripts/export_model_config.mjs` | Writes `hf/model-ranker/config.json` from the engine's constants. |
| `python scripts/render_model_card.py` | Fills the evaluation numbers in the Hugging Face model cards from `report.json`. |

## Test and publish
| Script | What it does |
|---|---|
| `node scripts/check_pages.cjs` (`npm run test:pages`) | End-to-end browser test of the built site (Playwright; build the site first). |
| `scripts/publish_hf.sh <hf-user>` | Builds everything above that publishing needs and uploads the Space, dataset and model cards. |

Typical order after changing labels: `evaluate_web.mjs` → `build_dataset.py` → `render_model_card.py`.
