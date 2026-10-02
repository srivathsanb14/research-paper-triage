#!/usr/bin/env bash
# Publish the Space, dataset and model cards to the Hugging Face Hub.
#
#   pip install -U huggingface_hub && hf auth login     # once, with a write token
#   scripts/publish_hf.sh <hf-user-or-org>
#
# Creates (or updates) four public repos:
#   <user>/paper-triage                    static Space (the website)
#   <user>/paper-triage-dataset            papers + manual + synthetic labels, with EDA
#   <user>/paper-triage-ranker             relevance ranker card, config and evaluation
#   <user>/paper-triage-minilm-embeddings  how the off-the-shelf MiniLM model is used
set -euo pipefail
USER_NS="${1:?usage: scripts/publish_hf.sh <hf-user-or-org>}"
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"

"$PY" scripts/build_pages.py --hf-space
"$PY" scripts/build_dataset.py
node scripts/export_model_config.mjs
RANKER="$(mktemp -d)"
trap 'rm -rf "$RANKER"' EXIT
cp -R hf/model-ranker/. "$RANKER/"
mkdir -p "$RANKER/evaluation"
cp docs/evaluation/report.md docs/evaluation/report.json "$RANKER/evaluation/" 2>/dev/null || true
sed -i.bak "s#<hf-user>#${USER_NS}#g" "$RANKER/README.md" && rm "$RANKER/README.md.bak"

hf repo create "${USER_NS}/paper-triage" --repo-type=space --space_sdk=static --exist-ok
hf upload "${USER_NS}/paper-triage" _site . --repo-type=space --delete='*' --commit-message="Deploy website"
hf repo create "${USER_NS}/paper-triage-dataset" --repo-type=dataset --exist-ok
hf upload "${USER_NS}/paper-triage-dataset" _dataset . --repo-type=dataset --commit-message="Update dataset"
hf repo create "${USER_NS}/paper-triage-ranker" --exist-ok
hf upload "${USER_NS}/paper-triage-ranker" "$RANKER" . --commit-message="Update model card"
hf repo create "${USER_NS}/paper-triage-minilm-embeddings" --exist-ok
hf upload "${USER_NS}/paper-triage-minilm-embeddings" hf/model-embeddings . --commit-message="Update model card"

echo
echo "Space:   https://huggingface.co/spaces/${USER_NS}/paper-triage"
echo "Dataset: https://huggingface.co/datasets/${USER_NS}/paper-triage-dataset"
echo "Models:  https://huggingface.co/${USER_NS}/paper-triage-ranker"
echo "         https://huggingface.co/${USER_NS}/paper-triage-minilm-embeddings"
