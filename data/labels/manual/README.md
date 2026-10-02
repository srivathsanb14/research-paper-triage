# Manual labels

Put label files exported from the app's **Label** tab here (`Labels CSV` or `Labels JSONL`).
One file per labeller and session is fine; `scripts/build_dataset.py` validates and merges them,
keeping the latest label per (labeller, profile, paper).

Only human judgements belong in this folder. Rule- or model-generated labels go in `../synthetic/`.
