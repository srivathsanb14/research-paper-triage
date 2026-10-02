"""Central configuration. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("TRIAGE_DATA_DIR", ROOT / "data"))

# --- Triage defaults (human-set, tunable in the UI) -----------------------
DEFAULT_READ_CUTOFF = 0.62
DEFAULT_SKIM_CUTOFF = 0.40
MINUTES_PER_READ = 30  # used to cap READ count by the user's time budget

LABELS = ("READ", "SKIM", "SKIP")
LABEL_VALUE = {"READ": 1.0, "SKIM": 0.5, "SKIP": 0.0}
