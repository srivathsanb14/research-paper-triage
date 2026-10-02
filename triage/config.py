"""Central configuration. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
import sys
from pathlib import Path

BROWSER_MODE = sys.platform == "emscripten" or os.environ.get("TRIAGE_BROWSER_MODE") == "1"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("TRIAGE_DATA_DIR", ROOT / "data"))
DB_PATH = Path(os.environ.get("TRIAGE_DB_PATH", DATA_DIR / "triage.db"))

# --- Triage defaults (human-set, tunable in the UI) -----------------------
DEFAULT_READ_CUTOFF = 0.62
DEFAULT_SKIM_CUTOFF = 0.40
MINUTES_PER_READ = 30  # used to cap READ count by the user's time budget
MINUTES_PER_SKIM = 5

LABELS = ("READ", "SKIM", "SKIP")
LABEL_VALUE = {"READ": 1.0, "SKIM": 0.5, "SKIP": 0.0}
