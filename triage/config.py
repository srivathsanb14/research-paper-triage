"""Central configuration. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("TRIAGE_DATA_DIR", ROOT / "data"))

# --- Triage defaults (human-set, tunable in the UI) -----------------------
