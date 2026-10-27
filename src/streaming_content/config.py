"""Paths, table naming and constants shared by local runs and Databricks jobs."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("SCA_RAW_DIR", ROOT / "data" / "raw"))
WAREHOUSE = Path(os.environ.get("SCA_WAREHOUSE", ROOT / "data" / "warehouse"))
REPORTS = ROOT / "reports"
# On Databricks results go to a UC volume (SCA_RESULTS_DIR) and are copied back into the repo with the CLI.
RESULTS_DIR = Path(os.environ.get("SCA_RESULTS_DIR", REPORTS / "results"))

# Unity Catalog catalog on Databricks; empty string = local Delta tables in the default catalog.
CATALOG = os.environ.get("SCA_CATALOG", "")

# Twitch: one slot = 10 minutes, 6148 slots over ~43 days.
SLOT_MINUTES = 10
SLOTS_PER_DAY = 144
N_SLOTS = 6148
# Sessionization gap rule: a new session starts when the gap between the end of everything the
# viewer has watched so far and the next start exceeds this many slots (3 slots = 30 min).
SESSION_GAP_SLOTS = 3


def table(layer: str, name: str) -> str:
    """Fully qualified table name: <catalog>.<layer>.<name> on Databricks, <layer>.<name> locally."""
    base = f"{layer}.{name}"
    return f"{CATALOG}.{base}" if CATALOG else base
