"""Smoke test of a deployment: the 100k-viewer sample through clean -> sessionize, with hard assertions."""

from __future__ import annotations

import json
from pathlib import Path

from streaming_content import config
from streaming_content.twitch import bronze, silver

EXPECTED_ROWS = 3_051_733  # 100k_a.csv, measured
EXPECTED_VIEWERS = 100_000


def run(spark) -> dict:
    df = bronze.read_raw(spark, Path(config.RAW) / "twitch" / "100k_a.csv")
    clean = silver.clean_views(df)
    s = silver.sessionize(clean)
    rows = s.count()
    viewers = s.select("user_id").distinct().count()
    sessions = s.select("user_id", "session_id").distinct().count()
    out = {"rows": rows, "viewers": viewers, "sessions": sessions, "ok": rows == EXPECTED_ROWS and viewers == EXPECTED_VIEWERS}
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "smoke.json").write_text(json.dumps(out) + "\n")
    assert out["ok"], f"smoke test failed: {out}"
    return out
