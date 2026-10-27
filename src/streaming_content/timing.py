"""Append-only timing/metric log (reports/results/timings.jsonl) used to back README numbers."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from streaming_content import config


def log(record: dict) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    # read-modify-write: append mode raises "Illegal seek" on UC volumes (FUSE)
    path = config.RESULTS_DIR / "timings.jsonl"
    old = path.read_text() if path.exists() else ""
    path.write_text(old + json.dumps(record) + "\n")


@contextmanager
def timed(step: str, **extra):
    t0 = time.perf_counter()
    out: dict = {}
    yield out
    log({"step": step, "seconds": round(time.perf_counter() - t0, 2), **extra, **out})
