"""Structured Streaming REPLAY of Twitch presence events as a simulated live feed.

This is a replay, not a live source: the historical intervals of a viewer sample are expanded into one event
per (viewer, channel, 10-minute slot), written as time-ordered parquet files (one file per chunk of slots), and
read back with `maxFilesPerTrigger` so micro-batches arrive in event-time order. The event time is a SYNTHETIC
timestamp (2018-01-01 + slot * 10 min): the source has no calendar date. Streaming queries (watermark 30 min):
  * concurrent viewers per channel and 10-minute window (a presence row is one viewer in one slot, so count = viewers);
  * platform-wide concurrent viewers per window.
Run with trigger(availableNow=True) so it works on serverless; per-batch timings come from the query progress.
"""

from __future__ import annotations

import json
import time

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from streaming_content import config
from streaming_content.timing import timed

EPOCH = "2018-01-01 00:00:00"  # synthetic
CHUNK_SLOTS = 36  # 6 hours of event time per replay file


def event_time(slot_col):
    return (F.lit(EPOCH).cast("timestamp").cast("long") + slot_col * config.SLOT_MINUTES * 60).cast("timestamp")


def prepare_replay(spark: SparkSession, base: str, viewer_pct: int = 2, slot_lo: int = 0, slot_hi: int = 1008) -> dict:
    """Expand a viewer sample into presence events and write one parquet file per chunk of slots."""
    s = spark.table(config.table("silver", "twitch_view_sessions"))
    sample = s.filter((F.pmod(F.hash("user_id"), F.lit(100)) < viewer_pct) & (F.col("start_slot") < slot_hi) & (F.col("stop_slot") > slot_lo))
    ev = (
        sample.select("user_id", "streamer_id", F.explode(F.sequence("start_slot", F.col("stop_slot") - 1)).alias("slot"))
        .filter((F.col("slot") >= slot_lo) & (F.col("slot") < slot_hi))
        .dropDuplicates(["user_id", "streamer_id", "slot"])
    )
    ev.write.format("delta").mode("overwrite").saveAsTable(config.table("silver", "twitch_presence_replay_source"))
    src = spark.table(config.table("silver", "twitch_presence_replay_source"))
    n = src.count()
    t0 = time.perf_counter()
    for c0 in range(slot_lo, slot_hi, CHUNK_SLOTS):
        (src.filter((F.col("slot") >= c0) & (F.col("slot") < c0 + CHUNK_SLOTS)).coalesce(1)
            .write.mode("overwrite").parquet(f"{base}/in/chunk={c0:05d}"))
    return {"events": n, "chunks": len(range(slot_lo, slot_hi, CHUNK_SLOTS)), "write_seconds": round(time.perf_counter() - t0, 1),
            "viewer_pct": viewer_pct, "slots": [slot_lo, slot_hi]}


def run_stream(spark: SparkSession, base: str, max_files_per_trigger: int = 1) -> dict:
    schema = "user_id INT, streamer_id INT, slot INT"
    t_chan = config.table("gold", "twitch_concurrent_viewers_stream")
    t_plat = config.table("gold", "twitch_concurrent_viewers_platform_stream")
    for t in (t_chan, t_plat):
        spark.sql(f"DROP TABLE IF EXISTS {t}")
    src = (spark.readStream.schema(schema).option("maxFilesPerTrigger", max_files_per_trigger)
           .option("recursiveFileLookup", "true").parquet(f"{base}/in")
           .withColumn("event_time", event_time(F.col("slot"))).withWatermark("event_time", "30 minutes"))
    per_channel = src.groupBy(F.window("event_time", "10 minutes").alias("w"), "streamer_id").agg(F.count("*").alias("concurrent_viewers")) \
        .select(F.col("w.start").alias("window_start"), "streamer_id", "concurrent_viewers")
    platform = src.groupBy(F.window("event_time", "10 minutes").alias("w")).agg(F.count("*").alias("concurrent_viewers")) \
        .select(F.col("w.start").alias("window_start"), "concurrent_viewers")
    out: dict = {"replay": True, "event_time": "synthetic: 2018-01-01 + slot*10min"}
    for name, df, table in (("per_channel", per_channel, t_chan), ("platform", platform, t_plat)):
        ck = f"{base}/ckpt_{name}"
        t0 = time.perf_counter()
        q = (df.writeStream.format("delta").outputMode("append").option("checkpointLocation", ck)
             .trigger(availableNow=True).toTable(table))
        q.awaitTermination()
        wall = time.perf_counter() - t0
        prog = [p for p in (q.recentProgress or [])]
        dur = np.array([p["batchDuration"] for p in prog if p.get("numInputRows", 0) > 0], dtype=float) / 1000.0
        rows = sum(p.get("numInputRows", 0) for p in prog)
        out[name] = {
            "wall_seconds": round(wall, 2), "batches": len(prog), "input_rows": rows, "rows_per_second_overall": rows / wall if wall else None,
            "batch_seconds_p50": float(np.percentile(dur, 50)) if len(dur) else None, "batch_seconds_p95": float(np.percentile(dur, 95)) if len(dur) else None,
            "batch_seconds_p99": float(np.percentile(dur, 99)) if len(dur) else None, "output_rows": spark.table(table).count(),
        }
    return out


def run(spark: SparkSession, base: str | None = None, viewer_pct: int = 2) -> dict:
    if base is None:
        base = f"/Volumes/{config.CATALOG}/bronze/raw/_replay" if config.CATALOG else str(config.WAREHOUSE / "_replay")
    with timed("stream_prepare") as t:
        prep = prepare_replay(spark, base, viewer_pct)
        t.update(prep)
    with timed("stream_replay") as t:
        res = run_stream(spark, base)
        t["input_rows"] = res["per_channel"]["input_rows"]
    res["prepare"] = prep
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "twitch_streaming.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    return res
