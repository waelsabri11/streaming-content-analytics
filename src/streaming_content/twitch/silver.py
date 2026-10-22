"""Silver: validated views and viewer sessions.

Sessionization rule (documented in docs/methodology.md): per viewer, intervals are sorted by
(start, stop); a new session starts when the next interval begins more than SESSION_GAP_SLOTS
after the latest stop seen so far in the current viewer (running max, so overlapping streams
watched in parallel never split a session). Watch time of a session is the *union* of its intervals,
not the sum, so parallel viewing is not double counted.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from streaming_content import config


def clean_views(bronze: DataFrame, dedupe: bool = True) -> DataFrame:
    """Drop corrupt/invalid rows (and exact duplicates when `dedupe`); add watch_slots (half-open [start, stop))."""
    ok = (
        F.col("_corrupt").isNull()
        & F.col("user_id").isNotNull()
        & F.col("streamer").isNotNull()
        & (F.col("start_slot") >= 0)
        & (F.col("stop_slot") > F.col("start_slot"))
        & (F.col("stop_slot") <= config.N_SLOTS)
    )
    cols = ["user_id", "stream_id", "streamer", "start_slot", "stop_slot"]
    df = bronze.filter(ok).select(*cols)
    if dedupe:
        df = df.dropDuplicates()
    return df.withColumn("watch_slots", F.col("stop_slot") - F.col("start_slot"))


def sessionize(views: DataFrame, gap_slots: int = config.SESSION_GAP_SLOTS) -> DataFrame:
    """Return the views with `session_id` (per viewer, 1-based) and `new_watch_slots`.

    `new_watch_slots` is the part of the interval not already covered by earlier intervals of the
    same viewer; summing it per session/viewer gives union watch time.
    """
    w = Window.partitionBy("user_id").orderBy("start_slot", "stop_slot", "stream_id")
    prev_max = F.max("stop_slot").over(w.rowsBetween(Window.unboundedPreceding, -1))
    v = views.withColumn("_prev_max_stop", prev_max)
    new_session = F.when(
        F.col("_prev_max_stop").isNull() | (F.col("start_slot") - F.col("_prev_max_stop") > gap_slots), 1
    ).otherwise(0)
    v = v.withColumn("_new", new_session)
    v = v.withColumn("session_id", F.sum("_new").over(w.rowsBetween(Window.unboundedPreceding, 0)))
    new_slots = F.when(
        F.col("_prev_max_stop").isNull(), F.col("watch_slots")
    ).otherwise(F.greatest(F.lit(0), F.col("stop_slot") - F.greatest(F.col("start_slot"), F.col("_prev_max_stop"))))
    return v.withColumn("new_watch_slots", new_slots).drop("_prev_max_stop", "_new")


def session_table(sessionized: DataFrame) -> DataFrame:
    """One row per (viewer, session)."""
    return sessionized.groupBy("user_id", "session_id").agg(
        F.min("start_slot").alias("start_slot"),
        F.max("stop_slot").alias("stop_slot"),
        F.sum("new_watch_slots").alias("watch_slots"),
        F.count("*").alias("n_intervals"),
        F.countDistinct("streamer_id").alias("n_streamers"),
    )
