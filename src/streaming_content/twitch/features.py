"""Gold: per-viewer behavioural features from sessionized intervals."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from streaming_content import config

SLOT_HOURS = config.SLOT_MINUTES / 60.0
FEATURE_COLS = [
    "log_hours", "log_sessions", "mean_session_min", "log_active_days", "sessions_per_active_day",
    "share_night", "share_morning", "share_afternoon", "share_evening", "share_peak_days",
    "log_streamers", "norm_entropy", "top1_share", "repeat_share", "loyalty_days_share",
]


def peak_day_pair(sessionized: DataFrame) -> tuple[int, int]:
    """Two adjacent days of the 7-day cycle with the highest global watch time.

    Slot 0 has no calendar date in the source, so 'weekend' cannot be known a priori; the busiest
    adjacent pair is used as a data-driven stand-in and named `peak_days` (see methodology).
    """
    rows = (
        sessionized.groupBy(((F.col("start_slot") / config.SLOTS_PER_DAY).cast("int") % 7).alias("dow"))
        .agg(F.sum("new_watch_slots").alias("w"))
        .collect()
    )
    w = {r["dow"]: r["w"] for r in rows}
    best = max(range(7), key=lambda d: w.get(d, 0) + w.get((d + 1) % 7, 0))
    return best, (best + 1) % 7


def viewer_features(sessionized: DataFrame, peak_days: tuple[int, int], lo: int = 0, hi: int = config.N_SLOTS) -> DataFrame:
    """One row per viewer with the raw behavioural features (windowed to start_slot in [lo, hi))."""
    # new_watch_slots == 0 rows are fully covered by an earlier interval (parallel stream): no watch time to add
    v = sessionized.filter((F.col("start_slot") >= lo) & (F.col("start_slot") < hi) & (F.col("new_watch_slots") > 0))
    slot_of_day = F.col("start_slot") % config.SLOTS_PER_DAY
    hour = (slot_of_day / 6).cast("int")
    day = (F.col("start_slot") / config.SLOTS_PER_DAY).cast("int")
    w = F.col("new_watch_slots")
    v = v.withColumn("day", day).withColumn("hour", hour).withColumn("dow", day % 7)

    sess = v.groupBy("user_id", "session_id").agg(F.sum(w).alias("slots"))
    s_agg = sess.groupBy("user_id").agg(
        F.count("*").alias("n_sessions"),
        F.avg("slots").alias("mean_session_slots"),
    )

    block = lambda a, b: F.sum(F.when((F.col("hour") >= a) & (F.col("hour") < b), w).otherwise(0))  # noqa: E731
    pk = F.sum(F.when(F.col("dow").isin(*peak_days), w).otherwise(0))
    u_agg = v.groupBy("user_id").agg(
        F.sum(w).alias("slots"),
        F.countDistinct("day").alias("active_days"),
        (block(0, 6) / F.sum(w)).alias("share_night"),
        (block(6, 12) / F.sum(w)).alias("share_morning"),
        (block(12, 18) / F.sum(w)).alias("share_afternoon"),
        (block(18, 24) / F.sum(w)).alias("share_evening"),
        (pk / F.sum(w)).alias("share_peak_days"),
    )

    us = v.groupBy("user_id", "streamer_id").agg(
        F.sum(w).alias("s_slots"),
        F.countDistinct("session_id").alias("s_sessions"),
        F.countDistinct("day").alias("s_days"),
    )
    tot = F.sum("s_slots").over(Window.partitionBy("user_id"))
    us = us.withColumn("p", F.col("s_slots") / tot)
    c_agg = us.groupBy("user_id").agg(
        F.count("*").alias("n_streamers"),
        (-F.sum(F.col("p") * F.log(F.col("p")))).alias("entropy"),
        F.max("p").alias("top1_share"),
        F.sum(F.when(F.col("s_sessions") >= 2, F.col("s_slots")).otherwise(0)).alias("repeat_slots"),
        F.max("s_days").alias("top_days"),
    )

    f = u_agg.join(s_agg, "user_id").join(c_agg, "user_id")
    return f.select(
        "user_id",
        (F.col("slots") * SLOT_HOURS).alias("hours"),
        "n_sessions",
        "active_days",
        "n_streamers",
        F.log1p(F.col("slots") * SLOT_HOURS).alias("log_hours"),
        F.log1p("n_sessions").alias("log_sessions"),
        (F.col("mean_session_slots") * config.SLOT_MINUTES).alias("mean_session_min"),
        F.log1p("active_days").alias("log_active_days"),
        (F.col("n_sessions") / F.col("active_days")).alias("sessions_per_active_day"),
        "share_night", "share_morning", "share_afternoon", "share_evening", "share_peak_days",
        F.log1p("n_streamers").alias("log_streamers"),
        F.when(F.col("n_streamers") > 1, F.col("entropy") / F.log(F.col("n_streamers").cast("double"))).otherwise(0.0).alias("norm_entropy"),
        "top1_share",
        (F.col("repeat_slots") / F.col("slots")).alias("repeat_share"),
        (F.col("top_days") / F.col("active_days")).alias("loyalty_days_share"),
    )
