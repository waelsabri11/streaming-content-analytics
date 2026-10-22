"""Twitch stages on Spark: silver (clean, encode, sessionize) and gold features."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from streaming_content import config
from streaming_content.timing import timed
from streaming_content.twitch import features as feat
from streaming_content.twitch import silver as sv

T_BRONZE = lambda: config.table("bronze", "twitch_interactions")  # noqa: E731
T_STREAMERS = lambda: config.table("silver", "twitch_streamers")  # noqa: E731
T_SESS_VIEWS = lambda: config.table("silver", "twitch_view_sessions")  # noqa: E731
T_SESSIONS = lambda: config.table("gold", "twitch_sessions")  # noqa: E731
T_FEATURES = lambda: config.table("gold", "twitch_viewer_features")  # noqa: E731


def streamer_dimension(spark: SparkSession, bronze: DataFrame) -> DataFrame:
    """Dictionary-encode channel names: 465k names -> int ids (broadcast-joined back)."""
    names = bronze.select("streamer").filter(F.col("streamer").isNotNull()).distinct()
    dim = names.withColumn("streamer_id", F.dense_rank().over(Window.orderBy("streamer")).cast("int"))
    dim.write.format("delta").mode("overwrite").saveAsTable(T_STREAMERS())
    return spark.table(T_STREAMERS())


def build_silver(spark: SparkSession, partitions: int = 192) -> dict:
    out: dict = {}
    bronze = spark.table(T_BRONZE())
    with timed("silver_streamer_dim") as t:
        dim = streamer_dimension(spark, bronze)
        t["streamers"] = dim.count()
    valid = sv.clean_views(bronze.withColumn("_corrupt", F.col("_corrupt")), dedupe=False)
    # one shuffle by viewer: dedupe, window and aggregations below all reuse this partitioning
    v = (
        valid.join(F.broadcast(dim), "streamer")
        .repartition(partitions, "user_id")
        .dropDuplicates(["user_id", "stream_id", "streamer_id", "start_slot", "stop_slot"])
        .drop("streamer")
    )
    s = sv.sessionize(v)
    with timed("silver_sessionize", partitions=partitions) as t:
        s.write.format("delta").mode("overwrite").saveAsTable(T_SESS_VIEWS())
        t["rows"] = spark.table(T_SESS_VIEWS()).count()
        out["silver_rows"] = t["rows"]
    with timed("gold_sessions") as t:
        sv.session_table(spark.table(T_SESS_VIEWS())).write.format("delta").mode("overwrite").saveAsTable(T_SESSIONS())
        t["sessions"] = spark.table(T_SESSIONS()).count()
        out["sessions"] = t["sessions"]
    return out


def build_features(spark: SparkSession) -> dict:
    s = spark.table(T_SESS_VIEWS())
    with timed("gold_features") as t:
        peak = feat.peak_day_pair(s)
        f = feat.viewer_features(s, peak)
        f.write.format("delta").mode("overwrite").saveAsTable(T_FEATURES())
        t["viewers"] = spark.table(T_FEATURES()).count()
        t["peak_days"] = list(peak)
    mid = config.N_SLOTS // 2
    for name, (lo, hi) in {"h1": (0, mid), "h2": (mid, config.N_SLOTS)}.items():
        with timed(f"gold_features_{name}") as t:
            f = feat.viewer_features(s, peak, lo, hi)
            f.write.format("delta").mode("overwrite").saveAsTable(config.table("gold", f"twitch_viewer_features_{name}"))
            t["viewers"] = spark.table(config.table("gold", f"twitch_viewer_features_{name}")).count()
    return {"peak_days": list(peak)}
