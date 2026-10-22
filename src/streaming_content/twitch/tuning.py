"""Spark tuning experiments, each measured before/after on a 10% viewer sample of silver.

Timings are wall-clock medians of REPEATS runs on a local[*] session (12 cores, 24 GB driver) and are
indicative only: Databricks serverless will behave differently. Results: reports/results/tuning.json.
"""

from __future__ import annotations

import json
import statistics
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from streaming_content import config
from streaming_content.twitch import silver as sv

REPEATS = 3


def bench(fn, repeats: int = REPEATS) -> dict:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {"median_s": round(statistics.median(times), 2), "runs_s": [round(x, 2) for x in times]}


def sample_tables(spark: SparkSession, frac: float = 0.1) -> tuple[str, str]:
    """Materialise a viewer-hash sample of silver as two layouts: raw order and by-viewer."""
    src = spark.table(config.table("silver", "twitch_view_sessions"))
    keep = F.pmod(F.hash("user_id"), F.lit(1000)) < int(frac * 1000)
    df = src.filter(keep)
    a, b = config.table("silver", "tune_sample_unclustered"), config.table("silver", "tune_sample_zorder")
    df.repartition(48).write.format("delta").mode("overwrite").saveAsTable(a)  # arbitrary order
    df.repartition(48).write.format("delta").mode("overwrite").saveAsTable(b)
    return a, b


def run(spark: SparkSession) -> dict:
    out: dict = {"sample_fraction": 0.1, "repeats": REPEATS}
    a, b = sample_tables(spark)
    base = spark.table(a)
    n = base.count()
    out["sample_rows"] = n
    out["sample_users"] = base.select("user_id").distinct().count()

    # ---- E0 skew diagnostics
    per_user = base.groupBy("user_id").count()
    q = per_user.approxQuantile("count", [0.5, 0.99, 0.999, 1.0], 0.001)
    out["intervals_per_viewer"] = {"p50": q[0], "p99": q[1], "p99.9": q[2], "max": q[3]}
    per_ch = base.groupBy("streamer_id").count()
    qc = per_ch.approxQuantile("count", [0.5, 0.99, 0.999, 1.0], 0.001)
    out["intervals_per_channel"] = {"p50": qc[0], "p99": qc[1], "p99.9": qc[2], "max": qc[3]}

    # ---- E1 shuffle partitions for the viewer-keyed window (sessionize + user features)
    raw = base.select("user_id", "stream_id", "streamer_id", "start_slot", "stop_slot", "watch_slots")

    def sess_run():
        sv.session_table(sv.sessionize(raw)).write.format("noop").mode("overwrite").save()

    out["E1_shuffle_partitions"] = {}
    for p in (8, 48, 192, 1000):
        spark.conf.set("spark.sql.adaptive.enabled", "false")
        spark.conf.set("spark.sql.shuffle.partitions", str(p))
        out["E1_shuffle_partitions"][str(p)] = bench(sess_run, 2)
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.shuffle.partitions", "192")
    out["E1_shuffle_partitions"]["aqe_on_192"] = bench(sess_run, 2)

    # ---- E2 broadcast vs sort-merge join of the channel dimension (790k rows)
    dim = spark.table(config.table("silver", "twitch_streamers")).select("streamer_id", "streamer")

    def join_run(hint):
        def f():
            d = F.broadcast(dim) if hint == "broadcast" else dim.hint("merge")
            base.join(d, "streamer_id").groupBy("streamer").agg(F.sum("watch_slots")).write.format("noop").mode("overwrite").save()
        return f

    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
    out["E2_join"] = {"sort_merge": bench(join_run("merge")), "broadcast": bench(join_run("broadcast"))}
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")

    # ---- E3 skewed aggregation: distinct viewers per channel, naive vs two-phase salted
    def naive():
        base.groupBy("streamer_id").agg(F.countDistinct("user_id")).write.format("noop").mode("overwrite").save()

    def salted():
        salt = F.pmod(F.hash("user_id"), F.lit(16))
        (base.select("streamer_id", "user_id", salt.alias("salt")).distinct()
             .groupBy("streamer_id", "salt").agg(F.count("*").alias("c"))
             .groupBy("streamer_id").agg(F.sum("c")).write.format("noop").mode("overwrite").save())

    spark.conf.set("spark.sql.adaptive.enabled", "false")
    out["E3_skewed_distinct"] = {"naive_countDistinct": bench(naive), "two_phase_dedupe_then_count": bench(salted)}
    spark.conf.set("spark.sql.adaptive.enabled", "true")

    # ---- E4 file layout: point lookup of 2,000 viewers
    users = [r[0] for r in base.select("user_id").distinct().limit(2000).collect()]
    user_df = spark.createDataFrame([(u,) for u in users], "user_id int")

    def lookup(tbl):
        def f():
            spark.table(tbl).join(F.broadcast(user_df), "user_id").agg(F.sum("watch_slots")).collect()
        return f

    out["E4_layout"] = {"unclustered": bench(lookup(a))}
    try:
        spark.sql(f"OPTIMIZE {b} ZORDER BY (user_id)")
        out["E4_layout"]["zorder_user_id"] = bench(lookup(b))
    except Exception as e:  # pragma: no cover
        out["E4_layout"]["zorder_user_id"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    c = config.table("silver", "tune_sample_liquid")
    try:
        spark.sql(f"DROP TABLE IF EXISTS {c}")
        spark.sql(f"CREATE TABLE {c} USING DELTA CLUSTER BY (user_id) AS SELECT * FROM {a}")
        spark.sql(f"OPTIMIZE {c}")
        out["E4_layout"]["liquid_clustering_user_id"] = bench(lookup(c))
    except Exception as e:  # pragma: no cover
        out["E4_layout"]["liquid_clustering_user_id"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    out["E4_layout"]["files"] = {
        t: int(spark.sql(f"DESCRIBE DETAIL {t}").first()["numFiles"]) for t in (a, b, c) if _exists(spark, t)
    }

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "tuning.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def _exists(spark, t: str) -> bool:
    try:
        spark.table(t)
        return True
    except Exception:
        return False
