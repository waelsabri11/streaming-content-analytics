"""Same-code throughput benchmark: sessionize -> viewer features -> MLlib KMeans on a viewer-hash sample of bronze.

Run identically on the Mac (local[*]) and on Databricks serverless: input is the bronze Delta table filtered to
pmod(hash(user_id), 100) < pct, each step writes its output as a Delta table, no explicit caching on either side.
Repeats run in one process: repeat 1 is "cold" (new session, first read of the data), later repeats are "warm".
"""

from __future__ import annotations

import json
import time

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from streaming_content import config
from streaming_content.twitch import features as feat
from streaming_content.twitch import silver as sv


def run_once(spark: SparkSession, pct: int, partitions: int = 192) -> dict:
    t = config.table
    steps: dict = {}
    t0 = time.perf_counter()
    bronze = spark.table(t("bronze", "twitch_interactions")).filter(F.pmod(F.hash("user_id"), F.lit(100)) < pct)
    dim = spark.table(t("silver", "twitch_streamers"))
    v = (sv.clean_views(bronze, dedupe=False).join(F.broadcast(dim), "streamer").repartition(partitions, "user_id")
         .dropDuplicates(["user_id", "stream_id", "streamer_id", "start_slot", "stop_slot"]).drop("streamer"))
    sv.sessionize(v).write.format("delta").mode("overwrite").saveAsTable(t("silver", "bench_sessions"))
    steps["sessionize_seconds"] = time.perf_counter() - t0
    steps["rows"] = spark.table(t("silver", "bench_sessions")).count()

    t0 = time.perf_counter()
    s = spark.table(t("silver", "bench_sessions"))
    feat.viewer_features(s, (6, 0)).write.format("delta").mode("overwrite").saveAsTable(t("gold", "bench_features"))
    steps["features_seconds"] = time.perf_counter() - t0
    f = spark.table(t("gold", "bench_features"))
    steps["viewers"] = f.count()

    t0 = time.perf_counter()
    pipe = Pipeline(stages=[VectorAssembler(inputCols=feat.FEATURE_COLS, outputCol="raw"),
                            StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True),
                            KMeans(k=7, seed=42, maxIter=20, featuresCol="features", predictionCol="segment")])
    pipe.fit(f).transform(f).groupBy("segment").count().collect()
    steps["kmeans_seconds"] = time.perf_counter() - t0
    steps["total_seconds"] = steps["sessionize_seconds"] + steps["features_seconds"] + steps["kmeans_seconds"]
    return {k: (round(x, 2) if isinstance(x, float) else x) for k, x in steps.items()}


def run(spark: SparkSession, pct: int, repeats: int = 3, label: str = "databricks", session_start_seconds: float | None = None) -> dict:
    runs = []
    for i in range(repeats):
        r = run_once(spark, pct)
        r["repeat"] = i + 1
        r["kind"] = "cold" if i == 0 else "warm"
        runs.append(r)
    out = {"label": label, "pct": pct, "runs": runs, "session_start_seconds": session_start_seconds}
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"benchmark_{label}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old[str(pct)] = out
    path.write_text(json.dumps(old, indent=1))
    return out
