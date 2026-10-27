"""Entry point for Databricks job tasks and local runs: python jobs/run.py <stage> [options]."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if "__file__" in globals():  # local runs; on Databricks the wheel provides the package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["netflix", "twitch-bronze", "twitch-silver", "twitch-features",
                                     "twitch-segments", "twitch-lifecurves", "twitch-tuning", "twitch-churn", "twitch-abtest", "twitch-streaming", "lakehouse", "optimize-demo", "export-app-data", "benchmark"])
    p.add_argument("--catalog", default="")
    p.add_argument("--raw-root", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--pct", type=int, default=100)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--driver-mem", default="24g")
    a = p.parse_args()
    if a.catalog:
        os.environ["SCA_CATALOG"] = a.catalog
    if a.raw_root:
        os.environ["SCA_RAW_DIR"] = a.raw_root
    if a.results_dir:
        os.environ["SCA_RESULTS_DIR"] = a.results_dir

    import time

    from streaming_content import config
    from streaming_content.spark import get_spark

    t_start = time.perf_counter()
    spark = get_spark(a.stage, driver_mem=a.driver_mem) if a.stage == "benchmark" else get_spark(a.stage)
    spark.range(1).count()  # first action = session ready
    session_seconds = time.perf_counter() - t_start
    if a.stage == "netflix":
        from streaming_content.netflix import pipeline

        pipeline.run(spark=spark)
    elif a.stage == "twitch-bronze":
        from streaming_content.twitch import bronze

        parts = Path(config.RAW) / "twitch_parts"
        # 24 re-split gzip parts load in parallel; the single 6.9 GB gzip is an unsplittable one-task read
        src = f"{parts}/part-*.csv.gz" if parts.exists() and any(parts.glob("part-*.csv.gz")) else Path(config.RAW) / "twitch" / "full_a.csv.gz"
        bronze.ingest(spark, src)
    elif a.stage == "twitch-silver":
        from streaming_content.twitch import pipeline

        print(json.dumps(pipeline.build_silver(spark)))
    elif a.stage == "twitch-features":
        from streaming_content.twitch import pipeline

        print(json.dumps(pipeline.build_features(spark)))
    elif a.stage == "twitch-segments":
        from streaming_content.twitch import segments_job

        segments_job.run(spark)
    elif a.stage == "twitch-lifecurves":
        from streaming_content.twitch import lifecurves_job

        lifecurves_job.run(spark)
    elif a.stage == "twitch-churn":
        from streaming_content.twitch import churn

        churn.run(spark)
    elif a.stage == "lakehouse":
        from streaming_content import lakehouse

        lakehouse.run(spark)
    elif a.stage == "benchmark":
        from streaming_content import benchmark
        from streaming_content.spark import on_databricks

        benchmark.run(spark, a.pct, a.repeats, label="databricks" if on_databricks() else f"local_{a.driver_mem}", session_start_seconds=session_seconds)
    elif a.stage == "export-app-data":
        from streaming_content import app_export

        app_export.main(spark)
    elif a.stage == "optimize-demo":
        from streaming_content import lakehouse

        lakehouse.run(spark, ("optimize_demo",))
    elif a.stage == "twitch-streaming":
        from streaming_content.twitch import streaming

        streaming.run(spark)
    elif a.stage == "twitch-abtest":
        from streaming_content.twitch import abtest

        abtest.run(spark)
    elif a.stage == "twitch-tuning":
        from streaming_content.twitch import tuning

        tuning.run(spark)


if __name__ == "__main__":
    main()
