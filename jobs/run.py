"""Entry point for Databricks job tasks and local runs: python jobs/run.py <stage> [options]."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["netflix", "twitch-bronze", "twitch-silver", "twitch-features",
                                     "twitch-segments", "twitch-lifecurves", "twitch-tuning"])
    p.add_argument("--catalog", default="")
    p.add_argument("--raw-root", default=None)
    a = p.parse_args()
    if a.catalog:
        os.environ["SCA_CATALOG"] = a.catalog
    if a.raw_root:
        os.environ["SCA_RAW_DIR"] = a.raw_root

    from streaming_content import config
    from streaming_content.spark import get_spark

    spark = get_spark(a.stage)
    if a.stage == "netflix":
        from streaming_content.netflix import pipeline

        pipeline.run(spark=spark)
    elif a.stage == "twitch-bronze":
        from streaming_content.twitch import bronze

        bronze.ingest(spark, Path(config.RAW) / "twitch" / "full_a.csv.gz")
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
    elif a.stage == "twitch-tuning":
        from streaming_content.twitch import tuning

        tuning.run(spark)


if __name__ == "__main__":
    main()
