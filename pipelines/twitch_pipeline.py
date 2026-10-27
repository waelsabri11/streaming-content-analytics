"""Lakeflow Declarative Pipeline: Twitch bronze -> silver (validated, sessionized).

Deployed by the Asset Bundle in resources/. Not executed locally (the `pyspark.pipelines` runtime
only exists on Databricks); the transformation logic lives in `streaming_content.twitch.silver`,
which is unit-tested with plain Spark, and tests/test_pipeline_wiring.py checks that this file
registers the expected datasets and expectations against a stub of the decorator API.
"""

import sys

from pyspark import pipelines as dp
from pyspark.sql import functions as F

# the bundle ships src/ next to the pipeline; make the package importable
sys.path.append(spark.conf.get("bundle.sourcePath", "."))  # noqa: F821

from streaming_content import config  # noqa: E402
from streaming_content.twitch import bronze as bronze_mod  # noqa: E402
from streaming_content.twitch import silver as silver_mod  # noqa: E402

RAW_PATH = spark.conf.get("twitch.raw_path")  # noqa: F821  e.g. /Volumes/<catalog>/bronze/raw/twitch/


@dp.table(name="twitch_interactions", comment="Raw Twitch viewing intervals (10-minute slots), as ingested")
@dp.expect("parsed_without_error", "_corrupt IS NULL")
def twitch_interactions():
    return (
        spark.readStream.format("cloudFiles")  # noqa: F821
        .option("cloudFiles.format", "csv")
        .schema(bronze_mod.SCHEMA + ", _corrupt STRING")
        .option("columnNameOfCorruptRecord", "_corrupt")
        .load(RAW_PATH)
        .withColumn("_source", F.col("_metadata.file_name"))
    )


@dp.materialized_view(name="twitch_views", comment="Validated, de-duplicated viewing intervals with watch_slots")
@dp.expect_or_drop("positive_duration", "stop_slot > start_slot")
@dp.expect_or_drop("slot_in_window", f"start_slot >= 0 AND stop_slot <= {config.N_SLOTS}")
@dp.expect_or_drop("has_viewer_and_channel", "user_id IS NOT NULL AND streamer IS NOT NULL")
def twitch_views():
    return silver_mod.clean_views(dp.read("twitch_interactions"))


@dp.materialized_view(
    name="twitch_view_sessions",
    comment=f"Intervals with per-viewer session_id (gap > {config.SESSION_GAP_SLOTS} slots) and union watch time",
)
@dp.expect_or_fail("session_ids_positive", "session_id >= 1")
@dp.expect("new_watch_not_above_raw", "new_watch_slots <= watch_slots")
def twitch_view_sessions():
    return silver_mod.sessionize(dp.read("twitch_views"))
