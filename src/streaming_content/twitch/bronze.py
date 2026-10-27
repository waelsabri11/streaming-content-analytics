"""Bronze: raw Twitch interactions, schema-enforced, no cleaning.

The source is a single gzip file, which Spark cannot split: the read is one task. That is the
reason bronze exists - decompress and parse once, then everything downstream reads splittable
Delta/Parquet.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from streaming_content import config
from streaming_content.timing import timed

SCHEMA = "user_id INT, stream_id BIGINT, streamer STRING, start_slot INT, stop_slot INT"


def read_raw(spark: SparkSession, path: str | Path) -> DataFrame:
    return (
        spark.read.schema(SCHEMA + ", _corrupt STRING")
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", "_corrupt")
        .csv(str(path))
    )


def ingest(spark: SparkSession, path: str | Path, target: str | None = None, max_rows_per_file: int = 8_000_000) -> int:
    """Single-pass load. No repartition: the gzip read is one task anyway, and a shuffle of the
    474M-row table would double the disk footprint; maxRecordsPerFile keeps output files splittable."""
    target = target or config.table("bronze", "twitch_interactions")
    with timed("bronze_ingest", source=Path(str(path)).name, target=target) as out:
        df = read_raw(spark, path).withColumn("_source", F.lit(Path(str(path)).name))
        (df.write.format("delta").mode("overwrite").option("maxRecordsPerFile", max_rows_per_file)
           .saveAsTable(target))
        n = spark.table(target).count()
        out["rows"] = n
    return n


if __name__ == "__main__":
    import sys

    from streaming_content.spark import get_spark

    ingest(get_spark("bronze"), sys.argv[1])
