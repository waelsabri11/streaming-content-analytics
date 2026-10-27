"""SparkSession factory: Spark Connect / serverless on Databricks, local Delta otherwise."""

from __future__ import annotations

import os
import shutil
import sys

from pyspark.sql import SparkSession

from streaming_content import config


def on_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ or os.environ.get("DATABRICKS_SERVERLESS") == "1"


def get_spark(app: str = "streaming-content", cores: str = "*", driver_mem: str = "24g") -> SparkSession:
    if on_databricks():
        # serverless Spark ML stages models/data through a UC volume path (probe: reports/results/platform_probe_*.json)
        os.environ.setdefault("SPARKML_TEMP_DFS_PATH", f"/Volumes/{config.CATALOG or 'workspace'}/bronze/raw/_ml_tmp")
        return SparkSession.builder.getOrCreate()
    from delta import configure_spark_with_delta_pip

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    config.WAREHOUSE.mkdir(parents=True, exist_ok=True)
    builder = (
        SparkSession.builder.appName(app)
        .master(f"local[{cores}]")
        .config("spark.driver.memory", driver_mem)
        .config("spark.sql.warehouse.dir", str(config.WAREHOUSE))
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "96")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.local.dir", str(config.WAREHOUSE / "_tmp"))
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    for layer in ("bronze", "silver", "gold"):
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {layer}")
        # the local catalog is in-memory: re-register Delta tables already on disk from earlier runs
        # a crashed write leaves a table directory without a transaction log: clear it, or the next
        # saveAsTable refuses to create the table
        for d in (config.WAREHOUSE / f"{layer}.db").glob("*"):
            if d.is_dir() and not any((d / "_delta_log").glob("*.json")):
                shutil.rmtree(d)
        for tdir in sorted((config.WAREHOUSE / f"{layer}.db").glob("*/_delta_log")):
            if not any(tdir.glob("*.json")):
                continue
            name = tdir.parent.name
            spark.sql(f"CREATE TABLE IF NOT EXISTS {layer}.{name} USING DELTA LOCATION '{tdir.parent}'")
    return spark


def cached(df):
    """df.cache() locally; a no-op on Databricks serverless, which does not support DataFrame caching."""
    return df if on_databricks() else df.cache()


def uncache(df) -> None:
    if not on_databricks():
        df.unpersist()
