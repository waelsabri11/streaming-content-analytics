# Platform notes: Databricks Free Edition (measured 2026-10-27)

Workspace: AWS-hosted Free Edition, serverless only. Probe script: [`jobs/probe/serverless_probe.py`](../jobs/probe/serverless_probe.py); raw outputs:
`reports/results/platform_probe_env3.json`, `platform_probe_env5_no_ml_tmp.json`, `platform_probe_env5_ml_tmp.json`. Tiny data (20k rows), so these show *whether* a feature works, not how fast.

| Feature | Result | Detail / workaround |
|---|---|---|
| Serverless Spark | works | Spark Connect, Spark 4.2.x server. Jobs use `environment_version` "5" (see below). |
| Unity Catalog | works | Catalog `workspace` (default); I created schemas `bronze`, `silver`, `gold` and volumes `bronze.raw`, `gold.results`. Raw files are uploaded to the volume with `databricks fs cp` (~4 MB/s measured on a 114 MB file). |
| `df.cache()` / `persist` | works **only with environment v5**; environment v3 raises `CACHE_REQUIRES_NEWER_CLIENT_ON_SERVERLESS_GC` | bundle uses v5. |
| MLlib KMeans, ClusteringEvaluator, ALS | work (v5) | |
| MLlib LogisticRegression, GBTClassifier, CrossValidator, TrainValidationSplit | work (v5) **after** setting `SPARKML_TEMP_DFS_PATH` to a UC volume path | `spark.py` sets it to `/Volumes/<catalog>/bronze/raw/_ml_tmp` on Databricks; without it: `SPARKML_TEMP_DFS_PATH ... must be set`. |
| MLlib BisectingKMeans | **fails** (`DBFS_DISABLED`: model cache path under the disabled public DBFS root) | Use KMeans on Databricks. |
| Spark confs | `spark.sql.shuffle.partitions` is readable/settable (value `auto`); `spark.sql.adaptive.enabled`, `autoBroadcastJoinThreshold`, `adaptive.skewJoin.enabled`, `spark.executor.memory` raise `CONFIG_NOT_AVAILABLE` | The partition / broadcast / skew tuning experiments run **locally** (fixed resources); Delta-level ones (OPTIMIZE, clustering, data skipping) can run on Databricks. Join strategy can still be forced with hints in code. |
| Structured Streaming | works with `trigger(availableNow=True)` (file source -> watermark -> windowed count -> Delta table, with a checkpoint on a volume): 30-67 s for 1,000 events including startup | Continuous triggers were not tested. |
| Delta | works | `CLUSTER BY` (liquid clustering), NOT NULL, `CHECK` constraint, `MERGE`, `OPTIMIZE`, `DESCRIBE HISTORY`, time travel (`VERSION AS OF`), change data feed (`table_changes`). |
| System tables | readable | `system.billing.usage` returned rows (used below for compute accounting). |
| Jobs / Asset Bundles | works | `databricks bundle validate/deploy/run` against the real workspace; serverless job tasks use `spark_python_task` + `environments` with a wheel built by `uv build`. |

Docs-stated limits (Databricks "Free Edition limitations", page updated 2026-10-18): serverless only; one 2X-Small SQL warehouse; max 5 concurrent job tasks; one active Lakeflow pipeline per type;
up to 3 Apps (auto-stopped after 24 h); limited serverless notebook size and usage; outbound internet restricted to trusted domains; compute stops for the rest of the day when the daily quota is exceeded.
The quota size itself is not published; compute actually used is reported in the benchmark section once the runs have happened.

Observed startup of the probe job: see `reports/results/platform_probe_*.json` run ids and `reports/local_vs_databricks.md`.
