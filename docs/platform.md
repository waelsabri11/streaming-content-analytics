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



## What was built on top (all created and run in this workspace, 2026-10-27)

| Item | Result |
|---|---|
| Upload of raw data | `databricks fs cp` of the single 6.9 GB gzip to a UC volume **failed** (`unexpected EOF` on the single PUT). Re-splitting the data into 24 gzip parts (8.7 GB total) and uploading 4 at a time worked (about 2 h for the whole set, throughput about 1-4 MB/s over a home connection). Parts load in parallel on serverless (bronze 140 s) whereas one gzip is a single task. |
| Jobs | `sca_twitch` DAG (bronze -> silver -> features -> segments / churn / A/B / life curves / streaming / OPTIMIZE demo), `sca_netflix`, `sca_lakehouse`, `sca_benchmark`: all ran. `max_concurrent_runs: 1` queued a second run of the same job for 30 minutes (setup_duration 1,895 s in the API); the platform also reports "reached maximum number of allowed active runs (5)" for tasks. |
| Lakeflow Declarative Pipeline | Ran on the 100k-viewer sample via Auto Loader. First two updates failed (an expectation referenced a column dropped by the upstream select; a streaming table cannot host a batch `dropDuplicates` read, so `twitch_views` is a materialized view). Not run on the full table. |
| Structured Streaming | `availableNow` micro-batches from a file source with watermark into a Delta sink worked; Spark Connect returns progress fields (`batchDuration`, `numInputRows`). A stale checkpoint from a crashed earlier attempt made a rerun write nothing: the code now clears checkpoints before a replay. |
| Delta | MERGE, CDF, time travel, RESTORE, CHECK constraints, `CLUSTER BY`, OPTIMIZE, VACUUM all worked (results section 10). |
| Unity Catalog | comments, tags, constraints, feature tables (`databricks-feature-engineering` in the job environment), lineage readable from `system.access.table_lineage`. The built-in lineage UI screenshot was not taken (no browser session on the workspace); the edges are in the results file instead. |
| MLflow + registry | UC registry works (`workspace.gold.netflix_performance_score`, aliases `champion` / `challenger`). A raw LightGBM flavor model failed at serving time (`train and valid dataset categorical_feature do not match`: JSON input has no pandas categoricals); a pyfunc wrapper fixes it (`netflix/serving_model.py`). |
| Model Serving | CPU endpoint with scale-to-zero created without error on Free Edition and answered 100/100 requests (p50 148 ms). |
| SQL warehouse | Serverless starter warehouse (2X-Small, auto-stops): cold first query about 10-18 s, warm dashboard queries 1.5-3 s via the REST API. |
| AI/BI dashboard | Created and published through the Lakeview API from six dataset queries (`jobs/create_dashboard.py`); all six queries return rows; I did not open the rendered dashboard. |
| Genie | Space created from a serialized payload; 5 questions answered in 17-20 s each, SQL stored in `reports/results/databricks/genie.json`. |
| Databricks Apps | `sca-streamlit` deployed (state RUNNING, compute ACTIVE); the CLI could not fetch app logs in this session; the app needs a workspace login, so I did not load its pages. |
| System tables | `system.billing.usage` and `system.access.table_lineage` are readable. Billing rows for today's compute were not yet present when queried (usage lands with a delay), so compute used is not reported; no quota error occurred. |

## Continuous deployment (GitHub Actions)
* **Credential**: a personal access token (generated by the workspace owner in Settings > Developer > Access tokens and stored as the GitHub secret `DATABRICKS_TOKEN`, with `DATABRICKS_HOST`). I did not try a service principal / OAuth M2M, so I cannot say whether Free Edition allows one; the workflow also accepts `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET`.
* **Workflow** (`.github/workflows/ci.yml`): pull requests run tests then `databricks bundle validate -t prod`; a push to `main` runs tests, validate and `databricks bundle deploy -t prod`; a manual run (`workflow_dispatch`, input `run_smoke`) deploys and runs the `sca_smoke` job (100k-viewer sample, asserts 3,051,733 rows and 100,000 viewers).
* **Proof of a real deploy**: push of commit `f0b7120` to `main` triggered [Actions run 40537604496](https://github.com/waelsabri11/streaming-content-analytics/actions/runs/40537604496): jobs `tests` and `databricks` succeeded (validate + deploy). `databricks bundle summary -t prod` afterwards lists the deployed jobs `sca-twitch-prod`, `sca-netflix-prod`, `sca-lakehouse-prod`, `sca-benchmark-prod`, `sca-smoke-prod` and the pipeline `twitch-silver-prod` under `/Workspace/Users/waelsabri4@gmail.com/.bundle/streaming-content-analytics/prod`. No screenshot of the workspace UI was taken (no browser session there).
* The same smoke job was run on the `dev` target before wiring CI (73 s, succeeded). The smoke run triggered from Actions is documented in the README once it has been run.
