# Deploying on Databricks Free Edition

Status: **not deployed.** There was no Databricks CLI profile, `DATABRICKS_HOST` or `~/.databrickscfg` on the machine that built this
project, and accounts are created by the owner, not by the build session. Everything below is written against the documented
Free Edition behaviour and is **untested on a real workspace**.

## Free Edition limits that shape the design (from the Databricks docs, page last updated 2026-10-18)
* Serverless compute only; custom clusters are not available. Daily compute quota: exceeding it shuts compute down for the rest of the day
  (in extreme cases the month). **The quota size is not published in the docs, so I could not measure it without a workspace.** The first
  job run will show whether the 474M-row stages fit; if not, scale down (below) and write the window into the README.
* Outbound internet is restricted to a list of trusted domains, so `ingest/download.py` probably cannot fetch the Twitch file from
  `mcauleylab.ucsd.edu` from inside the workspace. Download locally and upload to a Unity Catalog volume instead.
* Jobs: at most 5 concurrent tasks. Lakeflow pipelines: one active pipeline per type. Apps: up to 3 per account, auto-stopped after 24 h.
* Serverless does not support DataFrame caching; the code routes `.cache()` through `streaming_content.spark.cached`, a no-op there.
* `spark.conf` tuning flags used in `twitch/tuning.py` are only a local experiment; most are not settable on serverless.
* Spark ML on serverless runs through Spark Connect. Coverage of `KMeans`/`ALS` there was not checked.

## Steps for the owner
```bash
# 1. Create the Free Edition workspace in the browser, then:
databricks auth login --host https://<workspace>.cloud.databricks.com       # creates a profile
# 2. Create the volume and upload the raw files (~7 GB for the full Twitch file; use 100k_a.csv first)
databricks volumes create workspace bronze raw MANAGED
databricks fs cp data/raw/twitch/full_a.csv.gz dbfs:/Volumes/workspace/bronze/raw/twitch/
databricks fs cp -r data/raw/netflix dbfs:/Volumes/workspace/bronze/raw/netflix
databricks fs cp -r data/raw/imdb dbfs:/Volumes/workspace/bronze/raw/imdb
# 3. Validate and deploy the bundle, run the job
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run streaming_content_batch -t dev
```
The catalog and schemas (`bronze`, `silver`, `gold`) must exist: `CREATE SCHEMA IF NOT EXISTS workspace.bronze` etc. A `.gz` CSV is read by a single task; on
serverless the same one-pass load as `twitch/bronze.py` applies, and the Lakeflow pipeline (`pipelines/twitch_pipeline.py`) reads the volume with Auto Loader.

## Scaling down if the quota forces it
In order of preference, each documented in the README if used: (1) the 100k-viewer sample (`100k_a.csv`, 3.05M rows) end to end;
(2) the full file but only viewers with `pmod(hash(user_id), 10) = 0` (about 47M rows); (3) skip the ALS variant. The code paths are identical.

## What the pipeline file does not prove
`pipelines/twitch_pipeline.py` uses the `pyspark.pipelines` decorators. Locally only a wiring test runs (a stub of the decorator API checks that the
datasets and expectations are registered). The transformation functions it calls are the same ones tested with Spark in `tests/`.
The Databricks Asset Bundle YAML is checked for internal consistency by `tests/test_pipeline_wiring.py`; `databricks bundle validate` needs credentials
and runs in CI only when `DATABRICKS_VALIDATE=true` and secrets are set.
