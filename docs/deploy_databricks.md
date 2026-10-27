# Deploying on Databricks Free Edition

Status: **deployed and run** on a Free Edition workspace (dev target). What worked, what did not and the workarounds: [platform.md](platform.md).

## Steps (as done)
```bash
# 1. Install the official CLI (github.com/databricks/cli releases) and sign in yourself in the browser:
databricks auth login --host https://<workspace>.cloud.databricks.com
# 2. Catalog objects (catalog `workspace` exists on Free Edition)
for s in bronze silver gold lakeflow; do databricks schemas create $s workspace; done
databricks volumes create workspace bronze raw MANAGED
databricks volumes create workspace gold results MANAGED
# 3. Upload raw data. A single 6.9 GB gzip upload fails (unexpected EOF): split into parts first (see below) and upload them 4 at a time.
databricks fs cp -r data/raw/netflix dbfs:/Volumes/workspace/bronze/raw/netflix
databricks fs cp -r data/raw/imdb    dbfs:/Volumes/workspace/bronze/raw/imdb
databricks fs mkdirs dbfs:/Volumes/workspace/bronze/raw/twitch_parts
ls data/raw/twitch_parts/part-*.csv.gz | xargs -P4 -I{} sh -c 'databricks fs cp {} dbfs:/Volumes/workspace/bronze/raw/twitch_parts/$(basename {}) --overwrite'
# 4. Bundle
databricks bundle validate -t dev && databricks bundle deploy -t dev
databricks bundle run sca_twitch -t dev        # bronze -> silver -> features -> segments / churn / A/B / life curves / streaming / OPTIMIZE demo
databricks bundle run sca_lakehouse -t dev     # Netflix + governance + Delta demos + feature tables + app data export
# 5. Results land in /Volumes/workspace/gold/results; copy them into reports/results/databricks and regenerate the report:
databricks fs cp -r dbfs:/Volumes/workspace/gold/results reports/results/databricks --overwrite
python reports/make_results.py
```
Parts were produced locally from the bronze table (`spark.table(...).repartition(24).write.option("compression","gzip").csv(...)`, 8.7 GB); the original `full_a.csv.gz` is 6.9 GB.

Extras created by scripts: `jobs/create_dashboard.py` (AI/BI dashboard), `jobs/create_genie_space.py`, `jobs/genie_latency.py`, `jobs/serving_latency.py` (needs the endpoint `sca-netflix-score`, created with `databricks api post /api/2.0/serving-endpoints`),
the App (`app/`, deployed with `databricks apps create` + `databricks workspace import-dir` + `databricks apps deploy`; `app/app_data` comes from the `export-app-data` stage and is git-ignored).

## If the daily quota runs out
Scale down in this order and say so in the README: (1) the 100k-viewer sample (`100k_a.csv`); (2) the full file but only viewers with `pmod(hash(user_id), 10) = 0` (about 47M rows); (3) skip the ALS variant. The code paths are identical. No quota error occurred in the runs reported here, but the quota size is not published.

## Not covered
No `prod` target exists in `databricks.yml`; the Lakeflow pipeline ran on the sample only; the `sca_*` jobs are not scheduled.
