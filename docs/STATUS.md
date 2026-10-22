# STATUS (written at pause, 2026-10-22)

## Done and committed locally (5 commits, nothing pushed)
Full local pipeline on all 474,676,930 Twitch rows + Netflix/IMDb: bronze/silver/gold Delta, sessionization, viewer features, MLlib segmentation (k=9, best-of-3 restarts),
ALS variant (no gain), Twitch life curves, Netflix score / life curves / value index, tuning experiments (`reports/results/tuning.json`), Streamlit app, docs, tests (22 pass, clean clone checked), CI, bundle YAML.
Correction to an earlier relay: the earlier commit list is stale; `git log` shows 5 commits. `twitch-tuning` did NOT stop without timings: `reports/results/tuning.json` exists and is in results.md section 2.
The only pending reruns: none required. (The segments rerun with restarts finished; names were re-derived with `rename_from_saved_profile`, which the next full run reproduces.)

## Running now
Nothing. No local process, no Databricks job (there is no workspace connection yet; no run IDs).

## Broken / unverified
Everything Databricks-specific is untested (bundle, Lakeflow file, UC registration, MLlib on serverless). Databricks CLI is not installed and there is no profile.

## Partial downloads
None. Raw data complete in `data/raw/` (gitignored); checksums in `ingest/checksums.json`. Re-fetch: `python ingest/download.py all`.

## Exact next steps to resume
1. Get the workspace URL from Wael. Install the official Databricks CLI (github.com/databricks/cli releases), then Wael runs `databricks auth login --host <url>` himself (never handle tokens; never commit `.databrickscfg`).
2. FIRST, a tiny serverless probe job: does MLlib (KMeans, GBTClassifier, CrossValidator, ALS) run, does Structured Streaming `availableNow` work, upload/storage/compute quota for the 6.9 GB gz. Record in `docs/platform.md` (what worked / unavailable / workaround). Cite the Free Edition docs; the quota size is unpublished.
3. Upload raw files to a UC volume; deploy the bundle; run the DAG on Databricks; one codebase (`src/` functions), local = tests only. Use measured counts (474,676,930, never the page's 124M) everywhere.
4. Scope additions from the review session: (a) viewer re-engagement model: features before a cutoff, label = any viewing next 7 days, temporal split, pytest leakage test, MLlib LogisticRegression + GBTClassifier with CrossValidator/TrainValidationSplit, ROC-AUC, PR-AUC, calibration, top-decile lift, MLflow + UC registry; (b) `docs/ab_testing.md`: unit of randomization, metric, CUPED, power/MDE from real Twitch variance, SIMULATED A/A and A/B with injected effect (labelled).
5. Then Databricks features as available: Delta MERGE/time travel (table version in MLflow)/CDF/constraints/liquid clustering/OPTIMIZE/VACUUM, UC comments/tags/lineage, Auto Loader + expectations, Structured Streaming replay (labelled as replay), dev/prod bundle targets, feature tables, UC aliases champion/challenger, SQL warehouse + AI/BI dashboard + Genie, Apps, system tables.
6. Benchmark `reports/local_vs_databricks.md`: 1/10/50/100% volumes, >=3 runs, cold vs warm, p50/p95/p99 latencies; states it describes Free Edition vs one workstation. Tuning confs that serverless forbids stay local.
Priority order from the review session: core (medallion, sessions/features, segmentation, churn GBT, streaming replay, tuning) then Netflix, then extras. No push before the Opus review.
