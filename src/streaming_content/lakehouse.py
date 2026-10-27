"""Databricks lakehouse features applied on top of the tables: Unity Catalog governance, Delta demos, feature tables.

Every function takes a SparkSession and works with the `workspace.<layer>.<table>` naming (config.table).
Results (timings, versions, counts) go to reports/results/lakehouse.json.
"""

from __future__ import annotations

import json
import statistics
import time

from streaming_content import config

T = config.table

COMMENTS = {
    ("bronze", "twitch_interactions"): ("Raw Twitch viewing intervals as loaded from the McAuley-lab files (10-minute slots, 43 days). Source: Rappaz, McAuley, Aberer, RecSys 2021.",
                                       {"user_id": "anonymised viewer id", "stream_id": "broadcast segment id", "streamer": "channel username", "start_slot": "first 10-min slot of presence (inclusive)", "stop_slot": "end slot (exclusive)"}),
    ("silver", "twitch_view_sessions"): ("Validated viewing intervals with per-viewer session_id (gap > 3 slots = 30 min) and union watch time.",
                                        {"session_id": "1-based session counter within a viewer", "new_watch_slots": "slots of this interval not already covered by earlier intervals of the viewer", "streamer_id": "dictionary-encoded channel (silver.twitch_streamers)"}),
    ("gold", "twitch_viewer_features"): ("Per-viewer behavioural features over the full 43-day window (input of the segmentation).", {"hours": "union watch hours", "norm_entropy": "entropy of watch time across channels / log(n channels)", "top1_share": "share of watch time on the most-watched channel"}),
    ("gold", "twitch_viewer_segments"): ("Viewer segment assignment (MLlib KMeans, k chosen by silhouette + bootstrap stability).", {"segment": "cluster id; names in reports/results.md"}),
    ("gold", "netflix_cohort"): ("Netflix titles released since 2023-01-01 with PRE-RELEASE features and the target (log views in the release semester).", {"y": "TARGET: log(views in release semester)", "rel_sem": "release semester"}),
    ("gold", "netflix_performance_score"): ("Out-of-sample predictions with 80% conformal intervals for the three temporal test folds.", {"pred": "predicted log views", "lo": "interval lower bound (log)", "hi": "interval upper bound (log)"}),
    ("gold", "netflix_life_curves"): ("Title-semester points of the exposure-normalised life curves.", {"rate": "views per exposure day", "t_mid": "days since release at the middle of the exposed window"}),
    ("gold", "netflix_value_index"): ("Cumulative views over 4 semesters since release (observed + forecast) and index vs the kind median.", {"value_index": "value_views / median value_views of the same kind"}),
}
TAGS = {"bronze": {"layer": "bronze"}, "silver": {"layer": "silver"}, "gold": {"layer": "gold"}}
CONSTRAINTS = [
    ("silver", "twitch_view_sessions", "positive_duration", "stop_slot > start_slot"),
    ("silver", "twitch_view_sessions", "session_ge_1", "session_id >= 1"),
    ("gold", "twitch_viewer_features", "hours_positive", "hours > 0"),
    ("gold", "netflix_cohort", "views_positive", "views_rel_sem > 0"),
]


def _exists(spark, t: str) -> bool:
    # spark.table() is lazy on Spark Connect and would not raise: ask the catalog instead
    return bool(spark.catalog.tableExists(t))


def governance(spark) -> dict:
    """Table/column comments, tags, NOT NULL and CHECK constraints, change data feed on gold tables."""
    done: dict = {"comments": [], "tags": [], "constraints": [], "skipped": []}
    for (layer, name), (comment, cols) in COMMENTS.items():
        t = T(layer, name)
        if not _exists(spark, t):
            done["skipped"].append(t)
            continue
        spark.sql(f"COMMENT ON TABLE {t} IS '{comment}'")
        present = {f.name for f in spark.table(t).schema.fields}
        for c, text in cols.items():
            if c in present:
                spark.sql(f"ALTER TABLE {t} ALTER COLUMN {c} COMMENT '{text}'")
        spark.sql(f"ALTER TABLE {t} SET TAGS ('layer' = '{layer}', 'project' = 'streaming-content-analytics')")
        done["comments"].append(t)
        done["tags"].append(t)
    for layer, name, cname, expr in CONSTRAINTS:
        t = T(layer, name)
        if not _exists(spark, t):
            continue
        try:
            spark.sql(f"ALTER TABLE {t} ADD CONSTRAINT {cname} CHECK ({expr})")
            done["constraints"].append(f"{t}.{cname}: {expr}")
        except Exception as e:  # noqa: BLE001 - violated by data or other: record, do not hide
            msg = str(e)
            done["constraints"].append(f"{t}.{cname}: {expr} (already present)" if "ALREADY_EXISTS" in msg else f"{t}.{cname}: NOT ADDED ({msg[:120]})")
    for name in ("netflix_cohort", "twitch_viewer_features", "twitch_viewer_segments"):
        t = T("gold", name)
        if _exists(spark, t):
            spark.sql(f"ALTER TABLE {t} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    return done


def delta_demo(spark) -> dict:
    """MERGE upsert, change data feed, time travel and RESTORE on a DEEP CLONE (real tables stay untouched)."""
    src, demo = T("silver", "netflix_titles"), T("silver", "netflix_titles_merge_demo")
    spark.sql(f"CREATE OR REPLACE TABLE {demo} DEEP CLONE {src}")
    spark.sql(f"ALTER TABLE {demo} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    n0 = spark.table(demo).count()
    v0 = int(spark.sql(f"DESCRIBE HISTORY {demo}").first()["version"])
    # late restatement of one semester: 25 corrected rows + 10 brand-new titles, as one MERGE
    spark.sql(f"""CREATE OR REPLACE TEMP VIEW upd AS
        SELECT title_key, semester, views * 1.1 AS views, hours_viewed FROM {demo} WHERE semester = '2026H1' AND views IS NOT NULL LIMIT 25""")
    spark.sql("""CREATE OR REPLACE TEMP VIEW newrows AS
        SELECT concat('demo-new-', cast(id AS string)) AS title_key, '2026H1' AS semester, 1000000.0 AS views, 5000000.0 AS hours_viewed FROM range(10)""")
    t0 = time.perf_counter()
    spark.sql(f"""MERGE INTO {demo} t USING (SELECT * FROM upd UNION ALL SELECT * FROM newrows) s
        ON t.title_key = s.title_key AND t.semester = s.semester
        WHEN MATCHED THEN UPDATE SET t.views = s.views, t.hours_viewed = s.hours_viewed
        WHEN NOT MATCHED THEN INSERT (title_key, semester, views, hours_viewed) VALUES (s.title_key, s.semester, s.views, s.hours_viewed)""")
    merge_s = time.perf_counter() - t0
    n1 = spark.table(demo).count()
    v1 = int(spark.sql(f"DESCRIBE HISTORY {demo}").first()["version"])
    # second identical MERGE: idempotent for the inserted rows (matched now)
    spark.sql(f"""MERGE INTO {demo} t USING newrows s ON t.title_key = s.title_key AND t.semester = s.semester
        WHEN MATCHED THEN UPDATE SET t.views = s.views WHEN NOT MATCHED THEN INSERT (title_key, semester, views, hours_viewed) VALUES (s.title_key, s.semester, s.views, s.hours_viewed)""")
    n2 = spark.table(demo).count()
    cdf = {r["_change_type"]: r["c"] for r in spark.sql(f"SELECT _change_type, count(*) c FROM table_changes('{demo}', {v0 + 1}, {v1}) GROUP BY 1").collect()}
    n_tt = spark.sql(f"SELECT count(*) FROM {demo} VERSION AS OF {v0}").first()[0]
    spark.sql(f"RESTORE TABLE {demo} TO VERSION AS OF {v0}")
    n_restored = spark.table(demo).count()
    out = {"table": demo, "rows_before": n0, "rows_after_merge": n1, "rows_after_second_merge": n2, "merge_seconds": round(merge_s, 2),
           "change_data_feed": cdf, "time_travel_rows_at_v0": n_tt, "rows_after_restore": n_restored, "versions": [v0, v1]}
    return out


def reproduce_training_snapshot(spark, snapshot: dict) -> dict:
    """Reload the cohort exactly as it was when the model was trained (Delta time travel) and check the checksum."""
    t, v = snapshot["table"], snapshot["delta_version"]
    row = spark.sql(f"SELECT count(*) AS n, sum(y) AS s FROM {t} VERSION AS OF {v}").first()
    now = spark.sql(f"SELECT count(*) AS n, sum(y) AS s FROM {t}").first()
    return {"version": v, "rows_at_version": row["n"], "rows_logged": snapshot["rows"], "y_sum_matches_logged": abs(row["s"] - snapshot["y_sum"]) < 1e-6,
            "current_rows": now["n"], "current_version": int(spark.sql(f"DESCRIBE HISTORY {t}").first()["version"])}


def reproduce(spark) -> dict:
    """Time-travel check: the cohort at the Delta version recorded for the model has the logged rows and checksum."""
    snap = json.loads((config.RESULTS_DIR / "netflix.json").read_text())["training_snapshot"]
    return reproduce_training_snapshot(spark, snap)


def lineage(spark) -> dict:
    """Unity Catalog table lineage recorded in system.access.table_lineage for this project's tables."""
    rows = spark.sql(f"""SELECT DISTINCT source_table_full_name AS source, target_table_full_name AS target
        FROM system.access.table_lineage WHERE target_table_catalog = '{config.CATALOG or 'workspace'}'
        AND target_table_schema IN ('silver', 'gold') AND source_table_full_name IS NOT NULL ORDER BY 2, 1""").collect()
    return {"edges": [{"source": r["source"], "target": r["target"]} for r in rows]}


def feature_tables(spark) -> dict:
    """Register viewer and title features as Unity Catalog feature tables (primary key = entity id)."""
    from databricks.feature_engineering import FeatureEngineeringClient

    fe = FeatureEngineeringClient()
    out = {}
    for name, pk, src, desc in (
        ("fs_viewer_behaviour", ["user_id"], T("gold", "twitch_viewer_features"), "Behavioural features per Twitch viewer over the 43-day window"),
        ("fs_netflix_title_prerelease", ["title_key"], T("gold", "netflix_cohort"), "Pre-release features of Netflix titles (+ target column y; use only FEATURES_NUM/CAT for training)"),
    ):
        if not _exists(spark, src):
            out[name] = "source table missing"
            continue
        target = T("gold", name)
        df = spark.table(src)
        if name == "fs_netflix_title_prerelease":
            df = df.select("title_key", "kind", "primary_genre", "log_runtime", "exposure_days", "release_month", "season_no", "is_limited", "has_alt_title", "imdb_matched")
        spark.sql(f"DROP TABLE IF EXISTS {target}")
        fe.create_table(name=target, primary_keys=pk, df=df, description=desc)
        out[name] = {"rows": spark.table(target).count(), "primary_keys": pk}
    return out


def optimize_demo(spark, user_sample: int = 2000, repeats: int = 5) -> dict:
    """Before/after on a point lookup: unclustered Delta table vs OPTIMIZE vs liquid clustering by user_id."""
    src = T("silver", "twitch_view_sessions")
    base = T("silver", "tune_unclustered")
    clus = T("silver", "tune_liquid")
    spark.sql(f"CREATE OR REPLACE TABLE {base} AS SELECT * FROM {src} WHERE pmod(hash(user_id), 10) = 0")
    users = [r[0] for r in spark.sql(f"SELECT DISTINCT user_id FROM {base} LIMIT {user_sample}").collect()]
    spark.createDataFrame([(u,) for u in users], "user_id int").createOrReplaceTempView("lookup_users")

    def q(t):
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            spark.sql(f"SELECT sum(watch_slots) FROM {t} JOIN lookup_users USING (user_id)").collect()
            times.append(time.perf_counter() - t0)
        return {"median_s": round(statistics.median(times), 3), "runs_s": [round(x, 3) for x in times]}

    files = lambda t: int(spark.sql(f"DESCRIBE DETAIL {t}").first()["numFiles"])  # noqa: E731
    out = {"rows": spark.table(base).count(), "lookup_viewers": len(users), "before": q(base), "files_before": files(base)}
    t0 = time.perf_counter()
    spark.sql(f"OPTIMIZE {base}")
    out["optimize_compact_seconds"] = round(time.perf_counter() - t0, 1)
    out["after_optimize_compaction"] = q(base)
    out["files_after_optimize"] = files(base)
    spark.sql(f"CREATE OR REPLACE TABLE {clus} CLUSTER BY (user_id) AS SELECT * FROM {base}")
    t0 = time.perf_counter()
    spark.sql(f"OPTIMIZE {clus}")
    out["optimize_clustering_seconds"] = round(time.perf_counter() - t0, 1)
    out["after_liquid_clustering"] = q(clus)
    out["files_after_clustering"] = files(clus)
    t0 = time.perf_counter()
    spark.sql(f"VACUUM {base}")
    out["vacuum_seconds"] = round(time.perf_counter() - t0, 1)
    return out


def run(spark, parts: tuple[str, ...] = ("governance", "delta_demo", "feature_tables", "reproduce", "lineage")) -> dict:
    res: dict = {}
    for p in parts:
        t0 = time.perf_counter()
        try:
            res[p] = globals()[p](spark)
        except Exception as e:  # noqa: BLE001 - record failures honestly
            res[p] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
        res[p + "_seconds"] = round(time.perf_counter() - t0, 1)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / "lakehouse.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **res}, indent=2, default=float) + "\n")
    return res
