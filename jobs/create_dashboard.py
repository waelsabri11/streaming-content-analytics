"""Create and publish the AI/BI (Lakeview) dashboard over the gold tables via the Databricks CLI.

Usage: python jobs/create_dashboard.py [--catalog workspace]. Requires an authenticated CLI profile.
"""

import argparse
import json
import subprocess

DATASETS = {
    "segments": ("Viewer segments", "SELECT segment, count(*) AS viewers, round(sum(hours), 0) AS total_hours, round(avg(hours), 1) AS avg_hours, round(avg(top1_share), 2) AS top1_share, round(avg(share_night), 2) AS clock_00_06h, round(avg(share_evening), 2) AS clock_18_24h FROM {c}.gold.twitch_viewer_segments GROUP BY segment ORDER BY segment"),
    "netflix_curve": ("Netflix life curve", "SELECT kind, k AS semesters_after_release_semester, round(percentile_approx(rate, 0.5), 0) AS median_views_per_exposure_day, count(*) AS n FROM {c}.gold.netflix_life_curves WHERE NOT at_floor GROUP BY 1, 2 ORDER BY 1, 2"),
    "score": ("Performance score coverage", "SELECT fold AS test_semester, count(*) AS titles, round(avg(CASE WHEN y BETWEEN lo AND hi THEN 1 ELSE 0 END), 3) AS interval_coverage FROM {c}.gold.netflix_performance_score GROUP BY fold ORDER BY fold"),
    "value": ("Value index top 15", "SELECT title, kind, rel_sem, round(value_views / 1e6, 1) AS value_m_views, round(value_index, 1) AS value_index FROM {c}.gold.netflix_value_index ORDER BY value_views DESC LIMIT 15"),
    "retention": ("Twitch retention", "SELECT cast(r.first_week AS string) AS first_contact_week, r.week - r.first_week AS lag_weeks, round(r.pairs / b.pairs, 3) AS retention FROM {c}.gold.twitch_retention_weeks r JOIN (SELECT first_week, pairs FROM {c}.gold.twitch_retention_weeks WHERE week = first_week) b USING (first_week) WHERE r.week - r.first_week BETWEEN 1 AND 4 AND r.first_week <= 1 ORDER BY 1, 2"),
    "stream": ("Replay: concurrent viewers", "SELECT window_start, concurrent_viewers FROM {c}.gold.twitch_concurrent_viewers_platform_stream ORDER BY window_start"),
}


def field(n):
    return {"name": n, "expression": f"`{n}`"}


def chart(name, ds, kind, x, y, title, color=None, cols=None):
    fields = [field(x), field(y)] + ([field(color)] if color else [])
    enc = {"x": {"fieldName": x, "scale": {"type": "quantitative" if kind == "line" and x != "first_contact_week" and not x.startswith("window") else "categorical"}, "displayName": x},
           "y": {"fieldName": y, "scale": {"type": "quantitative"}, "displayName": y}}
    if x.startswith("window"):
        enc["x"]["scale"] = {"type": "temporal"}
    if color:
        enc["color"] = {"fieldName": color, "scale": {"type": "categorical"}, "displayName": color}
    return {"name": name, "queries": [{"name": "main_query", "query": {"datasetName": ds, "fields": fields, "disaggregated": True}}],
            "spec": {"version": 3, "widgetType": kind, "encodings": enc, "frame": {"showTitle": True, "title": title}}}


def table(name, ds, columns, title):
    return {"name": name, "queries": [{"name": "main_query", "query": {"datasetName": ds, "fields": [field(c) for c in columns], "disaggregated": True}}],
            "spec": {"version": 1, "widgetType": "table", "encodings": {"columns": [{"fieldName": c, "displayName": c} for c in columns]}, "frame": {"showTitle": True, "title": title}}}


def build(catalog: str) -> dict:
    w = [
        (chart("w_seg", "segments", "bar", "segment", "viewers", "Viewers per segment"), (0, 0, 3, 6)),
        (chart("w_seg_h", "segments", "bar", "segment", "total_hours", "Watch hours per segment"), (3, 0, 3, 6)),
        (table("w_seg_t", "segments", ["segment", "viewers", "avg_hours", "top1_share", "clock_00_06h", "clock_18_24h"], "Segment profile"), (0, 6, 6, 6)),
        (chart("w_curve", "netflix_curve", "line", "semesters_after_release_semester", "median_views_per_exposure_day", "Netflix life curve (median views per exposure day; late points are survivors)", color="kind"), (0, 12, 3, 7)),
        (chart("w_ret", "retention", "line", "lag_weeks", "retention", "Twitch viewer-channel retention", color="first_contact_week"), (3, 12, 3, 7)),
        (table("w_score", "score", ["test_semester", "titles", "interval_coverage"], "80% conformal interval coverage"), (0, 19, 3, 5)),
        (table("w_val", "value", ["title", "kind", "rel_sem", "value_m_views", "value_index"], "Value index, top 15"), (3, 19, 3, 8)),
        (chart("w_stream", "stream", "line", "window_start", "concurrent_viewers", "Streaming replay (synthetic time axis): concurrent viewers per 10 min", ), (0, 27, 6, 6)),
    ]
    return {"datasets": [{"name": k, "displayName": d, "queryLines": [q.format(c=catalog)]} for k, (d, q) in DATASETS.items()],
            "pages": [{"name": "p1", "displayName": "Streaming content analytics",
                       "layout": [{"widget": wd, "position": {"x": x, "y": y, "width": ww, "height": h}} for wd, (x, y, ww, h) in w]}]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="workspace")
    ap.add_argument("--warehouse", default="49d284e9ba6c044b")
    ap.add_argument("--path", default="/Workspace/Users/waelsabri4@gmail.com/sca")
    a = ap.parse_args()
    body = {"display_name": "Streaming content analytics", "parent_path": a.path, "warehouse_id": a.warehouse, "serialized_dashboard": json.dumps(build(a.catalog))}
    out = json.loads(subprocess.check_output(["databricks", "lakeview", "create", "--json", json.dumps(body)]))
    print(out["dashboard_id"])
    pub = subprocess.check_output(["databricks", "lakeview", "publish", out["dashboard_id"], "--json", json.dumps({"warehouse_id": a.warehouse, "embed_credentials": True})])
    print(pub.decode()[:300])
