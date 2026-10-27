"""Netflix end-to-end: parse -> silver -> IMDb match -> cohort -> score -> life curves -> value index.

Writes Delta tables when a SparkSession is given (bronze/silver/gold), JSON results to
reports/results/, figures to reports/figures/, and logs the models to MLflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from streaming_content import config
from streaming_content.netflix import features, imdb, parse, score, silver
from streaming_content.netflix import lifecurves as L
from streaming_content.netflix.parse import SEMESTERS
from streaming_content.timing import timed

HORIZON = 4  # value index horizon, in semesters since release (~24 months)
SHAPE_TRAIN_REL = ["2023H1", "2023H2"]
SHAPE_TRAIN_UNTIL = "2024H2"
VALIDATION_COHORTS = ["2024H1", "2024H2"]
TEST_SEMESTERS = ["2025H1", "2025H2", "2026H1"]


def _save(name: str, obj: dict) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float) + "\n")


def _to_delta(spark, df: pd.DataFrame, layer: str, name: str) -> None:
    if spark is None:
        return
    d = df.copy()
    for c in d.columns:  # Spark cannot infer object columns holding mixed None/NaN/pd.NA
        if d[c].dtype == object or str(d[c].dtype) == "string":
            d[c] = d[c].map(lambda x: None if pd.isna(x) else str(x)).astype(object)
        elif d[c].dtype.kind == "M":
            d[c] = pd.to_datetime(d[c]).astype("datetime64[us]")
    spark.createDataFrame(d).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        config.table(layer, name)
    )


def match_rates(meta: pd.DataFrame, silv: pd.DataFrame) -> dict:
    m = meta.merge(silv.drop_duplicates("title_key")[["title_key", "season_no", "content_type"]], on="title_key")
    m["is_series"] = m["season_no"].notna()
    return {
        "n_titles": int(len(m)),
        "matched": int(m["tconst"].notna().sum()),
        "match_rate": float(m["tconst"].notna().mean()),
        "match_rate_series_seasons": float(m.loc[m["is_series"], "tconst"].notna().mean()),
        "match_rate_other": float(m.loc[~m["is_series"], "tconst"].notna().mean()),
        "reasons": m["match_reason"].value_counts().to_dict(),
    }


def validate_value_index(curves: pd.DataFrame) -> tuple[dict, dict]:
    """Holdout of the life-curve forecasts: shapes fitted on early cohorts / early semesters only."""
    train = curves[curves["rel_sem"].isin(SHAPE_TRAIN_REL)
                   & (curves["semester"].map(SEMESTERS.index) <= SEMESTERS.index(SHAPE_TRAIN_UNTIL))]
    shapes = L.fit_all_shapes(train)
    test = curves[curves["rel_sem"].isin(VALIDATION_COHORTS)]
    methods = ["power", "exponential", "median_ratio", "persistence"]
    out: dict = {"horizon_semesters": HORIZON, "shape_train": {"cohorts": SHAPE_TRAIN_REL, "observations_until": SHAPE_TRAIN_UNTIL},
                 "test_cohorts": VALIDATION_COHORTS}
    for m_obs in (1, 2):
        rows = []
        for tk, g in test.groupby("title_key"):
            g = g.sort_values("k")
            obs = g[g["k"] < m_obs]
            if len(obs) < m_obs:
                continue
            real = g[(g["k"] >= m_obs) & (g["k"] < HORIZON)]["views"].sum()
            alive = bool((g["k"] == HORIZON - 1).any())
            r = {"title_key": tk, "realized": real, "survived": alive}
            for meth in methods:
                r[meth] = L.forecast_views(obs, g["release_date"].iloc[0], g["rel_sem"].iloc[0], g["kind"].iloc[0], shapes, meth, HORIZON)
            rows.append(r)
        d = pd.DataFrame(rows)
        block: dict = {"n_titles": len(d), "dropout_share": float(1 - d["survived"].mean())}
        for label, sub in (("all", d), ("survivors", d[d["survived"]])):
            sub = sub[sub["realized"] > 0]
            block[label] = {
                meth: {
                    "spearman": float(spearmanr(sub["realized"], sub[meth]).statistic),
                    "median_ape": float(np.median(np.abs(sub[meth] - sub["realized"]) / sub["realized"])),
                    "mean_log_ratio": float(np.mean(np.log(sub[meth] / sub["realized"]))),
                }
                for meth in methods
            }
        out[f"observed_{m_obs}_semester"] = block
    return out, shapes


def value_index(curves: pd.DataFrame, shapes: dict, method: str = "power") -> pd.DataFrame:
    """Cumulative views over HORIZON semesters: observed + forecast, then relative to the kind median."""
    rows = []
    for tk, g in curves.groupby("title_key"):
        g = g.sort_values("k")
        # titles need a contiguous start at k=0; semesters beyond the data are forecast
        last_k = int(g["k"].max())
        observed = float(g[g["k"] < HORIZON]["views"].sum())
        forecast = L.forecast_views(g[g["k"] <= min(last_k, HORIZON - 1)], g["release_date"].iloc[0], g["rel_sem"].iloc[0],
                                    g["kind"].iloc[0], shapes, method, HORIZON) if last_k < HORIZON - 1 else 0.0
        rows.append({"title_key": tk, "title": g["title_raw"].iloc[0], "kind": g["kind"].iloc[0], "rel_sem": g["rel_sem"].iloc[0],
                     "n_semesters_observed": int(min(last_k + 1, HORIZON)), "observed_views": observed,
                     "forecast_views": forecast, "value_views": observed + forecast})
    v = pd.DataFrame(rows)
    v["value_index"] = v["value_views"] / v.groupby("kind")["value_views"].transform("median")
    v["fully_observed"] = v["n_semesters_observed"] == HORIZON
    return v.sort_values("value_views", ascending=False)


def run(raw_dir: Path | None = None, spark=None, mlflow_tracking: bool = True) -> dict:
    raw = Path(raw_dir or config.RAW)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    with timed("netflix_total") as t:
        bronze = parse.parse_all(raw / "netflix")
        _to_delta(spark, bronze, "bronze", "netflix_engagement")
        silv = silver.build_silver(bronze)
        _to_delta(spark, silv, "silver", "netflix_titles")
        res["volumes"] = {
            "bronze_rows": int(len(bronze)), "semesters": len(SEMESTERS),
            "titles_with_release_date": int(silv.loc[silv["release_date"].notna(), "title_key"].nunique()),
            "views_imputed_from_hours_over_runtime": int(silv["views_imputed"].sum()),
        }

        basics = imdb.load_basics(raw / "imdb" / "title.basics.tsv.gz")
        index = imdb.build_index(basics)
        # match first: season episode counts need the matched parent ids
        meta0 = features.build_meta(silv, basics, pd.DataFrame(columns=["parentTconst", "seasonNumber", "n_episodes"]),
                                    lambda b, a, y, s: imdb.match_title(b, a, y, s, index))
        eps = imdb.season_episode_counts(raw / "imdb" / "title.episode.tsv.gz", set(meta0["tconst"].dropna()))
        meta = features.build_meta(silv, basics, eps, lambda b, a, y, s: imdb.match_title(b, a, y, s, index))
        res["imdb_match"] = match_rates(meta, silv)
        _to_delta(spark, meta, "silver", "netflix_imdb_match")

        cohort = features.build_cohort(silv, meta)
        res["cohort"] = {"n_titles": int(len(cohort)), "by_release_semester": cohort.groupby("rel_sem").size().to_dict(),
                         "by_kind": cohort["kind"].value_counts().to_dict(),
                         "imdb_matched_share": float(cohort["imdb_matched"].mean())}
        _to_delta(spark, cohort.drop(columns=["title_alt"], errors="ignore"), "gold", "netflix_cohort")

        # ---- performance score
        folds, preds = {}, []
        for sem in TEST_SEMESTERS:
            r = score.evaluate_fold(cohort, sem)
            test = r["_test"].assign(pred=r["_pred"], lo=r["_pred"] - r["conformal"]["halfwidth_log"],
                                     hi=r["_pred"] + r["conformal"]["halfwidth_log"], fold=sem)
            preds.append(test)
            folds[sem] = {k: v for k, v in r.items() if not k.startswith("_")}
            if sem == TEST_SEMESTERS[-1]:
                last = r
        res["score"] = {"target": "log(views in release semester)", "folds": folds, "primary_fold": TEST_SEMESTERS[-1]}
        pred_df = pd.concat(preds)
        res["score"]["shap_top"] = shap_importance(last)
        res["score"]["mlflow"] = log_mlflow(last, folds[TEST_SEMESTERS[-1]]) if mlflow_tracking else None
        keep = ["title_key", "title_raw", "kind", "primary_genre", "rel_sem", "release_date", "season_no", "y", "pred", "lo", "hi", "fold",
                "prior_season_log_views"]
        _to_delta(spark, pred_df[keep], "gold", "netflix_performance_score")
        pred_df[keep].to_csv(config.RESULTS_DIR / "netflix_performance_score.csv", index=False)

        # ---- life curves + value index
        curves = L.build_curve_table(silv)
        _to_delta(spark, curves, "gold", "netflix_life_curves")
        res["life_curves"] = curve_summary(curves)
        validation, _ = validate_value_index(curves)
        res["value_index_validation"] = validation
        shapes_all = L.fit_all_shapes(curves)
        res["life_curve_shapes_all_data"] = shapes_all
        vi = value_index(curves, shapes_all)
        _to_delta(spark, vi, "gold", "netflix_value_index")
        vi.head(200).to_csv(config.RESULTS_DIR / "netflix_value_index_top200.csv", index=False)
        res["value_index"] = {"horizon_semesters": HORIZON, "n_titles": int(len(vi)),
                              "n_fully_observed": int(vi["fully_observed"].sum()),
                              "top10": vi.head(10)[["title", "kind", "rel_sem", "value_views", "value_index"]].to_dict("records")}
        t["titles_in_cohort"] = int(len(cohort))
    _save("netflix", res)
    return res


def curve_summary(curves: pd.DataFrame) -> dict:
    out: dict = {"n_curve_points": int(len(curves)), "n_titles": int(curves["title_key"].nunique()),
                 "share_points_at_reporting_floor": float(curves["at_floor"].mean()), "by_kind": {}}
    for kind, g in curves.groupby("kind"):
        tau = L.best_tau(g)
        b = L.fit_shape(g, "exponential")
        gam = L.fit_shape(g, "power", tau)
        d = g[~g["at_floor"]].groupby("title_key").filter(lambda x: len(x) >= 2)
        sse = {}
        for fam, shape in (("exponential", b), ("power", gam)):
            x = L._basis(d["t_mid"].to_numpy(), fam, tau)
            xc = x - d.assign(x=x).groupby("title_key")["x"].transform("mean").to_numpy()
            yc = (d["log_rate"] - d.groupby("title_key")["log_rate"].transform("mean")).to_numpy()
            sse[fam] = float(((yc + shape * xc) ** 2).mean())
        out["by_kind"][kind] = {
            "n_titles": int(g["title_key"].nunique()), "exp_rate_per_day": b,
            "exp_half_life_days": float(np.log(2) / b) if b > 0 else None, "power_gamma": gam, "power_tau_days": tau,
            "within_title_mse_log_exponential": sse["exponential"], "within_title_mse_log_power": sse["power"],
            "median_log_rate_by_k": g.groupby("k")["log_rate"].median().round(3).to_dict(),
            "n_by_k": g.groupby("k").size().to_dict(),
        }
    return out


def shap_importance(fold_result: dict, top: int = 15) -> list[dict]:
    import shap

    from streaming_content.netflix.score import design

    X, _ = design(fold_result["_test"], fold_result["_cats"])
    sv = shap.TreeExplainer(fold_result["_model"]).shap_values(X)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
    return [{"feature": k, "mean_abs_shap": float(v)} for k, v in imp.head(top).items()]


def log_mlflow(fold_result: dict, metrics: dict) -> dict | None:
    """Log the primary-fold model. On Databricks the registry is Unity Catalog (`databricks-uc`)."""
    try:
        import mlflow
        import mlflow.lightgbm

        from streaming_content.spark import on_databricks

        if on_databricks():
            mlflow.set_registry_uri("databricks-uc")
            name = f"{config.CATALOG or 'workspace'}.gold.netflix_performance_score"
        else:
            mlflow.set_tracking_uri(f"file:{config.ROOT / 'data' / 'mlruns'}")
            name = None
        mlflow.set_experiment("netflix-performance-score" if not on_databricks() else "/Shared/netflix-performance-score")
        with mlflow.start_run(run_name=f"fold-{metrics['test_semester']}") as run:
            mlflow.log_params({**score.LGB_PARAMS, "test_semester": metrics["test_semester"], "n_train": metrics["n_train"]})
            for k, v in metrics["lightgbm"].items():
                mlflow.log_metric(f"test_{k}", v)
            mlflow.log_metric("conformal_coverage", metrics["conformal"]["empirical_coverage"])
            from mlflow.models import infer_signature

            from streaming_content.netflix.score import design

            x_ex, _ = design(fold_result["_test"].head(5), fold_result["_cats"])
            sig = infer_signature(x_ex, fold_result["_model"].predict(x_ex))
            info = mlflow.lightgbm.log_model(fold_result["_model"], "model", registered_model_name=name, signature=sig)
            return {"run_id": run.info.run_id, "registered_model": name, "model_uri": info.model_uri}
    except Exception as e:  # logging must never fail the pipeline
        return {"error": f"{type(e).__name__}: {e}"}
