"""Viewer re-engagement ("churn") model: will a viewer watch again in the following 7 days?

Distributed supervised ML on the full interaction table, strictly temporal:
  * features use only intervals that START before the cutoff (slot cutoff = end of the feature window);
  * label = any viewing with start in [cutoff, cutoff + 7 days);
  * train: features weeks 0-2 -> label week 3; test: features weeks 1-3 -> label week 4 (window shifted by one week);
  * viewers are split by hash: 80% train viewers, 20% test viewers, so no viewer appears on both sides.
Spark MLlib Pipelines: logistic regression baseline and GBTClassifier, tuned with TrainValidationSplit on a
viewer sample; final fit on all training viewers.
"""

from __future__ import annotations

import json

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from streaming_content import config
from streaming_content.spark import cached, uncache
from streaming_content.timing import timed

WEEK = 7 * config.SLOTS_PER_DAY  # 1008 slots
TRAIN = {"feat_lo": 0, "cutoff": 3 * WEEK}  # label week 3
TEST = {"feat_lo": 1 * WEEK, "cutoff": 4 * WEEK}  # label week 4
TEST_VIEWER_BUCKETS = 20  # hash buckets out of 100 held out for testing
FEATURES = [
    "recency_days", "tenure_days", "log_hours", "log_sessions", "log_active_days", "sessions_per_active_day",
    "mean_session_min", "log_streamers", "top1_share", "recent_hours_share", "last_week_hours", "last_week_active_days",
    "share_night", "share_evening", "max_gap_days",
]


def viewer_bucket(user_col: str = "user_id"):
    return F.pmod(F.hash(F.col(user_col), F.lit(7)), F.lit(100))


def build_features(sessionized: DataFrame, feat_lo: int, cutoff: int) -> DataFrame:
    """One row per viewer active in [feat_lo, cutoff); uses only rows with start_slot in that range."""
    v = sessionized.filter((F.col("start_slot") >= feat_lo) & (F.col("start_slot") < cutoff) & (F.col("new_watch_slots") > 0))
    day = (F.col("start_slot") / config.SLOTS_PER_DAY).cast("int")
    hour = ((F.col("start_slot") % config.SLOTS_PER_DAY) / 6).cast("int")
    # an interval that starts before the cutoff but runs past it only counts up to the cutoff
    clip = (F.least(F.col("stop_slot"), F.lit(cutoff)) - F.col("start_slot")) / (F.col("stop_slot") - F.col("start_slot"))
    w = F.col("new_watch_slots") * clip
    last_week = F.col("start_slot") >= cutoff - WEEK
    v = v.withColumn("day", day)
    base = v.groupBy("user_id").agg(
        F.max(F.least(F.col("stop_slot"), F.lit(cutoff))).alias("last_stop"),
        F.min("start_slot").alias("first_start"),
        F.sum(w).alias("slots"),
        F.countDistinct("session_id").alias("n_sessions"),
        F.countDistinct("day").alias("active_days"),
        F.countDistinct("streamer_id").alias("n_streamers"),
        F.sum(F.when(last_week, w).otherwise(0)).alias("lw_slots"),
        F.countDistinct(F.when(last_week, F.col("day"))).alias("lw_days"),
        F.sum(F.when((hour < 6), w).otherwise(0)).alias("night_slots"),
        F.sum(F.when((hour >= 18), w).otherwise(0)).alias("evening_slots"),
    )
    top = (
        v.groupBy("user_id", "streamer_id").agg(F.sum(w).alias("s"))
        .groupBy("user_id").agg(F.max("s").alias("top_s"))
    )
    # largest gap (in days) between consecutive active days: breaks in routine are a churn signal
    days = v.select("user_id", "day").distinct()
    wd = Window.partitionBy("user_id").orderBy("day")
    gaps = days.withColumn("gap", F.col("day") - F.lag("day").over(wd)).groupBy("user_id").agg(F.coalesce(F.max("gap"), F.lit(0)).alias("max_gap"))
    sess_len = v.groupBy("user_id", "session_id").agg(F.sum(w).alias("l")).groupBy("user_id").agg(F.avg("l").alias("mean_session_slots"))
    f = base.join(top, "user_id").join(gaps, "user_id").join(sess_len, "user_id")
    hours = F.col("slots") * config.SLOT_MINUTES / 60.0
    return f.select(
        "user_id",
        ((cutoff - F.col("last_stop")) / config.SLOTS_PER_DAY).alias("recency_days"),
        ((cutoff - F.col("first_start")) / config.SLOTS_PER_DAY).alias("tenure_days"),
        F.log1p(hours).alias("log_hours"),
        F.log1p("n_sessions").alias("log_sessions"),
        F.log1p("active_days").alias("log_active_days"),
        (F.col("n_sessions") / F.col("active_days")).alias("sessions_per_active_day"),
        (F.col("mean_session_slots") * config.SLOT_MINUTES).alias("mean_session_min"),
        F.log1p("n_streamers").alias("log_streamers"),
        (F.col("top_s") / F.col("slots")).alias("top1_share"),
        (F.col("lw_slots") / F.col("slots")).alias("recent_hours_share"),
        (F.col("lw_slots") * config.SLOT_MINUTES / 60.0).alias("last_week_hours"),
        F.col("lw_days").cast("double").alias("last_week_active_days"),
        (F.col("night_slots") / F.col("slots")).alias("share_night"),
        (F.col("evening_slots") / F.col("slots")).alias("share_evening"),
        F.col("max_gap").cast("double").alias("max_gap_days"),
    )


def build_labels(sessionized: DataFrame, cutoff: int) -> DataFrame:
    """Viewers with any interval starting in [cutoff, cutoff + 7 days)."""
    return (
        sessionized.filter((F.col("start_slot") >= cutoff) & (F.col("start_slot") < cutoff + WEEK))
        .select("user_id").distinct().withColumn("label", F.lit(1.0))
    )


def dataset(sessionized: DataFrame, spec: dict, test: bool) -> DataFrame:
    feats = build_features(sessionized, spec["feat_lo"], spec["cutoff"])
    lab = build_labels(sessionized, spec["cutoff"])
    bucket = viewer_bucket()
    side = (bucket < TEST_VIEWER_BUCKETS) if test else (bucket >= TEST_VIEWER_BUCKETS)
    return feats.filter(side).join(lab, "user_id", "left").fillna({"label": 0.0})


def prep_stages(scale: bool) -> list:
    st = [VectorAssembler(inputCols=FEATURES, outputCol="raw", handleInvalid="error")]
    if scale:
        st.append(StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True))
    else:
        st[0].setOutputCol("features")
    return st


def metrics(pred: DataFrame) -> dict:
    """ROC-AUC, PR-AUC, Brier, top-decile lift and a 10-bin calibration table (probability = column p)."""
    ev = lambda m: BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName=m).evaluate(pred)  # noqa: E731
    p = pred.select("label", vector_p("probability").alias("p"))
    n = p.count()
    base = p.agg(F.avg("label")).first()[0]
    brier = p.select(F.avg((F.col("p") - F.col("label")) ** 2)).first()[0]
    ranked = p.withColumn("decile", F.ntile(10).over(Window.orderBy(F.desc("p"))))
    cal = ranked.groupBy("decile").agg(F.avg("p").alias("mean_pred"), F.avg("label").alias("rate"), F.count("*").alias("n")).orderBy("decile").collect()
    top = cal[0]["rate"]
    riskiest = cal[-1]["rate"]  # lowest predicted return probability = highest churn risk
    return {
        "n": n, "positive_rate": base, "roc_auc": ev("areaUnderROC"), "pr_auc": ev("areaUnderPR"), "brier": brier,
        "brier_baseline_constant": base * (1 - base), "lift_top_decile": top / base, "precision_top_decile": top,
        "churn_rate": 1 - base, "churn_lift_top_risk_decile": (1 - riskiest) / (1 - base), "churn_precision_top_risk_decile": 1 - riskiest,
        "calibration_deciles": [{"decile": r["decile"], "mean_pred": r["mean_pred"], "observed_rate": r["rate"], "n": r["n"]} for r in cal],
    }


def vector_p(col: str):
    """P(label=1) from an MLlib probability vector (works on Spark Connect via vector_to_array)."""
    from pyspark.ml.functions import vector_to_array

    return vector_to_array(F.col(col))[1]


def run(spark: SparkSession, tune_fraction: float = 0.1) -> dict:
    s = spark.table(config.table("silver", "twitch_view_sessions"))
    res: dict = {"train_spec": TRAIN, "test_spec": TEST, "features": FEATURES, "test_viewer_bucket_pct": TEST_VIEWER_BUCKETS,
                 "label": "any viewing with start in [cutoff, cutoff + 1008 slots)"}
    with timed("churn_build_datasets") as t:
        train = dataset(s, TRAIN, test=False)
        test = dataset(s, TEST, test=True)
        train.write.format("delta").mode("overwrite").saveAsTable(config.table("gold", "twitch_churn_train"))
        test.write.format("delta").mode("overwrite").saveAsTable(config.table("gold", "twitch_churn_test"))
        train = cached(spark.table(config.table("gold", "twitch_churn_train")))
        test = cached(spark.table(config.table("gold", "twitch_churn_test")))
        res["n_train"], res["n_test"] = train.count(), test.count()
        t.update(n_train=res["n_train"], n_test=res["n_test"])
    res["train_positive_rate"] = train.agg(F.avg("label")).first()[0]

    tune = train.sample(False, tune_fraction, 3)
    with timed("churn_logreg") as t:
        lr = LogisticRegression(maxIter=50, labelCol="label", featuresCol="features")
        grid = ParamGridBuilder().addGrid(lr.regParam, [0.0, 0.01, 0.1]).build()
        # tune on a sample (explicit loop over the grid), refit on all training viewers
        best_reg, best_auc = 0.0, -1.0
        tr, va = tune.randomSplit([0.75, 0.25], seed=1)
        for g in grid:
            m = Pipeline(stages=prep_stages(True) + [LogisticRegression(maxIter=50, regParam=g[lr.regParam])]).fit(tr)
            auc = BinaryClassificationEvaluator(metricName="areaUnderPR").evaluate(m.transform(va))
            if auc > best_auc:
                best_reg, best_auc = g[lr.regParam], auc
        lr_model = Pipeline(stages=prep_stages(True) + [LogisticRegression(maxIter=50, regParam=best_reg)]).fit(train)
        res["logreg"] = {"best_regParam": best_reg, "tuning_pr_auc": best_auc, "tune_sample_rows": tune.count()}
        res["logreg"]["test"] = metrics(lr_model.transform(test))
        t["seconds_note"] = "includes tuning, final fit and test metrics"

    with timed("churn_gbt") as t:
        gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=40, seed=7)
        grid = ParamGridBuilder().addGrid(gbt.maxDepth, [4, 6]).addGrid(gbt.stepSize, [0.1, 0.2]).build()
        tr, va = tune.randomSplit([0.75, 0.25], seed=2)
        pre = Pipeline(stages=prep_stages(False)).fit(tr)
        tvs = TrainValidationSplit(
            estimator=gbt, estimatorParamMaps=grid, trainRatio=0.75, seed=2, parallelism=1,
            evaluator=BinaryClassificationEvaluator(metricName="areaUnderPR"),
        )
        tvs_model = tvs.fit(pre.transform(tune))
        bp = max(zip(tvs_model.validationMetrics, grid), key=lambda x: x[0])
        best = {k.name: v for k, v in bp[1].items()}
        res["gbt"] = {"best_params": best, "tuning_pr_auc": bp[0], "grid_size": len(grid), "tune_sample_rows": tune.count()}
        final = GBTClassifier(labelCol="label", featuresCol="features", maxIter=40, seed=7, maxDepth=best["maxDepth"], stepSize=best["stepSize"])
        gbt_model = Pipeline(stages=prep_stages(False) + [final]).fit(train)
        res["gbt"]["test"] = metrics(gbt_model.transform(test))
        res["gbt"]["feature_importance"] = sorted(
            zip(FEATURES, [float(x) for x in gbt_model.stages[-1].featureImportances.toArray()]), key=lambda x: -x[1])

    res["mlflow"] = log_mlflow(res, gbt_model, lr_model)
    uncache(train)
    uncache(test)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "twitch_churn.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    return res


def log_mlflow(res: dict, gbt_model, lr_model) -> dict:
    """Track both models; register the GBT in Unity Catalog when on Databricks. Never fails the job."""
    out: dict = {}
    try:
        import mlflow

        from streaming_content.spark import on_databricks

        if on_databricks():
            mlflow.set_registry_uri("databricks-uc")
            mlflow.set_experiment("/Shared/twitch-reengagement")
        else:
            mlflow.set_tracking_uri(f"file:{config.ROOT / 'data' / 'mlruns'}")
            mlflow.set_experiment("twitch-reengagement")
        for name, key in (("logreg", "logreg"), ("gbt", "gbt")):
            with mlflow.start_run(run_name=name) as run:
                m = res[key]["test"]
                mlflow.log_params({"model": name, "n_train": res["n_train"], "n_test": res["n_test"], **{k: str(v) for k, v in res[key].get("best_params", {"regParam": res[key].get("best_regParam")}).items()}})
                mlflow.log_metrics({"test_roc_auc": m["roc_auc"], "test_pr_auc": m["pr_auc"], "test_lift_top_decile": m["lift_top_decile"], "test_brier": m["brier"]})
                out[name] = {"run_id": run.info.run_id}
        return out
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:300]}", **out}

