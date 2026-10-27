"""Serverless capability probe: prints a JSON report (what works on Free Edition serverless)."""
import json
import os
import time
import traceback

os.environ.setdefault("SPARKML_TEMP_DFS_PATH", "/Volumes/workspace/bronze/raw/_ml_tmp")
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()
rep = {"environment_version": "5 + SPARKML_TEMP_DFS_PATH", "spark_version": spark.version, "runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION"), "checks": {}}


def check(name):
    def deco(fn):
        t0 = time.time()
        try:
            out = fn()
            rep["checks"][name] = {"ok": True, "seconds": round(time.time() - t0, 2), "detail": out}
        except Exception as e:
            rep["checks"][name] = {"ok": False, "seconds": round(time.time() - t0, 2), "error": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[-700:]}
        print("CHECK", name, json.dumps(rep["checks"][name], default=str)[:600], flush=True)
        return fn
    return deco


df = spark.range(0, 20000).select("id", (F.rand(1)).alias("a"), (F.rand(2) * 3).alias("b"), (F.rand(3) > 0.5).cast("int").alias("label"))


@check("kmeans")
def _():
    from pyspark.ml import Pipeline
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.feature import StandardScaler, VectorAssembler
    m = Pipeline(stages=[VectorAssembler(inputCols=["a", "b"], outputCol="r"), StandardScaler(inputCol="r", outputCol="features"), KMeans(k=3)]).fit(df)
    return m.transform(df).groupBy("prediction").count().count()


@check("bisecting_kmeans")
def _():
    from pyspark.ml.clustering import BisectingKMeans
    from pyspark.ml.feature import VectorAssembler
    v = VectorAssembler(inputCols=["a", "b"], outputCol="features").transform(df)
    return BisectingKMeans(k=3).fit(v).summary.k


@check("clustering_evaluator")
def _():
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.evaluation import ClusteringEvaluator
    from pyspark.ml.feature import VectorAssembler
    v = VectorAssembler(inputCols=["a", "b"], outputCol="features").transform(df)
    return ClusteringEvaluator().evaluate(KMeans(k=3).fit(v).transform(v))


@check("logreg_gbt_crossvalidator")
def _():
    from pyspark.ml.classification import GBTClassifier, LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    v = VectorAssembler(inputCols=["a", "b"], outputCol="features").transform(df)
    ev = BinaryClassificationEvaluator()
    lr = LogisticRegression()
    cv = CrossValidator(estimator=lr, estimatorParamMaps=ParamGridBuilder().addGrid(lr.regParam, [0.0, 0.1]).build(), evaluator=ev, numFolds=2)
    a = ev.evaluate(cv.fit(v).transform(v))
    b = ev.evaluate(GBTClassifier(maxIter=5).fit(v).transform(v))
    return {"lr_auc": a, "gbt_auc": b}


@check("train_validation_split")
def _():
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
    v = VectorAssembler(inputCols=["a", "b"], outputCol="features").transform(df)
    lr = LogisticRegression()
    TrainValidationSplit(estimator=lr, estimatorParamMaps=ParamGridBuilder().addGrid(lr.regParam, [0.0, 0.1]).build(), evaluator=BinaryClassificationEvaluator()).fit(v)
    return "ok"


@check("als")
def _():
    from pyspark.ml.recommendation import ALS
    r = df.select((F.col("id") % 100).alias("u"), (F.col("id") % 37).alias("i"), F.col("a").alias("r"))
    return ALS(userCol="u", itemCol="i", ratingCol="r", implicitPrefs=True, maxIter=3).fit(r).rank


@check("df_cache")
def _():
    df.cache().count()
    return "cache accepted"


for conf in ["spark.sql.shuffle.partitions", "spark.sql.adaptive.enabled", "spark.sql.autoBroadcastJoinThreshold",
             "spark.sql.adaptive.skewJoin.enabled", "spark.executor.memory", "spark.sql.files.maxPartitionBytes"]:
    @check(f"conf_set:{conf}")
    def _(conf=conf):
        cur = spark.conf.get(conf, None)
        spark.conf.set(conf, cur if cur is not None else "200")
        return f"current={cur}"


@check("structured_streaming_available_now")
def _():
    base = "/Volumes/workspace/bronze/raw/_probe"
    os.makedirs(base + "/in", exist_ok=True)
    spark.range(1000).selectExpr("id", "cast(id % 7 as int) as ch", "timestamp_seconds(1700000000 + id) as ts").write.mode("overwrite").json(base + "/in")
    q = (spark.readStream.schema("id long, ch int, ts timestamp").json(base + "/in").withWatermark("ts", "10 minutes")
         .groupBy(F.window("ts", "5 minutes"), "ch").count()
         .writeStream.format("delta").outputMode("append").option("checkpointLocation", base + "/ckpt")
         .trigger(availableNow=True).toTable("workspace.bronze._probe_stream"))
    q.awaitTermination()
    return spark.table("workspace.bronze._probe_stream").count()


@check("delta_features")
def _():
    spark.sql("CREATE OR REPLACE TABLE workspace.bronze._probe_delta (id BIGINT NOT NULL, v DOUBLE) CLUSTER BY (id) TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    spark.range(1000).selectExpr("id", "rand() v").write.mode("append").saveAsTable("workspace.bronze._probe_delta")
    spark.sql("ALTER TABLE workspace.bronze._probe_delta ADD CONSTRAINT v_ok CHECK (v >= 0)")
    spark.sql("MERGE INTO workspace.bronze._probe_delta t USING (SELECT 1 id, 0.5 v) s ON t.id = s.id WHEN MATCHED THEN UPDATE SET v = s.v")
    spark.sql("OPTIMIZE workspace.bronze._probe_delta")
    ver = spark.sql("DESCRIBE HISTORY workspace.bronze._probe_delta").count()
    tt = spark.sql("SELECT count(*) FROM workspace.bronze._probe_delta VERSION AS OF 1").first()[0]
    cdf = spark.sql("SELECT count(*) FROM table_changes('workspace.bronze._probe_delta', 1)").first()[0]
    return {"history_rows": ver, "time_travel_rows": tt, "cdf_rows": cdf}


@check("system_tables")
def _():
    return spark.sql("SELECT count(*) FROM system.billing.usage").first()[0]


print("PROBE_REPORT=" + json.dumps(rep, default=str))
