"""Viewer segmentation with Spark MLlib: scaling, KMeans, k selection, stability, profiling."""

from __future__ import annotations

import numpy as np
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from streaming_content.spark import cached, uncache


def adjusted_rand_index(contingency: np.ndarray) -> float:
    """ARI from a contingency table (rows: labelling A, cols: labelling B)."""
    c = np.asarray(contingency, dtype=np.float64)
    n = c.sum()
    comb2 = lambda x: x * (x - 1) / 2  # noqa: E731
    sum_ij = comb2(c).sum()
    sum_a = comb2(c.sum(axis=1)).sum()
    sum_b = comb2(c.sum(axis=0)).sum()
    expected = sum_a * sum_b / comb2(n)
    max_index = (sum_a + sum_b) / 2
    return float((sum_ij - expected) / (max_index - expected)) if max_index != expected else 1.0


def contingency(df: DataFrame, a: str, b: str) -> np.ndarray:
    rows = df.groupBy(a, b).count().collect()
    ia = {v: i for i, v in enumerate(sorted({r[a] for r in rows}))}
    ib = {v: i for i, v in enumerate(sorted({r[b] for r in rows}))}
    m = np.zeros((len(ia), len(ib)))
    for r in rows:
        m[ia[r[a]], ib[r[b]]] = r["count"]
    return m


def make_pipeline(cols: list[str], k: int, seed: int = 42, max_iter: int = 30) -> Pipeline:
    return Pipeline(
        stages=[
            VectorAssembler(inputCols=cols, outputCol="raw_features", handleInvalid="error"),
            StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True),
            KMeans(k=k, seed=seed, maxIter=max_iter, featuresCol="features", predictionCol="segment"),
        ]
    )


def fit_best_of(df: DataFrame, cols: list[str], k: int, seeds: tuple[int, ...] = (42, 43, 44), max_iter: int = 30) -> PipelineModel:
    """KMeans restarts: MLlib has no n_init, so fit one model per seed on the same scaled features and
    keep the lowest within-cluster cost. Single-seed KMeans on these skewed features lands in different
    local optima (seed-only ARI was ~0.6 before this), which made every stability number look worse than
    the structure really is, or by luck better."""
    prep = Pipeline(stages=make_pipeline(cols, k, seeds[0], max_iter).getStages()[:2]).fit(df)
    scaled = cached(prep.transform(df).select("user_id", "features"))
    best, best_cost = None, float("inf")
    for s in seeds:
        km = KMeans(k=k, seed=s, maxIter=max_iter, featuresCol="features", predictionCol="segment").fit(scaled)
        if km.summary.trainingCost < best_cost:
            best, best_cost = km, km.summary.trainingCost
    uncache(scaled)
    return PipelineModel(stages=[*prep.stages, best])


def silhouette(pred: DataFrame) -> float:
    return ClusteringEvaluator(
        featuresCol="features", predictionCol="segment", metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    ).evaluate(pred)


def select_k(
    feats: DataFrame, cols: list[str], ks: list[int], sample_frac: float = 0.02, seed: int = 42,
    n_boot: int = 3,
) -> list[dict]:
    """Silhouette and bootstrap stability (ARI between models fit on independent resamples).

    The sweep runs on a sample of viewers (silhouette is O(n*k) per evaluation and the sample is
    large enough that estimates are stable); the final model is refit on all viewers.
    """
    base = cached(feats.sample(False, sample_frac, seed).select("user_id", *cols))
    hold = cached(feats.sample(False, sample_frac, seed + 1).select("user_id", *cols))
    out = []
    for k in ks:
        full = fit_best_of(base, cols, k, (seed, seed + 1, seed + 2))
        sil = silhouette(full.transform(hold))
        labels = []
        for b in range(n_boot):
            boot = base.sample(True, 1.0, seed + 100 + b)
            m = fit_best_of(boot, cols, k, tuple(seed + 10 * (b + 1) + i for i in range(3)))
            labels.append(m.transform(hold).select("user_id", F.col("segment").alias(f"s{b}")))
        aris = []
        for i in range(n_boot):
            for j in range(i + 1, n_boot):
                joined = labels[i].join(labels[j], "user_id")
                aris.append(adjusted_rand_index(contingency(joined, f"s{i}", f"s{j}")))
        out.append({"k": k, "silhouette": float(sil), "bootstrap_ari_mean": float(np.mean(aris)),
                    "bootstrap_ari_min": float(np.min(aris)), "n_sample": base.count()})
    return out


def fit_final(feats: DataFrame, cols: list[str], k: int, seed: int = 42) -> PipelineModel:
    return fit_best_of(feats, cols, k, (seed, seed + 1, seed + 2))


def profile(assigned: DataFrame, cols: list[str]) -> DataFrame:
    """Per-segment size, share of watch time and mean of each raw feature (+ hours/sessions)."""
    extra = ["hours", "n_sessions", "active_days", "n_streamers"]
    aggs = [F.avg(c).alias(c) for c in cols + extra]
    return (
        assigned.groupBy("segment")
        .agg(F.count("*").alias("n_viewers"), F.sum("hours").alias("total_hours"), *aggs)
        .withColumn("viewer_share", F.col("n_viewers") / F.sum("n_viewers").over(Window.partitionBy()))
        .withColumn("hours_share", F.col("total_hours") / F.sum("total_hours").over(Window.partitionBy()))
        .orderBy("segment")
    )
