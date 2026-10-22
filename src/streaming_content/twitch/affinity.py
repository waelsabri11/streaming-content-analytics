"""Viewer-channel affinity: implicit ALS channel embeddings, channel families, viewer fold-in.

ALS is trained on a viewer sample and on channels with enough viewers in it. Channel factors are
clustered into "families" (channels that share audiences), and every viewer - not only the ALS
training sample - gets an embedding by folding in: the watch-time-weighted mean of the factors of
the channels they watched (a broadcast join, the factor table is small).
"""

from __future__ import annotations

from pyspark.ml.clustering import KMeans
from pyspark.ml.functions import array_to_vector
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def user_channel_matrix(sessionized: DataFrame) -> DataFrame:
    return (sessionized.groupBy("user_id", "streamer_id").agg(F.sum("new_watch_slots").alias("slots"))
            .filter(F.col("slots") > 0))


def train_als(
    uc: DataFrame, user_frac: float = 0.1, min_viewers: int = 200, rank: int = 16, seed: int = 42
):
    """Implicit ALS on log-watch-time. Returns (model, kept_channels_df, n_users, n_channels)."""
    sample = uc.filter(F.pmod(F.hash("user_id", F.lit(seed)), F.lit(1000)) < int(user_frac * 1000))
    keep = (
        sample.groupBy("streamer_id").agg(F.count("*").alias("viewers")).filter(F.col("viewers") >= min_viewers)
        .withColumn("item", F.dense_rank().over(Window.orderBy("streamer_id")) - 1)
    )
    data = (
        sample.join(F.broadcast(keep.select("streamer_id", "item")), "streamer_id")
        .withColumn("rating", F.log1p("slots").cast("float"))
        .select(F.col("user_id").alias("user"), "item", "rating")
    )
    als = ALS(
        rank=rank, maxIter=8, regParam=0.05, alpha=20.0, implicitPrefs=True, userCol="user", itemCol="item",
        ratingCol="rating", coldStartStrategy="drop", seed=seed, numUserBlocks=24, numItemBlocks=24,
    )
    model = als.fit(data)
    return model, keep, data.select("user").distinct().count(), keep.count()


def channel_families(model, keep: DataFrame, n_families: int = 12, seed: int = 42) -> DataFrame:
    """Cluster ALS item factors (L2-normalised) into channel families."""
    f = model.itemFactors.withColumn("vec", array_to_vector(F.col("features")))
    from pyspark.ml.feature import Normalizer

    f = Normalizer(inputCol="vec", outputCol="nvec", p=2.0).transform(f)
    km = KMeans(k=n_families, seed=seed, featuresCol="nvec", predictionCol="family", maxIter=40).fit(f)
    fam = km.transform(f).select(F.col("id").alias("item"), "family", "features")
    return fam.join(keep.select("item", "streamer_id"), "item")


def fold_in(uc: DataFrame, fam: DataFrame, rank: int = 16) -> DataFrame:
    """Per-viewer embedding (mean of channel factors weighted by log watch time) + family shares."""
    f = F.broadcast(fam.select("streamer_id", "family", "features"))
    j = uc.join(f, "streamer_id").withColumn("wt", F.log1p("slots"))
    emb = j.groupBy("user_id").agg(
        *[F.sum(F.col("wt") * F.col("features")[i]).alias(f"_e{i}") for i in range(rank)],
        F.sum("wt").alias("_w"),
    )
    for i in range(rank):
        emb = emb.withColumn(f"emb_{i}", F.col(f"_e{i}") / F.col("_w")).drop(f"_e{i}")
    return emb.drop("_w")


def family_shares(uc: DataFrame, fam: DataFrame, n_families: int = 12) -> DataFrame:
    f = F.broadcast(fam.select("streamer_id", "family"))
    j = uc.join(f, "streamer_id", "left").withColumn("family", F.coalesce(F.col("family"), F.lit(-1)))
    tot = j.groupBy("user_id").agg(F.sum("slots").alias("_t"))
    shares = j.groupBy("user_id").pivot("family", list(range(-1, n_families))).agg(F.sum("slots"))
    shares = shares.join(tot, "user_id")
    for c in [str(i) for i in range(-1, n_families)]:
        shares = shares.withColumn(f"fam_{c.replace('-', 'm')}", F.coalesce(F.col(f"`{c}`"), F.lit(0)) / F.col("_t")).drop(c)
    return shares.drop("_t")
