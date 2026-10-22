"""Twitch channel audience life curves: new vs returning viewers, retention cohorts, decay fits.

Left-censoring: the window opens at slot 0, so channels already live then have unknown age. Only
channels whose first activity is on or after day BIRTH_MIN_DAY (and that leave >= MIN_FOLLOW days of
follow-up) are treated as "born in window".
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from streaming_content import config
from streaming_content.timing import timed

BIRTH_MIN_DAY = 7
MIN_FOLLOW = 14
TIERS = [("small", 5, 30), ("medium", 30, 300), ("large", 300, 10**9)]  # by median daily viewers in first week
FIT_UNTIL_AGE = 14  # fit on ages [1, 14], forecast ages [15, 28]


def build_tables(spark: SparkSession) -> None:
    s = spark.table(config.table("silver", "twitch_view_sessions"))
    day = (F.col("start_slot") / config.SLOTS_PER_DAY).cast("int")
    with timed("lifecurves_user_channel_day") as t:
        ucd = s.groupBy("user_id", "streamer_id", day.alias("day")).agg(F.sum("new_watch_slots").alias("slots"))
        ucd.write.format("delta").mode("overwrite").saveAsTable(config.table("gold", "twitch_user_channel_day"))
        t["rows"] = spark.table(config.table("gold", "twitch_user_channel_day")).count()
    with timed("lifecurves_channel_daily") as t:
        u = spark.table(config.table("gold", "twitch_user_channel_day"))
        w = Window.partitionBy("user_id", "streamer_id")
        u = u.withColumn("first_day", F.min("day").over(w))
        cd = u.groupBy("streamer_id", "day").agg(
            F.count("*").alias("viewers"),
            F.sum((F.col("day") == F.col("first_day")).cast("int")).alias("new_viewers"),
            F.sum("slots").alias("slots"),
        ).withColumn("returning_viewers", F.col("viewers") - F.col("new_viewers"))
        cd.write.format("delta").mode("overwrite").saveAsTable(config.table("gold", "twitch_channel_daily"))
        t["rows"] = spark.table(config.table("gold", "twitch_channel_daily")).count()
    with timed("lifecurves_retention") as t:
        u = spark.table(config.table("gold", "twitch_user_channel_day"))
        wk = (F.col("day") / 7).cast("int")
        uw = u.groupBy("user_id", "streamer_id", wk.alias("week")).count().drop("count")
        first = uw.withColumn("first_week", F.min("week").over(Window.partitionBy("user_id", "streamer_id")))
        ret = first.groupBy("first_week", "week").agg(F.count("*").alias("pairs"))
        ret.write.format("delta").mode("overwrite").saveAsTable(config.table("gold", "twitch_retention_weeks"))
        t["rows"] = spark.table(config.table("gold", "twitch_retention_weeks")).count()


def retention_matrix(spark: SparkSession) -> pd.DataFrame:
    r = spark.table(config.table("gold", "twitch_retention_weeks")).toPandas()
    r["lag"] = r["week"] - r["first_week"]
    base = r[r["lag"] == 0].set_index("first_week")["pairs"]
    r["retention"] = r["pairs"] / r["first_week"].map(base)
    return r.pivot(index="first_week", columns="lag", values="retention")


def channel_curves(spark: SparkSession) -> pd.DataFrame:
    """Daily viewers of channels born in the window, by age, with a tier from first-week size."""
    cd = spark.table(config.table("gold", "twitch_channel_daily"))
    birth = cd.groupBy("streamer_id").agg(F.min("day").alias("birth_day"))
    last_day = int(cd.agg(F.max("day")).first()[0])
    born = birth.filter((F.col("birth_day") >= BIRTH_MIN_DAY) & (F.col("birth_day") <= last_day - MIN_FOLLOW))
    j = cd.join(F.broadcast(born), "streamer_id").withColumn("age", F.col("day") - F.col("birth_day"))
    j = j.filter(F.col("age") <= 28)
    return j.select("streamer_id", "age", "viewers", "new_viewers", "returning_viewers", "birth_day").toPandas()


def fit_decay(curves: pd.DataFrame) -> dict:
    """Per tier: exponential vs power decay of log(viewers) on age, with title-style fixed effects.

    Level per channel from ages 1..7; shape pooled within tier from ages 1..FIT_UNTIL_AGE (within-channel
    variation). Compared on the held-out ages FIT_UNTIL_AGE+1..28 of other channels' own level.
    """
    out: dict = {}
    c = curves.copy()
    # fill zero-viewer days (channel offline) as missing: a channel with no chatters that day is not in the table
    first_week = c[(c["age"] >= 0) & (c["age"] <= 6)].groupby("streamer_id")["viewers"].median()
    c["tier"] = c["streamer_id"].map(first_week).map(
        lambda v: next((n for n, lo, hi in TIERS if lo <= v < hi), None) if pd.notna(v) else None
    )
    c = c.dropna(subset=["tier"])
    for tier, g in c.groupby("tier"):
        g = g.assign(ly=np.log(g["viewers"]))
        fitg = g[(g["age"] >= 1) & (g["age"] <= FIT_UNTIL_AGE)]
        # exposure here is days online-with-audience; age is days since first activity
        res = {"n_channels": int(g["streamer_id"].nunique())}
        for fam, tau in (("exponential", 0.0), ("power", 3.0)):
            x = fitg["age"].to_numpy(float) if fam == "exponential" else np.log1p(fitg["age"].to_numpy(float) / tau)
            xs = pd.Series(x, index=fitg.index)
            xc = xs - xs.groupby(fitg["streamer_id"]).transform("mean")
            yc = fitg["ly"] - fitg.groupby("streamer_id")["ly"].transform("mean")
            shape = float(-(xc * yc).sum() / (xc * xc).sum())
            # holdout: level from ages 1..FIT_UNTIL_AGE, predict later ages
            hold = g[g["age"] > FIT_UNTIL_AGE]
            lvl = (fitg["ly"] + shape * xs).groupby(fitg["streamer_id"]).mean()
            hx = hold["age"].to_numpy(float) if fam == "exponential" else np.log1p(hold["age"].to_numpy(float) / tau)
            pred = hold["streamer_id"].map(lvl).to_numpy() - shape * hx
            ok = ~np.isnan(pred)
            res[fam] = {"shape": shape, "holdout_rmse_log": float(np.sqrt(np.mean((pred[ok] - hold["ly"].to_numpy()[ok]) ** 2))),
                        "holdout_bias_log": float(np.mean(pred[ok] - hold["ly"].to_numpy()[ok])), "n_holdout": int(ok.sum())}
        lvl0 = fitg.groupby("streamer_id")["ly"].mean()
        hold = g[g["age"] > FIT_UNTIL_AGE]
        pred = hold["streamer_id"].map(lvl0).to_numpy()
        ok = ~np.isnan(pred)
        res["persistence"] = {"holdout_rmse_log": float(np.sqrt(np.mean((pred[ok] - hold["ly"].to_numpy()[ok]) ** 2))),
                              "holdout_bias_log": float(np.mean(pred[ok] - hold["ly"].to_numpy()[ok])), "n_holdout": int(ok.sum())}
        by_age = g.groupby("age").agg(viewers=("viewers", "median"), returning_share=("returning_viewers", "sum"), tot=("viewers", "sum"))
        by_age["returning_share"] = by_age["returning_share"] / by_age["tot"]
        res["median_viewers_by_age"] = {int(k): float(v) for k, v in by_age["viewers"].items() if k in (0, 1, 3, 7, 14, 21, 28)}
        res["returning_share_by_age"] = {int(k): float(v) for k, v in by_age["returning_share"].items() if k in (0, 1, 3, 7, 14, 21, 28)}
        out[tier] = res
    return out


def run(spark: SparkSession, skip_build: bool = False) -> dict:
    if not skip_build:
        build_tables(spark)
    cd = spark.table(config.table("gold", "twitch_channel_daily"))
    res: dict = {"channel_days": cd.count(), "channels": cd.select("streamer_id").distinct().count()}
    ret = retention_matrix(spark)
    res["retention_pair_weeks"] = {int(k): {int(lag): round(float(v), 4) for lag, v in row.dropna().items()} for k, row in ret.iterrows()}
    curves = channel_curves(spark)
    res["born_in_window"] = {"channels": int(curves["streamer_id"].nunique()), "rule": f"first day >= {BIRTH_MIN_DAY} and >= {MIN_FOLLOW} days before window end"}
    res["decay"] = fit_decay(curves)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "twitch_lifecurves.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    return res
