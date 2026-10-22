import math

import pytest
from pyspark.sql import functions as F

from streaming_content.twitch.features import FEATURE_COLS, viewer_features
from streaming_content.twitch.silver import sessionize


def _feats(spark, rows, peak=(6, 0)):
    df = spark.createDataFrame(rows, "user_id int, stream_id long, streamer_id int, start_slot int, stop_slot int")
    df = df.withColumn("watch_slots", F.col("stop_slot") - F.col("start_slot"))
    return {r.user_id: r for r in viewer_features(sessionize(df), peak).collect()}


def test_viewer_features_values(spark):
    # viewer 1: channel 1 for 6 slots (hour 0, day 0), channel 2 for 2 slots on day 1 (session 2), channel 1 again day 1
    rows = [(1, 1, 1, 0, 6), (1, 2, 2, 144 + 60, 144 + 62), (1, 3, 1, 144 + 70, 144 + 72),
            (2, 4, 7, 10, 11)]
    f = _feats(spark, rows)
    v = f[1]
    assert v.hours == pytest.approx(10 * 10 / 60)  # 10 slots
    assert v.n_sessions == 3 and v.active_days == 2 and v.n_streamers == 2
    assert v.top1_share == pytest.approx(8 / 10)
    p = [0.8, 0.2]
    assert v.norm_entropy == pytest.approx(-sum(x * math.log(x) for x in p) / math.log(2))
    # channel 1 is watched in 2 sessions -> repeat share = its 8 slots / 10
    assert v.repeat_share == pytest.approx(0.8)
    assert v.share_night + v.share_morning + v.share_afternoon + v.share_evening == pytest.approx(1.0)
    assert f[2].n_streamers == 1 and f[2].norm_entropy == 0.0 and f[2].top1_share == 1.0


def test_feature_columns_present(spark):
    f = _feats(spark, [(1, 1, 1, 0, 3)])
    assert all(c in f[1].asDict() for c in FEATURE_COLS)


def test_fit_best_of_recovers_separated_clusters(spark):
    import random

    from streaming_content.twitch.segmentation import fit_best_of

    random.seed(0)
    rows = [(i, random.gauss(c, 0.1), random.gauss(c, 0.1)) for i in range(300) for c in [(i % 3) * 10]]
    df = spark.createDataFrame(rows, "user_id int, a double, b double")
    pred = fit_best_of(df, ["a", "b"], 3, seeds=(1, 2, 3)).transform(df).select("user_id", "segment").collect()
    groups = {}
    for r in pred:
        groups.setdefault(r.user_id % 3, set()).add(r.segment)
    assert all(len(v) == 1 for v in groups.values()) and len({next(iter(v)) for v in groups.values()}) == 3
