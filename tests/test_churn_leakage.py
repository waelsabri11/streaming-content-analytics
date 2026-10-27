from pyspark.sql import functions as F

from streaming_content.twitch import churn
from streaming_content.twitch.silver import sessionize

COLS = "user_id int, stream_id long, streamer_id int, start_slot int, stop_slot int"
W = churn.WEEK


def _sess(spark, rows):
    df = spark.createDataFrame(rows, COLS).withColumn("watch_slots", F.col("stop_slot") - F.col("start_slot"))
    return sessionize(df)


def test_features_ignore_everything_at_or_after_cutoff(spark):
    cut = 3 * W
    pre = [(1, 1, 1, 10, 14), (1, 2, 2, 500, 503), (2, 3, 1, 2000, 2004)]
    post = [(1, 4, 1, cut + 5, cut + 9), (2, 5, 3, cut + 100, cut + 130), (3, 6, 1, cut + 1, cut + 4)]
    a = {r.user_id: r.asDict() for r in churn.build_features(_sess(spark, pre), 0, cut).collect()}
    b = {r.user_id: r.asDict() for r in churn.build_features(_sess(spark, pre + post), 0, cut).collect()}
    assert a == b  # post-cutoff activity changes nothing, and viewer 3 (only active after the cutoff) has no row
    assert 3 not in b


def test_labels_cover_exactly_the_next_week(spark):
    cut = 3 * W
    rows = [(1, 1, 1, cut - 1, cut + 1), (5, 7, 1, cut, cut + 1), (2, 2, 1, cut + W - 1, cut + W + 3), (3, 3, 1, cut + W, cut + W + 2), (4, 4, 1, cut - 5, cut - 3)]
    lab = {r.user_id for r in churn.build_labels(_sess(spark, rows), cut).collect()}
    assert lab == {5, 2}  # start in [cut, cut + week); viewer 1 started before the cutoff, viewer 3 exactly one week later, viewer 4 before


def test_train_and_test_windows_are_temporal_and_viewers_disjoint(spark):
    assert churn.TRAIN["cutoff"] + churn.WEEK <= churn.TEST["cutoff"]  # train label window ends before test label starts
    assert churn.TEST["feat_lo"] < churn.TEST["cutoff"] and churn.TRAIN["feat_lo"] < churn.TRAIN["cutoff"]
    users = spark.range(2000).selectExpr("cast(id as int) as user_id")
    tr = users.filter(churn.viewer_bucket() >= churn.TEST_VIEWER_BUCKETS)
    te = users.filter(churn.viewer_bucket() < churn.TEST_VIEWER_BUCKETS)
    assert tr.join(te, "user_id").count() == 0 and abs(te.count() / 2000 - 0.2) < 0.05


def test_interval_straddling_the_cutoff_is_clipped(spark):
    cut = 3 * W
    straddle = [(1, 1, 1, cut - 4, cut + 50)]
    clipped = [(1, 1, 1, cut - 4, cut)]
    a = churn.build_features(_sess(spark, straddle), 0, cut).first().asDict()
    b = churn.build_features(_sess(spark, clipped), 0, cut).first().asDict()
    assert a["recency_days"] == 0.0
    for k in ("log_hours", "recency_days", "last_week_hours", "mean_session_min"):
        assert abs(a[k] - b[k]) < 1e-9, k
