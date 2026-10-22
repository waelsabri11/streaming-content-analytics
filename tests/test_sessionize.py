from streaming_content.twitch.silver import clean_views, session_table, sessionize

COLS = ["user_id", "stream_id", "streamer_id", "start_slot", "stop_slot"]


def _views(spark, rows):
    df = spark.createDataFrame(rows, COLS)
    from pyspark.sql import functions as F

    return df.withColumn("watch_slots", F.col("stop_slot") - F.col("start_slot"))


def test_gap_rule_splits_after_three_slots(spark):
    # gap of exactly 3 slots keeps one session, gap of 4 starts a new one
    rows = [(1, 1, 1, 0, 2), (1, 2, 1, 5, 6), (1, 3, 1, 10, 11)]
    s = sessionize(_views(spark, rows)).orderBy("start_slot").collect()
    assert [r.session_id for r in s] == [1, 1, 2]


def test_parallel_streams_do_not_split_and_union_not_double_counted(spark):
    # long stream 0-20 with a short parallel stream 5-7 and a later one at 22 (gap 2)
    rows = [(1, 1, 1, 0, 20), (1, 2, 2, 5, 7), (1, 3, 3, 22, 24)]
    sess = session_table(sessionize(_views(spark, rows))).collect()
    assert len(sess) == 1
    assert sess[0].watch_slots == 20 + 2  # union: [0,20) + [22,24)
    assert sess[0].n_streamers == 3


def test_sessions_are_per_viewer(spark):
    rows = [(1, 1, 1, 0, 2), (2, 1, 1, 0, 2), (2, 2, 2, 100, 101)]
    sess = session_table(sessionize(_views(spark, rows))).collect()
    assert sorted((r.user_id, r.session_id) for r in sess) == [(1, 1), (2, 1), (2, 2)]


def test_clean_views_drops_invalid(spark):

    rows = [(1, 1, "a", 0, 2, None), (1, 1, "a", 0, 2, None), (2, 1, "a", 5, 5, None), (3, 1, "a", 0, 9999, None)]
    b = spark.createDataFrame(rows, "user_id int, stream_id long, streamer string, start_slot int, stop_slot int, _corrupt string")
    out = clean_views(b).collect()
    assert len(out) == 1 and out[0].watch_slots == 2
