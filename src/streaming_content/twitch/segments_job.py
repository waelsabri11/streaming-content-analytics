"""Viewer segmentation job: choose k, fit, name and profile segments, check stability, test ALS.

Everything quoted in the README comes from reports/results/twitch_segments.json written here.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from streaming_content import config
from streaming_content.spark import cached
from streaming_content.timing import timed
from streaming_content.twitch import affinity, segmentation
from streaming_content.twitch.features import FEATURE_COLS

KS = [4, 5, 6, 7, 8, 9, 10]  # a-priori range: fewer than 4 segments is not useful for a business team
ARI_FLOOR = 0.80  # k is eligible if bootstrap ARI >= this; then highest silhouette wins
N_FAMILIES = 12
RANK = 16
EMB_COLS = [f"emb_{i}" for i in range(RANK)]


def choose_k(sweep: list[dict]) -> int:
    ok = [r for r in sweep if r["bootstrap_ari_mean"] >= ARI_FLOOR]
    pool = ok or sweep
    key = (lambda r: r["silhouette"]) if ok else (lambda r: r["bootstrap_ari_mean"])
    return max(pool, key=key)["k"]


HEAVY_HOURS, REGULAR_HOURS = 30.0, 10.0  # viewing hours over the 43-day window
CLOCK_BLOCKS = [("share_night", "00-06h"), ("share_morning", "06-12h"), ("share_afternoon", "12-18h"), ("share_evening", "18-24h")]


def name_segments(prof: pd.DataFrame) -> dict[int, str]:
    """Composite descriptive names built from fixed thresholds on the profile (no hand-picking).

    volume tier (hours over the window: Heavy >= 30, Regular 10-30, Light < 10) + optional tags:
    single-channel (top-1 share >= 0.8) or loyal (>= 0.55); long-session (mean session >= 60 min);
    clock window whose share is >= 1.4x the all-viewer average. Time-of-day refers to the dataset clock
    (slot-of-day), not to a known time zone. Duplicate names get a segment suffix.
    """
    w = prof["n_viewers"] / prof["n_viewers"].sum()
    avg = {c: float((prof[c] * w).sum()) for c, _ in CLOCK_BLOCKS}
    names: dict[int, str] = {}
    for _, r in prof.iterrows():
        vol = "Heavy" if r["hours"] >= HEAVY_HOURS else "Regular" if r["hours"] >= REGULAR_HOURS else "Light"
        tags = []
        if r["top1_share"] >= 0.8:
            tags.append("single-channel")
        elif r["top1_share"] >= 0.55:
            tags.append("loyal")
        if r["mean_session_min"] >= 60:
            tags.append("long-session")
        lifts = {lab: r[c] / avg[c] for c, lab in CLOCK_BLOCKS}
        best = max(lifts, key=lifts.get)
        if lifts[best] >= 1.4:
            tags.append(f"clock {best}")
        names[int(r["segment"])] = " ".join([vol, *tags, "viewers"])
    counts = pd.Series(names).value_counts()
    return {s: (n if counts[n] == 1 else f"{n} (S{s})") for s, n in names.items()}


def make_labels(df, k: int, seed: int):
    m = segmentation.fit_best_of(df, FEATURE_COLS, k, tuple(range(3 * seed, 3 * seed + 3)))
    return m.transform(df).select("user_id", F.col("segment").alias(f"s{seed - 1}"))


def eta_squared(df: pd.DataFrame, group: str, y: str) -> float:
    g = df.groupby(group)[y]
    ss_between = (g.count() * (g.mean() - df[y].mean()) ** 2).sum()
    ss_total = ((df[y] - df[y].mean()) ** 2).sum()
    return float(ss_between / ss_total)


def rename_from_saved_profile() -> dict:
    """Recompute segment names from the saved profile without refitting (names are deterministic)."""
    path = config.RESULTS_DIR / "twitch_segments.json"
    res = json.loads(path.read_text())
    prof = pd.DataFrame(res["profile"])
    names = name_segments(prof)
    for rec in res["profile"]:
        rec["name"] = names[int(rec["segment"])]
    res["segment_names"] = {str(k): v for k, v in names.items()}
    path.write_text(json.dumps(res, indent=2, default=float) + "\n")
    return names


def run(spark: SparkSession) -> dict:
    res: dict = {"features": FEATURE_COLS, "k_candidates": KS, "ari_floor": ARI_FLOOR}
    feats = cached(spark.table(config.table("gold", "twitch_viewer_features")))
    res["n_viewers"] = feats.count()

    with timed("segments_select_k") as t:
        sweep = segmentation.select_k(feats, FEATURE_COLS, KS, sample_frac=0.02)
        t["n_sample"] = sweep[0]["n_sample"]
    res["k_sweep"] = sweep
    k = choose_k(sweep)
    res["k"] = k
    with timed("segments_seed_stability", k=k):
        samp = cached(feats.sample(False, 0.02, 5).select("user_id", *FEATURE_COLS))
        labs = [make_labels(samp, k, sd) for sd in (1, 2, 3, 4)]  # four disjoint sets of three restarts
        aris = [segmentation.adjusted_rand_index(segmentation.contingency(labs[i].join(labs[j], "user_id"), f"s{i}", f"s{j}"))
                for i in range(4) for j in range(i + 1, 4)]
        res["kmeans_seed_stability"] = {"k": k, "n": samp.count(), "restarts_per_model": 3, "ari_pairwise_mean": float(np.mean(aris)), "ari_pairwise_min": float(np.min(aris))}
    sess_all = spark.table(config.table("silver", "twitch_view_sessions"))
    d_h = sess_all.groupBy(((F.col("start_slot") % config.SLOTS_PER_DAY) / 6).cast("int").alias("h")).agg(F.sum("new_watch_slots").alias("w")).orderBy("h").toPandas()
    res["diurnal_watch_share_by_clock_hour"] = (d_h["w"] / d_h["w"].sum()).round(4).tolist()

    with timed("segments_fit_final", k=k) as t:
        model = segmentation.fit_final(feats, FEATURE_COLS, k)
        assigned = model.transform(feats).select("user_id", "segment", "hours", "n_sessions", "active_days", "n_streamers", *FEATURE_COLS)
        assigned.write.format("delta").mode("overwrite").saveAsTable(config.table("gold", "twitch_viewer_segments"))
    assigned = spark.table(config.table("gold", "twitch_viewer_segments"))
    prof = segmentation.profile(assigned, FEATURE_COLS).toPandas()
    names = name_segments(prof)
    prof["name"] = prof["segment"].map(names)
    res["profile"] = prof.round(4).to_dict("records")

    # ---- stability across time halves: model fitted on H1 vs H2, and viewers' segment persistence
    h1 = spark.table(config.table("gold", "twitch_viewer_features_h1"))
    h2 = spark.table(config.table("gold", "twitch_viewer_features_h2"))
    with timed("segments_time_halves", k=k):
        m1 = segmentation.fit_final(h1, FEATURE_COLS, k)
        m2 = segmentation.fit_final(h2, FEATURE_COLS, k)
        both = (
            h1.select("user_id", *[F.col(c).alias(f"a_{c}") for c in FEATURE_COLS], F.col("hours").alias("hours_h1"))
            .join(h2.select("user_id", *FEATURE_COLS, F.col("hours").alias("hours_h2")), "user_id")
        )
        a = both.select("user_id", "hours_h1", "hours_h2", *[F.col(f"a_{c}").alias(c) for c in FEATURE_COLS])
        seg_h1_on_h1 = m1.transform(a).select("user_id", F.col("segment").alias("seg_h1"))
        b = both.select("user_id", *FEATURE_COLS)
        seg_h2_by_m1 = m1.transform(b).select("user_id", F.col("segment").alias("seg_h2_m1"))
        seg_h2_by_m2 = m2.transform(b).select("user_id", F.col("segment").alias("seg_h2_m2"))
        j = cached(seg_h1_on_h1.join(seg_h2_by_m1, "user_id").join(seg_h2_by_m2, "user_id"))
        n_both = j.count()
        res["time_halves"] = {
            "viewers_active_in_both_halves": n_both,
            "ari_model_h1_vs_model_h2_on_h2_features": segmentation.adjusted_rand_index(segmentation.contingency(j, "seg_h2_m1", "seg_h2_m2")),
            "ari_same_model_h1_features_vs_h2_features": segmentation.adjusted_rand_index(segmentation.contingency(j, "seg_h1", "seg_h2_m1")),
            "share_viewers_same_segment_h1_to_h2": float(j.filter(F.col("seg_h1") == F.col("seg_h2_m1")).count() / n_both),
        }

    # ---- predictive validity: H1 segment -> H2 outcomes (retention and volume) vs plain volume deciles
    with timed("segments_predictive_validity"):
        seg1 = m1.transform(h1).select("user_id", F.col("segment").alias("seg"), F.col("hours").alias("hours_h1"))
        out = seg1.join(h2.select("user_id", F.col("hours").alias("hours_h2")), "user_id", "left").fillna({"hours_h2": 0.0})
        out = out.withColumn("active_h2", (F.col("hours_h2") > 0).cast("int")).withColumn("log_hours_h2", F.log1p("hours_h2"))
        out = out.withColumn("hours_decile", F.ntile(10).over(Window.orderBy("hours_h1"))).withColumn(
            "hours_quintile", F.ntile(k).over(Window.orderBy("hours_h1")))
        pdf = out.sample(False, 0.05, 7).toPandas()
        res["predictive_validity"] = {
            "n_sample": len(pdf),
            "retained_in_h2_by_segment": pdf.groupby("seg")["active_h2"].mean().round(4).to_dict(),
            "eta2_log_hours_h2_by_segment": eta_squared(pdf, "seg", "log_hours_h2"),
            "eta2_log_hours_h2_by_h1_hours_decile": eta_squared(pdf, "hours_decile", "log_hours_h2"),
            "eta2_log_hours_h2_by_h1_hours_equal_k_groups": eta_squared(pdf, "hours_quintile", "log_hours_h2"),
            "eta2_retained_by_segment": eta_squared(pdf, "seg", "active_h2"),
            "eta2_retained_by_h1_hours_decile": eta_squared(pdf, "hours_decile", "active_h2"),
            "eta2_retained_by_h1_hours_equal_k_groups": eta_squared(pdf, "hours_quintile", "active_h2"),
            "equal_k_groups": k,
        }

    # ---- ALS channel-affinity embeddings as additional segmentation input.
    # Everything here uses H1 data only (ALS factors, fold-in, features), then H2 outcomes judge it.
    mid = config.N_SLOTS // 2
    s = spark.table(config.table("silver", "twitch_view_sessions"))
    uc = cached(affinity.user_channel_matrix(s))  # full window, for the segment x channel table below
    with timed("segments_als", rank=RANK) as t:
        uc1 = cached(affinity.user_channel_matrix(s.filter(F.col("start_slot") < mid)))
        model_als, keep, n_users, n_items = affinity.train_als(uc1, user_frac=0.1, min_viewers=200, rank=RANK)
        fam = cached(affinity.channel_families(model_als, keep, N_FAMILIES))
        t.update(als_users=n_users, als_channels=n_items)
        emb = affinity.fold_in(uc1, fam, RANK)
        fam.select("streamer_id", "family").write.format("delta").mode("overwrite").saveAsTable(config.table("gold", "twitch_channel_families"))
        emb.write.format("delta").mode("overwrite").saveAsTable(config.table("gold", "twitch_viewer_embeddings_h1"))
    res["als"] = {"rank": RANK, "trained_on": "first half (slots < 3074), 10% viewer sample", "n_train_users": n_users,
                  "n_channels_with_factors": n_items, "n_families": N_FAMILIES}
    embd = spark.table(config.table("gold", "twitch_viewer_embeddings_h1"))
    withemb = cached(h1.join(embd, "user_id", "inner"))
    res["als"]["h1_viewers_with_embedding"] = withemb.count()
    var_cols = FEATURE_COLS + EMB_COLS
    with timed("segments_als_variant", k=k):
        variants = {}
        outcomes = h2.select("user_id", F.col("hours").alias("hours_h2"))
        rows = withemb.join(outcomes, "user_id", "left").fillna({"hours_h2": 0.0}).withColumn(
            "log_hours_h2", F.log1p("hours_h2")).withColumn("active_h2", (F.col("hours_h2") > 0).cast("int"))
        for name, cols in (("behaviour_only", FEATURE_COLS), ("behaviour_plus_embedding", var_cols)):
            sweep1 = segmentation.select_k(withemb, cols, [k], sample_frac=0.05)[0]
            mdl = segmentation.fit_final(withemb, cols, k)
            pdf_v = mdl.transform(rows).select("log_hours_h2", "active_h2", "segment").sample(False, 0.1, 13).toPandas()
            variants[name] = {**sweep1, "n_features": len(cols),
                              "eta2_log_hours_h2": eta_squared(pdf_v, "segment", "log_hours_h2"),
                              "eta2_retained_h2": eta_squared(pdf_v, "segment", "active_h2")}
    res["als_variant"] = {"k": k, "variants": variants,
                          "note": "silhouette is not comparable across feature spaces of different dimension; ARI and eta2 are"}

    # ---- segment x channel affinity (lift of segment watch time on a channel vs everyone)
    with timed("segments_channel_affinity"):
        segs = assigned.select("user_id", "segment")
        j2 = uc.join(segs, "user_id")
        per = j2.groupBy("segment", "streamer_id").agg(F.sum("slots").alias("s"), F.count("*").alias("viewers"))
        seg_tot = per.groupBy("segment").agg(F.sum("s").alias("seg_total"))
        ch_tot = per.groupBy("streamer_id").agg(F.sum("s").alias("ch_total"), F.sum("viewers").alias("ch_viewers"))
        grand = per.agg(F.sum("s")).first()[0]
        aff = (per.join(seg_tot, "segment").join(ch_tot, "streamer_id")
               .filter(F.col("ch_viewers") >= 5000)
               .withColumn("lift", (F.col("s") / F.col("seg_total")) / (F.col("ch_total") / F.lit(float(grand))))
               .filter(F.col("viewers") >= 500))
        dim = spark.table(config.table("silver", "twitch_streamers"))
        w = Window.partitionBy("segment").orderBy(F.desc("lift"))
        top = aff.withColumn("rk", F.row_number().over(w)).filter("rk <= 10").join(dim, "streamer_id")
        top.select("segment", "rk", "streamer", "lift", "viewers", "s").write.format("delta").mode("overwrite").saveAsTable(
            config.table("gold", "twitch_segment_channel_affinity"))
        res["channel_affinity_top"] = {
            int(r["segment"]): [] for r in top.select("segment").distinct().collect()
        }
        for r in spark.table(config.table("gold", "twitch_segment_channel_affinity")).orderBy("segment", "rk").collect():
            res["channel_affinity_top"][int(r["segment"])].append({"channel": r["streamer"], "lift": round(float(r["lift"]), 2), "viewers": int(r["viewers"])})

    res["segment_names"] = {int(k_): v for k_, v in names.items()}
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "twitch_segments.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    return res
