"""Cohort definition and pre-release features for the performance score.

Unit: one (series season | film | other programme) with a release date >= 2023-01-01.
Target: log views in the title's *release semester* (see docs/methodology.md for why).
Every feature must be knowable at the release date; `tests/test_leakage.py` enforces that by
corrupting everything observed from the release semester onward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from streaming_content.netflix.parse import SEMESTERS, semester_bounds
from streaming_content.netflix.silver import semester_of

MIN_RELEASE = pd.Timestamp("2023-01-01")
GENRES = [
    "Drama", "Comedy", "Documentary", "Action", "Thriller", "Crime", "Animation", "Romance",
    "Reality-TV", "Family", "Adventure", "Horror", "Sci-Fi", "Fantasy", "Mystery", "Music",
    "Talk-Show", "Biography", "History", "Sport",
]
FEATURES_NUM = [
    "log_runtime", "exposure_days", "release_month", "release_dow", "season_no", "is_limited",
    "has_alt_title", "imdb_matched", "imdb_runtime", "imdb_n_episodes", "imdb_is_adult",
    "prior_season_log_views", "n_prior_seasons", "franchise_recent_log_rate",
    "title_words",
] + [f"g_{g}" for g in GENRES]
FEATURES_CAT = ["kind", "primary_genre"]


def kind_of(row_season_no, content_type) -> str:
    if pd.notna(row_season_no):
        return "series_season"
    return "film" if content_type == "movie" else "tv_other"


def build_cohort(silver: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """One row per title in the cohort, features + target.

    `meta` has one row per title_key: tconst, genres, imdb_runtime, imdb_is_adult, imdb_n_episodes,
    match_reason. It comes from a snapshot of IMDb; only fields fixed at release are used.
    """
    s = silver[~silver["event_like"] & silver["release_date"].notna()].copy()
    s = s[s["release_date"] >= MIN_RELEASE]
    s["rel_sem"] = s["release_date"].map(semester_of)

    # title-level attributes (take the first row of the title across semesters)
    first = s.sort_values("sem_idx").drop_duplicates("title_key")
    # target: views in the release semester
    tgt = s[s["semester"] == s["rel_sem"]][["title_key", "views", "hours_viewed"]]
    tgt = tgt.groupby("title_key").first().rename(columns={"views": "views_rel_sem", "hours_viewed": "hours_rel_sem"})
    c = first.merge(tgt, on="title_key", how="inner")
    c = c.merge(meta, on="title_key", how="left")
    c = c.dropna(subset=["views_rel_sem", "runtime_min"])
    c["kind"] = [kind_of(a, b) for a, b in zip(c["season_no"], c["content_type"])]
    c["y"] = np.log(c["views_rel_sem"])
    c["log_runtime"] = np.log(c["runtime_min"])
    c["exposure_days"] = c["release_date"].map(lambda d: (semester_bounds(semester_of(d))[1] - d).days + 1)
    c["release_month"] = c["release_date"].dt.month
    c["release_dow"] = c["release_date"].dt.day_of_week
    c["is_limited"] = c["is_limited"].astype(int)
    c["has_alt_title"] = c["has_alt_title"].astype(int)
    c["imdb_matched"] = c["tconst"].notna().astype(int)
    c["title_words"] = c["series_base"].str.split().str.len()
    g = c["genres"].astype("string").fillna("")
    for name in GENRES:
        c[f"g_{name}"] = g.str.contains(rf"(?:^|,){name}(?:,|$)", regex=True).astype(int)
    c["primary_genre"] = g.str.split(",").str[0].replace("", np.nan)
    c = add_prior_season(c, silver)
    return c.reset_index(drop=True)


def add_prior_season(c: pd.DataFrame, silver: pd.DataFrame) -> pd.DataFrame:
    """Franchise features from earlier seasons, using only semesters that ended before release.

    * prior_season_log_views: season N-1's views in its own release semester (needs that semester
      to be in the data and to have ended strictly before this season's release date).
    * franchise_recent_log_rate: max over earlier seasons of log(views per exposure day) in the
      latest semester that ended before this release (works for seasons released before 2023).
    """
    allv = silver[silver["season_no"].notna() & silver["views"].notna() & ~silver["event_like"]].copy()
    allv["sem_end"] = allv["semester"].map(lambda x: semester_bounds(x)[1])
    allv["rel_sem"] = allv["release_date"].map(lambda d: semester_of(d) if pd.notna(d) else None)
    allv["log_rate"] = np.log(allv["views"] / allv["exposure_days"].clip(lower=1))
    by_series = {k: g for k, g in allv.groupby("series_base")}
    prior, n_prior, fr_rate = [], [], []
    for sb, sn, d in zip(c["series_base"], c["season_no"], c["release_date"]):
        g = by_series.get(sb)
        n_prior.append(float(sn) - 1 if pd.notna(sn) else np.nan)
        if pd.isna(sn) or g is None:
            prior.append(np.nan)
            fr_rate.append(np.nan)
            continue
        past = g[(g["season_no"] < sn) & (g["sem_end"] < d)]
        if past.empty:
            prior.append(np.nan)
            fr_rate.append(np.nan)
            continue
        latest = past.sort_values("sem_idx").groupby("season_no").tail(1)
        fr_rate.append(latest["log_rate"].max())
        p = past[(past["season_no"] == sn - 1) & (past["semester"] == past["rel_sem"])]
        prior.append(np.log(p["views"].iat[0]) if len(p) else np.nan)
    c = c.copy()
    c["prior_season_log_views"], c["n_prior_seasons"], c["franchise_recent_log_rate"] = prior, n_prior, fr_rate
    return c


def build_meta(silver: pd.DataFrame, basics: pd.DataFrame, episodes: pd.DataFrame, match_fn) -> pd.DataFrame:
    """Match every cohort title to IMDb; return one row per title_key."""
    s = silver[~silver["event_like"] & silver["release_date"].notna()]
    s = s[s["release_date"] >= MIN_RELEASE].drop_duplicates("title_key")
    b = basics.set_index("tconst")
    ep = episodes.set_index(["parentTconst", "seasonNumber"])["n_episodes"].to_dict()
    rows = []
    for r in s.itertuples():
        is_series = pd.notna(r.season_no)
        tc, why = match_fn(r.series_base if is_series else r.title_en, r.title_alt, r.release_date.year, is_series)
        rec = {"title_key": r.title_key, "tconst": tc, "match_reason": why}
        if tc is not None:
            row = b.loc[tc]
            rec.update(
                genres=row["genres"], imdb_runtime=row["runtimeMinutes"], imdb_is_adult=row["isAdult"],
                imdb_n_episodes=ep.get((tc, float(r.season_no))) if is_series else np.nan,
            )
        rows.append(rec)
    return pd.DataFrame(rows)


__all__ = ["build_cohort", "build_meta", "FEATURES_NUM", "FEATURES_CAT", "SEMESTERS"]
