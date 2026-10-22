import numpy as np
import pandas as pd
import pytest
from fixtures import synthetic_bronze

from streaming_content.netflix import features as F
from streaming_content.netflix.parse import SEMESTERS
from streaming_content.netflix.silver import build_silver, exposure_days, parse_title

META_COLS = ["title_key", "tconst", "match_reason", "genres", "imdb_runtime", "imdb_is_adult", "imdb_n_episodes"]


def _meta(silver):
    keys = silver["title_key"].unique()
    return pd.DataFrame({"title_key": keys}).reindex(columns=META_COLS)


def _cohort(bronze):
    s = build_silver(bronze)
    return F.build_cohort(s, _meta(s)), s


def test_features_do_not_move_when_everything_from_release_semester_is_corrupted():
    bronze = synthetic_bronze()
    base, _ = _cohort(bronze)
    cut = "2025H1"
    corrupted = bronze.copy()
    m = corrupted["semester"].map(SEMESTERS.index) >= SEMESTERS.index(cut)
    corrupted.loc[m, ["views", "hours_viewed"]] *= 1000.0
    alt, _ = _cohort(corrupted)
    a = base[base["rel_sem"] == cut].set_index("title_key")[F.FEATURES_NUM].sort_index()
    b = alt[alt["rel_sem"] == cut].set_index("title_key")[F.FEATURES_NUM].sort_index()
    assert len(a) > 5
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    # sanity: the target did move, so the corruption was actually seen
    ya = base[base["rel_sem"] == cut].set_index("title_key")["y"].sort_index()
    yb = alt[alt["rel_sem"] == cut].set_index("title_key")["y"].sort_index()
    assert (yb - ya).mean() > 5


def test_prior_season_requires_semester_ended_before_release():
    rows = []
    def r(sem, title, rel, views):
        rows.append(dict(semester=sem, sheet_kind="tv", title_raw=title, available_globally="Yes",
                         release_date=pd.Timestamp(rel), hours_viewed=views * 5, runtime_min=300.0, views=views))
    # S1 released early 2024H2, S2 released late 2024H2 -> S1's release-semester views are NOT yet known
    r("2024H2", "Same Sem: Season 1", "2024-07-05", 5e6)
    r("2024H2", "Same Sem: Season 2", "2024-12-20", 9e6)
    # S1 in 2024H1, S2 in 2025H1 -> S1's release semester ended before S2's release
    r("2024H1", "Earlier: Season 1", "2024-02-01", 4e6)
    r("2025H1", "Earlier: Season 2", "2025-03-01", 8e6)
    r("2025H1", "Earlier: Season 1", "2024-02-01", 1e6)  # post-release for S2: must not be used
    s = build_silver(pd.DataFrame(rows))
    c = F.build_cohort(s, _meta(s)).set_index("title_raw")
    assert np.isnan(c.loc["Same Sem: Season 2", "prior_season_log_views"])
    assert c.loc["Earlier: Season 2", "prior_season_log_views"] == pytest.approx(np.log(4e6))


def test_exposure_normalisation_partial_semester():
    assert exposure_days(pd.Timestamp("2024-06-20"), "2024H1") == 11  # Jun 20..30 inclusive
    assert exposure_days(pd.Timestamp("2023-01-01"), "2024H1") == 182  # 2024 is a leap year
    assert exposure_days(pd.Timestamp("2024-08-01"), "2024H1") == 0


def test_no_target_derived_columns_in_feature_list():
    banned = ("views_rel_sem", "hours_rel_sem", "y", "views", "hours")
    assert not [f for f in F.FEATURES_NUM + F.FEATURES_CAT if f in banned]


def test_parse_title():
    p = parse_title("Dear Child: Limited Series // Liebes Kind: Miniserie")
    assert p["is_limited"] and p["season_no"] == 1 and p["series_base"] == "Dear Child"
    p = parse_title("Wednesday: Season 2")
    assert p["season_no"] == 2 and p["series_base"] == "Wednesday"
    assert parse_title("Leave the World Behind")["season_no"] is None
