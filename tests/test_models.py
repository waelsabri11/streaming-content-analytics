import numpy as np
import pandas as pd
import pytest

from streaming_content.netflix import imdb, score
from streaming_content.netflix import lifecurves as L
from streaming_content.twitch.segmentation import adjusted_rand_index
from streaming_content.twitch.segments_job import choose_k, name_segments


def test_ari_perfect_and_independent():
    perfect = np.array([[10, 0], [0, 20]])
    assert adjusted_rand_index(perfect) == pytest.approx(1.0)
    independent = np.array([[25, 25], [25, 25]])
    assert abs(adjusted_rand_index(independent)) < 0.02
    relabelled = np.array([[0, 10], [20, 0]])
    assert adjusted_rand_index(relabelled) == pytest.approx(1.0)


def test_choose_k_rule():
    sweep = [{"k": 4, "silhouette": 0.30, "bootstrap_ari_mean": 0.9}, {"k": 5, "silhouette": 0.40, "bootstrap_ari_mean": 0.7},
             {"k": 6, "silhouette": 0.35, "bootstrap_ari_mean": 0.85}]
    assert choose_k(sweep) == 6  # k=5 has the best silhouette but fails the stability floor
    assert choose_k([{"k": 4, "silhouette": 0.9, "bootstrap_ari_mean": 0.5}, {"k": 5, "silhouette": 0.1, "bootstrap_ari_mean": 0.6}]) == 5


def test_name_segments_thresholds_and_uniqueness():
    prof = pd.DataFrame({
        "segment": [0, 1, 2, 3], "n_viewers": [100, 100, 100, 100], "hours": [50.0, 20.0, 5.0, 5.0],
        "top1_share": [0.2, 0.3, 0.95, 0.3], "mean_session_min": [40.0, 70.0, 40.0, 40.0],
        "share_night": [0.25, 0.25, 0.25, 0.25], "share_morning": [0.25, 0.25, 0.25, 0.25],
        "share_afternoon": [0.25, 0.25, 0.25, 0.25], "share_evening": [0.25, 0.25, 0.25, 0.25]})
    prof.loc[3, ["share_evening", "share_night"]] = [0.7, -0.2 + 0.25]
    names = name_segments(prof)
    assert names[0] == "Heavy viewers"
    assert names[1] == "Regular long-session viewers"
    assert names[2] == "Light single-channel viewers"
    assert "clock 18-24h" in names[3] and names[3].startswith("Light")
    assert len(set(names.values())) == 4


def test_power_shape_recovered():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(300):
        a = rng.normal(10, 1)
        for k, t in enumerate([10, 120, 300, 480]):
            rows.append(dict(title_key=f"t{i}", kind="film", k=k, t_mid=t, at_floor=False,
                             log_rate=a - 1.3 * np.log1p(t / 7) + rng.normal(0, 0.05), rate=1.0, rel_sem="2023H1",
                             release_date=pd.Timestamp("2023-01-01"), views=1.0))
    d = pd.DataFrame(rows)
    assert L.fit_shape(d, "power", 7.0) == pytest.approx(1.3, abs=0.05)
    assert L.best_tau(d) == 7.0


def test_sem_label_and_future_window():
    assert L.sem_label("2024H2", 1) == "2025H1" and L.sem_label("2024H1", 3) == "2025H2"
    t, e = L.future_window(pd.Timestamp("2024-06-20"), "2024H1", 0)
    assert e == 11 and t == pytest.approx(5.0)


def test_conformal_halfwidth_and_coverage():
    rng = np.random.default_rng(1)
    cal, test = np.abs(rng.normal(0, 1, 2000)), np.abs(rng.normal(0, 1, 5000))
    hw = score.conformal_halfwidth(cal, 0.2)
    assert (test <= hw).mean() == pytest.approx(0.8, abs=0.03)


def test_metrics_and_top_decile():
    y = np.arange(100.0)
    assert score.top_decile_precision(y, y) == 1.0
    assert score.top_decile_precision(y, -y) == 0.0
    assert score.metrics(y + 1, y + 1)["spearman"] == pytest.approx(1.0)


def test_imdb_match_by_year_and_type():
    basics = pd.DataFrame({
        "tconst": ["t1", "t2", "t3"], "titleType": ["movie", "movie", "tvSeries"],
        "primaryTitle": ["Heat", "Heat", "Heat"], "originalTitle": ["Heat", "Heat", "Heat"],
        "isAdult": [0, 0, 0], "startYear": [1995, 2023, 2010], "runtimeMinutes": [170, 100, 45], "genres": ["Crime", "Drama", "Drama"],
        "is_series": [False, False, True]})
    idx = imdb.build_index(basics)
    assert imdb.match_title("Heat", None, 2023, False, idx)[0] == "t2"
    assert imdb.match_title("Heat", None, 2024, True, idx)[0] == "t3"
    assert imdb.match_title("Not There", None, 2023, False, idx) == (None, "no_name_match")
    assert imdb.match_title("Heat (1995)", None, 2023, False, idx)[0] == "t1"
