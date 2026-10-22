"""Title life curves: views per exposure day against days since release, and the value index.

Curve unit: for title i observed in semester k, rate_ik = views_ik / exposure_days_ik, located at
t_ik = days since release at the middle of the exposed window. Two parametric families are fitted
per content kind on log(rate):
    exponential:  log r = a - b t
    power:        log r = a - g * log(1 + t / tau)        (tau chosen on a grid)
Titles keep their own level a_i (random-intercept style): the shared shape (b or g) is estimated
from within-title changes, then each title's level is re-estimated and shrunk toward the kind mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from streaming_content.netflix.parse import SEMESTERS, semester_bounds
from streaming_content.netflix.silver import semester_of

FLOOR_VIEWS = 100_000  # Netflix reports views rounded to 100k; at/below is a censored floor
TAUS = [1.0, 3.0, 7.0, 14.0, 30.0, 60.0, 120.0]


def build_curve_table(silver: pd.DataFrame, min_release: str = "2023-01-01") -> pd.DataFrame:
    s = silver[~silver["event_like"] & silver["release_date"].notna() & silver["views"].notna()].copy()
    s = s[s["release_date"] >= pd.Timestamp(min_release)]
    s["rel_sem"] = s["release_date"].map(semester_of)
    s["k"] = s["sem_idx"] - s["rel_sem"].map({x: i for i, x in enumerate(SEMESTERS)})
    s = s[s["k"] >= 0].copy()
    ends = s["semester"].map(lambda x: semester_bounds(x)[1])
    starts = s["semester"].map(lambda x: semester_bounds(x)[0])
    win_start = np.maximum(starts.values, s["release_date"].values)
    mid = win_start + (ends.values - win_start) / 2
    s["t_mid"] = (pd.to_datetime(mid) - s["release_date"]).dt.total_seconds() / 86400.0
    s["rate"] = s["views"] / s["exposure_days"].clip(lower=1)
    s["log_rate"] = np.log(s["rate"])
    s["at_floor"] = s["views"] <= FLOOR_VIEWS
    s["kind"] = np.where(s["season_no"].notna(), "series_season", np.where(s["content_type"] == "movie", "film", "tv_other"))
    return s[["title_key", "title_raw", "kind", "release_date", "rel_sem", "semester", "k", "t_mid",
              "exposure_days", "views", "rate", "log_rate", "at_floor"]].reset_index(drop=True)


def _basis(t: np.ndarray, family: str, tau: float) -> np.ndarray:
    return t if family == "exponential" else np.log1p(t / tau)


def fit_shape(curves: pd.DataFrame, family: str, tau: float = 30.0) -> float:
    """Slope of log-rate on the basis using within-title variation (title fixed effects)."""
    d = curves[~curves["at_floor"]].copy()
    d["x"] = _basis(d["t_mid"].to_numpy(), family, tau)
    d = d.groupby("title_key").filter(lambda g: len(g) >= 2)
    if d.empty:
        return float("nan")
    xc = d["x"] - d.groupby("title_key")["x"].transform("mean")
    yc = d["log_rate"] - d.groupby("title_key")["log_rate"].transform("mean")
    return float(-(xc * yc).sum() / (xc * xc).sum())  # positive = decay


def best_tau(curves: pd.DataFrame) -> float:
    best, best_sse = TAUS[0], np.inf
    d = curves[~curves["at_floor"]]
    d = d.groupby("title_key").filter(lambda g: len(g) >= 2)
    for tau in TAUS:
        g = fit_shape(d, "power", tau)
        x = _basis(d["t_mid"].to_numpy(), "power", tau)
        xc = x - d.assign(x=x).groupby("title_key")["x"].transform("mean").to_numpy()
        yc = d["log_rate"] - d.groupby("title_key")["log_rate"].transform("mean")
        sse = float(((yc.to_numpy() + g * xc) ** 2).sum())
        if sse < best_sse:
            best, best_sse = tau, sse
    return best


def level(curves_i: pd.DataFrame, shape: float, family: str, tau: float) -> float:
    x = _basis(curves_i["t_mid"].to_numpy(), family, tau)
    return float(np.mean(curves_i["log_rate"].to_numpy() + shape * x))


def predict_log_rate(a: float, shape: float, t: np.ndarray, family: str, tau: float) -> np.ndarray:
    return a - shape * _basis(t, family, tau)


def cumulative_views(a: float, shape: float, family: str, tau: float, horizon_days: float, step: float = 1.0) -> float:
    """Integral of the fitted rate over [0, horizon] days (rate in views/day)."""
    t = np.arange(0.5 * step, horizon_days, step)
    return float(np.exp(predict_log_rate(a, shape, t, family, tau)).sum() * step)


def sem_label(rel_sem: str, k: int) -> str:
    y, h = int(rel_sem[:4]), int(rel_sem[-1])
    idx = (h - 1) + k
    return f"{y + idx // 2}H{idx % 2 + 1}"


def future_window(release: pd.Timestamp, rel_sem: str, k: int) -> tuple[float, float]:
    """(t_mid days since release, exposure days) of semester rel_sem+k for a title."""
    start, end = semester_bounds(sem_label(rel_sem, k))
    ws = max(start, release)
    exposure = (end - ws).days + 1
    t_mid = ((ws + (end - ws) / 2) - release).total_seconds() / 86400.0
    return t_mid, float(exposure)


def fit_all_shapes(train: pd.DataFrame) -> dict:
    """Per kind: exponential rate b, power (g, tau), plus nonparametric median log-ratios k-1 -> k."""
    out = {}
    for kind, g in train.groupby("kind"):
        tau = best_tau(g)
        piv = g.pivot_table(index="title_key", columns="k", values="log_rate")
        ratios = {k: float((piv[k] - piv[k - 1]).dropna().median()) for k in piv.columns if k > 0 and k - 1 in piv.columns and (piv[k] - piv[k - 1]).notna().sum() >= 15}
        out[kind] = {"exp_b": fit_shape(g, "exponential"), "power_g": fit_shape(g, "power", tau), "tau": tau,
                     "median_log_ratio": ratios, "n_titles": int(g["title_key"].nunique())}
    return out


def forecast_views(obs: pd.DataFrame, release: pd.Timestamp, rel_sem: str, kind: str, shapes: dict,
                   method: str, horizon: int) -> float:
    """Sum of predicted views for semesters k = len(obs) .. horizon-1 given observed k = 0..len-1."""
    sh = shapes[kind]
    m = int(obs["k"].max()) + 1
    total = 0.0
    if method == "exponential":
        a = level(obs, sh["exp_b"], "exponential", sh["tau"])
    elif method == "power":
        a = level(obs, sh["power_g"], "power", sh["tau"])
    for k in range(m, horizon):
        t, e = future_window(release, rel_sem, k)
        if method in ("exponential", "power"):
            fam = method
            shape = sh["exp_b"] if fam == "exponential" else sh["power_g"]
            total += float(np.exp(predict_log_rate(a, shape, np.array([t]), fam, sh["tau"]))[0]) * e
        elif method == "persistence":
            last = obs.sort_values("k").iloc[-1]
            total += float(last["rate"]) * e
        elif method == "median_ratio":
            last = obs.sort_values("k").iloc[-1]
            lr = float(last["log_rate"])
            for kk in range(m, k + 1):
                lr += sh["median_log_ratio"].get(kk, 0.0)
            total += float(np.exp(lr)) * e
    return total
