"""Performance score: predict release-semester views from pre-release features only.

Temporal evaluation: for a test semester T, train on cohorts released before T-1, calibrate the
conformal interval on cohort T-1, and score cohort T. Baselines: genre median, prior season.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from streaming_content.netflix.features import FEATURES_CAT, FEATURES_NUM
from streaming_content.netflix.parse import SEMESTERS

LGB_PARAMS = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1,
)


def design(df: pd.DataFrame, cats: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, dict]:
    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    cats = cats or {c: sorted(df[c].dropna().unique().tolist()) for c in FEATURES_CAT}
    for c in FEATURES_CAT:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    for c in FEATURES_NUM:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, cats


def fit_lgbm(train: pd.DataFrame) -> tuple[lgb.LGBMRegressor, dict]:
    X, cats = design(train)
    m = lgb.LGBMRegressor(**LGB_PARAMS).fit(X, train["y"])
    return m, cats


def genre_median_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    med = train.groupby(["kind", "primary_genre"])["y"].median()
    kind_med = train.groupby("kind")["y"].median()
    out = []
    for k, g in zip(test["kind"], test["primary_genre"]):
        out.append(med.get((k, g), kind_med.get(k, train["y"].median())))
    return np.asarray(out)


def prior_season_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Prior season's log views where it exists, else the genre-median prediction."""
    fallback = genre_median_baseline(train, test)
    p = test["prior_season_log_views"].to_numpy(dtype=float)
    return np.where(np.isnan(p), fallback, p)


def conformal_halfwidth(resid_abs: np.ndarray, alpha: float) -> float:
    n = len(resid_abs)
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(resid_abs, q, method="higher"))


def top_decile_precision(y: np.ndarray, yhat: np.ndarray) -> float:
    k = max(1, int(round(len(y) * 0.1)))
    true_top = set(np.argsort(-y)[:k])
    pred_top = set(np.argsort(-yhat)[:k])
    return len(true_top & pred_top) / k


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    raw_err = np.abs(np.exp(yhat) - np.exp(y)) / np.exp(y)
    return {
        "spearman": float(spearmanr(y, yhat).statistic),
        "top_decile_precision": top_decile_precision(y, yhat),
        "mae_log": float(np.mean(np.abs(yhat - y))),
        "mape_log_scale": float(np.mean(np.abs(yhat - y) / np.abs(y))),
        "median_ape_views": float(np.median(raw_err)),
    }


def evaluate_fold(cohort: pd.DataFrame, test_sem: str, alpha: float = 0.2) -> dict:
    i = SEMESTERS.index(test_sem)
    cal_sem = SEMESTERS[i - 1]
    train = cohort[cohort["rel_sem"].map(SEMESTERS.index) < i - 1]
    cal = cohort[cohort["rel_sem"] == cal_sem]
    test = cohort[cohort["rel_sem"] == test_sem]
    model, cats = fit_lgbm(train)
    Xc, _ = design(cal, cats)
    Xt, _ = design(test, cats)
    hw = conformal_halfwidth(np.abs(cal["y"].to_numpy() - model.predict(Xc)), alpha)
    pred = model.predict(Xt)
    y = test["y"].to_numpy()
    res = {
        "test_semester": test_sem, "n_train": len(train), "n_cal": len(cal), "n_test": len(test),
        "lightgbm": metrics(y, pred),
        "genre_median": metrics(y, genre_median_baseline(train, test)),
        "prior_season": metrics(y, prior_season_baseline(train, test)),
        "conformal": {
            "target_coverage": 1 - alpha, "halfwidth_log": hw,
            "empirical_coverage": float(np.mean(np.abs(y - pred) <= hw)),
        },
    }
    has_prior = test["prior_season_log_views"].notna().to_numpy()
    if has_prior.sum() >= 10:
        res["on_titles_with_prior_season"] = {
            "n": int(has_prior.sum()),
            "lightgbm": metrics(y[has_prior], pred[has_prior]),
            "prior_season": metrics(y[has_prior], test["prior_season_log_views"].to_numpy()[has_prior]),
        }
    return res | {"_pred": pred, "_model": model, "_test": test, "_cats": cats}
