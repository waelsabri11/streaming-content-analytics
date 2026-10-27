"""A/B test design on Twitch viewer data: variance, CUPED, power/MDE, sequential testing - all SIMULATED.

No experiment was run. Twitch gives (a) the real variance structure of a weekly engagement metric and of its
pre-period counterpart and (b) a population to resample from; the treatment effect is injected synthetically.
Unit of randomisation: the viewer. Pre-period: slots [0, 3 weeks); post (experiment) period: week 3.
"""

from __future__ import annotations

import json
from statistics import NormalDist

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from streaming_content import config
from streaming_content.timing import timed

WEEK = 7 * config.SLOTS_PER_DAY
PRE_END, POST_END = 3 * WEEK, 4 * WEEK
N = NormalDist()


# ---------------------------------------------------------------- pure statistics (unit tested)
def cuped_adjust(y: np.ndarray, x: np.ndarray, theta: float | None = None) -> tuple[np.ndarray, float]:
    """y - theta (x - mean(x)), theta = cov(x, y) / var(x) estimated on the pooled sample (both arms)."""
    if theta is None:
        theta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    return y - theta * (x - x.mean()), theta


def diff_in_means(y: np.ndarray, t: np.ndarray) -> tuple[float, float, float]:
    """(effect, standard error, two-sided p-value) for treatment minus control."""
    a, b = y[t == 1], y[t == 0]
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    d = float(a.mean() - b.mean())
    return d, se, 2 * (1 - N.cdf(abs(d) / se))


def mde(sigma: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable absolute effect for a two-arm, equal-split, two-sided z-test."""
    return (N.inv_cdf(1 - alpha / 2) + N.inv_cdf(power)) * sigma * np.sqrt(2.0 / n_per_arm)


def n_per_arm(sigma: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    return int(np.ceil(2 * ((N.inv_cdf(1 - alpha / 2) + N.inv_cdf(power)) * sigma / delta) ** 2))


def msprt_pvalues(d: np.ndarray, v: np.ndarray, tau2: float) -> np.ndarray:
    """Always-valid p-values of a normal mixture SPRT (Johari et al.).

    d: running effect estimates, v: their sampling variances at each look, tau2: prior variance of the effect.
    Lambda = sqrt(v / (v + tau2)) * exp(tau2 d^2 / (2 v (v + tau2))); p_t = min(1, min_{s<=t} 1/Lambda_s).
    """
    lam = np.sqrt(v / (v + tau2)) * np.exp(tau2 * d**2 / (2 * v * (v + tau2)))
    return np.minimum(1.0, np.minimum.accumulate(1.0 / lam))


# ---------------------------------------------------------------- simulation
def simulate_aa_ab(pop: pd.DataFrame, n_per_arm_sim: int, effect_rel: float, reps: int, seed: int = 0) -> dict:
    """Repeatedly draw 2n viewers, randomise, (optionally) inject a relative effect on post hours, test naive vs CUPED."""
    rng = np.random.default_rng(seed)
    y_all, x_all = pop["post_hours"].to_numpy(float), pop["pre_hours"].to_numpy(float)
    out = {k: {sc: {"p": [], "d": [], "se": []} for sc in ("aa", "ab")} for k in ("naive", "cuped")}
    for scenario, eff in (("aa", 0.0), ("ab", effect_rel)):
        for _ in range(reps):
            idx = rng.choice(len(y_all), size=2 * n_per_arm_sim, replace=False)
            y, x = y_all[idx].copy(), x_all[idx]
            t = np.zeros(len(idx), dtype=int)
            t[rng.choice(len(idx), size=n_per_arm_sim, replace=False)] = 1
            y = y + t * eff * y  # injected, SYNTHETIC multiplicative effect on post-period hours
            for name, yy in (("naive", y), ("cuped", cuped_adjust(y, x)[0])):
                d, se, p = diff_in_means(yy, t)
                for key, val in (("p", p), ("d", d), ("se", se)):
                    out[name][scenario][key].append(val)
    res: dict = {"n_per_arm": n_per_arm_sim, "reps": reps, "injected_relative_effect": effect_rel, "population_mean_post_hours": float(y_all.mean())}
    for name in ("naive", "cuped"):
        aa, ab = out[name]["aa"], out[name]["ab"]
        res[name] = {
            "aa_false_positive_rate": float(np.mean(np.array(aa["p"]) < 0.05)),
            "aa_mean_se": float(np.mean(aa["se"])),
            "ab_power": float(np.mean(np.array(ab["p"]) < 0.05)),
            "ab_mean_estimated_effect": float(np.mean(ab["d"])),
            "ab_true_effect": float(effect_rel * y_all.mean()),
            "ab_ci95_coverage_of_true_effect": float(np.mean(np.abs(np.array(ab["d"]) - effect_rel * y_all.mean()) <= 1.96 * np.array(ab["se"]))),
        }
    res["variance_reduction_se_ratio_squared"] = float(res["cuped"]["aa_mean_se"] ** 2 / res["naive"]["aa_mean_se"] ** 2)
    res["pre_post_correlation"] = float(np.corrcoef(x_all, y_all)[0, 1])
    return res


def simulate_sequential(pop: pd.DataFrame, n_max_per_arm: int, effect_rel: float, looks: int, reps: int, seed: int = 1) -> dict:
    """Peeking at `looks` equally spaced interim analyses: naive fixed-horizon p<0.05 at any look vs mSPRT."""
    rng = np.random.default_rng(seed)
    y_all, x_all = pop["post_hours"].to_numpy(float), pop["pre_hours"].to_numpy(float)
    sigma2 = float(np.var(y_all - np.cov(x_all, y_all)[0, 1] / np.var(x_all) * (x_all - x_all.mean())))
    tau2 = (0.02 * y_all.mean()) ** 2  # prior: effects of order 2% of the mean
    sizes = (np.arange(1, looks + 1) * n_max_per_arm // looks).astype(int)
    res = {}
    for scenario, eff in (("aa", 0.0), ("ab", effect_rel)):
        naive_hit = msprt_hit = 0
        stop_at = []
        for _ in range(reps):
            idx = rng.choice(len(y_all), size=2 * n_max_per_arm, replace=False)
            y, x = y_all[idx].copy(), x_all[idx]
            t = np.r_[np.ones(n_max_per_arm, dtype=int), np.zeros(n_max_per_arm, dtype=int)][rng.permutation(2 * n_max_per_arm)]
            y = y + t * eff * y
            yc, _ = cuped_adjust(y, x)
            ds, vs, ps = [], [], []
            for n in sizes:  # first n viewers of each arm, in arrival order
                a = yc[t == 1][:n]
                b = yc[t == 0][:n]
                d = a.mean() - b.mean()
                ds.append(d)
                vs.append(sigma2 * 2 / n)
                ps.append(diff_in_means(np.r_[a, b], np.r_[np.ones(n, int), np.zeros(n, int)])[2])
            naive_hit += bool(np.min(ps) < 0.05)
            pv = msprt_pvalues(np.array(ds), np.array(vs), tau2)
            hit = np.nonzero(pv < 0.05)[0]
            msprt_hit += bool(len(hit))
            if len(hit):
                stop_at.append(int(sizes[hit[0]]))
        res[scenario] = {"naive_peeking_reject_rate": naive_hit / reps, "msprt_reject_rate": msprt_hit / reps,
                         "msprt_median_stop_n_per_arm": float(np.median(stop_at)) if stop_at else None}
    res.update({"looks": looks, "n_max_per_arm": n_max_per_arm, "reps": reps, "injected_relative_effect": effect_rel, "tau2_prior": tau2})
    return res


def run(spark: SparkSession, sample_viewers: int = 3_000_000) -> dict:
    s = spark.table(config.table("silver", "twitch_view_sessions"))
    hours = F.col("new_watch_slots") * config.SLOT_MINUTES / 60.0
    with timed("ab_population") as t:
        pre = s.filter(F.col("start_slot") < PRE_END).groupBy("user_id").agg(F.sum(hours).alias("pre_hours"))
        post = s.filter((F.col("start_slot") >= PRE_END) & (F.col("start_slot") < POST_END)).groupBy("user_id").agg(F.sum(hours).alias("post_hours"))
        # population: viewers active in the pre-period (a treatment cannot be assigned to someone we never saw)
        pop = pre.join(post, "user_id", "left").fillna({"post_hours": 0.0}).withColumn("returned", (F.col("post_hours") > 0).cast("int"))
        n_pop = pop.count()
        stats = pop.agg(F.count("*").alias("n"), F.avg("post_hours").alias("mu_post"), F.stddev("post_hours").alias("sd_post"),
                        F.avg("pre_hours").alias("mu_pre"), F.stddev("pre_hours").alias("sd_pre"), F.corr("pre_hours", "post_hours").alias("rho"),
                        F.avg("returned").alias("return_rate"), F.expr("percentile_approx(post_hours, array(0.5, 0.9, 0.99, 0.999))").alias("q")).first().asDict()
        frac = min(1.0, sample_viewers / n_pop)
        sample = pop.sample(False, frac, 11).toPandas()
        t["viewers"] = n_pop
    res: dict = {"population": {"definition": "viewers with any activity in slots [0, 3 weeks); post = week 3", "n": n_pop,
                                **{k: (float(v) if not isinstance(v, list) else [float(x) for x in v]) for k, v in stats.items() if k != "n"}},
                 "sample_size_for_simulation": len(sample)}
    pop_pd = sample
    sigma = float(stats["sd_post"])
    res["mde_naive_relative"] = {str(n): float(mde(sigma, n) / stats["mu_post"]) for n in (10_000, 100_000, 500_000)}
    sig_c = sigma * np.sqrt(1 - stats["rho"] ** 2)
    res["mde_cuped_relative"] = {str(n): float(mde(sig_c, n) / stats["mu_post"]) for n in (10_000, 100_000, 500_000)}
    res["n_per_arm_for_2pct_effect"] = {"naive": n_per_arm(sigma, 0.02 * stats["mu_post"]), "cuped": n_per_arm(sig_c, 0.02 * stats["mu_post"])}
    with timed("ab_simulate") as t:
        res["aa_ab_simulation"] = simulate_aa_ab(pop_pd, n_per_arm_sim=20_000, effect_rel=0.03, reps=400)
        res["sequential"] = simulate_sequential(pop_pd, n_max_per_arm=40_000, effect_rel=0.03, looks=10, reps=300)
    res["label"] = "SIMULATION: injected synthetic effect on real Twitch variance; no real experiment was run."
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "ab_simulation.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    return res
