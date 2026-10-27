"""Figures for the README (docs/img/*.png), drawn from reports/results/*.json and reports/app_data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, APP, IMG = ROOT / "reports" / "results", ROOT / "reports" / "app_data", ROOT / "docs" / "img"
RESD = RES / "databricks"  # figures show the Databricks results
IMG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
C = ["#2a6f97", "#c8553d", "#6a994e", "#8d6a9f", "#b08900"]


def segments():
    r = json.loads((RESD / "twitch_segments.json").read_text())
    p = pd.DataFrame(r["profile"]).sort_values("hours_share", ascending=False)
    cols = ["log_hours", "log_sessions", "log_streamers", "norm_entropy", "top1_share", "repeat_share", "share_night", "share_evening"]
    z = p.set_index("name")[cols]
    z = (z - z.mean()) / z.std(ddof=0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3), gridspec_kw={"width_ratios": [1.5, 1]})
    im = ax[0].imshow(z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax[0].set_xticks(range(len(cols)), [c.replace("share_night", "clock 00-06h share").replace("share_evening", "clock 18-24h share").replace("_", " ") for c in cols], rotation=35, ha="right")
    ax[0].set_yticks(range(len(z)), z.index)
    ax[0].set_title("Segment means (z across segments)")
    plt.colorbar(im, ax=ax[0], fraction=0.03)
    y = np.arange(len(p))
    ax[1].barh(y - 0.2, p["viewer_share"], 0.4, color=C[0], label="share of viewers")
    ax[1].barh(y + 0.2, p["hours_share"], 0.4, color=C[1], label="share of watch time")
    ax[1].set_yticks(y, [""] * len(p))
    ax[1].invert_yaxis()
    ax[1].legend(frameon=False)
    ax[1].set_title("Size")
    fig.tight_layout()
    fig.savefig(IMG / "segments.png")


def score():
    s = pd.read_parquet(APP / "netflix_scores.parquet")
    d = s[s["fold"] == "2026H1"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    inside = (d["y"] >= d["lo"]) & (d["y"] <= d["hi"])
    ax.scatter(np.exp(d["pred"]) / 1e6, np.exp(d["y"]) / 1e6, s=10, c=np.where(inside, C[0], C[1]), alpha=0.7)
    lim = [0.1, max(np.exp(d["y"]).max(), np.exp(d["pred"]).max()) / 1e6 * 1.2]
    ax.plot(lim, lim, color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("predicted views, release semester (M)")
    ax.set_ylabel("reported views (M)")
    ax.set_title(f"2026H1 hold-out, n={len(d)}; red = outside the 80% interval")
    fig.tight_layout()
    fig.savefig(IMG / "score_2026H1.png")


def curves():
    c = pd.read_parquet(APP / "netflix_curves.parquet")
    sh = json.loads((APP / "netflix_shapes.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, (kind, g) in enumerate(c[~c["at_floor"]].groupby("kind")):
        m = g.groupby("k").agg(t=("t_mid", "median"), r=("rate", "median"))
        ax[0].plot(m["t"], m["r"], "o-", color=C[i], label=f"{kind} (median)")
        a = np.median(np.log(g["rate"]) + sh[kind]["power_g"] * np.log1p(g["t_mid"] / sh[kind]["tau"]))
        t = np.linspace(5, 900, 100)
        ax[0].plot(t, np.exp(a - sh[kind]["power_g"] * np.log1p(t / sh[kind]["tau"])), "--", color=C[i], lw=0.8)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("days since release")
    ax[0].set_ylabel("views per exposure day")
    ax[0].legend(frameon=False)
    ax[0].set_title("Netflix life curves (dashed: fitted power law)\nlate points are survivors")
    r = json.loads((RESD / "twitch_lifecurves.json").read_text())
    for i, (tier, d) in enumerate(r["decay"].items()):
        ages = sorted(int(a) for a in d["returning_share_by_age"])
        ax[1].plot(ages, [d["returning_share_by_age"][str(a)] for a in ages], "o-", color=C[i], label=tier)
    ax[1].set_xlabel("channel age (days since first activity)")
    ax[1].set_ylabel("share of viewers already seen")
    ax[1].legend(frameon=False, title="first-week size")
    ax[1].set_title("Twitch new channels: returning share")
    fig.tight_layout()
    fig.savefig(IMG / "life_curves.png")


if __name__ == "__main__":
    segments()
    score()
    curves()
    print("figures written to", IMG)
