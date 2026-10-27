"""Streamlit restitution: title/channel page and segments page.

Reads small exported tables from reports/app_data (written by `python jobs/export_app_data.py`);
on Databricks Apps the same files ship with the app bundle. No raw data is needed here.
Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_here = Path(__file__).resolve().parent
# Databricks Apps ships app_data next to this file; locally it is exported to reports/app_data
DATA = Path(os.environ.get("SCA_APP_DATA", _here / "app_data" if (_here / "app_data").exists() else _here.parent / "reports" / "app_data"))
st.set_page_config(page_title="Streaming content analytics", layout="wide")


@st.cache_data
def load():
    d = {n: pd.read_parquet(DATA / f"{n}.parquet") for n in
         ["netflix_scores", "netflix_curves", "netflix_value_index", "twitch_segment_profile",
          "twitch_channel_daily_top", "twitch_segment_affinity"]}
    d["shapes"] = json.loads((DATA / "netflix_shapes.json").read_text())
    return d


def power_curve(a: float, g: float, tau: float, t: np.ndarray) -> np.ndarray:
    return np.exp(a - g * np.log1p(t / tau))


def title_page(d):
    st.header("Title performance, life curve and value")
    sc, cu, vi = d["netflix_scores"], d["netflix_curves"], d["netflix_value_index"]
    kind = st.sidebar.selectbox("Content kind", sorted(sc["kind"].unique()))
    pool = sc[sc["kind"] == kind].sort_values("release_date", ascending=False)
    title = st.selectbox("Title", pool["title_raw"].tolist())
    row = pool[pool["title_raw"] == title].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted release-semester views (pre-release model)", f"{np.exp(row['pred'])/1e6:,.1f} M")
    c2.metric("80% conformal interval", f"{np.exp(row['lo'])/1e6:,.1f} - {np.exp(row['hi'])/1e6:,.1f} M")
    c3.metric("Actual (reported)", f"{np.exp(row['y'])/1e6:,.1f} M")
    st.caption(f"Out-of-sample prediction for fold {row['fold']} (trained only on titles released two or more semesters earlier).")

    v = vi[vi["title_key"] == row["title_key"]]
    if len(v):
        v = v.iloc[0]
        st.metric("Value index (cumulative views over 4 semesters, relative to the median title of the kind)", f"{v['value_index']:.1f}x",
                  help="Observed + forecast. Forecast share: " + f"{v['forecast_views']/max(v['value_views'],1):.0%}")

    pts = cu[cu["title_key"] == row["title_key"]].sort_values("t_mid")
    sh = d["shapes"].get(kind)
    fig = go.Figure()
    fig.add_scatter(x=pts["t_mid"], y=pts["rate"], mode="markers+lines", name="reported views / exposure day")
    if sh and len(pts):
        a = float(np.mean(np.log(pts["rate"]) + sh["power_g"] * np.log1p(pts["t_mid"] / sh["tau"])))
        t = np.linspace(1, 730, 200)
        fig.add_scatter(x=t, y=power_curve(a, sh["power_g"], sh["tau"], t), name="fitted power-law decay (kind shape, title level)", line=dict(dash="dash"))
    fig.update_yaxes(type="log", title="views per exposure day")
    fig.update_xaxes(title="days since release (middle of exposure window)")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Comparables")
    comp = pool[(pool["primary_genre"] == row["primary_genre"]) & (pool["title_key"] != row["title_key"])].copy()
    comp["dist"] = (comp["y"] - row["y"]).abs()
    st.dataframe(comp.nsmallest(8, "dist")[["title_raw", "rel_sem", "primary_genre"]].assign(
        views_M=lambda x: np.exp(comp.loc[x.index, "y"]) / 1e6), hide_index=True)

    st.header("Twitch channel audience")
    cd = d["twitch_channel_daily_top"]
    ch = st.selectbox("Channel (top by viewer-days)", cd["streamer"].drop_duplicates().tolist())
    g = cd[cd["streamer"] == ch].sort_values("day")
    f2 = px.area(g, x="day", y=["new_viewers", "returning_viewers"], labels={"value": "viewers", "day": "day in window"})
    st.plotly_chart(f2, width="stretch")


def segments_page(d):
    st.header("Viewer segments (Twitch viewing logs)")
    st.info("Twitch is a proxy for streaming consumption: live chat presence on a different platform and audience, not subscriber viewing.")
    prof = d["twitch_segment_profile"]
    show = prof[["name", "n_viewers", "viewer_share", "hours_share", "hours", "n_sessions", "active_days", "n_streamers", "top1_share", "repeat_share", "share_night"]]
    st.dataframe(show.round(3), hide_index=True)
    z = prof.set_index("name")[["log_hours", "log_sessions", "log_streamers", "norm_entropy", "top1_share", "repeat_share", "share_night", "share_evening"]]
    z = (z - z.mean()) / z.std(ddof=0)
    st.plotly_chart(px.imshow(z, aspect="auto", color_continuous_scale="RdBu_r", zmin=-2, zmax=2, title="Segment means (z-scored across segments)"),
                    width="stretch")
    aff = d["twitch_segment_affinity"]
    seg = st.selectbox("Segment", prof["name"].tolist())
    sid = int(prof.loc[prof["name"] == seg, "segment"].iloc[0])
    st.subheader("Channels over-indexed in this segment (lift vs all viewers)")
    st.dataframe(aff[aff["segment"] == sid][["rk", "streamer", "lift", "viewers"]], hide_index=True)


def main():
    d = load()
    page = st.sidebar.radio("Page", ["Titles & channels", "Segments"])
    (title_page if page == "Titles & channels" else segments_page)(d)


main()
