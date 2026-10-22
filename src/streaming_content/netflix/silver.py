"""Silver layer for Netflix reports: title parsing, content typing, exposure, view imputation."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from streaming_content.netflix.parse import SEMESTERS, semester_bounds, semester_days

_SEASON_RE = re.compile(
    r"^(?P<base>.+?): (?:Season|Series|Staffel|Saison|Temporada|Stagione|Part|Volume|Vol\.) (?P<n>\d+)$"
)
_LIMITED_SUFFIX = ": Limited Series"
_OTHER_ROWS = re.compile(r"^Other (Shows|Movies)$")
# Live events / recurring dated programmes are not "content" with a life curve.
_DATED = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r" \d{1,2}, \d{4}"
)
_EVENT = re.compile(
    r"^(Raw|WWE|F1 |NFL|NBA|MLB|UFC|Tyson|Jake Paul|Canelo|Love Is Blind: Live|2026 World|2023 World)"
)


def parse_title(title_raw: str) -> dict:
    """Split 'English // local' titles and extract series base / season number."""
    parts = [p.strip() for p in title_raw.split(" // ")]
    title_en = parts[0]
    alt = parts[1] if len(parts) > 1 else None
    season_no: int | None = None
    is_limited = False
    base = title_en
    if title_en.endswith(_LIMITED_SUFFIX):
        base, season_no, is_limited = title_en[: -len(_LIMITED_SUFFIX)], 1, True
    else:
        m = _SEASON_RE.match(title_en)
        if m:
            base, season_no = m["base"], int(m["n"])
    return {
        "title_en": title_en,
        "title_alt": alt,
        "series_base": base,
        "season_no": season_no,
        "is_limited": is_limited,
        "has_alt_title": alt is not None,
    }


def is_event_like(title_en: str) -> bool:
    return bool(_DATED.search(title_en) or _EVENT.match(title_en))


def semester_of(ts: pd.Timestamp) -> str:
    return f"{ts.year}H{1 if ts.month <= 6 else 2}"


def exposure_days(release_date: pd.Timestamp | None, sem: str) -> float:
    """Days the title could have been watched inside the semester (release day counts)."""
    start, end = semester_bounds(sem)
    full = semester_days(sem)
    if release_date is None or pd.isna(release_date) or release_date <= start:
        return float(full)
    if release_date > end:
        return 0.0
    return float((end - release_date).days + 1)


def build_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    """One row per (title, semester) with parsed attributes, imputed views and exposure.

    * 'Other Shows/Movies' aggregate rows are dropped (they are not titles).
    * 2023H1 has a single sheet and no runtime/views: type and runtime are taken from the same
      title's later rows, views are derived as hours / runtime (Netflix's own definition).
    """
    df = bronze[~bronze["title_raw"].str.match(_OTHER_ROWS)].copy()
    parsed = pd.DataFrame([parse_title(t) for t in df["title_raw"].unique()])
    parsed["title_raw"] = df["title_raw"].unique()
    df = df.merge(parsed, on="title_raw", how="left")

    # runtime and content type from any semester where the sheet reveals them
    known = df[df["sheet_kind"] != "unknown"]
    runtime_ref = known.dropna(subset=["runtime_min"]).groupby("title_raw")["runtime_min"].max()
    kind_ref = known.groupby("title_raw")["sheet_kind"].agg(lambda s: s.mode().iat[0])
    df["runtime_min"] = df["runtime_min"].fillna(df["title_raw"].map(runtime_ref))
    unk = df["sheet_kind"] == "unknown"
    df.loc[unk, "sheet_kind"] = df.loc[unk, "title_raw"].map(kind_ref)
    still = df["sheet_kind"].isna()
    df.loc[still, "sheet_kind"] = np.where(df.loc[still, "season_no"].notna(), "tv", "movie")
    df = df.rename(columns={"sheet_kind": "content_type"})

    df["views"] = pd.to_numeric(df["views"], errors="coerce")
    derived = df["hours_viewed"] * 60.0 / df["runtime_min"]
    df["views_imputed"] = df["views"].isna() & derived.notna()
    df["views"] = df["views"].fillna(derived)

    df["available_globally"] = df["available_globally"].eq("Yes")
    df["event_like"] = df["title_en"].map(is_event_like)
    df["title_key"] = df["title_raw"] + "|" + df["release_date"].dt.strftime("%Y-%m-%d").fillna("NA")
    df["exposure_days"] = [exposure_days(r, s) for r, s in zip(df["release_date"], df["semester"])]
    df["sem_idx"] = df["semester"].map({s: i for i, s in enumerate(SEMESTERS)})
    return df.reset_index(drop=True)
