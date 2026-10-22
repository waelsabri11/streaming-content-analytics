"""Match Netflix titles to IMDb non-commercial datasets and derive pre-release metadata.

IMDb files are a current snapshot. We only read fields that are properties of the title fixed at
release (type, genres, start year, runtime, season episode count) and never ratings or votes.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

BASIC_TYPES = {"movie", "tvMovie", "tvSeries", "tvMiniSeries", "tvSpecial"}
_YEAR_TAG = re.compile(r"\s*\((\d{4})\)\s*$")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s.replace("&", " and "))
    return re.sub(r"\s+", " ", s).strip()


def load_basics(path: Path, min_year: int = 1990) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        na_values="\\N",
        usecols=["tconst", "titleType", "primaryTitle", "originalTitle", "isAdult",
                 "startYear", "runtimeMinutes", "genres"],
        dtype={"tconst": "string", "titleType": "string", "primaryTitle": "string",
               "originalTitle": "string", "genres": "string"},
        quoting=3,
    )
    df = df[df["titleType"].isin(BASIC_TYPES)]
    df = df[df["startYear"].fillna(0).astype(float) >= min_year].copy()
    df["startYear"] = df["startYear"].astype(int)
    df["is_series"] = df["titleType"].isin(["tvSeries", "tvMiniSeries"])
    return df


class Index:
    """Normalized-name lookup over IMDb basics (primary and original titles)."""

    def __init__(self, basics: pd.DataFrame):
        a = basics.assign(key=basics["primaryTitle"].fillna("").map(norm))
        b = basics.assign(key=basics["originalTitle"].fillna("").map(norm))
        frame = pd.concat([a, b]).drop_duplicates(["tconst", "key"])
        self.frame = frame[frame["key"] != ""].reset_index(drop=True)
        self.rows = self.frame.groupby("key").indices

    def __contains__(self, key: str) -> bool:
        return key in self.rows

    def __getitem__(self, key: str) -> pd.DataFrame:
        return self.frame.iloc[self.rows[key]]


def build_index(basics: pd.DataFrame) -> Index:
    return Index(basics)


def match_title(
    base: str, alt: str | None, release_year: int, is_series: bool, index: Index
) -> tuple[str | None, str]:
    """Return (tconst, reason). Series match on series name with startYear <= release year."""
    yr = None
    m = _YEAR_TAG.search(base)
    if m:
        yr, base = int(m[1]), _YEAR_TAG.sub("", base)
    keys = [norm(base)] + ([norm(alt)] if isinstance(alt, str) and alt else [])
    cands = pd.concat([index[k] for k in keys if k in index]) if any(k in index for k in keys) else None
    if cands is None:
        return None, "no_name_match"
    cands = cands.drop_duplicates("tconst")
    cands = cands[cands["is_series"] == is_series] if is_series else cands[~cands["is_series"]]
    if is_series:
        cands = cands[cands["startYear"] <= release_year + 1]
        cands = cands[cands["startYear"] >= release_year - 25]
    else:
        target = yr if yr else release_year
        cands = cands[(cands["startYear"] - target).abs() <= (0 if yr else 1)]
    if cands.empty:
        return None, "year_mismatch"
    if len(cands) == 1:
        return str(cands["tconst"].iat[0]), "unique"
    if is_series:  # several series with the same name: keep the one that started latest before release
        cands = cands.sort_values("startYear", ascending=False)
        top = cands[cands["startYear"] == cands["startYear"].iat[0]]
    else:
        d = (cands["startYear"] - release_year).abs()
        top = cands[d == d.min()]
    if len(top) == 1:
        return str(top["tconst"].iat[0]), "disambiguated_year"
    return None, "ambiguous"


def season_episode_counts(episode_path: Path, parents: set[str]) -> pd.DataFrame:
    ep = pd.read_csv(episode_path, sep="\t", na_values="\\N", dtype="string", quoting=3)
    ep = ep[ep["parentTconst"].isin(parents)]
    ep["seasonNumber"] = pd.to_numeric(ep["seasonNumber"], errors="coerce")
    return (
        ep.groupby(["parentTconst", "seasonNumber"]).size().rename("n_episodes").reset_index()
    )
