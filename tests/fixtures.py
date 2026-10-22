"""Small synthetic Netflix bronze table with known structure (no network, no real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from streaming_content.netflix.parse import SEMESTERS, semester_bounds, semester_days


def synthetic_bronze(seed: int = 0, n_series: int = 30, n_films: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    def add(title, kind, rel, level, decay, runtime):
        for sem in SEMESTERS:
            start, end = semester_bounds(sem)
            if rel > end:
                continue
            exposure = semester_days(sem) if rel <= start else (end - rel).days + 1
            age_mid = max((start - rel).days, 0) + exposure / 2
            views = level * np.exp(-decay * age_mid) * exposure * rng.lognormal(0, 0.1)
            views = round(views / 1e5) * 1e5
            if views < 1e5:
                continue
            rows.append(dict(semester=sem, sheet_kind=kind, title_raw=title, available_globally="Yes",
                             release_date=rel, hours_viewed=views * runtime / 60.0, runtime_min=runtime,
                             views=views))

    for i in range(n_series):
        base = f"Show {i}"
        rel1 = pd.Timestamp("2023-01-10") + pd.Timedelta(days=int(rng.integers(0, 600)))
        level = rng.lognormal(10, 1)
        for s in (1, 2):
            rel = rel1 + pd.Timedelta(days=330 * (s - 1))
            if rel > pd.Timestamp("2026-06-30"):
                continue
            add(f"{base}: Season {s}", "tv", rel, level * (1.2 if s == 2 else 1), 0.01, 400)
    for i in range(n_films):
        rel = pd.Timestamp("2023-01-05") + pd.Timedelta(days=int(rng.integers(0, 1250)))
        add(f"Film {i}", "movie", rel, rng.lognormal(10.5, 1), 0.015, 100)
    df = pd.DataFrame(rows)
    # 2023H1 is a single, untyped sheet without runtime/views in the real file
    h1 = df["semester"] == "2023H1"
    df.loc[h1, ["sheet_kind", "runtime_min", "views"]] = ["unknown", np.nan, np.nan]
    return df
