"""Parse Netflix 'What We Watched' xlsx files into one tidy long table (pandas; files are small)."""

from __future__ import annotations

import re
from pathlib import Path

import openpyxl
import pandas as pd

SEMESTERS = ["2023H1", "2023H2", "2024H1", "2024H2", "2025H1", "2025H2", "2026H1"]

_SHEET_KIND = {"tv": "tv", "shows": "tv", "film": "movie", "movies": "movie", "engagement": "unknown"}


def semester_bounds(sem: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive first and last day of a semester label like '2024H2'."""
    y, h = int(sem[:4]), int(sem[-1])
    if h == 1:
        return pd.Timestamp(y, 1, 1), pd.Timestamp(y, 6, 30)
    return pd.Timestamp(y, 7, 1), pd.Timestamp(y, 12, 31)


def semester_days(sem: str) -> int:
    a, b = semester_bounds(sem)
    return (b - a).days + 1


def _runtime_minutes(x) -> float | None:
    if x is None or isinstance(x, (int, float)):
        return float(x) if x is not None else None
    m = re.fullmatch(r"\s*(\d+):(\d{1,2})\s*", str(x))
    return int(m[1]) * 60 + int(m[2]) if m else None


def parse_workbook(path: Path, semester: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        kind = _SHEET_KIND.get(ws.title.strip().lower(), "unknown")
        rows = list(ws.iter_rows(values_only=True))
        h = next(i for i, r in enumerate(rows[:15]) if r[1] == "Title")
        header = [str(c).strip() if c is not None else f"_c{j}" for j, c in enumerate(rows[h])]
        for r in rows[h + 1 :]:
            rec = dict(zip(header, r))
            hours = rec.get("Hours Viewed")
            if rec.get("Title") is None or not isinstance(hours, (int, float)):
                continue  # footnotes, blanks
            out.append(
                {
                    "semester": semester,
                    "sheet_kind": kind,
                    "title_raw": str(rec["Title"]).strip(),
                    "available_globally": rec.get("Available Globally?"),
                    "release_date": rec.get("Release Date"),
                    "hours_viewed": float(hours),
                    "runtime_min": _runtime_minutes(rec.get("Runtime")),
                    "views": float(rec["Views"]) if isinstance(rec.get("Views"), (int, float)) else None,
                }
            )
    df = pd.DataFrame(out)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    for c in ("runtime_min", "views"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def parse_all(netflix_dir: Path) -> pd.DataFrame:
    frames = [parse_workbook(netflix_dir / f"{s}.xlsx", s) for s in SEMESTERS]
    return pd.concat(frames, ignore_index=True)
