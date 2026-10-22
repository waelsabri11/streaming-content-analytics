"""Download raw sources into data/raw/ and record SHA-256 checksums.

Nothing downloaded here is committed or redistributed (data/ is gitignored).
Usage: python ingest/download.py [twitch|twitch100k|netflix|imdb|all]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
MANIFEST = Path(__file__).resolve().parents[1] / "ingest" / "checksums.json"

TWITCH_BASE = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/twitch"
NETFLIX = {
    "2023H1": "https://assets.ctfassets.net/4cd45et68cgf/1HyknFM84ISQpeua6TjM7A/97a0a393098937a8f29c9d29c48dbfa8/What_We_Watched_A_Netflix_Engagement_Report_2023Jan-Jun.xlsx",
    "2023H2": "https://assets.ctfassets.net/4cd45et68cgf/inuAnzotdsAEgbInGLzH5/1be323ba419b2af3a96bffa29acc31a3/What_We_Watched_A_Netflix_Engagement_Report_2023Jul-Dec.xlsx",
    "2024H1": "https://assets.ctfassets.net/4cd45et68cgf/2PoZlfdc46dH2gQvI8eUzI/9db5840720c47acfcf7b89ffe2402860/What_We_Watched_A_Netflix_Engagement_Report_2024Jan-Jun.xlsx",
    "2024H2": "https://assets.ctfassets.net/4cd45et68cgf/6XSmoEjBjVMPRtYybT9d1E/8c0b0b2645b8712d5597b0bdbe0d64e2/What_We_Watched_A_Netflix_Engagement_Report_2024Jul-Dec.xlsx",
    "2025H1": "https://assets.ctfassets.net/4cd45et68cgf/mplcXj5ulHDfbCPCr0f0I/5dbb6ec09f03df89706476e380e9b8bd/What_We_Watched_A_Netflix_Engagement_Report_2025Jan-Jun.xlsx",
    "2025H2": "https://assets.ctfassets.net/4cd45et68cgf/2vdDPGLKA0XX2cjF2APJFn/7f1c367b39ed73a6a588751d3c5d0252/What_We_Watched_A_Netflix_Engagement_Report_2025Jul-Dec__6_.xlsx",
    "2026H1": "https://assets.ctfassets.net/4cd45et68cgf/40WGcHJa9vRua31kU6Gbz5/43b15c7fbd6924392fc80e885839e39f/Netflix-s_What_We_Watched_Report_2026Jan-Jun__1_.xlsx",
}
IMDB = ["title.basics.tsv.gz", "title.episode.tsv.gz"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)


def record(path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest[str(path.relative_to(RAW))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main(which: str = "all") -> None:
    jobs: list[tuple[str, Path]] = []
    if which in ("twitch100k", "all"):
        jobs.append((f"{TWITCH_BASE}/100k_a.csv", RAW / "twitch" / "100k_a.csv"))
    if which in ("twitch", "all"):
        jobs.append((f"{TWITCH_BASE}/full_a.csv.gz", RAW / "twitch" / "full_a.csv.gz"))
    if which in ("netflix", "all"):
        jobs += [(u, RAW / "netflix" / f"{k}.xlsx") for k, u in NETFLIX.items()]
    if which in ("imdb", "all"):
        jobs += [(f"https://datasets.imdbws.com/{n}", RAW / "imdb" / n) for n in IMDB]
    for url, dest in jobs:
        print("fetch", url)
        fetch(url, dest)
        record(dest)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
