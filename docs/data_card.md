# Data card

Raw data is **never committed or redistributed**. `ingest/download.py` fetches each source and records
SHA-256 checksums in `ingest/checksums.json`. Everything below was read from the sources on 2026-10-22.

## Twitch viewing logs (core large-volume dataset)

| | |
|---|---|
| Source | McAuley lab datasets page (UCSD), "Twitch" section; files at `https://mcauleylab.ucsd.edu/public_datasets/gdrive/twitch/` |
| Paper | J. Rappaz, J. McAuley, K. Aberer. *Recommendation on Live-Streaming Platforms: Dynamic Availability and Repeat Consumption.* RecSys 2021 |
| Files used | `full_a.csv.gz` (6,943,679,726 bytes, last modified 2025-01-17), `100k_a.csv` (114,131,306 bytes; downloaded for development only) |
| Schema | `user_id` (anonymised int), `stream_id` (bigint), `streamer` (username), `start_slot`, `stop_slot` (ints, 10-minute slots; 0 .. 6148) |
| What a row is | A viewer was connected to a streamer's chat during slots `[start, stop)`. The authors crawled all live streamers and the users in their chats every 10 minutes for 43 days. |
| **Measured size** | **474,676,930 rows**, 15,524,309 viewers, 790,100 channels (see `reports/results.md`). |
| License | The page states no license; it asks that users cite the paper. Treated as research use with citation, no redistribution. |

**Discrepancy with the source page.** The datasets page lists the full dataset as "15.5M users, 465k streamers,
124M interactions". The file actually downloaded has 474.7M rows and 790k channels (the viewer count matches). The
task brief's figure of ~474M is the measured one; the page's interaction and streamer counts look stale. The 100k-viewer
sample file matches the page (3,051,733 rows, 100,000 viewers, 162,625 streamers).

**Known properties (measured)**: `stop > start` for every row, no duplicate rows, no row outside `[0, 6148]` (all
474,676,930 rows survive validation). About 6% of rows overlap another row of the same viewer (parallel streams), which is
why watch time is computed as a union. The last day of the window is partial (6,148 slots = 42.7 days).

**What it is not.** Presence in a live chat is not the same as watching, there is no calendar date or time zone for slot 0,
there are no content categories, and the platform, audience and consumption model (live, free, channel-centric) differ from a
subscription catalogue. Twitch is a **proxy** for consumption behaviour, not subscriber data.

## Netflix "What We Watched" engagement reports

| | |
|---|---|
| Source | `about.netflix.com` newsroom pages "What We Watched" (H1 2023 ... H1 2026), seven semi-annual `.xlsx` files linked from those pages (URLs in `ingest/download.py`) |
| Content | Per title and semester: hours viewed, runtime, views (= hours / runtime), release date, "available globally". Titles with fewer than 50k views are aggregated in "Other" rows. 2023 H1 has a single sheet with hours only. |
| License | No license in the files; public reports. Not redistributed; only aggregate results are in this repository (per-title derived tables are git-ignored). |

Quirks handled: titles are `English // local-language`; seasons are `Name: Season N` / `Limited Series`; live events and dated
programmes (`Raw: June 29, 2026`) are excluded; views are rounded to 100k so values at 100k form a censored floor; release dates
exist for roughly half of the shows and a fifth of the films (licensed titles mostly have none).

## IMDb non-commercial datasets

`title.basics.tsv.gz` and `title.episode.tsv.gz` from `datasets.imdbws.com`, used under IMDb's personal / non-commercial terms.
Only fields fixed at release are read (type, genres, start year, runtime, season episode count). **No ratings or votes** are read
(they are post-release). The files are a current snapshot, not a release-date snapshot.
