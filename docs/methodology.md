# Methodology note

Numbers live in [`reports/results.md`](../reports/results.md). This note records the decisions, the assumptions behind
them, the traps I tried to avoid, and what the analysis cannot support.

## Twitch is a proxy, not subscriber data
The Twitch logs show who sat in which live channel's chat every ten minutes for 43 days. A subscription catalogue
(CANAL+, Netflix) has on-demand titles, plans and content with a release date. What transfers is the *method*:
sessionization at scale, behavioural features, segmentation and its validation, audience-lifecycle analysis. What does
not transfer: any level or ratio (hours per viewer, repeat share, segment sizes).

## Sessionization (silver)
* Intervals are half-open slot ranges `[start, stop)`; a slot is 10 minutes. `watch_slots = stop - start`.
* Per viewer, sort by `(start, stop, stream_id)`. A new session starts when the next interval begins **more than 3 slots (30 min)**
  after the running maximum `stop` of everything before it in that viewer. The running maximum (not the previous row's stop) is
  used so a long stream watched in parallel with short ones never splits a session.
* Session watch time is the **union** of intervals (an interval contributes only the part not already covered), so
  parallel viewing is not double-counted. Unit tests cover the gap boundary, parallel streams and per-viewer isolation.
* The 30-minute gap is a judgement call; no sensitivity analysis on it was run.
* Intervals crossing midnight are attributed to the day of their start slot.

## Features and the time axis
* Slot 0 has no calendar date or time zone. "Clock" shares use `slot % 144`; the global profile (see results) has its trough
  around dataset hours 8-11 and peak around 18-22, but segments are named by dataset-clock window, not by local time.
* "Peak days" is the busiest adjacent pair of days in the 7-day cycle, found from the data, because weekdays cannot be known.
* Half-window features (for stability and prediction tests) keep intervals whose start slot is in the half and which add watch time.
  A session straddling the boundary is counted in the half where an interval of it starts.

## Choosing k, and what "stable" means here
* k range 4-10 is fixed beforehand (fewer than four segments is not actionable). Rule: highest silhouette among k with mean
  bootstrap ARI >= 0.80. The sweep runs on a 2% viewer sample; the final model uses all viewers.
* Bootstrap ARI: fit on independent bootstrap resamples, label a common hold-out sample, compare labellings with ARI. It mixes
  sample noise and KMeans initialisation noise; a separate seed-only ARI is reported.
* Time stability: model fitted on the first half vs second half; and whether a viewer's first-half and second-half behaviour fall in the
  same segment. Behaviour in 21 days is noisy for light viewers, so this is a lower bound on true stability.
* Predictive validity: first-half segment vs second-half outcomes, benchmarked against plain first-half volume cut in the same number of groups.
  A segmentation that does not beat "how much did they watch" would say little beyond volume.
* ALS embeddings were trained and evaluated on first-half data only (embeddings from the whole window would leak second-half behaviour into a
  test of second-half prediction).

## Netflix: cohort, target and the exposure problem
* Unit: a series *season*, a film or another programme with a release date on or after 2023-01-01 (so its release semester is inside the data).
* Target: `log(views in the release semester)`. The brief asks for "first full semester"; that target is missing for every title that drops below
  the report's 50k-view floor and does not exist yet for 2026H1 releases, so the release semester is used. The cost: exposure differs from 1 to 181 days.
  Exposure days is a pre-release feature (it follows from the release date) and curves elsewhere are normalised per exposure day.
* Views are rounded to 100k; values at the floor are censored. 2023H1 has no views column, so views are `hours / runtime` using the runtime of the same title
  in a later semester (Netflix's own definition); titles seen only in 2023H1 and without runtime are dropped from the cohort.
* Live events and recurring dated programmes (WWE, Raw, sports fixtures) are excluded: they have no life curve.

## Leakage traps and the tests that guard them
| Trap | Guard |
|---|---|
| Features built from the title's own later semesters | `test_features_do_not_move_when_everything_from_release_semester_is_corrupted`: multiplies every views/hours value from a cut-off semester onward by 1000 and asserts all features of titles released in that semester are unchanged (and that the target moved, so the test can fail). |
| Prior-season views that overlap the new season's release semester | `test_prior_season_requires_semester_ended_before_release`: a prior season is used only if the semester it is measured in ended strictly before the new release date. |
| Random train/test split | Rolling temporal folds; calibration semester sits between train and test. |
| Ratings/votes (post-release) | Not read from IMDb at all. |
| "Available globally" flag (can change after release) | Not used as a feature. |
| Life-curve shapes fitted on data from the test period | Shapes are fitted on cohorts released in 2023 using observations up to 2024H2, strictly before every hold-out label (2025H1 onward). |
| Embeddings built on the window being predicted | ALS and fold-in use first-half data only. |

Remaining, acknowledged leakage-like issues: the IMDb snapshot is current (genres, runtimes and episode counts could have been edited after release); the
"non-English title" flag uses the Netflix report title string, which is a post-release publication of a pre-release attribute; Netflix runtime comes from the report.

## Life curves
* Rate = views / exposure days; time axis = days since release at the middle of the exposed window. Shapes: exponential and power law `gamma * log(1 + t/tau)`.
* The shape is estimated from within-title variation (title fixed effects), so differences in title popularity do not masquerade as decay.
* **Survivorship**: titles that fall below the reporting floor disappear, so the observed late-age points are survivors; the median rate even rises at the oldest ages
  for that reason. Points at the floor are excluded from fits. A censored-regression fit would be the next step.
* Value index = observed + forecast cumulative views over four semesters, relative to the kind median. Validation uses titles released in 2024H1-H2 with the
  first one or two semesters observed; the realised value counts a dropped-out semester as zero (< 50k views).

## What is not claimed
* No business impact figure is estimated: no costs, revenues, subscribers or A/B tests are in the data.
* Segment names are labels for profiles on a different platform, not discovered audience types of any broadcaster.
* Spark timings are from one local machine; Databricks Free Edition serverless behaviour is not measured (no workspace yet).
