# Model card

Numbers are in [`reports/results.md`](../reports/results.md); this card describes what each model is and is not.

## 1. Viewer segmentation (Twitch)
* **Task**: group 15.5M viewers by behaviour so a business team can reason about audiences.
* **Model**: Spark MLlib `VectorAssembler -> StandardScaler -> KMeans`, 15 viewer-level features (volume, sessions, active days,
  clock-time shares, channel diversity and concentration, repeat consumption, loyalty). k chosen by a rule fixed beforehand:
  among k in 4..10 with mean bootstrap ARI >= 0.80, the highest silhouette.
* **Intended use**: descriptive profiling and a starting point for targeting hypotheses on streaming-like data.
* **Not for**: any claim about subscribers of a video-on-demand service; individual-level decisions.
* **Limits**: silhouette is low for every k (behaviour is a continuum); stability is moderate (see results: ARI across time halves vs the share of
  viewers who stay in the same segment); the segments predict later volume worse than first-half volume cut in the same number of groups; ALS embeddings did not improve
  stability or prediction; time-of-day features use the dataset clock, whose time zone is unknown. KMeans uses best-of-3 restarts. Segment names are built from the profile by fixed
  thresholds (volume tier, loyalty, session length, clock window), not hand-picked.

## 2. Performance score (Netflix)
* **Task**: predict `log(views in release semester)` from information known before release.
* **Model**: LightGBM regressor (400 trees, 15 leaves, seed 42) + split-conformal 80% interval (absolute residuals on the previous
  semester). Baselines: median by kind x primary genre; prior season's views.
* **Features** (all pre-release): kind, season number, limited-series flag, runtime, release month/weekday/exposure days in the release
  semester, IMDb genres/runtime/episode count (snapshot), non-English-title flag, franchise history from semesters that ended before release.
* **Excluded on purpose**: any views/hours observed from the release semester onward, IMDb ratings/votes, the title's later presence in reports,
  the "available globally" flag (can change after release).
* **Evaluation**: rolling temporal folds (test 2025H1, 2025H2, 2026H1); training uses cohorts released at least two semesters before the test semester.
* **Limits**: ~350-390 test titles per fold; the target is the release semester, whose exposure ranges from 1 to 181 days (exposure is a feature, but
  the target is not per-day); titles below Netflix's 50k-view floor are missing; IMDb matching is 85% (see results).

## 3. Life-curve model and value index (Netflix)
* **Model**: per content kind, `log(views per exposure day)` against days since release with title-level intercepts; exponential and
  power-law (`gamma * log(1 + t / tau)`, tau on a grid) shapes compared; a non-parametric median-ratio method and persistence as baselines.
* **Value index**: observed + forecast cumulative views over 4 semesters after release, divided by the median of the same kind.
* **Limits**: survivorship (titles drop out of the report when they fall below the floor, so late-age points are the survivors); semester resolution
  cannot see the first-week burst; the index is a forecast for titles with fewer than 4 observed semesters.

## 4. Twitch channel life curves
* New vs returning viewers per channel-day, week-cohort retention of viewer-channel pairs, exponential vs power decay for channels born inside the window.
* **Limits**: the 43-day window is short, channels are not episodic content (broadcasters stream repeatedly), and age is left-censored for channels live at day 0.
