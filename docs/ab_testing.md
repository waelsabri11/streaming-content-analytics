# Measuring the effect of promoting a content: A/B test design (simulated)

**No experiment was run.** Everything numeric below comes from `reports/results/ab_simulation.json` (section 8 of
[`reports/results.md`](../reports/results.md)): the *variance structure* is real (Twitch viewers), the *treatment effect is injected
synthetically*. Twitch is a proxy for a subscription service, so the absolute numbers (hours per viewer) do not transfer; the method does.

## Design
* **Question**: does promoting a title (homepage push) increase engagement of the viewers who are shown it?
* **Unit of randomisation: the viewer** (account/profile), assigned by hash of the id. Not the session (sessions of one viewer are correlated and would leak
  treatment across arms), not the title (a title-level test measures substitution between titles, not the effect on a viewer). Interference between viewers is assumed negligible (no shared queues or social features);
  a catalogue-wide push that cannibalises other titles must be measured with a viewer-level total-watch-time metric, not with the promoted title's views.
* **Population**: viewers with activity in the pre-period (a new viewer has no covariate and may not exist at assignment time).
* **Primary metric**: watch hours in the experiment week (heavy tailed: the real data has median about 1.2 h, 99.9th percentile about 33 h, so consider winsorising at a pre-registered quantile). **Secondary**: return rate (any viewing in the week, 79.6% here), plus a guardrail on total watch time of non-promoted content.
* **Pre-registered before launch**: metric, horizon (full weeks, to avoid day-of-week mix), alpha 0.05 two-sided, power 0.8, stopping rule, winsorisation, exclusion of bots/test accounts.

## CUPED (variance reduction with pre-period behaviour)
Adjusted outcome `y* = y - theta (x - mean(x))` with `x` = the same viewer's pre-period hours (weeks 0-2) and `theta = cov(x, y) / var(x)` estimated on the pooled sample (randomisation makes the adjustment unbiased).
The variance falls by about `rho^2` where `rho` is the pre/post correlation: here rho = 0.715, so the variance of the estimator drops to 48.9% of the unadjusted one in the simulation (standard error 0.031 vs 0.044 hours at 20,000 viewers per arm).
In practice this halves the required sample size: for a 2% relative effect, 92,792 viewers per arm unadjusted vs 45,408 with CUPED.

## Power and minimum detectable effect (from the real Twitch variance)
Weekly hours per viewer: mean 2.88, standard deviation 4.43 (14.2M viewers active in weeks 0-2). MDE (80% power, alpha 0.05, relative to the mean):

| viewers per arm | naive | CUPED |
|---|---|---|
| 10,000 | 6.1% | 4.3% |
| 100,000 | 1.9% | 1.3% |
| 500,000 | 0.86% | 0.60% |

## Simulated A/A and A/B (clearly synthetic)
400 repetitions each, 20,000 viewers per arm resampled from the real viewers, treatment effect +3% of post-period hours injected in the A/B arm.
* **A/A** (no effect): false positive rate 4.75% naive, 5.75% CUPED (400 repetitions, so about +-1 point of noise): the test is calibrated either way.
* **A/B**: power 48.5% naive vs 79.5% CUPED at the same sample size; both estimators are centred on the injected effect; 95% interval coverage 96.0% and 93.8%.

## Sequential testing
Looking at a fixed-horizon p-value repeatedly inflates the false positive rate. Simulation with 10 equally spaced looks (up to 40,000 viewers per arm, CUPED outcome), 300 repetitions:
* naive peeking (stop at the first p < 0.05): **19.3% false positives under no effect**;
* mixture SPRT (always-valid p-value, normal prior with standard deviation 2% of the mean): 0.7% false positives under no effect (conservative), 82% power for the +3% effect, median stopping point 24,000 viewers per arm.
mSPRT is valid under continuous monitoring but conservative; a group-sequential design (alpha spending, e.g. O'Brien-Fleming boundaries) with pre-planned looks is the standard alternative and would recover some power.

## Not covered
Novelty effects (a push works for a day), long-run retention, heterogeneous effects (use the segments of this project to pre-register subgroup analyses), multiple metrics (control the family-wise error or pre-declare one primary), and sample-ratio-mismatch checks (monitor the arm sizes against 50/50 with a chi-square test before reading any result).
