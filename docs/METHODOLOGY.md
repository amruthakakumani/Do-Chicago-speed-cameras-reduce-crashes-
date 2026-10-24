# Methodology

This document states the estimand, the design, the identification assumptions, and the estimators
that produce `results/`. It is written so that a reviewer can decide whether the assumptions are
credible before looking at the numbers.

---

## 1. Research question

What is the causal effect of a Chicago speed camera on the number of reported crashes in the
250 m around it, relative to what would have happened at a comparable location without a camera?

## 2. Estimand

**ATT** = the average effect of camera enforcement on **reported crashes per catchment-month**,
for cameras that go live between 2018 and 2024 and have a complete 24-month pre-period and
36-month post-period inside the observed window.

The unit of observation is a **site-month**:

- *treated site*: the 250 m catchment around a camera's reported coordinate;
- *comparison site*: a 500 m lattice point at least 600 m from every camera, with at least 100
  crashes over the observed period and at least 40% of months with a crash.

Crashes are counted once per site-month. A crash can belong to only one treated catchment in the
primary design (overlap removal), and treated/control catchments cannot overlap by construction
(600 m buffer > 2 × 250 m).

The estimand is **not** crashes per vehicle-mile travelled, and **not** crashes per road segment.
No exposure denominator is available, so the result must not be interpreted as a change in risk per
unit of travel.

## 3. Treatment and timing

- Treatment begins in the month of `go_live_date` (`treatment_month_offset = 0`), matching the
  original study.
- A sensitivity analysis starts treatment in the following month (`offset = 1`) because go-live
  dates are monthly and mid-month.
- Chicago's warning period is not recorded in the portal table. The baseline assumes enforcement
  begins at `go_live_date`; the 30-day warning period would attenuate the immediate effect.
- Treatment is absorbing: cameras do not switch off.

## 4. Comparison group

Each camera is matched to five comparison sites:

1. within `max_match_distance_m` of the camera (primary: 2,000 m; sensitivity: 1,000 m, 5,000 m,
   and citywide);
2. closest on a score combining the absolute difference in pre-period mean crashes and the absolute
   difference in pre-period linear slope, each standardised by the dispersion of the candidate pool
   **evaluated in that camera's own pre-period window**;
3. at least 600 m from every camera (so no control catchment can contain a camera catchment).

Comparison sites may be reused across cameras. Reuse creates dependence across stacks, which is why
inference is clustered at the camera level and cross-checked with a wild cluster bootstrap.

## 5. Identification assumptions

The DiD estimate is causal under:

1. **Parallel trends.** Absent the camera, camera catchments and their matched comparison sites
   would have followed the same trend. Matching on pre-period level and slope makes this more
   plausible but cannot prove it. The event study reports a joint pre-trend test, which can only
   fail to reject, never confirm.
2. **No anticipation.** Driver behaviour does not change before the recorded go-live date. A
   30-day warning period violates this slightly; the `offset = 1` sensitivity bounds the damage.
3. **No interference (SUTVA).** A camera's effect does not spill beyond its 250 m catchment and into
   comparison sites. The 600 m buffer rules out *catchment overlap*, not behavioural spillover or
   crash displacement; buffer sensitivity is reported.
4. **No differential shocks.** Conditional on the fixed effects, nothing else changes camera
   catchments relative to comparison sites at the same time. The stack × event-month fixed effects
   absorb any shock common to a camera and its controls, including citywide and seasonal shocks.
5. **Treatment timing is exogenous** conditional on the match. Cameras are sited where crashes were
   high, so this is the weakest assumption: regression to the mean remains a threat even after
   matching on pre-trends. Placebo go-live dates and geographic placebos test for it.
6. **No contamination from earlier cameras.** A camera whose catchment overlaps a camera that is
   already live is dropped. Keeping them is reported as a sensitivity.

## 6. Estimators

### 6.1 Original baseline (replication)

- `y ~ treated + post + treated×post`, OLS and Poisson, standard errors clustered by camera stack.
- This is the original notebook's specification, retained to show what the correction changes.
- A **Negative Binomial (NB2)** version with the same regressors is reported when the Pearson
dispersion diagnostic indicates overdispersion. Poisson standard errors are only valid under
equidispersion; NB2 estimates the variance function rather than assuming it.

### 6.2 Two-way fixed-effects OLS (preferred linear specification)

- Unit fixed effects (site) and stack × event-month fixed effects.
- Absorbed by alternating projections; standard errors clustered by camera.
- The stack × event-month FE absorb every shock common to a camera and its five matched sites, so
  identification comes from within-stack, within-site deviations.

### 6.3 Conditional Poisson FE (preferred count specification)

- Conditioning on each stack-month total turns the Poisson FE model into a binomial model for the
  treated count with the number of controls as exposure. This is numerically identical to Poisson
  with a full set of stack × event-month intercepts, without estimating thousands of dummies.
- Coefficient is a log rate ratio; `exp(β) − 1` is the percent change.

### 6.4 Cohort 2×2 (staggered-adoption-robust)

For each camera *i* and event month *e* ≥ 0:

```
ATT_i(e) = [Y_i,g+e − Y_i,g−1] − mean_j [Y_j,g+e − Y_j,g−1]
```

over camera *i*'s matched controls *j*. The camera-level ATT is the mean over *e* = 0…35; the
overall ATT is the mean across cameras. Inference is a percentile bootstrap that resamples cameras,
which respects the (small) number of independent treated units and avoids TWFE weighting.

### 6.5 Event study

Quarterly treated × event-time indicators, reference quarter −1, estimated with unit and stack ×
event-month FE. The joint pre-trend Wald statistic uses the cluster-robust covariance of the
pre-period coefficients. The test's low power is stated explicitly.

### 6.6 Synthetic control

- Treated series: mean crash count across the largest go-live cohort.
- Donors: 100 comparison sites with the smallest pre-period discrepancy (pre-period information
  only).
- Fit: simplex-constrained least squares on de-meaned series (the original specification), warm-
  started from a clipped least-squares solution.
- Inference: 200 placebo-in-space draws of the same cohort size; pseudo p-values for the post/pre
  RMSPE ratio and for a negative post gap.
- Diagnostics: pre-period R², pre-RMSPE, in-time placebos (−24, −12 months), leave-one-out on the
  five most heavily weighted donors.
- **Fit gate:** if pre-period R² < 0.50 the fit is reported as inadequate and the SC is not used as
  corroboration.

### 6.7 Bayesian negative binomial (not run)

The original optional PyMC model is preserved in the original notebook. It was **not** re-run in the
upgrade because PyMC is not part of the lightweight environment; its estimand (`n` = number of units
as exposure) is a cell-size convenience, not traffic exposure. Treat it as optional.

## 7. Inference

- Cluster-robust (camera) standard errors for the regression estimators.
- **Wild cluster bootstrap-t** (Rademacher weights, restricted null, 999 replications) for the
  preferred linear specification, because there are only 20 cameras and comparison sites are shared.
- Camera-level percentile bootstrap for the cohort 2×2 estimator.
- Placebo-in-space and placebo-in-time p-values for the synthetic control and DiD designs.
- Exploratory subgroups are Benjamini–Hochberg adjusted and labelled exploratory.

## 8. Robustness

Every variation is a declared change to `DesignConfig`:

| Dimension | Values |
|---|---|
| Catchment radius | 150, 250, 400 m |
| Exclusion buffer | 500, 600, 800 m |
| Matched controls | 3, 5, 10 |
| Maximum match distance | 1,000, 2,000, 5,000 m, citywide |
| Event window | −12/+36, −24/+24, −24/+48 |
| Lattice phase | 0, 125, 250, 375 m |
| Overlap handling | drop later camera, keep all |
| COVID-19 | exclude 2020‑03…2021‑12 |
| Outcomes | all, injury, severe/fatal |
| Placebo dates | −24, −12 months |
| Lagged treatment (persistence) | +12 months |
| Placebo geography | 200 random pseudo-camera locations |
| Reporting era | 2015–2018 cohorts vs 2019+ cohorts |

The headline result is **not** chosen from this table. It is the pre-specified primary design; the
table shows how much it moves.

## 9. Deviations from the original study

| Original | Upgrade | Why |
|---|---|---|
| Flat lon/lat projection | UTM 16N (EPSG:26916) | Correct metric distances |
| Citywide matching (despite README) | 2,000 m local matching as primary | Absorb local shocks; make the documented design feasible |
| Overlapping cameras kept | Drop cameras overlapping an already-live camera | Independent treated units; no double counting |
| Start 2015‑01 | Start 2015‑09 via coverage guard | Sparse early months are missing, not zeros |
| Pooled OLS/Poisson only | Adds FE OLS, conditional Poisson, cohort 2×2 | Staggered adoption and common shocks |
| No severity outcomes | Injury and severe/fatal outcomes | Policy-relevant margin |
| Descriptive violations curve | Within-camera slope + caveats | Enforcement ≠ behaviour |
| Placeholder README results | Validated numbers with CIs | Scientific integrity |

## 10. Interpretation rules

1. A coefficient is a causal effect only under §5. The report states which assumption is most at
   risk for each result.
2. A non-significant estimate is **not** evidence of no effect; with 20 cameras the confidence
   interval is wide. Report the interval, not just the p-value.
3. The synthetic control is corroborating evidence only when its pre-fit is adequate and its sign
   agrees with the DiD. In this study it does neither (R² = 0.74 is borderline and the sign is
   opposite), so it is reported as a disagreement, not as confirmation.
4. Subgroup estimates are exploratory.
5. The result does not generalise to the 2013–14 camera rollout, which cannot be evaluated with
   crash data that begin in 2015.
