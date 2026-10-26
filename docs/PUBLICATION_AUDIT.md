**Historical note:** this is the pre-bottom-up publication audit (2026-10-26). The final verified audit is `docs/FINAL_PUBLICATION_AUDIT.md`.

# Publication audit

Independent verification of every number intended for the paper, README,
dashboard and LinkedIn articles. Values were read from saved artifacts, not from
prior prose. Date: 2026-10-26.

---

## 1. Causal results

| Claim | Verified value | Source | Status |
|---|---:|---|---|
| Original Poisson DiD | −7.3% [−13.6%, −0.6%], p = 0.0342 | `scripts/verify_replication.py`; `results/estimates.json → replication.original_notebook` | PASS |
| Original OLS DiD | −0.3557 [−0.7110, −0.0005], p = 0.0497 | same | PASS |
| Original treated sample | 33 cameras, 11,880 panel rows | same | PASS |
| Revised conditional Poisson FE | −2.59% | `results/estimates.json` primary | PASS |
| Revised 95% CI | −11.47% to +7.19% | same | PASS |
| Revised p-value | 0.591 | same | PASS |
| Revised treated sample | **20 cameras** | same | PASS |
| Cohort 2×2 estimate | −0.095 crashes/site-month [−0.735, 0.520], p = 0.789 | `results/estimates.json` | PASS |
| Corrected-projection replication | −8.58% [−14.66%, −2.07%], p = 0.0105 | `results/estimates.json` replication | PASS |
| Event-study pre-trend test | Wald 5.20, df 7, p = 0.635 | `results/estimates.json → event_study_pre_trend` | PASS |
| Synthetic control | +4.18%, pre-R² 0.742, one-sided p = 0.495 | `results/estimates.json → synthetic_control` | PASS |
| Geographic placebos | one-sided p = 0.525 (DiD), 0.380 (cohort); median −3.07% | `results/estimates.json → placebo_geography` | PASS |
| Lattice phase range | −3.07% to −1.20%; mean −2.05% | `results/estimates.json → lattice_phase` | PASS |

**Reproduction check:** the original notebook's saved outputs reproduce exactly
on the current portal snapshot (`-0.3557` OLS; `-7.3%` Poisson; 33 cameras;
11,880 rows; pre-means 4.298/4.403). No arithmetic correction was made to the
original; the revised estimate comes from design changes documented in
`docs/AUDIT.md`.

## 2. Forecasting results

| Claim | Verified value | Source | Status |
|---|---:|---|---|
| V1 test MAE | 1.5683 | `results/ml/v1/metrics.json` | PASS |
| V1 test RMSE | 2.0931 | same | PASS |
| V1 test Poisson deviance | 1.2233 | same | PASS |
| V1 test R² | 0.6611 | same | PASS |
| V1 test bias | +0.163 | same | PASS |
| V2 retrospective MAE | 1.5642 | `results/ml_v2/metrics.json` | PASS |
| V2 retrospective RMSE | 2.0888 | same | PASS |
| V2 retrospective deviance | 1.2181 | same | PASS |
| V2 retrospective bias | +0.144 | same | PASS |
| V2 validation improvement | **0.243%** | `results/ml_v2/metrics.json → validation.relative_deviance_improvement` | PASS |
| Pre-declared 1% promotion rule | **not met**; V1 deployed | `results/ml_v2/deployment.json` | PASS |
| 90% interval coverage | **90.62%** | `results/ml_v2/metrics.json → conformal.interval_90` | PASS |
| Highest-risk decile coverage | **84.81%** | same, by risk decile | PASS |
| Camera-site interval coverage | 90.1% | same | PASS |
| Comparison-site interval coverage | 90.7% | same | PASS |
| Recursive 12-month V1 / V2 MAE | 1.6452 / 1.6231 | `results/ml_v2/recursive_metrics.json` | PASS |
| Recursive interval coverage | 90.2% | same | PASS |
| Geographic combined deviance (V2) | 1.2189 vs 1.2328 (V1 features) | `results/ml_v2/metrics.json → geographic` | PASS |

**Model selection:** V2 is a research candidate, not the deployed model. The
paper and README must say so explicitly. V2's retrospective and geographic gains
are small and do not satisfy the pre-declared 1% validation rule.

## 3. Data snapshot and observation window

| Check | Result | Status |
|---|---|---|
| Portal snapshot | 2026-10-24 | PASS |
| Raw crash rows | 1,106,882 | PASS |
| Cleaned crashes | 1,080,998 | PASS |
| First analysis month | 2015-09-01 | PASS |
| Last analysis month | 2026-08-01 | PASS |
| Last raw crash date | 2026-08-31 23:50 | PASS |
| Future-dated observations | none (max date precedes snapshot) | PASS |
| Reporting lag | last 2 source months excluded | PASS |
| Leading coverage guard | 2015-01…08 dropped (476 records) | PASS |
| Weather archive | 2015-01-01 … 2026-08-01 | PASS |

The 2015-09 start is the first month with reliable crash coverage; 2013–14 has
only 8 records and 2015-01…08 has 476. These are missing months, not observed
zeros.

## 4. Leakage and feature integrity

| Check | Result | Status |
|---|---|---|
| V1 lag/rolling perturbation test | passed | PASS |
| V2 lag 24/36 recomputation | passed | PASS |
| V2 neighbourhood prior-month recomputation | passed | PASS |
| Target columns absent from feature schema | passed | PASS |
| Weather lagged; no target-month weather | passed | PASS |
| No AADT/future exposure used | passed (excluded) | PASS |
| Schema validation at inference | passed | PASS |
| Test rows used in tuning | no | PASS |
| Test period previously inspected | yes — retrospective only | PASS (documented) |

## 5. Prose and table consistency

| Check | Result | Status |
|---|---|---|
| Causal numbers in README/REPORT match artifacts | passed | PASS |
| V1/V2 numbers in comparison docs match artifacts | passed | PASS |
| Interval coverage matches selected predictions | passed (90.62%) | PASS |
| No placeholder `[X]` or TBD values in publication docs | passed | PASS |
| Figures generated from saved data | passed | PASS |
| Root original notebook unchanged | zero diff; separate research copy in `notebooks/` | PASS |
| V1 artifacts preserved | `models/v1/`, `results/ml/v1/` | PASS |
| V2 candidate preserved | `models/v2_*`, `results/ml_v2/` | PASS |

## 6. Known discrepancies and exclusions

1. **V2 does not beat V1 by 1% on validation.** It improves 0.243%. Do not
   describe V2 as the deployed or "better" model. It is a candidate with better
   geographic/recursive diagnostics.
2. **The test period is not a new untouched confirmation.** It was inspected in
   V1; all V2 test numbers are retrospective.
3. **High-crash accuracy slightly regressed in V2** (top-decile MAE 2.922 vs
   2.897). Report this.
4. **Interval coverage is conditional, not uniform.** The highest-risk decile is
   covered at 84.8% at a nominal 90%; label intervals approximate.
5. **Causal estimate is imprecise.** The 95% CI (−11.5%, +7.2%) includes
   meaningful reductions and increases; do not call it a zero effect.
6. **The synthetic control disagrees** (+4.2%) and has a moderate pre-fit
   (R² 0.74); report as an inconsistency, not corroboration.
7. **V2 Optuna completed 11 trials**, not the 40-trial budget, because each
   trial is slower with 84 features. Report actual count.
8. **No traffic-volume exposure.** Current AADT is a 2026 snapshot; excluding it
   is deliberate.

## 7. Publication verdict

All headline numbers are verified. The scientific claims that can be made are:

- The original −7.3% is reproducible but not robust to local matching and
  overlap handling.
- The revised primary estimate is −2.6% with a wide interval; it is not
  statistically distinguishable from zero.
- V1 forecasting is reproducible and modestly accurate (MAE 1.568, deviance
  1.223), but high-risk months remain under-predicted and intervals under-cover
  the highest-risk decile.
- V2 adds real road/neighbourhood features and improves geographic and recursive
  diagnostics, but fails the pre-declared promotion rule.

No claim beyond these is supported by the saved evidence.
