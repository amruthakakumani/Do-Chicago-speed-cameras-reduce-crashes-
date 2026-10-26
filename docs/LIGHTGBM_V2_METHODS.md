# LightGBM V2 methods

V2 keeps **one** LightGBM Poisson model and the unchanged causal analysis. It
adds real road-network features, neighbourhood history, citywide regime
features, prediction intervals, geographic holdout validation and a recursive
12-month diagnostic.

---

## 1. Objective and scope

- Target: reported crashes in a 250 m catchment during a calendar month.
- Unit: location-month; 1,355 locations (129 overlap-deduplicated camera
  catchments + 1,226 comparison sites).
- Train: 2017-01 … 2025-08 (140,920 rows). Test: 2025-09 … 2026-08
  (16,260 rows).
- Prediction is **not** causal inference. Camera operational status is an
  observational feature, not evidence of an effect.

## 2. Data and provenance

| Source | Use | Coverage / vintage | Licence |
|---|---|---|---|
| Chicago Data Portal `85ca-t3if` | Crash outcome | 2015-09 … 2026-08 snapshot | Public |
| Chicago Data Portal `4i42-qv3h` | Camera locations and go-live | 209 cameras | Public |
| Open-Meteo archive | Lagged weather and climate normals | 2015-01 … 2026-08 | Open data |
| US Census TIGER/Line 2024 EDGES (`17031`, `17043`) | Road density, class, distance, intersections | Cook + DuPage, 2024 vintage | Public domain |
| `holidays` US calendar | Holiday counts | Forecast calendar | Open source |

Raw TIGER archives and SHA-256 checksums are in `data/raw/tiger/provenance.json`.
TIGER **EDGES** (not the cartographic ROADS layer) is used because its topology
nodes (`TNIDF`/`TNIDT`) make intersections unambiguous.

**Traffic volume is deliberately excluded.** The current Chicago AADT dataset
`gc7y-n4xa` is a 2026 snapshot; using it for 2017–2025 training would leak future
exposure. The historical 2006 study is too old and sparse. No values are
manufactured.

## 3. Features

V2 starts from the 44 V1 features and adds:

### Temporal / regime (15)
`lag_24`, `lag_36`, `same_month_mean_3y`, `seasonal_ratio`, `cv_12`,
`trend_6`, `shock_lag1`, `shock_lag3`, `months_since_last_crash`,
`months_since_last_injury`, `ever_crashed`, and citywide lag/3-month/12-month
counts, citywide trend, citywide injury share, city/citywide shock and
location-to-city 12-month ratio.

All are computed from months **strictly before** the forecast month.
`lag_24`/`lag_36` are missing for the first two/three years; missingness is
documented rather than imputed.

### Neighbourhood (6)
Gaussian-kernel (σ = 750 m, 3 km cutoff, row-normalised) weighted histories of
neighbouring locations: `neighbor_crash_lag1`, `neighbor_crash_mean_12`,
`neighbor_crash_std_12`, `neighbor_zero_share_12`,
`neighbor_injury_share_12`, `neighbor_shock_lag1`. Only prior months enter.

### Road network (13)
`road_density_500m_km_per_km2`, major/local road density, 500 m road length,
major-road share, road segment count, 1 km density, major 1 km density,
intersection density at 500 m/1 km, intersection counts, distance to nearest
major road, distance to nearest major intersection.

These are static built-environment attributes. They use 2024 TIGER geometry for
all training years, which assumes the network is broadly time-invariant; minor
network changes are a documented limitation.

### Weather normals (3)
`temp_normal`, `precip_normal`, `snow_normal` are expanding same-calendar-month
normals from prior years. They provide weather information for recursive
forecasts where future weather is unknown. One-step forecasts also retain the
V1 lagged weather features.

## 4. Leakage controls

V1 checks remain: shifted lags, shifted rolling windows, no target-derived
feature, chronological splits, schema validation, and the future-perturbation
test. V2 adds mechanical recomputation checks for lags 24/36 and for the
neighbourhood lag (prior-month weighted history must match stored values).
`validate_v2_leakage` fails the build on any mismatch.

## 5. Validation design

1. **Temporal rolling origin** (pre-test only): three expanding six-month
   validation blocks, 2024-03 … 2025-08.
2. **Geographic holdout**: 2 km blocks; 34 of 169 blocks are held out
   (294 locations: 24 cameras, 270 comparison sites). Training locations within
   1 km of any held-out location are removed, so nearby dependent catchments do
   not straddle the split. Minimum train-to-holdout distance ≈ 1,005 m.
3. **Combined temporal + geographic holdout**: train non-held-out locations
   through 2025-02; evaluate held-out locations in 2025-03 … 2025-08.
4. **Geographic-only**: train non-held-out locations through 2025-08; evaluate
   held-out locations in 2025-03 … 2025-08.

The old test period (2025-09 … 2026-08) has already been examined. Any V2 number
on it is a **retrospective diagnostic**, not a new untouched confirmation.

## 6. Tuning and selection

- Optuna TPE with median pruning; budget 40 trials / 900 s. **11 trials
  completed** in 922 s, selected on mean Poisson deviance across the three
  pre-test temporal folds.
- Pre-declared rule in `docs/V2_BASELINE_AUDIT.md`: V2 replaces V1 only if
  validation deviance improves by at least **1% relative** and validation MAE
  does not regress by more than **2%**.
- The rule is evaluated on pre-test validation only. The retrospective test is
  reported but not used to choose the deployed model.

## 7. Prediction intervals

Split conformal on **out-of-fold** rolling-validation predictions; no test row
and no second deployed model is used.

- Scores: absolute residual and normalised residual
  `|y − μ| / √(μ + 1)`.
- The normalised score is chosen a priori for count-scale heteroskedasticity;
  time-ordered cross-conformal coverage on the calibration period is reported
  as a diagnostic. The absolute score is retained as a comparator.
- Finite-sample corrected quantile; intervals `μ ± q·scale`, clipped at zero.
- 90% and 50% intervals; empirical coverage, width, and coverage by site type,
  cluster and predicted-risk decile are reported.
- Exchangeability is not assumed: spatial and temporal dependence can break the
  nominal marginal guarantee. Intervals are labelled **approximate**.

## 8. Recursive 12-month diagnostic

The headline one-step metrics use observed histories for lags. The recursive
diagnostic starts at 2025-08 and forecasts 2025-09 … 2026-08 without observing
intervening crashes: predictions feed subsequent lags, rolling windows,
neighbourhood histories and citywide totals. Future weather is replaced by
expanding climate normals. Recursive errors are reported separately and are not
comparable to one-step errors.

## 9. Reproduction

```bash
python scripts/train_lightgbm_v2.py --trials 40 --timeout 900
python scripts/evaluate_recursive_v2.py
streamlit run dashboard/app.py
```

Artifacts: `models/v2_lightgbm_crash_model.txt`,
`models/v2_model_metadata.json`, `results/ml_v2/`.
