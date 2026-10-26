# LightGBM methodology

This document describes the **predictive** layer. It is separate from the causal
analysis in `docs/METHODOLOGY.md` and must not be read as a causal design.

---

## 1. Purpose and scope

The model predicts **reported crashes in a 250 m catchment during a future
calendar month** at a fixed set of Chicago locations. It answers "how many crashes
should we expect here next month, given observed history and seasonality?" It does
**not** answer "what would happen if a camera were installed or removed?" That
question is reserved for the difference-in-differences, event-study,
synthetic-control and placebo analyses, which are unchanged.

A camera being operational is allowed as an observational feature because it is
known at forecast time. Its importance is **not** causal evidence; cameras are
sited where crashes are high, so the feature is confounded by placement.

## 2. Unit of observation and target

- Unit: **location-month**.
- Locations: camera catchments (overlapping 250 m catchments deduplicated) plus
  the causal design's 500 m comparison sites. The set is crash-dense and
  non-random; it is not a representative sample of all Chicago road segments.
- Target: `y_all`, the count of distinct reported crashes within 250 m of the
  location coordinate during the calendar month.
- Target window: **2017-01 … 2026-08** (116 months). The first eight target
  months precede citywide reporting (September 2017); a coverage-consistent
  sensitivity starts in 2018-01 and fails the 1% promotion rule. Earlier months
  (2015-09 … 2016-12) are feature history; the crash-reporting
  regime before 2017 is unstable.
- Counts: every location-month has an observed count, including genuine zeros.
  No missing target values are imputed.

## 3. Features

All features are available at the forecast date or are deterministic calendar
attributes.

| Group | Features | Construction |
|---|---|---|
| Calendar | `year`, `month_of_year`, `month_sin`, `month_cos`, `quarter`, `days_in_month`, `month_index` | Deterministic from the forecast month |
| Holidays | `n_holidays` | US federal holidays in the month |
| Crash history | `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12` | `y_all` shifted within location |
| Rolling history | `roll_mean_3/6/12`, `roll_std_3/6/12`, `expanding_mean`, `expanding_std`, `zero_share_12` | Shifted by one month before rolling; only prior months |
| Severity history | `injury_share_12`, `severe_share_12` | 12-month shifted sums, divided by shifted total |
| Momentum | `trend_12` | `lag_1 / (roll_mean_12 + 0.5)` |
| Weather | `temp_lag1`, `precip_lag1`, `snow_lag1`, `temp_mean_3`, `precip_mean_3`, `snow_mean_3` | Open-Meteo daily archive aggregated monthly, then shifted; **never the target month's weather** |
| Geography | `x`, `y`, `lat`, `lon`, `dist_center_m`, `dist_nearest_camera_m`, `n_cameras_1km`, `cluster` | Static; UTM 16N distances; k-means cluster (k = 10, seed 20261024) |
| Treatment context | `site_is_camera`, `camera_operational`, `camera_age_months`, `n_operational_cameras_1km` | Known at forecast time; observational only |

The weather archive is cached verbatim in `data/raw/weather_daily.csv.gz` with a
provenance record. If it is unavailable, the weather columns are dropped from the
schema, the dashboard and report record the blocker, and no values are invented.

## 4. Leakage prevention

1. **Lags** use `groupby(location).shift(k)` with `k ≥ 1`.
2. **Rolling statistics** are computed on `shift(1)` and only look backwards.
3. **Severity shares** use shifted 12-month sums.
4. **Weather** is lagged; the forecast month's own weather is never a feature.
5. **No target-derived feature** is included in the feature list.
6. **Temporal splits**: the test period is locked before tuning; folds are
   strictly earlier; months are never shuffled.
7. **No target encoding** and no global statistics fitted on validation/test.
8. **Mechanical checks** (`validate_no_leakage`) recompute lags/rolling means and
   perturb future outcomes to confirm earlier features do not change. The build
   fails if a check fails.
9. **Schema checks** at prediction time compare the saved model's feature order
   and names with the input frame; mismatch raises.

## 5. Model

- **One model only:** LightGBM, `objective="poisson"`, `metric="poisson"`,
  gradient-boosted decision trees.
- No XGBoost, CatBoost, random forest, neural network or ensemble is trained.
- `cluster` is treated as categorical; all other features are numeric.
- Deterministic settings: fixed seeds, `deterministic=True`,
  `force_col_wise=True`, fixed thread count.

## 6. Tuning

- Framework: Optuna, TPE sampler (`multivariate=True`), `MedianPruner`
  (`n_startup_trials=8`).
- Search space: `learning_rate` (0.01–0.15, log), `num_leaves` (15–255),
  `max_depth` (3–12), `min_child_samples` (10–100), `feature_fraction` (0.6–1),
  `bagging_fraction` (0.6–1), `bagging_freq` (1–7), `lambda_l1` and `lambda_l2`
  (1e-3–10, log), and the number of boosting rounds via early stopping
  (100 rounds patience, max 3,000).
- Validation: **rolling-origin** folds ending before the test period; each fold
  trains on all earlier months and validates on the next six months.
- Objective: mean Poisson deviance across folds. Hyperparameters are **never**
  selected on causal significance or on the test period.
- Budget: 40 trials or 900 seconds, whichever comes first; the actual number
  completed is recorded in `results/ml/metrics.json`.

## 7. Final training and test protocol

1. Tune on rolling-origin folds strictly before the test period.
2. Refit on **all** location-months before the test period (through 2025-08).
3. Use the mean best iteration from the folds, scaled by 1.1, as the number of
   boosting rounds; no test-period early stopping.
4. Evaluate on the locked test period **2025-09 … 2026-08**. No test month
   influences training or tuning; a five-trial smoke run was inspected during
   implementation and is disclosed in `docs/PREDICTION_RESULTS.md`.
5. Save the model only after successful training and evaluation.

## 8. Metrics and baselines

Reported on the untouched test rows only:

- MAE, RMSE, mean Poisson deviance, bias (mean predicted − mean actual) and
  relative bias, R², and MAPE on non-zero actuals.
- Calibration by predicted-risk decile (mean predicted vs mean actual).
- Error on zero-crash months and on the top decile of actual crashes.
- Monthly error and bias.
- Baselines evaluated on **exactly the same rows**: seasonal naive (same month
  last year), rolling 3-month mean, and historical expanding mean. These are
  reference forecasts, not competing ML models, and are reported only in
  documentation, not used for model selection.

## 9. Interpretability

SHAP values are computed with `shap.TreeExplainer` for held-out test rows. They
decompose the model's predicted log rate into feature contributions. They are
**predictive associations**, not causal effects. A large SHAP value for
`camera_operational` would reflect the model's use of camera siting and history,
not proof that the camera caused the crashes.

## 10. Performance

- Parquet caching for the processed crash table, location-month dataset and
  predictions.
- Vectorised feature engineering; rolling windows use `groupby.transform`.
- LightGBM `Dataset` objects reused within a fold.
- Optuna pruning and a configurable trial/time budget.
- The dashboard loads precomputed predictions and SHAP values; it **never**
  trains or tunes on reload.
- Metrics reported: training time, inference time, and artifact sizes in
  `results/ml/metrics.json`.

## 11. Limitations

- The location set is crash-dense and non-random; accuracy elsewhere is unknown.
- No verified road-network or intersection-density source was used. Static geography is
  limited to coordinates, distance to city centre and cameras, a k-means cluster and
  crash history; this is a deliberate choice to avoid unverified external joins.
- Monthly crash counts are overdispersed; Poisson deviance is the appropriate
  loss, but intervals are not calibrated probabilistic forecasts.
- The model is a one-step-ahead monthly forecaster. It is not validated at daily
  or sub-monthly resolution.
- `month_index` and `year` are in-range for the test period; tree models cannot
  extrapolate beyond the training range, so forecasts far beyond 2026 are not
  supported.
- Weather is lagged, not forecast; a true operational forecast would use a
  weather forecast and be evaluated separately.
- Reporting changes (notably around 2017) and camera operation changes are
  absorbed by history features, not explicitly modelled.
- Prediction errors are larger for high-crash months (shrinkage toward the mean).
- The model does not identify causal effects and must not be used to evaluate a
  camera's impact.

## 12. Reproduction

```bash
python scripts/download_data.py
python -m chicago_cameras.pipeline --config config/analysis.yaml
python scripts/train_lightgbm.py --trials 40 --timeout 900
python scripts/evaluate_lightgbm.py
streamlit run dashboard/app.py
```

Artefacts: `models/lightgbm_crash_model.txt`, `models/model_metadata.json`,
`results/ml/metrics.json`, `results/ml/predictions.parquet`,
`results/ml/shap_values.parquet`, `results/ml/figures/`.
