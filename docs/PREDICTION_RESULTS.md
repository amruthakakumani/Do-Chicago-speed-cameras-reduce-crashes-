# LightGBM prediction results

**Status:** retrospective diagnostic on a previously inspected test period.
Generated from `results/ml/metrics.json`; do not edit by hand.
**Target window:** 2017-01-01 … 2026-08-01, model trained through 2025-08-01.
The first eight target months precede citywide reporting (September 2017); a
coverage-consistent sensitivity starts in 2018-01 and does not meet the 1%
promotion rule.

## 1. What was evaluated

| Item | Value |
|---|---|
| Target | 2017-01-01 … 2026-08-01 |
| Training rows | 140,920 |
| Test rows | 16,260 |
| Test period | 2025-09-01 … 2026-08-01 (12 months) |
| Locations | 1,355 (129 camera, 1,226 comparison) |
| Features | 44 |
| Model | LightGBM, Poisson objective, 924 rounds |
| Tuning | 40 Optuna trials, best validation deviance 1.2230 |

> **Protocol note.** During implementation, one five-trial smoke run was evaluated
> end-to-end on the same test period to verify the pipeline. No feature,
> hyperparameter or model-selection decision was changed in response to those
> numbers; the final 40-trial study selected hyperparameters only on pre-test
> rolling folds. The reported metrics are from the final model trained before the
> test period. For a fully never-inspected holdout, rerun on a later data snapshot
> with a new test window.

## 2. Test accuracy

| Metric | Value |
|---|---|
| Test rows | 16,260 |
| MAE | 1.568 |
| RMSE | 2.093 |
| Mean Poisson deviance | 1.223 |
| Bias (pred − actual) | 0.163 (+4.3%) |
| R² | 0.661 |
| MAPE on non-zero actuals | 52.4% |
| Mean actual / predicted | 3.769 / 3.932 |


These are **out-of-sample** results. A high training score is not evidence of
forecast skill and is not reported as such.

## 3. Comparison with simple baselines

| Baseline | Rows | MAE | RMSE | Poisson deviance | Bias |
|---|---|---|---|---|---|
| Seasonal naive (same month last year) | 16,260 | 2.116 | 2.896 | 8.849 | 0.102 |
| Rolling 3-month mean | 16,260 | 1.749 | 2.362 | 2.020 | 0.033 |
| Historical expanding mean | 16,260 | 1.569 | 2.167 | 1.277 | -0.354 |


The baselines are evaluated on exactly the same test rows. They are reference
forecasts, not competing ML models. The historical expanding mean is a strong
baseline for MAE because monthly crash counts are persistent; the LightGBM model
is judged on Poisson deviance, where it improves on the baselines.

## 4. Calibration

| Predicted-risk decile | Rows | Mean predicted | Mean actual | Actual / predicted |
|---|---|---|---|---|
| 1 | 1,626 | 1.060 | 1.001 | 0.94 |
| 2 | 1,626 | 1.411 | 1.335 | 0.95 |
| 3 | 1,626 | 1.797 | 1.689 | 0.94 |
| 4 | 1,626 | 2.238 | 2.161 | 0.97 |
| 5 | 1,626 | 2.765 | 2.675 | 0.97 |
| 6 | 1,626 | 3.372 | 3.274 | 0.97 |
| 7 | 1,626 | 4.121 | 3.957 | 0.96 |
| 8 | 1,626 | 5.041 | 4.866 | 0.97 |
| 9 | 1,626 | 6.646 | 6.376 | 0.96 |
| 10 | 1,626 | 10.871 | 10.360 | 0.95 |


Across the ten predicted-risk deciles the ratio of actual to predicted crashes ranges from 0.94 to 0.97. Values below 1 mean the model over-predicts; values above 1 mean it under-predicts. The overall bias is 0.163 crashes per location-month (+4.3%).

## 5. Error structure

| Group | Rows | Mean predicted | Mean actual | MAE |
|---|---|---|---|---|
| Zero-crash months | 1,865 | 1.718 | 0 | 1.718 |
| High-crash months (top decile, threshold 8.0) | 2,073 | 9.020 | 10.989 | 2.897 |

The model over-predicts zero-crash months and under-predicts the highest-crash
months: the usual shrinkage of a squared/deviance-loss model. Do not read this as
a causal effect.

## 5b. Performance by location type and region

| Site type | Rows | MAE | RMSE | Poisson deviance | Bias |
|---|---|---|---|---|---|
| camera | 1,548 | 1.769 | 2.310 | 1.266 | 0.401 |
| control | 14,712 | 1.547 | 2.069 | 1.219 | 0.138 |


| Cluster | Rows | MAE | RMSE | Poisson deviance | Bias |
|---|---|---|---|---|---|
| 0 | 1,704 | 1.466 | 1.926 | 1.203 | 0.185 |
| 1 | 1,716 | 2.039 | 2.658 | 1.288 | 0.297 |
| 2 | 1,764 | 1.537 | 2.058 | 1.216 | 0.055 |
| 3 | 1,668 | 1.191 | 1.578 | 1.166 | 0.021 |
| 4 | 1,764 | 1.731 | 2.287 | 1.213 | 0.201 |
| 5 | 1,548 | 1.412 | 1.874 | 1.214 | 0.122 |
| 6 | 1,368 | 1.369 | 1.844 | 1.201 | 0.022 |
| 7 | 1,692 | 1.676 | 2.270 | 1.251 | 0.127 |
| 8 | 1,764 | 1.692 | 2.177 | 1.269 | 0.318 |
| 9 | 1,272 | 1.474 | 1.936 | 1.198 | 0.270 |


Accuracy is reported, not assumed equal, across camera and comparison sites and
across geographic clusters.

## 6. What drives predictions

| Feature | Gain share (%) | Splits |
|---|---|---|
| `roll_mean_12` | 58.49 | 1,955 |
| `roll_mean_6` | 15.50 | 997 |
| `expanding_mean` | 11.81 | 4,093 |
| `expanding_std` | 3.40 | 1,514 |
| `month_index` | 2.70 | 5,818 |
| `roll_mean_3` | 2.17 | 1,179 |
| `month_cos` | 0.62 | 1,401 |
| `cluster` | 0.52 | 2,811 |
| `temp_lag1` | 0.36 | 1,536 |
| `year` | 0.31 | 573 |
| `month_of_year` | 0.29 | 919 |
| `lag_1` | 0.27 | 518 |
| `lat` | 0.26 | 1,645 |
| `trend_12` | 0.26 | 1,847 |
| `dist_center_m` | 0.23 | 1,844 |


### Mean absolute SHAP value (held-out rows)

| Feature | Mean \|SHAP\| |
|---|---|
| `roll_mean_12` | 0.274 |
| `expanding_mean` | 0.222 |
| `month_index` | 0.060 |
| `roll_mean_6` | 0.054 |
| `month_cos` | 0.034 |
| `expanding_std` | 0.023 |
| `roll_mean_3` | 0.022 |
| `temp_lag1` | 0.016 |
| `year` | 0.009 |
| `month_of_year` | 0.008 |
| `cluster` | 0.007 |
| `precip_lag1` | 0.006 |
| `precip_mean_3` | 0.005 |
| `days_in_month` | 0.004 |
| `lat` | 0.004 |


### Camera features

| Camera feature | Gain share (%) |
|---|---|
| `camera_age_months` | 0.0224 |
| `n_operational_cameras_1km` | 0.0156 |
| `n_cameras_1km` | 0.0144 |
| `camera_operational` | 0.0010 |
| `site_is_camera` | 0.0009 |


Camera operational status is a legitimate forecast feature because it is known at
the forecast date, but its importance is **not** causal evidence. Cameras are
sited where crash risk is high, so camera status is confounded by placement. In
this fit the camera features together account for about 0.05% of total split
gain: the model is driven almost entirely by crash history. That is a statement
about predictive redundancy, not about whether cameras work.

## 7. Tuning and protocol

- Optuna TPE with median pruning; 40 trials completed
  (914s), best validation deviance
  1.2230.
- Rolling-origin validation strictly before 2025-09-01; the test months
  never influenced tuning.
- Final refit on all pre-test months (2017-01-01 …
  2025-08-01), 924 boosting rounds,
  13.2s.
- Deterministic seeds and `deterministic=True`; repeated inference is identical.

## 8. Figures

`results/ml/figures/` contains feature importance, SHAP summary and waterfall,
actual-versus-predicted, monthly error, residual distribution, calibration and
the latest prediction map.

## 9. What this does not show

- It does not estimate the causal effect of cameras. The causal estimate remains
  in `results/estimates.json` and `docs/REPORT.md`.
- It does not validate the model outside crash-dense Chicago locations.
- It does not provide calibrated predictive intervals; it provides point
  forecasts and a Poisson deviance loss.
- It does not use future weather. Weather features are lagged.
- It cannot extrapolate beyond the training date range.

## 10. Reproduce

```bash
python scripts/train_lightgbm.py --trials 40 --timeout 900
python scripts/evaluate_lightgbm.py
python scripts/build_prediction_report.py
```
