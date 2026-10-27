# Model card — LightGBM crash-count forecaster

## 1. Model details

| Field | Value |
|---|---|
| Name | `lightgbm_poisson_crash_model` |
| Version | 1.0.0 |
| Model type | Gradient-boosted decision trees (LightGBM), Poisson objective |
| Framework | LightGBM 4.7 |
| Target | Reported crashes in a 250 m catchment during a calendar month |
| Unit | Location-month |
| Artifact | `models/v1/lightgbm_crash_model.txt` |
| Metadata | `models/v1/model_metadata.json` |
| Seed | 20261024 |
| Owner | Chicago speed-camera research project (see repository contributors) |
| License | MIT (repository code and research materials) |

## 2. Intended use

- One-month-ahead **descriptive forecasting** of crash counts at crash-dense
  Chicago locations.
- Exploring which observed factors are associated with predicted crash risk.
- A map-based teaching and research interface that separates prediction from
  causal inference.
- Evaluating forecast accuracy on a temporally held-out period.

## 3. Out-of-scope use

- **Do not** use this model to estimate the causal effect of installing, keeping,
  or removing a speed camera. That is the job of the difference-in-differences,
  event-study, synthetic-control and placebo analyses.
- **Do not** assign a study-wide causal estimate to an individual camera.
- **Do not** use predictions for enforcement, ticketing, insurance, policing, or
  resource allocation without independent validation, governance and legal
  review.
- **Do not** use the model outside the City of Chicago or for dates far beyond
  the training window (it cannot extrapolate tree splits).
- **Do not** interpret SHAP values as causal effects.

## 4. Training data

| Item | Value |
|---|---|
| Source | City of Chicago Data Portal `85ca-t3if` (Traffic Crashes), snapshot 2026-10-24 |
| Geography | Camera catchments (129 after overlap deduplication) plus 1,226 comparison sites |
| Target window | 2017-01 … 2026-08 (116 months); the first eight target months precede citywide reporting |
| Training rows | 140,920 location-months (through 2025-08) |
| Test rows | 16,260 location-months (2025-09 … 2026-08) |
| Weather | Open-Meteo archive, lagged; cached with provenance |
| Holidays | `holidays` US federal calendar |
| Missing targets | None; genuine zeros are retained |

The location set is **crash-dense and non-random**: it is the causal design's
camera and comparison geography, not a random sample of Chicago streets. The
target window begins eight months before the portal's September 2017 citywide
coverage start; the coverage-consistent sensitivity is documented in
`docs/BOTTOM_UP_AUDIT.md`.

## 5. Features

44 features in six groups: calendar, US holidays, lagged crash counts
(1/2/3/6/12 months), rolling means/standard deviations and expanding statistics,
severity shares, lagged weather, static geography (UTM coordinates, distance to
centre and cameras, k-means cluster), and camera operational context. See
`docs/METHODOLOGY.md` and `docs/FEATURE_AVAILABILITY_LEDGER.md` for the feature
and leakage rules.

Weather is lagged by at least one month; the forecast month's own weather is
never used. No target-derived feature is in the feature list. See §7.

## 6. Evaluation

**Protocol:** hyperparameters tuned with Optuna on rolling-origin folds strictly
before the test period; final model refit on all pre-test months. No test month
influenced training or tuning. A five-trial smoke run was inspected during
implementation; no selection decision changed in response. See
`docs/REPRODUCIBILITY.md` for the full protocol and commands.

Test period: **2025-09-01 … 2026-08-01** (16,260 location-months).

| Metric | Value |
|---|---|
| Test rows | 16,260 |
| MAE | 1.568 |
| RMSE | 2.093 |
| Mean Poisson deviance | 1.2233 |
| Bias (pred − actual) | +0.163 (+4.3%) |
| R² | 0.661 |
| MAPE on non-zero actuals | 52.3% |
| Mean actual / predicted | 3.769 / 3.932 |


## 7. Leakage controls

- All history features are shifted within location; rolling windows use prior
  months only.
- Severity shares use shifted 12-month sums.
- Weather is lagged.
- Months are never shuffled; folds and test are chronological.
- No target encoding or statistics fitted on validation/test.
- Mechanical leakage tests recompute lags and perturb future outcomes; the build
  fails if any check fails.
- Schema checks at inference compare model feature names/order with the input.

## 8. Calibration

Across the ten predicted-risk deciles the ratio of actual to predicted crashes ranges from 0.94 to 0.98. Values below 1 mean the model over-predicts; values above 1 mean it under-predicts. The overall bias is 0.162 crashes per location-month (+4.3%).

| Predicted-risk decile | Rows | Mean predicted | Mean actual | Actual / predicted |
|---|---|---|---|---|
| 1 | 1,626 | 1.051 | 1.001 | 0.95 |
| 2 | 1,626 | 1.402 | 1.329 | 0.95 |
| 3 | 1,626 | 1.787 | 1.682 | 0.94 |
| 4 | 1,626 | 2.232 | 2.158 | 0.97 |
| 5 | 1,626 | 2.765 | 2.699 | 0.98 |
| 6 | 1,626 | 3.381 | 3.264 | 0.97 |
| 7 | 1,626 | 4.114 | 3.952 | 0.96 |
| 8 | 1,626 | 5.044 | 4.867 | 0.96 |
| 9 | 1,626 | 6.643 | 6.412 | 0.97 |
| 10 | 1,626 | 10.898 | 10.329 | 0.95 |


## 9. Limitations and risks

1. **Prediction ≠ causation.** Camera features may be predictive because of
   siting, not because enforcement changes crashes.
2. **Non-random geography.** Accuracy at unobserved locations is unknown.
3. **Overdispersion and shrinkage.** Errors are larger for high-crash months;
   the model under-predicts the top decile and over-predicts zero months.
4. **One-step horizon.** Not validated at daily resolution or multi-month
   horizons.
5. **Regime change.** Reporting changes, camera operation changes, construction
   and policy shocks are absorbed by history features, not explicitly modelled.
6. **Weather is lagged**, not forecast.
7. **Tree extrapolation.** `year` and `month_index` are in-range for the test
   period; forecasts beyond the training range are not supported.
8. **Approximate intervals only.** Scale-normalised split-conformal intervals are
   available, but coverage is conditional (84.8% in the highest-risk decile at
   nominal 90%), and they are not causal confidence intervals.

## 10. Ethical considerations

- Crash data contain no personally identifying driver information in the
  modelling pipeline; the model predicts **counts at aggregated locations**, not
  individuals.
- Speed cameras are a surveillance and enforcement technology. Predictive
  mapping can stigmatise neighbourhoods; results should be presented with
  denominators, uncertainty and community context.
- Do not use predicted crash counts to justify enforcement intensity without
  causal evidence and equity review.
- The location set over-represents crash-dense areas; maps should state this.

## 11. Maintenance

- Retrain only with a documented data snapshot and the same temporal protocol.
- Keep the test period untouched; if a new test period is chosen, archive the old
  one and do not re-use it for tuning.
- Re-run `scripts/evaluate_lightgbm.py` after any retrain and update
  `docs/MODEL_EXPERIMENT_RESULTS.md`.
- Re-check feature schema and leakage tests after changing the dataset builder.

## 12. Reproduction

```bash
python scripts/train_lightgbm.py --trials 40 --timeout 900
python scripts/evaluate_lightgbm.py
streamlit run dashboard/app.py
```
