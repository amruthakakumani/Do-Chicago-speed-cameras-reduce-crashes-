# Generated results

Everything in this directory is produced by the pipeline and should not be edited
by hand. Raw and processed data are not committed.

## Causal study

| Path | Contents |
|---|---|
| `estimates.json` | Machine-readable causal estimates, diagnostics and placebo results |
| `provenance.json` | Data-quality and acquisition record |
| `analysis_panel.parquet` | Site-month panel used by the estimators |
| `sites.parquet` | Camera and comparison-site locations (WGS84) |
| `tables/` | Main estimates, event study, balance, severity, sensitivity and placebo tables |
| `figures/` | Causal diagnostic figures used by the notebook and supporting documentation |

## Deployed forecasting model (V1)

| Path | Contents |
|---|---|
| `ml/metrics.json` | Dataset, tuning, held-out test metrics, calibration and baselines |
| `ml/predictions.parquet` | Location-month predictions with split labels and coordinates |
| `ml/shap_values.parquet` | SHAP values for held-out test rows |
| `ml/crash_grid_monthly.parquet` | 500 m observed-crash grid for the map |
| `ml/camera_metadata.parquet` | Camera identifiers, addresses and go-live dates |
| `ml/feature_importance.csv`, `ml/calibration.csv`, `ml/monthly_metrics.csv` | Supporting evaluation tables |
| `ml/figures/` | V1 feature importance, SHAP, accuracy, error, residual and map figures |
| `ml_v2/deployment.json` | Model-selection record; selects V1 |
| `ml_v2/selected_predictions.parquet` | V1 test rows with conformal interval endpoints |
| `ml_v2/metrics.json`, `ml_v2/calibration.csv`, `ml_v2/monthly_metrics.csv` | V1/V2 comparison and interval coverage |
| `ml_v2/recursive_metrics.json` | Recursive benchmark |

The deployed model artifact is `models/v1/lightgbm_crash_model.txt`, with
metadata in `models/v1/model_metadata.json`. Predictions are **not** causal
effects.

## Supporting experiments and coverage

| Path | Contents |
|---|---|
| `bottom_up/` | Reporting-coverage, exposure, road-assignment and reconstruction outputs; dashboard coverage page and documentation |
| `statistical_hybrid/` | Compact negative-result record for the optimized LightGBM, Negative Binomial and hybrid comparison |
| `ml_v2/` | Retained V2 evaluation metrics; the V2 binary checkpoint is not part of the deployed release |

The experiment summary is documented in `docs/MODEL_EXPERIMENT_RESULTS.md` and
`docs/MODEL_SELECTION_DECISION.md`. V1 remains deployed because no candidate met
the predeclared 1% validation improvement rule.
