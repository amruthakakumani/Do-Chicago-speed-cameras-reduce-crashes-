# Generated results

Everything in this directory is produced by the pipeline and should not be edited by hand:

```bash
python -m chicago_cameras.pipeline --config config/analysis.yaml
```

| Path | Contents |
|---|---|
| `estimates.json` | Machine-readable bundle used by the dashboard and notebook |
| `provenance.json` | Data-quality and acquisition record for this run |
| `analysis_panel.parquet` | Site-month panel used by the estimators |
| `sites.parquet` | Camera and comparison-site locations (WGS84) |
| `tables/main_estimates.csv` | Naive, original baseline, FE OLS, conditional Poisson, cohort 2×2, replication |
| `tables/replication_estimates.csv` | Original-sample replication with corrected projection |
| `tables/event_study.csv` | Quarterly event-study coefficients |
| `tables/cohort_event_effects.csv` | Per-event-month cohort 2×2 effects |
| `tables/balance.csv` | Pre-period balance |
| `tables/subgroups.csv` | Exploratory heterogeneity with BH-adjusted p-values |
| `tables/severity.csv` | Injury and severe/fatal outcomes |
| `tables/sensitivity.csv` | Every robustness and falsification specification |
| `tables/synthetic_control.csv` | Synthetic-control fit and pseudo p-values |
| `tables/sc_leave_one_out.csv` | Donor influence |
| `tables/sc_in_time_placebo.csv` | Fake treatment dates |
| `tables/sc_placebo_records.csv` | Placebo-in-space draws |
| `tables/placebo_geography.csv` | Random-location placebo draws |
| `tables/violations_curve.csv` | Mechanism check |
| `figures/*.png`, `figures/*.svg` | Ten publication figures |

Figures are written in both PNG (220 dpi) and SVG. The dashboard only reads these artefacts; it
never refits a model.

## Machine-learning artefacts (`results/ml/`)

| Path | Contents |
|---|---|
| `metrics.json` | Dataset, training, tuning, test metrics, calibration, baselines, top features |
| `evaluation.json` | Re-evaluation output from `scripts/evaluate_lightgbm.py` |
| `predictions.parquet` | Location-month predictions with split labels and coordinates |
| `shap_values.parquet` | SHAP values for held-out test rows |
| `crash_grid_monthly.parquet` | 500 m observed-crash grid for map rendering |
| `camera_metadata.parquet` | Camera identifiers, addresses and go-live dates for the map |
| `feature_importance.csv` | Gain-based importance |
| `calibration.csv` | Predicted-risk decile calibration |
| `monthly_metrics.csv` | Monthly out-of-sample metrics |
| `best_params.json` | Selected Optuna hyperparameters and best iterations |
| `tuning_trials.csv` | Every Optuna trial |
| `figures/` | Feature importance, SHAP, accuracy, error, residual, calibration and prediction map |

Model artifacts live in `models/lightgbm_crash_model.txt` and
`models/model_metadata.json`. Predictions are **not** causal effects.


## V2 machine-learning artefacts (`results/ml_v2/`)

| Path | Contents |
|---|---|
| `metrics.json` | V2 dataset, tuning, validation, retrospective test, intervals, geographic and recursive diagnostics |
| `deployment.json` | Final model selection decision and artefact paths |
| `predictions.parquet` | V2 candidate predictions for all location-months |
| `selected_predictions.parquet` | Deployed-model test rows with 50% and 90% interval endpoints |
| `shap_values.parquet` | V2 SHAP values for held-out rows |
| `feature_importance.csv` | V2 gain importance |
| `calibration.csv` | V2 calibration |
| `recursive_metrics.json`, `recursive_predictions.parquet` | Genuine 12-month recursive path |
| `recursive_v1_predictions.parquet` | V1 recursive comparator |
| `best_params.json`, `tuning_trials.csv` | V2 Optuna study |
| `figures/` | V2 feature importance, SHAP, accuracy, error, calibration, map |

Archived V1 artifacts are in `results/ml/v1/`; V1 model files are in `models/v1/`.
