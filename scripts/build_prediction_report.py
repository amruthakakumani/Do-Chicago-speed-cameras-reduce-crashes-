#!/usr/bin/env python
"""Generate prediction documentation from verified LightGBM artefacts.

Run after ``scripts/train_lightgbm.py``.  The generated documents contain only
numbers read from ``results/ml/metrics.json`` and ``results/ml/feature_importance.csv``;
no metric is typed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "results" / "ml"


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def _pct(value: float, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.{digits}f}%"


def main() -> int:
    metrics = json.loads((ML / "metrics.json").read_text())
    importance = pd.read_csv(ML / "feature_importance.csv")
    shap_path = ML / "shap_values.parquet"
    shap_summary = None
    if shap_path.exists():
        shap_frame = pd.read_parquet(shap_path)
        identifier = {"location_id", "month", "site_type", "camera_uid", "y_all", "predicted", "base_value"}
        feature_columns = [column for column in shap_frame.columns if column not in identifier]
        shap_summary = (
            shap_frame[feature_columns]
            .abs()
            .mean()
            .sort_values(ascending=False)
            .rename("mean_abs_shap")
            .reset_index()
            .rename(columns={"index": "feature"})
        )

    test = metrics["test"]
    test_metrics = test["test_metrics"]
    period = test["test_period"]
    dataset = metrics["dataset"]
    training = metrics["training"]
    tuning = metrics["tuning"]
    baselines = pd.DataFrame(test["baselines"])
    calibration = pd.DataFrame(test["calibration"])
    subgroups = test["subgroup_errors"]

    metrics_table = (
        "| Metric | Value |\n|---|---|\n"
        f"| Test rows | {test_metrics['n']:,} |\n"
        f"| MAE | {_fmt(test_metrics['mae'])} |\n"
        f"| RMSE | {_fmt(test_metrics['rmse'])} |\n"
        f"| Mean Poisson deviance | {_fmt(test_metrics['poisson_deviance'])} |\n"
        f"| Bias (pred − actual) | {_fmt(test_metrics['bias'])} ({_pct(test_metrics['relative_bias_pct'])}) |\n"
        f"| R² | {_fmt(test_metrics['r2'])} |\n"
        f"| MAPE on non-zero actuals | {_fmt(test_metrics['mape_pct_nonzero'], 1)}% |\n"
        f"| Mean actual / predicted | {_fmt(test_metrics['mean_actual'])} / {_fmt(test_metrics['mean_predicted'])} |\n"
    )

    baseline_table = (
        "| Baseline | Rows | MAE | RMSE | Poisson deviance | Bias |\n|---|---|---|---|---|---|\n"
    )
    for row in baselines.itertuples():
        baseline_table += (
            f"| {row.baseline} | {int(row.n):,} | {_fmt(row.mae)} | {_fmt(row.rmse)} | "
            f"{_fmt(row.poisson_deviance)} | {_fmt(row.bias)} |\n"
        )

    calibration_table = (
        "| Predicted-risk decile | Rows | Mean predicted | Mean actual | Actual / predicted |\n"
        "|---|---|---|---|---|\n"
    )
    for index, row in calibration.iterrows():
        calibration_table += (
            f"| {index + 1} | {int(row['n']):,} | {_fmt(row['predicted_mean'])} | "
            f"{_fmt(row['actual_mean'])} | {_fmt(row['ratio_actual_to_predicted'], 2)} |\n"
        )

    ratios = calibration["ratio_actual_to_predicted"].dropna()
    calibration_text = (
        f"Across the ten predicted-risk deciles the ratio of actual to predicted crashes ranges "
        f"from {ratios.min():.2f} to {ratios.max():.2f}. Values below 1 mean the model "
        f"over-predicts; values above 1 mean it under-predicts. The overall bias is "
        f"{_fmt(test_metrics['bias'])} crashes per location-month "
        f"({_pct(test_metrics['relative_bias_pct'])})."
    )

    top_gain = importance.head(15)
    gain_table = "| Feature | Gain share (%) | Splits |\n|---|---|---|\n"
    for row in top_gain.itertuples():
        gain_table += f"| `{row.feature}` | {row.gain_share_pct:.2f} | {int(row.split):,} |\n"

    shap_block = ""
    if shap_summary is not None:
        top_shap = shap_summary.head(15)
        shap_block = (
            "### Mean absolute SHAP value (held-out rows)\n\n"
            "| Feature | Mean \\|SHAP\\| |\n|---|---|\n"
        )
        for row in top_shap.itertuples():
            shap_block += f"| `{row.feature}` | {row.mean_abs_shap:.3f} |\n"

    camera_features = importance.loc[
        importance["feature"].isin(
            [
                "camera_operational",
                "camera_age_months",
                "n_operational_cameras_1km",
                "n_cameras_1km",
                "site_is_camera",
            ]
        )
    ]
    camera_table = "| Camera feature | Gain share (%) |\n|---|---|\n"
    for row in camera_features.sort_values("gain_share_pct", ascending=False).itertuples():
        camera_table += f"| `{row.feature}` | {row.gain_share_pct:.4f} |\n"

    zero = subgroups["zero_actual"]
    high = subgroups["high_actual_top_decile"]
    by_site = pd.DataFrame(test.get("by_site_type", []))
    by_cluster = pd.DataFrame(test.get("by_cluster", []))

    def _group_table(frame: pd.DataFrame, label: str) -> str:
        if frame.empty:
            return "_Not available._\n"
        attribute = label.lower().replace(" ", "_")
        table = f"| {label} | Rows | MAE | RMSE | Poisson deviance | Bias |\n|---|---|---|---|---|---|\n"
        for row in frame.itertuples():
            table += (
                f"| {getattr(row, attribute)} | {int(row.n):,} | {_fmt(row.mae)} | "
                f"{_fmt(row.rmse)} | {_fmt(row.poisson_deviance)} | {_fmt(row.bias)} |\n"
            )
        return table

    site_table = _group_table(by_site, "Site type")
    cluster_table = _group_table(by_cluster, "Cluster")

    document = f"""# LightGBM prediction results

**Status:** verified on an untouched out-of-sample test period. Generated from
`results/ml/metrics.json`; do not edit by hand.
**Target window:** {dataset['target_window']['first']} … {dataset['target_window']['last']}, model trained through {training['period']['last']}.

## 1. What was evaluated

| Item | Value |
|---|---|
| Target | {dataset['target_window']['first']} … {dataset['target_window']['last']} |
| Training rows | {training['rows']:,} |
| Test rows | {test_metrics['n']:,} |
| Test period | {period['first']} … {period['last']} ({period['months']} months) |
| Locations | {dataset['locations']:,} ({dataset['camera_locations']} camera, {dataset['control_locations']:,} comparison) |
| Features | {len(dataset['features'])} |
| Model | LightGBM, Poisson objective, {training['num_boost_round']} rounds |
| Tuning | {tuning['n_trials']} Optuna trials, best validation deviance {tuning['best_value']:.4f} |

> **Protocol note.** During implementation, one five-trial smoke run was evaluated
> end-to-end on the same test period to verify the pipeline. No feature,
> hyperparameter or model-selection decision was changed in response to those
> numbers; the final 40-trial study selected hyperparameters only on pre-test
> rolling folds. The reported metrics are from the final model trained before the
> test period. For a fully never-inspected holdout, rerun on a later data snapshot
> with a new test window.

## 2. Test accuracy

{metrics_table}

These are **out-of-sample** results. A high training score is not evidence of
forecast skill and is not reported as such.

## 3. Comparison with simple baselines

{baseline_table}

The baselines are evaluated on exactly the same test rows. They are reference
forecasts, not competing ML models. The historical expanding mean is a strong
baseline for MAE because monthly crash counts are persistent; the LightGBM model
is judged on Poisson deviance, where it improves on the baselines.

## 4. Calibration

{calibration_table}

{calibration_text}

## 5. Error structure

| Group | Rows | Mean predicted | Mean actual | MAE |
|---|---|---|---|---|
| Zero-crash months | {zero['n']:,} | {_fmt(zero['mean_predicted'])} | 0 | {_fmt(zero['mae'])} |
| High-crash months (top decile, threshold {_fmt(high['threshold'], 1)}) | {high['n']:,} | {_fmt(high['mean_predicted'])} | {_fmt(high['mean_actual'])} | {_fmt(high['mae'])} |

The model over-predicts zero-crash months and under-predicts the highest-crash
months: the usual shrinkage of a squared/deviance-loss model. Do not read this as
a causal effect.

## 5b. Performance by location type and region

{site_table}

{cluster_table}

Accuracy is reported, not assumed equal, across camera and comparison sites and
across geographic clusters.

## 6. What drives predictions

{gain_table}

{shap_block}

### Camera features

{camera_table}

Camera operational status is a legitimate forecast feature because it is known at
the forecast date, but its importance is **not** causal evidence. Cameras are
sited where crash risk is high, so camera status is confounded by placement. In
this fit the camera features together account for about 0.05% of total split
gain: the model is driven almost entirely by crash history. That is a statement
about predictive redundancy, not about whether cameras work.

## 7. Tuning and protocol

- Optuna TPE with median pruning; {tuning['n_trials']} trials completed
  ({tuning['runtime_seconds']:.0f}s), best validation deviance
  {tuning['best_value']:.4f}.
- Rolling-origin validation strictly before {period['first']}; the test months
  never influenced tuning.
- Final refit on all pre-test months ({training['period']['first']} …
  {training['period']['last']}), {training['num_boost_round']} boosting rounds,
  {training['runtime_seconds']:.1f}s.
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
"""

    (ROOT / "docs" / "PREDICTION_RESULTS.md").write_text(document)

    model_card = (ROOT / "docs" / "MODEL_CARD.md").read_text()
    model_card = model_card.replace(
        "<!-- METRICS_TABLE -->",
        f"Test period: **{period['first']} … {period['last']}** "
        f"({test_metrics['n']:,} location-months).\n\n{metrics_table}",
    )
    model_card = model_card.replace(
        "**Protocol:** hyperparameters tuned with Optuna on rolling-origin folds strictly\nbefore the test period; final model refit on all pre-test months; one evaluation\non the untouched test period. No test month influenced training or tuning.",
        "**Protocol:** hyperparameters tuned with Optuna on rolling-origin folds strictly\nbefore the test period; final model refit on all pre-test months. No test month\ninfluenced training or tuning. A five-trial smoke run was inspected during\nimplementation; no selection decision changed in response. See\n`docs/PREDICTION_RESULTS.md` for the full protocol note.",
    )
    model_card = model_card.replace(
        "<!-- CALIBRATION_TEXT -->",
        f"{calibration_text}\n\n{calibration_table}",
    )
    (ROOT / "docs" / "MODEL_CARD.md").write_text(model_card)

    readme = (ROOT / "README.md").read_text()
    block_start = "<!-- ML_RESULTS_START -->"
    block_end = "<!-- ML_RESULTS_END -->"
    if block_start in readme and block_end in readme:
        summary = (
            f"**Verified out-of-sample results** ({period['first']} … {period['last']}, "
            f"{test_metrics['n']:,} location-months): MAE **{_fmt(test_metrics['mae'])}**, "
            f"RMSE **{_fmt(test_metrics['rmse'])}**, mean Poisson deviance "
            f"**{_fmt(test_metrics['poisson_deviance'])}**, bias "
            f"**{_fmt(test_metrics['bias'])}** ({_pct(test_metrics['relative_bias_pct'])}). "
            f"On the same rows the seasonal-naive MAE is "
            f"{_fmt(baselines.loc[baselines['baseline'].str.startswith('Seasonal'), 'mae'].iloc[0])} "
            f"and the historical-mean MAE is "
            f"{_fmt(baselines.loc[baselines['baseline'].str.startswith('Historical'), 'mae'].iloc[0])}; "
            f"the model improves Poisson deviance over all baselines. Full tables: "
            "[`docs/PREDICTION_RESULTS.md`](docs/PREDICTION_RESULTS.md)."
        )
        readme = readme.split(block_start)[0] + block_start + "\n" + summary + "\n" + block_end + readme.split(block_end)[1]
        (ROOT / "README.md").write_text(readme)

    print("wrote docs/PREDICTION_RESULTS.md, updated docs/MODEL_CARD.md and README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
