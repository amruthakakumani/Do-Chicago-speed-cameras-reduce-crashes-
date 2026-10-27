#!/usr/bin/env python
"""Train and evaluate LightGBM V2 with prediction intervals and geographic holdout.

V1 artifacts are never overwritten.  V2 is deployed only if it passes the
pre-declared validation criterion in ``docs/MODEL_SELECTION_DECISION.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chicago_cameras.config import load_config
from chicago_cameras.ml.conformal import apply_conformal, evaluate_methods, select_method
from chicago_cameras.ml.dataset import FEATURE_COLUMNS as V1_FEATURE_COLUMNS
from chicago_cameras.ml.dataset import build_dataset
from chicago_cameras.ml.dataset_v2 import build_dataset_v2
from chicago_cameras.ml.evaluate import (
    calibration_table,
    full_evaluation,
    regression_metrics,
    subgroup_errors,
)
from chicago_cameras.ml.explain import (
    compute_shap_values,
    gain_importance,
    plot_actual_vs_predicted,
    plot_calibration,
    plot_feature_importance,
    plot_monthly_error,
    plot_prediction_map,
    plot_residuals,
    plot_shap_summary,
    plot_shap_waterfall,
)
from chicago_cameras.ml.geographic import geographic_split
from chicago_cameras.ml.train import (
    evaluate_folds,
    merge_params,
    model_metadata,
    save_model,
    train_final,
    write_metadata,
)
from chicago_cameras.ml.tune import run_tuning
from chicago_cameras.ml.validation import fold_frames, rolling_origin_folds

LOGGER = logging.getLogger("train_lightgbm_v2")


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _fold_predictions(
    bundle,
    params: dict,
    folds,
    num_threads: int = 8,
    max_rounds: int = 3000,
    early_stopping: int = 100,
) -> pd.DataFrame:
    """Out-of-fold predictions for conformal calibration."""
    records: list[pd.DataFrame] = []
    for fold, train, validation in fold_frames(list(folds), bundle.frame):
        dtrain = lgb.Dataset(
            train[bundle.feature_columns],
            label=train["y_all"],
            categorical_feature=bundle.categorical_features,
            free_raw_data=True,
        )
        dvalid = lgb.Dataset(
            validation[bundle.feature_columns],
            label=validation["y_all"],
            categorical_feature=bundle.categorical_features,
            reference=dtrain,
            free_raw_data=True,
        )
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=max_rounds,
            valid_sets=[dvalid],
            valid_names=["validation"],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(0)],
        )
        prediction = model.predict(
            validation[bundle.feature_columns],
            num_iteration=int(model.best_iteration or max_rounds),
        )
        held = validation[["location_id", "month", "y_all", "site_type", "cluster"]].copy()
        held["predicted"] = prediction
        held["fold"] = fold.name
        records.append(held)
    return pd.concat(records, ignore_index=True)


def _train_subset(
    bundle,
    params: dict,
    train: pd.DataFrame,
    features: list[str],
    num_boost_round: int,
):
    categorical = [column for column in bundle.categorical_features if column in features]
    dataset = lgb.Dataset(
        train[features],
        label=train["y_all"],
        categorical_feature=categorical,
        free_raw_data=True,
    )
    return lgb.train(params, dataset, num_boost_round=int(num_boost_round))


def _predict_subset(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.clip(model.predict(frame[features]), 0.0, None)


def _metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    return {
        "metrics": regression_metrics(frame["y_all"], prediction),
        "subgroups": subgroup_errors(frame["y_all"].to_numpy(), prediction),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--quick", action="store_true", help="5 Optuna trials")
    parser.add_argument("--reuse-tuning", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    started = time.time()
    config = load_config(args.config)
    results_dir = ROOT / "results" / "ml_v2"
    figures_dir = results_dir / "figures"
    models_dir = ROOT / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    v1_metrics = _json(ROOT / "results" / "ml" / "v1" / "metrics.json")
    v1_test_predictions = pd.read_parquet(ROOT / "results" / "ml" / "v1" / "predictions.parquet")
    v1_test_predictions = v1_test_predictions.loc[
        v1_test_predictions["split"] == "test"
    ].reset_index(drop=True)
    v1_params = merge_params(
        v1_metrics["training"]["hyperparameters"], seed=int(config["inference"]["seed"]), num_threads=args.threads
    )
    v1_rounds = int(v1_metrics["training"]["num_boost_round"])

    print("building V2 dataset and V1 comparison dataset")
    bundle = build_dataset_v2(args.config, refresh=args.refresh)
    v1_bundle = build_dataset(args.config, refresh=False)
    print(
        f"V2: {len(bundle.frame):,} rows | {bundle.metadata['n_locations']:,} locations | "
        f"{len(bundle.feature_columns)} features"
    )

    best_path = results_dir / "best_params.json"
    if args.reuse_tuning and best_path.exists():
        payload = _json(best_path)
        tuning = SimpleNamespace(
            best_params=payload["params"],
            best_iterations=payload["best_iterations"],
            best_value=payload["best_value"],
            n_trials=payload["n_trials"],
            runtime_seconds=payload.get("runtime_seconds", 0.0),
        )
        print(f"reusing V2 tuning: {tuning.n_trials} trials, deviance {tuning.best_value:.5f}")
    else:
        tuning = run_tuning(
            bundle,
            n_trials=5 if args.quick else args.trials,
            timeout=args.timeout,
            seed=int(config["inference"]["seed"]),
            num_threads=args.threads,
        )
        _save(
            best_path,
            {
                "params": tuning.best_params,
                "best_iterations": tuning.best_iterations,
                "best_value": tuning.best_value,
                "n_trials": tuning.n_trials,
                "runtime_seconds": tuning.runtime_seconds,
            },
        )
        tuning.trials.to_csv(results_dir / "tuning_trials.csv", index=False)
        print(f"V2 Optuna: {tuning.n_trials} trials, best deviance {tuning.best_value:.5f}")

    v2_params = merge_params(
        tuning.best_params, seed=int(config["inference"]["seed"]), num_threads=args.threads
    )
    folds = rolling_origin_folds(
        bundle.metadata["target_window"]["first"], bundle.metadata["test_start"], 3, 6
    )

    print("evaluating V1 and V2 on identical rolling-origin folds")
    v1_fold_metrics = evaluate_folds(v1_bundle, v1_params, folds, num_threads=args.threads)
    v2_fold_metrics = evaluate_folds(bundle, v2_params, folds, num_threads=args.threads)
    v1_validation = {
        "mean_deviance": float(v1_fold_metrics["poisson_deviance"].mean()),
        "mean_mae": float(v1_fold_metrics["mae"].mean()),
        "folds": v1_fold_metrics.to_dict(orient="records"),
    }
    v2_validation = {
        "mean_deviance": float(v2_fold_metrics["poisson_deviance"].mean()),
        "mean_mae": float(v2_fold_metrics["mae"].mean()),
        "folds": v2_fold_metrics.to_dict(orient="records"),
    }
    dev_improvement = 1.0 - v2_validation["mean_deviance"] / v1_validation["mean_deviance"]
    mae_change = v2_validation["mean_mae"] / v1_validation["mean_mae"] - 1.0
    passed = dev_improvement >= 0.01 and mae_change <= 0.02
    selected_model = "v2" if passed else "v1"
    print(
        f"validation: V1 dev {v1_validation['mean_deviance']:.5f} / MAE {v1_validation['mean_mae']:.4f}; "
        f"V2 dev {v2_validation['mean_deviance']:.5f} / MAE {v2_validation['mean_mae']:.4f}; "
        f"dev improvement {dev_improvement:+.2%}; selected {selected_model}"
    )

    print("training final V2 candidate")
    trained = train_final(
        bundle,
        tuning.best_params,
        tuning.best_iterations,
        seed=int(config["inference"]["seed"]),
        num_threads=args.threads,
    )
    v2_model_path = models_dir / "v2_lightgbm_crash_model.txt"
    save_model(trained, v2_model_path)

    inference_started = time.perf_counter()
    from chicago_cameras.ml.predict import attach_predictions

    v2_predictions = attach_predictions(bundle, trained.booster)
    inference_seconds = time.perf_counter() - inference_started
    v2_predictions.to_parquet(results_dir / "predictions.parquet", index=False)
    v2_test = v2_predictions.loc[v2_predictions["split"] == "test"].reset_index(drop=True)
    v2_evaluation = full_evaluation(v2_test)
    v2_test_metrics = v2_evaluation["test_metrics"]
    print(
        f"V2 retrospective test: MAE {v2_test_metrics['mae']:.4f}, "
        f"deviance {v2_test_metrics['poisson_deviance']:.4f}, bias {v2_test_metrics['bias']:+.4f}"
    )

    # Fair V1 vs V2 retrospective comparison on identical rows.
    comparison_frame = v2_test[["location_id", "month", "y_all", "site_type", "cluster"]].merge(
        v1_test_predictions[["location_id", "month", "predicted"]].rename(
            columns={"predicted": "v1_predicted"}
        ),
        on=["location_id", "month"],
        how="inner",
    ).merge(
        v2_test[["location_id", "month", "predicted"]].rename(columns={"predicted": "v2_predicted"}),
        on=["location_id", "month"],
        how="inner",
    )
    v1_metrics_compare = regression_metrics(comparison_frame["y_all"], comparison_frame["v1_predicted"])
    v2_metrics_compare = regression_metrics(comparison_frame["y_all"], comparison_frame["v2_predicted"])
    comparison = {
        "n_rows": len(comparison_frame),
        "v1": v1_metrics_compare,
        "v2": v2_metrics_compare,
        "delta_mae": float(v2_metrics_compare["mae"] - v1_metrics_compare["mae"]),
        "relative_mae_change": float(v2_metrics_compare["mae"] / v1_metrics_compare["mae"] - 1.0),
        "delta_deviance": float(
            v2_metrics_compare["poisson_deviance"] - v1_metrics_compare["poisson_deviance"]
        ),
        "relative_deviance_change": float(
            v2_metrics_compare["poisson_deviance"] / v1_metrics_compare["poisson_deviance"] - 1.0
        ),
        "note": "Retrospective diagnostic on the previously inspected 2025-09..2026-08 period.",
    }
    v1_comparison_errors = subgroup_errors(
        comparison_frame["y_all"].to_numpy(), comparison_frame["v1_predicted"].to_numpy()
    )
    v2_comparison_errors = subgroup_errors(
        comparison_frame["y_all"].to_numpy(), comparison_frame["v2_predicted"].to_numpy()
    )

    print("conformal calibration on out-of-fold predictions")
    calibration_bundle = v1_bundle if selected_model == "v1" else bundle
    calibration_params = v1_params if selected_model == "v1" else v2_params
    calibration = _fold_predictions(
        calibration_bundle, calibration_params, folds, num_threads=args.threads
    )
    method_selection = select_method(calibration, alpha=0.1)
    method = method_selection["selected"]
    interval_90, interval_metrics_90 = apply_conformal(calibration, v2_test if selected_model == "v2" else v1_test_predictions, method, 0.1)
    interval_50, interval_metrics_50 = apply_conformal(calibration, v2_test if selected_model == "v2" else v1_test_predictions, method, 0.5)
    method_comparison = evaluate_methods(calibration, v2_test if selected_model == "v2" else v1_test_predictions)
    selected_test = (v2_test if selected_model == "v2" else v1_test_predictions).copy().reset_index(drop=True)
    selected_test["lower_90"] = interval_90.lower
    selected_test["upper_90"] = interval_90.upper
    selected_test["lower_50"] = interval_50.lower
    selected_test["upper_50"] = interval_50.upper
    selected_test.to_parquet(results_dir / "selected_predictions.parquet", index=False)
    print(
        f"90% intervals: coverage {interval_metrics_90['empirical_coverage']:.3f}, "
        f"mean width {interval_metrics_90['mean_width']:.3f}"
    )

    print("geographic holdout")
    geo = geographic_split(bundle.locations)
    holdout_ids = set(geo.holdout["location_id"])
    train_ids = set(geo.train["location_id"])
    frame = bundle.frame
    geo_val = frame.loc[
        frame["location_id"].isin(holdout_ids)
        & frame["month"].between(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-08-01"))
    ].copy()
    geo_only_train = frame.loc[
        frame["location_id"].isin(train_ids) & (frame["month"] <= pd.Timestamp("2025-08-01"))
    ].copy()
    combined_train = frame.loc[
        frame["location_id"].isin(train_ids) & (frame["month"] <= pd.Timestamp("2025-02-01"))
    ].copy()

    geo_only_model = _train_subset(
        bundle, v2_params, geo_only_train, bundle.feature_columns, trained.num_boost_round
    )
    geo_only_pred = _predict_subset(geo_only_model, geo_val, bundle.feature_columns)
    geo_only = _metrics(geo_val, geo_only_pred)

    combined_model = _train_subset(
        bundle, v2_params, combined_train, bundle.feature_columns, trained.num_boost_round
    )
    combined_pred = _predict_subset(combined_model, geo_val, bundle.feature_columns)
    combined = _metrics(geo_val, combined_pred)

    v1_features = [column for column in bundle.feature_columns if column in V1_FEATURE_COLUMNS]
    v1_feature_model = _train_subset(
        bundle, v1_params, combined_train, v1_features, v1_rounds
    )
    v1_feature_pred = _predict_subset(v1_feature_model, geo_val, v1_features)
    v1_feature = _metrics(geo_val, v1_feature_pred)

    geographic = {
        "split": geo.info,
        "geo_only": geo_only,
        "combined_temporal_geographic": combined,
        "feature_ablation_v1_vs_v2": {
            "v1_features": v1_feature,
            "v2_features": combined,
            "note": "Same training rows and evaluation rows; V1 hyperparameters for the V1 feature set, V2 hyperparameters for the V2 feature set.",
        },
    }

    # SHAP diagnostics for the V2 candidate.
    shap_source = bundle.frame.loc[bundle.frame["split"] == "test"].reset_index(drop=True)
    if len(shap_source) != len(v2_test):  # pragma: no cover - alignment guard
        raise RuntimeError("V2 test frames are not aligned for SHAP")
    shap_table, shap_features = compute_shap_values(
        trained.booster, shap_source, bundle.feature_columns, max_rows=20_000, seed=int(config["inference"]["seed"])
    )
    shap_output = pd.concat(
        [
            v2_test[["location_id", "month", "site_type", "camera_uid", "y_all", "predicted"]].reset_index(drop=True),
            shap_table.reset_index(drop=True),
        ],
        axis=1,
    )
    shap_output.to_parquet(results_dir / "shap_values.parquet", index=False)

    importance = gain_importance(trained.booster)
    importance.to_csv(results_dir / "feature_importance.csv", index=False)
    pd.DataFrame(v2_evaluation["calibration"]).to_csv(results_dir / "calibration.csv", index=False)
    pd.DataFrame(v2_evaluation["monthly"]).to_csv(results_dir / "monthly_metrics.csv", index=False)

    plot_feature_importance(importance, figures_dir)
    plot_shap_summary(
        shap_table[bundle.feature_columns].to_numpy(),
        shap_features[bundle.feature_columns].reset_index(drop=True),
        figures_dir,
    )
    top_row = int(np.argmax(v2_test["predicted"].to_numpy()))
    position = top_row if top_row in shap_table.index else 0
    plot_shap_waterfall(
        shap_table[bundle.feature_columns].to_numpy(),
        shap_features[bundle.feature_columns].reset_index(drop=True),
        float(shap_table["base_value"].iloc[0]),
        position,
        figures_dir,
    )
    plot_actual_vs_predicted(v2_test["y_all"].to_numpy(), v2_test["predicted"].to_numpy(), figures_dir)
    plot_monthly_error(pd.DataFrame(v2_evaluation["monthly"]), figures_dir)
    plot_residuals(v2_test["y_all"].to_numpy(), v2_test["predicted"].to_numpy(), figures_dir)
    plot_calibration(pd.DataFrame(v2_evaluation["calibration"]), figures_dir)
    plot_prediction_map(v2_predictions.loc[v2_predictions["split"] == "test"], figures_dir)

    top_features = importance.head(20).to_dict(orient="records")
    metadata = model_metadata(
        bundle,
        trained,
        tuning,
        version="2.0.0",
        model_file="models/v2_lightgbm_crash_model.txt",
        extra={
            "validation_comparison": {"v1": v1_validation, "v2": v2_validation},
            "selection_decision": {
                "selected_model": selected_model,
                "relative_deviance_improvement": dev_improvement,
                "relative_mae_change": mae_change,
                "rule": "deviance improvement >= 1% and MAE regression <= 2% on pre-test folds",
            },
            "retrospective_test_comparison": comparison,
            "conformal": {
                "method_selection": method_selection,
                "interval_90": interval_metrics_90,
                "interval_50": interval_metrics_50,
            },
            "geographic": geographic,
            "feature_importance_top_20": top_features,
            "inference_seconds": round(inference_seconds, 3),
        },
    )
    write_metadata(metadata, models_dir / "v2_model_metadata.json")

    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset": {
            "rows": len(bundle.frame),
            "training_rows": trained.train_rows,
            "test_rows": len(v2_test),
            "locations": bundle.metadata["n_locations"],
            "camera_locations": bundle.metadata["n_camera_locations"],
            "control_locations": bundle.metadata["n_control_locations"],
            "features": bundle.feature_columns,
            "new_features": bundle.metadata["new_features"],
            "target_window": bundle.metadata["target_window"],
            "test_start": bundle.metadata["test_start"],
            "road_network": bundle.metadata["road_network"],
            "traffic_volume": bundle.metadata["traffic_volume"],
        },
        "training": {
            "rows": trained.train_rows,
            "period": {
                "first": str(trained.train_period[0].date()),
                "last": str(trained.train_period[1].date()),
            },
            "num_boost_round": trained.num_boost_round,
            "runtime_seconds": trained.training_seconds,
            "hyperparameters": trained.params,
        },
        "tuning": {
            "n_trials": tuning.n_trials,
            "best_value": tuning.best_value,
            "runtime_seconds": tuning.runtime_seconds,
            "best_iterations": tuning.best_iterations,
        },
        "validation": {
            "v1": v1_validation,
            "v2": v2_validation,
            "relative_deviance_improvement": dev_improvement,
            "relative_mae_change": mae_change,
            "selection_rule": "deviance improvement >= 1% and MAE regression <= 2% on pre-test folds",
            "selected_model": selected_model,
        },
        "test_retrospective": {
            "v1": v1_metrics_compare,
            "v2": v2_metrics_compare,
            "comparison": comparison,
            "v1_zero_high": v1_comparison_errors,
            "v2_zero_high": v2_comparison_errors,
            "v1_full": {
                "calibration": calibration_table(
                    comparison_frame["y_all"].to_numpy(), comparison_frame["v1_predicted"].to_numpy()
                ).to_dict(orient="records")
            },
            "v2_full": v2_evaluation,
        },
        "conformal": {
            "selected_model": selected_model,
            "method_selection": method_selection,
            "interval_90": interval_metrics_90,
            "interval_50": interval_metrics_50,
            "method_comparison": method_comparison,
        },
        "geographic": geographic,
        "test_metrics_selected": (
            v2_test_metrics
            if selected_model == "v2"
            else regression_metrics(v1_test_predictions["y_all"], v1_test_predictions["predicted"])
        ),
        "performance": {
            "training_seconds": trained.training_seconds,
            "inference_seconds_for_all_rows": inference_seconds,
            "peak_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024**2 if platform.system() == "Darwin" else 1024),
                1,
            ),
            "rows_scored": len(v2_predictions),
        },
        "top_features": top_features,
        "total_runtime_seconds": time.time() - started,
    }
    _save(results_dir / "metrics.json", payload)

    deployment = {
        "selected_model": selected_model,
        "model_version": "2.0.0" if selected_model == "v2" else "1.0.0",
        "model_file": (
            "models/v2_lightgbm_crash_model.txt" if selected_model == "v2" else "models/v1/lightgbm_crash_model.txt"
        ),
        "predictions_path": (
            "results/ml_v2/predictions.parquet" if selected_model == "v2" else "results/ml/v1/predictions.parquet"
        ),
        "test_predictions_path": "results/ml_v2/selected_predictions.parquet",
        "selection_rule": payload["validation"]["selection_rule"],
        "validation": {
            "v1": {"deviance": v1_validation["mean_deviance"], "mae": v1_validation["mean_mae"]},
            "v2": {"deviance": v2_validation["mean_deviance"], "mae": v2_validation["mean_mae"]},
            "relative_deviance_improvement": dev_improvement,
            "relative_mae_change": mae_change,
        },
        "retrospective_test": {
            "v1": v1_metrics_compare,
            "v2": v2_metrics_compare,
            "note": "Retrospective only; this period was already examined in V1.",
        },
        "conformal_method": method,
        "interval_coverage_90": interval_metrics_90["empirical_coverage"],
        "interval_mean_width_90": interval_metrics_90["mean_width"],
    }
    _save(results_dir / "deployment.json", deployment)

    print(f"selected model: {selected_model}")
    print(f"V2 test MAE {v2_test_metrics['mae']:.4f} deviance {v2_test_metrics['poisson_deviance']:.4f}")
    print(f"artefacts: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
