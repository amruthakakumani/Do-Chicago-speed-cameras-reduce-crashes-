#!/usr/bin/env python
"""Train, tune, evaluate and explain the single LightGBM crash model.

Usage::

    python scripts/train_lightgbm.py                 # 40 Optuna trials
    python scripts/train_lightgbm.py --quick         # 5 trials (development)
    python scripts/train_lightgbm.py --reuse-tuning  # refit with saved best params
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chicago_cameras.config import load_config  # noqa: E402
from chicago_cameras.ml.dataset import build_crash_grid, build_dataset  # noqa: E402
from chicago_cameras.ml.evaluate import full_evaluation  # noqa: E402
from chicago_cameras.ml.explain import (  # noqa: E402
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
from chicago_cameras.ml.predict import attach_predictions  # noqa: E402
from chicago_cameras.ml.train import (  # noqa: E402
    model_metadata,
    save_model,
    train_final,
    write_metadata,
)
from chicago_cameras.ml.tune import run_tuning  # noqa: E402

LOGGER = logging.getLogger("train_lightgbm")


def _load_tuning(path: Path) -> SimpleNamespace | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return SimpleNamespace(
        best_params=payload["params"],
        best_iterations=payload["best_iterations"],
        best_value=payload["best_value"],
        n_trials=payload["n_trials"],
        runtime_seconds=payload.get("runtime_seconds", 0.0),
    )


def _write_tuning(path: Path, tuning: SimpleNamespace) -> None:
    path.write_text(
        json.dumps(
            {
                "params": tuning.best_params,
                "best_iterations": tuning.best_iterations,
                "best_value": tuning.best_value,
                "n_trials": tuning.n_trials,
                "runtime_seconds": tuning.runtime_seconds,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--quick", action="store_true", help="5 Optuna trials")
    parser.add_argument("--reuse-tuning", action="store_true")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Re-download source data")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    started = time.time()
    config = load_config(args.config)
    results_dir = ROOT / "results" / "ml"
    figures_dir = results_dir / "figures"
    models_dir = ROOT / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("building location-month dataset")
    bundle = build_dataset(args.config, refresh=args.refresh)
    print(
        f"dataset: {len(bundle.frame):,} rows | {bundle.metadata['n_locations']:,} locations "
        f"| {bundle.metadata['n_camera_locations']} cameras | "
        f"{bundle.metadata['n_control_locations']:,} comparison sites | "
        f"{len(bundle.feature_columns)} features"
    )
    print(
        f"target window: {bundle.metadata['target_window']['first']} .. "
        f"{bundle.metadata['target_window']['last']} | test starts {bundle.metadata['test_start']}"
    )

    best_path = results_dir / "best_params.json"
    tuning: SimpleNamespace | None = None
    if args.reuse_tuning:
        tuning = _load_tuning(best_path)
        if tuning is None:
            raise SystemExit("--reuse-tuning requested but results/ml/best_params.json is missing")
        print(f"reusing tuning: {tuning.n_trials} trials, best deviance {tuning.best_value:.4f}")
    elif not args.skip_tuning:
        n_trials = 5 if args.quick else args.trials
        tuning = run_tuning(
            bundle,
            n_trials=n_trials,
            timeout=args.timeout,
            seed=int(config["inference"]["seed"]),
            num_threads=args.threads,
        )
        _write_tuning(best_path, tuning)
        tuning.trials.to_csv(results_dir / "tuning_trials.csv", index=False)
        print(
            f"Optuna: {tuning.n_trials} trials in {tuning.runtime_seconds:.0f}s; "
            f"best validation deviance {tuning.best_value:.4f}"
        )

    tuned_params = tuning.best_params if tuning else {}
    best_iterations = tuning.best_iterations if tuning else [800]
    trained = train_final(
        bundle,
        tuned_params,
        best_iterations,
        seed=int(config["inference"]["seed"]),
        num_threads=args.threads,
    )
    model_path = models_dir / "lightgbm_crash_model.txt"
    save_model(trained, model_path)
    print(
        f"trained: {trained.num_boost_round} rounds on {trained.train_rows:,} rows in "
        f"{trained.training_seconds:.1f}s"
    )

    evaluation_started = time.time()
    predictions = attach_predictions(bundle, trained.booster)
    test_frame = predictions.loc[predictions["split"] == "test"].reset_index(drop=True)
    evaluation = full_evaluation(test_frame)
    inference_seconds = time.time() - evaluation_started
    metrics = evaluation["test_metrics"]
    print(
        f"test ({evaluation['test_period']['first']}..{evaluation['test_period']['last']}): "
        f"MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} "
        f"deviance={metrics['poisson_deviance']:.3f} bias={metrics['bias']:+.3f}"
    )

    predictions.to_parquet(results_dir / "predictions.parquet", index=False)
    pd.DataFrame(evaluation["monthly"]).to_csv(results_dir / "monthly_metrics.csv", index=False)
    pd.DataFrame(evaluation["calibration"]).to_csv(results_dir / "calibration.csv", index=False)

    importance = gain_importance(trained.booster)
    importance.to_csv(results_dir / "feature_importance.csv", index=False)

    shap_source = (
        bundle.frame.loc[bundle.frame["split"] == "test"].reset_index(drop=True)
    )
    if len(shap_source) != len(test_frame):  # pragma: no cover - alignment guard
        raise RuntimeError("test frames are not aligned for SHAP")
    shap_table, shap_features = compute_shap_values(
        trained.booster,
        shap_source,
        bundle.feature_columns,
        max_rows=20_000,
        seed=int(config["inference"]["seed"]),
    )
    shap_output = pd.concat(
        [
            test_frame.loc[shap_table.index, ["location_id", "month", "site_type", "camera_uid", "y_all", "predicted"]]
            .reset_index(drop=True),
            shap_table.reset_index(drop=True),
        ],
        axis=1,
    )
    shap_output.to_parquet(results_dir / "shap_values.parquet", index=False)

    plot_feature_importance(importance, figures_dir)
    plot_shap_summary(
        shap_table[bundle.feature_columns].to_numpy(),
        shap_features[bundle.feature_columns].reset_index(drop=True),
        figures_dir,
    )
    top_row = int(np.argmax(test_frame["predicted"].to_numpy()))
    # Locate the same row inside the SHAP sample by index label.
    label = test_frame.index[top_row]
    if label in shap_table.index:
        position = int(np.where(shap_table.index == label)[0][0])
    else:  # pragma: no cover - sample may exclude the row
        position = int(np.argmax(test_frame.loc[shap_table.index, "predicted"].to_numpy()))
    plot_shap_waterfall(
        shap_table[bundle.feature_columns].to_numpy(),
        shap_features[bundle.feature_columns].reset_index(drop=True),
        float(shap_table["base_value"].iloc[0]),
        position,
        figures_dir,
    )
    plot_actual_vs_predicted(
        test_frame["y_all"].to_numpy(), test_frame["predicted"].to_numpy(), figures_dir
    )
    plot_monthly_error(pd.DataFrame(evaluation["monthly"]), figures_dir)
    plot_residuals(test_frame["y_all"].to_numpy(), test_frame["predicted"].to_numpy(), figures_dir)
    plot_calibration(pd.DataFrame(evaluation["calibration"]), figures_dir)
    plot_prediction_map(predictions.loc[predictions["split"] == "test"], figures_dir)

    cameras = bundle.study_data.cameras[
        ["camera_uid", "address", "go_live", "go_month", "lat", "lon"]
    ].copy()
    cameras.to_parquet(results_dir / "camera_metadata.parquet", index=False)

    crash_grid = build_crash_grid(bundle.study_data)
    crash_grid.to_parquet(results_dir / "crash_grid_monthly.parquet", index=False)

    top_features = importance.head(15).to_dict(orient="records")
    metadata = model_metadata(
        bundle,
        trained,
        tuning,
        extra={
            "test_metrics": metrics,
            "test_period": evaluation["test_period"],
            "feature_importance_top_15": top_features,
            "inference_seconds_for_test": round(inference_seconds, 3),
            "artifacts": {
                "predictions": "results/ml/predictions.parquet",
                "shap_values": "results/ml/shap_values.parquet",
                "metrics": "results/ml/metrics.json",
                "figures": "results/ml/figures/",
                "crash_grid": "results/ml/crash_grid_monthly.parquet",
            },
        },
    )
    write_metadata(metadata, models_dir / "model_metadata.json")

    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset": {
            "rows": int(len(bundle.frame)),
            "train_rows": int((bundle.frame["split"] == "train").sum()),
            "test_rows": int((bundle.frame["split"] == "test").sum()),
            "locations": bundle.metadata["n_locations"],
            "camera_locations": bundle.metadata["n_camera_locations"],
            "control_locations": bundle.metadata["n_control_locations"],
            "features": bundle.feature_columns,
            "target_window": bundle.metadata["target_window"],
            "test_start": bundle.metadata["test_start"],
            "weather_available": bool(bundle.metadata["weather"].get("available", False)),
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
            "n_trials": getattr(tuning, "n_trials", None),
            "best_value": getattr(tuning, "best_value", None),
            "runtime_seconds": getattr(tuning, "runtime_seconds", None),
            "best_iterations": getattr(tuning, "best_iterations", None),
        },
        "test": evaluation,
        "top_features": top_features,
        "performance": {
            "training_seconds": trained.training_seconds,
            "inference_seconds_for_test": inference_seconds,
            "peak_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024**2 if platform.system() == "Darwin" else 1024),
                1,
            ),
            "rows_scored": int(len(predictions)),
        },
        "total_runtime_seconds": time.time() - started,
        "model_file": str(model_path.relative_to(ROOT)),
    }
    (results_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"artefacts written to {results_dir} and {models_dir}")

    # A compact console summary used by the final report.
    print("\n=== summary ===")
    print(f"training observations : {trained.train_rows:,}")
    print(f"test observations     : {evaluation['test_period']['rows']:,}")
    print(f"unique locations      : {bundle.metadata['n_locations']:,}")
    print(f"train period          : {payload['training']['period']['first']} .. {payload['training']['period']['last']}")
    print(f"test period           : {evaluation['test_period']['first']} .. {evaluation['test_period']['last']}")
    print(f"mae / rmse / deviance : {metrics['mae']:.3f} / {metrics['rmse']:.3f} / {metrics['poisson_deviance']:.3f}")
    print(f"bias                  : {metrics['bias']:+.3f} ({metrics['relative_bias_pct']:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
