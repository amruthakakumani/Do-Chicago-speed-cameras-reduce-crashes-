"""Coverage-consistent model experiments and validation.

The functions in this module reuse the existing LightGBM family and evaluation
code.  They deliberately keep hyperparameters fixed during feature ablation so
that differences are attributable to feature groups, not tuning noise.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .conformal import apply_conformal, select_method
from .dataset import FEATURE_COLUMNS as V1_FEATURES
from .evaluate import (
    calibration_table,
    monthly_metrics,
    regression_metrics,
    subgroup_errors,
)
from .validation import fold_frames, rolling_origin_folds, temporal_split

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "results/bottom_up/coverage_model"


def load_enriched() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    frame = pd.read_parquet(PROCESSED / "ml_coverage_enriched.parquet")
    frame["month"] = pd.to_datetime(frame["month"])
    groups = json.loads((PROCESSED / "ml_coverage_feature_groups.json").read_text())
    return frame, groups


def ablation_feature_sets(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """Cumulative groups, with V1 as the historical comparison set."""
    minimal = groups["calendar"] + groups["history"] + groups["geography"]
    camera = minimal + groups["camera"]
    weather = camera + groups["weather"]
    road = weather + groups["road"]
    neighborhood = road + groups["neighborhood"]
    citywide = neighborhood + groups["citywide"]
    full = citywide + groups["long_history"]
    return {
        "M0_minimal": minimal,
        "M1_camera": camera,
        "M2_weather": weather,
        "M3_road": road,
        "M4_neighborhood": neighborhood,
        "M5_citywide": citywide,
        "M6_full": full,
        "V1_historical_features": list(V1_FEATURES),
    }


def fixed_params(tuned: dict[str, Any], seed: int = 20261024, threads: int = 8) -> dict[str, Any]:
    params = {
        "objective": "poisson",
        "metric": "poisson",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": seed,
        "bagging_seed": seed + 1,
        "feature_fraction_seed": seed + 2,
        "data_random_seed": seed + 3,
        "num_threads": threads,
        "deterministic": True,
        "force_col_wise": True,
        "min_data_in_bin": 5,
    }
    params.update(
        {
            key: tuned[key]
            for key in (
                "learning_rate",
                "num_leaves",
                "max_depth",
                "min_child_samples",
                "feature_fraction",
                "bagging_fraction",
                "bagging_freq",
                "lambda_l1",
                "lambda_l2",
            )
            if key in tuned
        }
    )
    return params


def _fit(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
    max_rounds: int = 2500,
    early_stopping: int = 80,
) -> tuple[lgb.Booster, pd.DataFrame]:
    categorical = ["cluster"] if "cluster" in features else []
    dtrain = lgb.Dataset(
        train[features], label=train["y_all"], categorical_feature=categorical, free_raw_data=True
    )
    dvalid = lgb.Dataset(
        validation[features],
        label=validation["y_all"],
        categorical_feature=categorical,
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
        validation[features], num_iteration=int(model.best_iteration or max_rounds)
    )
    out = validation[["location_id", "month", "y_all", "site_type", "cluster"]].copy()
    out["predicted"] = prediction
    return model, out


def train_evaluate(
    frame: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
    name: str,
    target_start: str = "2018-01-01",
    test_start: str = "2025-09-01",
    max_rounds: int = 2500,
    early_stopping: int = 80,
) -> dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    train_all, test = temporal_split(frame, test_start)
    folds = rolling_origin_folds(target_start, test_start, n_folds=3, val_months=6)
    oof: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    best_iterations: list[int] = []
    started = time.time()
    for fold, fold_train, fold_validation in fold_frames(folds, frame):
        model, prediction = _fit(
            fold_train, fold_validation, features, params, max_rounds, early_stopping
        )
        best_iterations.append(int(model.best_iteration or max_rounds))
        metrics = regression_metrics(prediction["y_all"], prediction["predicted"])
        metrics.update({"fold": fold.name})
        fold_metrics.append(metrics)
        oof.append(prediction)
    calibration = pd.concat(oof, ignore_index=True)
    rounds = max(50, int(np.mean(best_iterations) * 1.10))
    categorical = ["cluster"] if "cluster" in features else []
    dtrain = lgb.Dataset(
        train_all[features], label=train_all["y_all"], categorical_feature=categorical, free_raw_data=True
    )
    final = lgb.train(params, dtrain, num_boost_round=rounds)
    test = test.copy()
    test["predicted"] = final.predict(test[features])
    train_all = train_all.copy()
    train_all["predicted"] = final.predict(train_all[features])

    oof_metrics = regression_metrics(calibration["y_all"], calibration["predicted"])
    test_metrics = regression_metrics(test["y_all"], test["predicted"])
    calibration_table_frame = calibration_table(test["y_all"], test["predicted"])
    subgroups = subgroup_errors(test["y_all"], test["predicted"])
    threshold = float(np.quantile(train_all["y_all"], 0.90))
    actual_high = test["y_all"] >= threshold
    k = max(1, round(0.10 * len(test)))
    top = test.nlargest(k, "predicted")
    top_high = top["y_all"] >= threshold
    hotspot = {
        "actual_high_threshold_from_training_y": threshold,
        "top_fraction": 0.10,
        "k": k,
        "precision": float(top_high.mean()),
        "recall": float(top_high.sum() / max(actual_high.sum(), 1)),
        "lift": float(top_high.mean() / max(actual_high.mean(), 1e-12)),
        "n_actual_high": int(actual_high.sum()),
    }
    # Rank-based overlap uses the test outcome only for post-hoc evaluation.
    top_actual = test.nlargest(k, "y_all")
    hotspot["top_decile_rank_overlap"] = float(
        len(set(top.index) & set(top_actual.index)) / k
    )

    method = select_method(calibration, alpha=0.1)
    interval, interval_metrics = apply_conformal(
        calibration, test, method["selected"], alpha=0.1  # type: ignore[arg-type]
    )
    test["lower_90"] = interval.lower
    test["upper_90"] = interval.upper
    test["interval_method"] = method["selected"]

    monthly = monthly_metrics(test, "predicted")
    feature_importance = pd.DataFrame(
        {
            "feature": features,
            "gain": final.feature_importance(importance_type="gain"),
            "split": final.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    feature_importance["gain_share_pct"] = (
        100 * feature_importance["gain"] / max(feature_importance["gain"].sum(), 1)
    )
    model_path = ROOT / "models/bottom_up" / f"{name}.txt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(str(model_path))
    feature_importance.to_csv(RESULTS / f"{name}_feature_importance.csv", index=False)
    test.to_parquet(RESULTS / f"{name}_test_predictions.parquet", index=False)
    calibration.to_parquet(RESULTS / f"{name}_oof_predictions.parquet", index=False)
    metrics = {
        "name": name,
        "n_features": len(features),
        "features": features,
        "params": params,
        "rounds": int(rounds),
        "best_iterations": best_iterations,
        "training_seconds": round(time.time() - started, 2),
        "train_rows": len(train_all),
        "test_rows": len(test),
        "train_period": {
            "first": str(train_all["month"].min().date()),
            "last": str(train_all["month"].max().date()),
        },
        "test_period": {
            "first": str(test["month"].min().date()),
            "last": str(test["month"].max().date()),
        },
        "validation": {
            "fold_metrics": fold_metrics,
            "mean_deviance": float(np.mean([row["poisson_deviance"] for row in fold_metrics])),
            "mean_mae": float(np.mean([row["mae"] for row in fold_metrics])),
            "oof_metrics": oof_metrics,
        },
        "test": test_metrics,
        "calibration": calibration_table_frame.to_dict(orient="records"),
        "subgroup_errors": subgroups,
        "hotspot": hotspot,
        "interval_90": interval_metrics,
        "interval_method": method,
        "monthly": monthly.to_dict(orient="records"),
        "feature_importance_top": feature_importance.head(20).to_dict(orient="records"),
        "model_file": str(model_path.relative_to(ROOT)),
    }
    (RESULTS / f"{name}_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n"
    )
    return metrics


def main() -> int:
    frame, groups = load_enriched()
    v1_params = json.loads((ROOT / "results/ml/best_params.json").read_text())["params"]
    params = fixed_params(v1_params)
    feature_sets = ablation_feature_sets(groups)
    # Run V1 first; the remaining groups are separate experiments.
    train_evaluate(frame, feature_sets["V1_historical_features"], params, "coverage_v1")
    train_evaluate(frame, feature_sets["M6_full"], params, "coverage_full")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
