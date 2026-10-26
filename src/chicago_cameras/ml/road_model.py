"""Separate road-segment forecasting experiment.

The road-segment target is not the camera-zone target.  It covers a fixed
sample of road segments (all major segments plus a seeded random sample of local
segments), and many segment-months are legitimate zeros.  Its raw MAE is
therefore not comparable to the 250 m camera-zone task.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .conformal import coverage_by_group, coverage_by_risk_decile, interval_metrics as interval_metrics_fn, select_method, build_intervals, conformal_quantile, nonconformity
from .evaluate import calibration_table, monthly_metrics, regression_metrics, subgroup_errors
from .reconstruction import fixed_params
from .validation import fold_frames, rolling_origin_folds, temporal_split

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "results/bottom_up/road_model"

FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "roll_mean_3",
    "roll_std_3",
    "roll_mean_6",
    "roll_std_6",
    "roll_mean_12",
    "roll_std_12",
    "expanding_mean",
    "zero_share_12",
    "month_index",
    "month_of_year",
    "month_sin",
    "month_cos",
    "days_in_month",
    "is_covid",
    "length_m",
    "major_road",
    "road_class",
    "from_degree",
    "to_degree",
    "node_degree_max",
    "is_intersection_edge",
    "centroid_x",
    "centroid_y",
]


def _fit(train: pd.DataFrame, validation: pd.DataFrame, params: dict[str, Any], name: str):
    dtrain = lgb.Dataset(
        train[FEATURES], label=train["y_all"], categorical_feature=["road_class"], free_raw_data=True
    )
    dvalid = lgb.Dataset(
        validation[FEATURES],
        label=validation["y_all"],
        categorical_feature=["road_class"],
        reference=dtrain,
        free_raw_data=True,
    )
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2200,
        valid_sets=[dvalid],
        valid_names=["validation"],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    pred = model.predict(validation[FEATURES], num_iteration=int(model.best_iteration or 2200))
    out = validation[["segment_id", "month", "y_all", "major_road", "road_class"]].copy()
    out["predicted"] = pred
    return model, out


def train_road_model() -> dict[str, Any]:
    frame = pd.read_parquet(PROCESSED / "road_model_panel_features.parquet")
    frame["month"] = pd.to_datetime(frame["month"])
    frame = frame.loc[frame["month"] >= "2018-01-01"].copy()
    frame["road_class"] = frame["road_class"].astype("category")
    train_all, test = temporal_split(frame, "2025-09-01")
    folds = rolling_origin_folds("2018-01-01", "2025-09-01", n_folds=3, val_months=6)
    params = fixed_params(json.loads((ROOT / "results/ml/best_params.json").read_text())["params"])
    params["num_threads"] = 8
    started = time.time()
    oof: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    iterations: list[int] = []
    for fold, fold_train, fold_validation in fold_frames(folds, frame):
        model, prediction = _fit(fold_train, fold_validation, params, fold.name)
        iterations.append(int(model.best_iteration or 2200))
        metrics = regression_metrics(prediction["y_all"], prediction["predicted"])
        metrics["fold"] = fold.name
        fold_metrics.append(metrics)
        oof.append(prediction)
        print(fold.name, metrics["poisson_deviance"], flush=True)
    calib = pd.concat(oof, ignore_index=True)
    rounds = max(50, int(np.mean(iterations) * 1.1))
    dtrain = lgb.Dataset(
        train_all[FEATURES], label=train_all["y_all"], categorical_feature=["road_class"], free_raw_data=True
    )
    final = lgb.train(params, dtrain, num_boost_round=rounds)
    test = test.copy()
    test["predicted"] = final.predict(test[FEATURES])
    train_all = train_all.copy()
    train_all["predicted"] = final.predict(train_all[FEATURES])
    test_metrics = regression_metrics(test["y_all"], test["predicted"])
    threshold = float(np.quantile(train_all["y_all"], 0.90))
    high = test["y_all"] >= threshold
    k = max(1, int(round(0.10 * len(test))))
    top = test.nlargest(k, "predicted")
    hotspot = {
        "actual_high_threshold_from_training_y": threshold,
        "k": k,
        "precision": float((top["y_all"] >= threshold).mean()),
        "recall": float((top["y_all"] >= threshold).sum() / max(high.sum(), 1)),
        "lift": float((top["y_all"] >= threshold).mean() / max(high.mean(), 1e-12)),
        "n_actual_high": int(high.sum()),
    }
    method = select_method(calib, alpha=0.1)
    scores = nonconformity(calib["y_all"].to_numpy(), calib["predicted"].to_numpy(), method["selected"])
    quantile = conformal_quantile(scores, 0.1)
    interval = build_intervals(test["predicted"].to_numpy(), quantile, method["selected"], 0.1)
    interval_metrics = interval_metrics_fn(test["y_all"].to_numpy(), interval)
    interval_metrics["by_risk_decile"] = coverage_by_risk_decile(test, interval).to_dict(orient="records")
    interval_metrics["by_major_road"] = coverage_by_group(test, interval, "major_road").to_dict(orient="records")
    test["lower_90"] = interval.lower
    test["upper_90"] = interval.upper
    model_path = ROOT / "models/bottom_up/road_segment_lgbm.txt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(str(model_path))
    RESULTS.mkdir(parents=True, exist_ok=True)
    test.to_parquet(RESULTS / "road_test_predictions.parquet", index=False)
    calib.to_parquet(RESULTS / "road_oof_predictions.parquet", index=False)
    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "gain": final.feature_importance(importance_type="gain"),
            "split": final.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(RESULTS / "road_feature_importance.csv", index=False)
    metrics = {
        "target": "Crashes assigned to a road segment within 30 m",
        "n_features": len(FEATURES),
        "features": FEATURES,
        "rounds": int(rounds),
        "best_iterations": iterations,
        "training_seconds": round(time.time() - started, 2),
        "train_rows": int(len(train_all)),
        "test_rows": int(len(test)),
        "validation": {
            "fold_metrics": fold_metrics,
            "mean_deviance": float(np.mean([row["poisson_deviance"] for row in fold_metrics])),
            "mean_mae": float(np.mean([row["mae"] for row in fold_metrics])),
        },
        "test": test_metrics,
        "calibration": calibration_table(test["y_all"], test["predicted"]).to_dict(orient="records"),
        "subgroup_errors": subgroup_errors(test["y_all"], test["predicted"]),
        "hotspot": hotspot,
        "interval_90": interval_metrics,
        "interval_method": method,
        "monthly": monthly_metrics(test).to_dict(orient="records"),
        "feature_importance_top": importance.head(20).to_dict(orient="records"),
        "model_file": str(model_path.relative_to(ROOT)),
        "comparison_warning": (
            "Road-segment MAE is for a different unit and population than the 250 m "
            "camera-zone task; do not compare raw MAE across tasks."
        ),
    }
    (RESULTS / "road_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n"
    )
    return metrics


if __name__ == "__main__":
    result = train_road_model()
    print(json.dumps({k: result[k] for k in ("test", "validation", "hotspot", "interval_90")}, indent=2, default=str))
