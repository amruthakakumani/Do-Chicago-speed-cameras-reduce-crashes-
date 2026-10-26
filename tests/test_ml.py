"""Tests for the LightGBM prediction layer.

All fixtures are synthetic and tiny; none of these tests touch the real Chicago
data or the network.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from chicago_cameras.ml.dataset import DatasetBundle, _add_history_features, validate_no_leakage
from chicago_cameras.ml.evaluate import (
    calibration_table,
    poisson_deviance,
    regression_metrics,
)
from chicago_cameras.ml.predict import (
    attach_predictions,
    load_model,
    predict_counts,
    validate_feature_schema,
)
from chicago_cameras.ml.train import save_model, train_final
from chicago_cameras.ml.validation import rolling_origin_folds, temporal_split


def _history_frame() -> pd.DataFrame:
    rows = []
    for location in ("A", "B"):
        for index in range(30):
            month = pd.Timestamp("2018-01-01") + pd.DateOffset(months=index)
            value = index + (10 if location == "B" else 0)
            rows.append(
                {
                    "location_id": location,
                    "month": month,
                    "y_all": value,
                    "y_injury": value // 3,
                    "y_severe": value // 6,
                }
            )
    return pd.DataFrame(rows)


def test_lag_features_use_only_prior_months() -> None:
    frame = _add_history_features(_history_frame())
    location_a = frame.loc[frame["location_id"] == "A"].sort_values("month").reset_index(drop=True)
    assert np.isnan(location_a.loc[0, "lag_1"])
    assert location_a.loc[5, "lag_1"] == location_a.loc[4, "y_all"]
    assert location_a.loc[5, "lag_3"] == location_a.loc[2, "y_all"]
    expected = location_a.loc[2:4, "y_all"].mean()
    assert location_a.loc[5, "roll_mean_3"] == pytest.approx(expected)


def test_rolling_features_are_unchanged_by_future_outcomes() -> None:
    frame = _add_history_features(_history_frame())
    features = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "roll_mean_3"]
    result = validate_no_leakage(frame, features)
    assert result["ok"], result["problems"]


def test_leakage_check_detects_a_future_feature() -> None:
    frame = _add_history_features(_history_frame())
    frame["roll_mean_3"] = frame["y_all"]  # deliberately leaky
    result = validate_no_leakage(frame, ["roll_mean_3"])
    assert not result["ok"]


def test_temporal_split_never_overlaps() -> None:
    frame = _history_frame()
    frame["split"] = np.where(frame["month"] < "2019-01-01", "train", "test")
    train, test = temporal_split(frame, "2019-01-01")
    assert train["month"].max() < test["month"].min()


def test_rolling_origin_folds_are_chronological() -> None:
    folds = rolling_origin_folds("2017-01-01", "2025-09-01", n_folds=3, val_months=6)
    assert len(folds) == 3
    for fold in folds:
        assert fold.train_start < fold.train_end < fold.val_start <= fold.val_end
    assert folds[-1].val_end == pd.Timestamp("2025-08-01")
    assert all(fold.val_end < pd.Timestamp("2025-09-01") for fold in folds)


def test_poisson_deviance_is_zero_for_perfect_fit() -> None:
    y = np.array([0, 1, 4, 9], dtype=float)
    assert poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-8)


def test_regression_metrics_and_calibration_schema() -> None:
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    pred = y + 0.5
    metrics = regression_metrics(y, pred)
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["bias"] == pytest.approx(0.5)
    table = calibration_table(y, pred, n_bins=5)
    assert {"predicted_mean", "actual_mean", "n", "ratio_actual_to_predicted"} <= set(table.columns)


@pytest.fixture
def bundle() -> DatasetBundle:
    rng = np.random.default_rng(7)
    rows = []
    for location in range(20):
        for index in range(24):
            month = pd.Timestamp("2020-01-01") + pd.DateOffset(months=index)
            f1 = rng.uniform(0, 5)
            f2 = rng.uniform(0, 2)
            rows.append(
                {
                    "location_id": f"L{location}",
                    "month": month,
                    "y_all": int(rng.poisson(1 + f1)),
                    "y_injury": 0,
                    "y_severe": 0,
                    "f1": f1,
                    "f2": f2,
                    "split": "train" if month < pd.Timestamp("2021-09-01") else "test",
                    "site_type": "camera" if location % 2 else "control",
                    "camera_uid": f"C{location}",
                    "lat": 41.8,
                    "lon": -87.6,
                    "x": 0.0,
                    "y": 0.0,
                    "cluster": location % 3,
                    "lag_12": None,
                    "roll_mean_3": None,
                    "expanding_mean": None,
                }
            )
    frame = pd.DataFrame(rows)
    return DatasetBundle(
        frame=frame,
        locations=pd.DataFrame({"location_id": frame["location_id"].unique()}),
        feature_columns=["f1", "f2"],
        categorical_features=[],
        metadata={"test_start": "2021-09-01", "target_window": {"first": "2020-01-01"}},
    )


def test_model_serialization_and_reproducible_predictions(bundle: DatasetBundle, tmp_path) -> None:
    trained = train_final(
        bundle,
        {"learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 5},
        [25],
        seed=1,
        num_threads=1,
        num_boost_round=25,
    )
    path = tmp_path / "model.txt"
    save_model(trained, path)
    loaded = load_model(path)
    validate_feature_schema(bundle.frame, bundle.feature_columns, loaded)
    first = predict_counts(loaded, bundle.frame, bundle.feature_columns)
    second = predict_counts(loaded, bundle.frame, bundle.feature_columns)
    assert np.allclose(first, second)
    assert (first >= 0).all()


def test_feature_schema_mismatch_raises(bundle: DatasetBundle, tmp_path) -> None:
    trained = train_final(
        bundle,
        {"learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 5},
        [10],
        seed=1,
        num_threads=1,
        num_boost_round=10,
    )
    path = tmp_path / "model.txt"
    save_model(trained, path)
    loaded = load_model(path)
    with pytest.raises(ValueError):
        validate_feature_schema(bundle.frame, ["f1"], loaded)


def test_attach_predictions_keeps_nonnegative_counts(bundle: DatasetBundle) -> None:
    trained = train_final(
        bundle,
        {"learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 5},
        [15],
        seed=1,
        num_threads=1,
        num_boost_round=15,
    )
    predictions = attach_predictions(bundle, trained.booster)
    assert {"location_id", "month", "predicted", "split"} <= set(predictions.columns)
    assert (predictions["predicted"] >= 0).all()
    assert len(predictions) == len(bundle.frame)


def test_crash_grid_geographic_aggregation() -> None:
    from chicago_cameras.design import StudyData
    from chicago_cameras.ml.dataset import build_crash_grid

    months = pd.date_range("2019-01-01", "2019-03-01", freq="MS")
    cameras = pd.DataFrame(
        {
            "camera_uid": ["C1"],
            "x": [1000.0],
            "y": [1000.0],
            "go_month": [pd.Timestamp("2019-01-01")],
        }
    )
    crashes = pd.DataFrame(
        {
            "crash_record_id": [f"X{i}" for i in range(20)],
            "x": np.linspace(900, 1100, 20),
            "y": np.linspace(900, 1100, 20),
            "month": [months[i % len(months)] for i in range(20)],
            "injury_crash": [i % 2 for i in range(20)],
            "severe_crash": [i % 3 == 0 for i in range(20)],
        }
    )
    data = StudyData(cameras, crashes, months)
    grid = build_crash_grid(data, start_month="2019-01-01", grid_m=500, radius_m=250)
    assert {
        "cell_id",
        "year",
        "month",
        "lat",
        "lon",
        "crashes_all",
        "crashes_injury",
        "crashes_severe",
    } <= set(grid.columns)
    assert len(grid) > 0
    assert (grid[["crashes_all", "crashes_injury", "crashes_severe"]] >= 0).all().all()


def test_dashboard_artifacts_are_consistent() -> None:
    """If committed ML artefacts exist, dashboard inputs must agree with metrics."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    predictions_path = root / "results" / "ml" / "predictions.parquet"
    metrics_path = root / "results" / "ml" / "metrics.json"
    if not predictions_path.exists() or not metrics_path.exists():
        pytest.skip("ML artefacts not generated in this checkout")

    predictions = pd.read_parquet(predictions_path)
    metrics = json.loads(metrics_path.read_text())
    assert {"location_id", "month", "predicted", "split", "site_type"} <= set(predictions.columns)
    assert (predictions["predicted"] >= 0).all()
    test = predictions.loc[predictions["split"] == "test"]
    assert len(test) == metrics["test"]["test_period"]["rows"]
    assert test["predicted"].mean() == pytest.approx(
        metrics["test"]["test_metrics"]["mean_predicted"], abs=1e-6
    )
    assert test["y_all"].mean() == pytest.approx(
        metrics["test"]["test_metrics"]["mean_actual"], abs=1e-6
    )

    camera_meta_path = root / "results" / "ml" / "camera_metadata.parquet"
    if camera_meta_path.exists():
        cameras = pd.read_parquet(camera_meta_path)
        assert cameras["camera_uid"].is_unique
        assert set(test.loc[test["site_type"] == "camera", "camera_uid"]).issubset(
            set(cameras["camera_uid"])
        )
