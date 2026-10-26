"""Tests for the V2 feature, geographic, conformal and recursive layers."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from chicago_cameras.ml.conformal import (
    apply_conformal,
    evaluate_methods,
    select_method,
)
from chicago_cameras.ml.dataset import DatasetBundle, _add_history_features
from chicago_cameras.ml.dataset_v2 import _add_temporal_features_v2
from chicago_cameras.ml.geographic import geographic_split
from chicago_cameras.ml.predict import load_model
from chicago_cameras.ml.recursive import recursive_forecast
from chicago_cameras.ml.spatial import _build_intersections


def _temporal_frame() -> pd.DataFrame:
    rows = []
    for location in ("A", "B"):
        for index in range(48):
            rows.append(
                {
                    "location_id": location,
                    "month": pd.Timestamp("2018-01-01") + pd.DateOffset(months=index),
                    "month_index": index,
                    "y_all": index + (5 if location == "B" else 0),
                    "y_injury": (index // 3) if location == "A" else 0,
                    "y_severe": 0,
                }
            )
    frame = _add_history_features(pd.DataFrame(rows))
    return _add_temporal_features_v2(frame)


def test_v2_temporal_features_use_prior_months() -> None:
    frame = _temporal_frame()
    location = frame.loc[frame["location_id"] == "A"].sort_values("month").reset_index(drop=True)
    assert location.loc[24, "lag_24"] == location.loc[0, "y_all"]
    assert location.loc[40, "lag_36"] == location.loc[4, "y_all"]
    # Every month has a crash in this fixture, so months-since is always 1.
    assert location.loc[10, "months_since_last_crash"] == 1
    assert location.loc[10, "ever_crashed"] == 1
    expected = np.mean(
        [location.loc[40 - 12, "y_all"], location.loc[40 - 24, "y_all"], location.loc[40 - 36, "y_all"]]
    )
    assert location.loc[40, "same_month_mean_3y"] == pytest.approx(expected)


def test_months_since_last_crash_counts_gaps() -> None:
    rows = []
    values = [1, 0, 0, 2, 0, 0, 0, 1] + [0] * 8
    for index, value in enumerate(values):
        rows.append(
            {
                "location_id": "A",
                "month": pd.Timestamp("2018-01-01") + pd.DateOffset(months=index),
                "month_index": index,
                "y_all": value,
                "y_injury": 0,
                "y_severe": 0,
            }
        )
    frame = _add_temporal_features_v2(_add_history_features(pd.DataFrame(rows)))
    location = frame.sort_values("month").reset_index(drop=True)
    assert location.loc[4, "months_since_last_crash"] == 1  # last crash at t=3
    assert location.loc[6, "months_since_last_crash"] == 3  # last crash at t=3
    assert location.loc[8, "months_since_last_crash"] == 1  # last crash at t=7


def _locations() -> pd.DataFrame:
    records = []
    for x_index in range(10):
        for y_index in range(10):
            records.append(
                {
                    "location_id": f"L{x_index}_{y_index}",
                    "x": x_index * 1000.0,
                    "y": y_index * 1000.0,
                    "lat": 41.8,
                    "lon": -87.6,
                    "site_type": "camera" if (x_index + y_index) % 7 == 0 else "control",
                    "camera_uid": f"C{x_index}_{y_index}",
                    "cluster": (x_index + y_index) % 5,
                }
            )
    return pd.DataFrame(records)


def test_geographic_split_isolates_blocks_and_buffers() -> None:
    locations = _locations()
    split = geographic_split(locations, holdout_fraction=0.2, buffer_m=1000.0, seed=1)
    train_ids = set(split.train["location_id"])
    holdout_ids = set(split.holdout["location_id"])
    assert train_ids and holdout_ids
    assert not train_ids.intersection(holdout_ids)
    assert split.info["min_train_to_holdout_distance_m"] > 1000.0
    assert not set(split.train["block_id"]).intersection(set(split.holdout["block_id"]))


def test_conformal_prefers_scale_adaptive_intervals() -> None:
    rng = np.random.default_rng(0)
    months = pd.date_range("2020-01-01", periods=18, freq="MS")
    calibration = pd.DataFrame(
        {
            "month": np.repeat(months, 20),
            "y_all": rng.poisson(4, 18 * 20),
            "predicted": rng.uniform(1, 10, 18 * 20),
            "site_type": ["camera"] * (18 * 20),
            "cluster": [0] * (18 * 20),
        }
    )
    test = calibration.copy()
    selection = select_method(calibration, alpha=0.1)
    assert selection["selected"] == "normalized"
    interval, metrics = apply_conformal(calibration, test, "normalized", 0.1)
    assert (interval.lower >= 0).all()
    assert (interval.upper >= interval.lower).all()
    assert 0.7 <= metrics["empirical_coverage"] <= 1.0
    both = evaluate_methods(calibration, test)
    assert {"normalized", "absolute"} <= set(both["methods"])


def test_tiger_intersection_nodes() -> None:
    from shapely.geometry import LineString

    lines = [
        LineString([(0, 0), (10, 0)]),
        LineString([(0, 0), (0, 10)]),
        LineString([(0, 0), (-10, 0)]),
    ]
    points, major = _build_intersections(
        lines, ["S1200", "S1400", "S1400"], [(1, 2), (1, 3), (1, 4)]
    )
    assert len(points) == 1
    assert bool(major[0])


def _recursive_bundle(perturb_future: bool = False) -> DatasetBundle:
    rng = np.random.default_rng(3)
    months = pd.date_range("2020-01-01", "2022-06-01", freq="MS")
    locations = _locations().head(12).reset_index(drop=True)
    features = [
        "lag_1",
        "roll_mean_3",
        "expanding_mean",
        "temp_lag1",
        "temp_normal",
        "precip_normal",
        "snow_normal",
        "cluster",
    ]
    rows = []
    for location in locations.itertuples():
        for index, month in enumerate(months):
            value = int(rng.poisson(3))
            if perturb_future and month > pd.Timestamp("2021-12-01"):
                value *= 10
            rows.append(
                {
                    "location_id": location.location_id,
                    "month": month,
                    "month_index": index,
                    "y_all": value,
                    "y_injury": 0,
                    "y_severe": 0,
                    "lag_1": value,
                    "roll_mean_3": value,
                    "expanding_mean": value,
                    "temp_lag1": 10.0,
                    "temp_normal": 10.0,
                    "precip_normal": 1.0,
                    "snow_normal": 0.0,
                    "cluster": location.cluster,
                }
            )
    frame = pd.DataFrame(rows)
    return DatasetBundle(
        frame=frame,
        locations=locations,
        feature_columns=features,
        categorical_features=[],
        metadata={"test_start": "2022-01-01"},
    )


def test_recursive_forecast_does_not_use_future_actuals() -> None:
    bundle = _recursive_bundle()
    features = bundle.feature_columns
    train = bundle.frame.loc[bundle.frame["month"] <= pd.Timestamp("2021-12-01")]
    model = lgb.train(
        {"objective": "poisson", "verbosity": -1, "min_data_in_leaf": 2},
        lgb.Dataset(train[features], label=train["y_all"]),
        num_boost_round=30,
    )
    result = recursive_forecast(bundle, model, features, "2021-12-01", horizon=5)
    assert len(result.predictions) == 12 * 5
    assert (result.predictions["predicted"] >= 0).all()

    perturbed = _recursive_bundle(perturb_future=True)
    perturbed_result = recursive_forecast(perturbed, model, features, "2021-12-01", horizon=5)
    assert np.allclose(
        result.predictions["predicted"].to_numpy(),
        perturbed_result.predictions["predicted"].to_numpy(),
    )


def test_v2_model_metadata_matches_saved_model() -> None:
    root = Path(__file__).resolve().parents[1]
    metadata_path = root / "models" / "v2_model_metadata.json"
    model_path = root / "models" / "v2_lightgbm_crash_model.txt"
    if not metadata_path.exists() or not model_path.exists():
        pytest.skip("V2 model artefacts not generated in this checkout")
    metadata = json.loads(metadata_path.read_text())
    booster = load_model(model_path)
    assert metadata["feature_schema"] == booster.feature_name()
    assert metadata["model_version"] == "2.0.0"


def test_dashboard_selected_artifacts_are_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    deployment_path = root / "results" / "ml_v2" / "deployment.json"
    if not deployment_path.exists():
        pytest.skip("V2 deployment artefacts not generated in this checkout")
    deployment = json.loads(deployment_path.read_text())
    selected_path = root / deployment["test_predictions_path"]
    assert selected_path.exists()
    selected = pd.read_parquet(selected_path)
    assert {"lower_90", "upper_90", "predicted", "y_all"} <= set(selected.columns)
    assert (selected["lower_90"] >= 0).all()
    assert (selected["upper_90"] >= selected["lower_90"]).all()
    assert ((selected["predicted"] >= selected["lower_90"]) & (selected["predicted"] <= selected["upper_90"])).all()
    observed = ((selected["y_all"] >= selected["lower_90"]) & (selected["y_all"] <= selected["upper_90"])).mean()
    assert abs(observed - deployment["interval_coverage_90"]) < 0.01
    assert deployment["selected_model"] in {"v1", "v2"}
    assert (root / deployment["model_file"]).exists()
