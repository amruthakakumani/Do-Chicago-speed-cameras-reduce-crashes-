"""Tests for the hybrid forecasting experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from chicago_cameras.ml.statistical_experiment import (
    NB_FEATURES,
    _candidate_metrics,
    _hybrid_weight,
    _nb_design,
)


def _synthetic_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({feature: rng.normal(size=n) for feature in NB_FEATURES})
    frame["cluster"] = rng.integers(0, 4, size=n)
    frame["expanding_mean"] = rng.uniform(1, 5, size=n)
    frame["y_all"] = rng.poisson(3, size=n)
    frame["site_type"] = "control"
    frame.loc[frame.index[:10], "site_type"] = "camera"
    frame["month"] = pd.date_range("2024-01-01", periods=n, freq="MS")
    return frame


def test_hybrid_weight_is_bounded_and_nonnegative():
    rng = np.random.default_rng(11)
    y = rng.poisson(3, size=1000)
    calibration = pd.DataFrame(
        {
            "y_all": y,
            "lgbm": np.clip(y + rng.normal(0, 0.5, size=len(y)), 0, None),
            "nb": np.clip(y + rng.normal(0, 2.0, size=len(y)), 0, None),
        }
    )
    weight, curve = _hybrid_weight(calibration)
    assert 0 <= weight <= 1
    assert len(curve) == 101
    assert (curve["poisson_deviance"] >= 0).all()
    assert (curve["mae"] >= 0).all()


def test_nb_design_is_finite_and_has_cluster_indicators():
    frame = _synthetic_frame()
    frame.loc[frame.index[:3], "lag_1"] = np.nan
    design, medians = _nb_design(frame)
    assert "const" in design.columns
    assert not design.isna().any().any()
    assert any(column.startswith("cluster_") for column in design.columns)
    assert medians.notna().all()


def test_nb_features_exclude_target():
    assert "y_all" not in NB_FEATURES
    assert "y_injury" not in NB_FEATURES
    assert "y_severe" not in NB_FEATURES


def test_candidate_metrics_and_nonnegative_predictions():
    frame = _synthetic_frame(200)
    frame["prediction"] = np.clip(frame["y_all"] + 0.2, 0, None)
    metrics = _candidate_metrics(frame, "prediction", frame["y_all"].to_numpy(float))
    assert metrics["n"] == 200
    assert metrics["mae"] >= 0
    assert metrics["poisson_deviance"] >= 0
    assert 0 <= metrics["hotspot"]["precision"] <= 1
    assert 0 <= metrics["hotspot"]["recall"] <= 1
