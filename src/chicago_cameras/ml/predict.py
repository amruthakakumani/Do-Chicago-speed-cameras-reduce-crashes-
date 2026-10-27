"""Load the saved model and generate reproducible predictions."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .dataset import DatasetBundle


def load_model(path: str | Path) -> lgb.Booster:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    booster = lgb.Booster(model_file=str(model_path))
    if booster.num_feature() == 0:  # pragma: no cover - defensive
        raise ValueError(f"model at {model_path} has no features")
    return booster


def validate_feature_schema(
    frame: pd.DataFrame, features: Iterable[str], booster: lgb.Booster
) -> None:
    features = list(features)
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"missing model features: {missing}")
    model_features = booster.feature_name()
    if list(model_features) != features:
        raise ValueError(
            "feature schema does not match the trained model; "
            f"expected {model_features}, got {features}"
        )


def predict_counts(
    booster: lgb.Booster,
    frame: pd.DataFrame,
    features: list[str],
    clip_min: float = 0.0,
) -> np.ndarray:
    validate_feature_schema(frame, features, booster)
    prediction = booster.predict(frame[features])
    prediction = np.asarray(prediction, dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("model produced non-finite predictions")
    return np.clip(prediction, clip_min, None)


def attach_predictions(
    bundle: DatasetBundle,
    booster: lgb.Booster,
    frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return location-month rows with a ``predicted`` column and split label."""
    source = (frame if frame is not None else bundle.frame).copy()
    source["predicted"] = predict_counts(booster, source, bundle.feature_columns)
    kept = [
        "location_id",
        "month",
        "y_all",
        "y_injury",
        "y_severe",
        "predicted",
        "split",
        "site_type",
        "camera_uid",
        "lat",
        "lon",
        "easting_m",
        "northing_m",
        "cluster",
        "lag_12",
        "roll_mean_3",
        "expanding_mean",
    ]
    return source[[column for column in kept if column in source.columns]].reset_index(drop=True)
