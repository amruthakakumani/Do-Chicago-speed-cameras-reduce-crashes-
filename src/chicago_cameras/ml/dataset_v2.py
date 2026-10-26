"""V2 location-month dataset: road network, neighbourhood and temporal features.

V2 preserves the V1 construction and adds only features that are available at
the forecast date:

* static TIGER road-network features (density, class, distance, intersections);
* spatially lagged neighbouring crash history (strictly prior months);
* longer history (24/36 months), seasonal ratios, months since last crash;
* expanding same-month weather normals (prior years only).

No traffic-volume feature is used: the public AADT snapshot is current rather
than historical and would leak future exposure into 2017–2025 training months.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from ..config import load_config, path_for
from ..pipeline import prepare_data
from .dataset import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS as V1_FEATURE_COLUMNS,
    MLDatasetConfig,
    DatasetBundle,
    add_features,
    assert_valid_features,
    build_counts,
    build_locations,
)
from .external import (
    fetch_daily_weather,
    monthly_holidays,
    monthly_weather,
)
from .spatial import ROAD_FEATURES, build_road_features

LOGGER = logging.getLogger(__name__)

NEW_TEMPORAL_FEATURES = [
    "lag_24",
    "lag_36",
    "same_month_mean_3y",
    "seasonal_ratio",
    "cv_12",
    "trend_6",
    "months_since_last_crash",
    "months_since_last_injury",
    "ever_crashed",
    "temp_normal",
    "precip_normal",
    "snow_normal",
]
NEW_CITYWIDE_FEATURES = [
    "city_crash_lag1",
    "city_crash_mean_3",
    "city_crash_mean_12",
    "city_crash_trend",
    "city_injury_share_12",
    "location_city_ratio_12",
    "shock_lag1",
    "shock_lag3",
    "city_shock_lag1",
]
NEW_NEIGHBOR_FEATURES = [
    "neighbor_crash_lag1",
    "neighbor_crash_mean_12",
    "neighbor_crash_std_12",
    "neighbor_zero_share_12",
    "neighbor_injury_share_12",
    "neighbor_shock_lag1",
]
FEATURE_COLUMNS_V2 = (
    V1_FEATURE_COLUMNS
    + NEW_TEMPORAL_FEATURES
    + NEW_CITYWIDE_FEATURES
    + ROAD_FEATURES
    + NEW_NEIGHBOR_FEATURES
)


def _add_temporal_features_v2(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["location_id", "month"]).copy()
    group = frame.groupby("location_id", sort=False)

    for lag in (24, 36):
        frame[f"lag_{lag}"] = group["y_all"].shift(lag)
    frame["same_month_mean_3y"] = frame[["lag_12", "lag_24", "lag_36"]].mean(axis=1)
    frame["seasonal_ratio"] = frame["same_month_mean_3y"] / (frame["expanding_mean"] + 0.5)
    frame["cv_12"] = frame["roll_std_12"] / (frame["roll_mean_12"] + 0.5)
    frame["trend_6"] = frame["lag_1"] / (frame["roll_mean_6"] + 0.5)
    frame["shock_lag1"] = frame["lag_1"] - frame["roll_mean_12"]
    frame["shock_lag3"] = frame[["lag_1", "lag_2", "lag_3"]].mean(axis=1) - frame["roll_mean_12"]

    # Months since the last observed crash/injury, using only months before t.
    work = frame[["location_id", "month_index", "y_all", "y_injury"]].copy()
    for outcome, label in (("y_all", "crash"), ("y_injury", "injury")):
        hit = np.where(work[outcome] > 0, work["month_index"], np.nan)
        previous = (
            pd.Series(hit, index=work.index)
            .groupby(work["location_id"], sort=False)
            .ffill()
            .groupby(work["location_id"], sort=False)
            .shift(1)
        )
        frame[f"months_since_last_{label}"] = work["month_index"] - previous
    ever = group["y_all"].transform(lambda series: (series > 0).cummax().shift(1))
    frame["ever_crashed"] = ever.fillna(0).astype(int)
    return frame


def _add_weather_normals(frame: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Expanding same-calendar-month weather normals from prior years only."""
    if weather.empty:
        for column in ("temp_normal", "precip_normal", "snow_normal"):
            frame[column] = np.nan
        return frame
    work = weather.copy()
    work["month"] = pd.to_datetime(work["month"])
    work["month_of_year"] = work["month"].dt.month
    for source, target in (
        ("temp_mean_c", "temp_normal"),
        ("precip_mm", "precip_normal"),
        ("snow_cm", "snow_normal"),
    ):
        work[target] = work.groupby("month_of_year")[source].transform(
            lambda series: series.shift(1).expanding(min_periods=1).mean()
        )
    return frame.merge(
        work[["month", "temp_normal", "precip_normal", "snow_normal"]], on="month", how="left"
    )


def _neighborhood_weights(
    locations: pd.DataFrame, sigma_m: float = 750.0, max_distance_m: float = 3000.0
) -> csr_matrix:
    xy = locations[["x", "y"]].to_numpy(float)
    distance = cdist(xy, xy)
    weights = np.exp(-((distance / sigma_m) ** 2))
    weights[distance > max_distance_m] = 0.0
    np.fill_diagonal(weights, 0.0)
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums > 0)
    return csr_matrix(weights)


def _add_citywide_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Citywide crash regime, lagged so the target month is never used."""
    city = (
        frame.groupby("month", as_index=False)
        .agg(city_crashes=("y_all", "sum"), city_injury=("y_injury", "sum"))
        .sort_values("month")
    )
    city["city_crash_lag1"] = city["city_crashes"].shift(1)
    city["city_crash_mean_3"] = city["city_crashes"].shift(1).rolling(3, min_periods=1).mean()
    city["city_crash_mean_12"] = city["city_crashes"].shift(1).rolling(12, min_periods=6).mean()
    city["city_crash_trend"] = city["city_crash_lag1"] / (city["city_crash_mean_12"] + 1.0)
    city_injury_12 = city["city_injury"].shift(1).rolling(12, min_periods=6).sum()
    city_total_12 = city["city_crashes"].shift(1).rolling(12, min_periods=6).sum()
    city["city_injury_share_12"] = np.divide(
        city_injury_12,
        city_total_12,
        out=np.full(len(city), np.nan),
        where=city_total_12 > 0,
    )
    city["city_shock_lag1"] = city["city_crash_lag1"] - city["city_crash_mean_12"]
    city = city[
        [
            "month",
            "city_crash_lag1",
            "city_crash_mean_3",
            "city_crash_mean_12",
            "city_crash_trend",
            "city_injury_share_12",
            "city_shock_lag1",
        ]
    ]
    merged = frame.merge(city, on="month", how="left")
    merged["location_city_ratio_12"] = merged["roll_mean_12"] / (
        merged["city_crash_mean_12"] / max(merged["location_id"].nunique(), 1) + 0.5
    )
    return merged


def _add_neighbor_features(frame: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    """Gaussian-kernel neighbourhood crash history using prior months only."""
    location_ids = locations["location_id"].tolist()
    months = pd.DatetimeIndex(sorted(frame["month"].unique()))
    n_locations, n_months = len(location_ids), len(months)

    def matrix(column: str) -> np.ndarray:
        pivot = (
            frame.pivot(index="location_id", columns="month", values=column)
            .reindex(index=location_ids, columns=months)
            .fillna(0.0)
        )
        return pivot.to_numpy(float)

    y_all = matrix("y_all")
    y_injury = matrix("y_injury")
    weights = _neighborhood_weights(locations)
    weighted_all = np.asarray(weights @ y_all)
    weighted_injury = np.asarray(weights @ y_injury)
    weighted_zero = np.asarray(weights @ (y_all == 0).astype(float))

    lag1 = np.full((n_locations, n_months), np.nan)
    mean12 = np.full((n_locations, n_months), np.nan)
    std12 = np.full((n_locations, n_months), np.nan)
    zero12 = np.full((n_locations, n_months), np.nan)
    injury12 = np.full((n_locations, n_months), np.nan)
    for index in range(1, n_months):
        lag1[:, index] = weighted_all[:, index - 1]
        lo = max(0, index - 12)
        window_all = weighted_all[:, lo:index]
        window_injury = weighted_injury[:, lo:index]
        window_zero = weighted_zero[:, lo:index]
        mean12[:, index] = window_all.mean(axis=1)
        std12[:, index] = window_all.std(axis=1, ddof=0)
        zero12[:, index] = window_zero.mean(axis=1)
        denominator = window_all.sum(axis=1)
        injury12[:, index] = np.divide(
            window_injury.sum(axis=1), denominator, out=np.full(n_locations, np.nan), where=denominator > 0
        )

    columns = {
        "neighbor_crash_lag1": lag1,
        "neighbor_crash_mean_12": mean12,
        "neighbor_crash_std_12": std12,
        "neighbor_zero_share_12": zero12,
        "neighbor_injury_share_12": injury12,
        "neighbor_shock_lag1": lag1 - mean12,
    }
    long = pd.DataFrame(
        {
            "location_id": np.repeat(location_ids, n_months),
            "month": np.tile(months.to_numpy(), n_locations),
        }
    )
    for name, matrix_values in columns.items():
        long[name] = matrix_values.reshape(-1)
    return frame.merge(long, on=["location_id", "month"], how="left")


def validate_v2_leakage(frame: pd.DataFrame, locations: pd.DataFrame) -> dict[str, Any]:
    """Recompute the spatial-history features and verify prior-month construction."""
    problems: list[str] = []
    location_ids = locations["location_id"].tolist()
    months = pd.DatetimeIndex(sorted(frame["month"].unique()))
    pivot = (
        frame.pivot(index="location_id", columns="month", values="y_all")
        .reindex(index=location_ids, columns=months)
        .fillna(0.0)
        .to_numpy(float)
    )
    weights = _neighborhood_weights(locations)
    weighted = np.asarray(weights @ pivot)
    stored = (
        frame.pivot(index="location_id", columns="month", values="neighbor_crash_lag1")
        .reindex(index=location_ids, columns=months)
        .to_numpy(float)
    )
    if len(months) > 2:
        expected = weighted[:, :-1]
        actual = stored[:, 1:]
        mask = np.isfinite(actual)
        if not np.allclose(actual[mask], expected[mask], atol=1e-6):
            problems.append("neighbor_crash_lag1 does not use only the prior month")

    # Lags 24/36 are mechanical checks in addition to the V1 leakage tests.
    ordered = frame.sort_values(["location_id", "month"])
    group = ordered.groupby("location_id", sort=False)["y_all"]
    for lag in (24, 36):
        expected = group.shift(lag)
        actual = ordered[f"lag_{lag}"]
        mask = expected.notna() & actual.notna()
        if not np.allclose(actual[mask], expected[mask], atol=1e-9):
            problems.append(f"lag_{lag} does not use only prior months")
    return {"ok": not problems, "problems": problems}


@dataclass
class V2DatasetBundle(DatasetBundle):
    version: str = "2.0.0"


def build_dataset_v2(
    config_path: str | Path = "config/analysis.yaml",
    ml_config: MLDatasetConfig | None = None,
    refresh: bool = False,
) -> V2DatasetBundle:
    """Build the V2 dataset from real portal and TIGER data."""
    ml_config = ml_config or MLDatasetConfig()
    config = load_config(config_path)
    data, quality = prepare_data(config, refresh=refresh, reuse_processed=True)
    raw = path_for(config, "raw")
    processed = path_for(config, "processed")

    daily_weather, weather_provenance = fetch_daily_weather(
        start="2015-01-01",
        end=str(pd.Timestamp(ml_config.target_end).date()),
        cache_dir=raw,
        refresh=refresh,
    )
    weather = monthly_weather(daily_weather)
    holidays = monthly_holidays(pd.DatetimeIndex(data.months))

    locations = build_locations(data, ml_config)
    locations.to_parquet(processed / "ml_v2_locations.parquet", index=False)
    counts = build_counts(data, locations, ml_config.radius_m)

    frame = add_features(counts, locations, data, weather, holidays, ml_config)
    frame = _add_temporal_features_v2(frame)
    frame = _add_weather_normals(frame, weather)

    road_cache = processed / "ml_v2_road_features.parquet"
    if road_cache.exists() and not refresh:
        road_features = pd.read_parquet(road_cache)
        road_metadata = json.loads((processed / "ml_v2_road_metadata.json").read_text())
    else:
        road_result = build_road_features(locations, raw / "tiger", epsg=data.metric_epsg, refresh=refresh)
        road_features = road_result.features
        road_metadata = road_result.metadata
        road_features.to_parquet(road_cache, index=False)
        (processed / "ml_v2_road_metadata.json").write_text(
            json.dumps(road_metadata, indent=2, sort_keys=True) + "\n"
        )
    frame = frame.merge(road_features, on="location_id", how="left")
    frame = _add_citywide_features(frame)
    frame = _add_neighbor_features(frame, locations)

    # The V1 leakage checks run on the full history frame before target filtering.
    from .dataset import validate_no_leakage

    v1_leakage = validate_no_leakage(frame, FEATURE_COLUMNS_V2)
    v2_leakage = validate_v2_leakage(frame, locations)
    problems = v1_leakage["problems"] + v2_leakage["problems"]
    if problems:
        raise RuntimeError(f"V2 feature leakage checks failed: {problems}")

    frame = frame.loc[
        frame["month"].between(pd.Timestamp(ml_config.target_start), pd.Timestamp(ml_config.target_end))
    ].reset_index(drop=True)

    all_nan = [column for column in FEATURE_COLUMNS_V2 if frame[column].isna().all()]
    feature_columns = [column for column in FEATURE_COLUMNS_V2 if column not in all_nan]
    if all_nan:
        LOGGER.warning("dropping all-missing V2 features: %s", all_nan)
    assert_valid_features(frame, feature_columns)

    metadata = {
        "version": "2.0.0",
        "data_snapshot": quality.get("generated_at_utc"),
        "target_window": {
            "first": str(pd.Timestamp(ml_config.target_start).date()),
            "last": str(pd.Timestamp(ml_config.target_end).date()),
        },
        "test_start": str(pd.Timestamp(ml_config.test_start).date()),
        "n_location_months": int(len(frame)),
        "n_locations": int(locations["location_id"].nunique()),
        "n_camera_locations": int((locations["site_type"] == "camera").sum()),
        "n_control_locations": int((locations["site_type"] == "control").sum()),
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "radius_m": ml_config.radius_m,
        "target_definition": (
            "Reported crashes within 250 m of the location coordinate during the "
            "calendar month. Not crashes per vehicle-mile."
        ),
        "location_set": (
            "Camera catchments (overlapping catchments deduplicated) plus the causal "
            "design's comparison sites. Crash-dense, non-random sample of Chicago."
        ),
        "categorical_features": CATEGORICAL_FEATURES,
        "weather": weather_provenance,
        "road_network": road_metadata,
        "leakage_checks": {"v1": v1_leakage, "v2": v2_leakage},
        "new_features": {
            "temporal": NEW_TEMPORAL_FEATURES,
            "road": ROAD_FEATURES,
            "citywide": NEW_CITYWIDE_FEATURES,
            "neighbor": NEW_NEIGHBOR_FEATURES,
        },
        "traffic_volume": {
            "used": False,
            "reason": (
                "The current Chicago AADT dataset is a 2026 snapshot and the historical "
                "2006 study is too old/sparse. Using either for 2017–2025 training would "
                "leak future exposure or misrepresent current exposure; no volume values "
                "were manufactured."
            ),
        },
    }
    frame.to_parquet(processed / "ml_v2_dataset.parquet", index=False)
    return V2DatasetBundle(
        frame=frame,
        locations=locations,
        feature_columns=feature_columns,
        categorical_features=CATEGORICAL_FEATURES,
        metadata=metadata,
        study_data=data,
        version="2.0.0",
    )
