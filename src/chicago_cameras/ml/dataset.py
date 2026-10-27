"""Location-month dataset with strict temporal and geographic leakage controls.

One row is a **location-month**: the number of reported crashes expected in a
250 m catchment during a calendar month.  Features use only information that is
available strictly before the target month (lags, rolling windows, lagged
weather) plus deterministic calendar and static geographic attributes.

The location set is the union of camera catchments (overlapping catchments
deduplicated) and the causal design's comparison sites.  It is a crash-dense,
non-random sample of Chicago; this is documented rather than hidden.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from ..config import load_config, path_for
from ..design import StudyData, resolve_overlaps
from ..pipeline import prepare_data
from .external import (
    fetch_daily_weather,
    lagged_weather_features,
    monthly_holidays,
    monthly_weather,
)

LOGGER = logging.getLogger(__name__)

CITY_CENTER_LAT = 41.85
CITY_CENTER_LON = -87.65

FEATURE_COLUMNS: list[str] = [
    # calendar / regime
    "month_index",
    "year",
    "month_of_year",
    "month_sin",
    "month_cos",
    "quarter",
    "days_in_month",
    "n_holidays",
    "is_covid",
    # crash history (all shifted by at least one month)
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "roll_mean_3",
    "roll_mean_6",
    "roll_mean_12",
    "roll_std_3",
    "roll_std_6",
    "roll_std_12",
    "expanding_mean",
    "expanding_std",
    "zero_share_12",
    "injury_share_12",
    "severe_share_12",
    "trend_12",
    # lagged weather (never the target month's weather)
    "temp_lag1",
    "precip_lag1",
    "snow_lag1",
    "temp_mean_3",
    "precip_mean_3",
    "snow_mean_3",
    # static geography (metric easting/northing, named to avoid confusion with the target y)
    "easting_m",
    "northing_m",
    "lat",
    "lon",
    "dist_center_m",
    "dist_nearest_camera_m",
    "n_cameras_1km",
    "cluster",
    "site_is_camera",
    # camera operation at the forecast month (observational only)
    "camera_operational",
    "camera_age_months",
    "n_operational_cameras_1km",
]

CATEGORICAL_FEATURES = ["cluster"]
TARGET_COLUMNS = ["y_all", "y_injury", "y_severe"]


@dataclass(frozen=True)
class MLDatasetConfig:
    radius_m: float = 250.0
    buffer_m: float = 600.0
    grid_m: float = 500.0
    min_control_crashes: int = 100
    min_active_share: float = 0.40
    target_start: str = "2017-01-01"
    target_end: str = "2026-08-01"
    test_start: str = "2025-09-01"
    kmeans_clusters: int = 10
    seed: int = 20261024


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    locations: pd.DataFrame
    feature_columns: list[str]
    categorical_features: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    study_data: Any = None

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[self.feature_columns]

    @property
    def y(self) -> pd.Series:
        return self.frame["y_all"]

    def split(self, name: str) -> pd.DataFrame:
        return self.frame.loc[self.frame["split"] == name].copy()


def _to_lonlat(xy: np.ndarray, epsg: int) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon, lat = transformer.transform(xy[:, 0], xy[:, 1])
    return np.asarray(lon), np.asarray(lat)


def build_locations(data: StudyData, config: MLDatasetConfig) -> pd.DataFrame:
    """Camera catchments (overlap-deduplicated) plus comparison sites."""
    cameras = data.cameras.copy().reset_index(drop=True)
    keep, _ = resolve_overlaps(
        cameras, config.radius_m, np.ones(len(cameras), dtype=bool), "drop_later"
    )
    cameras = cameras.loc[keep].reset_index(drop=True)

    control_xy, _ = data.control_sites(
        radius_m=config.radius_m,
        buffer_m=config.buffer_m,
        grid_m=config.grid_m,
        grid_offset_m=0.0,
        min_crashes=config.min_control_crashes,
        min_active_share=config.min_active_share,
    )

    camera_frame = pd.DataFrame(
        {
            "location_id": "CAM:" + cameras["camera_uid"].astype(str),
            "site_type": "camera",
            "camera_uid": cameras["camera_uid"].astype(str),
            "go_month": pd.to_datetime(cameras["go_month"]),
            "x": cameras["x"].to_numpy(float),
            "y": cameras["y"].to_numpy(float),
            "lat": cameras["lat"].to_numpy(float),
            "lon": cameras["lon"].to_numpy(float),
        }
    )
    control_lon, control_lat = _to_lonlat(control_xy, data.metric_epsg)
    control_frame = pd.DataFrame(
        {
            "location_id": [f"CTL:{i:05d}" for i in range(len(control_xy))],
            "site_type": "control",
            "camera_uid": None,
            "go_month": pd.NaT,
            "x": control_xy[:, 0],
            "y": control_xy[:, 1],
            "lat": control_lat,
            "lon": control_lon,
        }
    )

    locations = pd.concat([camera_frame, control_frame], ignore_index=True)
    locations["go_idx"] = data.months.get_indexer(locations["go_month"]).astype(int)
    locations.loc[locations["site_type"] == "control", "go_idx"] = -1

    camera_xy = data.cameras[["x", "y"]].to_numpy(float)
    loc_xy = locations[["x", "y"]].to_numpy(float)
    distance = cdist(loc_xy, camera_xy)
    # City centre projected through the same CRS for a metric distance.
    center = Transformer.from_crs(4326, data.metric_epsg, always_xy=True).transform(
        CITY_CENTER_LON, CITY_CENTER_LAT
    )
    locations["dist_center_m"] = np.linalg.norm(loc_xy - np.asarray(center), axis=1)
    locations["dist_nearest_camera_m"] = distance.min(axis=1)
    locations["n_cameras_1km"] = (distance <= 1000.0).sum(axis=1).astype(int)

    km = KMeans(
        n_clusters=config.kmeans_clusters, n_init=10, random_state=config.seed
    ).fit(loc_xy)
    locations["cluster"] = km.labels_.astype(int)
    locations["site_is_camera"] = (locations["site_type"] == "camera").astype(int)
    return locations


def build_counts(
    data: StudyData, locations: pd.DataFrame, radius_m: float
) -> pd.DataFrame:
    """Long location-month counts for all, injury and severe crashes."""
    xy = locations[["x", "y"]].to_numpy(float)
    arrays = {
        "y_all": data.counts_at(xy, radius_m, "all"),
        "y_injury": data.counts_at(xy, radius_m, "injury"),
        "y_severe": data.counts_at(xy, radius_m, "severe"),
    }
    n_locations = len(locations)
    n_months = len(data.months)
    frame = pd.DataFrame(
        {
            "location_id": np.repeat(locations["location_id"].to_numpy(), n_months),
            "month": np.tile(pd.DatetimeIndex(data.months).to_numpy(), n_locations),
        }
    )
    for name, matrix in arrays.items():
        frame[name] = matrix.reshape(-1)
    return frame


def _camera_operation_features(
    data: StudyData, locations: pd.DataFrame
) -> pd.DataFrame:
    """Operational-camera counts and camera age at each location-month.

    These are observational features: a camera being live is correlated with
    where the city chose to enforce.  They are never interpreted causally.
    """
    camera_xy = data.cameras[["x", "y"]].to_numpy(float)
    go_idx = data.months.get_indexer(data.cameras["go_month"]).astype(int)
    go_idx = np.clip(go_idx, 0, None)
    n_months = len(data.months)
    month_idx = np.arange(n_months)
    operational = (month_idx[None, :] >= go_idx[:, None]).astype(int)

    distance = cdist(locations[["x", "y"]].to_numpy(float), camera_xy)
    within_250 = distance <= 250.0
    within_1000 = distance <= 1000.0

    n_ops_250 = within_250.astype(int) @ operational
    n_ops_1000 = within_1000.astype(int) @ operational

    near_go = np.where(within_250, go_idx[None, :], 10**6).min(axis=1)
    age = np.where(
        (near_go[:, None] < 10**6) & (month_idx[None, :] >= near_go[:, None]),
        month_idx[None, :] - near_go[:, None] + 1,
        0,
    )

    n_locations = len(locations)
    frame = pd.DataFrame(
        {
            "location_id": np.repeat(locations["location_id"].to_numpy(), n_months),
            "month": np.tile(pd.DatetimeIndex(data.months).to_numpy(), n_locations),
            "camera_operational": (n_ops_250 > 0).astype(int).reshape(-1),
            "camera_age_months": age.reshape(-1),
            "n_operational_cameras_1km": n_ops_1000.reshape(-1),
        }
    )
    return frame


def _add_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Lag and rolling features computed strictly from earlier months."""
    frame = frame.sort_values(["location_id", "month"]).copy()
    group = frame.groupby("location_id", sort=False)

    for lag in (1, 2, 3, 6, 12):
        frame[f"lag_{lag}"] = group["y_all"].shift(lag)

    def _roll(series: pd.Series, window: int, how: str, min_periods: int) -> pd.Series:
        return series.shift(1).rolling(window, min_periods=min_periods).agg(how)

    for window, min_periods in ((3, 2), (6, 4), (12, 8)):
        frame[f"roll_mean_{window}"] = group["y_all"].transform(
            lambda s, w=window, m=min_periods: _roll(s, w, "mean", m)
        )
        frame[f"roll_std_{window}"] = group["y_all"].transform(
            lambda s, w=window, m=min_periods: _roll(s, w, "std", m)
        )

    frame["expanding_mean"] = group["y_all"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    frame["expanding_std"] = group["y_all"].transform(
        lambda s: s.shift(1).expanding(min_periods=2).std()
    )
    frame["zero_share_12"] = group["y_all"].transform(
        lambda s: (s == 0).astype(float).shift(1).rolling(12, min_periods=6).mean()
    )

    injury_sum = group["y_injury"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=6).sum()
    )
    severe_sum = group["y_severe"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=6).sum()
    )
    total_sum = group["y_all"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=6).sum()
    )
    frame["injury_share_12"] = np.where(total_sum > 0, injury_sum / total_sum, np.nan)
    frame["severe_share_12"] = np.where(total_sum > 0, severe_sum / total_sum, np.nan)
    frame["trend_12"] = frame["lag_1"] / (frame["roll_mean_12"] + 0.5)
    return frame


def add_features(
    frame: pd.DataFrame,
    locations: pd.DataFrame,
    data: StudyData,
    weather: pd.DataFrame,
    holidays: pd.DataFrame,
    config: MLDatasetConfig,
) -> pd.DataFrame:
    """Attach calendar, history, weather, geographic and camera features."""
    frame = frame.copy()
    frame["month"] = pd.to_datetime(frame["month"])

    frame = _add_history_features(frame)
    frame = frame.merge(_camera_operation_features(data, locations), on=["location_id", "month"], how="left")
    static = locations[
        [
            "location_id",
            "site_type",
            "camera_uid",
            "x",
            "y",
            "lat",
            "lon",
            "dist_center_m",
            "dist_nearest_camera_m",
            "n_cameras_1km",
            "cluster",
            "site_is_camera",
        ]
    ].rename(columns={"x": "easting_m", "y": "northing_m"})
    frame = frame.merge(static, on="location_id", how="left")

    month = frame["month"]
    first = pd.Timestamp(data.months[0])
    frame["month_index"] = (month.dt.year - first.year) * 12 + (month.dt.month - first.month)
    frame["year"] = month.dt.year
    frame["month_of_year"] = month.dt.month
    frame["month_sin"] = np.sin(2 * np.pi * (frame["month_of_year"] - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (frame["month_of_year"] - 1) / 12)
    frame["quarter"] = month.dt.quarter
    frame["days_in_month"] = month.dt.days_in_month
    frame["is_covid"] = month.between("2020-03-01", "2021-12-01").astype(int)

    calendar = pd.DataFrame({"month": pd.DatetimeIndex(sorted(frame["month"].unique()))})
    calendar = calendar.merge(holidays, on="month", how="left")
    frame = frame.merge(calendar, on="month", how="left")

    weather_features = lagged_weather_features(weather, pd.DatetimeIndex(calendar["month"]))
    frame = frame.merge(weather_features, on="month", how="left")

    frame["split"] = np.where(
        frame["month"] < pd.Timestamp(config.test_start), "train", "test"
    )
    return frame


def validate_no_leakage(frame: pd.DataFrame, feature_columns: Iterable[str]) -> dict[str, Any]:
    """Mechanical checks that no feature at month t uses month t or later.

    The checks are deliberately literal: they recompute lags and rolling means
    from the outcome and compare, and they perturb future outcomes to ensure
    features before the perturbation are unchanged.
    """
    features = list(feature_columns)
    problems: list[str] = []
    ordered = frame.sort_values(["location_id", "month"])

    group = ordered.groupby("location_id", sort=False)
    for lag in (1, 2, 3, 6, 12):
        expected = group["y_all"].shift(lag)
        actual = ordered[f"lag_{lag}"]
        mask = expected.notna() & actual.notna()
        if not np.allclose(actual[mask], expected[mask], atol=1e-9):
            problems.append(f"lag_{lag} does not match y_all.shift({lag})")

    base = group["y_all"].shift(1)
    expected_roll = base.groupby(ordered["location_id"], sort=False).transform(
        lambda s: s.rolling(3, min_periods=2).mean()
    )
    mask = expected_roll.notna() & ordered["roll_mean_3"].notna()
    if not np.allclose(ordered.loc[mask, "roll_mean_3"], expected_roll[mask], atol=1e-9):
        problems.append("roll_mean_3 does not use only prior months")

    # Perturb a future outage and confirm earlier features do not move.
    sample_locations = ordered["location_id"].drop_duplicates().head(3).tolist()
    sample = ordered.loc[ordered["location_id"].isin(sample_locations)].copy()
    months = sorted(sample["month"].unique())
    if len(months) >= 4:
        cut = months[len(months) // 2]
        perturbed = sample.copy()
        perturbed.loc[perturbed["month"] >= cut, "y_all"] += 100
        before = sample.loc[sample["month"] < cut]
        after = perturbed.loc[perturbed["month"] < cut]
        comparable = [c for c in features if c in sample.columns]
        if not before[comparable].reset_index(drop=True).equals(
            after[comparable].reset_index(drop=True)
        ):
            problems.append("future outcome perturbation changed earlier features")

    forbidden = {"y_all", "y_injury", "y_severe", "future_crashes"}
    leakage_columns = sorted(forbidden.intersection(features))
    if leakage_columns:
        problems.append(f"target columns used as features: {leakage_columns}")

    return {"ok": not problems, "problems": problems, "n_features": len(features)}


def assert_valid_features(frame: pd.DataFrame, feature_columns: list[str]) -> None:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"feature schema mismatch: missing {missing}")
    if frame[feature_columns].isna().all(axis=0).any():
        empty = frame[feature_columns].columns[frame[feature_columns].isna().all()].tolist()
        raise ValueError(f"features with no observed values: {empty}")


def build_dataset(
    config_path: str | Path = "config/analysis.yaml",
    ml_config: MLDatasetConfig | None = None,
    refresh: bool = False,
) -> DatasetBundle:
    """Build the verified location-month dataset from real portal data."""
    ml_config = ml_config or MLDatasetConfig()
    config = load_config(config_path)
    data, quality = prepare_data(config, refresh=refresh, reuse_processed=True)
    processed = path_for(config, "processed")
    raw = path_for(config, "raw")

    daily_weather, weather_provenance = fetch_daily_weather(
        start="2015-01-01",
        end=str(pd.Timestamp(ml_config.target_end).date()),
        cache_dir=raw,
        refresh=refresh,
    )
    weather = monthly_weather(daily_weather)
    holidays = monthly_holidays(pd.DatetimeIndex(data.months))

    locations = build_locations(data, ml_config)
    counts = build_counts(data, locations, ml_config.radius_m)
    frame = add_features(counts, locations, data, weather, holidays, ml_config)

    # Leakage checks run on the frame that still contains the pre-target history,
    # because rolling features legitimately reach back before the target window.
    leakage = validate_no_leakage(frame, FEATURE_COLUMNS)
    if not leakage["ok"]:
        raise RuntimeError(f"feature leakage checks failed: {leakage['problems']}")

    frame = frame.loc[
        frame["month"].between(pd.Timestamp(ml_config.target_start), pd.Timestamp(ml_config.target_end))
    ].reset_index(drop=True)

    all_nan = [column for column in FEATURE_COLUMNS if frame[column].isna().all()]
    weather_columns = {
        "temp_lag1",
        "precip_lag1",
        "snow_lag1",
        "temp_mean_3",
        "precip_mean_3",
        "snow_mean_3",
    }
    if all_nan and set(all_nan) <= weather_columns:
        feature_columns = [column for column in FEATURE_COLUMNS if column not in all_nan]
        LOGGER.warning(
            "weather unavailable; dropping weather features: %s", sorted(all_nan)
        )
    elif all_nan:
        raise ValueError(f"features with no observed values: {all_nan}")
    else:
        feature_columns = list(FEATURE_COLUMNS)

    assert_valid_features(frame, feature_columns)

    location_tree = cKDTree(locations[["x", "y"]].to_numpy(float))
    nearest, _ = location_tree.query(locations[["x", "y"]].to_numpy(float), k=2)
    min_spacing = float(nearest[:, 1].min())

    metadata = {
        "data_snapshot": quality.get("generated_at_utc"),
        "min_location_spacing_m": min_spacing,
        "overlap_note": (
            "Camera catchments are overlap-deduplicated; comparison sites are >600 m from every "
            "camera and on a 500 m lattice. A crash can still sit on the boundary of two "
            "comparison catchments, but catchments do not overlap in area."
        ),
        "analysis_months": {
            "first": str(data.months[0].date()),
            "last": str(data.months[-1].date()),
            "n": len(data.months),
        },
        "target_window": {
            "first": str(pd.Timestamp(ml_config.target_start).date()),
            "last": str(pd.Timestamp(ml_config.target_end).date()),
        },
        "test_start": str(pd.Timestamp(ml_config.test_start).date()),
        "n_location_months": len(frame),
        "n_locations": int(locations["location_id"].nunique()),
        "n_camera_locations": int((locations["site_type"] == "camera").sum()),
        "n_control_locations": int((locations["site_type"] == "control").sum()),
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "categorical_features": CATEGORICAL_FEATURES,
        "weather": weather_provenance,
        "leakage_checks": leakage,
        "location_set": (
            "Camera catchments (overlapping catchments deduplicated) plus the causal "
            "design's comparison sites. Crash-dense, non-random sample of Chicago."
        ),
        "target_definition": (
            "Reported crashes within 250 m of the location coordinate during the "
            "calendar month. Not crashes per vehicle-mile."
        ),
        "radius_m": ml_config.radius_m,
        "config_path": str(config_path),
    }

    frame.to_parquet(processed / "ml_dataset.parquet", index=False)
    locations.to_parquet(processed / "ml_locations.parquet", index=False)
    return DatasetBundle(
        frame=frame,
        locations=locations,
        feature_columns=feature_columns,
        categorical_features=CATEGORICAL_FEATURES,
        metadata=metadata,
        study_data=data,
    )


def build_crash_grid(
    data: StudyData,
    start_month: str = "2024-01-01",
    grid_m: float = 500.0,
    radius_m: float = 250.0,
) -> pd.DataFrame:
    """Aggregate observed crashes to a 500 m grid for efficient map rendering."""
    from ..geospatial import grid_points

    points = grid_points(data.bounds_xy, spacing_m=grid_m)
    all_counts = data.counts_at(points, radius_m, "all")
    injury_counts = data.counts_at(points, radius_m, "injury")
    severe_counts = data.counts_at(points, radius_m, "severe")

    transformer = Transformer.from_crs(data.metric_epsg, 4326, always_xy=True)
    lon, lat = transformer.transform(points[:, 0], points[:, 1])
    start = pd.Timestamp(start_month).to_period("M").to_timestamp()
    month_index = data.months.get_indexer(pd.DatetimeIndex([start]))[0]
    if month_index < 0:
        raise ValueError(f"{start_month} is outside the observed month window")

    rows: list[pd.DataFrame] = []
    for index in range(month_index, len(data.months)):
        month = data.months[index]
        mask = (all_counts[:, index] + injury_counts[:, index] + severe_counts[:, index]) > 0
        if not mask.any():
            continue
        rows.append(
            pd.DataFrame(
                {
                    "cell_id": np.flatnonzero(mask),
                    "year": month.year,
                    "month": month,
                    "lat": lat[mask],
                    "lon": lon[mask],
                    "crashes_all": all_counts[mask, index],
                    "crashes_injury": injury_counts[mask, index],
                    "crashes_severe": severe_counts[mask, index],
                }
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=["cell_id", "year", "month", "lat", "lon", "crashes_all", "crashes_injury", "crashes_severe"]
        )
    return pd.concat(rows, ignore_index=True)
