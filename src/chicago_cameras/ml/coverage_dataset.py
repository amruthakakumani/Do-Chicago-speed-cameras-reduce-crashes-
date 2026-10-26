"""Coverage-consistent camera-zone dataset.

The existing V1/V2 datasets begin in January 2017, eight months before the
portal's citywide reporting start.  This module rebuilds the same 250 m
camera/control geography from September 2017 onward and starts the target window
in January 2018, so every feature is derived from the citywide regime.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
import pandas as pd

from ..config import load_config, path_for
from ..pipeline import prepare_data
from .dataset import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    DatasetBundle,
    MLDatasetConfig,
    add_features,
    assert_valid_features,
    build_counts,
    build_locations,
    validate_no_leakage,
)
from .external import fetch_daily_weather, monthly_holidays, monthly_weather

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoverageDatasetConfig:
    config_path: str = "config/analysis.yaml"
    target_start: str = "2018-01-01"
    target_end: str = "2026-08-01"
    test_start: str = "2025-09-01"
    citywide_start: str = "2017-09-01"
    radius_m: float = 250.0
    buffer_m: float = 600.0
    grid_m: float = 500.0
    min_control_crashes: int = 100
    min_active_share: float = 0.40
    seed: int = 20261024


def build_coverage_bundle(config: CoverageDatasetConfig = CoverageDatasetConfig()) -> DatasetBundle:
    cfg = load_config(config.config_path)
    data, quality = prepare_data(cfg, reuse_processed=True)
    citywide = data.subset_start(config.citywide_start)
    processed = path_for(cfg, "processed")
    raw = path_for(cfg, "raw")
    ml_config = MLDatasetConfig(
        radius_m=config.radius_m,
        buffer_m=config.buffer_m,
        grid_m=config.grid_m,
        min_control_crashes=config.min_control_crashes,
        min_active_share=config.min_active_share,
        target_start=config.target_start,
        target_end=config.target_end,
        test_start=config.test_start,
        seed=config.seed,
    )
    weather_start = "2017-01-01"
    daily_weather, weather_provenance = fetch_daily_weather(
        start=weather_start, end=config.target_end, cache_dir=raw, refresh=False
    )
    weather = monthly_weather(daily_weather)
    holidays = monthly_holidays(pd.DatetimeIndex(citywide.months))
    locations = build_locations(citywide, ml_config)
    counts = build_counts(citywide, locations, config.radius_m)
    frame = add_features(counts, locations, citywide, weather, holidays, ml_config)
    leakage = validate_no_leakage(frame, FEATURE_COLUMNS)
    if not leakage["ok"]:
        raise RuntimeError(f"coverage feature leakage: {leakage['problems']}")
    frame = frame.loc[
        frame["month"].between(pd.Timestamp(config.target_start), pd.Timestamp(config.target_end))
    ].reset_index(drop=True)

    feature_columns = [column for column in FEATURE_COLUMNS if not frame[column].isna().all()]
    assert_valid_features(frame, feature_columns)
    frame.to_parquet(processed / "ml_coverage_dataset.parquet", index=False)
    locations.to_parquet(processed / "ml_coverage_locations.parquet", index=False)
    metadata = {
        "data_snapshot": quality.get("generated_at_utc"),
        "coverage_start": config.citywide_start,
        "target_window": {"first": config.target_start, "last": config.target_end},
        "test_start": config.test_start,
        "n_locations": int(locations["location_id"].nunique()),
        "n_camera_locations": int((locations["site_type"] == "camera").sum()),
        "n_control_locations": int((locations["site_type"] == "control").sum()),
        "n_rows": int(len(frame)),
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "categorical_features": CATEGORICAL_FEATURES,
        "leakage_checks": leakage,
        "weather": weather_provenance,
        "location_set": "Camera catchments + 250 m comparison catchments selected on 2017-09+ counts.",
        "target_definition": "Reported crashes within 250 m of location coordinate in calendar month.",
        "note": (
            "This is not a direct replication of the V1/V2 task: both the reporting "
            "regime and the target start differ. Comparisons are made on common test "
            "location-months."
        ),
    }
    (processed / "ml_coverage_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n"
    )
    return DatasetBundle(
        frame=frame,
        locations=locations,
        feature_columns=feature_columns,
        categorical_features=CATEGORICAL_FEATURES,
        metadata=metadata,
        study_data=citywide,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bundle = build_coverage_bundle()
    print(bundle.frame.shape)
    print(bundle.metadata["n_locations"], bundle.metadata["n_camera_locations"], bundle.metadata["n_control_locations"])
    print(bundle.frame.groupby("split").size().to_dict())
