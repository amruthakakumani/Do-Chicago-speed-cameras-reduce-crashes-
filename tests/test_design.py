"""End-to-end design construction on a small synthetic city."""

from __future__ import annotations

import numpy as np
import pandas as pd

from chicago_cameras.design import DesignConfig, StudyData, build_design


def _synthetic_city(seed: int = 11) -> StudyData:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2016-01-01", "2020-12-01", freq="MS")
    cameras = pd.DataFrame(
        {
            "camera_uid": ["C1", "C2", "C3"],
            "location_id": ["C1", "C2", "C3"],
            "address": ["a", "b", "c"],
            "x": [1000.0, 2500.0, 3800.0],
            "y": [1000.0, 1500.0, 800.0],
            "go_live": pd.to_datetime(["2018-02-01"] * 3),
            "go_month": pd.to_datetime(["2018-02-01"] * 3),
        }
    )
    n_months = len(months)
    n_crashes = 90 * n_months
    crashes = pd.DataFrame(
        {
            "crash_record_id": [f"X{i:06d}" for i in range(n_crashes)],
            "x": rng.uniform(0, 5000, n_crashes),
            "y": rng.uniform(0, 4000, n_crashes),
            "month": rng.choice(months, size=n_crashes),
            "injury_crash": rng.integers(0, 2, n_crashes),
            "severe_crash": rng.integers(0, 2, n_crashes),
            "posted_speed_limit": rng.choice([25.0, 30.0, 35.0], size=n_crashes),
        }
    )
    return StudyData(cameras, crashes, months)


def test_build_design_end_to_end() -> None:
    data = _synthetic_city()
    config = DesignConfig(
        radius_m=250,
        buffer_m=600,
        grid_m=500,
        min_control_crashes=20,
        min_active_month_share=0.2,
        max_match_distance_m=2000,
        n_controls=2,
        pre_months=12,
        post_months=12,
        overlap_rule="drop_later",
    )
    design = build_design(data, config)
    assert design.validation["problems"] == []
    assert design.n_cameras >= 1
    assert design.diagnostics["treated_control_catchment_overlap"] is False
    assert design.diagnostics["control_min_distance_to_camera_m"] > 600
    assert design.diagnostics["camera_attribution_matched"]["max_sites_per_crash"] == 1


def test_buffer_must_exceed_twice_radius_in_config_validation() -> None:
    import pytest

    from chicago_cameras.config import load_config

    with pytest.raises(ValueError):
        load_config("tests/_bad_config.yaml")


def test_subset_start_filters_months() -> None:
    data = _synthetic_city()
    subset = data.subset_start("2019-01-01")
    assert subset.months[0] == pd.Timestamp("2019-01-01")
    assert subset.crashes["month"].min() >= pd.Timestamp("2019-01-01")
