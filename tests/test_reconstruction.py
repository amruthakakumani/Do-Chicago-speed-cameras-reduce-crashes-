"""Feature-group tests: no target columns and stable group construction."""

from __future__ import annotations

from chicago_cameras.ml.reconstruction import ablation_feature_sets, fixed_params


def test_feature_groups_exclude_targets_and_are_disjoint():
    groups = {
        "calendar": ["month_index"],
        "history": ["lag_1"],
        "geography": ["easting_m"],
        "camera": ["camera_operational"],
        "weather": ["temp_lag1"],
        "road": ["road_density_500m_km_per_km2"],
        "neighborhood": ["neighbor_crash_lag1"],
        "citywide": ["city_crash_lag1"],
        "long_history": ["lag_24"],
    }
    sets = ablation_feature_sets(groups)
    assert sets["M0_minimal"] == ["month_index", "lag_1", "easting_m"]
    assert "y_all" not in sets["M6_full"]
    assert all(not feature.startswith("y_") for feature in sets["M6_full"])
    params = fixed_params({"learning_rate": 0.02, "num_leaves": 31})
    assert params["objective"] == "poisson"
    assert params["learning_rate"] == 0.02
    assert params["num_leaves"] == 31
