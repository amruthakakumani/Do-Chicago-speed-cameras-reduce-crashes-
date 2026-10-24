"""Tests for stacked-panel construction and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chicago_cameras.panel import build_panel, validate_panel


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    Y_cam = rng.poisson(5, size=(2, 12))
    Y_ctl = rng.poisson(5, size=(4, 12))
    return build_panel(
        Y_camera=Y_cam,
        Y_control=Y_ctl,
        matches={0: np.array([0, 1, 2]), 1: np.array([1, 2, 3])},
        camera_go_idx={0: 4, 1: 6},
        camera_uids={0: "A", 1: "B"},
        control_ids=np.array(["G0", "G1", "G2", "G3"]),
        pre_months=4,
        post_months=4,
        camera_go_month={0: pd.Timestamp("2016-02-01"), 1: pd.Timestamp("2016-04-01")},
    )


def test_panel_shape_and_composition() -> None:
    panel = _panel()
    assert len(panel) == 2 * 4 * 8
    assert panel["stack"].nunique() == 2
    assert panel.groupby("stack")["treated"].sum().eq(8).all()
    assert set(panel["rel"].unique()) == set(range(-4, 4))


def test_panel_keeps_observed_zeros() -> None:
    rng = np.random.default_rng(1)
    Y_cam = rng.poisson(5, size=(2, 12))
    Y_cam[0, 0] = 0
    Y_ctl = rng.poisson(5, size=(4, 12))
    panel = build_panel(
        Y_camera=Y_cam,
        Y_control=Y_ctl,
        matches={0: np.array([0, 1, 2]), 1: np.array([1, 2, 3])},
        camera_go_idx={0: 4, 1: 6},
        camera_uids={0: "A", 1: "B"},
        control_ids=np.array(["G0", "G1", "G2", "G3"]),
        pre_months=4,
        post_months=4,
    )
    assert ((panel["stack"] == "SA") & (panel["rel"] == -4) & (panel["treated"] == 1)).any()
    assert panel.loc[(panel["stack"] == "SA") & (panel["rel"] == -4) & (panel["treated"] == 1), "y"].iloc[0] == 0
    assert panel["y"].isna().sum() == 0


def test_calendar_month_aligns_with_event_time() -> None:
    panel = _panel()
    stack = panel.loc[panel["stack"] == "SA"]
    zero = stack.loc[stack["rel"] == 0, "calendar_month"].iloc[0]
    assert zero == pd.Timestamp("2016-02-01")
    minus_two = stack.loc[stack["rel"] == -2, "calendar_month"].iloc[0]
    assert minus_two == pd.Timestamp("2015-12-01")


def test_validation_reports_no_problems_for_valid_panel() -> None:
    result = validate_panel(_panel(), pre_months=4, post_months=4)
    assert result["problems"] == []
    assert result["n_stacks"] == 2
    assert result["controls_per_stack"] == {"3": 2}


def test_validation_detects_missing_rows() -> None:
    panel = _panel().iloc[:-1]
    result = validate_panel(panel, pre_months=4, post_months=4)
    assert result["problems"]


def test_build_panel_rejects_impossible_window() -> None:
    with pytest.raises(ValueError):
        build_panel(
            Y_camera=np.zeros((1, 5)),
            Y_control=np.zeros((1, 5)),
            matches={0: np.array([0])},
            camera_go_idx={0: 2},
            camera_uids={0: "A"},
            control_ids=np.array(["G0"]),
            pre_months=4,
            post_months=4,
        )
