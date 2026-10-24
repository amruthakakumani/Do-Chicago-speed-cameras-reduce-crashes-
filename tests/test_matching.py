"""Tests for comparison-group matching."""

from __future__ import annotations

import numpy as np

from chicago_cameras.matching import match_controls


def _frames():
    n_controls, n_months = 2, 60
    camera_xy = np.array([[0.0, 0.0]])
    control_xy = np.array([[100.0, 0.0], [200.0, 0.0]])
    Y_control = np.zeros((n_controls, n_months))
    # Control 0 is high before month 24 and low afterwards.
    Y_control[0, :24] = 10.0
    Y_control[0, 24:] = 1.0
    # Control 1 is low before month 24 and high afterwards.
    Y_control[1, :24] = 1.0
    Y_control[1, 24:] = 10.0
    # The camera follows control 1 exactly.
    Y_camera = Y_control[[1]].copy()
    return camera_xy, control_xy, Y_camera, Y_control


def test_matching_uses_each_camera_own_pre_window() -> None:
    """Regression test: control statistics must be evaluated on the camera's window."""
    camera_xy, control_xy, Y_camera, Y_control = _frames()
    result = match_controls(
        camera_xy=camera_xy,
        camera_rows=np.array([0]),
        camera_go_idx=np.array([40]),
        Y_camera=Y_camera,
        control_xy=control_xy,
        Y_control=Y_control,
        pre_months=24,
        n_controls=1,
        max_distance_m=1000.0,
    )
    # The camera pre-window is months 16..39, where control 1 matches exactly.
    assert result.matches[0].tolist() == [1]


def test_distance_cap_excludes_far_controls() -> None:
    camera_xy, control_xy, Y_camera, Y_control = _frames()
    result = match_controls(
        camera_xy=camera_xy,
        camera_rows=np.array([0]),
        camera_go_idx=np.array([40]),
        Y_camera=Y_camera,
        control_xy=control_xy,
        Y_control=Y_control,
        pre_months=24,
        n_controls=2,
        max_distance_m=150.0,
    )
    assert result.matches == {}
    assert result.failures[0] == "too_few_nearby_controls"


def test_caliper_rejects_poor_matches() -> None:
    camera_xy, control_xy, Y_camera, Y_control = _frames()
    Y_camera = Y_camera + 50.0  # far from every control
    result = match_controls(
        camera_xy=camera_xy,
        camera_rows=np.array([0]),
        camera_go_idx=np.array([40]),
        Y_camera=Y_camera,
        control_xy=control_xy,
        Y_control=Y_control,
        pre_months=24,
        n_controls=1,
        max_distance_m=1000.0,
        caliper=0.01,
    )
    assert result.matches == {}
    assert result.failures[0] == "caliper_not_met"


def test_unrestricted_matching_uses_whole_pool() -> None:
    camera_xy, control_xy, Y_camera, Y_control = _frames()
    result = match_controls(
        camera_xy=camera_xy,
        camera_rows=np.array([0]),
        camera_go_idx=np.array([40]),
        Y_camera=Y_camera,
        control_xy=control_xy + 100_000.0,
        Y_control=Y_control,
        pre_months=24,
        n_controls=2,
        max_distance_m=None,
    )
    assert len(result.matches[0]) == 2
