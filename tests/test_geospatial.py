"""Tests for distance, counting, overlap and lattice primitives."""

from __future__ import annotations

import numpy as np

from chicago_cameras.geospatial import (
    assignment_pairs,
    attribution_diagnostics,
    connected_components,
    count_site_months,
    grid_points,
    haversine_m,
    overlap_pairs,
    project_lonlat,
)


def test_projection_matches_great_circle_distance() -> None:
    # Two points about 1 km apart in Chicago.
    lon = np.array([-87.65, -87.63852])
    lat = np.array([41.85, 41.85])
    xy = project_lonlat(lon, lat)
    projected = float(np.linalg.norm(xy[1] - xy[0]))
    geodesic = float(haversine_m(lat[:1], lon[:1], lat[1:], lon[1:])[0])
    assert abs(projected - geodesic) / geodesic < 0.004


def test_projection_rejects_mismatched_shapes() -> None:
    import pytest

    with pytest.raises(ValueError):
        project_lonlat([-87.65], [41.85, 41.86])


def test_counts_retain_observed_zeros() -> None:
    sites = np.array([[0.0, 0.0], [1000.0, 0.0]])
    crashes = np.array([[0.0, 0.0], [0.0, 0.0]])
    months = np.array([1, 1])
    counts = count_site_months(sites, crashes, months, n_months=3, radius_m=250)
    assert counts.shape == (2, 3)
    assert counts[0, 1] == 2
    assert counts[0, 0] == 0 and counts[0, 2] == 0
    assert counts[1].sum() == 0


def test_crash_outside_radius_not_counted() -> None:
    sites = np.array([[0.0, 0.0]])
    crashes = np.array([[300.0, 0.0]])
    counts = count_site_months(sites, crashes, np.array([0]), n_months=1, radius_m=250)
    assert counts[0, 0] == 0


def test_shared_crash_attribution_diagnostics() -> None:
    sites = np.array([[0.0, 0.0], [100.0, 0.0]])
    crashes = np.array([[50.0, 0.0], [150.0, 0.0]])
    crash_idx, site_idx = assignment_pairs(sites, crashes, radius_m=100)
    assert len(crash_idx) == 3  # crash 0 goes to both sites; crash 1 to site 1 only
    diagnostics = attribution_diagnostics(crash_idx, n_crashes=2)
    assert diagnostics.n_shared_crashes == 1
    assert diagnostics.max_sites_per_crash == 2
    assert diagnostics.histogram[1] == 1 and diagnostics.histogram[2] == 1
    assert len(site_idx) == 3


def test_overlap_pairs_and_components() -> None:
    points = np.array([[0.0, 0.0], [100.0, 0.0], [1000.0, 0.0], [1050.0, 0.0]])
    pairs = overlap_pairs(points, min_distance_m=200)
    assert sorted(map(tuple, pairs.tolist())) == [(0, 1), (2, 3)]
    labels = connected_components(4, pairs)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_grid_spacing_and_offset() -> None:
    points = grid_points((0.0, 0.0, 1000.0, 1000.0), spacing_m=500, offset=(250.0, 0.0))
    assert points.shape == (4, 2)  # x in {250, 750}, y in {0, 500}
    assert set(np.unique(points[:, 0])) == {250.0, 750.0}
