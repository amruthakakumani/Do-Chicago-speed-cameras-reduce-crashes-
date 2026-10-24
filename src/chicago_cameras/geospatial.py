"""Geospatial primitives used by the research design.

All distances are computed in a projected metric CRS (default EPSG:26916,
NAD83 / UTM zone 16N) rather than the flat degrees-to-metres approximation used
by the original notebook.  The functions are deliberately small and pure so
that they can be unit-tested with synthetic fixtures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

LOGGER = logging.getLogger(__name__)

WGS84_EPSG = 4326
DEFAULT_METRIC_EPSG = 26916  # NAD83 / UTM zone 16N


def make_transformer(metric_epsg: int = DEFAULT_METRIC_EPSG) -> Transformer:
    """Return a lon/lat -> metric transformer (always_xy keeps lon, lat order)."""
    return Transformer.from_crs(WGS84_EPSG, metric_epsg, always_xy=True)


def project_lonlat(
    lon: Sequence[float] | np.ndarray,
    lat: Sequence[float] | np.ndarray,
    metric_epsg: int = DEFAULT_METRIC_EPSG,
) -> np.ndarray:
    """Project WGS84 lon/lat to metres, returning an ``(n, 2)`` array of x, y."""
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    if lon_arr.shape != lat_arr.shape:
        raise ValueError("lon and lat must have the same shape")
    transformer = make_transformer(metric_epsg)
    x, y = transformer.transform(lon_arr, lat_arr)
    points = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    if not np.isfinite(points).all():
        raise ValueError("Projection produced non-finite coordinates")
    return points


def haversine_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Great-circle distance in metres (used to validate the projection)."""
    radius = 6_371_008.8
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def build_tree(points: np.ndarray) -> cKDTree:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an (n, 2) array")
    if len(points) == 0:
        raise ValueError("cannot build a KD-tree from zero points")
    return cKDTree(points)


def count_site_months(
    site_xy: np.ndarray,
    crash_xy: np.ndarray,
    crash_month_index: np.ndarray,
    n_months: int,
    radius_m: float,
    tree: cKDTree | None = None,
) -> np.ndarray:
    """Count distinct crashes within ``radius_m`` of each site, by month.

    Returns an integer matrix of shape ``(n_sites, n_months)``.  A crash is
    counted at most once per site, because each crash index appears once in the
    KD-tree result for that site.
    """
    site_xy = np.asarray(site_xy, dtype=float)
    crash_month_index = np.asarray(crash_month_index, dtype=int)
    if len(site_xy) == 0:
        return np.zeros((0, n_months), dtype=np.int32)
    if crash_month_index.min(initial=0) < 0:
        raise ValueError("crash_month_index contains -1 (crash outside the month window)")

    tree = tree or build_tree(crash_xy)
    neighbours = tree.query_ball_point(site_xy, r=float(radius_m))
    counts = np.zeros((len(site_xy), n_months), dtype=np.int32)
    for row, indices in enumerate(neighbours):
        if indices:
            idx = crash_month_index[np.asarray(indices, dtype=int)]
            counts[row] = np.bincount(idx, minlength=n_months)
    return counts


def assignment_pairs(
    site_xy: np.ndarray,
    crash_xy: np.ndarray,
    radius_m: float,
    tree: cKDTree | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(crash_index, site_index)`` for every site-crash assignment.

    Unlike :func:`count_site_months`, this keeps the long form so that repeated
    attribution of one crash to several sites can be measured.
    """
    site_xy = np.asarray(site_xy, dtype=float)
    if len(site_xy) == 0:
        empty = np.array([], dtype=int)
        return empty, empty
    tree = tree or build_tree(crash_xy)
    neighbours = tree.query_ball_point(site_xy, r=float(radius_m))
    crash_idx: list[np.ndarray] = []
    site_idx: list[np.ndarray] = []
    for site, indices in enumerate(neighbours):
        if indices:
            arr = np.asarray(indices, dtype=int)
            crash_idx.append(arr)
            site_idx.append(np.full(len(arr), site, dtype=int))
    if not crash_idx:
        empty = np.array([], dtype=int)
        return empty, empty
    return np.concatenate(crash_idx), np.concatenate(site_idx)


@dataclass
class AttributionDiagnostics:
    n_crashes: int
    n_assignments: int
    n_shared_crashes: int
    shared_share: float
    max_sites_per_crash: int
    histogram: dict[int, int]

    def to_dict(self) -> dict[str, float | int | dict[int, int]]:
        return {
            "n_crashes": self.n_crashes,
            "n_assignments": self.n_assignments,
            "n_shared_crashes": self.n_shared_crashes,
            "shared_share": round(self.shared_share, 4),
            "max_sites_per_crash": self.max_sites_per_crash,
            "histogram": self.histogram,
        }


def attribution_diagnostics(
    crash_index: np.ndarray,
    n_crashes: int,
) -> AttributionDiagnostics:
    """Summarise how many distinct crashes each site set receives."""
    if len(crash_index) == 0:
        return AttributionDiagnostics(n_crashes, 0, 0, 0.0, 0, {})
    counts = np.bincount(np.asarray(crash_index, dtype=int), minlength=n_crashes)
    observed = counts[counts > 0]
    histogram = {int(k): int(v) for k, v in zip(*np.unique(observed, return_counts=True))}
    return AttributionDiagnostics(
        n_crashes=n_crashes,
        n_assignments=int(len(crash_index)),
        n_shared_crashes=int((observed > 1).sum()),
        shared_share=float((observed > 1).sum()) / max(len(observed), 1),
        max_sites_per_crash=int(observed.max(initial=0)),
        histogram=histogram,
    )


def overlap_pairs(points: np.ndarray, min_distance_m: float) -> np.ndarray:
    """Pairs of points closer than ``min_distance_m`` (catchment overlap)."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.empty((0, 2), dtype=int)
    tree = cKDTree(points)
    return np.asarray(sorted(tree.query_pairs(r=float(min_distance_m))), dtype=int).reshape(-1, 2)


def connected_components(n_nodes: int, pairs: np.ndarray) -> np.ndarray:
    """Deterministic union-find labels for an undirected pair list.

    Components are labelled by their smallest node index, so the output is
    reproducible across runs.
    """
    parent = list(range(n_nodes))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in np.asarray(pairs, dtype=int).reshape(-1, 2):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    labels = np.array([find(i) for i in range(n_nodes)], dtype=int)
    _, canonical = np.unique(labels, return_inverse=True)
    return canonical.astype(int)


def grid_points(
    bounds_xy: tuple[float, float, float, float],
    spacing_m: float,
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Regular lattice of candidate comparison sites in projected metres."""
    xmin, ymin, xmax, ymax = bounds_xy
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    xs = np.arange(xmin + offset[0], xmax, spacing_m)
    ys = np.arange(ymin + offset[1], ymax, spacing_m)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def distances_to_nearest(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest reference point."""
    if len(reference) == 0:
        return np.full(len(points), np.inf)
    tree = cKDTree(np.asarray(reference, dtype=float))
    distance, _ = tree.query(np.asarray(points, dtype=float), k=1)
    return np.asarray(distance, dtype=float)


def mean_slope(values: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Pre-period mean level and least-squares slope for each row.

    ``values`` is ``(n_units, n_months)``.  The slope uses a centred linear time
    index so that it is independent of the window offset.
    """
    segment = np.asarray(values, dtype=float)[:, lo:hi]
    if segment.shape[1] < 2:
        raise ValueError("need at least two months to compute a slope")
    time = np.arange(segment.shape[1], dtype=float) - (segment.shape[1] - 1) / 2
    return segment.mean(axis=1), (segment * time).sum(axis=1) / (time**2).sum()
