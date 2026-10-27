"""Study design construction: catchments, comparison sites, matching, panel.

This is the boundary between data and estimation.  Every design choice that can
affect identification is an explicit argument, which makes robustness
comparisons honest rather than ad hoc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geospatial import (
    assignment_pairs,
    attribution_diagnostics,
    count_site_months,
    distances_to_nearest,
    grid_points,
    overlap_pairs,
)
from .matching import MatchResult, match_controls
from .panel import build_panel, validate_panel

LOGGER = logging.getLogger(__name__)

Outcome = Literal["all", "injury", "severe"]


@dataclass
class DesignConfig:
    radius_m: float = 250
    buffer_m: float = 600
    grid_m: float = 500
    grid_offset_m: float = 0
    min_control_crashes: int = 100
    min_active_month_share: float = 0.40
    max_match_distance_m: float | None = 1000.0
    n_controls: int = 5
    pre_months: int = 24
    post_months: int = 36
    treatment_month_offset: int = 0
    warning_days: int = 0
    overlap_rule: Literal["drop_later", "keep_all"] = "drop_later"
    min_pre_months: int | None = None
    caliper: float | None = None


@dataclass
class DesignResult:
    panel: pd.DataFrame
    matches: MatchResult
    cameras: pd.DataFrame
    control_xy: np.ndarray
    control_ids: np.ndarray
    camera_counts: np.ndarray
    control_counts: np.ndarray
    camera_go_idx: dict[int, int]
    camera_go_month: dict[int, pd.Timestamp]
    validation: dict
    diagnostics: dict
    config: DesignConfig

    @property
    def n_cameras(self) -> int:
        return len(self.matches.matches)


class StudyData:
    """Cleaned data plus cached spatial computations."""

    def __init__(
        self,
        cameras: pd.DataFrame,
        crashes: pd.DataFrame,
        months: pd.DatetimeIndex,
        metric_epsg: int = 26916,
    ) -> None:
        if "x" not in cameras.columns or "y" not in cameras.columns:
            raise ValueError("cameras must already be projected (x, y)")
        if "x" not in crashes.columns or "y" not in crashes.columns:
            raise ValueError("crashes must already be projected (x, y)")
        self.cameras = cameras.reset_index(drop=True).copy()
        self.crashes = crashes.reset_index(drop=True).copy()
        self.months = pd.DatetimeIndex(months)
        self.metric_epsg = metric_epsg
        self._month_index = self.months.get_indexer(self.crashes["month"])
        if (self._month_index < 0).any():
            raise ValueError("crashes contain months outside the declared month window")
        self._camera_xy = self.cameras[["x", "y"]].to_numpy(float)
        self._crash_xy = self.crashes[["x", "y"]].to_numpy(float)
        self._count_cache: dict[tuple[float, str], np.ndarray] = {}
        self._site_cache: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}

        xs, ys = self._crash_xy[:, 0], self._crash_xy[:, 1]
        self.bounds_xy = (xs.min(), ys.min(), xs.max(), ys.max())

    def subset_start(self, start_month: str | pd.Timestamp) -> StudyData:
        """Return a copy restricted to months at or after ``start_month``.

        Used to reproduce the original study's 2015 start exactly, without
        re-cleaning the source tables.
        """
        start = pd.Timestamp(start_month).normalize()
        keep = self.crashes["month"] >= start
        crashes = self.crashes.loc[keep].copy()
        months = pd.date_range(start, self.months.max(), freq="MS")
        return StudyData(self.cameras.copy(), crashes, months, metric_epsg=self.metric_epsg)

    @property
    def n_months(self) -> int:
        return len(self.months)

    def camera_counts(self, radius_m: float, outcome: Outcome = "all") -> np.ndarray:
        key = (float(radius_m), outcome)
        if key not in self._count_cache:
            mask = self._outcome_mask(outcome)
            self._count_cache[key] = count_site_months(
                self._camera_xy,
                self._crash_xy[mask],
                self._month_index[mask],
                self.n_months,
                radius_m,
            )
        return self._count_cache[key]

    def counts_at(self, xy: np.ndarray, radius_m: float, outcome: Outcome = "all") -> np.ndarray:
        mask = self._outcome_mask(outcome)
        return count_site_months(
            xy,
            self._crash_xy[mask],
            self._month_index[mask],
            self.n_months,
            radius_m,
        )

    def control_sites(
        self,
        radius_m: float,
        buffer_m: float,
        grid_m: float,
        grid_offset_m: float,
        min_crashes: int,
        min_active_share: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Candidate comparison sites and their all-crash monthly counts."""
        key = (float(radius_m), float(buffer_m), float(grid_m) * 1e6 + float(grid_offset_m))
        if key in self._site_cache:
            return self._site_cache[key]

        points = grid_points(
            self.bounds_xy,
            spacing_m=grid_m,
            offset=(grid_offset_m, grid_offset_m),
        )
        distance = distances_to_nearest(points, self._camera_xy)
        points = points[distance > buffer_m]
        counts = self.counts_at(points, radius_m, "all")
        active_share = (counts > 0).mean(axis=1)
        keep = (counts.sum(axis=1) >= min_crashes) & (active_share >= min_active_share)
        points, counts = points[keep], counts[keep]
        self._site_cache[key] = (points, counts)
        return points, counts

    def _outcome_mask(self, outcome: Outcome) -> np.ndarray:
        if outcome == "all":
            return np.ones(len(self.crashes), dtype=bool)
        if outcome == "injury":
            return self.crashes["injury_crash"].to_numpy(bool)
        if outcome == "severe":
            return self.crashes["severe_crash"].to_numpy(bool)
        raise ValueError(f"unknown outcome: {outcome}")


def _treatment_offset_months(config: DesignConfig) -> int:
    warning_months = int(np.ceil(config.warning_days / 30.0)) if config.warning_days else 0
    return int(config.treatment_month_offset) + warning_months


def resolve_overlaps(
    cameras: pd.DataFrame,
    radius_m: float,
    eligible_mask: np.ndarray,
    rule: str,
) -> tuple[np.ndarray, dict[str, int]]:
    """Flag eligible cameras whose catchment is contaminated by an earlier camera.

    A camera whose 250 m catchment overlaps a camera that was already live is not
    a clean experiment: part of its outcome is already treated.  ``drop_later``
    keeps the earliest camera in each overlap chain; ``keep_all`` reproduces the
    original behaviour and is used only as a sensitivity check.
    """
    xy = cameras[["x", "y"]].to_numpy(float)
    pairs = overlap_pairs(xy, 2 * radius_m)
    keep = np.ones(len(cameras), dtype=bool)
    if rule == "keep_all" or len(pairs) == 0:
        return np.ones(len(cameras), dtype=bool), {
            "overlap_pairs": len(pairs),
            "dropped_for_overlap": 0,
        }

    if rule != "drop_later":
        raise ValueError(f"unknown overlap rule: {rule}")

    order = np.lexsort((cameras["camera_uid"].to_numpy(), cameras["go_live"].to_numpy()))
    rank = np.empty(len(cameras), dtype=int)
    rank[order] = np.arange(len(cameras))

    adjacency: dict[int, list[int]] = {i: [] for i in range(len(cameras))}
    for a, b in pairs:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))

    dropped = 0
    drop_distances: list[float] = []
    for node in range(len(cameras)):
        if not eligible_mask[node]:
            continue
        earlier = [n for n in adjacency[node] if rank[n] < rank[node]]
        if earlier:
            keep[node] = False
            dropped += 1
            drop_distances.append(
                float(min(np.linalg.norm(xy[node] - xy[n]) for n in earlier))
            )
    stats = {
        "overlap_pairs": len(pairs),
        "dropped_for_overlap": dropped,
    }
    if drop_distances:
        stats["overlap_drop_distance_m_median"] = float(np.median(drop_distances))
        stats["overlap_drop_distance_m_max"] = float(np.max(drop_distances))
        stats["overlap_drop_within_100m"] = int(sum(d <= 100 for d in drop_distances))
    return keep, stats


def build_design(
    data: StudyData,
    config: DesignConfig,
    outcome: Outcome = "all",
    camera_subset: np.ndarray | None = None,
) -> DesignResult:
    """Construct the matched, stacked panel for one design specification."""
    cameras = data.cameras.copy()
    offset = _treatment_offset_months(config)
    cameras["go_idx"] = data.months.get_indexer(cameras["go_month"]) + offset
    cameras["eligible_window"] = (cameras["go_idx"] >= config.pre_months) & (
        cameras["go_idx"] + config.post_months <= data.n_months
    )
    if config.min_pre_months is not None:
        cameras["eligible_window"] &= cameras["go_idx"] >= config.min_pre_months

    eligible = cameras["eligible_window"].to_numpy(bool).copy()
    if camera_subset is not None:
        eligible &= np.asarray(camera_subset, dtype=bool)

    keep_overlap, overlap_stats = resolve_overlaps(
        cameras, config.radius_m, eligible, config.overlap_rule
    )
    eligible &= keep_overlap
    if not eligible.any():
        raise ValueError(
            "no camera satisfies the event-window and overlap requirements; "
            "try a shorter pre-period or the keep_all overlap rule"
        )

    control_xy, _ = data.control_sites(
        radius_m=config.radius_m,
        buffer_m=config.buffer_m,
        grid_m=config.grid_m,
        grid_offset_m=config.grid_offset_m,
        min_crashes=config.min_control_crashes,
        min_active_share=config.min_active_month_share,
    )
    control_counts = data.counts_at(control_xy, config.radius_m, outcome)
    camera_counts = data.camera_counts(config.radius_m, outcome)

    camera_rows = np.flatnonzero(eligible)
    match_result = match_controls(
        camera_xy=cameras.loc[camera_rows, ["x", "y"]].to_numpy(float),
        camera_rows=camera_rows,
        camera_go_idx=cameras.loc[camera_rows, "go_idx"].to_numpy(int),
        Y_camera=camera_counts,
        control_xy=control_xy,
        Y_control=control_counts,
        pre_months=config.pre_months,
        n_controls=config.n_controls,
        max_distance_m=config.max_match_distance_m,
        caliper=config.caliper,
    )
    if match_result.n_cameras == 0:
        raise ValueError(
            "matching produced no cameras; check max_match_distance_m and the "
            "number of candidate comparison sites"
        )

    go_idx = dict(zip(camera_rows, cameras.loc[camera_rows, "go_idx"].to_numpy(int)))
    camera_uids = dict(zip(cameras.index, cameras["camera_uid"]))
    camera_go_month = dict(zip(cameras.index, cameras["go_month"]))
    control_ids = np.array([f"G{i:05d}" for i in range(len(control_xy))], dtype=object)

    panel = build_panel(
        Y_camera=camera_counts,
        Y_control=control_counts,
        matches=match_result.matches,
        camera_go_idx=go_idx,
        camera_uids=camera_uids,
        control_ids=control_ids,
        pre_months=config.pre_months,
        post_months=config.post_months,
        camera_go_month=camera_go_month,
    )
    validation = validate_panel(panel, config.pre_months, config.post_months)

    matched_rows = np.array(sorted(match_result.matches), dtype=int)
    cam_tree = cKDTree(cameras[["x", "y"]].to_numpy(float))
    treated_control_distance = distances_to_nearest(
        control_xy, cameras[["x", "y"]].to_numpy(float)
    )
    diagnostics = {
        "n_cameras_total": len(cameras),
        "n_cameras_in_window": int(cameras["eligible_window"].sum()),
        "n_cameras_dropped_overlap": int(
            (cameras["eligible_window"] & ~keep_overlap).sum()
        ),
        "n_cameras_matched": len(matched_rows),
        "n_control_candidates": len(control_xy),
        "n_controls_used_unique": len(np.unique(match_result.controls_used)),
        "n_controls_reused": int(match_result.n_multi_used_controls),
        "match_failures": {str(k): v for k, v in match_result.failures.items()},
        "control_min_distance_to_camera_m": float(treated_control_distance.min())
        if len(control_xy)
        else np.nan,
        "camera_xy_tree_size": int(cam_tree.n),
        **overlap_stats,
    }

    # Empirical check that treated and control catchments never overlap.
    if len(control_xy):
        cross = distances_to_nearest(control_xy, cameras[["x", "y"]].to_numpy(float))
        diagnostics["treated_control_catchment_overlap"] = bool(
            (cross <= 2 * config.radius_m).any()
        )
    else:  # pragma: no cover - defensive
        diagnostics["treated_control_catchment_overlap"] = False

    # Comparison sites on a 500 m lattice touch at 500 m but do not overlap in
    # area. Use a 1 m tolerance so that boundary adjacency is not reported as
    # overlap, while a genuinely overlapping pair is.
    control_overlap = overlap_pairs(control_xy, max(2 * config.radius_m - 1.0, 0.0))
    diagnostics["control_overlap_pairs"] = len(control_overlap)

    # Repeated attribution of one crash to several camera catchments.
    crash_idx, _ = assignment_pairs(
        cameras[["x", "y"]].to_numpy(float), data._crash_xy, config.radius_m
    )
    attribution = attribution_diagnostics(crash_idx, len(data.crashes))
    diagnostics["camera_attribution_all_cameras"] = attribution.to_dict()
    crash_idx_matched, _ = assignment_pairs(
        cameras.loc[matched_rows, ["x", "y"]].to_numpy(float),
        data._crash_xy,
        config.radius_m,
    )
    diagnostics["camera_attribution_matched"] = attribution_diagnostics(
        crash_idx_matched, len(data.crashes)
    ).to_dict()
    control_crash_idx, _ = assignment_pairs(control_xy, data._crash_xy, config.radius_m)
    diagnostics["control_attribution"] = attribution_diagnostics(
        control_crash_idx, len(data.crashes)
    ).to_dict()

    # Road-characteristic balance: median posted speed limit of crashes within
    # each catchment, for matched cameras vs the comparison sites actually used.
    speed = data.crashes["posted_speed_limit"].to_numpy(float)
    crash_tree = cKDTree(data._crash_xy)

    def _median_speed(points: np.ndarray) -> np.ndarray:
        neighbours = crash_tree.query_ball_point(points, config.radius_m)
        out = np.full(len(points), np.nan)
        for i, idx in enumerate(neighbours):
            if idx:
                values = speed[np.asarray(idx, dtype=int)]
                values = values[np.isfinite(values)]
                if len(values):
                    out[i] = float(np.median(values))
        return out

    camera_speed = _median_speed(cameras.loc[matched_rows, ["x", "y"]].to_numpy(float))
    used_control_rows = np.unique(match_result.controls_used)
    control_speed = _median_speed(control_xy[used_control_rows])
    both = np.concatenate([camera_speed, control_speed])
    both = both[np.isfinite(both)]
    pooled = float(np.std(both, ddof=0)) if len(both) > 1 else np.nan
    mean_gap = float(np.nanmean(camera_speed) - np.nanmean(control_speed)) if len(both) else np.nan
    if pooled is not None and np.isfinite(pooled) and pooled > 0 and np.isfinite(mean_gap):
        speed_smd: float | None = float(mean_gap / pooled)
    elif np.isfinite(mean_gap) and abs(mean_gap) < 1e-9:
        speed_smd = 0.0
    else:
        speed_smd = None
    diagnostics["speed_limit_balance"] = {
        "camera_median_mph": float(np.nanmedian(camera_speed)) if len(camera_speed) else np.nan,
        "control_median_mph": float(np.nanmedian(control_speed)) if len(control_speed) else np.nan,
        "standardised_mean_difference": speed_smd,
        "n_cameras_with_speed": int(np.isfinite(camera_speed).sum()),
        "n_controls_with_speed": int(np.isfinite(control_speed).sum()),
    }

    return DesignResult(
        panel=panel,
        matches=match_result,
        cameras=cameras,
        control_xy=control_xy,
        control_ids=control_ids,
        camera_counts=camera_counts,
        control_counts=control_counts,
        camera_go_idx=go_idx,
        camera_go_month=camera_go_month,
        validation=validation,
        diagnostics=diagnostics,
        config=config,
    )
