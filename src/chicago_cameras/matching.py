"""Matched comparison groups for the stacked event-study design."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geospatial import mean_slope

LOGGER = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Matched controls for each eligible camera."""

    matches: dict[int, np.ndarray] = field(default_factory=dict)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    failures: dict[int, str] = field(default_factory=dict)
    controls_used: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    @property
    def n_cameras(self) -> int:
        return len(self.matches)

    @property
    def n_multi_used_controls(self) -> int:
        if len(self.controls_used) == 0:
            return 0
        counts = np.bincount(self.controls_used)
        return int((counts > 1).sum())


def _standardise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = np.nanstd(values, ddof=0)
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return (values - np.nanmean(values)) / scale


def match_controls(
    camera_xy: np.ndarray,
    camera_rows: np.ndarray,
    camera_go_idx: np.ndarray,
    Y_camera: np.ndarray,
    control_xy: np.ndarray,
    Y_control: np.ndarray,
    pre_months: int,
    n_controls: int,
    max_distance_m: float,
    caliper: float | None = None,
) -> MatchResult:
    """Match each camera to nearby comparison sites with similar pre-trends.

    The match score follows the original study (absolute difference in
    standardised pre-period level plus absolute difference in standardised
    pre-period slope) but adds an optional caliper and explicit failure
    reporting, so that a camera with no credible comparison group is dropped
    rather than matched to a bad one.

    ``camera_rows`` indexes rows of ``Y_camera`` / ``camera_xy``; ``camera_go_idx``
    is the aligned month index of go-live for those rows.
    """
    camera_xy = np.asarray(camera_xy, dtype=float)
    control_xy = np.asarray(control_xy, dtype=float)
    if len(camera_rows) == 0:
        raise ValueError("no cameras supplied for matching")
    if len(control_xy) == 0:
        raise ValueError("no candidate comparison sites supplied")

    unrestricted = max_distance_m is None or not np.isfinite(max_distance_m)
    control_tree = cKDTree(control_xy)

    matches: dict[int, np.ndarray] = {}
    failures: dict[int, str] = {}
    rows: list[dict[str, object]] = []

    for position, camera_row in enumerate(camera_rows):
        go_idx = int(camera_go_idx[position])
        lo = go_idx - pre_months
        if lo < 0:
            failures[int(camera_row)] = "insufficient_pre_period"
            continue

        # Level and slope are computed on *each camera's own* pre-period window
        # for both the treated unit and every candidate comparison site.
        cam_level, cam_slope = mean_slope(Y_camera[[camera_row]], lo, go_idx)
        cam_level, cam_slope = float(cam_level[0]), float(cam_slope[0])
        control_level, control_slope = mean_slope(Y_control, lo, go_idx)

        nearby = (
            np.arange(len(control_xy), dtype=int)
            if unrestricted
            else np.asarray(
                control_tree.query_ball_point(camera_xy[position], max_distance_m), dtype=int
            )
        )
        if len(nearby) < n_controls:
            failures[int(camera_row)] = "too_few_nearby_controls"
            continue

        level_ref = float(np.nanmean(control_level[nearby]))
        slope_ref = float(np.nanmean(control_slope[nearby]))
        level_scale = float(np.nanstd(control_level[nearby])) or 1.0
        slope_scale = float(np.nanstd(control_slope[nearby])) or 1.0

        score = np.abs(control_level[nearby] - cam_level) / level_scale + np.abs(
            control_slope[nearby] - cam_slope
        ) / slope_scale
        order = np.argsort(score, kind="stable")
        chosen_local = order[:n_controls]
        chosen = nearby[chosen_local]

        best_score = float(score[chosen_local[0]])
        if caliper is not None and best_score > caliper:
            failures[int(camera_row)] = "caliper_not_met"
            continue

        distances = np.linalg.norm(control_xy[chosen] - camera_xy[position], axis=1)
        matches[int(camera_row)] = chosen

        rows.append(
            {
                "camera_row": int(camera_row),
                "go_idx": go_idx,
                "camera_pre_level": cam_level,
                "camera_pre_slope": cam_slope,
                "control_pre_level": float(np.nanmean(control_level[chosen])),
                "control_pre_slope": float(np.nanmean(control_slope[chosen])),
                "level_gap": cam_level - float(np.nanmean(control_level[chosen])),
                "slope_gap": cam_slope - float(np.nanmean(control_slope[chosen])),
                "mean_match_distance_m": float(np.mean(distances)),
                "max_match_distance_m": float(np.max(distances)),
                "match_score": best_score,
                "n_candidates_in_radius": int(len(nearby)),
                "level_ref": level_ref,
                "slope_ref": slope_ref,
                "level_scale": level_scale,
                "slope_scale": slope_scale,
            }
        )

    table = pd.DataFrame(rows)
    controls_used = (
        np.concatenate([matches[key] for key in sorted(matches)]).astype(int)
        if matches
        else np.array([], dtype=int)
    )
    return MatchResult(
        matches=matches,
        table=table,
        failures=failures,
        controls_used=controls_used,
    )
