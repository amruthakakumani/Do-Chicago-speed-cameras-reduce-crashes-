"""Geographic grouped validation with spatial buffering.

Nearby camera and comparison catchments are dependent.  A naive random split
would put overlapping or adjacent sites on both sides.  V2 assigns locations to
2 km blocks, holds out entire blocks, and removes any training location within a
1 km buffer of a held-out location.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass
class GeographicSplit:
    train: pd.DataFrame
    holdout: pd.DataFrame
    info: dict


def assign_blocks(locations: pd.DataFrame, block_size_m: float = 2000.0) -> pd.DataFrame:
    frame = locations.copy()
    x0, y0 = frame["x"].min(), frame["y"].min()
    frame["block_x"] = np.floor((frame["x"] - x0) / block_size_m).astype(int)
    frame["block_y"] = np.floor((frame["y"] - y0) / block_size_m).astype(int)
    frame["block_id"] = frame["block_x"].astype(str) + "_" + frame["block_y"].astype(str)
    return frame


def geographic_split(
    locations: pd.DataFrame,
    holdout_fraction: float = 0.20,
    buffer_m: float = 1000.0,
    seed: int = 20261024,
    block_size_m: float = 2000.0,
) -> GeographicSplit:
    """Hold out whole 2 km blocks and buffer training locations away from them."""
    frame = assign_blocks(locations, block_size_m=block_size_m)
    blocks = np.array(sorted(frame["block_id"].unique()))
    rng = np.random.default_rng(seed)
    n_holdout = max(1, int(round(len(blocks) * holdout_fraction)))
    held_blocks = set(rng.choice(blocks, size=n_holdout, replace=False).tolist())

    holdout_mask = frame["block_id"].isin(held_blocks).to_numpy()
    holdout = frame.loc[holdout_mask].copy()
    if holdout.empty:
        raise ValueError("geographic split produced an empty holdout")

    holdout_tree = cKDTree(holdout[["x", "y"]].to_numpy(float))
    distance, _ = holdout_tree.query(frame[["x", "y"]].to_numpy(float), k=1)
    train_mask = (~holdout_mask) & (distance > buffer_m)
    train = frame.loc[train_mask].copy()
    if train.empty:
        raise ValueError("geographic split produced an empty training set")

    info = {
        "block_size_m": block_size_m,
        "n_blocks": int(len(blocks)),
        "n_holdout_blocks": int(len(held_blocks)),
        "n_train_locations": int(len(train)),
        "n_holdout_locations": int(len(holdout)),
        "holdout_fraction_actual": float(len(holdout) / len(frame)),
        "buffer_m": buffer_m,
        "min_train_to_holdout_distance_m": float(distance[train_mask].min()),
        "holdout_camera_locations": int((holdout["site_type"] == "camera").sum()),
        "holdout_control_locations": int((holdout["site_type"] == "control").sum()),
        "held_blocks": sorted(held_blocks),
    }
    return GeographicSplit(train=train, holdout=holdout, info=info)


def geographic_frames(
    frame: pd.DataFrame,
    split: GeographicSplit,
    train_last_month: str,
    validation_first: str,
    validation_last: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build temporally and geographically separated train/validation frames."""
    train_locations = set(split.train["location_id"])
    holdout_locations = set(split.holdout["location_id"])
    train = frame.loc[
        frame["location_id"].isin(train_locations)
        & (frame["month"] <= pd.Timestamp(train_last_month))
    ].copy()
    validation = frame.loc[
        frame["location_id"].isin(holdout_locations)
        & frame["month"].between(pd.Timestamp(validation_first), pd.Timestamp(validation_last))
    ].copy()
    if train.empty or validation.empty:
        raise ValueError("geographic temporal split produced an empty side")
    return train, validation
