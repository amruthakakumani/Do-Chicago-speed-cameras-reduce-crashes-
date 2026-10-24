#!/usr/bin/env python
"""Reproduce the original notebook's published estimates exactly.

This script deliberately re-implements the *original* logic — flat lon/lat
projection, citywide matching, no overlap handling, 2015 start — so that a
reviewer can confirm the saved notebook outputs against the current portal
snapshot.  It is an audit tool, not part of the upgraded analysis.

Usage::

    python scripts/verify_replication.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree

from chicago_cameras.acquisition import read_raw
from chicago_cameras.config import load_config, path_for


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = path_for(config, "raw")
    cameras = read_raw(raw_dir / "cameras.csv.gz")
    crashes = read_raw(raw_dir / "crashes.csv.gz")

    # --- Original cell 4/7 -------------------------------------------------
    keep = ["crash_record_id", "crash_date", "latitude", "longitude", "injuries_total"]
    crashes = crashes[keep].copy()
    frame = crashes.copy()
    frame["crash_date"] = pd.to_datetime(frame["crash_date"], errors="coerce")
    for column in ("latitude", "longitude"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["crash_date", "latitude", "longitude"])
    frame = frame.loc[frame["crash_date"] >= "2015-01-01"]
    frame = frame.rename(columns={"crash_date": "date", "latitude": "lat", "longitude": "lon"})
    frame = frame.loc[frame["lat"].between(41.6, 42.05) & frame["lon"].between(-87.95, -87.5)]
    frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp()
    months = pd.date_range(
        "2015-01-01", frame["month"].max() - pd.DateOffset(months=2), freq="MS"
    )
    frame = frame.loc[frame["month"].isin(months)].reset_index(drop=True)
    T = len(months)

    cams = cameras.copy()
    cams["go_live"] = pd.to_datetime(cams["go_live_date"], errors="coerce")
    cams["lat"] = pd.to_numeric(cams["latitude"], errors="coerce")
    cams["lon"] = pd.to_numeric(cams["longitude"], errors="coerce")
    cams = (
        cams.dropna(subset=["go_live", "lat", "lon"])
        .drop_duplicates(subset=["lat", "lon"])
        .reset_index(drop=True)
    )
    cams["go_month"] = cams["go_live"].dt.to_period("M").dt.to_timestamp()
    cams["go_idx"] = months.get_indexer(cams["go_month"])

    # --- Original cell 8: flat projection ----------------------------------
    LAT0, LON0 = 41.85, -87.65

    def to_xy(lat, lon):
        x = (np.asarray(lon) - LON0) * 111_320 * np.cos(np.radians(LAT0))
        y = (np.asarray(lat) - LAT0) * 110_574
        return np.column_stack([x, y])

    crash_xy = to_xy(frame["lat"], frame["lon"])
    crash_tree = cKDTree(crash_xy)
    crash_month_idx = months.get_indexer(frame["month"])
    cam_xy = to_xy(cams["lat"], cams["lon"])

    def monthly_counts(sites_xy, radius=250):
        neighbours = crash_tree.query_ball_point(sites_xy, r=radius)
        out = np.zeros((len(sites_xy), T), dtype=np.int32)
        for i, idx in enumerate(neighbours):
            if idx:
                out[i] = np.bincount(crash_month_idx[idx], minlength=T)
        return out

    Y_cam = monthly_counts(cam_xy)

    # --- Original cell 9 ----------------------------------------------------
    cam_tree = cKDTree(cam_xy)
    xmin, ymin = crash_xy.min(0)
    xmax, ymax = crash_xy.max(0)
    gx, gy = np.meshgrid(np.arange(xmin, xmax, 500), np.arange(ymin, ymax, 500))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    grid = grid[cam_tree.query(grid)[0] > 600]
    Y_grid = monthly_counts(grid)
    busy = Y_grid.sum(1) >= 100
    Y_ctl = Y_grid[busy]

    # --- Original cell 10 ---------------------------------------------------
    eligible = cams.index[(cams.go_idx >= 24) & (cams.go_idx + 36 <= T)].to_numpy()

    def mean_slope(Y, lo, hi):
        seg = Y[:, lo:hi].astype(float)
        t = np.arange(hi - lo) - (hi - lo - 1) / 2
        return seg.mean(1), (seg * t).sum(1) / (t**2).sum()

    rel = np.arange(-24, 36)
    pieces = []
    for c in eligible:
        g = int(cams.at[c, "go_idx"])
        m_t, s_t = mean_slope(Y_cam[[c]], g - 24, g)
        m_c, s_c = mean_slope(Y_ctl, g - 24, g)
        dist = np.abs(m_c - m_t) / m_c.std() + np.abs(s_c - s_t) / (s_c.std() + 1e-9)
        best = np.argsort(dist)[:5]
        cols = np.arange(g - 24, g + 36)
        pieces.append(
            pd.DataFrame({"stack": c, "unit": f"T{c}", "treated": 1, "rel": rel, "y": Y_cam[c, cols]})
        )
        for j in best:
            pieces.append(
                pd.DataFrame(
                    {"stack": c, "unit": f"C{j}_s{c}", "treated": 0, "rel": rel, "y": Y_ctl[j, cols]}
                )
            )
    panel = pd.concat(pieces, ignore_index=True)
    panel["post"] = (panel["rel"] >= 0).astype(int)
    panel["tp"] = panel["treated"] * panel["post"]

    # --- Original cell 12 ---------------------------------------------------
    cluster = {"groups": panel["stack"].values}
    ols = smf.ols("y ~ treated + post + tp", data=panel).fit(
        cov_type="cluster", cov_kwds=cluster
    )
    pois = smf.glm(
        "y ~ treated + post + tp", data=panel, family=sm.families.Poisson()
    ).fit(cov_type="cluster", cov_kwds=cluster)

    ols_low, ols_high = ols.conf_int().loc["tp"]
    po_low, po_high = pois.conf_int().loc["tp"]
    po_pct = (np.exp(pois.params["tp"]) - 1) * 100
    po_low_pct = (np.exp(po_low) - 1) * 100
    po_high_pct = (np.exp(po_high) - 1) * 100

    print("=== Exact replication of the original notebook ===")
    print(f"crashes kept          {len(frame):,}")
    print(f"camera sample         {len(eligible)} of {len(cams)}")
    print(f"panel rows            {len(panel):,}")
    print(f"pre means             twins {panel.loc[(panel.post==0)&(panel.treated==0),'y'].mean():.3f}"
          f" | cameras {panel.loc[(panel.post==0)&(panel.treated==1),'y'].mean():.3f}")
    print()
    print(f"OLS      {ols.params['tp']:+.4f} [{ols_low:+.4f}, {ols_high:+.4f}]  p={ols.pvalues['tp']:.4f}")
    print(f"Poisson  {po_pct:+.1f}% [{po_low_pct:+.1f}%, {po_high_pct:+.1f}%]  p={pois.pvalues['tp']:.4f}")
    print()
    print("Published in the notebook: OLS -0.356 [-0.711, -0.000] p=0.0497;"
          " Poisson -7.3% [-13.6%, -0.6%] p=0.0342")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
