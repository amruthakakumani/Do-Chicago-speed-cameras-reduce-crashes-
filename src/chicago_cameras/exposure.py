"""Historical traffic-exposure audit for the Chicago AADT dataset.

The portal metadata says ADT data begin in October 2020 and that quality improves
from May 2026.  The current extract starts later than that claim.  This module
records what is actually present, builds a segment-by-month availability matrix,
and decides whether exposure can enter a 2017-2026 forecast without backfilling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExposureAuditConfig:
    processed_dir: Path = Path("data/processed")
    results_dir: Path = Path("results/bottom_up/exposure")
    quality_regime_start: str = "2026-05-01"


def build_exposure_audit(config: ExposureAuditConfig = ExposureAuditConfig()) -> dict[str, Any]:
    monthly = pd.read_parquet(config.processed_dir / "aadt_segment_month.parquet")
    segments = pd.read_parquet(config.processed_dir / "aadt_segments.parquet")
    monthly["month"] = pd.to_datetime(monthly["month"])
    monthly["mean_adt"] = pd.to_numeric(monthly["mean_adt"], errors="coerce")
    monthly["min_adt"] = pd.to_numeric(monthly["min_adt"], errors="coerce")
    monthly["max_adt"] = pd.to_numeric(monthly["max_adt"], errors="coerce")
    monthly = monthly.loc[monthly["mean_adt"].notna()].copy()
    all_months = pd.date_range(monthly["month"].min(), monthly["month"].max(), freq="MS")
    matrix = (
        monthly.pivot_table(index="segmentid", columns="month", values="mean_adt", aggfunc="mean")
        .reindex(columns=all_months)
    )

    by_month = (
        monthly.groupby("month", as_index=False)
        .agg(
            rows=("segmentid", "size"),
            segments=("segmentid", "nunique"),
            mean_adt=("mean_adt", "mean"),
            median_adt=("mean_adt", "median"),
            p99_adt=("mean_adt", lambda series: float(series.quantile(0.99))),
            negative_or_zero=("mean_adt", lambda series: int((series <= 0).sum())),
            over_100k=("mean_adt", lambda series: int((series > 100_000).sum())),
        )
        .sort_values("month")
    )
    by_month["share_of_segments"] = by_month["segments"] / len(matrix)

    observed_months = matrix.notna().sum(axis=1)
    first_month = matrix.apply(lambda row: row.first_valid_index(), axis=1)
    last_month = matrix.apply(lambda row: row.last_valid_index(), axis=1)
    persistence = pd.DataFrame(
        {
            "segmentid": matrix.index,
            "months_observed": observed_months.to_numpy(),
            "first_month": first_month.to_numpy(),
            "last_month": last_month.to_numpy(),
        }
    )
    quality_start = pd.Timestamp(config.quality_regime_start)
    pre_quality = monthly.loc[monthly["month"] < quality_start]
    post_quality = monthly.loc[monthly["month"] >= quality_start]

    # A nearest-camera-zone match is a *coverage* check, not a causal or
    # traffic-normalisation step.  It asks how many camera catchments have any
    # AADT segment within 250 m.
    camera_coverage = None
    camera_path = Path("data/processed/cameras.parquet")
    if camera_path.exists():
        from scipy.spatial import cKDTree

        cameras = pd.read_parquet(camera_path)
        segment_xy = segments[["midpointlon", "midpointlat"]].to_numpy(float)
        # Approximate metric conversion is sufficient for a 250 m coverage screen.
        xy = np.column_stack(
            [
                (segment_xy[:, 0] + 87.65) * 111_320 * np.cos(np.radians(41.85)),
                (segment_xy[:, 1] - 41.85) * 110_574,
            ]
        )
        camera_xy = np.column_stack(
            [
                (cameras["lon"].to_numpy(float) + 87.65) * 111_320 * np.cos(np.radians(41.85)),
                (cameras["lat"].to_numpy(float) - 41.85) * 110_574,
            ]
        )
        tree = cKDTree(xy)
        distance, _ = tree.query(camera_xy, k=1)
        camera_coverage = {
            "n_cameras": int(len(cameras)),
            "cameras_with_segment_within_250m": int((distance <= 250).sum()),
            "share_with_segment_within_250m": float((distance <= 250).mean()),
            "cameras_with_segment_within_1000m": int((distance <= 1000).sum()),
            "share_with_segment_within_1000m": float((distance <= 1000).mean()),
            "note": "UTM-like local approximation; coverage screen only, not an exposure merge.",
        }

    summary = {
        "source": "Chicago Data Portal Average Daily Traffic Counts (gc7y-n4xa)",
        "portal_statement": (
            "The portal description says data are available from October 2020 and "
            "that accuracy/consistency improve from May 2026. The current extract "
            "does not contain October 2020- June 2022 records."
        ),
        "actual_first_month": str(monthly["month"].min().date()),
        "actual_last_month": str(monthly["month"].max().date()),
        "n_months": int(len(all_months)),
        "n_rows": int(len(monthly)),
        "n_segments": int(monthly["segmentid"].nunique()),
        "mean_adt_quantiles": {
            "p01": float(monthly["mean_adt"].quantile(0.01)),
            "p50": float(monthly["mean_adt"].median()),
            "p99": float(monthly["mean_adt"].quantile(0.99)),
            "max": float(monthly["mean_adt"].max()),
        },
        "zero_or_negative_rows": int((monthly["mean_adt"] <= 0).sum()),
        "over_100k_rows": int((monthly["mean_adt"] > 100_000).sum()),
        "months_observed_per_segment": {
            "min": int(observed_months.min()),
            "p50": float(observed_months.median()),
            "p95": float(observed_months.quantile(0.95)),
            "max": int(observed_months.max()),
        },
        "coverage_by_month": by_month.to_dict(orient="records"),
        "quality_regime": {
            "pre_2026_05": {
                "rows": int(len(pre_quality)),
                "segments": int(pre_quality["segmentid"].nunique()),
                "mean_adt": float(pre_quality["mean_adt"].mean()),
            },
            "from_2026_05": {
                "rows": int(len(post_quality)),
                "segments": int(post_quality["segmentid"].nunique()),
                "mean_adt": float(post_quality["mean_adt"].mean()),
            },
        },
        "camera_zone_coverage": camera_coverage,
        "decision": {
            "use_in_final_forecast": False,
            "reason": (
                "No AADT before July 2022; only 2,525 segments; camera-derived counts "
                "are operational measurements and may respond to enforcement; merging "
                "current values backward would leak future exposure."
            ),
            "what_would_be_needed": (
                "A time-stamped, segment-stable historical count series covering the "
                "training period, ideally from an independent traffic-count program."
            ),
        },
    }
    config.results_dir.mkdir(parents=True, exist_ok=True)
    by_month.to_csv(config.results_dir / "aadt_monthly_coverage.csv", index=False)
    persistence.to_parquet(config.results_dir / "aadt_segment_persistence.parquet", index=False)
    matrix.to_parquet(config.results_dir / "aadt_availability_matrix.parquet")
    (config.results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(build_exposure_audit(), indent=2, sort_keys=True, default=str))
