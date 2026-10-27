"""Coverage-aware diagnostics for Chicago crash reporting.

The Chicago Data Portal description is explicit: E-Crash data exist for some
police districts in 2015, but citywide reporting is not available until
September 2017.  This module treats that date as an observation-regime boundary,
not as an arbitrary sample cut.

The functions here do **not** infer completeness from crash counts alone.  They
join the portal's police-beat and notification fields, expose monthly and
district-level patterns, and leave the official metadata as the primary source
for the regime boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CITYWIDE_START = pd.Timestamp("2017-09-01")
VALID_DISTRICTS = set(range(1, 26))


@dataclass(frozen=True)
class CoverageConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    results_dir: Path = Path("results/bottom_up/coverage")
    citywide_start: pd.Timestamp = CITYWIDE_START
    reporting_lag_months: int = 2


def load_crash_coverage(config: CoverageConfig | None = None) -> pd.DataFrame:
    config = config or CoverageConfig()
    """Join the analysis snapshot to current portal beat/notification fields.

    The crash snapshot remains authoritative for the modeled outcome.  The
    coverage extract is used only for audit columns; records added after the
    snapshot are retained in the audit frame and flagged, but never enter the
    modeling panel.
    """
    crashes = pd.read_csv(
        config.raw_dir / "crashes.csv.gz",
        usecols=[
            "crash_record_id",
            "crash_date",
            "latitude",
            "longitude",
            "injuries_total",
            "injuries_fatal",
            "injuries_incapacitating",
            "report_type",
        ],
    )
    crashes["crash_date"] = pd.to_datetime(crashes["crash_date"], errors="coerce")
    current = pd.read_parquet(config.raw_dir / "crash_coverage_fields.parquet")
    current["date_police_notified"] = pd.to_datetime(
        current["date_police_notified"], errors="coerce"
    )
    frame = crashes.merge(
        current[["crash_record_id", "beat_of_occurrence", "date_police_notified"]],
        on="crash_record_id",
        how="outer",
        indicator=True,
    )
    frame["in_snapshot"] = frame["_merge"].isin(["both", "left_only"])
    frame["in_current"] = frame["_merge"].isin(["both", "right_only"])
    frame = frame.drop(columns="_merge")
    # Records that arrived after the modeling snapshot have no crash_date in the
    # extract; use the police-notification date only for the coverage audit.
    frame["crash_date"] = frame["crash_date"].fillna(frame["date_police_notified"])
    frame["beat"] = pd.to_numeric(frame["beat_of_occurrence"], errors="coerce")
    frame["district"] = frame["beat"] // 100
    frame.loc[~frame["district"].isin(VALID_DISTRICTS), "district"] = np.nan
    frame["month"] = frame["crash_date"].dt.to_period("M").dt.to_timestamp()
    frame["year"] = frame["crash_date"].dt.year
    frame["injury"] = (pd.to_numeric(frame["injuries_total"], errors="coerce") > 0).astype(int)
    frame["severe"] = (
        (pd.to_numeric(frame["injuries_fatal"], errors="coerce") > 0)
        | (pd.to_numeric(frame["injuries_incapacitating"], errors="coerce") > 0)
    ).astype(int)
    frame["delay_days"] = (
        frame["date_police_notified"] - frame["crash_date"]
    ).dt.total_seconds() / 86400.0
    frame["period"] = np.select(
        [
            frame["month"] < config.citywide_start,
            frame["month"] <= frame["month"].max() - pd.DateOffset(months=config.reporting_lag_months),
        ],
        ["pre_citywide", "citywide"],
        default="recent_incomplete",
    )
    frame.loc[frame["in_snapshot"].eq(False), "period"] = "post_snapshot"
    return frame


def monthly_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Monthly counts and composition, with explicit regime labels."""
    work = frame.loc[frame["in_snapshot"]].copy()
    table = (
        work.groupby("month", as_index=False)
        .agg(
            crashes=("crash_record_id", "size"),
            injury_crashes=("injury", "sum"),
            severe_crashes=("severe", "sum"),
            injuries=("injuries_total", "sum"),
            districts=("district", "nunique"),
            beats=("beat", "nunique"),
            missing_beat=("beat", lambda series: int(series.isna().sum())),
            period=("period", "first"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    table["injury_share"] = table["injury_crashes"] / table["crashes"]
    table["severe_share"] = table["severe_crashes"] / table["crashes"]
    table["month"] = pd.to_datetime(table["month"])
    return table


def district_month_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.loc[frame["in_snapshot"] & frame["district"].notna()].copy()
    table = (
        work.groupby(["district", "month"], as_index=False)
        .agg(
            crashes=("crash_record_id", "size"),
            injury_crashes=("injury", "sum"),
            beats=("beat", "nunique"),
        )
        .sort_values(["district", "month"])
        .reset_index(drop=True)
    )
    table["injury_share"] = table["injury_crashes"] / table["crashes"]
    return table


def beat_month_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.loc[frame["in_snapshot"] & frame["beat"].notna()].copy()
    return (
        work.groupby(["district", "beat", "month"], as_index=False)
        .agg(crashes=("crash_record_id", "size"), injury_crashes=("injury", "sum"))
        .sort_values(["district", "beat", "month"])
        .reset_index(drop=True)
    )


def coverage_summary(frame: pd.DataFrame, config: CoverageConfig | None = None) -> dict[str, Any]:
    config = config or CoverageConfig()
    snapshot = frame.loc[frame["in_snapshot"]]
    current_new = frame.loc[~frame["in_snapshot"]]
    removed_from_current = int((snapshot["in_snapshot"] & ~snapshot["in_current"]).sum())
    monthly = monthly_coverage(frame)
    before = monthly.loc[monthly["month"].between("2016-09-01", "2017-08-01")]
    after = monthly.loc[monthly["month"].between("2017-09-01", "2018-08-01")]
    delay = frame["delay_days"].dropna()
    missing_coords = snapshot[["latitude", "longitude"]].isna().any(axis=1)
    zero_coords = (snapshot["latitude"].eq(0) | snapshot["longitude"].eq(0))
    outside = (
        ~snapshot["latitude"].between(41.60, 42.06)
        | ~snapshot["longitude"].between(-87.95, -87.50)
    )
    return {
        "source": "Chicago Data Portal Traffic Crashes - Crashes (85ca-t3if)",
        "official_statement": (
            "E-Crash data are available for some police districts in 2015, but "
            "citywide data are not available until September 2017. The portal "
            "description also notes that records are added or amended after reports "
            "are finalized."
        ),
        "citywide_start": str(config.citywide_start.date()),
        "snapshot_rows": len(snapshot),
        "snapshot_unique_ids": int(snapshot["crash_record_id"].nunique()),
        "duplicate_ids": int(snapshot["crash_record_id"].duplicated().sum()),
        "current_portal_rows": int(frame["in_current"].sum()),
        "new_after_snapshot": len(current_new),
        "removed_from_current": removed_from_current,
        "snapshot_first": str(snapshot["crash_date"].min()),
        "snapshot_last": str(snapshot["crash_date"].max()),
        "missing_coordinates": int(missing_coords.sum()),
        "zero_coordinates": int(zero_coords.sum()),
        "outside_bbox_coordinates": int(outside.sum()),
        "invalid_district_or_beat": int(snapshot["district"].isna().sum()),
        "amended_report_type": int(snapshot["report_type"].eq("AMENDED").sum()),
        "notification_delay_days": {
            "n": int(delay.size),
            "median": float(delay.median()),
            "p95": float(delay.quantile(0.95)),
            "p99": float(delay.quantile(0.99)),
            "share_gt_30d": float((delay > 30).mean()),
            "share_gt_90d": float((delay > 90).mean()),
            "missing": int(frame["date_police_notified"].isna().sum()),
        },
        "monthly_regime": {
            "pre_citywide_mean_crashes": float(before["crashes"].mean()),
            "post_citywide_mean_crashes": float(after["crashes"].mean()),
            "pre_citywide_mean_injury_share": float(before["injury_share"].mean()),
            "post_citywide_mean_injury_share": float(after["injury_share"].mean()),
            "aug_2017_crashes": int(monthly.loc[monthly["month"].eq("2017-08-01"), "crashes"].iloc[0]),
            "sep_2017_crashes": int(monthly.loc[monthly["month"].eq("2017-09-01"), "crashes"].iloc[0]),
        },
        "districts_present": sorted(
            int(value) for value in snapshot["district"].dropna().unique()
        ),
        "n_districts_present": int(snapshot["district"].nunique()),
        "n_beats_present": int(snapshot.loc[snapshot["district"].notna(), "beat"].nunique()),
        "recent_incomplete_months": [
            str(value.date()) for value in monthly.loc[monthly["period"].eq("recent_incomplete"), "month"]
        ],
        "notes": [
            "All 22 CPD districts have at least one record in every month from 2015-09 onward, so district presence alone cannot measure completeness.",
            "The official metadata is the primary evidence for the September 2017 regime boundary; monthly counts and injury composition are corroborating diagnostics, not proof of completeness.",
            "The portal does not expose a per-record finalization timestamp, so portal reporting delay cannot be reconstructed; the two-month lag is a conservative operational rule.",
        ],
    }


def write_coverage_outputs(config: CoverageConfig | None = None) -> dict[str, Any]:
    config = config or CoverageConfig()
    frame = load_crash_coverage(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    monthly = monthly_coverage(frame)
    district = district_month_coverage(frame)
    beat = beat_month_coverage(frame)
    summary = coverage_summary(frame, config)
    monthly.to_csv(config.results_dir / "monthly_coverage.csv", index=False)
    district.to_csv(config.results_dir / "district_month_coverage.csv", index=False)
    beat.to_csv(config.results_dir / "beat_month_coverage.csv", index=False)
    (config.results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    frame.to_parquet(config.processed_dir / "crash_coverage_audit.parquet", index=False)
    return summary


if __name__ == "__main__":
    print(json.dumps(write_coverage_outputs(), indent=2, sort_keys=True))
