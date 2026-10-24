"""Cleaning and validation of the raw Chicago Data Portal tables.

Design principles
-----------------
* Raw values are never overwritten; cleaning produces new frames plus an
  explicit data-quality record.
* Drops are counted and reported, never silent.
* Months with no crashes are genuine zeros only if the source data cover that
  month; partially reported months at the end of the series are removed using
  a configurable reporting lag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# The portal uses 0 to mean "posted speed limit unknown" in this dataset.
SPEED_LIMIT_UNKNOWN = 0


@dataclass
class QualityRecord:
    """Structured data-quality notes for one table."""

    name: str
    raw_rows: int
    kept_rows: int
    dropped: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.name,
            "raw_rows": self.raw_rows,
            "kept_rows": self.kept_rows,
            "dropped": dict(self.dropped),
            "notes": list(self.notes),
            **self.extra,
        }


def _count_drop(record: QualityRecord, reason: str, n: int) -> None:
    if n:
        record.dropped[reason] = record.dropped.get(reason, 0) + int(n)


def clean_cameras(
    raw: pd.DataFrame,
    bbox: dict[str, float],
) -> tuple[pd.DataFrame, QualityRecord]:
    """Normalise the Speed Camera Locations table.

    Returns one row per distinct physical camera location with a stable
    ``camera_uid``. Exact duplicate coordinates are collapsed; when duplicates
    disagree on the go-live date the earliest date is retained because the
    later record cannot represent a second device at the same coordinates.
    """
    record = QualityRecord(name="cameras", raw_rows=len(raw), kept_rows=0)

    frame = raw.copy()
    frame["go_live"] = pd.to_datetime(frame["go_live_date"], errors="coerce")
    frame["lat"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["lon"] = pd.to_numeric(frame["longitude"], errors="coerce")

    missing_date = frame["go_live"].isna()
    missing_coords = frame["lat"].isna() | frame["lon"].isna()
    outside = ~(
        frame["lat"].between(bbox["lat_min"], bbox["lat_max"])
        & frame["lon"].between(bbox["lon_min"], bbox["lon_max"])
    )
    _count_drop(record, "missing_go_live_date", missing_date.sum())
    _count_drop(record, "missing_coordinates", missing_coords.sum())
    _count_drop(record, "coordinates_outside_city_bbox", (outside & ~missing_coords).sum())

    frame = frame.loc[~(missing_date | missing_coords | outside)].copy()

    # Collapse exact duplicates. Rounded coordinates avoid float representation
    # differences between records that describe the same device.
    frame["_lat_r"] = frame["lat"].round(5)
    frame["_lon_r"] = frame["lon"].round(5)
    before = len(frame)
    frame = (
        frame.sort_values("go_live")
        .drop_duplicates(subset=["_lat_r", "_lon_r"], keep="first")
        .reset_index(drop=True)
    )
    _count_drop(record, "duplicate_camera_coordinates", before - len(frame))

    # Stable, human-readable identifiers.
    if "location_id" in frame.columns and frame["location_id"].notna().all():
        base = frame["location_id"].astype(str)
    else:  # pragma: no cover - defensive
        base = pd.Series(
            [f"CAM{i:04d}" for i in range(len(frame))], index=frame.index, dtype="object"
        )
    counts = base.groupby(base).cumcount()
    frame["camera_uid"] = np.where(counts == 0, base, base + "_" + counts.astype(str))
    if frame["camera_uid"].duplicated().any():  # pragma: no cover - defensive
        frame["camera_uid"] = [f"CAM{i:04d}" for i in range(len(frame))]

    frame["go_month"] = frame["go_live"].dt.to_period("M").dt.to_timestamp()
    frame["go_live_year"] = frame["go_live"].dt.year
    frame = frame.drop(columns=["_lat_r", "_lon_r"])

    record.kept_rows = len(frame)
    record.extra["go_live_year_counts"] = (
        frame["go_live_year"].value_counts().sort_index().to_dict()
    )
    record.extra["n_distinct_coordinates"] = int(
        frame[["lat", "lon"]].drop_duplicates().shape[0]
    )
    record.notes.append(
        "go_live_date is taken as the first month of enforcement; the portal does "
        "not expose a separate installation date or warning-period end date."
    )
    return frame, record


def clean_crashes(
    raw: pd.DataFrame,
    bbox: dict[str, float],
    start_month: str | pd.Timestamp,
    reporting_lag_months: int,
) -> tuple[pd.DataFrame, QualityRecord]:
    """Normalise the Traffic Crashes table and define the observed month window."""
    record = QualityRecord(name="crashes", raw_rows=len(raw), kept_rows=0)

    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["crash_date"], errors="coerce")
    frame["lat"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["lon"] = pd.to_numeric(frame["longitude"], errors="coerce")

    for column in (
        "injuries_total",
        "injuries_fatal",
        "injuries_incapacitating",
        "posted_speed_limit",
        "num_units",
    ):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")

    missing_date = frame["date"].isna()
    missing_coords = frame["lat"].isna() | frame["lon"].isna()
    outside = ~(
        frame["lat"].between(bbox["lat_min"], bbox["lat_max"])
        & frame["lon"].between(bbox["lon_min"], bbox["lon_max"])
    )
    _count_drop(record, "missing_crash_date", missing_date.sum())
    _count_drop(record, "missing_coordinates", missing_coords.sum())
    _count_drop(record, "coordinates_outside_city_bbox", (outside & ~missing_coords).sum())
    frame = frame.loc[~(missing_date | missing_coords | outside)].copy()

    before = len(frame)
    frame = frame.drop_duplicates(subset=["crash_record_id"], keep="first").reset_index(drop=True)
    _count_drop(record, "duplicate_crash_record_id", before - len(frame))

    frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp()
    last_data_month = frame["month"].max()
    last_analysis_month = (last_data_month - pd.DateOffset(months=reporting_lag_months)).normalize()
    start = pd.Timestamp(start_month).normalize()

    before_month_filter = len(frame)
    frame = frame.loc[frame["month"].between(start, last_analysis_month)].reset_index(drop=True)
    _count_drop(
        record,
        "outside_analysis_month_window",
        before_month_filter - len(frame),
    )

    # Coverage guard: the portal's crash table begins in 2015. If the configured
    # start predates real coverage, the early months would look like genuine
    # zeros. Drop leading months whose volume is below 10% of the median month.
    months = pd.date_range(frame["month"].min(), frame["month"].max(), freq="MS")
    monthly = frame.groupby("month").size().reindex(months, fill_value=0)
    median_month = float(monthly.median())
    coverage_start = months[0]
    if median_month > 0:
        covered = monthly.loc[monthly >= 0.10 * median_month]
        if len(covered):
            coverage_start = covered.index.min()
    if coverage_start > months[0]:
        dropped_early = int((frame["month"] < coverage_start).sum())
        _count_drop(record, "months_before_reliable_coverage", dropped_early)
        frame = frame.loc[frame["month"] >= coverage_start].reset_index(drop=True)
        months = pd.date_range(coverage_start, frame["month"].max(), freq="MS")
        monthly = frame.groupby("month").size().reindex(months, fill_value=0)
        record.notes.append(
            f"Leading months before {coverage_start.date()} were excluded because their "
            "crash volume was below 10% of the median month; treating them as observed "
            "zeros would be wrong."
        )

    frame["injury_crash"] = (frame["injuries_total"].fillna(0) > 0).astype("int8")
    frame["severe_crash"] = (
        (frame["injuries_fatal"].fillna(0) > 0)
        | (frame["injuries_incapacitating"].fillna(0) > 0)
    ).astype("int8")
    frame["posted_speed_limit"] = frame["posted_speed_limit"].replace(
        SPEED_LIMIT_UNKNOWN, np.nan
    )

    record.kept_rows = len(frame)
    months = pd.date_range(frame["month"].min(), frame["month"].max(), freq="MS")
    monthly = frame.groupby("month").size().reindex(months, fill_value=0)
    record.extra.update(
        {
            "first_crash_date": str(frame["date"].min()),
            "last_crash_date": str(frame["date"].max()),
            "last_source_month": str(last_data_month.date()),
            "last_analysis_month": str(last_analysis_month.date()),
            "coverage_start": str(months[0].date()),
            "analysis_months": len(months),
            "distinct_crash_ids": int(frame["crash_record_id"].nunique()),
            "injury_crash_share": round(float(frame["injury_crash"].mean()), 4),
            "severe_crash_share": round(float(frame["severe_crash"].mean()), 4),
            "monthly_counts": {str(k.date()): int(v) for k, v in monthly.items()},
            "months_below_half_median": [
                str(d.date()) for d, v in monthly.items() if v < 0.5 * monthly.median()
            ],
        }
    )
    record.notes.append(
        f"The last {reporting_lag_months} calendar months of source data are excluded "
        "because recent crash reports are still being entered."
    )
    record.notes.append(
        "Zero crashes in a site-month are treated as observed zeros: the portal table "
        "is a census of reported crashes for the analysis window."
    )
    return frame, record


def clean_violations(raw: pd.DataFrame) -> tuple[pd.DataFrame, QualityRecord]:
    """Aggregate the Speed Camera Violations table to camera-month totals."""
    record = QualityRecord(name="violations", raw_rows=len(raw), kept_rows=0)
    frame = raw.copy()
    frame["violation_date"] = pd.to_datetime(frame["violation_date"], errors="coerce")
    frame["violations"] = pd.to_numeric(frame["violations"], errors="coerce")

    bad = frame["violation_date"].isna() | frame["violations"].isna()
    _count_drop(record, "missing_date_or_violations", bad.sum())
    frame = frame.loc[~bad].copy()

    frame["month"] = frame["violation_date"].dt.to_period("M").dt.to_timestamp()
    grouped = (
        frame.groupby(["camera_id", "month"], as_index=False)
        .agg(violations=("violations", "sum"), observed_days=("violation_date", "nunique"))
        .sort_values(["camera_id", "month"])
        .reset_index(drop=True)
    )
    record.kept_rows = len(grouped)
    record.extra.update(
        {
            "n_cameras": int(grouped["camera_id"].nunique()),
            "first_month": str(grouped["month"].min().date()),
            "last_month": str(grouped["month"].max().date()),
        }
    )
    record.notes.append(
        "Violation counts measure enforcement output, not necessarily underlying "
        "speeding behaviour or exposure; they are analysed as a mechanism check only."
    )
    return grouped, record
