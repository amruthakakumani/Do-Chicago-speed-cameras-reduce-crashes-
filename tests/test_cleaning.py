"""Tests for raw-table cleaning and the coverage guard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from chicago_cameras.cleaning import clean_cameras, clean_crashes, clean_violations

BBOX = {"lat_min": 41.6, "lat_max": 42.06, "lon_min": -87.95, "lon_max": -87.5}


def _camera_row(location_id: str, lat: float, lon: float, go_live: str) -> dict:
    return {
        "id": location_id,
        "location_id": location_id,
        "address": "x",
        "latitude": str(lat),
        "longitude": str(lon),
        "go_live_date": go_live,
    }


def test_camera_dedup_keeps_earliest_date() -> None:
    raw = pd.DataFrame(
        [
            _camera_row("A", 41.88, -87.70, "2020-05-01T00:00:00.000"),
            _camera_row("B", 41.88, -87.70, "2018-01-01T00:00:00.000"),
            _camera_row("C", 41.90, -87.65, "2019-01-01T00:00:00.000"),
        ]
    )
    cleaned, quality = clean_cameras(raw, BBOX)
    assert len(cleaned) == 2
    duplicate = cleaned.loc[cleaned["lat"].round(5) == 41.88].iloc[0]
    assert duplicate["go_live"].year == 2018
    assert quality.dropped["duplicate_camera_coordinates"] == 1
    assert cleaned["camera_uid"].is_unique


def test_camera_missing_coordinates_dropped() -> None:
    raw = pd.DataFrame([_camera_row("A", np.nan, -87.70, "2020-01-01T00:00:00.000")])
    cleaned, quality = clean_cameras(raw, BBOX)
    assert cleaned.empty
    assert quality.dropped["missing_coordinates"] == 1


def _crash_row(crash_id: str, date: str, lat: float = 41.88, lon: float = -87.70) -> dict:
    return {
        "crash_record_id": crash_id,
        "crash_date": date,
        "latitude": lat,
        "longitude": lon,
        "injuries_total": "1",
        "injuries_fatal": "0",
        "injuries_incapacitating": "0",
        "posted_speed_limit": "30",
        "crash_type": "INJURY AND / OR TOW DUE TO CRASH",
        "first_crash_type": "PEDESTRIAN",
        "num_units": "2",
        "report_type": "ON SCENE",
    }


def test_crash_coverage_guard_drops_sparse_leading_months() -> None:
    rows = [
        _crash_row("old1", "2014-06-01T10:00:00.000"),
        _crash_row("old2", "2014-07-01T10:00:00.000"),
    ]
    # 2015 onward, a dense and stable regime.
    for month in pd.date_range("2015-01-01", "2015-12-01", freq="MS"):
        for day in range(20):
            rows.append(
                _crash_row(
                    f"{month:%Y%m}_{day}",
                    f"{month + pd.Timedelta(days=day):%Y-%m-%d}T10:00:00.000",
                )
            )
    raw = pd.DataFrame(rows)
    cleaned, quality = clean_crashes(
        raw, BBOX, start_month="2013-01-01", reporting_lag_months=0
    )
    assert cleaned["month"].min() == pd.Timestamp("2015-01-01")
    assert quality.dropped["months_before_reliable_coverage"] == 2
    assert quality.extra["coverage_start"] == "2015-01-01"


def test_crash_severity_flags() -> None:
    raw = pd.DataFrame(
        [
            _crash_row("a", "2018-01-01T10:00:00.000"),
            {**_crash_row("b", "2018-02-01T10:00:00.000"), "injuries_total": "0",
             "injuries_incapacitating": "1"},
        ]
    )
    cleaned, _ = clean_crashes(raw, BBOX, start_month="2018-01-01", reporting_lag_months=0)
    by_id = cleaned.set_index("crash_record_id")
    assert by_id.loc["a", "injury_crash"] == 1
    assert by_id.loc["a", "severe_crash"] == 0
    assert by_id.loc["b", "severe_crash"] == 1


def test_violations_aggregate_to_camera_month() -> None:
    raw = pd.DataFrame(
        {
            "camera_id": ["CHI1", "CHI1", "CHI2"],
            "violation_date": ["2018-01-03", "2018-01-20", "2018-01-05"],
            "violations": ["3", "4", "1"],
        }
    )
    grouped, quality = clean_violations(raw)
    assert len(grouped) == 2
    assert grouped.loc[grouped["camera_id"] == "CHI1", "violations"].iloc[0] == 7
    assert quality.kept_rows == 2
