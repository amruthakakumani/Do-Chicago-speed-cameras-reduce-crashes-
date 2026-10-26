"""Exposure chronology tests: never backfill pre-2022 AADT."""

from __future__ import annotations

import pandas as pd

from chicago_cameras.exposure import ExposureAuditConfig, build_exposure_audit


def test_exposure_audit_records_actual_start_and_disables_use(tmp_path):
    processed = tmp_path / "processed"
    results = tmp_path / "results"
    processed.mkdir()
    results.mkdir()
    monthly = pd.DataFrame(
        {
            "segmentid": ["s1", "s1", "s2", "s2"],
            "month": pd.to_datetime(["2022-07-01", "2022-08-01", "2022-07-01", "2022-08-01"]),
            "n_rows": [10, 10, 10, 10],
            "mean_adt": [1000.0, 1100.0, 2000.0, 2100.0],
            "min_adt": [900.0, 1000.0, 1900.0, 2000.0],
            "max_adt": [1100.0, 1200.0, 2100.0, 2200.0],
        }
    )
    segments = pd.DataFrame(
        {
            "segmentid": ["s1", "s2"],
            "midpointlat": [41.85, 41.86],
            "midpointlon": [-87.65, -87.64],
        }
    )
    monthly.to_parquet(processed / "aadt_segment_month.parquet", index=False)
    segments.to_parquet(processed / "aadt_segments.parquet", index=False)
    summary = build_exposure_audit(
        ExposureAuditConfig(processed_dir=processed, results_dir=results)
    )
    assert summary["actual_first_month"] == "2022-07-01"
    assert summary["n_segments"] == 2
    assert summary["decision"]["use_in_final_forecast"] is False
    assert "No AADT before July 2022" in summary["decision"]["reason"]
