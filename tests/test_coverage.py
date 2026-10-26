"""Coverage diagnostics: unit tests with synthetic records."""

from __future__ import annotations

import pandas as pd

from chicago_cameras.coverage import CITYWIDE_START, monthly_coverage


def test_monthly_coverage_labels_and_injury_share():
    frame = pd.DataFrame(
        {
            "crash_record_id": ["a", "b", "c", "d"],
            "month": pd.to_datetime(
                ["2017-08-01", "2017-09-01", "2017-09-01", "2026-10-01"]
            ),
            "in_snapshot": [True, True, True, False],
            "injury": [1, 0, 1, 0],
            "severe": [0, 0, 1, 0],
            "injuries_total": [1, 0, 2, 0],
            "district": [1, 1, 2, 2],
            "beat": [101, 101, 201, 201],
            "period": ["pre_citywide", "citywide", "citywide", "post_snapshot"],
        }
    )
    table = monthly_coverage(frame)
    assert len(table) == 2
    assert not table["month"].eq("2026-10-01").any()
    august = table.loc[table["month"].eq("2017-08-01")].iloc[0]
    september = table.loc[table["month"].eq("2017-09-01")].iloc[0]
    assert august["crashes"] == 1
    assert september["crashes"] == 2
    assert september["injury_share"] == 0.5
    assert CITYWIDE_START == pd.Timestamp("2017-09-01")
