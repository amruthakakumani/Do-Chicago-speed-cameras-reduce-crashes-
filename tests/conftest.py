"""Shared synthetic fixtures. No real data and no network access are used."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261024)


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """A small stacked panel with a known ATT of -2 crashes per site-month."""
    rng = np.random.default_rng(0)
    rows = []
    for stack in range(6):
        for unit_offset in range(6):
            treated = int(unit_offset == 0)
            unit = f"{'T' if treated else 'C'}{stack}_{unit_offset}"
            for rel in range(-6, 6):
                base = 8.0 if treated else 5.0
                post_effect = -2.0 if (treated and rel >= 0) else 0.0
                y = max(0, round(rng.poisson(base + post_effect)))
                rows.append(
                    {
                        "stack": f"S{stack}",
                        "unit": unit,
                        "treated": treated,
                        "rel": rel,
                        "post": int(rel >= 0),
                        "tp": int(treated and rel >= 0),
                        "y": y,
                    }
                )
    panel = pd.DataFrame(rows)
    panel["quarter"] = np.floor_divide(panel["rel"], 3)
    return panel


@pytest.fixture
def camera_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "camera_uid": ["CHI001", "CHI002", "CHI003"],
            "location_id": ["CHI001", "CHI002", "CHI003"],
            "address": ["A", "B", "C"],
            "lat": [41.88, 41.90, 41.85],
            "lon": [-87.70, -87.65, -87.60],
            "go_live": pd.to_datetime(["2018-01-15", "2019-06-01", "2023-03-10"]),
            "go_month": pd.to_datetime(["2018-01-01", "2019-06-01", "2023-03-01"]),
            "x": [0.0, 2000.0, 6000.0],
            "y": [0.0, 1000.0, 0.0],
        }
    )
