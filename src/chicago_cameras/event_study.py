"""Event-study aggregation and persistence helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_percent(table: pd.DataFrame, columns: tuple[str, ...] = ("beta", "ci_low", "ci_high")) -> pd.DataFrame:
    """Convert log-point coefficients to percent changes for count models."""
    out = table.copy()
    for column in columns:
        if column in out.columns:
            out[f"{column}_pct"] = (np.exp(out[column]) - 1) * 100
    return out


def cumulative_effect(event_table: pd.DataFrame, months: int) -> float:
    """Cumulative ATT over the first ``months`` post-treatment months."""
    subset = event_table.loc[event_table["event_month"] < months, "att"]
    return float(subset.sum())


def persistence_summary(event_table: pd.DataFrame) -> dict[str, float]:
    """Average effect in early, middle and late parts of the post period."""
    post = event_table.loc[event_table["event_month"] >= 0]
    if post.empty:
        return {}
    n = len(post)
    thirds = np.array_split(np.arange(n), 3)
    return {
        "early": float(post.iloc[thirds[0]]["att"].mean()),
        "middle": float(post.iloc[thirds[1]]["att"].mean()),
        "late": float(post.iloc[thirds[2]]["att"].mean()),
    }


def annualised_change(event_table: pd.DataFrame) -> dict[str, float]:
    """Linear change in the event-time effect per 12 months (level scale)."""
    post = event_table.loc[event_table["event_month"] >= 0].copy()
    if len(post) < 6:
        return {}
    x = post["event_month"].to_numpy(float) / 12.0
    y = post["att"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope_per_year": float(slope), "intercept": float(intercept)}
