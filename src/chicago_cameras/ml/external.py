"""External, real-data features: Chicago weather and US federal holidays.

Only data that would genuinely be available at a forecast date are used:

* weather is **lagged** (previous month and prior three-month mean); the forecast
  month's own weather is never used;
* holidays are deterministic calendar events known in advance.

If the weather archive cannot be reached, the columns are created with missing
values and the run proceeds with a documented blocker.  No weather values are
invented.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import holidays as holidays_pkg
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)

CHICAGO_LAT = 41.8781
CHICAGO_LON = -87.6298
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    return session


def fetch_daily_weather(
    start: str,
    end: str,
    cache_dir: Path,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download daily Chicago weather from Open-Meteo and cache it verbatim."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / "weather_daily.csv.gz"
    provenance_path = cache_dir / "weather_provenance.json"

    if raw_path.exists() and not refresh:
        frame = pd.read_csv(raw_path, compression="gzip", parse_dates=["date"])
        provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
        return frame, provenance

    params = {
        "latitude": CHICAGO_LAT,
        "longitude": CHICAGO_LON,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_mean,precipitation_sum,snowfall_sum",
        "timezone": "America/Chicago",
    }
    started = datetime.now(UTC)
    try:
        response = _session().get(ARCHIVE_URL, params=params, timeout=90)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        LOGGER.warning("weather archive unavailable: %s", exc)
        return pd.DataFrame(columns=["date", "temp_c", "precip_mm", "snow_cm"]), {
            "available": False,
            "error": str(exc),
            "fetched_at_utc": started.isoformat(),
        }

    daily = payload.get("daily", {})
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(daily.get("time", [])),
            "temp_c": daily.get("temperature_2m_mean", []),
            "precip_mm": daily.get("precipitation_sum", []),
            "snow_cm": daily.get("snowfall_sum", []),
        }
    )
    if frame.empty:
        return frame, {"available": False, "error": "empty response"}

    frame.to_csv(raw_path, index=False, compression="gzip")
    checksum = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    provenance = {
        "available": True,
        "url": ARCHIVE_URL,
        "params": params,
        "rows": len(frame),
        "first_date": str(frame["date"].min().date()),
        "last_date": str(frame["date"].max().date()),
        "fetched_at_utc": started.isoformat(),
        "sha256": checksum,
        "cache_file": str(raw_path),
        "note": "Daily values are cached verbatim; monthly aggregation happens in features.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return frame, provenance


def monthly_weather(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily weather to calendar months."""
    if daily.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "temp_mean_c",
                "precip_mm",
                "snow_cm",
                "days_below_zero",
                "days_precip",
            ]
        )
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp()
    grouped = (
        frame.groupby("month")
        .agg(
            temp_mean_c=("temp_c", "mean"),
            precip_mm=("precip_mm", "sum"),
            snow_cm=("snow_cm", "sum"),
            days_below_zero=("temp_c", lambda s: int((s < 0).sum())),
            days_precip=("precip_mm", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    return grouped


def lagged_weather_features(monthly: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    """Weather features for month ``t`` built only from months strictly before ``t``.

    Columns are missing when no earlier observation exists; they are never filled
    with invented values.
    """
    base = pd.DataFrame({"month": pd.DatetimeIndex(months)})
    if monthly.empty:
        for column in (
            "temp_lag1",
            "precip_lag1",
            "snow_lag1",
            "temp_mean_3",
            "precip_mean_3",
            "snow_mean_3",
        ):
            base[column] = np.nan
        return base

    weather = monthly.sort_values("month").copy()
    weather["month"] = pd.to_datetime(weather["month"])
    shifted = weather.set_index("month").shift(1)

    lag1 = shifted[["temp_mean_c", "precip_mm", "snow_cm"]].rename(
        columns={
            "temp_mean_c": "temp_lag1",
            "precip_mm": "precip_lag1",
            "snow_cm": "snow_lag1",
        }
    )
    rolling3 = (
        shifted[["temp_mean_c", "precip_mm", "snow_cm"]]
        .rolling(3, min_periods=1)
        .mean()
        .rename(
            columns={
                "temp_mean_c": "temp_mean_3",
                "precip_mm": "precip_mean_3",
                "snow_cm": "snow_mean_3",
            }
        )
    )
    features = base.merge(lag1, left_on="month", right_index=True, how="left")
    features = features.merge(rolling3, left_on="month", right_index=True, how="left")
    return features


def monthly_holidays(months: pd.DatetimeIndex) -> pd.DataFrame:
    """Count US federal holidays in each month (deterministic calendar feature)."""
    years = sorted({int(month.year) for month in months})
    calendar = holidays_pkg.US(years=years)
    counts: dict[pd.Timestamp, int] = {}
    for month in months:
        start = pd.Timestamp(month)
        end = start + pd.offsets.MonthEnd(1)
        counts[start] = sum(1 for day in calendar if start <= pd.Timestamp(day) <= end)
    return pd.DataFrame({"month": list(counts), "n_holidays": list(counts.values())})
