"""Genuine recursive multi-month forecasting.

V1's evaluation is one-step-ahead with observed histories.  A deployment that
issues a 12-month forecast cannot observe the intervening crash counts.  This
module feeds each month's predictions back into the lag/rolling/neighbourhood
features so that the 12-month path uses no future outcomes.

Static and deterministic features (calendar, road network, camera operation,
climate normals) are known at the forecast issue date.  Future weather is taken
from expanding climate normals, never from the future weather archive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataset_v2 import _neighborhood_weights

LOGGER = logging.getLogger(__name__)

DYNAMIC_FEATURES = {
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "lag_24",
    "lag_36",
    "roll_mean_3",
    "roll_mean_6",
    "roll_mean_12",
    "roll_std_3",
    "roll_std_6",
    "roll_std_12",
    "expanding_mean",
    "expanding_std",
    "zero_share_12",
    "injury_share_12",
    "severe_share_12",
    "trend_12",
    "trend_6",
    "same_month_mean_3y",
    "seasonal_ratio",
    "cv_12",
    "shock_lag1",
    "shock_lag3",
    "months_since_last_crash",
    "months_since_last_injury",
    "ever_crashed",
    "temp_lag1",
    "precip_lag1",
    "snow_lag1",
    "temp_mean_3",
    "precip_mean_3",
    "snow_mean_3",
    "city_crash_lag1",
    "city_crash_mean_3",
    "city_crash_mean_12",
    "city_crash_trend",
    "city_injury_share_12",
    "city_shock_lag1",
    "location_city_ratio_12",
    "neighbor_crash_lag1",
    "neighbor_crash_mean_12",
    "neighbor_crash_std_12",
    "neighbor_zero_share_12",
    "neighbor_injury_share_12",
    "neighbor_shock_lag1",
}


@dataclass
class RecursiveResult:
    predictions: pd.DataFrame
    metrics: dict


def _rolling(matrix: np.ndarray, pos: int, window: int, statistic: str) -> np.ndarray:
    start = max(0, pos - window)
    window_values = matrix[:, start:pos]
    if window_values.shape[1] == 0:
        return np.full(matrix.shape[0], np.nan)
    if statistic == "mean":
        return window_values.mean(axis=1)
    if statistic == "std":
        return window_values.std(axis=1, ddof=0)
    raise ValueError(f"unknown statistic: {statistic}")


def _mean_or_nan(values: list[float]) -> float:
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def recursive_forecast(
    bundle,
    booster,
    features: list[str],
    origin_month: str | pd.Timestamp = "2025-08-01",
    horizon: int = 12,
) -> RecursiveResult:
    """Forecast ``horizon`` months ahead for every location, recursively.

    The first forecast month is the month after ``origin_month``.  Actual crashes
    are available only through ``origin_month``; every later lag is a prediction.
    """
    frame = bundle.frame.copy()
    frame["month"] = pd.to_datetime(frame["month"])
    origin = pd.Timestamp(origin_month).to_period("M").to_timestamp()
    months = pd.DatetimeIndex(sorted(frame["month"].unique()))
    origin_pos = int(np.where(months == origin)[0][0])
    horizon = min(horizon, len(months) - origin_pos - 1)
    if horizon <= 0:
        raise ValueError("no future months available for recursive forecasting")

    locations = bundle.locations.reset_index(drop=True)
    location_ids = locations["location_id"].tolist()
    n_locations = len(location_ids)

    def matrix(column: str) -> np.ndarray:
        return (
            frame.pivot(index="location_id", columns="month", values=column)
            .reindex(index=location_ids, columns=months)
            .to_numpy(float)
        )

    actual_y = matrix("y_all")
    actual_injury = matrix("y_injury")
    actual_severe = matrix("y_severe")
    history_y = actual_y.copy()
    history_injury = actual_injury.copy()
    history_severe = actual_severe.copy()
    history_y[:, origin_pos + 1 :] = np.nan
    history_injury[:, origin_pos + 1 :] = np.nan
    history_severe[:, origin_pos + 1 :] = np.nan

    last_window = slice(max(0, origin_pos - 12), origin_pos + 1)
    observed_total = history_y[:, last_window].sum(axis=1)
    injury_rate = np.divide(
        history_injury[:, last_window].sum(axis=1),
        observed_total,
        out=np.zeros(n_locations),
        where=observed_total > 0,
    )
    severe_rate = np.divide(
        history_severe[:, last_window].sum(axis=1),
        observed_total,
        out=np.zeros(n_locations),
        where=observed_total > 0,
    )

    weights = _neighborhood_weights(locations)
    if {"temp_normal", "precip_normal", "snow_normal"} <= set(frame.columns):
        normals = (
            frame.groupby("month")[["temp_normal", "precip_normal", "snow_normal"]]
            .first()
            .sort_index()
        )
    else:
        # V1 has no explicit normals: use the historical mean lagged weather for
        # each calendar month, computed only from months up to the origin.
        history = frame.loc[frame["month"] <= origin].copy()
        by_month = history.groupby("month_of_year")[
            ["temp_lag1", "precip_lag1", "snow_lag1"]
        ].mean()
        month_frame = pd.DataFrame({"month": months, "month_of_year": months.month})
        normals = (
            month_frame.merge(by_month, left_on="month_of_year", right_index=True)
            .set_index("month")[
                ["temp_lag1", "precip_lag1", "snow_lag1"]
            ]
            .rename(
                columns={
                    "temp_lag1": "temp_normal",
                    "precip_lag1": "precip_normal",
                    "snow_lag1": "snow_normal",
                }
            )
            .sort_index()
        )

    static_features = [column for column in features if column not in DYNAMIC_FEATURES]
    unknown = set(features) - set(static_features) - DYNAMIC_FEATURES
    if unknown:
        raise ValueError(f"unclassified features for recursive forecasting: {sorted(unknown)}")

    records: list[pd.DataFrame] = []
    for step in range(1, horizon + 1):
        pos = origin_pos + step
        month = months[pos]
        static = (
            frame.loc[frame["month"] == month]
            .set_index("location_id")
            .reindex(location_ids)
        )
        if static["cluster"].isna().any():
            raise ValueError(f"missing static features for {month.date()}")
        X = static[features].copy()

        # History features from the recursive series.
        for lag in (1, 2, 3, 6, 12, 24, 36):
            X[f"lag_{lag}"] = history_y[:, pos - lag] if pos - lag >= 0 else np.nan
        for window in (3, 6, 12):
            X[f"roll_mean_{window}"] = _rolling(history_y, pos, window, "mean")
            X[f"roll_std_{window}"] = _rolling(history_y, pos, window, "std")
        before = history_y[:, :pos]
        X["expanding_mean"] = before.mean(axis=1)
        X["expanding_std"] = before.std(axis=1, ddof=0)
        zero_window = history_y[:, max(0, pos - 12) : pos]
        X["zero_share_12"] = (zero_window == 0).mean(axis=1)
        injury_window = history_injury[:, max(0, pos - 12) : pos].sum(axis=1)
        total_window = history_y[:, max(0, pos - 12) : pos].sum(axis=1)
        X["injury_share_12"] = np.divide(
            injury_window, total_window, out=np.full(n_locations, np.nan), where=total_window > 0
        )
        severe_window = history_severe[:, max(0, pos - 12) : pos].sum(axis=1)
        X["severe_share_12"] = np.divide(
            severe_window, total_window, out=np.full(n_locations, np.nan), where=total_window > 0
        )
        X["trend_12"] = X["lag_1"] / (X["roll_mean_12"] + 0.5)
        X["trend_6"] = X["lag_1"] / (X["roll_mean_6"] + 0.5)
        X["shock_lag1"] = X["lag_1"] - X["roll_mean_12"]
        X["shock_lag3"] = X[["lag_1", "lag_2", "lag_3"]].mean(axis=1) - X["roll_mean_12"]

        same_month = []
        for offset in (12, 24, 36):
            if pos - offset >= 0:
                same_month.append(history_y[:, pos - offset])
        X["same_month_mean_3y"] = (
            np.mean(np.column_stack(same_month), axis=1)
            if same_month
            else np.full(n_locations, np.nan)
        )
        X["seasonal_ratio"] = X["same_month_mean_3y"] / (X["expanding_mean"] + 0.5)
        X["cv_12"] = X["roll_std_12"] / (X["roll_mean_12"] + 0.5)

        months_since_crash = np.full(n_locations, np.nan)
        months_since_injury = np.full(n_locations, np.nan)
        ever_crashed = np.zeros(n_locations, dtype=int)
        for index in range(n_locations):
            hits = np.flatnonzero(history_y[index, :pos] > 0)
            if len(hits):
                months_since_crash[index] = pos - hits[-1]
                ever_crashed[index] = 1
            injury_hits = np.flatnonzero(history_injury[index, :pos] > 0)
            if len(injury_hits):
                months_since_injury[index] = pos - injury_hits[-1]
        X["months_since_last_crash"] = months_since_crash
        X["months_since_last_injury"] = months_since_injury
        X["ever_crashed"] = ever_crashed

        # Neighbourhood history, using predicted values for prior future months.
        weighted_y = np.asarray(weights @ history_y)
        weighted_injury = np.asarray(weights @ history_injury)
        weighted_zero = np.asarray(weights @ (history_y == 0).astype(float))
        X["neighbor_crash_lag1"] = weighted_y[:, pos - 1]
        n_start = max(0, pos - 12)
        X["neighbor_crash_mean_12"] = weighted_y[:, n_start:pos].mean(axis=1)
        X["neighbor_crash_std_12"] = weighted_y[:, n_start:pos].std(axis=1, ddof=0)
        X["neighbor_zero_share_12"] = weighted_zero[:, n_start:pos].mean(axis=1)
        neighbour_total = weighted_y[:, n_start:pos].sum(axis=1)
        X["neighbor_injury_share_12"] = np.divide(
            weighted_injury[:, n_start:pos].sum(axis=1),
            neighbour_total,
            out=np.full(n_locations, np.nan),
            where=neighbour_total > 0,
        )
        X["neighbor_shock_lag1"] = X["neighbor_crash_lag1"] - X["neighbor_crash_mean_12"]

        # Citywide history.
        city_y = np.nansum(history_y, axis=0)
        city_injury = np.nansum(history_injury, axis=0)
        city_lag1 = city_y[pos - 1]
        city_mean3 = city_y[max(0, pos - 3) : pos].mean()
        city_mean12 = city_y[max(0, pos - 12) : pos].mean()
        city_injury12 = city_injury[max(0, pos - 12) : pos].sum()
        city_total12 = city_y[max(0, pos - 12) : pos].sum()
        X["city_crash_lag1"] = city_lag1
        X["city_crash_mean_3"] = city_mean3
        X["city_crash_mean_12"] = city_mean12
        X["city_crash_trend"] = city_lag1 / (city_mean12 + 1.0)
        X["city_injury_share_12"] = city_injury12 / city_total12 if city_total12 > 0 else np.nan
        X["city_shock_lag1"] = city_lag1 - city_mean12
        X["location_city_ratio_12"] = X["roll_mean_12"] / (
            city_mean12 / n_locations + 0.5
        )

        # Future weather is unknown: use expanding same-month normals.
        def normal_for(offset: int, column: str, pos: int = pos) -> float:
            index = pos - offset
            if index < 0:
                return float("nan")
            month_key = months[index]
            if month_key not in normals.index:
                return float("nan")
            return float(normals.loc[month_key, column])

        X["temp_lag1"] = normal_for(1, "temp_normal")
        X["precip_lag1"] = normal_for(1, "precip_normal")
        X["snow_lag1"] = normal_for(1, "snow_normal")
        for target, column in (
            ("temp_mean_3", "temp_normal"),
            ("precip_mean_3", "precip_normal"),
            ("snow_mean_3", "snow_normal"),
        ):
            X[target] = _mean_or_nan([normal_for(offset, column) for offset in (1, 2, 3)])

        prediction = np.clip(booster.predict(X[features]), 0.0, None)
        history_y[:, pos] = prediction
        history_injury[:, pos] = prediction * injury_rate
        history_severe[:, pos] = prediction * severe_rate

        records.append(
            pd.DataFrame(
                {
                    "location_id": location_ids,
                    "month": month,
                    "horizon": step,
                    "actual": actual_y[:, pos],
                    "predicted": prediction,
                    "site_type": locations["site_type"].to_numpy(),
                    "camera_uid": locations["camera_uid"].to_numpy(),
                    "cluster": locations["cluster"].to_numpy(),
                    "lat": locations["lat"].to_numpy(),
                    "lon": locations["lon"].to_numpy(),
                }
            )
        )

    predictions = pd.concat(records, ignore_index=True)
    from .evaluate import regression_metrics

    overall = regression_metrics(predictions["actual"], predictions["predicted"])
    by_horizon = []
    for horizon_value, group in predictions.groupby("horizon", sort=True):
        metrics = regression_metrics(group["actual"], group["predicted"])
        metrics["horizon"] = int(horizon_value)
        by_horizon.append(metrics)
    return RecursiveResult(
        predictions=predictions,
        metrics={
            "origin_month": str(origin.date()),
            "horizon": horizon,
            "overall": overall,
            "by_horizon": by_horizon,
            "note": (
                "Recursive path: after the first forecast month, lags and neighbourhood "
                "features use predictions, not actuals. Future weather is taken from "
                "expanding climate normals. This is not the same estimand as the "
                "one-step-ahead evaluation."
            ),
        },
    )
