"""Out-of-sample evaluation for the crash-count model.

Metrics are computed on the untouched test period only.  Baselines are evaluated
on exactly the same rows so that the comparison is honest and no improvement is
invented.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Poisson deviance per observation (0 for a perfect fit)."""
    y = np.asarray(y_true, dtype=float).reshape(-1)
    mu = np.clip(np.asarray(y_pred, dtype=float).reshape(-1), 1e-9, None)
    if y.shape != mu.shape:
        raise ValueError("y_true and y_pred must have the same length")
    contribution = np.where(y > 0, y * np.log(np.clip(y, 1e-12, None) / mu), 0.0) - (y - mu)
    return float(np.mean(2.0 * contribution))


def _safe_mape(y: np.ndarray, pred: np.ndarray) -> float:
    mask = y > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y[mask] - pred[mask]) / y[mask]) * 100)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    residual = pred - y
    return {
        "n": len(y),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "poisson_deviance": poisson_deviance(y, pred),
        "bias": float(np.mean(residual)),
        "relative_bias_pct": float(100 * np.mean(residual) / np.mean(y)) if y.mean() > 0 else float("nan"),
        "mean_actual": float(np.mean(y)),
        "mean_predicted": float(np.mean(pred)),
        "mape_pct_nonzero": _safe_mape(y, pred),
        "r2": float(1 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2))
        if np.sum((y - y.mean()) ** 2) > 0
        else float("nan"),
    }


def calibration_table(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed vs predicted counts by predicted-risk decile."""
    frame = pd.DataFrame({"actual": np.asarray(y_true, float), "predicted": np.asarray(y_pred, float)})
    try:
        frame["bin"] = pd.qcut(frame["predicted"], q=n_bins, duplicates="drop")
    except ValueError:  # pragma: no cover - degenerate predictions
        frame["bin"] = pd.cut(frame["predicted"], bins=n_bins, duplicates="drop")
    table = (
        frame.groupby("bin", observed=True)
        .agg(
            n=("actual", "size"),
            predicted_mean=("predicted", "mean"),
            actual_mean=("actual", "mean"),
            actual_sd=("actual", "std"),
        )
        .reset_index()
    )
    table["ratio_actual_to_predicted"] = table["actual_mean"] / table["predicted_mean"].replace(0, np.nan)
    table["calibration_error"] = table["actual_mean"] - table["predicted_mean"]
    table["bin"] = table["bin"].astype(str)
    return table


def subgroup_errors(
    y_true: np.ndarray, y_pred: np.ndarray, n_quantiles: int = 10
) -> dict[str, Any]:
    y = np.asarray(y_true, float)
    pred = np.asarray(y_pred, float)
    zero = y == 0
    threshold = float(np.quantile(y, 0.90)) if len(y) else np.nan
    high = y >= threshold
    return {
        "zero_actual": {
            "n": int(zero.sum()),
            "mean_predicted": float(pred[zero].mean()) if zero.any() else float("nan"),
            "mae": float(np.mean(np.abs(pred[zero] - y[zero]))) if zero.any() else float("nan"),
        },
        "high_actual_top_decile": {
            "n": int(high.sum()),
            "threshold": threshold,
            "mean_actual": float(y[high].mean()) if high.any() else float("nan"),
            "mean_predicted": float(pred[high].mean()) if high.any() else float("nan"),
            "mae": float(np.mean(np.abs(pred[high] - y[high]))) if high.any() else float("nan"),
        },
    }


def monthly_metrics(frame: pd.DataFrame, prediction_column: str = "predicted") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, group in frame.groupby("month", sort=True):
        metrics = regression_metrics(group["y_all"], group[prediction_column])
        metrics["month"] = pd.Timestamp(month)
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def baseline_comparison(test_frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate simple, already-available baselines on the same test rows.

    These are reference forecasts, not competing ML models.
    """
    baselines = {
        "Seasonal naive (same month last year)": "lag_12",
        "Rolling 3-month mean": "roll_mean_3",
        "Historical expanding mean": "expanding_mean",
    }
    rows: list[dict[str, Any]] = []
    for name, column in baselines.items():
        subset = test_frame.loc[test_frame[column].notna()]
        if subset.empty:
            continue
        metrics = regression_metrics(subset["y_all"], subset[column])
        metrics["baseline"] = name
        metrics["n"] = len(subset)
        rows.append(metrics)
    return pd.DataFrame(rows)


def grouped_metrics(
    frame: pd.DataFrame, group_column: str, prediction_column: str = "predicted"
) -> pd.DataFrame:
    """Metrics by site type or geographic cluster, on the same test rows."""
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(group_column, observed=True, sort=True):
        metrics = regression_metrics(group["y_all"], group[prediction_column])
        metrics[group_column] = value
        rows.append(metrics)
    return pd.DataFrame(rows)


def full_evaluation(test_frame: pd.DataFrame, prediction_column: str = "predicted") -> dict[str, Any]:
    """All required out-of-sample diagnostics for the final model."""
    y = test_frame["y_all"].to_numpy(float)
    pred = test_frame[prediction_column].to_numpy(float)
    metrics = regression_metrics(y, pred)
    return {
        "test_metrics": metrics,
        "calibration": calibration_table(y, pred).to_dict(orient="records"),
        "subgroup_errors": subgroup_errors(y, pred),
        "monthly": monthly_metrics(test_frame, prediction_column).to_dict(orient="records"),
        "baselines": baseline_comparison(test_frame).to_dict(orient="records"),
        "by_site_type": grouped_metrics(test_frame, "site_type", prediction_column).to_dict(orient="records"),
        "by_cluster": grouped_metrics(test_frame, "cluster", prediction_column).to_dict(orient="records"),
        "test_period": {
            "first": str(pd.Timestamp(test_frame["month"].min()).date()),
            "last": str(pd.Timestamp(test_frame["month"].max()).date()),
            "months": int(test_frame["month"].nunique()),
            "rows": len(test_frame),
        },
    }
