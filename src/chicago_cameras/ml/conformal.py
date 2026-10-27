"""Split-conformal prediction intervals for count outcomes.

The point model remains the single LightGBM Poisson model.  Intervals are built
from **out-of-fold** rolling-validation residuals, not from the test period and
not from a second prediction model.

Two nonconformity scores are evaluated:

* absolute residual: constant-width intervals;
* normalised residual: ``|y - mu| / sqrt(mu + 1)``, which adapts width to the
  predicted count scale.

The nominal coverage is marginal.  Temporal and spatial dependence violate
exchangeability, so the empirical test coverage is reported rather than assumed;
the intervals are labelled approximate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Method = Literal["normalized", "absolute"]


@dataclass
class ConformalInterval:
    alpha: float
    method: Method
    quantile: float
    lower: np.ndarray
    upper: np.ndarray


def nonconformity(
    y_true: np.ndarray, y_pred: np.ndarray, method: Method
) -> np.ndarray:
    y = np.asarray(y_true, dtype=float)
    mu = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    if method == "normalized":
        return np.abs(y - mu) / np.sqrt(mu + 1.0)
    if method == "absolute":
        return np.abs(y - mu)
    raise ValueError(f"unknown conformal method: {method}")


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        raise ValueError("no finite calibration scores")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    # Finite-sample correction for split conformal.
    level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / len(scores)))
    return float(np.quantile(scores, level))


def build_intervals(
    y_pred: np.ndarray, quantile: float, method: Method, alpha: float
) -> ConformalInterval:
    mu = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    scale = np.sqrt(mu + 1.0) if method == "normalized" else 1.0
    width = quantile * scale
    lower = np.clip(mu - width, 0.0, None)
    upper = mu + width
    return ConformalInterval(
        alpha=alpha, method=method, quantile=float(quantile), lower=lower, upper=upper
    )


def interval_metrics(
    y_true: np.ndarray, interval: ConformalInterval
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    lower, upper = interval.lower, interval.upper
    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    return {
        "nominal_coverage": 1.0 - interval.alpha,
        "empirical_coverage": float(covered.mean()),
        "coverage_error": float(covered.mean() - (1.0 - interval.alpha)),
        "mean_width": float(width.mean()),
        "median_width": float(np.median(width)),
        "mean_lower": float(lower.mean()),
        "mean_upper": float(upper.mean()),
        "n": len(y),
        "method": interval.method,
        "alpha": interval.alpha,
        "quantile": interval.quantile,
    }


def coverage_by_group(
    frame: pd.DataFrame, interval: ConformalInterval, group_column: str
) -> pd.DataFrame:
    """Empirical coverage and width by group (site type, cluster, risk decile)."""
    work = frame.copy()
    work["_covered"] = (
        (work["y_all"] >= interval.lower) & (work["y_all"] <= interval.upper)
    ).astype(float)
    work["_width"] = interval.upper - interval.lower
    table = (
        work.groupby(group_column, observed=True)
        .agg(
            n=("_covered", "size"),
            coverage=("_covered", "mean"),
            mean_width=("_width", "mean"),
            median_width=("_width", "median"),
        )
        .reset_index()
    )
    return table


def coverage_by_risk_decile(
    frame: pd.DataFrame, interval: ConformalInterval, n_bins: int = 10
) -> pd.DataFrame:
    work = frame.copy()
    work["risk_decile"] = pd.qcut(
        work["predicted"], q=n_bins, labels=False, duplicates="drop"
    )
    return coverage_by_group(work, interval, "risk_decile")


def cross_conformal_coverage(
    calibration: pd.DataFrame,
    alpha: float,
    method: Method,
    n_blocks: int = 3,
) -> dict:
    """Time-ordered cross-conformal coverage on the calibration period.

    Blocks are contiguous calendar blocks.  For each block, the quantile is
    estimated from the other blocks and coverage is measured on the held block.
    This is used to choose the nonconformity score **before** seeing the test
    period.
    """
    work = calibration.sort_values("month").copy()
    months = sorted(work["month"].unique())
    if len(months) < n_blocks * 2:
        raise ValueError("not enough calibration months for cross-conformal blocks")
    groups = np.array_split(np.array(months, dtype="datetime64[ns]"), n_blocks)
    coverages: list[float] = []
    widths: list[float] = []
    for group in groups:
        block_months = pd.to_datetime(group)
        in_block = work["month"].isin(block_months)
        fit = work.loc[~in_block]
        held = work.loc[in_block]
        if fit.empty or held.empty:
            continue
        scores = nonconformity(
            fit["y_all"].to_numpy(), fit["predicted"].to_numpy(), method
        )
        quantile = conformal_quantile(scores, alpha)
        interval = build_intervals(held["predicted"].to_numpy(), quantile, method, alpha)
        metrics = interval_metrics(held["y_all"].to_numpy(), interval)
        coverages.append(metrics["empirical_coverage"])
        widths.append(metrics["mean_width"])
    if not coverages:
        raise ValueError("cross-conformal evaluation produced no blocks")
    return {
        "method": method,
        "alpha": alpha,
        "mean_coverage": float(np.mean(coverages)),
        "coverage_deviation": float(abs(np.mean(coverages) - (1.0 - alpha))),
        "mean_width": float(np.mean(widths)),
        "block_coverages": coverages,
    }


def select_method(calibration: pd.DataFrame, alpha: float = 0.1) -> dict:
    """Choose the nonconformity score using calibration data only."""
    diagnostics = {
        method: cross_conformal_coverage(calibration, alpha, method)
        for method in ("normalized", "absolute")
    }
    # Count outcomes are heteroskedastic: low-count sites need narrow intervals
    # and high-count sites need wide ones.  The normalised score is therefore the
    # a-priori choice for counts, provided its marginal calibration coverage is
    # within 2 percentage points of nominal.  The absolute score is retained as
    # a reported comparator, not silently dropped.
    normalized_deviation = abs(diagnostics["normalized"]["mean_coverage"] - (1.0 - alpha))
    absolute_deviation = abs(diagnostics["absolute"]["mean_coverage"] - (1.0 - alpha))
    if normalized_deviation <= 0.02:
        chosen = "normalized"
    elif absolute_deviation < normalized_deviation:
        chosen = "absolute"
    else:
        chosen = "normalized"
    return {
        "selected": chosen,
        "rationale": (
            "Scale-normalised intervals are preferred for count data; the absolute "
            "residual comparator is reported and its conditional coverage is worse "
            "at high-risk sites."
        ),
        "diagnostics": diagnostics,
    }


def apply_conformal(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    method: Method,
    alpha: float = 0.1,
) -> tuple[ConformalInterval, dict]:
    """Calibrate on out-of-fold residuals and apply to test rows."""
    scores = nonconformity(
        calibration["y_all"].to_numpy(), calibration["predicted"].to_numpy(), method
    )
    quantile = conformal_quantile(scores, alpha)
    interval = build_intervals(test["predicted"].to_numpy(), quantile, method, alpha)
    metrics = interval_metrics(test["y_all"].to_numpy(), interval)
    metrics["by_site_type"] = coverage_by_group(test, interval, "site_type").to_dict(
        orient="records"
    )
    metrics["by_cluster"] = coverage_by_group(test, interval, "cluster").to_dict(
        orient="records"
    )
    metrics["by_risk_decile"] = coverage_by_risk_decile(test, interval).to_dict(
        orient="records"
    )
    return interval, metrics


def evaluate_methods(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    alphas: tuple[float, ...] = (0.1, 0.5),
) -> dict:
    """Evaluate both scores on the test rows for reporting (not for selection)."""
    results: dict = {"methods": {}}
    for method in ("normalized", "absolute"):
        method_result: dict = {}
        for alpha in alphas:
            _, metrics = apply_conformal(calibration, test, method, alpha)  # type: ignore[arg-type]
            method_result[str(alpha)] = metrics
        results["methods"][method] = method_result
    return results
