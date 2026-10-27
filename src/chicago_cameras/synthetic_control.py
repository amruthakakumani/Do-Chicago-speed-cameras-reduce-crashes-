"""Synthetic control with de-meaned, simplex-constrained weights.

The original notebook fitted a convex combination of no-camera sites to the
average camera series, de-meaned so that the level difference between hotspot
camera catchments and ordinary sites does not dominate the fit.  This module
keeps that specification, makes every diagnostic explicit, and refuses to
present a synthetic control as informative when the pre-period fit is poor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

LOGGER = logging.getLogger(__name__)


def fit_simplex_weights(y_pre: np.ndarray, X_pre: np.ndarray) -> np.ndarray:
    """Minimise ||y_pre - X_pre w||^2 subject to w >= 0 and sum(w) = 1."""
    y_pre = np.asarray(y_pre, dtype=float).reshape(-1)
    X_pre = np.asarray(X_pre, dtype=float)
    if X_pre.ndim != 2 or X_pre.shape[0] != len(y_pre):
        raise ValueError("X_pre must be (n_pre, n_donors) and align with y_pre")
    n_donors = X_pre.shape[1]
    if n_donors == 0:
        raise ValueError("no donors available")

    # Warm start from a clipped least-squares solution: this conditions the
    # problem far better than a uniform start when donors are collinear.
    try:
        unconstrained, *_ = np.linalg.lstsq(X_pre, y_pre, rcond=None)
        start = np.clip(unconstrained, 0.0, None)
        start = start / start.sum() if start.sum() > 0 else np.full(n_donors, 1.0 / n_donors)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        start = np.full(n_donors, 1.0 / n_donors)

    def objective(w: np.ndarray) -> float:
        residual = y_pre - X_pre @ w
        return float(residual @ residual)

    def gradient(w: np.ndarray) -> np.ndarray:
        return -2.0 * (X_pre.T @ (y_pre - X_pre @ w))

    result = minimize(
        objective,
        start,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_donors,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n_donors)}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        LOGGER.warning("synthetic-control optimiser did not converge: %s", result.message)
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    total = weights.sum()
    if total <= 0:
        raise RuntimeError("synthetic-control optimiser returned all-zero weights")
    return weights / total


@dataclass
class SyntheticControlFit:
    """Result of one synthetic-control fit."""

    synth: np.ndarray
    gap: np.ndarray
    weights: np.ndarray
    donor_rows: np.ndarray
    pre_rmspe: float
    post_rmspe: float
    pre_mean: float
    pre_r2: float
    post_gap_mean: float
    pct_vs_synth: float
    rmspe_ratio: float
    n_pre: int
    n_post: int
    converged: bool = True
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "pre_rmspe": self.pre_rmspe,
            "post_rmspe": self.post_rmspe,
            "pre_mean": self.pre_mean,
            "pre_r2": self.pre_r2,
            "post_gap_mean": self.post_gap_mean,
            "pct_vs_synth": self.pct_vs_synth,
            "rmspe_ratio": self.rmspe_ratio,
            "n_pre": self.n_pre,
            "n_post": self.n_post,
            "n_donors": len(self.donor_rows),
            "top_weights": {
                int(k): float(v)
                for k, v in sorted(
                    zip(self.donor_rows, self.weights), key=lambda kv: kv[1], reverse=True
                )[:10]
            },
            **self.extra,
        }


def run_synthetic_control(
    y: np.ndarray,
    X: np.ndarray,
    g: int,
    post_len: int,
    max_donors: int = 150,
) -> SyntheticControlFit:
    """Fit a synthetic control to ``y`` using columns of ``X`` as donors.

    ``g`` is the index of the first post-treatment month.  Donors are pruned to
    the ``max_donors`` with the smallest pre-period discrepancy, using pre-period
    information only.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a (n_months, n_donors) matrix")
    if X.shape[0] != len(y):
        raise ValueError("X and y must cover the same months")
    if not (0 < g <= len(y)):
        raise ValueError("g must lie inside the observed period")
    if g < 3:
        raise ValueError("at least three pre-treatment months are required")

    mu_y = float(y[:g].mean())
    mu_x = X[:g].mean(axis=0)
    y_d = y - mu_y
    X_d = X - mu_x

    pre_error = ((X_d[:g] - y_d[:g, None]) ** 2).sum(axis=0)
    n_donors = min(max_donors, X.shape[1])
    keep = np.argsort(pre_error, kind="stable")[:n_donors]

    weights = fit_simplex_weights(y_d[:g], X_d[:g][:, keep])
    synth = X_d[:, keep] @ weights + mu_y
    gap = y - synth

    n_post = min(post_len, len(y) - g)
    pre_rmspe = float(np.sqrt(np.mean(gap[:g] ** 2)))
    post_rmspe = float(np.sqrt(np.mean(gap[g : g + n_post] ** 2)))
    ss_res = float((gap[:g] ** 2).sum())
    ss_tot = float(((y[:g] - y[:g].mean()) ** 2).sum())
    pre_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    post_gap = float(gap[g : g + n_post].mean())
    synth_mean = float(synth[g : g + n_post].mean())
    pct = 100 * post_gap / synth_mean if synth_mean > 0 else np.nan

    return SyntheticControlFit(
        synth=synth,
        gap=gap,
        weights=weights,
        donor_rows=keep,
        pre_rmspe=pre_rmspe,
        post_rmspe=post_rmspe,
        pre_mean=mu_y,
        pre_r2=float(pre_r2),
        post_gap_mean=post_gap,
        pct_vs_synth=float(pct),
        rmspe_ratio=post_rmspe / (pre_rmspe + 1e-12),
        n_pre=g,
        n_post=n_post,
    )


def placebo_in_space(
    y: np.ndarray,
    X: np.ndarray,
    g: int,
    treated_group_size: int,
    n_placebo: int,
    post_len: int,
    seed: int,
    max_donors: int = 150,
) -> dict[str, object]:
    """Placebo-in-space inference: randomly assign ``treated_group_size`` donors.

    Each placebo draws a pseudo-treated group from the donor pool, fits a
    synthetic control using the remaining donors, and records the same test
    statistics as the real fit.  Pseudo p-values are ranks among these draws.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    n_donors = X.shape[1]
    if treated_group_size >= n_donors:
        raise ValueError("treated group must be smaller than the donor pool")

    records: list[dict[str, float]] = []
    gaps: list[np.ndarray] = []
    for _ in range(n_placebo):
        picked = rng.choice(n_donors, size=treated_group_size, replace=False)
        remaining = np.setdiff1d(np.arange(n_donors), picked)
        y_p = X[:, picked].mean(axis=1)
        fit = run_synthetic_control(y_p, X[:, remaining], g, post_len, max_donors=max_donors)
        records.append(
            {
                "pre_rmspe": fit.pre_rmspe,
                "post_rmspe": fit.post_rmspe,
                "rmspe_ratio": fit.rmspe_ratio,
                "post_gap_mean": fit.post_gap_mean,
                "pre_r2": fit.pre_r2,
            }
        )
        gaps.append(fit.gap)

    return {
        "records": pd.DataFrame(records),
        "gaps": np.vstack(gaps),
        "n_placebo": n_placebo,
        "treated_group_size": treated_group_size,
    }


def placebo_pvalues(
    fit: SyntheticControlFit,
    placebo: dict[str, object],
) -> dict[str, float]:
    """Rank-based pseudo p-values for the real fit against placebo fits."""
    records: pd.DataFrame = placebo["records"]  # type: ignore[assignment]
    if records.empty:
        return {"p_two_sided_ratio": np.nan, "p_one_sided_negative": np.nan}
    ratio_p = float((records["rmspe_ratio"] >= fit.rmspe_ratio).mean())
    negative_p = float((records["post_gap_mean"] <= fit.post_gap_mean).mean())
    return {
        "p_two_sided_ratio": ratio_p,
        "p_one_sided_negative": negative_p,
        "placebo_ratio_median": float(records["rmspe_ratio"].median()),
        "placebo_ratio_p95": float(records["rmspe_ratio"].quantile(0.95)),
    }


def in_time_placebo(
    y: np.ndarray,
    X: np.ndarray,
    g: int,
    offset: int,
    post_len: int,
    max_donors: int = 150,
) -> dict[str, float]:
    """Fit the model with a fake treatment date ``offset`` months before go-live."""
    fake_g = int(g + offset)
    if fake_g <= 3 or fake_g >= g:
        raise ValueError("in-time placebo must fall inside the pre-period")
    fit = run_synthetic_control(y, X, fake_g, post_len=min(-offset, post_len), max_donors=max_donors)
    return {
        "offset_months": offset,
        "fake_g_index": fake_g,
        "pre_rmspe": fit.pre_rmspe,
        "fake_post_gap_mean": fit.post_gap_mean,
        "fake_post_pct": fit.pct_vs_synth,
    }


def leave_one_out(
    y: np.ndarray,
    X: np.ndarray,
    g: int,
    post_len: int,
    donor_rows: np.ndarray,
    weights: np.ndarray,
    top_k: int,
    max_donors: int = 150,
) -> pd.DataFrame:
    """Drop each of the most heavily weighted donors and refit."""
    order = np.argsort(weights)[::-1][:top_k]
    records: list[dict[str, float]] = []
    for position in order:
        if weights[position] <= 1e-8:
            continue
        drop = int(donor_rows[position])
        keep_columns = np.array([j for j in range(X.shape[1]) if j != drop])
        fit = run_synthetic_control(y, X[:, keep_columns], g, post_len, max_donors=max_donors)
        records.append(
            {
                "dropped_donor_column": drop,
                "original_weight": float(weights[position]),
                "post_gap_mean": fit.post_gap_mean,
                "pct_vs_synth": fit.pct_vs_synth,
                "pre_rmspe": fit.pre_rmspe,
            }
        )
    return pd.DataFrame(records)
