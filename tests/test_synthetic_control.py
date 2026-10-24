"""Tests for the synthetic-control fit and placebo utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chicago_cameras.synthetic_control import (
    fit_simplex_weights,
    placebo_pvalues,
    run_synthetic_control,
)


def test_weights_are_on_the_simplex() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    y = X @ np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    weights = fit_simplex_weights(y, X)
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= -1e-9).all()


def test_exact_convex_combination_is_recovered() -> None:
    rng = np.random.default_rng(1)
    T = 60
    X = rng.poisson(4.0, size=(T, 3)).astype(float)
    y = 0.6 * X[:, 0] + 0.4 * X[:, 2]
    fit = run_synthetic_control(y, X, g=40, post_len=10)
    assert fit.pre_r2 == pytest.approx(1.0, abs=1e-6)
    assert fit.pre_rmspe < 1e-6
    assert abs(fit.gap).max() < 1e-6
    weights = dict(zip(fit.donor_rows.tolist(), fit.weights.tolist()))
    assert weights[0] == pytest.approx(0.6, abs=1e-3)
    assert weights[2] == pytest.approx(0.4, abs=1e-3)


def test_poor_pre_fit_is_visible() -> None:
    rng = np.random.default_rng(2)
    T = 48
    y = rng.poisson(4.0, size=T).astype(float)
    X = rng.poisson(4.0, size=(T, 20)).astype(float)
    fit = run_synthetic_control(y, X, g=30, post_len=10)
    assert fit.pre_r2 < 0.9  # independent series cannot be fit well


def test_synthetic_control_validates_inputs() -> None:
    with pytest.raises(ValueError):
        run_synthetic_control(np.zeros(10), np.zeros((9, 2)), g=5, post_len=2)
    with pytest.raises(ValueError):
        run_synthetic_control(np.zeros(10), np.zeros((10, 2)), g=0, post_len=2)


def test_placebo_pvalues_rank_the_real_fit() -> None:
    rng = np.random.default_rng(3)
    T = 50
    X = rng.poisson(4.0, size=(T, 30)).astype(float)
    y = X[:, :5].mean(axis=1) - 2.0 * (np.arange(T) >= 36)
    fit = run_synthetic_control(y, X, g=36, post_len=10)
    records = pd.DataFrame(
        {
            "rmspe_ratio": rng.uniform(0.5, 2.0, 40),
            "post_gap_mean": rng.normal(0, 0.5, 40),
        }
    )
    result = placebo_pvalues(fit, {"records": records})
    assert 0 <= result["p_two_sided_ratio"] <= 1
    assert 0 <= result["p_one_sided_negative"] <= 1
