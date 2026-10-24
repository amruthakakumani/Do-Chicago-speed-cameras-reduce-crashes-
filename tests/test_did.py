"""Tests for the DiD estimators and inference utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chicago_cameras.did import (
    _codes,
    baseline_negative_binomial,
    baseline_ols,
    baseline_poisson,
    cohort_did,
    cohort_did_from_panel,
    did_conditional_poisson,
    event_study_fe,
    feols,
    naive_before_after,
    poisson_dispersion,
    wild_cluster_bootstrap,
)
from chicago_cameras.panel import build_panel


def test_feols_recovers_known_effect() -> None:
    rng = np.random.default_rng(3)
    n_units, n_months = 20, 12
    units = np.repeat(np.arange(n_units), n_months)
    months = np.tile(np.arange(n_months), n_units)
    treated = units < 8
    post = months >= 6
    x = (treated & post).astype(float)

    unit_fe = rng.normal(0, 2, n_units)[units]
    time_fe = rng.normal(0, 1, n_months)[months]
    y = unit_fe + time_fe + 1.5 * x

    result = feols(
        y,
        x,
        fe_codes=[_codes(units), _codes(months)],
        cluster_codes=_codes(units),
        names=["x"],
        n_fe_params=(n_units - 1) + (n_months - 1),
    )
    assert result["beta"][0] == pytest.approx(1.5, abs=1e-8)
    assert result["se"][0] < 1e-6


def test_feols_rejects_collinear_regressor() -> None:
    n = 10
    units = np.arange(n)
    y = np.arange(n, dtype=float)
    constant = np.ones(n)
    with pytest.raises(ValueError, match="collinear"):
        feols(y, constant, fe_codes=[_codes(units)], cluster_codes=_codes(units), names=["const"])


def test_conditional_poisson_recovers_rate_ratio() -> None:
    rng = np.random.default_rng(7)
    rows = []
    n_stacks, k_controls = 24, 4
    true_effect = -0.30
    for stack in range(n_stacks):
        for rel in range(-12, 12):
            mu = rng.uniform(1.5, 5.0)
            treated_mean = mu * np.exp(true_effect) if rel >= 0 else mu
            rows.append(
                {
                    "stack": f"S{stack}",
                    "unit": f"T{stack}",
                    "treated": 1,
                    "rel": rel,
                    "post": int(rel >= 0),
                    "tp": int(rel >= 0),
                    "y": rng.poisson(treated_mean),
                }
            )
            for control in range(k_controls):
                rows.append(
                    {
                        "stack": f"S{stack}",
                        "unit": f"C{stack}_{control}",
                        "treated": 0,
                        "rel": rel,
                        "post": int(rel >= 0),
                        "tp": 0,
                        "y": rng.poisson(mu),
                    }
                )
    panel = pd.DataFrame(rows)
    estimate = did_conditional_poisson(panel)
    assert estimate.beta == pytest.approx(true_effect, abs=0.12)
    assert estimate.scale == "log"
    assert estimate.n_clusters == n_stacks


def test_cohort_did_recovers_known_step() -> None:
    n_cameras, n_months, go_idx, post = 4, 40, 20, 12
    Y_camera = np.full((n_cameras, n_months), 10.0)
    Y_camera[:, go_idx:] -= 3.0
    Y_control = np.full((6, n_months), 10.0)
    matches = {i: np.array([0, 1, 2]) for i in range(n_cameras)}
    estimate, events = cohort_did(
        Y_camera,
        Y_control,
        matches,
        camera_go_idx={i: go_idx for i in range(n_cameras)},
        post_months=post,
        n_boot=99,
        seed=0,
    )
    assert estimate.beta == pytest.approx(-3.0, abs=1e-9)
    assert events["att"].eq(-3.0).all()
    assert estimate.n_clusters == n_cameras


def test_baseline_estimators_run_and_report_expected_schema(synthetic_panel: pd.DataFrame) -> None:
    for estimator in (baseline_ols, baseline_poisson):
        estimate = estimator(synthetic_panel)
        row = estimate.to_row()
        assert {"method", "estimate", "ci_low", "ci_high", "pvalue"} <= set(row)
        if estimate.scale == "log":
            assert row["pct"] == pytest.approx((np.exp(estimate.beta) - 1) * 100)

    naive = naive_before_after(synthetic_panel)
    assert np.isfinite(naive.beta)


def test_event_study_returns_reference_quarter(synthetic_panel: pd.DataFrame) -> None:
    table = event_study_fe(synthetic_panel, reference_quarter=-1)
    reference = table.loc[table["quarter"] == -1].iloc[0]
    assert reference["beta"] == 0
    assert 0 <= table.attrs["pre_trend_pvalue"] <= 1


def test_cohort_from_panel_matches_matrix_version() -> None:
    rng = np.random.default_rng(5)
    n_cameras, n_months, go_idx, post, pre = 4, 36, 20, 10, 8
    Y_camera = rng.poisson(6, size=(n_cameras, n_months)).astype(float)
    Y_control = rng.poisson(5, size=(6, n_months)).astype(float)
    matches = {i: np.array([0, 1, 2]) for i in range(n_cameras)}
    panel = build_panel(
        Y_camera,
        Y_control,
        matches,
        {i: go_idx for i in range(n_cameras)},
        {i: f"C{i}" for i in range(n_cameras)},
        np.array([f"G{i}" for i in range(6)]),
        pre_months=pre,
        post_months=post,
    )
    matrix_estimate, _ = cohort_did(
        Y_camera, Y_control, matches, {i: go_idx for i in range(n_cameras)}, post, 49, 0
    )
    panel_estimate = cohort_did_from_panel(panel, post_months=post, n_boot=49, seed=0)
    assert panel_estimate.beta == pytest.approx(matrix_estimate.beta, abs=1e-9)
    assert panel_estimate.n_clusters == matrix_estimate.n_clusters


def test_wild_cluster_bootstrap_smoke(synthetic_panel: pd.DataFrame) -> None:
    codes = _codes(synthetic_panel["unit"])
    clusters = _codes(synthetic_panel["stack"])
    result = wild_cluster_bootstrap(
        synthetic_panel["y"].to_numpy(float),
        synthetic_panel["tp"].to_numpy(float),
        fe_codes=[codes],
        cluster_codes=clusters,
        n_boot=49,
        seed=1,
    )
    assert 0 < result["pvalue"] <= 1


def test_negative_binomial_and_poisson_dispersion(synthetic_panel: pd.DataFrame) -> None:
    estimate = baseline_negative_binomial(synthetic_panel)
    assert estimate.scale == "log"
    assert np.isfinite(estimate.beta)
    assert estimate.n_clusters == synthetic_panel["stack"].nunique()
    dispersion = poisson_dispersion(synthetic_panel)
    assert np.isfinite(dispersion) and dispersion > 0
