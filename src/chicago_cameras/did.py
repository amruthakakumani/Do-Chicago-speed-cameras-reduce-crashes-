"""Difference-in-differences estimators and dependence-robust inference.

The module keeps the original stacked OLS/Poisson specifications as baselines
and adds:

* a within-transformed fixed-effects estimator (unit fixed effects and
  stack-by-event-time fixed effects), which is the standard specification for a
  stacked event-study design and does not rely on a common post-period shock;
* an exact conditional Poisson estimator with stack-by-event-time fixed effects,
  obtained by conditioning on each stack-month total (equivalent to Poisson FE);
* a transparent cohort 2x2 estimator with camera-level bootstrap inference,
  which avoids the negative-weighting problem of two-way fixed effects under
  staggered adoption and treatment-effect heterogeneity;
* wild cluster bootstrap p-values for the preferred linear specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

LOGGER = logging.getLogger(__name__)


@dataclass
class Estimate:
    """A single scalar causal estimate with uncertainty."""

    name: str
    estimand: str
    beta: float
    se: float
    pvalue: float
    ci_low: float
    ci_high: float
    n_obs: int
    n_clusters: int
    scale: str = "log"  # "log" => report percent change; "level" => raw units
    method: str = ""
    details: dict = field(default_factory=dict)

    @property
    def pct(self) -> float | None:
        if self.scale == "log":
            return float((np.exp(self.beta) - 1) * 100)
        return None

    @property
    def pct_ci(self) -> tuple[float, float] | None:
        if self.scale != "log":
            return None
        return (
            float((np.exp(self.ci_low) - 1) * 100),
            float((np.exp(self.ci_high) - 1) * 100),
        )

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "method": self.method or self.name,
            "estimand": self.estimand,
            "estimate": self.beta,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "pvalue": self.pvalue,
            "n_obs": self.n_obs,
            "n_clusters": self.n_clusters,
            "scale": self.scale,
        }
        if self.scale == "log":
            lo, hi = self.pct_ci or (np.nan, np.nan)
            row.update({"pct": self.pct, "pct_ci_low": lo, "pct_ci_high": hi})
        row.update({f"detail_{k}": v for k, v in self.details.items() if np.isscalar(v)})
        return row


# ---------------------------------------------------------------------------
# Fixed-effect absorption
# ---------------------------------------------------------------------------


def _codes(values: Iterable[object]) -> np.ndarray:
    codes, _ = pd.factorize(pd.Series(list(values)), sort=True)
    if (codes < 0).any():
        raise ValueError("grouping variable contains missing values")
    return codes.astype(np.int64)


def _demean(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    counts = np.bincount(codes, minlength=int(codes.max()) + 1).astype(float)
    sums = np.bincount(codes, weights=values, minlength=len(counts))
    return values - (sums / counts)[codes]


def absorb(
    values: np.ndarray,
    fe_codes: Sequence[np.ndarray],
    tol: float = 1e-10,
    max_iter: int = 500,
) -> np.ndarray:
    """Residualise ``values`` on additive fixed effects by alternating projections."""
    out = np.asarray(values, dtype=float).copy()
    if not fe_codes:
        return out
    for _ in range(max_iter):
        previous = out.copy()
        for codes in fe_codes:
            out = _demean(out, codes)
        if np.max(np.abs(out - previous), initial=0.0) < tol:
            return out
    raise RuntimeError(
        "fixed-effect absorption did not converge; check for disconnected groups "
        "or pass fewer fixed effects"
    )


def _cluster_meat(X: np.ndarray, resid: np.ndarray, cluster_codes: np.ndarray) -> np.ndarray:
    n_clusters = int(cluster_codes.max()) + 1
    scores = np.zeros((n_clusters, X.shape[1]))
    for column in range(X.shape[1]):
        scores[:, column] = np.bincount(
            cluster_codes, weights=X[:, column] * resid, minlength=n_clusters
        )
    return scores.T @ scores


def feols(
    y: np.ndarray,
    X: np.ndarray,
    fe_codes: Sequence[np.ndarray],
    cluster_codes: np.ndarray,
    names: Sequence[str],
    n_fe_params: int | None = None,
) -> dict[str, object]:
    """OLS with absorbed fixed effects and one-way cluster-robust inference."""
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if len(y) != len(X):
        raise ValueError("y and X must have the same number of rows")
    if len(names) != X.shape[1]:
        raise ValueError("names must match the number of columns in X")
    if np.isnan(y).any() or np.isnan(X).any():
        raise ValueError("design matrix contains missing values")

    y_t = absorb(y, fe_codes)
    X_t = np.column_stack([absorb(X[:, j], fe_codes) for j in range(X.shape[1])])

    scale = np.sqrt((X_t**2).sum(axis=0))
    if (scale < 1e-10).any():
        dead = [name for name, s in zip(names, scale) if s < 1e-10]
        raise ValueError(f"collinear or absorbed regressor(s): {dead}")

    xtx_inv = np.linalg.pinv(X_t.T @ X_t)
    beta = xtx_inv @ (X_t.T @ y_t)
    resid = y_t - X_t @ beta

    n_obs = len(y)
    k = int(n_fe_params if n_fe_params is not None else 0) + X.shape[1]
    n_clusters = int(cluster_codes.max()) + 1
    meat = _cluster_meat(X_t, resid, cluster_codes)
    cov = xtx_inv @ meat @ xtx_inv
    if n_clusters > 1:
        cov *= (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / max(n_obs - k, 1))

    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, beta / se, np.nan)
    from scipy import stats as st

    pvalue = 2 * st.t.sf(np.abs(tstat), df=max(n_clusters - 1, 1))
    crit = st.t.ppf(0.975, df=max(n_clusters - 1, 1))
    return {
        "names": list(names),
        "beta": beta,
        "se": se,
        "tstat": tstat,
        "pvalue": pvalue,
        "cov": cov,
        "ci_low": beta - crit * se,
        "ci_high": beta + crit * se,
        "n_obs": n_obs,
        "n_clusters": n_clusters,
    }


def _fe_param_count(panel: pd.DataFrame, columns: Sequence[str]) -> int:
    total = 0
    for column in columns:
        total += panel[column].nunique() - 1
    return total


# ---------------------------------------------------------------------------
# Baseline specifications (faithful replications)
# ---------------------------------------------------------------------------


def baseline_ols(panel: pd.DataFrame) -> Estimate:
    """Original notebook: y ~ treated + post + treated:post, cluster by stack."""
    model = smf.ols("y ~ treated + post + tp", data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["stack"].values}
    )
    low, high = model.conf_int().loc["tp"]
    return Estimate(
        name="did_ols_baseline",
        estimand="ATT (crashes per site-month)",
        beta=float(model.params["tp"]),
        se=float(model.bse["tp"]),
        pvalue=float(model.pvalues["tp"]),
        ci_low=float(low),
        ci_high=float(high),
        n_obs=int(model.nobs),
        n_clusters=int(panel["stack"].nunique()),
        scale="level",
        method="Stacked OLS, treated + post + treated x post (original specification)",
        details={"r2": float(model.rsquared)},
    )


def baseline_poisson(panel: pd.DataFrame) -> Estimate:
    """Original notebook: Poisson GLM with the same regressors, cluster by stack."""
    model = smf.glm(
        "y ~ treated + post + tp",
        data=panel,
        family=sm.families.Poisson(),
    ).fit(cov_type="cluster", cov_kwds={"groups": panel["stack"].values})
    low, high = model.conf_int().loc["tp"]
    return Estimate(
        name="did_poisson_baseline",
        estimand="ATT (percent change in crash rate)",
        beta=float(model.params["tp"]),
        se=float(model.bse["tp"]),
        pvalue=float(model.pvalues["tp"]),
        ci_low=float(low),
        ci_high=float(high),
        n_obs=int(model.nobs),
        n_clusters=int(panel["stack"].nunique()),
        scale="log",
        method="Stacked Poisson GLM, treated + post + treated x post (original specification)",
        details={"deviance": float(model.deviance)},
    )


def baseline_negative_binomial(panel: pd.DataFrame) -> Estimate:
    """Negative Binomial (NB2) DiD, the count model for overdispersed outcomes.

    Poisson standard errors are only valid under equidispersion.  If the
    dispersion diagnostic indicates overdispersion, the NB2 model is the more
    appropriate baseline; it is reported alongside Poisson rather than replacing
    it, and the conditional Poisson FE remains the preferred count estimator
    because it absorbs stack-by-month shocks.
    """
    model = sm.NegativeBinomial.from_formula("y ~ treated + post + tp", data=panel)
    try:
        result = model.fit(
            disp=0,
            maxiter=300,
            cov_type="cluster",
            cov_kwds={"groups": panel["stack"].values},
        )
    except (ValueError, np.linalg.LinAlgError) as exc:  # pragma: no cover - numerical guard
        raise RuntimeError(f"negative binomial did not converge: {exc}") from exc

    low, high = result.conf_int().loc["tp"]
    return Estimate(
        name="did_negative_binomial",
        estimand="ATT (percent change in crash rate)",
        beta=float(result.params["tp"]),
        se=float(result.bse["tp"]),
        pvalue=float(result.pvalues["tp"]),
        ci_low=float(low),
        ci_high=float(high),
        n_obs=int(result.nobs),
        n_clusters=int(panel["stack"].nunique()),
        scale="log",
        method="Stacked Negative Binomial (NB2) DiD, cluster by camera",
        details={"alpha": float(result.params.get("alpha", np.nan))},
    )


def poisson_dispersion(panel: pd.DataFrame) -> float:
    """Pearson dispersion statistic for the baseline Poisson model.

    Values substantially above 1 indicate overdispersion that Poisson standard
    errors do not account for.
    """
    model = smf.glm("y ~ treated + post + tp", data=panel, family=sm.families.Poisson()).fit()
    mu = np.clip(np.asarray(model.fittedvalues, dtype=float), 1e-9, None)
    y = panel["y"].to_numpy(float)
    dof = max(len(y) - model.df_model - 1, 1)
    return float((((y - mu) ** 2) / mu).sum() / dof)


def naive_before_after(panel: pd.DataFrame) -> Estimate:
    """Treated-only before/after comparison, shown to demonstrate the bias."""
    treated = panel.loc[panel["treated"] == 1]
    pre = treated.loc[treated["post"] == 0, "y"].mean()
    post = treated.loc[treated["post"] == 1, "y"].mean()
    beta = float(np.log(post / pre)) if pre > 0 and post > 0 else np.nan
    return Estimate(
        name="naive_before_after",
        estimand="Naive before/after (no comparison group)",
        beta=beta,
        se=np.nan,
        pvalue=np.nan,
        ci_low=np.nan,
        ci_high=np.nan,
        n_obs=int(len(treated)),
        n_clusters=int(treated["stack"].nunique()),
        scale="log",
        method="Treated-only before/after mean comparison (biased by mean reversion)",
        details={"pre_mean": float(pre), "post_mean": float(post)},
    )


# ---------------------------------------------------------------------------
# Preferred specification: unit FE + stack-by-event-time FE
# ---------------------------------------------------------------------------


def did_fe_ols(panel: pd.DataFrame) -> Estimate:
    """Within estimator with unit and stack-by-relative-month fixed effects."""
    work = panel.copy()
    work["stack_rel"] = work["stack"].astype(str) + "|" + work["rel"].astype(str)
    fe = [_codes(work["unit"]), _codes(work["stack_rel"])]
    result = feols(
        work["y"].to_numpy(dtype=float),
        work[["tp"]].to_numpy(dtype=float),
        fe_codes=fe,
        cluster_codes=_codes(work["stack"]),
        names=["tp"],
        n_fe_params=_fe_param_count(work, ["unit", "stack_rel"]),
    )
    return Estimate(
        name="did_fe_ols",
        estimand="ATT (crashes per site-month)",
        beta=float(result["beta"][0]),
        se=float(result["se"][0]),
        pvalue=float(result["pvalue"][0]),
        ci_low=float(result["ci_low"][0]),
        ci_high=float(result["ci_high"][0]),
        n_obs=int(result["n_obs"]),
        n_clusters=int(result["n_clusters"]),
        scale="level",
        method="Two-way FE OLS: unit FE + stack x event-month FE, cluster by camera",
    )


def did_conditional_poisson(panel: pd.DataFrame) -> Estimate:
    """Exact conditional Poisson with stack-by-event-time fixed effects.

    Conditioning on each stack-month total turns the Poisson FE model into a
    binomial model for the treated count, with the number of matched controls as
    exposure.  This avoids high-dimensional dummy estimation and yields the
    same likelihood as Poisson FE with a full set of stack-month intercepts.
    """
    treated = panel.loc[panel["treated"] == 1].set_index(["stack", "rel"])["y"]
    controls = (
        panel.loc[panel["treated"] == 0]
        .groupby(["stack", "rel"], as_index=True)
        .agg(y_controls=("y", "sum"), k=("y", "size"))
    )
    cells = pd.concat([treated.rename("y_treated"), controls], axis=1).dropna()
    cells = cells.loc[(cells["k"] > 0) & (cells["y_treated"] + cells["y_controls"] > 0)]
    if cells.empty:
        raise ValueError("no complete stack-month cells for the conditional Poisson model")

    endog = np.column_stack([cells["y_treated"].to_numpy(float), cells["y_controls"].to_numpy(float)])
    exog = sm.add_constant((cells.index.get_level_values("rel") >= 0).astype(float))
    offset = -np.log(cells["k"].to_numpy(float))
    model = sm.GLM(endog, exog, family=sm.families.Binomial(), offset=offset).fit(
        cov_type="cluster",
        cov_kwds={"groups": cells.index.get_level_values("stack").to_numpy()},
    )
    beta = float(model.params[1])
    low, high = model.conf_int()[1]
    return Estimate(
        name="did_conditional_poisson",
        estimand="ATT (percent change in crash rate)",
        beta=beta,
        se=float(model.bse[1]),
        pvalue=float(model.pvalues[1]),
        ci_low=float(low),
        ci_high=float(high),
        n_obs=int(len(cells)),
        n_clusters=int(cells.index.get_level_values("stack").nunique()),
        scale="log",
        method="Conditional Poisson FE (stack x event-month fixed effects), cluster by camera",
        details={"conditional_cells": int(len(cells))},
    )


# ---------------------------------------------------------------------------
# Cohort 2x2 estimator with camera-level bootstrap
# ---------------------------------------------------------------------------


def cohort_event_effects(
    Y_camera: np.ndarray,
    Y_control: np.ndarray,
    matches: dict[int, np.ndarray],
    camera_go_idx: dict[int, int],
    post_months: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-camera 2x2 ATT by event month, relative to the month before go-live.

    Returns ``(att, valid)`` where ``att`` is ``(n_cameras, post_months)``.
    Difference-in-differences at event time ``e`` for camera ``i`` is

        (Y_i,g+e - Y_i,g-1) - mean_j(Y_j,g+e - Y_j,g-1)

    over that camera's matched controls.  Each camera therefore supplies its own
    comparison, and cameras are never combined with heterogeneous TWFE weights.
    """
    cameras = sorted(matches)
    att = np.full((len(cameras), post_months), np.nan)
    valid = np.zeros(len(cameras), dtype=bool)
    for row, camera_row in enumerate(cameras):
        g = int(camera_go_idx[camera_row])
        controls = np.asarray(matches[camera_row], dtype=int)
        if g < 1 or g + post_months > Y_camera.shape[1]:
            continue
        base_treated = float(Y_camera[camera_row, g - 1])
        base_control = Y_control[controls, g - 1].mean()
        treated_path = Y_camera[camera_row, g : g + post_months].astype(float)
        control_path = Y_control[controls, g : g + post_months].mean(axis=0)
        att[row] = (treated_path - base_treated) - (control_path - base_control)
        valid[row] = True
    return att, valid


def _bootstrap_mean(
    values: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI and two-sided p-value for a mean."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Bootstrap p-value for H0: mean = 0, based on the mass of the bootstrap
    # distribution on the other side of zero (two-sided).
    p_low = float((draws <= 0).mean())
    p_high = float((draws >= 0).mean())
    pvalue = min(1.0, 2 * min(p_low, p_high))
    return float(low), float(high), pvalue


def cohort_did_from_panel(
    panel: pd.DataFrame,
    post_months: int,
    n_boot: int,
    seed: int,
    name: str = "cohort_panel",
) -> Estimate:
    """Cohort 2x2 ATT computed directly from a (possibly filtered) panel.

    This is used for robustness specifications such as the COVID exclusion, where
    the raw count matrices would otherwise silently include the excluded months.
    """
    treated = (
        panel.loc[panel["treated"] == 1]
        .pivot_table(index="stack", columns="rel", values="y", aggfunc="sum")
        .sort_index(axis=1)
    )
    controls = (
        panel.loc[panel["treated"] == 0]
        .groupby(["stack", "rel"])["y"]
        .mean()
        .unstack("rel")
        .sort_index(axis=1)
    )
    stacks = treated.index.intersection(controls.index)
    if len(stacks) == 0 or -1 not in treated.columns or -1 not in controls.columns:
        raise ValueError("panel lacks a baseline month (rel = -1) for the cohort estimator")

    post = [r for r in range(0, post_months) if r in treated.columns and r in controls.columns]
    if not post:
        raise ValueError("panel has no post-treatment months for the cohort estimator")

    treated = treated.loc[stacks]
    controls = controls.loc[stacks]
    baseline_gap = treated[-1] - controls[-1]
    diffs = (treated[post] - controls[post]).sub(baseline_gap, axis=0)
    per_camera_series = diffs.mean(axis=1)
    per_camera_series = per_camera_series[np.isfinite(per_camera_series)]
    if per_camera_series.empty:
        raise ValueError(
            "no camera has a complete post-treatment comparison after filtering"
        )
    per_camera = per_camera_series.to_numpy(float)

    low, high, pvalue = _bootstrap_mean(per_camera, n_boot=n_boot, seed=seed)
    return Estimate(
        name=name,
        estimand="ATT (crashes per site-month, cohort 2x2)",
        beta=float(np.mean(per_camera)),
        se=float(np.std(per_camera, ddof=1) / np.sqrt(len(per_camera)))
        if len(per_camera) > 1
        else np.nan,
        pvalue=pvalue,
        ci_low=low,
        ci_high=high,
        n_obs=int(len(per_camera) * len(post)),
        n_clusters=int(len(per_camera)),
        scale="level",
        method="Matched 2x2 DiD per camera, average across cameras, bootstrap by camera",
        details={"n_cameras": int(len(per_camera)), "n_post_months": len(post)},
    )


def cohort_did(
    Y_camera: np.ndarray,
    Y_control: np.ndarray,
    matches: dict[int, np.ndarray],
    camera_go_idx: dict[int, int],
    post_months: int,
    n_boot: int,
    seed: int,
    name: str = "did_cohort_2x2",
) -> tuple[Estimate, pd.DataFrame]:
    """Average cohort 2x2 ATT across cameras with camera-level bootstrap CI."""
    att_matrix, valid = cohort_event_effects(
        Y_camera, Y_control, matches, camera_go_idx, post_months
    )
    if not valid.any():
        raise ValueError("no camera has a complete post period for the 2x2 estimator")

    per_camera = np.nanmean(att_matrix[valid], axis=1)
    event_means = np.nanmean(att_matrix[valid], axis=0)
    low, high, pvalue = _bootstrap_mean(per_camera, n_boot=n_boot, seed=seed)

    estimate = Estimate(
        name=name,
        estimand="ATT (crashes per site-month, cohort 2x2)",
        beta=float(per_camera.mean()),
        se=float(np.std(per_camera, ddof=1) / np.sqrt(len(per_camera))) if len(per_camera) > 1 else np.nan,
        pvalue=pvalue,
        ci_low=low,
        ci_high=high,
        n_obs=int(valid.sum() * post_months),
        n_clusters=int(valid.sum()),
        scale="level",
        method="Matched 2x2 DiD per camera, average across cameras, bootstrap by camera",
        details={
            "bootstrap": "percentile, camera-level",
            "n_cameras": int(valid.sum()),
            "per_camera_sd": float(np.std(per_camera, ddof=1)) if len(per_camera) > 1 else np.nan,
        },
    )
    event_table = pd.DataFrame(
        {
            "event_month": np.arange(post_months),
            "att": event_means,
            "n_cameras": valid.sum(),
        }
    )
    return estimate, event_table


def wild_cluster_bootstrap(
    y: np.ndarray,
    x: np.ndarray,
    fe_codes: Sequence[np.ndarray],
    cluster_codes: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Wild cluster bootstrap-t p-value for a single regressor (Rademacher weights).

    The null imposes beta = 0.  Residuals from the restricted model are
    re-signed at the cluster level, which is the standard remedy when the number
    of clusters is modest.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 2 and x.shape[1] != 1:
        raise ValueError("wild_cluster_bootstrap supports exactly one regressor")

    y_t = absorb(y, fe_codes)
    x_t = absorb(x, fe_codes)
    denominator = float(x_t @ x_t)
    beta = float(x_t @ y_t) / denominator
    resid_restricted = absorb(y, fe_codes)  # beta = 0 under the null
    n_clusters = int(cluster_codes.max()) + 1

    def statistic(y_values: np.ndarray) -> float:
        y_star = absorb(y_values, fe_codes)
        b = float(x_t @ y_star) / denominator
        resid = y_star - b * x_t
        meat = np.array(
            [((x_t * resid)[cluster_codes == g]).sum() ** 2 for g in range(n_clusters)]
        ).sum()
        variance = meat / denominator**2
        variance *= (n_clusters / (n_clusters - 1))
        se = np.sqrt(max(variance, 1e-18))
        return b / se

    observed = statistic(y)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_boot):
        weights = rng.choice([-1.0, 1.0], size=n_clusters)
        y_star = y - resid_restricted + resid_restricted * weights[cluster_codes]
        if abs(statistic(y_star)) >= abs(observed):
            exceed += 1
    return {
        "beta": beta,
        "t_stat": observed,
        "pvalue": (exceed + 1) / (n_boot + 1),
        "n_boot": n_boot,
    }


def event_study_fe(
    panel: pd.DataFrame,
    reference_quarter: int = -1,
    cluster_col: str = "stack",
) -> pd.DataFrame:
    """Quarterly event-study coefficients from the within estimator.

    Unit fixed effects absorb time-invariant site differences; stack-by-quarter
    fixed effects absorb every shock common to a camera and its matched controls.
    The reference quarter is omitted, so effects are relative to it.
    """
    work = panel.copy()
    work["stack_rel"] = work["stack"].astype(str) + "|" + work["rel"].astype(str)
    quarters = sorted(work["quarter"].unique())
    if reference_quarter not in quarters:
        raise ValueError(f"reference quarter {reference_quarter} not present in the panel")

    columns: list[np.ndarray] = []
    names: list[str] = []
    kept_quarters: list[int] = []
    for quarter in quarters:
        if quarter == reference_quarter:
            continue
        columns.append(((work["treated"] == 1) & (work["quarter"] == quarter)).to_numpy(float))
        names.append(f"q_{quarter}".replace("-", "m"))
        kept_quarters.append(int(quarter))

    X = np.column_stack(columns)
    fe = [_codes(work["unit"]), _codes(work["stack_rel"])]
    result = feols(
        work["y"].to_numpy(float),
        X,
        fe_codes=fe,
        cluster_codes=_codes(work[cluster_col]),
        names=names,
        n_fe_params=_fe_param_count(work, ["unit", "stack_rel"]),
    )

    table = pd.DataFrame(
        {
            "quarter": kept_quarters,
            "beta": result["beta"],
            "se": result["se"],
            "pvalue": result["pvalue"],
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
        }
    )
    reference = pd.DataFrame(
        [
            {
                "quarter": reference_quarter,
                "beta": 0.0,
                "se": 0.0,
                "pvalue": np.nan,
                "ci_low": 0.0,
                "ci_high": 0.0,
            }
        ]
    )
    table = pd.concat([table, reference], ignore_index=True).sort_values("quarter")

    pre = table.loc[table["quarter"] < reference_quarter]
    if len(pre):
        from scipy import stats as st

        idx = [names.index(f"q_{int(q)}".replace("-", "m")) for q in pre["quarter"]]
        beta_pre = result["beta"][idx]
        cov_pre = result["cov"][np.ix_(idx, idx)]
        wald = float(beta_pre @ np.linalg.pinv(cov_pre) @ beta_pre)
        table.attrs["pre_trend_wald"] = wald
        table.attrs["pre_trend_df"] = len(idx)
        table.attrs["pre_trend_pvalue"] = float(st.chi2.sf(wald, df=len(idx)))
        table.attrs["pre_trend_note"] = (
            "A joint test cannot prove parallel trends; it can only fail to reject "
            "them. Low power is expected with a few dozen cameras."
        )
    return table
