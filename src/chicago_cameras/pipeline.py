"""End-to-end research pipeline.

Run with::

    python -m chicago_cameras.pipeline --config config/analysis.yaml

Stages: acquire -> clean -> project -> design -> estimate -> event study ->
synthetic control -> robustness -> figures -> results artefacts.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .acquisition import acquire, specs_from_config
from .cleaning import clean_cameras, clean_crashes, clean_violations
from .config import load_config, path_for, write_json
from .design import DesignConfig, StudyData, build_design
from .did import (
    baseline_negative_binomial,
    baseline_ols,
    baseline_poisson,
    cohort_did,
    cohort_event_effects,
    did_conditional_poisson,
    did_fe_ols,
    event_study_fe,
    naive_before_after,
    poisson_dispersion,
    wild_cluster_bootstrap,
)
from .geospatial import project_lonlat
from .robustness import (
    evaluate_design,
    geographic_placebo,
    placebo_summary,
    sensitivity_table,
)
from .synthetic_control import (
    in_time_placebo,
    leave_one_out,
    placebo_in_space,
    placebo_pvalues,
    run_synthetic_control,
)
from .visualization import (
    plot_balance,
    plot_event_study,
    plot_main_estimates,
    plot_map,
    plot_placebo_distribution,
    plot_raw_trends,
    plot_sensitivity,
    plot_severity,
    plot_subgroups,
    plot_synthetic_control,
)

LOGGER = logging.getLogger(__name__)

PRIMARY = "did_conditional_poisson"


def _ticks() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def prepare_data(
    config: dict[str, Any],
    refresh: bool = False,
    reuse_processed: bool = True,
) -> tuple[StudyData, dict[str, Any]]:
    """Acquire, clean and project all tables; persist processed parquet files."""
    raw_dir = path_for(config, "raw")
    processed_dir = path_for(config, "processed")
    quality_path = processed_dir / "quality_report.json"
    data_files = {
        "cameras": processed_dir / "cameras.parquet",
        "crashes": processed_dir / "crashes.parquet",
        "violations": processed_dir / "violations.parquet",
    }
    fingerprint = {
        "bbox": config["data"]["bbox"],
        "start_month": config["data"]["start_month"],
        "reporting_lag_months": int(config["data"]["reporting_lag_months"]),
        "epsg": int(config["spatial"]["crs_epsg"]),
        "datasets": {
            name: entry["dataset_id"] for name, entry in config["data"]["datasets"].items()
        },
    }

    if reuse_processed and not refresh and quality_path.exists() and all(
        path.exists() for path in data_files.values()
    ):
        cached_quality = json.loads(quality_path.read_text())
        if cached_quality.get("fingerprint") == fingerprint:
            LOGGER.info("reusing processed parquet files in %s", processed_dir)
            cameras = pd.read_parquet(data_files["cameras"])
            crashes = pd.read_parquet(data_files["crashes"])
            violations = pd.read_parquet(data_files["violations"])
            months = pd.date_range(
                cached_quality["analysis"]["first_month"],
                cached_quality["analysis"]["last_month"],
                freq="MS",
            )
            data = StudyData(cameras, crashes, months, metric_epsg=int(config["spatial"]["crs_epsg"]))
            data.violations = violations  # type: ignore[attr-defined]
            return data, cached_quality

    datasets = acquire(specs_from_config(config), raw_dir, refresh=refresh)

    cameras, camera_quality = clean_cameras(datasets["cameras"], config["data"]["bbox"])
    crashes, crash_quality = clean_crashes(
        datasets["crashes"],
        config["data"]["bbox"],
        start_month=config["data"]["start_month"],
        reporting_lag_months=int(config["data"]["reporting_lag_months"]),
    )
    violations, violation_quality = clean_violations(datasets["violations"])

    epsg = int(config["spatial"]["crs_epsg"])
    cameras[["x", "y"]] = project_lonlat(cameras["lon"].to_numpy(), cameras["lat"].to_numpy(), epsg)
    crashes[["x", "y"]] = project_lonlat(crashes["lon"].to_numpy(), crashes["lat"].to_numpy(), epsg)

    months = pd.date_range(
        crashes["month"].min(), crashes["month"].max(), freq="MS"
    )
    crashes["month_index"] = months.get_indexer(crashes["month"])
    if (crashes["month_index"] < 0).any():  # pragma: no cover - defensive
        raise RuntimeError("crash months are not aligned to the month index")

    quality = {
        "generated_at_utc": _ticks(),
        "fingerprint": fingerprint,
        "cameras": camera_quality.to_dict(),
        "crashes": crash_quality.to_dict(),
        "violations": violation_quality.to_dict(),
        "analysis": {
            "months": len(months),
            "first_month": str(months[0].date()),
            "last_month": str(months[-1].date()),
            "crashes": len(crashes),
            "cameras": len(cameras),
            "epsg": epsg,
        },
    }

    cameras.to_parquet(processed_dir / "cameras.parquet", index=False)
    crashes.to_parquet(processed_dir / "crashes.parquet", index=False)
    violations.to_parquet(processed_dir / "violations.parquet", index=False)
    write_json(processed_dir / "quality_report.json", quality)

    data = StudyData(cameras, crashes, months, metric_epsg=epsg)
    data.violations = violations  # type: ignore[attr-defined]
    return data, quality


def _design_config(config: dict[str, Any]) -> DesignConfig:
    spatial = config["spatial"]
    design = config["design"]
    return DesignConfig(
        radius_m=float(spatial["radius_m"]),
        buffer_m=float(spatial["buffer_m"]),
        grid_m=float(spatial["grid_m"]),
        grid_offset_m=float(spatial.get("grid_offset_m", 0)),
        min_control_crashes=int(spatial["min_control_crashes"]),
        min_active_month_share=float(spatial["min_active_month_share"]),
        max_match_distance_m=float(spatial["max_match_distance_m"]),
        n_controls=int(design["n_controls"]),
        pre_months=int(design["pre_months"]),
        post_months=int(design["post_months"]),
        treatment_month_offset=int(design["treatment_month_offset"]),
        warning_days=int(design.get("warning_days", 0)),
        overlap_rule=str(design["overlap_rule"]),
        min_pre_months=int(design.get("min_pre_months", 0)) or None,
    )


def balance_table(panel: pd.DataFrame, cluster_col: str = "stack") -> pd.DataFrame:
    """Pre-period balance and differential pre-trend for treated vs controls."""
    import statsmodels.formula.api as smf

    pre = panel.loc[panel["post"] == 0].copy()
    summary = (
        pre.groupby("treated")["y"]
        .agg(["mean", "std", "count"])
        .rename(index={0: "control", 1: "camera"})
    )
    mean_t = float(summary.loc["camera", "mean"])
    mean_c = float(summary.loc["control", "mean"])
    var_t = float(summary.loc["camera", "std"] ** 2)
    var_c = float(summary.loc["control", "std"] ** 2)
    pooled = np.sqrt((var_t + var_c) / 2) if (var_t + var_c) > 0 else np.nan

    pre["rel_centred"] = pre["rel"] + 0.5
    trend = smf.ols("y ~ treated * rel_centred", data=pre).fit(
        cov_type="cluster", cov_kwds={"groups": pre[cluster_col].values}
    )
    low, high = trend.conf_int().loc["treated:rel_centred"]
    table = pd.DataFrame(
        [
            {
                "group": "camera",
                "pre_mean_crashes": mean_t,
                "pre_sd": float(summary.loc["camera", "std"]),
                "n_site_months": int(summary.loc["camera", "count"]),
            },
            {
                "group": "control",
                "pre_mean_crashes": mean_c,
                "pre_sd": float(summary.loc["control", "std"]),
                "n_site_months": int(summary.loc["control", "count"]),
            },
        ]
    )
    table.attrs["standardised_mean_difference"] = (
        float((mean_t - mean_c) / pooled) if pooled and pooled > 0 else np.nan
    )
    table.attrs["pre_trend_interaction_per_month"] = float(
        trend.params.get("treated:rel_centred", np.nan)
    )
    table.attrs["pre_trend_ci"] = (float(low), float(high))
    table.attrs["pre_trend_pvalue"] = float(trend.pvalues.get("treated:rel_centred", np.nan))
    return table


def subgroup_analysis(
    design,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Exploratory heterogeneity using per-camera 2x2 effects."""

    camera_rows = sorted(design.matches.matches)
    att_matrix, valid = cohort_event_effects(
        design.camera_counts,
        design.control_counts,
        design.matches.matches,
        design.camera_go_idx,
        design.config.post_months,
    )
    per_camera = pd.DataFrame(
        {
            "camera_row": camera_rows,
            "att": np.nanmean(att_matrix, axis=1),
            "valid": valid,
        }
    ).loc[lambda d: d["valid"]]
    cameras = design.cameras.loc[per_camera["camera_row"]]
    per_camera["x"] = cameras["x"].to_numpy()
    per_camera["y"] = cameras["y"].to_numpy()
    per_camera["go_live"] = cameras["go_live"].to_numpy()
    per_camera["pre_level"] = design.matches.table.set_index("camera_row").loc[
        per_camera["camera_row"], "camera_pre_level"
    ].to_numpy()

    # Geographic regions from k-means on projected camera coordinates.
    k = min(4, max(2, len(per_camera) // 5))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(per_camera[["x", "y"]])
    centres = km.cluster_centers_
    labels = []
    for cluster in km.labels_:
        cx, cy = centres[cluster]
        labels.append(("N" if cy > np.median(per_camera["y"]) else "S") + ("E" if cx > np.median(per_camera["x"]) else "W"))
    per_camera["region"] = [f"{label} (cluster {int(c)})" for label, c in zip(labels, km.labels_)]

    per_camera["baseline_risk"] = pd.qcut(
        per_camera["pre_level"], 3, labels=["Low baseline", "Middle baseline", "High baseline"],
        duplicates="drop",
    )
    per_camera["era"] = np.where(
        pd.to_datetime(per_camera["go_live"]).dt.year < 2020, "2015-2019", "2020+"
    )

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for dimension in ("region", "baseline_risk", "era"):
        for group, subset in per_camera.groupby(dimension, observed=True):
            values = subset["att"].to_numpy(float)
            if len(values) < 2:
                continue
            draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
            low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            rows.append(
                {
                    "dimension": dimension,
                    "group": str(group),
                    "n_cameras": len(values),
                    "att_level": float(values.mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "pvalue": float((np.abs(draws - values.mean()) >= abs(values.mean())).mean()),
                }
            )

    table = pd.DataFrame(rows)
    if not table.empty:
        table = _bh_adjust(table, "pvalue")
    return table


def _bh_adjust(table: pd.DataFrame, column: str, alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg adjusted p-values for exploratory comparisons."""
    out = table.copy()
    p = out[column].fillna(1).to_numpy()
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    out[f"{column}_bh"] = result
    out["significant_bh"] = out[f"{column}_bh"] <= alpha
    return out


def synthetic_control_analysis(design, config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Pooled synthetic control for the largest go-live cohort."""
    camera_rows = np.array(sorted(design.matches.matches), dtype=int)
    cameras = design.cameras.loc[camera_rows]
    go_idx = cameras["go_idx"].to_numpy(int)

    values, counts_by_idx = np.unique(go_idx, return_counts=True)
    g = int(values[np.argmax(counts_by_idx)])
    cohort_rows = camera_rows[np.flatnonzero(go_idx == g)]
    y_treated = design.camera_counts[cohort_rows].mean(axis=0).astype(float)
    X_donors = design.control_counts.T.astype(float)
    post_months = design.config.post_months

    fit = run_synthetic_control(
        y_treated,
        X_donors,
        g=g,
        post_len=post_months,
        max_donors=int(config["synthetic_control"].get("max_donors", 150)),
    )
    threshold = float(config["synthetic_control"].get("min_pre_fit_r2", 0.5))
    informative = bool(fit.pre_r2 >= threshold)

    placebo = placebo_in_space(
        y_treated,
        X_donors,
        g=g,
        treated_group_size=len(cohort_rows),
        n_placebo=int(config["synthetic_control"].get("n_placebo", 200)),
        post_len=post_months,
        seed=seed,
        max_donors=int(config["synthetic_control"].get("max_donors", 150)),
    )
    pvalues = placebo_pvalues(fit, placebo)

    in_time = []
    for offset in config["synthetic_control"].get("in_time_placebo_offsets_months", [-24, -12]):
        try:
            in_time.append(in_time_placebo(y_treated, X_donors, g, offset, post_months, max_donors=150))
        except ValueError as exc:
            LOGGER.warning("in-time placebo %s skipped: %s", offset, exc)

    loo = leave_one_out(
        y_treated,
        X_donors,
        g,
        post_months,
        fit.donor_rows,
        fit.weights,
        top_k=int(config["synthetic_control"].get("top_weighted_donors_loo", 5)),
        max_donors=int(config["synthetic_control"].get("max_donors", 150)),
    )

    return {
        "cohort_go_month": str(design.cameras.loc[cohort_rows, "go_month"].iloc[0].date()),
        "cohort_size": len(cohort_rows),
        "cohort_camera_uids": design.cameras.loc[cohort_rows, "camera_uid"].tolist(),
        "g_index": g,
        "fit": fit,
        "pvalues": pvalues,
        "placebo": placebo,
        "in_time": pd.DataFrame(in_time),
        "leave_one_out": loo,
        "informative": informative,
        "fit_threshold_r2": threshold,
        "_treated_series": y_treated,
    }


def mechanism_analysis(violations: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Violations over camera life, with explicit non-causal interpretation."""
    if violations is None or violations.empty:
        return {"available": False}

    frame = violations.copy()
    first = frame.groupby("camera_id")["month"].transform("min")
    frame["months_since_first"] = (
        (frame["month"].dt.year - first.dt.year) * 12 + (frame["month"].dt.month - first.dt.month)
    )
    for value in (12, 24, 36):
        frame[f"years_since_first_{value}"] = frame["months_since_first"] / value

    curve = (
        frame.loc[frame["months_since_first"].between(1, 48)]
        .groupby("months_since_first")
        .agg(mean_violations=("violations", "mean"), n_cameras=("camera_id", "nunique"))
        .reset_index()
    )

    # Camera fixed effects via within transformation, cluster-robust by camera.
    work = frame.loc[frame["months_since_first"].between(1, 48)].copy()
    work["log_violations"] = np.log1p(work["violations"].clip(lower=0))
    codes, _ = pd.factorize(work["camera_id"], sort=True)
    from .did import _cluster_meat, absorb

    y = absorb(work["log_violations"].to_numpy(float), [codes])
    x = absorb((work["months_since_first"] / 12).to_numpy(float), [codes])
    denominator = float(x @ x)
    beta = float(x @ y) / denominator
    resid = y - beta * x
    n_clusters = int(codes.max()) + 1
    meat = _cluster_meat(x[:, None], resid, codes)
    var = float(meat[0, 0] / denominator**2) * (n_clusters / (n_clusters - 1))
    se = float(np.sqrt(max(var, 1e-18)))

    return {
        "available": True,
        "curve": curve,
        "slope_per_year_log": beta,
        "slope_se": se,
        "slope_pct_per_year": float((np.exp(beta) - 1) * 100),
        "n_camera_months": len(work),
        "n_cameras": int(n_clusters),
        "caveat": (
            "Violations are enforcement output, not speeding behaviour or exposure. "
            "A decline is consistent with adaptation, enforcement saturation, or "
            "changes in camera operation, and cannot establish a causal mechanism."
        ),
    }


def run(config: dict[str, Any], refresh: bool = False, quick: bool = False) -> dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    seed = int(config["inference"]["seed"])
    results_dir = path_for(config, "results")
    figures_dir = path_for(config, "figures")
    tables_dir = path_for(config, "tables")
    n_boot = int(config["inference"]["n_bootstrap"])
    if quick:
        n_boot = 99

    LOGGER.info("stage 1/8: data acquisition and cleaning")
    data, quality = prepare_data(config, refresh=refresh)
    write_json(results_dir / "provenance.json", quality)

    LOGGER.info("stage 2/8: baseline design")
    design_config = _design_config(config)
    design = build_design(data, design_config, outcome="all")
    LOGGER.info(
        "matched cameras=%s of %s in window; control sites=%s",
        design.n_cameras,
        design.diagnostics["n_cameras_in_window"],
        design.diagnostics["n_control_candidates"],
    )

    LOGGER.info("stage 2b/8: faithful replication of the original sample")
    replication_config = replace(
        design_config,
        max_match_distance_m=None,   # original code matched against the whole city
        overlap_rule="keep_all",     # original code kept overlapping cameras
        min_active_month_share=0.0,  # original code had no activity screen
    )
    replication_data = data.subset_start("2015-01-01")
    replication_design = build_design(replication_data, replication_config, outcome="all")
    replication_estimates = {
        "ols": baseline_ols(replication_design.panel),
        "poisson": baseline_poisson(replication_design.panel),
        "conditional_poisson": did_conditional_poisson(replication_design.panel),
    }
    LOGGER.info(
        "replication sample: %s cameras, %s panel rows",
        replication_design.n_cameras,
        len(replication_design.panel),
    )

    LOGGER.info("stage 3/8: estimation")
    estimates = [
        naive_before_after(design.panel),
        baseline_ols(design.panel),
        baseline_poisson(design.panel),
        did_fe_ols(design.panel),
        did_conditional_poisson(design.panel),
    ]
    try:
        estimates.insert(3, baseline_negative_binomial(design.panel))
    except RuntimeError as exc:
        LOGGER.warning("negative binomial not estimable: %s", exc)
    dispersion = poisson_dispersion(design.panel)
    cohort, cohort_event = cohort_did(
        Y_camera=design.camera_counts,
        Y_control=design.control_counts,
        matches=design.matches.matches,
        camera_go_idx=design.camera_go_idx,
        post_months=design.config.post_months,
        n_boot=n_boot,
        seed=seed,
    )
    estimates.append(cohort)

    wild = wild_cluster_bootstrap(
        y=design.panel["y"].to_numpy(float),
        x=design.panel["tp"].to_numpy(float),
        fe_codes=[
            pd.factorize(design.panel["unit"], sort=True)[0],
            pd.factorize(
                design.panel["stack"].astype(str) + "|" + design.panel["rel"].astype(str),
                sort=True,
            )[0],
        ],
        cluster_codes=pd.factorize(design.panel["stack"], sort=True)[0],
        n_boot=int(config["inference"].get("wild_bootstrap_reps", 999)) if not quick else 99,
        seed=seed,
    )

    estimate_table = pd.DataFrame([estimate.to_row() for estimate in estimates])
    replication_rows = []
    for estimate in replication_estimates.values():
        row = estimate.to_row()
        row["method"] = f"Replication (2015+ sample): {estimate.method}"
        replication_rows.append(row)
    replication_table = pd.DataFrame(replication_rows)
    replication_table.to_csv(tables_dir / "replication_estimates.csv", index=False)
    estimate_table = pd.concat([estimate_table, replication_table], ignore_index=True)
    estimate_table.to_csv(tables_dir / "main_estimates.csv", index=False)

    LOGGER.info("stage 4/8: event study")
    event_table = event_study_fe(design.panel)
    event_table.to_csv(tables_dir / "event_study.csv", index=False)
    cohort_event.to_csv(tables_dir / "cohort_event_effects.csv", index=False)

    balance = balance_table(design.panel)
    balance.to_csv(tables_dir / "balance.csv", index=False)

    LOGGER.info("stage 5/8: synthetic control")
    sc = synthetic_control_analysis(design, config, seed=seed)
    pd.DataFrame([sc["fit"].to_dict()]).to_csv(tables_dir / "synthetic_control.csv", index=False)
    sc["leave_one_out"].to_csv(tables_dir / "sc_leave_one_out.csv", index=False)
    sc["in_time"].to_csv(tables_dir / "sc_in_time_placebo.csv", index=False)
    sc["placebo"]["records"].to_csv(tables_dir / "sc_placebo_records.csv", index=False)

    LOGGER.info("stage 6/8: robustness")
    sensitivity = sensitivity_table(
        data,
        design_config,
        config,
        n_boot=199 if quick else 499,
        baseline_camera_rows=np.array(sorted(design.matches.matches), dtype=int),
    )
    sensitivity.to_csv(tables_dir / "sensitivity.csv", index=False)

    # Lattice phase is an arbitrary design choice. Report the spread explicitly
    # rather than selecting the most favourable phase.
    phase_mask = sensitivity["analysis"].str.contains("Lattice phase") | sensitivity[
        "analysis"
    ].str.startswith("Baseline")
    phase_values = sensitivity.loc[phase_mask, "did_pct"].dropna()
    lattice_phase = {
        "n_phases": len(phase_values),
        "mean_pct": float(phase_values.mean()) if len(phase_values) else np.nan,
        "min_pct": float(phase_values.min()) if len(phase_values) else np.nan,
        "max_pct": float(phase_values.max()) if len(phase_values) else np.nan,
        "values": phase_values.round(3).tolist(),
    }

    subgroups = subgroup_analysis(design, n_boot=199 if quick else 999, seed=seed)
    subgroups.to_csv(tables_dir / "subgroups.csv", index=False)

    severity_rows = []
    for outcome in ("injury", "severe"):
        try:
            severity_design = build_design(data, design_config, outcome=outcome)
            summary = evaluate_design(
                severity_design, n_boot=n_boot, seed=seed, label=outcome
            )
            severity_rows.append(
                {
                    "outcome": outcome,
                    "did_pct": summary["did_pct"],
                    "ci_low": summary["did_pct_ci_low"],
                    "ci_high": summary["did_pct_ci_high"],
                    "pvalue": summary["did_pvalue"],
                    "n_cameras": summary["n_cameras"],
                }
            )
        except (ValueError, RuntimeError) as exc:
            LOGGER.warning("severity outcome %s not estimable: %s", outcome, exc)
            severity_rows.append({"outcome": outcome, "did_pct": np.nan, "error": str(exc)})
    severity = pd.DataFrame(severity_rows)
    severity.to_csv(tables_dir / "severity.csv", index=False)

    LOGGER.info("stage 7/8: falsification")
    placebo = geographic_placebo(
        data,
        design_config,
        n_placebo=50 if quick else int(config["robustness"].get("n_geographic_placebos", 200)),
        seed=seed,
        n_boot=99,
    )
    placebo.to_csv(tables_dir / "placebo_geography.csv", index=False)

    primary_row = estimate_table.loc[
        estimate_table["method"].str.contains("Conditional Poisson")
        & ~estimate_table["method"].str.contains("Replication")
    ].iloc[0]
    real_for_placebo = {
        "did_pct": float(primary_row["pct"]),
        "cohort_att_level": float(cohort.beta),
    }
    placebo_stats = placebo_summary(real_for_placebo, placebo)

    mechanism = mechanism_analysis(getattr(data, "violations", None), seed=seed)
    if mechanism.get("available"):
        mechanism["curve"].to_csv(tables_dir / "violations_curve.csv", index=False)

    LOGGER.info("stage 8/8: figures and artefacts")
    artefacts = {
        "generated_at_utc": _ticks(),
        "config": config.get("_config_path"),
        "sample": {
            "crashes": quality["analysis"]["crashes"],
            "cameras_total": design.diagnostics["n_cameras_total"],
            "cameras_in_window": design.diagnostics["n_cameras_in_window"],
            "cameras_matched": design.n_cameras,
            "cameras_dropped_overlap": design.diagnostics["n_cameras_dropped_overlap"],
            "control_candidates": design.diagnostics["n_control_candidates"],
            "months": quality["analysis"]["months"],
            "first_month": quality["analysis"]["first_month"],
            "last_month": quality["analysis"]["last_month"],
            "panel_rows": len(design.panel),
        },
        "design": asdict(design_config),
        "diagnostics": design.diagnostics,
        "validation": design.validation,
        "balance": {
            "standardised_mean_difference": balance.attrs.get("standardised_mean_difference"),
            "pre_trend_interaction_per_month": balance.attrs.get("pre_trend_interaction_per_month"),
            "pre_trend_ci": balance.attrs.get("pre_trend_ci"),
            "pre_trend_pvalue": balance.attrs.get("pre_trend_pvalue"),
        },
        "estimates": estimate_table.to_dict(orient="records"),
        "replication": {
            "n_cameras": replication_design.n_cameras,
            "n_panel_rows": len(replication_design.panel),
            "first_month": str(replication_data.months[0].date()),
            "last_month": str(replication_data.months[-1].date()),
            "estimates": replication_table.to_dict(orient="records"),
            "original_notebook": {
                "n_cameras": 33,
                "panel_rows": 11880,
                "ols_beta": -0.356,
                "ols_ci": [-0.711, -0.000],
                "ols_pvalue": 0.0497,
                "poisson_pct": -7.3,
                "poisson_ci_pct": [-13.6, -0.6],
                "poisson_pvalue": 0.0342,
            },
            "note": (
                "The replication uses the original 2015 start, citywide matching and no "
                "overlap handling. Small differences are expected because the portal "
                "extract date, the projection (UTM instead of a flat approximation) and "
                "duplicate handling differ."
            ),
        },
        "wild_cluster_bootstrap": wild,
        "poisson_dispersion": dispersion,
        "cohort_event_effects": cohort_event.to_dict(orient="records"),
        "event_study": event_table.drop(columns=["se"]).to_dict(orient="records"),
        "event_study_pre_trend": {
            key: event_table.attrs.get(key) for key in event_table.attrs
        },
        "synthetic_control": {
            **sc["fit"].to_dict(),
            **sc["pvalues"],
            "cohort_go_month": sc["cohort_go_month"],
            "cohort_size": sc["cohort_size"],
            "informative": sc["informative"],
            "fit_threshold_r2": sc["fit_threshold_r2"],
            "in_time_placebo": sc["in_time"].to_dict(orient="records"),
            "leave_one_out": sc["leave_one_out"].to_dict(orient="records"),
            "placebo_median_ratio": float(sc["placebo"]["records"]["rmspe_ratio"].median())
            if not sc["placebo"]["records"].empty
            else np.nan,
        },
        "subgroups": subgroups.to_dict(orient="records"),
        "severity": severity.to_dict(orient="records"),
        "sensitivity": sensitivity.to_dict(orient="records"),
        "lattice_phase": lattice_phase,
        "placebo_geography": placebo_stats,
        "mechanism": {
            "available": mechanism.get("available", False),
            "slope_per_year_log": mechanism.get("slope_per_year_log"),
            "slope_pct_per_year": mechanism.get("slope_pct_per_year"),
            "slope_se": mechanism.get("slope_se"),
            "n_cameras": mechanism.get("n_cameras"),
            "n_camera_months": mechanism.get("n_camera_months"),
            "caveat": mechanism.get("caveat"),
        },
        "quality": {
            "crashes_dropped": quality["crashes"]["dropped"],
            "cameras_dropped": quality["cameras"]["dropped"],
            "injury_crash_share": quality["crashes"]["injury_crash_share"],
            "severe_crash_share": quality["crashes"]["severe_crash_share"],
        },
    }
    write_json(results_dir / "estimates.json", artefacts)

    # Persist analysis substrates for the dashboard.
    design.panel.to_parquet(results_dir / "analysis_panel.parquet", index=False)
    sites = design.cameras.loc[sorted(design.matches.matches)].copy()
    sites["role"] = "matched camera"
    controls = pd.DataFrame(
        {"camera_uid": design.control_ids, "x": design.control_xy[:, 0], "y": design.control_xy[:, 1], "role": "comparison site"}
    )
    controls["go_month"] = pd.NaT
    controls["go_live"] = pd.NaT
    sites_out = pd.concat(
        [
            sites[["camera_uid", "x", "y", "go_month", "go_live", "role"]],
            controls[["camera_uid", "x", "y", "go_month", "go_live", "role"]],
        ],
        ignore_index=True,
    )
    # Convert projected coordinates back to lon/lat for mapping.
    from pyproj import Transformer

    transformer = Transformer.from_crs(int(config["spatial"]["crs_epsg"]), 4326, always_xy=True)
    lon, lat = transformer.transform(sites_out["x"].to_numpy(), sites_out["y"].to_numpy())
    sites_out["lon"] = lon
    sites_out["lat"] = lat
    sites_out.to_parquet(results_dir / "sites.parquet", index=False)

    LOGGER.info("generating figures")
    plot_map(design, data, figures_dir)
    plot_balance(balance, design.panel, figures_dir)
    plot_raw_trends(design.panel, figures_dir)
    plot_main_estimates(estimate_table, design.panel, figures_dir)
    plot_event_study(event_table, figures_dir)
    plot_synthetic_control(sc, figures_dir)
    plot_placebo_distribution(sc, placebo, float(primary_row["pct"]), figures_dir)
    plot_subgroups(subgroups, figures_dir)
    plot_severity(severity, figures_dir)
    plot_sensitivity(sensitivity, figures_dir)

    LOGGER.info("done: results in %s", results_dir)
    return artefacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--refresh", action="store_true", help="Re-download raw data")
    parser.add_argument("--quick", action="store_true", help="Fewer bootstrap draws (development)")
    args = parser.parse_args()

    config = load_config(args.config)
    run(config, refresh=args.refresh, quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
