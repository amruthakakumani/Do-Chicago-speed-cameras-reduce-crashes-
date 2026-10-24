"""Publication-quality figures generated from analysis outputs.

Every figure is written as both PNG and SVG.  Captions and footnotes state the
estimand, sample and units so that a reader cannot mistake a descriptive plot
for a causal estimate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

INK = "#1f2933"
MUTED = "#7b8794"
CAMERA = "#c0392b"
CONTROL = "#2c7fb8"
ACCENT = "#6a51a3"
GRID = "#d9e2ec"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "font.family": "DejaVu Sans",
            "figure.dpi": 120,
        }
    )


def _save(fig: plt.Figure, outdir: Path, name: str, note: str = "") -> None:
    _style()
    if note:
        fig.text(0.01, -0.06, note, fontsize=6.8, color=MUTED, ha="left", va="top")
    outdir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(outdir / f"{name}.{suffix}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def _pre_mean(panel: pd.DataFrame) -> float:
    return float(panel.loc[(panel["treated"] == 1) & (panel["post"] == 0), "y"].mean())


def plot_map(design, data, outdir: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(7.6, 9.2))
    sample = data.crashes.sample(min(len(data.crashes), 60_000), random_state=0)
    ax.hexbin(sample["x"], sample["y"], gridsize=90, cmap="Greys", mincnt=1, alpha=0.35, linewidths=0)

    cameras = design.cameras
    matched = design.matches.matches
    matched_rows = sorted(matched)
    other = cameras.loc[~cameras.index.isin(matched_rows)]
    ax.scatter(other["x"], other["y"], s=16, c=MUTED, marker="^", alpha=0.6, label="Camera (not in matched sample)")
    ax.scatter(
        design.cameras.loc[matched_rows, "x"],
        design.cameras.loc[matched_rows, "y"],
        s=42,
        c=CAMERA,
        marker="^",
        edgecolor="white",
        linewidth=0.5,
        label=f"Matched camera (n={len(matched_rows)})",
        zorder=5,
    )
    ax.scatter(
        design.control_xy[:, 0],
        design.control_xy[:, 1],
        s=12,
        c=CONTROL,
        alpha=0.55,
        marker="o",
        linewidth=0,
        label=f"Comparison site (n={len(design.control_xy)})",
    )
    used_controls = design.control_xy[np.unique(design.matches.controls_used)]
    ax.scatter(
        used_controls[:, 0],
        used_controls[:, 1],
        s=26,
        facecolors="none",
        edgecolors=ACCENT,
        linewidth=0.8,
        label="Comparison site used in at least one match",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("UTM zone 16N easting (m)")
    ax.set_ylabel("UTM zone 16N northing (m)")
    ax.set_title("Speed cameras and comparison sites, Chicago")
    ax.legend(loc="upper left", fontsize=8)
    _save(
        fig,
        outdir,
        "01_sites_map",
        "Treatment catchment = 250 m around a camera. Comparison sites are lattice points "
        ">600 m from every camera with >=100 crashes. Grey hexagons: crash density (all years).",
    )


def plot_balance(balance: pd.DataFrame, panel: pd.DataFrame, outdir: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    frame = balance.set_index("group")
    means = [frame.loc["camera", "pre_mean_crashes"], frame.loc["control", "pre_mean_crashes"]]
    sds = [frame.loc["camera", "pre_sd"], frame.loc["control", "pre_sd"]]
    axes[0].bar(["Camera\ncatchments", "Matched\ncomparison sites"], means, yerr=sds, capsize=4,
                color=[CAMERA, CONTROL], alpha=0.85)
    axes[0].set_ylabel("Crashes per site-month, pre-period")
    axes[0].set_title("Pre-period level")
    smd = balance.attrs.get("standardised_mean_difference", np.nan)
    axes[0].text(0.02, 0.95, f"Standardised mean difference: {smd:.2f}", transform=axes[0].transAxes,
                 va="top", fontsize=8.5, color=INK)

    pre = panel.loc[panel["post"] == 0]
    axes[1].hist(
        [pre.loc[pre["treated"] == 0, "y"], pre.loc[pre["treated"] == 1, "y"]],
        bins=np.arange(0, 20, 1),
        density=True,
        color=[CONTROL, CAMERA],
        alpha=0.55,
        label=["Comparison sites", "Camera catchments"],
    )
    axes[1].set_xlabel("Crashes per site-month, pre-period")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Pre-period distribution")
    axes[1].legend()
    _save(
        fig,
        outdir,
        "02_balance",
        "Error bars: one standard deviation. Matching is on pre-period level and slope; "
        "remaining level differences are absorbed by site fixed effects in the preferred model.",
    )


def plot_raw_trends(panel: pd.DataFrame, outdir: Path) -> None:
    _style()
    agg = panel.groupby(["treated", "rel"])["y"].agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    for treated, colour, label in ((0, CONTROL, "Matched comparison sites"), (1, CAMERA, "Camera catchments")):
        subset = agg.loc[agg["treated"] == treated]
        ax.plot(subset["rel"], subset["mean"], color=colour, lw=1.8, label=label)
        ax.fill_between(
            subset["rel"],
            subset["mean"] - 1.96 * subset["sem"],
            subset["mean"] + 1.96 * subset["sem"],
            color=colour,
            alpha=0.15,
            linewidth=0,
        )
    ax.axvline(-0.5, color=INK, ls="--", lw=1)
    ax.annotate("Camera go-live", xy=(-0.3, ax.get_ylim()[1]), xytext=(-0.3, ax.get_ylim()[1]),
                fontsize=8, va="top", color=INK)
    ax.set_xlabel("Months relative to camera go-live")
    ax.set_ylabel("Crashes per site-month within 250 m")
    ax.set_title("Raw trends: camera catchments versus matched comparison sites")
    ax.legend(loc="upper left")
    _save(
        fig,
        outdir,
        "03_raw_trends",
        "Unadjusted means with 95% confidence bands. Parallel pre-trends are an identifying "
        "assumption, not a finding; see the event study and the pre-trend test.",
    )


def _short_label(method: str) -> str:
    """Compact labels for the forest plot; full descriptions live in the tables."""
    replication = method.startswith("Replication")
    core = method.removeprefix("Replication (2015+ sample): ")
    if "Treated-only" in core:
        label = "Naive before/after (no comparison)"
    elif "Stacked OLS" in core:
        label = "DiD OLS (original specification)"
    elif "Stacked Poisson" in core:
        label = "DiD Poisson (original specification)"
    elif "Negative Binomial" in core:
        label = "DiD Negative Binomial (NB2)"
    elif "Two-way FE OLS" in core:
        label = "FE OLS: unit + stack-month FE"
    elif "Conditional Poisson" in core:
        label = "Conditional Poisson FE (preferred)"
    elif "Matched 2x2" in core:
        label = "Cohort 2x2 DiD (camera bootstrap)"
    else:
        label = core
    return f"Replication: {label}" if replication else label


def _estimates_for_plot(estimate_table: pd.DataFrame, pre_mean: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in estimate_table.iterrows():
        if row.get("scale") == "log" and pd.notna(row.get("pct")):
            value, low, high = row["pct"], row.get("pct_ci_low"), row.get("pct_ci_high")
            label = _short_label(str(row["method"]))
        elif row.get("scale") == "level" and pd.notna(row.get("estimate")):
            value = 100 * row["estimate"] / pre_mean if pre_mean else np.nan
            low = 100 * row.get("ci_low", np.nan) / pre_mean if pre_mean else np.nan
            high = 100 * row.get("ci_high", np.nan) / pre_mean if pre_mean else np.nan
            label = _short_label(str(row["method"])) + " (level, % of pre-mean)"
        else:
            continue
        rows.append({"label": label, "pct": value, "low": low, "high": high, "naive": "Naive" in label})
    return pd.DataFrame(rows)


def plot_main_estimates(estimate_table: pd.DataFrame, panel: pd.DataFrame, outdir: Path) -> None:
    _style()
    frame = _estimates_for_plot(estimate_table, _pre_mean(panel))
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(9.8, 0.62 * len(frame) + 1.8))
    y = np.arange(len(frame))
    colours = [MUTED if naive else (ACCENT if "Conditional" in label or "Two-way" in label else CONTROL)
               for label, naive in zip(frame["label"], frame["naive"])]
    ax.errorbar(
        frame["pct"],
        y,
        xerr=[frame["pct"] - frame["low"], frame["high"] - frame["pct"]],
        fmt="o",
        capsize=3,
        color=INK,
        ecolor=MUTED,
        markersize=6,
        zorder=3,
    )
    ax.scatter(frame["pct"], y, c=colours, s=46, zorder=4)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["label"], fontsize=8)
    ax.set_xlabel("Estimated change in crashes (%)   negative = fewer crashes")
    ax.set_title("Main estimates: cameras versus matched comparison sites")
    ax.invert_yaxis()
    _save(
        fig,
        outdir,
        "04_main_estimates",
        "Intervals are 95%. Level estimates are divided by the pre-period treated mean for display only; "
        "purple markers are the preferred fixed-effects specifications. Confidence intervals are "
        "cluster-robust at the camera level where available.",
    )


def plot_event_study(event_table: pd.DataFrame, outdir: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.errorbar(
        event_table["quarter"],
        event_table["beta"],
        yerr=[event_table["beta"] - event_table["ci_low"], event_table["ci_high"] - event_table["beta"]],
        fmt="o",
        capsize=3,
        color=INK,
        ecolor=MUTED,
        markersize=4.5,
    )
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(-0.5, color=CAMERA, ls="--", lw=1)
    ax.set_xlabel("Quarters relative to camera go-live (reference: quarter -1)")
    ax.set_ylabel("Change in crashes per site-month")
    ax.set_title("Event study: dynamic effects with 95% confidence intervals")
    wald_p = event_table.attrs.get("pre_trend_pvalue")
    if wald_p is not None:
        ax.text(
            0.02,
            0.05,
            f"Joint pre-trend test p={wald_p:.3f}\n(a test cannot prove parallel trends)",
            transform=ax.transAxes,
            fontsize=8,
            color=MUTED,
        )
    _save(
        fig,
        outdir,
        "05_event_study",
        "Unit and stack-by-event-month fixed effects; standard errors clustered by camera. "
        "Effects are in crashes per site-month, not percent.",
    )


def plot_synthetic_control(sc: dict[str, Any], outdir: Path) -> None:
    _style()
    fit = sc["fit"]
    months = fit.synth
    g = sc["g_index"]
    x = np.arange(len(months))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))
    treated = sc.get("_treated_series")
    axes[0].plot(x, treated, color=CAMERA, lw=1.8, label="Camera cohort (observed)")
    axes[0].plot(x, months, color=CONTROL, lw=1.8, ls="--", label="Synthetic control")
    axes[0].axvline(g, color=INK, ls=":", lw=1)
    axes[0].set_xlabel("Month index")
    axes[0].set_ylabel("Crashes per site-month")
    axes[0].set_title("Observed versus synthetic control")
    axes[0].legend(fontsize=8)

    axes[1].plot(x, fit.gap, color=CAMERA, lw=1.6)
    axes[1].axhline(0, color=INK, lw=0.8)
    axes[1].axvline(g, color=INK, ls=":", lw=1)
    axes[1].set_xlabel("Month index")
    axes[1].set_ylabel("Observed minus synthetic")
    axes[1].set_title("Gap")
    status = "adequate" if sc.get("informative") else "INADEQUATE"
    axes[1].text(
        0.02,
        0.05,
        f"Pre-period R²={fit.pre_r2:.2f} ({status} fit)\nPre-RMSPE={fit.pre_rmspe:.2f} crashes/month",
        transform=axes[1].transAxes,
        fontsize=8,
        color=CAMERA if not sc.get("informative") else MUTED,
    )
    _save(
        fig,
        outdir,
        "06_synthetic_control",
        f"Largest go-live cohort ({sc['cohort_size']} cameras, {sc['cohort_go_month']}). Donor pool: "
        "comparison sites >600 m from every camera. De-meaned simplex fit. "
        + ("Pre-period fit is poor; the synthetic control is not informative." if not sc.get("informative") else ""),
    )


def plot_placebo_distribution(sc: dict[str, Any], placebo: pd.DataFrame, primary_pct: float, outdir: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))
    records = sc["placebo"]["records"]
    if not records.empty:
        axes[0].hist(records["rmspe_ratio"], bins=24, color=CONTROL, alpha=0.7)
        axes[0].axvline(sc["fit"].rmspe_ratio, color=CAMERA, lw=2, label="Real cameras")
        axes[0].set_xlabel("Post/pre RMSPE ratio")
        axes[0].set_ylabel("Placebo draws")
        axes[0].set_title("Synthetic-control placebos in space")
        axes[0].legend(fontsize=8)

    did = placebo["did_pct"].dropna() if not placebo.empty else pd.Series(dtype=float)
    if not did.empty:
        axes[1].hist(did, bins=24, color=CONTROL, alpha=0.7)
        axes[1].axvline(primary_pct, color=CAMERA, lw=2, label="Real cameras")
        axes[1].axvline(0, color=INK, lw=0.8)
        axes[1].set_xlabel("Estimated effect on crashes (%)")
        axes[1].set_ylabel("Placebo draws")
        axes[1].set_title(f"Geographic placebos (n={len(did)})")
        axes[1].legend(fontsize=8)
    _save(
        fig,
        outdir,
        "07_placebo_distribution",
        "Placebo cameras are randomly located comparison sites with go-live months drawn from "
        "the real camera distribution; the full design is rebuilt for every draw.",
    )


def plot_subgroups(subgroups: pd.DataFrame, outdir: Path) -> None:
    _style()
    frame = subgroups.dropna(subset=["att_level"]) if not subgroups.empty else subgroups
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 0.5 * len(frame) + 1.8))
    y = np.arange(len(frame))
    ax.errorbar(
        frame["att_level"],
        y,
        xerr=[frame["att_level"] - frame["ci_low"], frame["ci_high"] - frame["att_level"]],
        fmt="o",
        capsize=3,
        color=INK,
        ecolor=MUTED,
        markersize=5,
    )
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row.dimension}: {row.group} (n={row.n_cameras})" for row in frame.itertuples()], fontsize=8)
    ax.set_xlabel("Change in crashes per site-month (per-camera 2x2 ATT)")
    ax.set_title("Exploratory heterogeneity")
    _save(
        fig,
        outdir,
        "08_subgroups",
        "Exploratory, not confirmatory. Bootstrap intervals resample cameras; BH-adjusted p-values "
        "are reported in results/tables/subgroups.csv because many comparisons are examined.",
    )


def plot_severity(severity: pd.DataFrame, outdir: Path) -> None:
    _style()
    frame = severity.dropna(subset=["did_pct"]) if not severity.empty else severity
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    y = np.arange(len(frame))
    ax.errorbar(
        frame["did_pct"],
        y,
        xerr=[frame["did_pct"] - frame["ci_low"], frame["ci_high"] - frame["did_pct"]],
        fmt="o",
        capsize=3,
        color=INK,
        ecolor=MUTED,
    )
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([row.outcome.capitalize() for row in frame.itertuples()])
    ax.set_xlabel("Estimated change in crashes (%)")
    ax.set_title("Crash severity outcomes")
    _save(
        fig,
        outdir,
        "09_severity",
        "Injury = injuries_total > 0. Severe = fatal or incapacitating injury. Same design and "
        "conditional Poisson specification as the main result.",
    )


def plot_sensitivity(sensitivity: pd.DataFrame, outdir: Path) -> None:
    _style()
    frame = sensitivity.loc[sensitivity["status"].eq("ok")] if "status" in sensitivity else sensitivity
    frame = frame.dropna(subset=["did_pct"])
    if frame.empty:
        return
    frame = frame.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.4, 0.42 * len(frame) + 1.8))
    y = np.arange(len(frame))
    is_placebo = frame["analysis"].str.contains("Placebo", case=False)
    ax.errorbar(
        frame["did_pct"],
        y,
        xerr=[frame["did_pct"] - frame["did_pct_ci_low"], frame["did_pct_ci_high"] - frame["did_pct"]],
        fmt="o",
        capsize=2.5,
        color=INK,
        ecolor=MUTED,
        markersize=4,
    )
    ax.scatter(frame["did_pct"], y, c=np.where(is_placebo, MUTED, ACCENT), s=34, zorder=4)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["analysis"], fontsize=7.5)
    ax.set_xlabel("Estimated change in crashes (%)")
    ax.set_title("Sensitivity of the main estimate to design choices")
    _save(
        fig,
        outdir,
        "10_sensitivity",
        "Grey markers are falsification tests expected to be near zero. Intervals are 95%. "
        "The headline result is not selected from this table.",
    )
