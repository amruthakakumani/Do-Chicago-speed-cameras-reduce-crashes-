#!/usr/bin/env python
"""Figures and residual diagnostics for the hybrid forecasting experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/statistical_hybrid"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#1f2933", "#6b7785", "#d9e2ec"
RED, BLUE, TEAL, GOLD, PURPLE = "#b03a2e", "#2c6e9b", "#2a9d8f", "#b7791f", "#6a51a3"

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 10.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "legend.frameon": False, "legend.fontsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def save(fig, name: str, note: str) -> None:
    fig.text(0.01, -0.02, note, fontsize=6.7, color=MUTED, ha="left", va="top")
    for suffix in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def load() -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(ROOT / "data/processed/ml_coverage_enriched.parquet")
    summary = json.loads((RESULTS / "summary.json").read_text())
    test = pd.read_parquet(RESULTS / "test_predictions.parquet")
    geo = pd.read_parquet(RESULTS / "geographic_predictions.parquet")
    frame["month"] = pd.to_datetime(frame["month"])
    test["month"] = pd.to_datetime(test["month"])
    geo["month"] = pd.to_datetime(geo["month"])
    return frame, summary, test, geo


def feature_correlation() -> None:
    corr = pd.read_csv(RESULTS / "feature_correlation.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=65, ha="right", fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Feature correlation among the diagnostic set")
    fig.colorbar(im, ax=ax, label="Pearson correlation", shrink=0.8)
    save(fig, "feature_correlation", "Linear correlations only; tree models can use nonlinear interactions that this matrix does not show.")


def target_distribution(frame: pd.DataFrame) -> None:
    y = frame["y_all"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    axes[0].hist(np.clip(y, 0, 20), bins=np.arange(0, 22) - 0.5, color=BLUE, alpha=0.85)
    axes[0].set_xlabel("Crashes per location-month (clipped at 20)")
    axes[0].set_ylabel("Rows")
    axes[0].set_title("Target distribution")
    zero = frame.groupby("site_type")["y_all"].apply(lambda s: float((s == 0).mean()))
    axes[1].bar(zero.index, zero.values * 100, color=[RED, BLUE])
    axes[1].set_ylabel("Zero-crash rows (%)")
    axes[1].set_title("Zero share by location type")
    fig.suptitle(f"Mean {y.mean():.2f} · variance {y.var():.2f} · zero share {100 * (y == 0).mean():.1f}%", fontsize=10.5)
    save(fig, "target_distribution", "Coverage-consistent camera-zone data, 2018-01 to 2026-08. Genuine zero crash months are retained.")


def monthly_seasonal(frame: pd.DataFrame) -> None:
    monthly = frame.groupby("month", as_index=False).agg(crashes=("y_all", "sum"))
    seasonal = frame.assign(month_of_year=frame["month"].dt.month).groupby("month_of_year").agg(mean=("y_all", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))
    axes[0].plot(monthly["month"], monthly["crashes"], color=INK, lw=1.4)
    axes[0].set_title("Monthly reported crashes")
    axes[0].set_ylabel("Crashes")
    axes[0].set_xlabel("Month")
    axes[1].plot(seasonal.index, seasonal["mean"], color=TEAL, marker="o")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_title("Seasonal mean")
    axes[1].set_xlabel("Month of year")
    axes[1].set_ylabel("Mean crashes / location")
    save(fig, "monthly_seasonal", "Reporting-era and seasonal patterns are descriptive; they are not causal effects.")


def mean_variance(frame: pd.DataFrame) -> None:
    stats = frame.groupby("location_id")["y_all"].agg(["mean", "var"])
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    ax.scatter(stats["mean"], stats["var"], s=8, alpha=0.35, color=BLUE)
    lim = max(stats["mean"].max(), stats["var"].max()) * 1.1
    ax.plot([0, lim], [0, lim], color=RED, ls="--", label="Poisson (var = mean)")
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_xlabel("Location mean crashes")
    ax.set_ylabel("Location variance")
    ax.set_title("Mean–variance relationship")
    ax.legend()
    save(fig, "mean_variance", f"Global variance/mean = {frame['y_all'].var() / frame['y_all'].mean():.2f}; points above the line are overdispersed.")


def geographic_risk(frame: pd.DataFrame) -> None:
    locations = frame.groupby("location_id").agg(x=("easting_m", "first"), y=("northing_m", "first"), mean=("y_all", "mean"), site=("site_type", "first"))
    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    sc = ax.scatter(locations["x"] / 1000, locations["y"] / 1000, c=locations["mean"], cmap="YlOrRd", s=10, alpha=0.7)
    ax.set_aspect("equal")
    ax.set_xlabel("UTM easting (km)")
    ax.set_ylabel("UTM northing (km)")
    ax.set_title("Geographic variation in mean crash counts")
    fig.colorbar(sc, ax=ax, label="Mean crashes/location-month")
    save(fig, "geographic_risk", "The location set is crash-dense and non-random; this is not a citywide road census.")


def deviance_residual(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(mu, float), 1e-9, None)
    term = np.where(y > 0, y * np.log(np.clip(y, 1e-12, None) / mu) - (y - mu), -mu)
    return np.sign(y - mu) * np.sqrt(np.clip(2 * term, 0, None))


def residual_diagnostics(test: pd.DataFrame) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    summary = {}
    for name, color, label in (("coverage_tuned", BLUE, "Optimized LGB"), ("nb", RED, "Negative Binomial"), ("hybrid", PURPLE, "Hybrid")):
        residual = deviance_residual(test["y_all"], test[name])
        axes[0].hist(residual, bins=50, density=True, histtype="step", lw=1.3, color=color, label=label)
        binned = pd.DataFrame({"pred": test[name], "resid": residual}).assign(bin=lambda d: pd.qcut(d["pred"], 10, duplicates="drop"))
        means = binned.groupby("bin", observed=True).agg(pred=("pred", "mean"), resid=("resid", "mean"))
        axes[1].plot(means["pred"], means["resid"], marker="o", color=color, label=label)
        summary[name] = {
            "mean": float(residual.mean()),
            "sd": float(residual.std()),
            "p01": float(np.quantile(residual, 0.01)),
            "p99": float(np.quantile(residual, 0.99)),
        }
    axes[0].set_title("Deviance residual distribution")
    axes[0].set_xlabel("Deviance residual")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[1].axhline(0, color=INK, lw=0.8)
    axes[1].set_title("Mean residual by predicted-risk decile")
    axes[1].set_xlabel("Mean prediction")
    axes[1].set_ylabel("Mean deviance residual")
    axes[1].legend()
    save(fig, "residual_diagnostics", "Deviance residuals use the observed counts and fitted means; positive residuals mean under-prediction.")
    # temporal autocorrelation of residual by location
    ac = {}
    work = test.sort_values(["location_id", "month"])
    for name in ("coverage_tuned", "nb", "hybrid"):
        work[name + "_resid"] = deviance_residual(work["y_all"], work[name])
        values = []
        for _, group in work.groupby("location_id"):
            if len(group) < 3:
                continue
            values.append(group[name + "_resid"].autocorr(lag=1))
        ac[name] = {"mean_lag1_autocorrelation": float(np.nanmean(values)) if values else float("nan")}
    summary["autocorrelation"] = ac
    # geographic cluster residual clustering
    cluster = test.copy()
    for name in ("coverage_tuned", "nb", "hybrid"):
        cluster[name + "_resid"] = deviance_residual(cluster["y_all"], cluster[name])
    summary["cluster_mean_residual"] = {
        name: cluster.groupby("cluster")[name + "_resid"].mean().to_dict() for name in ("coverage_tuned", "nb", "hybrid")
    }
    (RESULTS / "residual_diagnostics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def candidate_comparison(summary: dict) -> None:
    names = ["coverage_v1", "coverage_tuned", "nb", "hybrid"]
    labels = ["V1 ref.", "LGB", "NB", "Hybrid"]
    colors = [MUTED, BLUE, RED, PURPLE]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for ax, metric, title in ((axes[0], "poisson_deviance", "Validation deviance"), (axes[1], "mae", "Validation MAE")):
        values = [summary["temporal_validation"][name][metric] for name in names]
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=7)
    save(fig, "candidate_comparison", "Temporal rolling-origin OOF predictions (23,868 rows). The 2025-2026 test period was not used to select or weight candidates.")


def geographic_comparison(summary: dict) -> None:
    names = ["coverage_v1", "coverage_tuned", "nb", "hybrid"]
    labels = ["V1 ref.", "LGB", "NB", "Hybrid"]
    colors = [MUTED, BLUE, RED, PURPLE]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for ax, section, title in (
        (axes[0], "geographic", "Geographic block holdout"),
        (axes[1], "combined", "Geographic + later-period"),
    ):
        values = [summary[section][name]["poisson_deviance"] for name in names]
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    save(fig, "geographic_performance", "Held-out 2 km blocks with a 1 km buffer. The combined panel is retrospective on 2025-2026.")


def high_risk_calibration(summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for name, color, label in (("coverage_tuned", BLUE, "Optimized LGB"), ("nb", RED, "Negative Binomial"), ("hybrid", PURPLE, "Hybrid")):
        table = pd.DataFrame(summary["test"][name]["calibration"])
        ax.plot(table["predicted_mean"], table["actual_mean"], marker="o", lw=1.4, color=color, label=label)
    lim = max(table["predicted_mean"].max(), table["actual_mean"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=INK, ls="--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted crashes")
    ax.set_ylabel("Mean actual crashes")
    ax.set_title("Calibration by predicted-risk decile")
    ax.legend()
    save(fig, "high_risk_calibration", "The high-crash decile remains under-predicted. The hybrid is close to optimized LightGBM.")


def hybrid_weight() -> None:
    curve = pd.read_csv(RESULTS / "hybrid_weight_curve.csv")
    summary = json.loads((RESULTS / "summary.json").read_text())
    weight = summary["hybrid"]["weight_lightgbm"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(curve["weight_lightgbm"], curve["poisson_deviance"], color=PURPLE, lw=1.8, label="Poisson deviance")
    ax.axvline(weight, color=RED, ls="--", lw=1.2, label=f"Selected weight = {weight:.3f}")
    ax.set_xlabel("Weight on LightGBM")
    ax.set_ylabel("OOF Poisson deviance")
    ax.set_title("Hybrid weight sensitivity")
    ax2 = ax.twinx()
    ax2.plot(curve["weight_lightgbm"], curve["mae"], color=GOLD, lw=1.2, alpha=0.8, label="MAE")
    ax2.set_ylabel("OOF MAE")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax.text(0.04, 0.93, f"Error correlation: {summary['hybrid']['oof_error_correlation']:.3f}", transform=ax.transAxes, fontsize=8, color=MUTED)
    save(fig, "hybrid_weight", "Weights are learned only from out-of-fold training predictions; the test period is not used.")


def main() -> None:
    frame, summary, test, _ = load()
    feature_correlation()
    target_distribution(frame)
    monthly_seasonal(frame)
    mean_variance(frame)
    geographic_risk(frame)
    residual_diagnostics(test)
    candidate_comparison(summary)
    geographic_comparison(summary)
    high_risk_calibration(summary)
    hybrid_weight()
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
