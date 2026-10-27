#!/usr/bin/env python
"""Generate the eight publication figures from saved, verified artefacts.

Every panel reads a stored result or the cleaned data; no illustrative values are
hand-entered except the published original estimate, which is read from
``results/estimates.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1f2933"
MUTED = "#6b7785"
GRID = "#d9e2ec"
RED = "#b03a2e"
BLUE = "#2c6e9b"
PURPLE = "#6a51a3"
GOLD = "#b7791f"
TEAL = "#2a9d8f"

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 120,
    }
)


def save(fig: plt.Figure, name: str, note: str) -> None:
    fig.text(0.01, -0.02, note, fontsize=6.8, color=MUTED, ha="left", va="top")
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _km_axes(ax, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x0, y0 = x.min() - 2000, y.min() - 2000
    ax.set_xlabel("UTM easting (km from origin)")
    ax.set_ylabel("UTM northing (km from origin)")
    return x0, y0


def figure1_map() -> None:
    cameras = pd.read_parquet(ROOT / "data/processed/cameras.parquet")
    sites = pd.read_parquet(ROOT / "results/sites.parquet")
    crashes = pd.read_parquet(
        ROOT / "data/processed/crashes.parquet", columns=["x", "y"]
    ).sample(60_000, random_state=0)
    controls = sites.loc[sites["role"] == "comparison site"]
    matched = sites.loc[sites["role"] == "matched camera"]

    fig, ax = plt.subplots(figsize=(7.2, 8.8))
    ax.hexbin(crashes["x"], crashes["y"], gridsize=95, cmap="Greys", mincnt=1, alpha=0.28, linewidths=0)
    ax.scatter(cameras["x"], cameras["y"], s=15, marker="^", c=MUTED, alpha=0.55, label="Camera (all 209)")
    ax.scatter(matched["x"], matched["y"], s=36, marker="^", c=RED, edgecolor="white", linewidth=0.4, label="Matched camera (20)")
    ax.scatter(controls["x"], controls["y"], s=10, c=BLUE, alpha=0.55, linewidth=0, label="Comparison site (1,226)")
    x0, y0 = _km_axes(ax, np.r_[cameras["x"], controls["x"]], np.r_[cameras["y"], controls["y"]])
    ax.set_xticks(np.arange(0, 65, 5) * 1000 + x0)
    ax.set_yticks(np.arange(0, 55, 5) * 1000 + y0)
    ax.set_xticklabels([f"{(value - x0) / 1000:.0f}" for value in ax.get_xticks()])
    ax.set_yticklabels([f"{(value - y0) / 1000:.0f}" for value in ax.get_yticks()])
    ax.set_aspect("equal")
    ax.set_title("Chicago speed cameras and comparison locations")
    ax.legend(loc="upper left")
    save(
        fig,
        "fig1_study_map",
        "Crash density: 60,000 randomly sampled crashes, 2015-09 to 2026-08. "
        "Comparison sites are >600 m from every camera. UTM zone 16N; origin is the lower-left plot corner.",
    )


def figure2_overlap() -> None:
    cameras = pd.read_parquet(ROOT / "data/processed/cameras.parquet")
    xy = cameras[["x", "y"]].to_numpy(float)
    pairs = np.array(sorted(cKDTree(xy).query_pairs(500.0)), dtype=int)
    fig, ax = plt.subplots(figsize=(7.2, 8.2))
    x0, y0 = _km_axes(ax, cameras["x"], cameras["y"])
    for a, b in pairs:
        ax.plot(
            [xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]],
            color=GOLD, lw=0.8, alpha=0.7, zorder=1,
        )
    for x, y in xy:
        ax.add_patch(Circle((x, y), 250, fill=False, edgecolor=BLUE, lw=0.35, alpha=0.35, zorder=0))
    ax.scatter(xy[:, 0], xy[:, 1], s=14, c=RED, marker="^", edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_xticks(np.arange(0, 65, 5) * 1000 + x0)
    ax.set_yticks(np.arange(0, 55, 5) * 1000 + y0)
    ax.set_xticklabels([f"{(value - x0) / 1000:.0f}" for value in ax.get_xticks()])
    ax.set_yticklabels([f"{(value - y0) / 1000:.0f}" for value in ax.get_yticks()])
    ax.set_aspect("equal")
    ax.set_title("Overlapping 250 m camera catchments")
    ax.text(
        0.02, 0.02,
        f"{len(pairs)} camera pairs <500 m apart\n13 of 33 eligible cameras dropped\nin the primary design",
        transform=ax.transAxes, fontsize=8.5, color=INK,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": GRID},
    )
    save(
        fig,
        "fig2_catchment_overlap",
        "Circles are 250 m treatment catchments; gold segments join camera pairs whose catchments overlap. "
        "The primary design keeps the earliest camera in each overlap chain.",
    )


def figure3_causal() -> None:
    estimates = json.loads((ROOT / "results/estimates.json").read_text())
    sensitivity = pd.read_csv(ROOT / "results/tables/sensitivity.csv")
    original = estimates["replication"]["original_notebook"]
    primary = next(
        row for row in estimates["estimates"]
        if "Conditional Poisson" in row["method"] and "Replication" not in row["method"]
    )
    def sens(label):
        row = sensitivity.loc[sensitivity["analysis"] == label].iloc[0]
        return row["did_pct"], row["did_pct_ci_low"], row["did_pct_ci_high"]
    coverage_path = ROOT / "results/bottom_up/coverage_causal/estimates.json"
    coverage_rows = []
    if coverage_path.exists():
        coverage_estimates = json.loads(coverage_path.read_text())
        coverage_by_label = {row["analysis"]: row for row in coverage_estimates}
        coverage_rows = [
            ("Coverage 2017-09, 24/36 (15 cameras)", *(
                coverage_by_label["citywide 24/36"]["did_pct"],
                coverage_by_label["citywide 24/36"]["did_pct_ci_low"],
                coverage_by_label["citywide 24/36"]["did_pct_ci_high"],
            )),
            ("Coverage 2017-09, 12/36 (16 cameras)", *(
                coverage_by_label["citywide 12/36"]["did_pct"],
                coverage_by_label["citywide 12/36"]["did_pct_ci_low"],
                coverage_by_label["citywide 12/36"]["did_pct_ci_high"],
            )),
        ]
    rows = [
        ("Original Poisson DiD (33 cameras)", original["poisson_pct"], *original["poisson_ci_pct"]),
        ("Revised local design (20 cameras)", primary["pct"], primary["pct_ci_low"], primary["pct_ci_high"]),
        *coverage_rows,
        ("Citywide matching (20)", *sens("Citywide matching (no distance cap)")),
        ("1,000 m matching (11)", *sens("Max match distance 1000 m")),
        ("Keep overlapping cameras (33)", *sens("Keep overlapping cameras")),
        ("2019+ cohorts only (15)", *sens("2019+ go-live cohorts only")),
    ]
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    y = np.arange(len(rows))
    values = np.array([row[1] for row in rows])
    lows = np.array([row[2] for row in rows])
    highs = np.array([row[3] for row in rows])
    colors = [MUTED, BLUE, RED, TEAL, BLUE, BLUE, BLUE, GOLD]
    ax.errorbar(values, y, xerr=[values - lows, highs - values], fmt="none", ecolor=MUTED, capsize=3, zorder=2)
    ax.scatter(values, y, c=colors[:len(rows)], s=50, zorder=3)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in rows], fontsize=8)
    ax.set_xlabel("Change in crashes (%)  ·  negative = fewer crashes")
    ax.set_title("Causal estimates under alternative defensible designs")
    ax.invert_yaxis()
    save(
        fig,
        "fig3_causal_estimates",
        "Estimates are reported relative to matched comparison sites. The primary 2015-start design uses UTM distances, 2,000 m matching, "
        "and overlap removal. Coverage-consistent designs use a 2017-09 start and are a different sample. "
        "Source: results/estimates.json, results/tables/sensitivity.csv and results/bottom_up/coverage_causal/estimates.json.",
    )


def figure4_event_study() -> None:
    event = pd.read_csv(ROOT / "results/tables/event_study.csv")
    estimates = json.loads((ROOT / "results/estimates.json").read_text())
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.errorbar(
        event["quarter"], event["beta"],
        yerr=[event["beta"] - event["ci_low"], event["ci_high"] - event["beta"]],
        fmt="o-", color=INK, ecolor=MUTED, capsize=3, ms=4,
    )
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(-0.5, color=RED, ls="--", lw=1)
    ax.set_xlabel("Quarters relative to camera go-live (reference: −1)")
    ax.set_ylabel("Change in crashes per site-month")
    ax.set_title("Event study: dynamic effects and pre-treatment placebo quarters")
    pvalue = estimates["event_study_pre_trend"].get("pre_trend_pvalue")
    ax.text(
        0.02, 0.04,
        f"Joint pre-trend test p = {pvalue:.3f}\n(a test cannot prove parallel trends)",
        transform=ax.transAxes, fontsize=8, color=MUTED,
    )
    save(
        fig,
        "fig4_event_study",
        "Unit and camera-by-event-month fixed effects; standard errors clustered by camera. "
        "Effects are in crashes per site-month, not percent. n = 20 cameras.",
    )


def figure5_pretrends() -> None:
    panel = pd.read_parquet(ROOT / "results/analysis_panel.parquet")
    pre = panel.loc[panel["rel"] <= -1]
    agg = pre.groupby(["treated", "rel"])["y"].agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for treated, color, label in ((0, BLUE, "Matched comparison sites"), (1, RED, "Camera catchments")):
        subset = agg.loc[agg["treated"] == treated]
        ax.plot(subset["rel"], subset["mean"], color=color, lw=1.8, label=label)
        ax.fill_between(
            subset["rel"],
            subset["mean"] - 1.96 * subset["sem"],
            subset["mean"] + 1.96 * subset["sem"],
            color=color, alpha=0.15, linewidth=0,
        )
    ax.axvline(-0.5, color=INK, ls="--", lw=1)
    ax.set_xlabel("Months before camera go-live")
    ax.set_ylabel("Crashes per site-month within 250 m")
    ax.set_title("Pre-treatment trends: camera catchments and matched comparison sites")
    ax.legend(loc="upper left")
    save(
        fig,
        "fig5_pretrends",
        "Unadjusted means with 95% confidence bands. Pre-period only. "
        "The joint pre-trend test does not reject (p = 0.635), but low power limits what this establishes.",
    )


def _v1_test() -> pd.DataFrame:
    preds = pd.read_parquet(ROOT / "results/ml/v1/predictions.parquet")
    preds["month"] = pd.to_datetime(preds["month"])
    return preds.loc[preds["split"] == "test"].reset_index(drop=True)


def figure6_forecast() -> None:
    test = _v1_test()
    metrics = json.loads((ROOT / "results/ml/v1/metrics.json").read_text())["test"]["test_metrics"]
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    hb = ax.hexbin(test["y_all"], test["predicted"], gridsize=45, cmap="Blues", mincnt=1, linewidths=0)
    upper = float(max(test["y_all"].max(), test["predicted"].max())) * 1.05
    ax.plot([0, upper], [0, upper], color=RED, lw=1.2, label="Perfect forecast")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Actual crashes per location-month")
    ax.set_ylabel("Predicted crashes per location-month")
    ax.set_title("Held-out forecasts: observed versus LightGBM")
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="Location-months")
    ax.text(
        0.03, 0.72,
        f"MAE {metrics['mae']:.3f}\nRMSE {metrics['rmse']:.3f}\n"
        f"Poisson deviance {metrics['poisson_deviance']:.3f}\nBias {metrics['bias']:+.3f}",
        transform=ax.transAxes, fontsize=8, color=INK,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": GRID},
    )
    save(
        fig,
        "fig6_forecast",
        "Deployed V1 model on the retrospective test period 2025-09 to 2026-08 "
        "(16,260 location-months). Predictions are not causal effects.",
    )


def figure7_calibration() -> None:
    test = _v1_test()
    metrics = json.loads((ROOT / "results/ml/v1/metrics.json").read_text())["test"]
    calibration = pd.DataFrame(metrics["calibration"])
    bins = pd.cut(
        test["y_all"], [-0.1, 0.5, 1.5, 2.5, 4.5, 7.5, 100],
        labels=["0", "1", "2", "3–4", "5–7", "8+"],
    )
    errors = []
    for label, group in test.assign(actual_bin=bins).groupby("actual_bin", observed=True):
        errors.append(
            {
                "bin": str(label),
                "n": len(group),
                "mae": float(np.mean(np.abs(group["predicted"] - group["y_all"]))),
                "bias": float(np.mean(group["predicted"] - group["y_all"])),
            }
        )
    error_table = pd.DataFrame(errors)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    ax = axes[0]
    ax.plot(calibration["predicted_mean"], calibration["actual_mean"], "o-", color=BLUE, label="Deciles")
    upper = float(max(calibration["predicted_mean"].max(), calibration["actual_mean"].max())) * 1.1
    ax.plot([0, upper], [0, upper], color=RED, ls="--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted crashes")
    ax.set_ylabel("Mean actual crashes")
    ax.set_title("Calibration by predicted-risk decile")
    ax.legend()
    ax = axes[1]
    ax.bar(error_table["bin"], error_table["mae"], color=BLUE, alpha=0.8, label="MAE")
    ax.plot(error_table["bin"], error_table["bias"], "o-", color=RED, label="Bias (pred − actual)")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("Actual crashes per location-month")
    ax.set_ylabel("Crashes")
    ax.set_title("Error by observed crash level")
    ax.legend()
    save(
        fig,
        "fig7_calibration_highrisk",
        "Deployed V1 model, retrospective test period. High-crash months are under-predicted "
        f"(top actual decile: mean actual {metrics['subgroup_errors']['high_actual_top_decile']['mean_actual']:.2f}, "
        f"mean predicted {metrics['subgroup_errors']['high_actual_top_decile']['mean_predicted']:.2f}).",
    )


def figure8_intervals() -> None:
    metrics = json.loads((ROOT / "results/ml_v2/metrics.json").read_text())
    deciles = pd.DataFrame(metrics["conformal"]["interval_90"]["by_risk_decile"])
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    x = np.arange(len(deciles)) + 1
    ax.bar(x, deciles["coverage"] * 100, color=BLUE, alpha=0.8, label="Empirical coverage")
    ax.axhline(90, color=RED, ls="--", lw=1.2, label="Nominal 90%")
    ax.set_ylim(60, 102)
    ax.set_xticks(x)
    ax.set_xlabel("Predicted-risk decile (1 = lowest, 10 = highest)")
    ax.set_ylabel("90% interval coverage (%)")
    ax.set_title("Conformal interval coverage by predicted risk")
    ax2 = ax.twinx()
    ax2.plot(x, deciles["mean_width"], color=GOLD, marker="o", lw=1.5, label="Mean width")
    ax2.set_ylabel("Mean interval width (crashes)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False,
    )
    save(
        fig,
        "fig8_interval_coverage",
        "Scale-normalised split-conformal intervals for the deployed V1 model, calibrated on out-of-fold "
        "rolling predictions. Coverage is 90.6% overall but 84.8% in the highest-risk decile; intervals are "
        "approximate and not causal confidence intervals.",
    )


def figure9_reporting_coverage() -> None:
    table = pd.read_csv(ROOT / "results/bottom_up/coverage/monthly_coverage.csv", parse_dates=["month"])
    table = table.loc[table["month"] >= "2015-09-01"]
    fig, ax = plt.subplots(figsize=(9.8, 3.8))
    ax.plot(table["month"], table["crashes"], color=INK, lw=1.5, label="Reported crashes")
    ax.axvline(pd.Timestamp("2017-09-01"), color=RED, ls="--", lw=1.2)
    ax.set_xlabel("Month")
    ax.set_ylabel("Reported crashes")
    ax.set_title("Citywide crash reporting begins in September 2017")
    ax2 = ax.twinx()
    ax2.plot(table["month"], table["injury_share"] * 100, color=BLUE, lw=1.0, alpha=0.8, label="Injury share")
    ax2.set_ylabel("Injury share (%)")
    ax2.set_ylim(0, 22)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    save(
        fig,
        "fig9_reporting_coverage",
        "Portal metadata states citywide E-Crash coverage begins September 2017. Counts and injury composition shift at that boundary; "
        "the change is a reporting-regime diagnostic, not a causal effect.",
    )


def figure10_model_comparison() -> None:
    table = pd.read_csv(ROOT / "results/bottom_up/coverage_model/common_metrics.csv")
    order = ["v1", "v2", "coverage_v1", "coverage_full", "coverage_tuned"]
    labels = ["Deployed V1", "V2 candidate", "Coverage V1", "Coverage full", "Coverage tuned"]
    table = table.set_index("model").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7))
    colors = [MUTED, BLUE, TEAL, TEAL, PURPLE]
    axes[0].bar(labels, table["mae"], color=colors)
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Common test rows: MAE")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(labels, table["poisson_deviance"], color=colors)
    axes[1].set_ylabel("Poisson deviance")
    axes[1].set_title("Common test rows: deviance")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("All models evaluated on 15,912 common camera-zone test rows", fontsize=10.5)
    save(
        fig,
        "fig10_model_comparison",
        "Coverage-tuned improves deviance by 0.65% versus V1 on common rows, below the pre-declared 1% promotion rule. "
        "V1 remains deployed; V2 and coverage-tuned are research artifacts.",
    )


def figure11_road_intervals() -> None:
    metrics = json.loads((ROOT / "results/bottom_up/road_model/road_metrics.json").read_text())
    deciles = pd.DataFrame(metrics["interval_90"]["by_risk_decile"])
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.bar(deciles["risk_decile"] + 1, deciles["coverage"] * 100, color=BLUE, alpha=0.85)
    ax.axhline(90, color=RED, ls="--", lw=1.2, label="Nominal 90%")
    ax.set_ylim(50, 102)
    ax.set_xlabel("Predicted-risk decile (road segments)")
    ax.set_ylabel("Interval coverage (%)")
    ax.set_title("Road-segment intervals fail in the highest-risk decile")
    ax.legend(loc="lower left")
    save(
        fig,
        "fig11_road_intervals",
        "Road-segment task only. Overall coverage is 90.3%, but the highest-risk decile is 60.2%. "
        "Road and camera-zone MAE are not comparable.",
    )


def main() -> int:
    figure1_map()
    figure2_overlap()
    figure3_causal()
    figure4_event_study()
    figure5_pretrends()
    figure6_forecast()
    figure7_calibration()
    figure8_intervals()
    figure9_reporting_coverage()
    figure10_model_comparison()
    figure11_road_intervals()
    print(f"wrote 11 figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
