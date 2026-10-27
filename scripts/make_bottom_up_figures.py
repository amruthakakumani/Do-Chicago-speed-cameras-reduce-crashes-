#!/usr/bin/env python
"""Generate bottom-up reconstruction figures from saved results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/bottom_up/figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1f2933"
MUTED = "#6b7785"
GRID = "#d9e2ec"
RED = "#b03a2e"
BLUE = "#2c6e9b"
TEAL = "#2a9d8f"
GOLD = "#b7791f"
PURPLE = "#6a51a3"

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.titlesize": 10.5,
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


def save(fig, name: str, note: str) -> None:
    fig.text(0.01, -0.02, note, fontsize=6.8, color=MUTED, ha="left", va="top")
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def coverage_monthly() -> None:
    table = pd.read_csv(ROOT / "results/bottom_up/coverage/monthly_coverage.csv", parse_dates=["month"])
    table = table.loc[table["month"] >= "2015-09-01"]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(table["month"], table["crashes"], color=INK, lw=1.4, label="Reported crashes")
    ax.axvspan(table["month"].min(), pd.Timestamp("2017-09-01"), color=GOLD, alpha=0.12, linewidth=0)
    ax.axvline(pd.Timestamp("2017-09-01"), color=RED, ls="--", lw=1.2)
    ax.set_xlabel("Month")
    ax.set_ylabel("Reported crashes per month")
    ax.set_title("Monthly reported crashes: the September 2017 citywide reporting break")
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(table["month"], table["injury_share"] * 100, color=BLUE, lw=1.1, alpha=0.8, label="Injury share")
    ax2.set_ylabel("Injury crash share (%)")
    ax2.set_ylim(0, 22)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax.text(
        pd.Timestamp("2017-10-01"), table["crashes"].max() * 0.93,
        "Citywide coverage starts\nSeptember 2017 (official metadata)",
        fontsize=8, color=RED,
    )
    save(
        fig,
        "coverage_monthly",
        "Source: Chicago Data Portal 85ca-t3if snapshot plus police-beat coverage extract. "
        "The shaded period is pre-citywide; count increases and injury-share changes are not interpreted as causal.",
    )


def coverage_transition() -> None:
    table = pd.read_csv(ROOT / "results/bottom_up/coverage/monthly_coverage.csv", parse_dates=["month"])
    table = table.loc[table["month"].between("2016-01-01", "2018-12-01")]
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.bar(table["month"], table["crashes"], width=25, color=MUTED, alpha=0.65, label="Reported crashes")
    ax.axvline(pd.Timestamp("2017-09-01"), color=RED, ls="--", lw=1.2)
    ax.set_ylabel("Reported crashes")
    ax.set_xlabel("Month")
    ax.set_title("Transition detail: August to September 2017")
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(table["month"], table["injury_share"] * 100, color=BLUE, marker="o", ms=3, lw=1.2)
    ax2.set_ylabel("Injury share (%)")
    ax2.set_ylim(0, 20)
    save(
        fig,
        "coverage_transition",
        "The level and injury composition change discontinuously at the official citywide start. "
        "All 22 districts have records before the break, so district presence alone is not a completeness measure.",
    )


def district_heatmap() -> None:
    table = pd.read_csv(ROOT / "results/bottom_up/coverage/district_month_coverage.csv", parse_dates=["month"])
    table = table.loc[table["month"].between("2016-09-01", "2018-12-01")]
    pivot = table.pivot_table(index="district", columns="month", values="crashes", aggfunc="sum").fillna(0)
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"District {int(value)}" for value in pivot.index])
    ticks = np.linspace(0, len(pivot.columns) - 1, 6).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([pivot.columns[i].strftime("%b %Y") for i in ticks], rotation=25, ha="right")
    ax.axvline(list(pivot.columns).index(pd.Timestamp("2017-09-01")), color=BLUE, lw=1.5)
    ax.set_title("Reported crashes by CPD district, September 2016–December 2018")
    fig.colorbar(image, ax=ax, label="Reported crashes per district-month")
    save(
        fig,
        "coverage_district_heatmap",
        "All 22 districts are present throughout, but the level shift is not uniform. "
        "This is a geographic-composition diagnostic, not proof of reporting completeness.",
    )


def exposure_availability() -> None:
    table = pd.read_csv(ROOT / "results/bottom_up/exposure/aadt_monthly_coverage.csv", parse_dates=["month"])
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.plot(table["month"], table["segments"], color=BLUE, lw=1.8, marker="o", ms=3, label="Segments observed")
    ax.axvline(pd.Timestamp("2026-05-01"), color=GOLD, ls="--", lw=1.2)
    ax.set_ylabel("AADT segments observed")
    ax.set_xlabel("Month")
    ax.set_title("AADT coverage: actual first month is July 2022, not October 2020")
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(table["month"], table["mean_adt"], color=RED, lw=1.1, alpha=0.7)
    ax2.set_ylabel("Mean ADT (vehicles/day)")
    ax.text(
        pd.Timestamp("2026-05-15"), table["segments"].max() * 0.96,
        "Portal says quality\nimproves from May 2026",
        fontsize=8, color=GOLD, va="top",
    )
    save(
        fig,
        "exposure_availability",
        "Source: Chicago AADT gc7y-n4xa aggregate. The current extract has 2,525 segments and 83,232 segment-months; "
        "no exposure exists for 2017–2022. AADT is therefore excluded from the historical model.",
    )


def road_assignment() -> None:
    summary = json.loads((ROOT / "results/bottom_up/roads/assignment_summary.json").read_text())
    assignments = pd.read_parquet(ROOT / "data/processed/crash_road_assignments.parquet", columns=["assignment_distance_m", "n_candidate_segments"])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    axes[0].hist(assignments["assignment_distance_m"].dropna(), bins=60, color=BLUE, alpha=0.85)
    axes[0].set_xlabel("Nearest-segment distance (m)")
    axes[0].set_ylabel("Assigned crashes")
    axes[0].set_title("Assignment distance")
    axes[1].hist(assignments["n_candidate_segments"].clip(upper=12), bins=np.arange(0, 14) - 0.5, color=PURPLE, alpha=0.85)
    axes[1].set_xlabel("Candidate segments within 30 m")
    axes[1].set_ylabel("Crashes")
    axes[1].set_title("Assignment ambiguity")
    fig.suptitle(
        f"Road assignment: {100 * summary['assignment_rate']:.2f}% assigned; "
        f"{100 * summary['ambiguous_assignments'] / summary['assigned']:.1f}% have a near-tie",
        fontsize=10.5,
    )
    save(
        fig,
        "road_assignment",
        "A crash is assigned once to the nearest segment; ambiguity means a second segment is within 5 m of the winner. "
        "Ambiguous assignments are retained but flagged, and this is a known limitation of the road task.",
    )


def feature_ablation() -> None:
    names = ["M0_minimal", "M1_camera", "M2_weather", "M3_road", "M4_neighborhood", "M5_citywide", "coverage_full"]
    labels = ["Minimal", "+ camera", "+ weather", "+ road", "+ neighbourhood", "+ citywide", "+ long history"]
    values = []
    for name in names:
        metrics = json.loads((ROOT / f"results/bottom_up/coverage_model/{name}_metrics.json").read_text())
        values.append(metrics["validation"]["mean_deviance"])
    fig, ax = plt.subplots(figsize=(9.0, 3.7))
    bars = ax.bar(labels, values, color=[MUTED] + [BLUE] * 5 + [PURPLE])
    v1 = json.loads((ROOT / "results/bottom_up/coverage_model/coverage_v1_metrics.json").read_text())["validation"]["mean_deviance"]
    ax.axhline(v1, color=RED, ls="--", lw=1.2, label="Coverage-aware V1")
    ax.set_ylabel("Mean validation Poisson deviance")
    ax.set_title("Feature-group ablation (lower is better)")
    ax.set_ylim(min(values) - 0.003, max(values) + 0.003)
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0003, f"{value:.4f}", ha="center", fontsize=7)
    ax.legend(loc="upper right")
    save(
        fig,
        "feature_ablation",
        "Cumulative groups; all models use the same fixed V1 hyperparameters. "
        "The full group lowers validation deviance by about 0.67% versus minimal, still below the 1% promotion rule.",
    )


def model_comparison() -> None:
    table = pd.read_csv(ROOT / "results/bottom_up/coverage_model/common_metrics.csv")
    order = ["v1", "v2", "coverage_v1", "coverage_full", "coverage_tuned"]
    labels = ["Deployed V1", "V2 candidate", "Coverage V1", "Coverage full", "Coverage tuned"]
    table = table.set_index("model").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    axes[0].bar(labels, table["mae"], color=[MUTED, BLUE, TEAL, TEAL, PURPLE])
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Common test rows: MAE")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(labels, table["poisson_deviance"], color=[MUTED, BLUE, TEAL, TEAL, PURPLE])
    axes[1].set_ylabel("Poisson deviance")
    axes[1].set_title("Common test rows: deviance")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("All models evaluated on the same 15,912 camera-zone test rows", fontsize=10.5)
    save(
        fig,
        "model_comparison",
        "Differences are small: coverage-tuned improves deviance by 0.65% versus deployed V1 on common rows, "
        "below the pre-declared 1% promotion rule. V1 remains deployed.",
    )


def causal_coverage() -> None:
    existing = json.loads((ROOT / "results/estimates.json").read_text())
    original = existing["replication"]["original_notebook"]
    original_pct, original_lo, original_hi = original["poisson_pct"], *original["poisson_ci_pct"]
    revised = next(row for row in existing["estimates"] if "Conditional Poisson" in row["method"] and "Replication" not in row["method"])
    new = json.loads((ROOT / "results/bottom_up/coverage_causal/estimates.json").read_text())
    by_label = {row["analysis"]: row for row in new}
    rows = [
        ("Original 2015-start (33 cams)", original_pct, original_lo, original_hi, MUTED),
        ("Revised 2015-start (20 cams)", revised["pct"], revised["pct_ci_low"], revised["pct_ci_high"], BLUE),
        ("Coverage 2017-09, 24/36 (15 cams)", by_label["citywide 24/36"]["did_pct"], by_label["citywide 24/36"]["did_pct_ci_low"], by_label["citywide 24/36"]["did_pct_ci_high"], RED),
        ("Coverage 2017-09, 12/36 (16 cams)", by_label["citywide 12/36"]["did_pct"], by_label["citywide 12/36"]["did_pct_ci_low"], by_label["citywide 12/36"]["did_pct_ci_high"], TEAL),
    ]
    fig, ax = plt.subplots(figsize=(9.8, 3.8))
    y = np.arange(len(rows))
    for index, (label, value, low, high, color) in enumerate(rows):
        ax.errorbar(value, index, xerr=[[value - low], [high - value]], fmt="o", color=color, capsize=4, ms=6)
        ax.text(high + 0.3, index, f"{value:+.1f}% [{low:+.1f}, {high:+.1f}]", va="center", fontsize=7.5, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlabel("Change in crashes (%)")
    ax.set_title("Causal sensitivity to the reporting regime")
    ax.invert_yaxis()
    ax.set_xlim(-22, 13)
    save(
        fig,
        "causal_coverage",
        "The coverage-consistent design is a new sample, not a direct replication: only 15 cameras have 24 citywide pre-months "
        "and 36 post-months. The estimate is imprecise and the geographic placebo p-value is 0.25.",
    )


def road_intervals() -> None:
    metrics = json.loads((ROOT / "results/bottom_up/road_model/road_metrics.json").read_text())
    deciles = pd.DataFrame(metrics["interval_90"]["by_risk_decile"])
    fig, ax = plt.subplots(figsize=(8.6, 3.7))
    ax.bar(deciles["risk_decile"] + 1, deciles["coverage"] * 100, color=BLUE, alpha=0.85)
    ax.axhline(90, color=RED, ls="--", lw=1.2, label="Nominal 90%")
    ax.set_ylim(50, 102)
    ax.set_xlabel("Predicted-risk decile (road segments)")
    ax.set_ylabel("Interval coverage (%)")
    ax.set_title("Road-segment intervals fail in the highest-risk decile")
    ax.legend(loc="lower left")
    save(
        fig,
        "road_interval_coverage",
        "Road-segment task, separate from camera zones. Overall coverage is 90.3%, but the top risk decile is 60.2%; "
        "raw MAE is not comparable across the two tasks.",
    )


def road_sample_map() -> None:
    sample = pd.read_csv(ROOT / "results/bottom_up/roads/model_sample_segments.csv")
    sample["segment_id"] = sample["segment_id"].astype(str)
    panel = pd.read_parquet(ROOT / "data/processed/road_model_panel_features.parquet", columns=["segment_id", "centroid_x", "centroid_y", "y_all"])
    points = panel.groupby("segment_id", as_index=False).agg(
        centroid_x=("centroid_x", "first"), centroid_y=("centroid_y", "first"), total_crashes=("y_all", "sum")
    )
    points["segment_id"] = points["segment_id"].astype(str)
    points = points.merge(sample[["segment_id"]], on="segment_id", how="inner")
    fig, ax = plt.subplots(figsize=(7.0, 8.0))
    scatter = ax.scatter(
        points["centroid_x"] / 1000, points["centroid_y"] / 1000,
        c=np.log1p(points["total_crashes"]), cmap="YlOrRd", s=8, alpha=0.55,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("UTM easting (km)")
    ax.set_ylabel("UTM northing (km)")
    ax.set_title("Road modelling sample: assigned crash burden")
    fig.colorbar(scatter, ax=ax, label="log(1 + crashes)")
    save(
        fig,
        "road_sample_map",
        "All 3,126 major segments plus 12,000 seeded local segments; 2,440 sampled segments have zero assigned crashes. "
        "This sample covers 23.6% of all assigned crashes and is not a citywide census.",
    )


def main() -> None:
    coverage_monthly()
    coverage_transition()
    district_heatmap()
    exposure_availability()
    road_assignment()
    feature_ablation()
    model_comparison()
    causal_coverage()
    road_intervals()
    road_sample_map()
    print(f"wrote bottom-up figures to {OUT}")


if __name__ == "__main__":
    main()
