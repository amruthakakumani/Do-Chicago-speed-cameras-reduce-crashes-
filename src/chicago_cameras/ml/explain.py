"""SHAP explanations and publication figures for the LightGBM model.

SHAP values describe the model's predictive decomposition.  They are **not**
causal effects: a large SHAP value for a camera-related feature means the model
uses that feature to predict, not that the camera caused the crash count.
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

LOGGER = logging.getLogger(__name__)

INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#2c7fb8"
HIGHLIGHT = "#c0392b"
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


def gain_importance(booster: lgb.Booster) -> pd.DataFrame:
    names = booster.feature_name()
    gains = booster.feature_importance(importance_type="gain")
    splits = booster.feature_importance(importance_type="split")
    table = pd.DataFrame({"feature": names, "gain": gains, "split": splits})
    table["gain_share_pct"] = 100 * table["gain"] / table["gain"].sum()
    return table.sort_values("gain", ascending=False).reset_index(drop=True)


def compute_shap_values(
    booster: lgb.Booster,
    frame: pd.DataFrame,
    features: list[str],
    max_rows: int | None = 20_000,
    seed: int = 20261024,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return SHAP values (rows x features) and base values for tree ensemble."""
    sample = frame
    if max_rows is not None and len(frame) > max_rows:
        sample = frame.sample(max_rows, random_state=seed).sort_index()
    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(sample[features], check_additivity=False)
    if isinstance(values, list):  # pragma: no cover - binary/multiclass guard
        values = values[0]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:  # pragma: no cover
        values = values[..., 0]
    table = pd.DataFrame(values, columns=features, index=sample.index)
    table.insert(0, "base_value", np.asarray(explainer.expected_value).reshape(-1)[0])
    return table, sample


def plot_feature_importance(importance: pd.DataFrame, outdir: Path, top_n: int = 25) -> None:
    _style()
    frame = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 0.32 * len(frame) + 1.6))
    ax.barh(frame["feature"], frame["gain_share_pct"], color=ACCENT, alpha=0.85)
    ax.set_xlabel("Share of total split gain (%)")
    ax.set_title("LightGBM feature importance (gain)")
    _save(
        fig,
        outdir,
        "01_feature_importance",
        "Gain importance describes predictive use, not causal importance. "
        "A feature can be important because it proxies for camera siting.",
    )


def plot_shap_summary(
    values: np.ndarray, features_frame: pd.DataFrame, outdir: Path, max_display: int = 20
) -> None:
    _style()
    plt.figure(figsize=(9.2, 7.0))
    shap.summary_plot(
        values,
        features_frame,
        max_display=max_display,
        show=False,
        plot_type="dot",
        color_bar_label="Feature value (low → high)",
    )
    fig = plt.gcf()
    fig.suptitle("SHAP summary: drivers of predicted crash counts", y=1.01, fontsize=12, fontweight="semibold")
    _save(
        fig,
        outdir,
        "02_shap_summary",
        "Each point is a held-out location-month. Colour is the feature value; horizontal position is "
        "the SHAP contribution to the predicted log rate. Predictive association only — not causal.",
    )


def plot_shap_waterfall(
    values: np.ndarray,
    features_frame: pd.DataFrame,
    base_value: float,
    row_position: int,
    outdir: Path,
) -> None:
    _style()
    explanation = shap.Explanation(
        values=values[row_position],
        base_values=base_value,
        data=features_frame.iloc[row_position].to_numpy(),
        feature_names=list(features_frame.columns),
    )
    plt.figure(figsize=(9.0, 6.2))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle("Single prediction explanation (SHAP waterfall)", y=1.02, fontsize=12, fontweight="semibold")
    _save(
        fig,
        outdir,
        "03_shap_waterfall",
        "One held-out location-month. Values are contributions to the predicted log rate; "
        "they are not causal effects.",
    )


def plot_actual_vs_predicted(
    actual: np.ndarray, predicted: np.ndarray, outdir: Path
) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    hb = ax.hexbin(actual, predicted, gridsize=45, cmap="Blues", mincnt=1, linewidths=0)
    upper = float(max(np.max(actual), np.max(predicted))) * 1.05
    ax.plot([0, upper], [0, upper], color=HIGHLIGHT, lw=1.2, label="Perfect prediction")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Actual crashes per location-month (test)")
    ax.set_ylabel("Predicted crashes per location-month")
    ax.set_title("Out-of-sample accuracy")
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="Location-months")
    _save(
        fig,
        outdir,
        "04_actual_vs_predicted",
        "Untouched test period only. Overdispersion is expected for monthly crash counts; "
        "the calibration plot is the more informative summary.",
    )


def plot_monthly_error(monthly: pd.DataFrame, outdir: Path) -> None:
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.8), sharex=True)
    axes[0].plot(monthly["month"], monthly["mae"], color=ACCENT, lw=1.8, label="MAE")
    axes[0].plot(monthly["month"], monthly["rmse"], color=MUTED, lw=1.4, label="RMSE")
    axes[0].set_ylabel("Crashes per location-month")
    axes[0].set_title("Monthly out-of-sample error")
    axes[0].legend()
    axes[1].bar(monthly["month"], monthly["bias"], width=20, color=HIGHLIGHT, alpha=0.8)
    axes[1].axhline(0, color=INK, lw=0.8)
    axes[1].set_ylabel("Bias (pred − actual)")
    axes[1].set_xlabel("Month")
    _save(
        fig,
        outdir,
        "05_monthly_error",
        "Test period is fixed before training; no month in this plot influenced model selection.",
    )


def plot_residuals(actual: np.ndarray, predicted: np.ndarray, outdir: Path) -> None:
    _style()
    residual = np.asarray(predicted) - np.asarray(actual)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.hist(residual, bins=80, color=ACCENT, alpha=0.8)
    ax.axvline(0, color=INK, lw=1)
    ax.axvline(float(np.mean(residual)), color=HIGHLIGHT, lw=1.4, label=f"Mean {np.mean(residual):+.2f}")
    ax.set_xlabel("Residual (predicted − actual crashes)")
    ax.set_ylabel("Location-months")
    ax.set_title("Residual distribution, held-out test")
    ax.legend()
    _save(
        fig,
        outdir,
        "06_residual_distribution",
        f"Mean residual {np.mean(residual):+.3f}; MAE {np.mean(np.abs(residual)):.3f}; "
        f"n = {len(residual):,}.",
    )


def plot_calibration(calibration: pd.DataFrame, outdir: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.plot(calibration["predicted_mean"], calibration["actual_mean"], "o-", color=ACCENT, label="Deciles")
    upper = float(max(calibration["predicted_mean"].max(), calibration["actual_mean"].max())) * 1.1
    ax.plot([0, upper], [0, upper], color=HIGHLIGHT, ls="--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted crashes")
    ax.set_ylabel("Mean actual crashes")
    ax.set_title("Calibration by predicted-risk decile")
    ax.legend()
    _save(
        fig,
        outdir,
        "07_calibration",
        "Points below the diagonal mean over-prediction; points above mean under-prediction.",
    )


def plot_prediction_map(
    predictions: pd.DataFrame, outdir: Path, month: str | pd.Timestamp | None = None
) -> None:
    _style()
    frame = predictions.copy()
    if month is not None:
        frame = frame.loc[frame["month"] == pd.Timestamp(month)]
    elif not frame.empty:
        latest = frame["month"].max()
        frame = frame.loc[frame["month"] == latest]
    fig, ax = plt.subplots(figsize=(7.4, 8.8))
    scatter = ax.scatter(
        frame["lon"],
        frame["lat"],
        c=frame["predicted"],
        s=18,
        cmap="viridis",
        alpha=0.85,
        linewidths=0,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Predicted crashes, {pd.Timestamp(frame['month'].iloc[0]).strftime('%B %Y') if len(frame) else ''}")
    fig.colorbar(scatter, ax=ax, label="Predicted crashes per month")
    _save(
        fig,
        outdir,
        "08_prediction_map",
        "Predictions are shown for the latest held-out month. The location set is crash-dense "
        "and not a random sample of the city.",
    )
