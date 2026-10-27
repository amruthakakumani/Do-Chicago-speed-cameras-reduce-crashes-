"""Streamlit dashboard for the Chicago speed-camera study.

The dashboard is a *reader* of precomputed, validated artefacts.  It never
refits a model, and it never displays illustrative or fabricated numbers: if the
pipeline has not been run, it shows an explicit empty state.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
BOTTOM_UP = RESULTS / "bottom_up"

st.set_page_config(
    page_title="Chicago speed cameras & crashes",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
  :root { --ink:#1f2933; --muted:#616e7c; --rule:#d9e2ec; --accent:#6a51a3; }
  html, body, [class*="css"] { color: var(--ink); }
  h1, h2, h3 { letter-spacing: -0.015em; }
  h1 { font-size: 2.05rem; font-weight: 650; margin-bottom: .2rem; }
  h2 { font-size: 1.3rem; font-weight: 600; margin-top: 1.4rem; }
  h3 { font-size: 1.02rem; font-weight: 600; }
  .lede { color: var(--muted); font-size: 1.02rem; max-width: 60rem; }
  .card { border: 1px solid var(--rule); border-radius: 10px; padding: 1rem 1.15rem; background: #fff; }
  .kpi { font-size: 1.85rem; font-weight: 650; line-height: 1.1; }
  .kpi-label { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; }
  .note { color: var(--muted); font-size: .82rem; }
  .warn { border-left: 3px solid #b7791f; padding: .6rem .9rem; background: #fffbea; }
  hr { border: none; border-top: 1px solid var(--rule); }
  [data-testid="stSidebar"] { background: #f8fafc; }
  .stDataFrame { font-size: .86rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_estimates() -> dict | None:
    path = RESULTS / "estimates.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame | None:
    path = TABLES / name
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_cameras() -> pd.DataFrame | None:
    path = ROOT / "data" / "processed" / "cameras.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_bottom_json(relative: str) -> dict | None:
    path = BOTTOM_UP / relative
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_bottom_csv(relative: str) -> pd.DataFrame | None:
    path = BOTTOM_UP / relative
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_sites() -> pd.DataFrame | None:
    path = RESULTS / "sites.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


ML_DIR = RESULTS / "ml"
ML_V2 = RESULTS / "ml_v2"


@st.cache_data(show_spinner=False)
def load_deployment() -> dict | None:
    path = ML_V2 / "deployment.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


@st.cache_data(show_spinner=False)
def load_ml_metrics(path: str | None = None) -> dict | None:
    target = Path(path) if path else ML_DIR / "metrics.json"
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text())
    except json.JSONDecodeError:
        return None


@st.cache_data(show_spinner=False)
def load_ml_predictions(path: str | None = None) -> pd.DataFrame | None:
    target = Path(path) if path else ML_DIR / "predictions.parquet"
    if not target.exists():
        return None
    frame = pd.read_parquet(target)
    frame["month"] = pd.to_datetime(frame["month"])
    return frame


@st.cache_data(show_spinner=False)
def load_shap_values(path: str | None = None) -> pd.DataFrame | None:
    target = Path(path) if path else ML_DIR / "shap_values.parquet"
    if not target.exists():
        return None
    frame = pd.read_parquet(target)
    frame["month"] = pd.to_datetime(frame["month"])
    return frame


@st.cache_data(show_spinner=False)
def load_crash_grid(path: str | None = None) -> pd.DataFrame | None:
    target = Path(path) if path else ML_DIR / "crash_grid_monthly.parquet"
    if not target.exists():
        return None
    frame = pd.read_parquet(target)
    frame["month"] = pd.to_datetime(frame["month"])
    return frame


@st.cache_data(show_spinner=False)
def load_camera_metadata(path: str | None = None) -> pd.DataFrame | None:
    target = Path(path) if path else ML_DIR / "camera_metadata.parquet"
    if not target.exists():
        return None
    return pd.read_parquet(target)


@st.cache_data(show_spinner=False)
def load_intervals(path: str | None = None) -> pd.DataFrame | None:
    target = Path(path) if path else ML_V2 / "selected_predictions.parquet"
    if not target.exists():
        return None
    frame = pd.read_parquet(target)
    frame["month"] = pd.to_datetime(frame["month"])
    return frame


def _rgb_scale(values, low=(255, 237, 160), high=(189, 0, 38), alpha=210):
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return []
    span = np.nanmax(array) - np.nanmin(array)
    scaled = np.zeros_like(array) if span <= 0 else (array - np.nanmin(array)) / span
    return [
        [int(low[0] + (high[0] - low[0]) * value),
         int(low[1] + (high[1] - low[1]) * value),
         int(low[2] + (high[2] - low[2]) * value),
         alpha]
        for value in scaled
    ]


def _diverging_colors(values, alpha=200):
    array = np.asarray(values, dtype=float)
    scale = np.nanpercentile(np.abs(array), 90) if len(array) else 1.0
    scale = scale if np.isfinite(scale) and scale > 0 else 1.0

    def one(value):
        clipped = max(-1.0, min(1.0, float(value) / scale))
        if clipped >= 0:
            return [192, 57, 43, alpha] if clipped else [220, 220, 220, alpha]
        return [44, 127, 184, alpha]

    return [one(value) for value in array]


def empty_state() -> None:
    st.markdown("# Results not generated")
    st.markdown(
        '<p class="lede">This dashboard reads only validated artefacts. '
        "No results were found in <code>results/</code>, so there is nothing to show.</p>",
        unsafe_allow_html=True,
    )
    st.code(
        "python scripts/download_data.py\n"
        "python -m chicago_cameras.pipeline --config config/analysis.yaml",
        language="bash",
    )
    st.info(
        "The pipeline downloads the three public Chicago Data Portal datasets, builds the "
        "matched panel, runs the estimators and placebos, and writes tables, figures and "
        "`results/estimates.json`."
    )
    st.stop()


def fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.1f}%"


def pct_interval(low: float | None, high: float | None) -> str:
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return "—"
    return f"[{low:+.1f}%, {high:+.1f}%]"


def ml_figure(name: str, caption: str = "", directory: Path | None = None) -> None:
    base = directory or ML_DIR / "figures"
    png = base / f"{name}.png"
    svg = base / f"{name}.svg"
    if png.exists():
        st.image(str(png), use_container_width=True)
        if caption:
            st.markdown(f'<p class="note">{caption}</p>', unsafe_allow_html=True)
    elif svg.exists():
        st.image(str(svg), use_container_width=True)
    else:
        st.info(f"Figure `{name}` has not been generated.")


def figure(name: str, caption: str = "") -> None:
    png = FIGURES / f"{name}.png"
    svg = FIGURES / f"{name}.svg"
    if png.exists():
        st.image(str(png), use_container_width=True)
        if caption:
            st.markdown(f'<p class="note">{caption}</p>', unsafe_allow_html=True)
    elif svg.exists():
        st.image(str(svg), use_container_width=True)
    else:
        st.info(f"Figure `{name}` has not been generated.")


estimates = load_estimates()
if estimates is None:
    empty_state()

main = pd.DataFrame(estimates.get("estimates", []))
sensitivity = pd.DataFrame(estimates.get("sensitivity", []))
subgroups = pd.DataFrame(estimates.get("subgroups", []))
severity = pd.DataFrame(estimates.get("severity", []))
sample = estimates.get("sample", {})
sc = estimates.get("synthetic_control", {})
lattice = estimates.get("lattice_phase", {})

PRIMARY_METHOD = "Conditional Poisson FE (stack x event-month fixed effects), cluster by camera"
primary_row = main.loc[main["method"].str.contains("Conditional Poisson", na=False) & ~main["method"].str.contains("Replication", na=False)]
primary = primary_row.iloc[0] if not primary_row.empty else None

st.sidebar.markdown("### Chicago speed cameras")
st.sidebar.markdown(
    '<p class="note">Causal inference study · Chicago Data Portal</p>', unsafe_allow_html=True
)
page = st.sidebar.radio(
    "View",
    [
        "Research overview",
        "Chicago map",
        "Causal inference",
        "Crash forecasting",
        "Robustness & sensitivity",
        "Coverage & reconstruction",
        "Methods & limitations",
    ],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<p class="note">Snapshot: {sample.get("first_month", "?")} – {sample.get("last_month", "?")}<br>'
    f'{sample.get("cameras_matched", 0)} matched cameras · '
    f'{sample.get("control_candidates", 0):,} comparison sites</p>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<p class="note">Dashboard reads precomputed results; it does not refit models.</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
if page == "Research overview":
    st.markdown("# Do Chicago speed cameras reduce crashes?")
    st.markdown(
        '<p class="lede">A difference-in-differences study comparing crash counts within 250 m of '
        "speed cameras with matched Chicago locations that never had a camera. Estimates are "
        "reported with confidence intervals, not just p-values.</p>",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    if primary is not None:
        with cols[0]:
            st.markdown(
                f'<div class="card"><div class="kpi-label">Primary estimate</div>'
                f'<div class="kpi">{fmt_pct(primary.get("pct"))}</div>'
                f'<div class="note">crashes near cameras vs comparison sites</div></div>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown(
                f'<div class="card"><div class="kpi-label">95% interval</div>'
                f'<div class="kpi" style="font-size:1.35rem">'
                f'{pct_interval(primary.get("pct_ci_low"), primary.get("pct_ci_high"))}</div>'
                f'<div class="note">conditional Poisson FE, cluster-robust</div></div>',
                unsafe_allow_html=True,
            )
    with cols[2]:
        st.markdown(
            f'<div class="card"><div class="kpi-label">Cameras analysed</div>'
            f'<div class="kpi">{sample.get("cameras_matched", "—")}</div>'
            f'<div class="note">of {sample.get("cameras_in_window", "—")} in the event window; '
            f'{sample.get("cameras_dropped_overlap", 0)} dropped for catchment overlap</div></div>',
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(
            f'<div class="card"><div class="kpi-label">Crashes analysed</div>'
            f'<div class="kpi">{sample.get("crashes", 0):,}</div>'
            f'<div class="note">{sample.get("months", "—")} months, all reported crashes</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("## What the evidence supports")
    if primary is not None:
        ci_low = primary.get("pct_ci_low")
        ci_high = primary.get("pct_ci_high")
        crosses = ci_low is not None and ci_high is not None and ci_low < 0 < ci_high
        if crosses:
            st.markdown(
                f'<div class="warn"><strong>The primary local design does not detect a precise effect.</strong> '
                f"The point estimate is {fmt_pct(primary.get('pct'))} with a 95% interval of "
                f"{pct_interval(ci_low, ci_high)}, which includes both meaningful reductions and "
                "increases. This is an imprecise null, not evidence that cameras do nothing.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"The primary estimate is **{fmt_pct(primary.get('pct'))}** "
                f"(95% CI {pct_interval(ci_low, ci_high)}), which excludes zero."
            )

    st.markdown(
        """
- The original study's **−7.3%** estimate is reproducible from the saved notebook, but it relies on
  citywide matching that the README does not describe, keeps overlapping camera catchments, and is
  sensitive to arbitrary comparison-lattice placement.
- With **local comparison sites (≤2 km)** and **overlapping cameras removed**, the estimate moves
  toward zero and becomes statistically indistinguishable from no effect.
- The synthetic control has only a moderate pre-period fit and points in the opposite direction; it
  does **not** corroborate a reduction.
- Geographic placebo tests put the real estimate near the middle of the placebo distribution.

**Read the interval, not the sign.** With roughly twenty cameras, the design can rule out neither a
modest benefit nor a modest harm.
"""
    )

    st.markdown("## Design in one paragraph")
    st.markdown(
        "Each camera is its own experiment. For every camera we take the 250 m catchment and five "
        "nearby comparison sites with a similar pre-period level and trend. Event time runs from 24 "
        "months before to 35 months after go-live. The preferred estimator is a conditional Poisson "
        "model with site fixed effects and camera-by-month fixed effects, so any shock shared by a "
        "camera and its comparison sites — including citywide traffic and COVID-19 — is absorbed. "
        "Standard errors are clustered by camera and cross-checked with a wild cluster bootstrap."
    )

    st.markdown("## Read next")
    st.markdown(
        "- **Spatial explorer** — where the cameras and comparison sites are, and how overlap was handled.\n"
        "- **Causal estimates** — every estimator, the event study and the synthetic control.\n"
        "- **Robustness** — 25+ specifications, placebo dates, placebo locations and severity outcomes.\n"
        "- **Methodology** — assumptions, deviations from the original study, provenance."
    )


# ---------------------------------------------------------------------------
# Predictions & interactive map (LightGBM)
# ---------------------------------------------------------------------------
elif page == "Crash forecasting":
    st.markdown("# Crash forecasting")
    st.markdown(
        '<p class="lede">A single LightGBM Poisson model predicts monthly crashes at camera '
        "catchments and matched comparison sites. The map overlays observations, predictions, "
        "approximate 90% prediction intervals and the causal design. Predictions are predictive "
        "associations, not causal effects.</p>",
        unsafe_allow_html=True,
    )

    deployment = load_deployment()
    if deployment:
        selected_version = deployment.get("selected_model", "v1")
        metrics_path = str(ML_V2 / "metrics.json")
        predictions_path = str(ROOT / deployment["predictions_path"])
        intervals_path = str(ROOT / deployment["test_predictions_path"])
        shap_path = (
            str(ML_V2 / "shap_values.parquet")
            if selected_version == "v2"
            else str(ML_DIR / "shap_values.parquet")
        )
        figure_dir = ML_V2 / "figures" if selected_version == "v2" else ML_DIR / "figures"
    else:  # V1 fallback
        selected_version = "v1"
        metrics_path = str(ML_DIR / "metrics.json")
        predictions_path = str(ML_DIR / "predictions.parquet")
        intervals_path = None
        shap_path = str(ML_DIR / "shap_values.parquet")
        figure_dir = ML_DIR / "figures"

    ml_metrics = load_ml_metrics(metrics_path)
    predictions = load_ml_predictions(predictions_path)
    intervals = load_intervals(intervals_path) if intervals_path else None
    grid = load_crash_grid()
    shap_values = load_shap_values(shap_path)
    camera_meta = load_camera_metadata()

    if ml_metrics is None or predictions is None:
        st.markdown(
            '<div class="warn"><strong>Model results not generated.</strong> Run '
            "<code>python scripts/train_lightgbm_v2.py</code> (or the V1 script) to write "
            "<code>results/ml*</code>. The dashboard does not train models on reload.</div>",
            unsafe_allow_html=True,
        )
    else:
        selected_metrics = ml_metrics.get("test_metrics_selected") or ml_metrics["test"]["test_metrics"]
        test = predictions.loc[predictions["split"] == "test"].copy()
        if intervals is not None:
            interval_columns = [
                "location_id", "month", "lower_90", "upper_90", "lower_50", "upper_50",
            ]
            available = [column for column in interval_columns if column in intervals.columns]
            test = test.merge(intervals[available], on=["location_id", "month"], how="left")

        coverage = ml_metrics.get("conformal", {}).get("interval_90", {}).get("empirical_coverage")
        cols = st.columns(5)
        cols[0].metric("Test MAE", f"{selected_metrics['mae']:.2f}", help="Held-out test rows")
        cols[1].metric("Test RMSE", f"{selected_metrics['rmse']:.2f}")
        cols[2].metric("Poisson deviance", f"{selected_metrics['poisson_deviance']:.2f}")
        cols[3].metric(
            "Bias", f"{selected_metrics['bias']:+.2f}",
            delta=f"{selected_metrics['relative_bias_pct']:+.1f}%", delta_color="inverse",
        )
        cols[4].metric(
            "90% interval coverage",
            f"{coverage:.1%}" if coverage is not None and np.isfinite(coverage) else "—",
            help="Empirical coverage of approximate split-conformal intervals on test rows",
        )

        test_period = (test["month"].min(), test["month"].max())
        st.markdown(
            f'<p class="note">Deployed model: <strong>{selected_version.upper()}</strong> · '
            f'trained on {ml_metrics["training"]["rows"]:,} location-months through '
            f'{ml_metrics["training"]["period"]["last"]} · test {test_period[0].date()} to '
            f'{test_period[1].date()} ({len(test):,} rows) · '
            f'{ml_metrics["tuning"]["n_trials"]} Optuna trials · '
            f'{len(ml_metrics["dataset"]["features"])} features · '
            f'data freshness {ml_metrics.get("generated_at_utc", deployment.get("generated_at_utc", "—"))}</p>',
            unsafe_allow_html=True,
        )
        if selected_version == "v1" and deployment:
            validation = deployment.get("validation", {})
            st.markdown(
                f'<p class="note">V2 was evaluated but not deployed: validation deviance changed '
                f'{100 * validation.get("relative_deviance_improvement", 0):+.2f}% '
                f'(pre-declared threshold: +1%). V1 remains deployed; V2 results are retained for '
                "research and are shown in the comparison section below.</p>",
                unsafe_allow_html=True,
            )

        years = sorted(test["month"].dt.year.unique().tolist())
        filter_cols = st.columns([1, 1, 1.4, 1.4, 1.2])
        with filter_cols[0]:
            year = st.selectbox("Year", years, index=len(years) - 1)
        months_in_year = sorted(test.loc[test["month"].dt.year == year, "month"].unique())
        with filter_cols[1]:
            month = st.selectbox(
                "Month", months_in_year, index=len(months_in_year) - 1,
                format_func=lambda value: pd.Timestamp(value).strftime("%b %Y"),
            )
        with filter_cols[2]:
            site_types = st.multiselect("Location type", ["camera", "control"], default=["camera", "control"])
        available_clusters = sorted(test["cluster"].dropna().unique().tolist())
        with filter_cols[3]:
            clusters = st.multiselect("Region cluster", available_clusters, default=available_clusters)
        with filter_cols[4]:
            severity = st.selectbox("Observed severity", ["all", "injury", "severe"])

        layer_options = [
            "Cameras",
            "Treatment buffers (250 m)",
            "Comparison sites",
            "Observed crash density",
            "Predicted crashes",
            "Predicted risk intensity",
            "90% interval width",
            "Prediction errors (test)",
            "Historical crash density",
        ]
        selected_layers = st.multiselect(
            "Map layers", layer_options,
            default=["Cameras", "Comparison sites", "Observed crash density", "Predicted crashes"],
        )

        camera_options = (
            test.loc[test["site_type"] == "camera", ["location_id", "camera_uid"]]
            .drop_duplicates()
            .sort_values("camera_uid")
        )
        label_map = {
            row.location_id: f"{row.camera_uid} ({row.location_id})"
            for row in camera_options.itertuples()
        }
        camera_filter = st.selectbox(
            "Camera filter (map)", ["All cameras"] + camera_options["location_id"].tolist(),
            format_func=lambda value: value if value == "All cameras" else label_map.get(value, value),
        )

        month_predictions = test.loc[
            (test["month"] == pd.Timestamp(month))
            & (test["site_type"].isin(site_types))
            & (test["cluster"].isin(clusters))
        ].copy()
        if camera_filter != "All cameras":
            month_predictions = month_predictions.loc[
                month_predictions["location_id"] == camera_filter
            ]

        cameras = (
            test.loc[test["site_type"] == "camera", ["location_id", "camera_uid", "lat", "lon"]]
            .drop_duplicates("location_id")
        )
        if camera_filter != "All cameras":
            cameras = cameras.loc[cameras["location_id"] == camera_filter]
        controls = (
            test.loc[test["site_type"] == "control", ["location_id", "lat", "lon"]]
            .drop_duplicates("location_id")
        )

        severity_column = {"all": "crashes_all", "injury": "crashes_injury", "severe": "crashes_severe"}[severity]
        if grid is not None and len(grid):
            grid_month = grid.loc[grid["month"] == pd.Timestamp(month)].copy()
            grid_month["crashes"] = grid_month[severity_column]
            grid_month = grid_month.loc[grid_month["crashes"] > 0]
            historical = (
                grid.loc[grid["month"] <= pd.Timestamp(month)]
                .groupby(["cell_id", "lat", "lon"], as_index=False)[["crashes_all", "crashes_injury", "crashes_severe"]]
                .sum()
            )
            historical["crashes"] = historical[severity_column]
        else:
            grid_month = pd.DataFrame()
            historical = pd.DataFrame()

        layers = []
        if "Cameras" in selected_layers:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer", data=cameras, get_position="[lon, lat]",
                    get_radius=70, get_fill_color=[192, 57, 43, 230], pickable=True,
                )
            )
        if "Treatment buffers (250 m)" in selected_layers:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer", data=cameras, get_position="[lon, lat]",
                    get_radius=250, radius_units="meters", filled=False, stroked=True,
                    get_line_color=[192, 57, 43, 180], line_width_min_pixels=1, pickable=False,
                )
            )
        if "Comparison sites" in selected_layers:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer", data=controls, get_position="[lon, lat]",
                    get_radius=45, get_fill_color=[44, 127, 184, 150], pickable=True,
                )
            )
        if "Historical crash density" in selected_layers and len(historical):
            layers.append(
                pdk.Layer(
                    "ColumnLayer", data=historical, get_position="[lon, lat]",
                    get_elevation="crashes", elevation_scale=0.8, radius=230,
                    get_fill_color=[120, 120, 120, 70], extruded=False, pickable=True,
                )
            )
        if "Observed crash density" in selected_layers and len(grid_month):
            layers.append(
                pdk.Layer(
                    "ColumnLayer", data=grid_month, get_position="[lon, lat]",
                    get_elevation="crashes", elevation_scale=8, radius=230,
                    get_fill_color=_rgb_scale(grid_month["crashes"], low=(255, 237, 160), high=(189, 0, 38), alpha=150),
                    extruded=True, pickable=True,
                )
            )
        if "Predicted crashes" in selected_layers and len(month_predictions):
            layer_frame = month_predictions.copy()
            layer_frame["color"] = _rgb_scale(layer_frame["predicted"], low=(222, 235, 247), high=(106, 81, 163))
            layer_frame["radius"] = 50 + 18 * np.sqrt(layer_frame["predicted"].clip(lower=0))
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer", data=layer_frame, get_position="[lon, lat]",
                    get_radius="radius", get_fill_color="color", opacity=0.85, pickable=True,
                )
            )
        if "Predicted risk intensity" in selected_layers and len(month_predictions):
            layer_frame = month_predictions.copy()
            layer_frame["risk"] = (
                layer_frame["predicted"] / (layer_frame["expanding_mean"] + 0.5)
            ).clip(lower=0, upper=3)
            layer_frame["color"] = _rgb_scale(layer_frame["risk"], low=(237, 248, 251), high=(189, 0, 38))
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer", data=layer_frame, get_position="[lon, lat]",
                    get_radius=85, get_fill_color="color", opacity=0.9, pickable=True,
                )
            )
        if "90% interval width" in selected_layers and "upper_90" in month_predictions.columns:
            layer_frame = month_predictions.dropna(subset=["upper_90", "lower_90"]).copy()
            if len(layer_frame):
                layer_frame["width90"] = layer_frame["upper_90"] - layer_frame["lower_90"]
                layer_frame["color"] = _rgb_scale(layer_frame["width90"], low=(254, 240, 217), high=(140, 81, 10))
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer", data=layer_frame, get_position="[lon, lat]",
                        get_radius=95, get_fill_color="color", opacity=0.85, pickable=True,
                    )
                )
        if "Prediction errors (test)" in selected_layers and len(month_predictions):
            layer_frame = month_predictions.copy()
            layer_frame["error"] = layer_frame["predicted"] - layer_frame["y_all"]
            layer_frame["color"] = _diverging_colors(layer_frame["error"])
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer", data=layer_frame, get_position="[lon, lat]",
                    get_radius=75, get_fill_color="color", opacity=0.9, pickable=True,
                )
            )

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=41.85, longitude=-87.65, zoom=10.3, pitch=0),
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            tooltip={
                "html": "<b>{location_id}</b><br/>Predicted: {predicted}<br/>Actual: {y_all}",
                "style": {"backgroundColor": "#1f2933", "color": "white", "fontSize": "12px"},
            },
        )
        st.pydeck_chart(deck, use_container_width=True)
        st.markdown(
            f'<p class="note">Showing {pd.Timestamp(month).strftime("%B %Y")} · '
            f'{len(month_predictions):,} locations · {severity} crashes. Observed density is a '
            "500 m grid; columns are counts, not a continuous surface. Colour key: predicted "
            "crashes light → dark purple; risk intensity / interval width light → dark; error "
            "points are <span style='color:#2c7fb8'>blue for under-prediction</span> and "
            "<span style='color:#c0392b'>red for over-prediction</span>. Prediction intervals are "
            "approximate and are <strong>not</strong> causal confidence intervals.</p>",
            unsafe_allow_html=True,
        )

        st.markdown("## Camera drill-down")
        selected_camera = st.selectbox(
            "Select a camera", camera_options["location_id"].tolist(),
            format_func=lambda value: label_map.get(value, value), index=0,
        )
        camera_history = predictions.loc[predictions["location_id"] == selected_camera].sort_values("month").copy()
        if intervals is not None:
            interval_columns = [
                "month", "lower_90", "upper_90", "lower_50", "upper_50",
            ]
            available = [column for column in interval_columns if column in intervals.columns]
            camera_intervals = intervals.loc[
                intervals["location_id"] == selected_camera, available
            ]
            camera_history = camera_history.merge(camera_intervals, on="month", how="left")

        if len(camera_history):
            meta = {}
            if camera_meta is not None:
                match = camera_meta.loc[camera_meta["camera_uid"] == camera_history["camera_uid"].iloc[0]]
                if len(match):
                    meta = match.iloc[0].to_dict()
            latest_interval = camera_history.dropna(subset=["lower_90"]).tail(1) if "lower_90" in camera_history else pd.DataFrame()
            col_a, col_b = st.columns([1.0, 2.2])
            with col_a:
                st.markdown("### Camera details")
                st.write(
                    {
                        "Camera": camera_history["camera_uid"].iloc[0],
                        "Location id": selected_camera,
                        "Activation": meta.get("go_live", "see data dictionary"),
                        "Address": meta.get("address", "—"),
                        "Treatment radius": "250 m",
                    }
                )
                if len(latest_interval):
                    row = latest_interval.iloc[0]
                    st.metric(
                        f"Predicted {pd.Timestamp(row['month']).strftime('%b %Y')}",
                        f"{row['predicted']:.1f} crashes",
                        help="Held-out test month",
                    )
                    st.metric(
                        "90% prediction interval",
                        f"{max(0, row['lower_90']):.1f} – {row['upper_90']:.1f}",
                    )
                test_history = camera_history.loc[camera_history["split"] == "test"]
                if len(test_history):
                    errors = test_history["predicted"] - test_history["y_all"]
                    st.metric("Test MAE (this camera)", f"{errors.abs().mean():.2f}")
                    st.metric("Test bias (this camera)", f"{errors.mean():+.2f}")
                st.markdown(
                    '<p class="note">Camera-level errors are descriptive. The causal estimate below '
                    "is a study-wide average and is <strong>not</strong> a camera-specific effect. "
                    "Intervals are approximate split-conformal intervals, not causal confidence "
                    "intervals.</p>",
                    unsafe_allow_html=True,
                )
            with col_b:
                st.markdown("### Forecast: observed, predicted and 90% interval")
                chart_frame = camera_history.tail(36).copy()
                base = alt.Chart(chart_frame).encode(
                    x=alt.X("month:T", title="Month")
                )
                band = base.mark_area(opacity=0.18, color="#6a51a3").encode(
                    y=alt.Y("lower_90:Q", title="Crashes per month"),
                    y2="upper_90:Q",
                )
                lines = base.transform_fold(
                    ["y_all", "predicted"], as_=["series", "value"]
                ).mark_line().encode(
                    y=alt.Y("value:Q", title="Crashes per month"),
                    color=alt.Color(
                        "series:N",
                        scale=alt.Scale(
                            domain=["y_all", "predicted"],
                            range=["#c0392b", "#2c7fb8"],
                        ),
                        title=None,
                    ),
                )
                st.altair_chart((band + lines).properties(height=280), use_container_width=True)
                recent = camera_history.tail(12)[
                    ["month", "y_all", "predicted", "lower_90", "upper_90", "split"]
                ].copy()
                recent["error"] = recent["predicted"] - recent["y_all"]
                st.dataframe(recent, hide_index=True, use_container_width=True)

        st.markdown("### Why this prediction? (SHAP)")
        if shap_values is None:
            st.info("SHAP values not found for the selected model.")
        else:
            camera_shap = shap_values.loc[shap_values["location_id"] == selected_camera].copy()
            if camera_shap.empty:
                st.info("No held-out SHAP rows for this camera.")
            else:
                shap_months = sorted(camera_shap["month"].unique())
                shap_month = st.selectbox(
                    "Explanation month", shap_months, index=len(shap_months) - 1,
                    format_func=lambda value: pd.Timestamp(value).strftime("%b %Y"),
                    key="shap_month",
                )
                row = camera_shap.loc[camera_shap["month"] == pd.Timestamp(shap_month)].iloc[0]
                feature_columns = [
                    column for column in shap_values.columns
                    if column not in {
                        "location_id", "month", "site_type", "camera_uid", "y_all", "predicted", "base_value",
                    }
                ]
                contributions = (
                    row[feature_columns]
                    .astype(float)
                    .sort_values(key=np.abs, ascending=False)
                    .head(12)
                )
                st.bar_chart(contributions.sort_values())
                st.markdown(
                    f'<p class="note">Base value {row["base_value"]:.3f}; predicted '
                    f'{row["predicted"]:.2f}; observed {row["y_all"]:.0f}. SHAP values are '
                    "contributions to the predicted log rate and are not causal effects.</p>",
                    unsafe_allow_html=True,
                )

        st.markdown("## V1 vs V2 candidate (retrospective)")
        comparison = ml_metrics.get("test_retrospective", {}).get("comparison", {})
        if comparison:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Model": "V1", "MAE": comparison["v1"]["mae"], "RMSE": comparison["v1"]["rmse"],
                         "Poisson deviance": comparison["v1"]["poisson_deviance"], "Bias": comparison["v1"]["bias"]},
                        {"Model": "V2", "MAE": comparison["v2"]["mae"], "RMSE": comparison["v2"]["rmse"],
                         "Poisson deviance": comparison["v2"]["poisson_deviance"], "Bias": comparison["v2"]["bias"]},
                    ]
                ).round(3),
                hide_index=True,
                use_container_width=True,
            )
            st.markdown(
                '<p class="note">Retrospective diagnostic on the previously examined '
                "2025-09…2026-08 period. The deployment decision uses pre-test validation only; "
                "this table does not drive model selection.</p>",
                unsafe_allow_html=True,
            )

        st.markdown("## Group-level causal estimate")
        estimates = load_estimates()
        if estimates:
            main = pd.DataFrame(estimates.get("estimates", []))
            primary = main[
                main["method"].str.contains("Conditional Poisson", na=False)
                & ~main["method"].str.contains("Replication", na=False)
            ]
            if len(primary):
                row = primary.iloc[0]
                st.markdown(
                    f"Conditional Poisson FE: **{row['pct']:+.1f}%** "
                    f"(95% CI {row['pct_ci_low']:+.1f}% to {row['pct_ci_high']:+.1f}%), "
                    f"p = {row['pvalue']:.2f}, {int(row['n_clusters'])} cameras. "
                    "This is a **study-wide group-level estimate**, not a camera-specific effect. "
                    "The LightGBM model does not estimate this quantity."
                )

        st.markdown("## Model diagnostics")
        diag_cols = st.columns(2)
        with diag_cols[0]:
            ml_figure("04_actual_vs_predicted", "Held-out test accuracy.", figure_dir)
        with diag_cols[1]:
            ml_figure("07_calibration", "Calibration by predicted-risk decile.", figure_dir)

# ---------------------------------------------------------------------------
# Spatial explorer
# ---------------------------------------------------------------------------
elif page == "Chicago map":
    st.markdown("# Chicago map")
    st.markdown(
        '<p class="lede">Camera catchments (250 m radius) and comparison sites. Comparison sites are '
        "at least 600 m from every camera, so treated and comparison catchments cannot overlap.</p>",
        unsafe_allow_html=True,
    )

    sites = load_sites()
    if sites is None:
        st.info("`results/sites.parquet` not found. Run the pipeline.")
    else:
        roles = st.multiselect(
            "Show", sorted(sites["role"].unique()), default=sorted(sites["role"].unique())
        )
        subset = sites.loc[sites["role"].isin(roles)].dropna(subset=["lat", "lon"]).copy()
        subset["color"] = subset["role"].map(
            {"matched camera": "#b03a2e", "camera": "#b03a2e", "comparison site": "#2c6e9b"}
        ).fillna("#6b7785")
        st.map(subset, latitude="lat", longitude="lon", color="color", size=18)
        overlay = st.checkbox("Show reporting-era camera overlay", value=False)
        if overlay:
            cameras = load_cameras()
            if cameras is not None:
                camera_subset = cameras.dropna(subset=["lat", "lon"]).copy()
                camera_subset["go_live"] = pd.to_datetime(camera_subset["go_live"], errors="coerce")
                camera_subset["role"] = np.where(
                    camera_subset["go_live"] < pd.Timestamp("2017-09-01"),
                    "pre-citywide camera",
                    "post-citywide camera",
                )
                camera_subset["color"] = camera_subset["role"].map(
                    {"pre-citywide camera": "#b7791f", "post-citywide camera": "#6a51a3"}
                )
                st.map(
                    camera_subset[["lat", "lon", "color", "role"]],
                    latitude="lat", longitude="lon", color="color", size=26,
                )
                st.markdown(
                    '<p class="note">Gold cameras went live before citywide crash coverage '
                    "(September 2017); purple cameras follow it. The publication causal/forecast "
                    "window begins in 2015-09 and the V1/V2 target in 2017-01; the coverage-consistent "
                    "reconstruction starts in 2017-09. Cameras are not causal effects.</p>",
                    unsafe_allow_html=True,
                )
        st.markdown(
            f'<p class="note">{len(subset):,} points shown. Grey shading in the static map '
            "(Figure 1) is crash density. Coordinates are WGS84 for mapping; all analysis "
            "distances use UTM zone 16N metres.</p>",
            unsafe_allow_html=True,
        )
        figure("01_sites_map", "Figure 1. Cameras, matched cameras and comparison sites.")

    diagnostics = estimates.get("diagnostics", {})
    st.markdown("## Spatial validity diagnostics")
    rows = [
        ("Camera locations in source", diagnostics.get("n_cameras_total")),
        ("Cameras inside the event window", diagnostics.get("n_cameras_in_window")),
        ("Cameras dropped for catchment overlap", diagnostics.get("n_cameras_dropped_overlap")),
        ("Cameras with a matched comparison group", diagnostics.get("n_cameras_matched")),
        ("Candidate comparison sites", diagnostics.get("n_control_candidates")),
        ("Unique comparison sites used", diagnostics.get("n_controls_used_unique")),
        ("Comparison sites reused across cameras", diagnostics.get("n_controls_reused")),
        ("Minimum control-to-camera distance (m)", round(diagnostics.get("control_min_distance_to_camera_m", 0), 1)),
        ("Camera pairs with overlapping 250 m catchments", diagnostics.get("overlap_pairs")),
        ("Comparison-site pairs that overlap", diagnostics.get("control_overlap_pairs")),
        ("Treated/control catchment overlap", diagnostics.get("treated_control_catchment_overlap")),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Diagnostic", "Value"]),
        hide_index=True,
        use_container_width=True,
    )
    attribution = diagnostics.get("camera_attribution_all_cameras", {})
    if attribution:
        st.markdown("## Repeated crash attribution")
        st.markdown(
            f"Among all {diagnostics.get('n_cameras_total', 0)} cameras, "
            f"**{attribution.get('n_shared_crashes', 0):,} crashes** "
            f"({100 * attribution.get('shared_share', 0):.1f}% of attributed crashes) fall inside two "
            "or more 250 m catchments, and a crash can be counted by up to "
            f"**{attribution.get('max_sites_per_crash', 0)}** cameras. This is why cameras whose "
            "catchment overlaps an already-live camera are removed from the primary design."
        )
        speed = diagnostics.get("speed_limit_balance", {})
        if speed:
            smd = speed.get("standardised_mean_difference")
            smd_text = "not estimable (no variation)" if smd is None or pd.isna(smd) else f"{smd:+.2f}"
            st.markdown(
                f"Posted-speed-limit balance: median {speed.get('camera_median_mph', float('nan')):.0f} mph "
                f"at camera catchments vs {speed.get('control_median_mph', float('nan')):.0f} mph at "
                f"comparison sites (standardised difference {smd_text})."
            )

# ---------------------------------------------------------------------------
# Causal estimates
# ---------------------------------------------------------------------------
elif page == "Causal inference":
    st.markdown("# Causal inference")
    st.markdown(
        '<p class="lede">Effects are on reported crashes per site-month. Percent changes are '
        "reported for count models; level effects are converted to percent of the pre-period mean "
        "for comparability only.</p>",
        unsafe_allow_html=True,
    )

    if not main.empty:
        display = main.copy()
        display["estimate"] = display.apply(
            lambda row: fmt_pct(row["pct"])
            if row.get("scale") == "log" and pd.notna(row.get("pct"))
            else (f"{row['estimate']:+.3f}" if pd.notna(row.get("estimate")) else "—"),
            axis=1,
        )
        display["95% interval"] = display.apply(
            lambda row: pct_interval(row.get("pct_ci_low"), row.get("pct_ci_high"))
            if row.get("scale") == "log"
            else (
                f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
                if pd.notna(row.get("ci_low"))
                else "—"
            ),
            axis=1,
        )
        display["p"] = display["pvalue"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        display["cameras"] = display["n_clusters"]
        st.dataframe(
            display[["method", "estimate", "95% interval", "p", "cameras", "n_obs"]].rename(
                columns={"method": "Estimator", "n_obs": "Observations"}
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown(
            f'<p class="note">Preferred specifications: the conditional Poisson FE and the cohort '
            f"2x2 DiD. The replication rows use the original 2015+ sample and citywide matching. "
            f"Wild cluster bootstrap p-value for the FE OLS coefficient: "
            f'{estimates.get("wild_cluster_bootstrap", {}).get("pvalue", float("nan")):.3f} '
            f'(999 Rademacher replications).</p>',
            unsafe_allow_html=True,
        )

    st.markdown("## Event study")
    figure(
        "05_event_study",
        "Dynamic effects by quarter, relative to the quarter before go-live. "
        f"Joint pre-trend test p = {estimates.get('event_study_pre_trend', {}).get('pre_trend_pvalue', float('nan')):.3f}. "
        "A pre-trend test cannot prove parallel trends.",
    )

    st.markdown("## Synthetic control")
    fit_note = (
        "adequate" if sc.get("informative") else "INADEQUATE"
    )
    st.markdown(
        f"Largest go-live cohort: **{sc.get('cohort_size', '—')} cameras** ({sc.get('cohort_go_month', '—')}). "
        f"Pre-period R² = **{sc.get('pre_r2', float('nan')):.2f}** ({fit_note}), pre-RMSPE = "
        f"{sc.get('pre_rmspe', float('nan')):.2f} crashes/month against a pre-mean of "
        f"{sc.get('pre_mean', float('nan')):.2f}. Post-period gap = "
        f"**{sc.get('pct_vs_synth', float('nan')):+.1f}%** of the synthetic level "
        f"(pseudo-p for a negative gap = {sc.get('p_one_sided_negative', float('nan')):.2f})."
    )
    figure("06_synthetic_control", "Observed camera cohort vs synthetic comparison path.")

    st.markdown("## Balance")
    balance = estimates.get("balance", {})
    st.markdown(
        f"Standardised mean difference in pre-period crash level: "
        f"**{balance.get('standardised_mean_difference', float('nan')):+.2f}**. "
        f"Differential pre-trend: {balance.get('pre_trend_interaction_per_month', float('nan')):+.3f} "
        f"crashes/site-month per month (95% CI "
        f"{balance.get('pre_trend_ci', [float('nan'), float('nan')])[0]:+.3f} to "
        f"{balance.get('pre_trend_ci', [float('nan'), float('nan')])[1]:+.3f}), "
        f"p = {balance.get('pre_trend_pvalue', float('nan')):.3f}."
    )
    figure("02_balance", "Pre-period level and distribution: camera catchments vs comparison sites.")

# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
elif page == "Robustness & sensitivity":
    st.markdown("# Robustness and sensitivity")
    st.markdown(
        '<p class="lede">Every row is a declared design change, not a search for a favourable '
        "specification. Placebo tests are expected to be near zero.</p>",
        unsafe_allow_html=True,
    )

    if lattice:
        st.markdown(
            f"**Lattice phase.** The 500 m comparison lattice can be shifted arbitrarily. Across "
            f"{lattice.get('n_phases', 0)} phases the primary estimate ranges from "
            f"**{lattice.get('min_pct', float('nan')):+.1f}%** to "
            f"**{lattice.get('max_pct', float('nan')):+.1f}%** "
            f"(mean {lattice.get('mean_pct', float('nan')):+.1f}%). The phase is not identified from "
            "the data, so this spread is part of the uncertainty."
        )

    figure("10_sensitivity", "Figure 10. Sensitivity of the main estimate.")

    if not sensitivity.empty:
        show = sensitivity.copy()
        cols = ["analysis", "n_cameras", "did_pct", "did_pct_ci_low", "did_pct_ci_high", "did_pvalue", "status"]
        show = show[[c for c in cols if c in show.columns]]
        show.columns = [
            {"analysis": "Specification", "n_cameras": "Cameras", "did_pct": "Effect (%)",
             "did_pct_ci_low": "CI low", "did_pct_ci_high": "CI high", "did_pvalue": "p",
             "status": "Status"}.get(c, c)
            for c in show.columns
        ]
        st.dataframe(
            show,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Effect (%)": st.column_config.NumberColumn(format="%+.1f"),
                "CI low": st.column_config.NumberColumn(format="%+.1f"),
                "CI high": st.column_config.NumberColumn(format="%+.1f"),
                "p": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    st.markdown("## Placebo tests")
    figure(
        "07_placebo_distribution",
        "Left: synthetic-control placebos in space. Right: geographic placebos that rebuild the "
        "entire design at random comparison-site locations.",
    )
    placebo = estimates.get("placebo_geography", {})
    if placebo:
        st.markdown(
            f"Real estimate vs {placebo.get('n_placebo', 0)} geographic placebos: "
            f"one-sided p = **{placebo.get('p_one_sided_did', float('nan')):.2f}**; "
            f"placebo median {placebo.get('placebo_did_pct_median', float('nan')):+.1f}%, "
            f"5th–95th percentile [{placebo.get('placebo_did_pct_p05', float('nan')):+.1f}%, "
            f"{placebo.get('placebo_did_pct_p95', float('nan')):+.1f}%]."
        )

    st.markdown("## Subgroups (exploratory)")
    figure("08_subgroups", "Per-camera 2x2 effects grouped by geography, baseline risk and era.")
    if not subgroups.empty:
        show = subgroups.copy()
        keep = [c for c in ["dimension", "group", "n_cameras", "att_level", "ci_low", "ci_high", "pvalue", "pvalue_bh"] if c in show.columns]
        st.dataframe(show[keep], hide_index=True, use_container_width=True)

    st.markdown("## Severity outcomes")
    figure("09_severity", "Injury and severe/fatal crash outcomes.")
    if not severity.empty:
        st.dataframe(severity, hide_index=True, use_container_width=True)

    st.markdown("## Mechanism check")
    mech = estimates.get("mechanism", {})
    if mech.get("available"):
        st.markdown(
            f"Within-camera violations slope: **{mech.get('slope_pct_per_year', float('nan')):+.1f}% "
            f"per year** (SE {mech.get('slope_se', float('nan')):.3f} log points), "
            f"{mech.get('n_cameras', 0)} cameras, {mech.get('n_camera_months', 0):,} camera-months. "
            f"{mech.get('caveat', '')}"
        )
    else:
        st.info("Violations data were not available for the mechanism check.")

# ---------------------------------------------------------------------------
# Coverage and reconstruction
# ---------------------------------------------------------------------------
elif page == "Coverage & reconstruction":
    st.markdown("# Coverage & reconstruction")
    st.markdown(
        '<p class="lede">The official Chicago crash table is not citywide before September 2017. '
        "This page separates that reporting limitation from the camera-zone forecast and the "
        "road-segment research task.</p>",
        unsafe_allow_html=True,
    )
    coverage = load_bottom_json("coverage/summary.json")
    exposure = load_bottom_json("exposure/summary.json")
    road = load_bottom_json("roads/assignment_summary.json")
    road_metrics = load_bottom_json("road_model/road_metrics.json")
    common = load_bottom_csv("coverage_model/common_metrics.csv")
    causal = load_bottom_json("coverage_causal/estimates.json")

    cols = st.columns(4)
    if coverage:
        cols[0].metric("Snapshot crash rows", f"{coverage['snapshot_rows']:,}")
        cols[1].metric("New portal records after snapshot", f"{coverage['new_after_snapshot']:,}")
    if exposure:
        cols[2].metric("AADT first month", exposure["actual_first_month"])
        cols[3].metric("Road crashes assigned", f"{100 * road['assignment_rate']:.2f}%" if road else "—")

    st.markdown("## Citywide reporting break")
    monthly = load_bottom_csv("coverage/monthly_coverage.csv")
    if monthly is not None:
        monthly["month"] = pd.to_datetime(monthly["month"])
        chart = (
            alt.Chart(monthly)
            .mark_line(color="#2c6e9b", strokeWidth=1.6)
            .encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("crashes:Q", title="Reported crashes"),
                tooltip=["month:T", "crashes:Q", "injury_share:Q"],
            )
            .properties(height=280)
        )
        rule = (
            alt.Chart(pd.DataFrame({"month": [pd.Timestamp("2017-09-01")]}))
            .mark_rule(color="#b03a2e", strokeDash=[4, 4])
            .encode(x="month:T")
        )
        st.altair_chart(chart + rule, use_container_width=True)
        st.markdown(
            '<p class="note">The vertical line is the official citywide start (2017-09). '
            "The V1/V2 target starts in 2017-01; the coverage-consistent rebuild starts in "
            "2018-01 after four months of citywide history. Counts are not a completeness rate.</p>",
            unsafe_allow_html=True,
        )
        st.image(str(BOTTOM_UP / "figures" / "coverage_monthly.png"), caption="Coverage diagnostic: monthly crashes and injury share.", use_container_width=True)

    st.markdown("## Model comparison on common test rows")
    if common is not None:
        labels = {
            "v1": "Deployed V1",
            "v2": "V2 candidate",
            "coverage_v1": "Coverage V1",
            "coverage_full": "Coverage full",
            "coverage_tuned": "Coverage tuned",
        }
        view = common.copy()
        view["model"] = view["model"].map(labels).fillna(view["model"])
        st.dataframe(
            view[["model", "mae", "poisson_deviance", "bias", "r2"]].round(4),
            hide_index=True, use_container_width=True,
        )
        st.markdown(
            '<p class="note">All rows use the same 15,912 camera-zone test location-months. '
            "Coverage-tuned improves deviance by 0.65% versus deployed V1 on common rows, below "
            "the pre-declared 1% promotion rule. V1 remains deployed; this is not a new model release.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Common model comparison not found. Run the reconstruction scripts.")

    st.markdown("## Road-segment research task")
    if road and road_metrics:
        road_cols = st.columns(4)
        road_cols[0].metric("Road segments", "73,088")
        road_cols[1].metric("Assignment rate", f"{100 * road['assignment_rate']:.2f}%")
        road_cols[2].metric("Ambiguous assignments", f"{100 * road['ambiguous_assignments'] / road['assigned']:.1f}%")
        road_cols[3].metric("Road-model deviance", f"{road_metrics['test']['poisson_deviance']:.3f}")
        st.markdown(
            "<p class='note'>The road task uses all major segments plus a seeded local sample. "
            "It is predictive only. Its MAE is not comparable to the camera-zone task, and its "
            "highest-risk interval coverage is only 60.2%.</p>",
            unsafe_allow_html=True,
        )
        st.image(str(BOTTOM_UP / "figures" / "road_interval_coverage.png"), caption="Road-segment interval coverage by risk decile.", use_container_width=True)
    else:
        st.info("Road-segment audit not found. Run `python -m chicago_cameras.road_network`.")

    st.markdown("## Causal sensitivity")
    if causal:
        rows = []
        for row in causal:
            rows.append(
                {
                    "design": row["analysis"],
                    "cameras": row["n_cameras"],
                    "estimate_pct": row["did_pct"],
                    "ci_low": row["did_pct_ci_low"],
                    "ci_high": row["did_pct_ci_high"],
                    "p": row["did_pvalue"],
                    "pre_trend_p": row.get("event_pre_trend", {}).get("p"),
                }
            )
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
        st.markdown(
            '<p class="note">The coverage-consistent design is a new sample, not a direct '
            "replication. It leaves 15 cameras with 24 citywide pre-months and 36 post-months; "
            "the estimate is imprecise and the geographic placebo p-value is 0.25.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Coverage-consistent causal results not found.")

    st.markdown("## What changes, and what does not")
    st.markdown(
        """
- **Changes:** the publication's V1/V2 target starts eight months before citywide coverage;
  the coverage-consistent sensitivity and the road task are new evidence and should be
  disclosed.
- **Does not change:** V1 remains deployed; V2 remains a research candidate; the published
  causal estimate remains a historical specification; no new DOI, release or deployment is
  created.
- **Not comparable:** road-segment MAE and camera-zone MAE describe different populations and
  targets. Neither is a camera-specific causal effect.
"""
    )
    if coverage:
        st.caption(
            f"Snapshot {coverage['snapshot_first'][:10]} to {coverage['snapshot_last'][:10]}; "
            f"citywide start {coverage['citywide_start']}; AADT first month "
            f"{exposure['actual_first_month'] if exposure else 'unknown'}."
        )

# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------
else:
    st.markdown("# Methods and limitations")
    st.markdown(
        '<p class="lede">Definitions, identification assumptions, deviations from the original study '
        "and reproduction instructions.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("## Estimand")
    st.markdown(
        "The average effect of camera enforcement on **reported crashes per catchment-month** "
        "within 250 m, for cameras going live in 2018–2024 with a complete 24-month pre-period and "
        "36-month post-period. This is not crashes per vehicle-mile: no exposure denominator exists."
    )

    st.markdown("## Identification assumptions")
    st.markdown(
        """
1. **Parallel trends** conditional on matching pre-period level and slope. Tested but not proven.
2. **No anticipation** beyond the recorded go-live month (warning periods are not observed).
3. **No interference**: a 600 m buffer keeps catchments apart, but spillover beyond 250 m is untested.
4. **No differential shocks**: camera-by-month fixed effects absorb shocks shared with comparison sites.
5. **Camera siting is not driven by unobserved, time-varying local trends.** This is the weakest
   assumption; regression to the mean remains a threat.
6. **No contamination from earlier cameras**: overlapping-catchment cameras are dropped.
"""
    )

    st.markdown("## Deviations from the original study")
    st.markdown(
        """
| Original | Upgrade | Reason |
|---|---|---|
| Flat lon/lat projection | UTM 16N (EPSG:26916) | Correct metric distances |
| Citywide matching (README claimed 1 km) | 2 km local matching | Absorb local shocks; make the design feasible |
| Overlapping cameras kept | Drop cameras overlapping an already-live camera | Independent treated units |
| 2015-01 start | 2015-09 via coverage guard | Early months are missing, not zeros |
| Pooled OLS/Poisson | + FE OLS, conditional Poisson, cohort 2x2 | Staggered adoption and common shocks |
| No severity outcomes | Injury and severe/fatal outcomes | Policy-relevant margin |
"""
    )

    st.markdown("## Data provenance")
    provenance_path = ROOT / "data" / "raw" / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        rows = []
        for name, record in provenance.items():
            if not isinstance(record, dict):
                continue
            rows.append(
                {
                    "Dataset": name,
                    "Portal id": record.get("dataset_id"),
                    "Rows": record.get("rows"),
                    "Downloaded (UTC)": record.get("completed_at_utc"),
                    "SHA-256": (record.get("raw_sha256") or "")[:16] + "…",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.markdown(
            '<p class="note">Full URLs, query parameters and checksums are in '
            "<code>data/raw/provenance.json</code>.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No provenance file found. Run `python scripts/download_data.py`.")

    st.markdown("## Reproduction")
    st.code(
        "python -m venv .venv && source .venv/bin/activate\n"
        'pip install -e ".[dev,dashboard]"\n'
        "python scripts/download_data.py\n"
        "python -m chicago_cameras.pipeline --config config/analysis.yaml\n"
        "python scripts/verify_replication.py\n"
        "python -m pytest -q",
        language="bash",
    )
    st.markdown(
        "Detailed documentation: `docs/METHODOLOGY.md`, `docs/DATA_DICTIONARY.md`, "
        "`docs/AUDIT.md`, `docs/REPRODUCIBILITY.md`, `docs/FINAL_RELEASE_AUDIT.md`."
    )

    st.markdown("## What this dashboard does not do")
    st.markdown(
        "- It does not refit models in response to interactions.\n"
        "- It does not display demo or fabricated results.\n"
        "- It does not select the most favourable specification; the primary design is fixed in "
        "`config/analysis.yaml`."
    )
