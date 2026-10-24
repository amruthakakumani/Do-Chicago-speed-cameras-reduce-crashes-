"""Streamlit dashboard for the Chicago speed-camera study.

The dashboard is a *reader* of precomputed, validated artefacts.  It never
refits a model, and it never displays illustrative or fabricated numbers: if the
pipeline has not been run, it shows an explicit empty state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

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
def load_sites() -> pd.DataFrame | None:
    path = RESULTS / "sites.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


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
    ["Overview", "Spatial explorer", "Causal estimates", "Robustness", "Methodology"],
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
if page == "Overview":
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
# Spatial explorer
# ---------------------------------------------------------------------------
elif page == "Spatial explorer":
    st.markdown("# Spatial explorer")
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
        subset = sites.loc[sites["role"].isin(roles)].dropna(subset=["lat", "lon"])
        st.map(subset, latitude="lat", longitude="lon", color="role", size=18)
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
elif page == "Causal estimates":
    st.markdown("# Causal estimates")
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
elif page == "Robustness":
    st.markdown("# Robustness and falsification")
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
# Methodology
# ---------------------------------------------------------------------------
else:
    st.markdown("# Methodology and provenance")
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
        "`docs/AUDIT.md`, `docs/REPRODUCIBILITY.md`, `docs/REPORT.md`."
    )

    st.markdown("## What this dashboard does not do")
    st.markdown(
        "- It does not refit models in response to interactions.\n"
        "- It does not display demo or fabricated results.\n"
        "- It does not select the most favourable specification; the primary design is fixed in "
        "`config/analysis.yaml`."
    )
