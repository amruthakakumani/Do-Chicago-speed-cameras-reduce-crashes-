"""Robustness, falsification and sensitivity analyses.

Every specification is a small, explicit change to :class:`DesignConfig`, so the
comparison table cannot hide a design choice.  Placebo tests are implemented for
both fake treatment dates and fake treated locations.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .design import DesignConfig, DesignResult, StudyData, build_design
from .did import (
    cohort_did_from_panel,
    did_conditional_poisson,
    did_fe_ols,
)

LOGGER = logging.getLogger(__name__)


def _subset_covid(panel: pd.DataFrame, window: Iterable[str]) -> pd.DataFrame:
    start, end = (pd.Timestamp(value) for value in window)
    months = panel["calendar_month"]
    return panel.loc[~months.between(start, end)].copy()


def evaluate_design(
    design: DesignResult,
    n_boot: int,
    seed: int,
    label: str,
    covid_window: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run the three main estimators on one design and return a flat row."""
    panel = design.panel
    if covid_window is not None:
        panel = _subset_covid(panel, covid_window)
        if panel.empty:
            raise ValueError("COVID exclusion removed every observation")

    fe = did_fe_ols(panel)
    conditional = did_conditional_poisson(panel)
    cohort = cohort_did_from_panel(
        panel,
        post_months=design.config.post_months,
        n_boot=n_boot,
        seed=seed,
    )

    pct_low, pct_high = conditional.pct_ci or (np.nan, np.nan)
    return {
        "analysis": label,
        "n_cameras": design.n_cameras,
        "n_obs": int(len(panel)),
        "did_pct": conditional.pct,
        "did_pct_ci_low": pct_low,
        "did_pct_ci_high": pct_high,
        "did_pvalue": conditional.pvalue,
        "fe_att_level": fe.beta,
        "fe_att_ci_low": fe.ci_low,
        "fe_att_ci_high": fe.ci_high,
        "fe_pvalue": fe.pvalue,
        "cohort_att_level": cohort.beta,
        "cohort_ci_low": cohort.ci_low,
        "cohort_ci_high": cohort.ci_high,
        "cohort_pvalue": cohort.pvalue,
        "pre_months": design.config.pre_months,
        "post_months": design.config.post_months,
        "radius_m": design.config.radius_m,
        "buffer_m": design.config.buffer_m,
        "n_controls": design.config.n_controls,
        "max_match_distance_m": design.config.max_match_distance_m,
        "treatment_month_offset": design.config.treatment_month_offset,
        "overlap_rule": design.config.overlap_rule,
    }


def _row(summary: dict[str, Any], **extra: Any) -> dict[str, Any]:
    row = {k: v for k, v in summary.items() if k != "event_table"}
    row.update(extra)
    return row


def _safe_evaluate(
    data: StudyData,
    config: DesignConfig,
    label: str,
    outcome: str,
    n_boot: int,
    seed: int,
    covid_window: Iterable[str] | None = None,
    camera_subset: np.ndarray | None = None,
) -> dict[str, Any] | None:
    try:
        design = build_design(
            data, config, outcome=outcome, camera_subset=camera_subset  # type: ignore[arg-type]
        )
        summary = evaluate_design(design, n_boot=n_boot, seed=seed, label=label, covid_window=covid_window)
        return _row(summary, outcome=outcome, status="ok")
    except (ValueError, RuntimeError) as exc:
        LOGGER.warning("skipping %s: %s", label, exc)
        return {
            "analysis": label,
            "outcome": outcome,
            "status": f"not_estimable: {exc}",
        }


def sensitivity_table(
    data: StudyData,
    base_config: DesignConfig,
    config: dict[str, Any],
    n_boot: int = 499,
    baseline_camera_rows: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run every pre-registered robustness variation and collect the results."""
    robustness = config.get("robustness", {})
    seed = int(config["inference"]["seed"])
    rows: list[dict[str, Any]] = []
    if baseline_camera_rows is not None:
        keep_mask = np.zeros(len(data.cameras), dtype=bool)
        keep_mask[np.asarray(baseline_camera_rows, dtype=int)] = True
    else:
        keep_mask = None

    def add(
        label: str,
        cfg: DesignConfig,
        outcome: str = "all",
        camera_subset: np.ndarray | None = None,
    ) -> None:
        result = _safe_evaluate(
            data, cfg, label, outcome, n_boot=n_boot, seed=seed, camera_subset=camera_subset
        )
        if result is not None:
            row = {
                "analysis": label,
                "outcome": outcome,
                "did_pct": np.nan,
                "did_pct_ci_low": np.nan,
                "did_pct_ci_high": np.nan,
                "did_pvalue": np.nan,
                "cohort_att_level": np.nan,
                "n_cameras": np.nan,
            }
            row.update(result)
            rows.append(row)

    add("Baseline: 250 m radius, 5 controls, 2,000 m matching, -24/+36 months", base_config)

    if robustness.get("run_radius", True):
        for radius in config["spatial"]["radius_sensitivity_m"]:
            add(
                f"Radius {radius} m",
                replace(base_config, radius_m=radius, buffer_m=max(base_config.buffer_m, 2 * radius + 100)),
            )

    if robustness.get("run_buffer", True):
        for buffer_m in config["spatial"]["buffer_sensitivity_m"]:
            add(f"Exclusion buffer {buffer_m} m", replace(base_config, buffer_m=buffer_m))

    if robustness.get("run_n_controls", True):
        for n_controls in config["design"]["n_controls_sensitivity"]:
            add(f"{n_controls} matched controls", replace(base_config, n_controls=n_controls))

    if robustness.get("run_match_distance", True):
        for distance in config["spatial"]["max_match_distance_sensitivity_m"]:
            add(
                f"Max match distance {distance} m",
                replace(base_config, max_match_distance_m=distance),
            )
        add(
            "Citywide matching (no distance cap)",
            replace(base_config, max_match_distance_m=None),
        )

    if robustness.get("run_horizon", True):
        for pre, post in config["design"]["pre_post_sensitivity"]:
            add(
                f"Window -{pre}/+{post} months",
                replace(base_config, pre_months=pre, post_months=post),
            )

    if robustness.get("run_covid", True):
        row = _safe_evaluate(
            data,
            base_config,
            "Excluding March 2020 - December 2021",
            "all",
            n_boot=n_boot,
            seed=seed,
            covid_window=robustness.get("covid_exclusion_months", ["2020-03-01", "2021-12-01"]),
        )
        if row is not None:
            rows.append(row)

    if robustness.get("run_injury", True):
        add("Injury crashes only", base_config, outcome="injury")
        add("Severe/fatal crashes only", base_config, outcome="severe")

    if robustness.get("run_grid_offset", True):
        for offset in config["spatial"].get("grid_phase_sensitivity_m", [125, 250, 375]):
            add(
                f"Lattice phase +{offset} m",
                replace(base_config, grid_offset_m=float(offset)),
            )

    if robustness.get("run_overlap", True):
        add("Keep overlapping cameras", replace(base_config, overlap_rule="keep_all"))

    if robustness.get("run_cohort_era", True):
        years = pd.to_datetime(data.cameras["go_live"]).dt.year
        if keep_mask is not None:
            early = keep_mask & (years < 2019).to_numpy()
            late = keep_mask & (years >= 2019).to_numpy()
        else:  # pragma: no cover - defensive
            early = (years < 2019).to_numpy()
            late = (years >= 2019).to_numpy()
        add(
            "2015-2018 go-live cohorts only (reporting transition in pre-period)",
            base_config,
            camera_subset=early,
        )
        add(
            "2019+ go-live cohorts only",
            base_config,
            camera_subset=late,
        )

    if robustness.get("run_placebo_dates", True):
        for offset in robustness.get("placebo_treatment_offsets_months", [-24, -12]):
            add(
                f"Placebo go-live shifted {offset} months",
                replace(base_config, treatment_month_offset=offset),
                # Hold the camera sample fixed so that only the fake treatment
                # date changes, not the set of treated units.
                camera_subset=keep_mask,
            )

    add(
        "Go-live month excluded (start next month)",
        replace(base_config, treatment_month_offset=1),
        camera_subset=keep_mask,
    )
    # A forward shift is not a placebo: it asks whether the effect persists.
    add(
        "Treatment lagged 12 months (persistence, not a placebo)",
        replace(base_config, treatment_month_offset=12),
        camera_subset=keep_mask,
    )

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.drop(columns=["event_table"], errors="ignore")
    return table


def _estimate_from_panel(
    panel: pd.DataFrame,
    post_months: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    fe = did_fe_ols(panel)
    conditional = did_conditional_poisson(panel)
    cohort = cohort_did_from_panel(
        panel, post_months=post_months, n_boot=n_boot, seed=seed
    )
    return {
        "did_pct": conditional.pct,
        "did_pvalue": conditional.pvalue,
        "fe_att_level": fe.beta,
        "cohort_att_level": cohort.beta,
        "cohort_pvalue": cohort.pvalue,
        "n_cameras": cohort.n_clusters,
    }


def geographic_placebo(
    data: StudyData,
    base_config: DesignConfig,
    n_placebo: int,
    seed: int,
    n_boot: int = 199,
) -> pd.DataFrame:
    """Assign cameras to random comparison-site locations and re-estimate.

    Pseudo-cameras are drawn from the *real* comparison-site pool, so their
    catchment counts are already computed and a full lattice rebuild is not
    needed.  Donors within the 600 m buffer of a pseudo-camera are removed, and
    overlapping pseudo-cameras are resolved with the same rule as the real design.
    A real effect should sit in the tail of this distribution; a null should not.
    """
    from scipy.spatial.distance import cdist

    from .design import resolve_overlaps
    from .matching import match_controls
    from .panel import build_panel

    baseline = build_design(data, base_config, outcome="all")
    camera_rows = sorted(baseline.matches.matches)
    n_cameras = len(camera_rows)
    if n_cameras == 0:
        raise ValueError("no matched cameras to build geographic placebos from")

    go_months = baseline.cameras.loc[camera_rows, "go_month"].to_numpy()
    control_xy = baseline.control_xy
    control_counts = baseline.control_counts
    control_ids = baseline.control_ids
    months = data.months
    offset = int(base_config.treatment_month_offset) + (
        int(np.ceil(base_config.warning_days / 30.0)) if base_config.warning_days else 0
    )
    rng = np.random.default_rng(seed)

    records: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(n_placebo * 5, 50)
    while len(records) < n_placebo and attempts < max_attempts:
        attempts += 1
        picked = rng.choice(len(control_xy), size=n_cameras, replace=False)
        pseudo_xy = control_xy[picked]
        go_live = pd.to_datetime(rng.choice(go_months, size=n_cameras, replace=True))
        go_idx = months.get_indexer(pd.to_datetime(go_live).values) + offset
        eligible = (go_idx >= base_config.pre_months) & (
            go_idx + base_config.post_months <= len(months)
        )
        if not eligible.any():
            continue

        pseudo = pd.DataFrame(
            {
                "camera_uid": [f"PLACEBO{attempts:04d}_{j:03d}" for j in range(n_cameras)],
                "x": pseudo_xy[:, 0],
                "y": pseudo_xy[:, 1],
                "go_live": go_live,
                "go_month": go_live,
            }
        )
        keep, _ = resolve_overlaps(
            pseudo, base_config.radius_m, eligible, base_config.overlap_rule
        )
        eligible &= keep
        rows = np.flatnonzero(eligible)
        if len(rows) == 0:
            continue

        # Donors: drop the pseudo-camera sites themselves and anything inside the
        # exclusion buffer of a pseudo-camera.
        distance = cdist(pseudo_xy[rows], control_xy)
        donor_mask = np.ones(len(control_xy), dtype=bool)
        donor_mask[picked] = False
        donor_mask &= distance.min(axis=0) > base_config.buffer_m
        donor_rows = np.flatnonzero(donor_mask)
        if len(donor_rows) < base_config.n_controls + 1:
            continue

        match = match_controls(
            camera_xy=pseudo_xy[rows],
            camera_rows=rows,
            camera_go_idx=go_idx[rows],
            Y_camera=control_counts,
            control_xy=control_xy[donor_rows],
            Y_control=control_counts[donor_rows],
            pre_months=base_config.pre_months,
            n_controls=base_config.n_controls,
            max_distance_m=base_config.max_match_distance_m,
            caliper=base_config.caliper,
        )
        if match.n_cameras == 0:
            continue

        panel = build_panel(
            Y_camera=control_counts,
            Y_control=control_counts[donor_rows],
            matches=match.matches,
            camera_go_idx={row: int(go_idx[row]) for row in match.matches},
            camera_uids={row: pseudo.at[row, "camera_uid"] for row in match.matches},
            control_ids=control_ids[donor_rows],
            pre_months=base_config.pre_months,
            post_months=base_config.post_months,
            camera_go_month={row: pd.Timestamp(go_live[row]) for row in match.matches},
        )
        try:
            summary = _estimate_from_panel(
                panel,
                post_months=base_config.post_months,
                n_boot=n_boot,
                seed=seed + attempts,
            )
        except (ValueError, RuntimeError) as exc:
            LOGGER.debug("placebo estimate skipped: %s", exc)
            continue

        records.append({"draw": attempts, **summary})

    return pd.DataFrame(records)


def placebo_summary(real: dict[str, Any], placebo: pd.DataFrame) -> dict[str, float]:
    """Rank the real estimate against the geographic placebo distribution."""
    if placebo.empty:
        return {}
    did = placebo["did_pct"].dropna().to_numpy()
    cohort = placebo["cohort_att_level"].dropna().to_numpy()
    return {
        "n_placebo": int(len(placebo)),
        "placebo_did_pct_median": float(np.median(did)) if len(did) else np.nan,
        "placebo_did_pct_p05": float(np.percentile(did, 5)) if len(did) else np.nan,
        "placebo_did_pct_p95": float(np.percentile(did, 95)) if len(did) else np.nan,
        "p_one_sided_did": float((did <= real["did_pct"]).mean()) if len(did) else np.nan,
        "placebo_cohort_median": float(np.median(cohort)) if len(cohort) else np.nan,
        "p_one_sided_cohort": float((cohort <= real["cohort_att_level"]).mean())
        if len(cohort)
        else np.nan,
    }
