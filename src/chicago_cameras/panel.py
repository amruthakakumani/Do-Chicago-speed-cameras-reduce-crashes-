"""Construction and validation of the stacked site-month panel."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_panel(
    Y_camera: np.ndarray,
    Y_control: np.ndarray,
    matches: dict[int, np.ndarray],
    camera_go_idx: dict[int, int],
    camera_uids: dict[int, str],
    control_ids: np.ndarray,
    pre_months: int,
    post_months: int,
    camera_go_month: dict[int, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Stack one treated unit and its matched controls into a long panel.

    Each camera contributes ``pre_months + post_months`` rows per unit.  Event
    time ``rel`` runs from ``-pre_months`` to ``post_months - 1``; ``rel = 0`` is
    the go-live month, matching the original study's convention.
    """
    rel = np.arange(-pre_months, post_months)
    if (rel >= 0).sum() != post_months:  # pragma: no cover - defensive
        raise ValueError("event-time window is inconsistent with the design")

    pieces: list[pd.DataFrame] = []
    for camera_row in sorted(matches):
        go_idx = int(camera_go_idx[camera_row])
        lo, hi = go_idx - pre_months, go_idx + post_months
        if lo < 0 or hi > Y_camera.shape[1]:
            raise ValueError(
                f"camera row {camera_row} has go_idx={go_idx}, outside the observed window"
            )
        stack_id = f"S{camera_uids[camera_row]}"
        if camera_go_month is not None:
            go_month = pd.Timestamp(camera_go_month[camera_row]).normalize()
            calendar = pd.date_range(
                go_month + pd.DateOffset(months=-pre_months),
                periods=pre_months + post_months,
                freq="MS",
            )
        else:  # pragma: no cover - used only by low-level tests
            calendar = pd.array([pd.NaT] * len(rel), dtype="datetime64[ns]")
        pieces.append(
            pd.DataFrame(
                {
                    "stack": stack_id,
                    "camera_row": camera_row,
                    "camera_uid": camera_uids[camera_row],
                    "unit": f"T{camera_uids[camera_row]}",
                    "treated": 1,
                    "rel": rel,
                    "calendar_month": calendar,
                    "y": Y_camera[camera_row, lo:hi].astype(np.int64),
                }
            )
        )
        for control_row in matches[camera_row]:
            pieces.append(
                pd.DataFrame(
                    {
                        "stack": stack_id,
                        "camera_row": camera_row,
                        "camera_uid": camera_uids[camera_row],
                        "unit": f"C{control_ids[control_row]}",
                        "treated": 0,
                        "rel": rel,
                        "calendar_month": calendar,
                        "y": Y_control[control_row, lo:hi].astype(np.int64),
                    }
                )
            )

    if not pieces:
        raise ValueError("no matched cameras: the panel would be empty")

    panel = pd.concat(pieces, ignore_index=True)
    panel["post"] = (panel["rel"] >= 0).astype("int8")
    panel["tp"] = (panel["treated"] * panel["post"]).astype("int8")
    panel["quarter"] = np.floor_divide(panel["rel"], 3).astype(int)
    return panel.sort_values(["stack", "treated", "unit", "rel"], ascending=[True, False, True, True]).reset_index(drop=True)


def validate_panel(panel: pd.DataFrame, pre_months: int, post_months: int) -> dict[str, object]:
    """Check completeness, uniqueness and the treated/control composition."""
    expected_per_stack = pre_months + post_months
    problems: list[str] = []

    if panel["y"].isna().any():
        problems.append("missing outcome values")
    if (panel["y"] < 0).any():
        problems.append("negative crash counts")
    if panel.duplicated(subset=["stack", "unit", "rel"]).any():
        problems.append("duplicate stack-unit-month rows")
    if not set(panel["rel"].unique()) == set(range(-pre_months, post_months)):
        problems.append("event-time grid is incomplete")

    per_stack = panel.groupby("stack")["unit"].nunique()
    per_stack_rows = panel.groupby("stack").size()
    if per_stack_rows.ne(per_stack * expected_per_stack).any():
        problems.append("stack row counts do not equal units x window length")
    if panel.groupby("stack")["treated"].sum().ne(expected_per_stack).any():
        problems.append("each stack must have exactly one treated unit per month")

    treated_series = (
        panel.loc[panel["treated"] == 1].groupby("stack")["y"].sum()
        - panel.loc[panel["treated"] == 0].groupby(["stack", "unit"])["y"].sum().groupby("stack").mean()
    )

    return {
        "n_rows": int(len(panel)),
        "n_stacks": int(panel["stack"].nunique()),
        "n_units": int(panel["unit"].nunique()),
        "controls_per_stack": {str(k): int(v) for k, v in (per_stack - 1).value_counts().sort_index().items()},
        "months_per_stack": int(expected_per_stack),
        "zero_crash_share": round(float((panel["y"] == 0).mean()), 4),
        "problems": problems,
        "mean_treated_minus_control_crash_gap": round(float(treated_series.mean()), 3),
    }
