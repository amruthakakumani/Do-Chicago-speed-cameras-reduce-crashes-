"""Temporal validation for crash-count forecasting.

Months are never shuffled.  The test period is locked first; rolling-origin
folds are built strictly before it, so a model can never learn from the future.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def _month_start(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp()


def rolling_origin_folds(
    target_start: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    n_folds: int = 3,
    val_months: int = 6,
) -> list[Fold]:
    """Chronological folds ending before the untouched test period.

    With ``test_start = 2025-09`` and three six-month folds the validation
    periods are 2024-01..06, 2024-07..12 and 2025-01..06.  The two months
    immediately before the test period (2025-07..08) are reserved for the final
    refit and are never used to select hyperparameters.
    """
    start = _month_start(target_start)
    test = _month_start(test_start)
    if test <= start:
        raise ValueError("test_start must be after target_start")
    if n_folds < 1 or val_months < 1:
        raise ValueError("n_folds and val_months must be positive")

    folds: list[Fold] = []
    last_val_end = test - pd.offsets.MonthBegin(1)  # month before the test period
    for index in range(n_folds):
        val_end = last_val_end - pd.DateOffset(months=val_months * index)
        val_start = val_end - pd.DateOffset(months=val_months - 1)
        train_end = val_start - pd.offsets.MonthBegin(1)
        if val_start <= start:
            raise ValueError(
                "not enough history for the requested folds; shorten val_months "
                "or move test_start later"
            )
        folds.append(
            Fold(
                name=f"fold_{n_folds - index}",
                train_start=start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
            )
        )
    return list(reversed(folds))


def fold_frames(folds: list[Fold], frame: pd.DataFrame) -> list[tuple[Fold, pd.DataFrame, pd.DataFrame]]:
    """Materialise (fold, train, validation) frames with immutable month ordering."""
    out: list[tuple[Fold, pd.DataFrame, pd.DataFrame]] = []
    for fold in folds:
        train = frame.loc[frame["month"].between(fold.train_start, fold.train_end)].copy()
        validation = frame.loc[frame["month"].between(fold.val_start, fold.val_end)].copy()
        if train.empty or validation.empty:
            raise ValueError(f"{fold.name} has an empty train or validation frame")
        if train["month"].max() >= validation["month"].min():
            raise ValueError(f"{fold.name} leaks future months into training")
        out.append((fold, train, validation))
    return out


def temporal_split(
    frame: pd.DataFrame,
    test_start: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into pre-test (train/validation) and untouched test frames."""
    test = _month_start(test_start)
    train = frame.loc[frame["month"] < test].copy()
    test_frame = frame.loc[frame["month"] >= test].copy()
    if train.empty or test_frame.empty:
        raise ValueError("temporal split produced an empty side")
    if train["month"].max() >= test_frame["month"].min():
        raise ValueError("temporal split overlaps")
    return train, test_frame
