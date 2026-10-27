"""Fit and persist the single final LightGBM model."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .dataset import DatasetBundle
from .validation import temporal_split

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    booster: lgb.Booster
    params: dict[str, Any]
    num_boost_round: int
    training_seconds: float
    train_rows: int
    train_period: tuple[pd.Timestamp, pd.Timestamp]


def merge_params(tuned: dict[str, Any], seed: int, num_threads: int) -> dict[str, Any]:
    params = {
        "objective": "poisson",
        "metric": "poisson",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": seed,
        "bagging_seed": seed + 1,
        "feature_fraction_seed": seed + 2,
        "data_random_seed": seed + 3,
        "num_threads": num_threads,
        "deterministic": True,
        "force_col_wise": True,
        "min_data_in_bin": 5,
    }
    params.update({key: value for key, value in tuned.items() if key in {
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
        "lambda_l1",
        "lambda_l2",
    }})
    return params


def train_final(
    bundle: DatasetBundle,
    tuned_params: dict[str, Any],
    best_iterations: list[int],
    seed: int = 20261024,
    num_threads: int = 8,
    num_boost_round: int | None = None,
    max_rounds: int = 3000,
) -> TrainedModel:
    """Refit on all data before the untouched test period."""
    train, _ = temporal_split(bundle.frame, bundle.metadata["test_start"])
    params = merge_params(tuned_params, seed=seed, num_threads=num_threads)

    if num_boost_round is None:
        if best_iterations:
            num_boost_round = int(np.clip(np.mean(best_iterations) * 1.10, 50, max_rounds))
        else:  # pragma: no cover - defensive
            num_boost_round = 500

    dtrain = lgb.Dataset(
        train[bundle.feature_columns],
        label=train["y_all"],
        categorical_feature=bundle.categorical_features,
        free_raw_data=True,
    )
    started = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=int(num_boost_round))
    runtime = time.time() - started
    LOGGER.info(
        "final model trained on %s rows (%s..%s) for %s rounds in %.1fs",
        f"{len(train):,}",
        train["month"].min().date(),
        train["month"].max().date(),
        num_boost_round,
        runtime,
    )
    return TrainedModel(
        booster=booster,
        params=params,
        num_boost_round=int(num_boost_round),
        training_seconds=float(runtime),
        train_rows=len(train),
        train_period=(pd.Timestamp(train["month"].min()), pd.Timestamp(train["month"].max())),
    )


def save_model(model: TrainedModel, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster.save_model(str(path))
    return str(path)


def model_metadata(
    bundle: DatasetBundle,
    model: TrainedModel,
    tuning: Any | None,
    extra: dict[str, Any] | None = None,
    version: str = "1.0.0",
    model_file: str = "models/v1/lightgbm_crash_model.txt",
) -> dict[str, Any]:
    metadata = {
        "model_name": "lightgbm_poisson_crash_model",
        "model_version": version,
        "model_file": model_file,
        "objective": "poisson",
        "framework": f"lightgbm {lgb.__version__}",
        "seed": model.params.get("seed"),
        "training_period": {
            "first": str(model.train_period[0].date()),
            "last": str(model.train_period[1].date()),
        },
        "test_start": bundle.metadata["test_start"],
        "target_definition": bundle.metadata["target_definition"],
        "target_window": bundle.metadata["target_window"],
        "geographic_coverage": {
            "n_locations": bundle.metadata["n_locations"],
            "n_camera_locations": bundle.metadata["n_camera_locations"],
            "n_control_locations": bundle.metadata["n_control_locations"],
            "radius_m": bundle.metadata["radius_m"],
            "note": bundle.metadata["location_set"],
        },
        "data_snapshot": bundle.metadata["data_snapshot"],
        "feature_schema": bundle.feature_columns,
        "categorical_features": bundle.categorical_features,
        "n_features": len(bundle.feature_columns),
        "hyperparameters": model.params,
        "num_boost_round": model.num_boost_round,
        "training_rows": model.train_rows,
        "training_seconds": round(model.training_seconds, 3),
        "optuna": {
            "n_trials": getattr(tuning, "n_trials", None),
            "best_value": getattr(tuning, "best_value", None),
            "best_iterations": getattr(tuning, "best_iterations", None),
            "runtime_seconds": getattr(tuning, "runtime_seconds", None),
        },
        "causal_disclaimer": (
            "This model predicts observed crash counts. It does not estimate the causal "
            "effect of cameras, and feature importance must not be read causally."
        ),
    }
    if extra:
        metadata.update(extra)
    return metadata


def write_metadata(metadata: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")


def evaluate_folds(
    bundle: DatasetBundle,
    params: dict[str, Any],
    folds,
    num_threads: int = 8,
    max_rounds: int = 3000,
    early_stopping: int = 100,
) -> pd.DataFrame:
    """Out-of-fold validation metrics for a fixed hyperparameter set."""
    from .validation import fold_frames

    rows: list[dict[str, Any]] = []
    for fold, train, validation in fold_frames(list(folds), bundle.frame):
        dtrain = lgb.Dataset(
            train[bundle.feature_columns],
            label=train["y_all"],
            categorical_feature=bundle.categorical_features,
            free_raw_data=True,
        )
        dvalid = lgb.Dataset(
            validation[bundle.feature_columns],
            label=validation["y_all"],
            categorical_feature=bundle.categorical_features,
            reference=dtrain,
            free_raw_data=True,
        )
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=max_rounds,
            valid_sets=[dvalid],
            valid_names=["validation"],
            callbacks=[
                lgb.early_stopping(early_stopping, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        prediction = model.predict(
            validation[bundle.feature_columns],
            num_iteration=int(model.best_iteration or max_rounds),
        )
        from .evaluate import regression_metrics

        metrics = regression_metrics(validation["y_all"], prediction)
        metrics.update(
            {
                "fold": fold.name,
                "train_first": str(fold.train_start.date()),
                "train_last": str(fold.train_end.date()),
                "val_first": str(fold.val_start.date()),
                "val_last": str(fold.val_end.date()),
                "best_iteration": int(model.best_iteration or max_rounds),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)
