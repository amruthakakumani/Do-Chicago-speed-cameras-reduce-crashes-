"""Optuna tuning for the single LightGBM Poisson model.

Tuning uses rolling-origin validation only.  The locked test period is never
used for tuning or model selection, and hyperparameters are selected on Poisson
deviance, never on causal statistical significance.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from .dataset import DatasetBundle
from .evaluate import poisson_deviance
from .validation import Fold, fold_frames, rolling_origin_folds

LOGGER = logging.getLogger(__name__)


@dataclass
class TuningResult:
    best_params: dict[str, Any]
    best_value: float
    best_iterations: list[int]
    n_trials: int
    runtime_seconds: float
    trials: pd.DataFrame = field(default_factory=pd.DataFrame)


def suggest_params(trial: optuna.Trial, seed: int, num_threads: int) -> dict[str, Any]:
    """Search space for the Poisson LightGBM model."""
    return {
        "objective": "poisson",
        "metric": "poisson",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.60, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.60, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "seed": seed,
        "bagging_seed": seed + 1,
        "feature_fraction_seed": seed + 2,
        "data_random_seed": seed + 3,
        "num_threads": num_threads,
        "deterministic": True,
        "force_col_wise": True,
        "min_data_in_bin": 5,
    }


def _objective(
    trial: optuna.Trial,
    bundle: DatasetBundle,
    folds: list[Fold],
    seed: int,
    num_threads: int,
    max_rounds: int,
    early_stopping: int,
) -> float:
    scores: list[float] = []
    best_iterations: list[int] = []
    prepared = fold_frames(folds, bundle.frame)
    for step, (fold, train, validation) in enumerate(prepared):
        params = suggest_params(trial, seed=seed + step, num_threads=num_threads)
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
        best_iteration = int(model.best_iteration or max_rounds)
        prediction = model.predict(
            validation[bundle.feature_columns], num_iteration=best_iteration
        )
        score = poisson_deviance(validation["y_all"].to_numpy(), prediction)
        scores.append(score)
        best_iterations.append(best_iteration)
        LOGGER.debug("%s trial=%s step=%s deviance=%.4f", fold.name, trial.number, step, score)

        trial.report(float(np.mean(scores)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("best_iters", best_iterations)
    trial.set_user_attr("fold_scores", scores)
    return float(np.mean(scores))


def run_tuning(
    bundle: DatasetBundle,
    n_trials: int = 40,
    timeout: int | None = 900,
    seed: int = 20261024,
    num_threads: int = 8,
    max_rounds: int = 3000,
    early_stopping: int = 100,
    n_folds: int = 3,
    val_months: int = 6,
) -> TuningResult:
    """Tune with TPE and median pruning on rolling-origin folds."""
    folds = rolling_origin_folds(
        bundle.metadata["target_window"]["first"],
        bundle.metadata["test_start"],
        n_folds=n_folds,
        val_months=val_months,
    )
    sampler = TPESampler(seed=seed, multivariate=True)
    pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=0)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    started = time.time()
    study.optimize(
        lambda trial: _objective(
            trial,
            bundle,
            folds,
            seed=seed,
            num_threads=num_threads,
            max_rounds=max_rounds,
            early_stopping=early_stopping,
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )
    runtime = time.time() - started

    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "duration"))
    best = study.best_trial
    best_iters = best.user_attrs.get("best_iters", [500] * len(folds))
    LOGGER.info(
        "Optuna finished: %s trials, best deviance %.4f, %.1fs",
        len(study.trials),
        study.best_value,
        runtime,
    )
    return TuningResult(
        best_params=dict(best.params),
        best_value=float(study.best_value),
        best_iterations=[int(value) for value in best_iters],
        n_trials=len(study.trials),
        runtime_seconds=float(runtime),
        trials=trials,
    )
