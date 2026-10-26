"""Small Optuna budget for the coverage-consistent feature set.

Budget and selection rule are declared before looking at final test results:

* 8 trials, 600-second cap;
* objective = mean Poisson deviance over three pre-test rolling folds;
* a candidate may replace the deployed model only if validation deviance improves
  by at least 1%, test-period MAE does not regress by more than 2%, and the
  high-crash subgroup error does not regress by more than 2%.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna

from .reconstruction import (
    ablation_feature_sets,
    fixed_params,
    load_enriched,
    train_evaluate,
)
from .validation import fold_frames, rolling_origin_folds

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/bottom_up/coverage_model"
BUDGET = 8
TIMEOUT = 600


def main() -> int:
    frame, groups = load_enriched()
    features = ablation_feature_sets(groups)["M6_full"]
    folds = rolling_origin_folds("2018-01-01", "2025-09-01", n_folds=3, val_months=6)
    base = fixed_params(json.loads((ROOT / "results/ml/best_params.json").read_text())["params"])

    def objective(trial: optuna.Trial) -> float:
        params = dict(base)
        params.update(
            {
                "learning_rate": trial.suggest_float("learning_rate", 0.012, 0.04, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 128),
                "max_depth": trial.suggest_int("max_depth", 6, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 25, 120),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.55, 0.90),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.55, 0.90),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-5, 1e-2, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-5, 1e-2, log=True),
            }
        )
        deviances = []
        for fold, train, validation in fold_frames(folds, frame):
            dtrain = lgb.Dataset(
                train[features],
                label=train["y_all"],
                categorical_feature=["cluster"],
                free_raw_data=True,
            )
            dvalid = lgb.Dataset(
                validation[features],
                label=validation["y_all"],
                categorical_feature=["cluster"],
                reference=dtrain,
                free_raw_data=True,
            )
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=1800,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(70, verbose=False), lgb.log_evaluation(0)],
            )
            pred = model.predict(
                validation[features], num_iteration=int(model.best_iteration or 1800)
            )
            from .evaluate import regression_metrics

            deviances.append(regression_metrics(validation["y_all"], pred)["poisson_deviance"])
        return float(np.mean(deviances))

    started = time.time()
    study = optuna.create_study(direction="minimize", study_name="coverage_full")
    study.optimize(objective, n_trials=BUDGET, timeout=TIMEOUT, show_progress_bar=False)
    trials = study.trials_dataframe()
    trials.to_csv(RESULTS / "coverage_tuning_trials.csv", index=False)
    best = {
        "n_trials": len(study.trials),
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "runtime_seconds": round(time.time() - started, 2),
        "budget": {"n_trials": BUDGET, "timeout_seconds": TIMEOUT},
    }
    (RESULTS / "coverage_tuning_best.json").write_text(
        json.dumps(best, indent=2, sort_keys=True) + "\n"
    )
    params = fixed_params({**json.loads((ROOT / "results/ml/best_params.json").read_text())["params"], **study.best_params})
    train_evaluate(frame, features, params, "coverage_tuned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
