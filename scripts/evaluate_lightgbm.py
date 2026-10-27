#!/usr/bin/env python
"""Evaluate the saved LightGBM model on the untouched test period.

This script never trains and never tunes.  It loads the persisted model, rebuilds
the verified location-month dataset, checks the feature schema and prints the
out-of-sample metrics, calibration and baselines.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chicago_cameras.ml.dataset import build_dataset
from chicago_cameras.ml.evaluate import full_evaluation
from chicago_cameras.ml.predict import attach_predictions, load_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--model", default="models/v1/lightgbm_crash_model.txt")
    parser.add_argument("--json", default="results/ml/evaluation.json")
    args = parser.parse_args()

    bundle = build_dataset(args.config, refresh=False)
    booster = load_model(ROOT / args.model)
    inference_started = time.perf_counter()
    predictions = attach_predictions(bundle, booster)
    inference_seconds = time.perf_counter() - inference_started
    test = predictions.loc[predictions["split"] == "test"].reset_index(drop=True)
    evaluation = full_evaluation(test)

    metrics = evaluation["test_metrics"]
    print(f"test period: {evaluation['test_period']['first']} .. {evaluation['test_period']['last']}")
    print(f"rows: {metrics['n']:,}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Poisson deviance: {metrics['poisson_deviance']:.4f}")
    print(f"Bias: {metrics['bias']:+.4f} ({metrics['relative_bias_pct']:+.2f}%)")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"inference: {inference_seconds:.2f}s for {len(predictions):,} location-months")
    print("\nbaselines on the same rows")
    print(
        pd.DataFrame(evaluation["baselines"])[
            ["baseline", "n", "mae", "rmse", "poisson_deviance", "bias"]
        ]
        .round(4)
        .to_string(index=False)
    )

    destination = ROOT / args.json
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evaluation, indent=2, default=str) + "\n")
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
