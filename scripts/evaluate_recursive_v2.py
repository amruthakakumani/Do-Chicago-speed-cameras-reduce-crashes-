#!/usr/bin/env python
"""Evaluate genuine 12-month recursive forecasts for V1 and V2.

One-step metrics use observed histories.  This script starts at 2025-08 and
forecasts 2025-09 … 2026-08 without observing intervening crashes.  Results are
reported separately and must not be confused with the one-step headline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chicago_cameras.ml.dataset import build_dataset
from chicago_cameras.ml.dataset_v2 import build_dataset_v2
from chicago_cameras.ml.predict import load_model
from chicago_cameras.ml.recursive import recursive_forecast


def _interval_coverage(predictions: pd.DataFrame, quantile: float, method: str) -> dict:
    scale = (
        np.sqrt(predictions["predicted"].to_numpy() + 1.0)
        if method == "normalized"
        else np.ones(len(predictions))
    )
    width = quantile * scale
    lower = np.clip(predictions["predicted"].to_numpy() - width, 0.0, None)
    upper = predictions["predicted"].to_numpy() + width
    covered = (predictions["actual"] >= lower) & (predictions["actual"] <= upper)
    return {
        "nominal": 0.90,
        "coverage": float(covered.mean()),
        "mean_width": float((upper - lower).mean()),
        "quantile": float(quantile),
        "note": "One-step conformal quantile applied to recursive forecasts; approximate.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default="2025-08-01")
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args()

    results_dir = ROOT / "results" / "ml_v2"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("building V1 and V2 datasets")
    v1_bundle = build_dataset("config/analysis.yaml")
    v2_bundle = build_dataset_v2("config/analysis.yaml")

    print("loading models")
    v1_model = load_model(ROOT / "models" / "v1" / "lightgbm_crash_model.txt")
    v2_model = load_model(ROOT / "models" / "v2_lightgbm_crash_model.txt")

    print("V1 recursive forecast")
    v1_result = recursive_forecast(
        v1_bundle, v1_model, v1_bundle.feature_columns, args.origin, args.horizon
    )
    print("V2 recursive forecast")
    v2_result = recursive_forecast(
        v2_bundle, v2_model, v2_bundle.feature_columns, args.origin, args.horizon
    )

    v1_result.predictions.to_parquet(results_dir / "recursive_v1_predictions.parquet", index=False)
    v2_result.predictions.to_parquet(results_dir / "recursive_predictions.parquet", index=False)

    deployment = json.loads((results_dir / "deployment.json").read_text())
    selected = deployment["selected_model"]
    conformal = json.loads((results_dir / "metrics.json").read_text())["conformal"]
    quantile = conformal["interval_90"]["quantile"]
    method = deployment.get("conformal_method", "normalized")

    payload = {
        "origin_month": args.origin,
        "horizon": args.horizon,
        "v1": {
            **v1_result.metrics,
            "recursive_interval_90": _interval_coverage(v1_result.predictions, quantile, method),
        },
        "v2": {
            **v2_result.metrics,
            "recursive_interval_90": _interval_coverage(v2_result.predictions, quantile, method),
        },
        "selected_model": selected,
        "note": (
            "Recursive forecasts feed predictions into subsequent lags and neighbourhood "
            "histories; future weather is from expanding climate normals. They are not "
            "comparable to the one-step-ahead headline metrics."
        ),
    }
    (results_dir / "recursive_metrics.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )

    print("\nV1 recursive (one path, 12 months)")
    print(
        pd.DataFrame(v1_result.metrics["by_horizon"])[
            ["horizon", "mae", "rmse", "poisson_deviance", "bias"]
        ].round(3).to_string(index=False)
    )
    print("\nV2 recursive (one path, 12 months)")
    print(
        pd.DataFrame(v2_result.metrics["by_horizon"])[
            ["horizon", "mae", "rmse", "poisson_deviance", "bias"]
        ].round(3).to_string(index=False)
    )
    print(f"\nwrote {results_dir / 'recursive_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
