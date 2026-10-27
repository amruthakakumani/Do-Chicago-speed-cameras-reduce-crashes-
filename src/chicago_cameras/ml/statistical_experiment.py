"""Statistical diagnostics and hybrid forecasting experiment.

This module is deliberately separate from the causal/publication pipeline.  All
candidates use the coverage-consistent camera-zone dataset and the same
location-month rows, target definition and validation folds.
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from statsmodels.genmod.families import NegativeBinomial, Poisson
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .conformal import apply_conformal, select_method
from .evaluate import calibration_table, regression_metrics, subgroup_errors
from .geographic import assign_blocks, geographic_frames, geographic_split
from .validation import fold_frames, rolling_origin_folds, temporal_split

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "results/statistical_hybrid"
FIGURES = RESULTS / "figures"
NB_FEATURES = [
    "lag_1",
    "lag_3",
    "roll_mean_3",
    "roll_mean_12",
    "expanding_mean",
    "roll_std_12",
    "zero_share_12",
    "trend_12",
    "seasonal_ratio",
    "trend_6",
    "injury_share_12",
    "month_sin",
    "month_cos",
    "is_covid",
    "temp_lag1",
    "precip_lag1",
    "snow_lag1",
    "camera_operational",
    "n_cameras_1km",
    "dist_center_m",
    "road_density_500m_km_per_km2",
    "major_road_share_500m",
    "n_intersections_1000m",
    "dist_major_intersection_m",
    "neighbor_crash_mean_12",
    "neighbor_crash_lag1",
]


def load_frame() -> pd.DataFrame:
    frame = pd.read_parquet(PROCESSED / "ml_coverage_enriched.parquet")
    frame["month"] = pd.to_datetime(frame["month"])
    return frame


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nb_design(frame: pd.DataFrame, medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    numeric = frame[NB_FEATURES].copy()
    clusters = pd.get_dummies(frame["cluster"].astype("category"), prefix="cluster", drop_first=True, dtype=float)
    design = pd.concat([numeric, clusters], axis=1)
    if medians is None:
        medians = design.median(numeric_only=True)
    design = design.fillna(medians)
    design = sm.add_constant(design, has_constant="add")
    return design, medians


def _fit_nb(train: pd.DataFrame) -> tuple[Any, pd.Series, float, pd.DataFrame]:
    X, medians = _nb_design(train)
    y = train["y_all"].to_numpy(float)
    offset = np.log(train["expanding_mean"].clip(lower=0.5).to_numpy(float))
    poisson = sm.GLM(y, X, family=Poisson(), offset=offset).fit()
    mu = np.clip(poisson.predict(X, offset=offset), 1e-9, None)
    pearson = float(np.sum((y - mu) ** 2 / mu) / max(len(y) - X.shape[1], 1))
    alpha = max((pearson - 1.0) / max(float(mu.mean()), 1e-9), 1e-6)
    model = sm.GLM(y, X, family=NegativeBinomial(alpha=alpha), offset=offset).fit()
    vif_rows = []
    for index, column in enumerate(X.columns):
        try:
            value = float(variance_inflation_factor(X.to_numpy(float), index))
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            value = float("nan")
        vif_rows.append({"feature": str(column), "vif": value})
    return model, medians, alpha, pd.DataFrame(vif_rows)


def _predict_nb(model, medians: pd.Series, frame: pd.DataFrame) -> np.ndarray:
    X, _ = _nb_design(frame, medians)
    X = X.reindex(columns=model.model.exog_names, fill_value=0.0)
    offset = np.log(frame["expanding_mean"].clip(lower=0.5).to_numpy(float))
    return np.clip(np.asarray(model.predict(X, offset=offset), dtype=float), 0.0, None)


def _fit_nb_folds(frame: pd.DataFrame, folds) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    alphas = []
    for fold, train, validation in fold_frames(list(folds), frame):
        model, medians, alpha, _ = _fit_nb(train)
        prediction = _predict_nb(model, medians, validation)
        out = validation[["location_id", "month", "y_all", "site_type", "cluster"]].copy()
        out["predicted"] = prediction
        out["fold"] = fold.name
        rows.append(out)
        alphas.append(alpha)
    return pd.concat(rows, ignore_index=True), {"alphas": alphas, "mean_alpha": float(np.mean(alphas))}


def _fit_nb_final(frame: pd.DataFrame, test_start: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    train, test = temporal_split(frame, test_start)
    model, medians, alpha, vif = _fit_nb(train)
    prediction = _predict_nb(model, medians, test)
    out = test[["location_id", "month", "y_all", "site_type", "cluster"]].copy()
    out["predicted"] = prediction
    return out, {"alpha": alpha, "vif": vif.to_dict(orient="records")}


def _hybrid_weight(calibration: pd.DataFrame, lgbm_column: str = "lgbm", nb_column: str = "nb") -> tuple[float, pd.DataFrame]:
    work = calibration.dropna(subset=[lgbm_column, nb_column, "y_all"]).copy()
    def objective(weight: float) -> float:
        prediction = weight * work[lgbm_column].to_numpy(float) + (1 - weight) * work[nb_column].to_numpy(float)
        return regression_metrics(work["y_all"], prediction)["poisson_deviance"]
    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-5})
    curve = pd.DataFrame(
        [
            {
                "weight_lightgbm": weight,
                "poisson_deviance": objective(weight),
                "mae": regression_metrics(
                    work["y_all"],
                    weight * work[lgbm_column].to_numpy(float) + (1 - weight) * work[nb_column].to_numpy(float),
                )["mae"],
            }
            for weight in np.linspace(0, 1, 101)
        ]
    )
    return float(result.x), curve


def _hotspot(frame: pd.DataFrame, prediction: str, training_y: np.ndarray) -> dict[str, Any]:
    threshold = float(np.quantile(training_y, 0.90))
    high = frame["y_all"] >= threshold
    k = max(1, round(0.10 * len(frame)))
    top = frame.nlargest(k, prediction)
    top_high = top["y_all"] >= threshold
    return {
        "actual_high_threshold_from_training_y": threshold,
        "k": k,
        "precision": float(top_high.mean()),
        "recall": float(top_high.sum() / max(high.sum(), 1)),
        "lift": float(top_high.mean() / max(high.mean(), 1e-12)),
        "n_actual_high": int(high.sum()),
    }


def _candidate_metrics(frame: pd.DataFrame, prediction: str, training_y: np.ndarray) -> dict[str, Any]:
    metrics = regression_metrics(frame["y_all"], frame[prediction])
    metrics["calibration"] = calibration_table(frame["y_all"], frame[prediction]).to_dict(orient="records")
    metrics["subgroup_errors"] = subgroup_errors(frame["y_all"], frame[prediction])
    metrics["hotspot"] = _hotspot(frame, prediction, training_y)
    return metrics


def _interval_result(calibration: pd.DataFrame, test: pd.DataFrame, prediction: str) -> dict[str, Any]:
    method = select_method(calibration, alpha=0.1)
    cal = calibration[["y_all", prediction, "month"]].rename(columns={prediction: "predicted"})
    test_frame = test.copy()
    test_frame["predicted"] = test_frame[prediction]
    interval, metrics = apply_conformal(cal, test_frame, method["selected"], alpha=0.1)
    test_frame["lower_90"] = interval.lower
    test_frame["upper_90"] = interval.upper
    metrics["method_selection"] = method
    return {"metrics": metrics, "test": test_frame}


def _lgbm_predict(train: pd.DataFrame, validation: pd.DataFrame, metrics: dict[str, Any]) -> np.ndarray:
    params = dict(metrics["params"])
    params["verbosity"] = -1
    model = lgb.train(
        params,
        lgb.Dataset(
            train[metrics["features"]],
            label=train["y_all"],
            categorical_feature=["cluster"] if "cluster" in metrics["features"] else [],
            free_raw_data=True,
        ),
        num_boost_round=int(metrics["rounds"]),
    )
    return np.clip(model.predict(validation[metrics["features"]]), 0.0, None)


def _bootstrapped_differences(test: pd.DataFrame, locations: pd.DataFrame, baseline: str = "coverage_v1") -> dict[str, Any]:
    blocks = assign_blocks(locations[["location_id", "x", "y"]], 2000)[["location_id", "block_id"]]
    work = test.merge(blocks, on="location_id", how="left")
    rng = np.random.default_rng(20261024)
    groups = {block: np.flatnonzero(work["block_id"].eq(block).to_numpy()) for block in work["block_id"].dropna().unique()}
    blocks = list(groups)
    result: dict[str, Any] = {}
    for candidate in ("coverage_v1", "coverage_tuned", "nb", "hybrid"):
        if candidate not in work or candidate == baseline:
            continue
        values_dev, values_mae = [], []
        for _ in range(1000):
            sample = rng.choice(blocks, size=len(blocks), replace=True)
            idx = np.concatenate([groups[block] for block in sample])
            y = work["y_all"].to_numpy(float)[idx]
            a = work[candidate].to_numpy(float)[idx]
            b = work[baseline].to_numpy(float)[idx]
            values_dev.append(
                regression_metrics(y, a)["poisson_deviance"] - regression_metrics(y, b)["poisson_deviance"]
            )
            values_mae.append(regression_metrics(y, a)["mae"] - regression_metrics(y, b)["mae"])
        result[f"{candidate}_minus_{baseline}"] = {
            "deviance": {
                "mean": float(np.mean(values_dev)),
                "ci_low": float(np.percentile(values_dev, 2.5)),
                "ci_high": float(np.percentile(values_dev, 97.5)),
            },
            "mae": {
                "mean": float(np.mean(values_mae)),
                "ci_low": float(np.percentile(values_mae, 2.5)),
                "ci_high": float(np.percentile(values_mae, 97.5)),
            },
        }
    return result


def run_experiment() -> dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    locations = pd.read_parquet(PROCESSED / "ml_coverage_locations.parquet")
    coverage_v1 = pd.read_parquet(RESULTS.parent / "bottom_up/coverage_model/coverage_v1_test_predictions.parquet")
    coverage_tuned = pd.read_parquet(RESULTS.parent / "bottom_up/coverage_model/coverage_tuned_test_predictions.parquet")
    lgb_oof = pd.read_parquet(RESULTS.parent / "bottom_up/coverage_model/coverage_tuned_oof_predictions.parquet")
    tuned_metrics = json.loads((RESULTS.parent / "bottom_up/coverage_model/coverage_tuned_metrics.json").read_text())
    v1_metrics = json.loads((RESULTS.parent / "bottom_up/coverage_model/coverage_v1_metrics.json").read_text())
    test = frame.loc[frame["month"] >= "2025-09-01"].copy()
    train = frame.loc[frame["month"] < "2025-09-01"].copy()

    # EDA / data audit
    y = frame["y_all"]
    monthly = frame.groupby("month", as_index=False).agg(crashes=("y_all", "sum"), mean=("y_all", "mean"), zeros=("y_all", lambda s: int(s.eq(0).sum())))
    monthly["month_of_year"] = monthly["month"].dt.month
    seasonal = monthly.groupby("month_of_year", as_index=False)["crashes"].mean()
    year_stats = frame.assign(year=frame["month"].dt.year).groupby("year", as_index=False).agg(crashes=("y_all", "sum"), mean=("y_all", "mean"), zeros=("y_all", lambda s: float(s.eq(0).mean())))
    missing = frame[NB_FEATURES].isna().mean().rename("missing_share").reset_index().rename(columns={"index": "feature"})
    corr_features = [
        "lag_1", "roll_mean_3", "roll_mean_12", "expanding_mean", "zero_share_12", "trend_12",
        "temp_lag1", "precip_lag1", "month_sin", "camera_operational", "road_density_500m_km_per_km2",
        "n_cameras_1km", "dist_center_m", "site_is_camera",
    ]
    corr = frame[corr_features].corr()
    feature_redundancy = [
        {"feature_a": corr.columns[i], "feature_b": corr.columns[j], "correlation": float(corr.iloc[i, j])}
        for i in range(len(corr))
        for j in range(i + 1, len(corr))
        if abs(float(corr.iloc[i, j])) >= 0.90
    ]
    data_audit = {
        "rows": len(frame),
        "locations": int(frame["location_id"].nunique()),
        "camera_locations": int(frame.loc[frame["site_type"] == "camera", "location_id"].nunique()),
        "comparison_locations": int(frame.loc[frame["site_type"] == "control", "location_id"].nunique()),
        "target_first": str(frame["month"].min().date()),
        "target_last": str(frame["month"].max().date()),
        "months": int(frame["month"].nunique()),
        "test_rows": len(test),
        "test_first": str(test["month"].min().date()),
        "test_last": str(test["month"].max().date()),
        "training_rows": len(train),
        "target_mean": float(y.mean()),
        "target_variance": float(y.var()),
        "zero_share": float(y.eq(0).mean()),
        "overdispersion_variance_mean": float(y.var() / y.mean()),
        "max_target": int(y.max()),
        "min_location_spacing_m": float(
            cdist(locations[["x", "y"]].to_numpy(float), locations[["x", "y"]].to_numpy(float))[
                np.triu_indices(len(locations), 1)
            ].min()
        ),
        "data_fingerprint_sha256": fingerprint(PROCESSED / "ml_coverage_enriched.parquet"),
        "nb_missing": missing.to_dict(orient="records"),
        "feature_redundancy_abs_corr_ge_0_90": feature_redundancy,
        "monthly": monthly.to_dict(orient="records"),
        "seasonal": seasonal.to_dict(orient="records"),
        "year_stats": year_stats.to_dict(orient="records"),
    }
    (RESULTS / "data_audit.json").write_text(json.dumps(data_audit, indent=2, sort_keys=True, default=str) + "\n")
    (RESULTS / "feature_missing.csv").write_text(missing.to_csv(index=False))
    (RESULTS / "feature_correlation.csv").write_text(corr.to_csv())

    # Temporal folds and NB
    folds = rolling_origin_folds("2018-01-01", "2025-09-01", n_folds=3, val_months=6)
    started = time.time()
    nb_oof, nb_fold_info = _fit_nb_folds(frame, folds)
    nb_test, nb_test_info = _fit_nb_final(frame, "2025-09-01")
    nb_seconds = time.time() - started
    nb_oof.to_parquet(RESULTS / "nb_oof_predictions.parquet", index=False)
    nb_test.to_parquet(RESULTS / "nb_test_predictions.parquet", index=False)
    (RESULTS / "nb_diagnostics.json").write_text(
        json.dumps({"fold_info": nb_fold_info, "test_info": nb_test_info, "fit_seconds": nb_seconds}, indent=2, sort_keys=True, default=str) + "\n"
    )

    # Align OOF and learn hybrid weight; test predictions
    lgb_oof = lgb_oof.rename(columns={"predicted": "lgbm"})
    nb_oof = nb_oof.rename(columns={"predicted": "nb"})
    calibration = lgb_oof[["location_id", "month", "y_all", "lgbm"]].merge(
        nb_oof[["location_id", "month", "nb"]], on=["location_id", "month"], how="inner"
    )
    weight, weight_curve = _hybrid_weight(calibration)
    weight_curve.to_csv(RESULTS / "hybrid_weight_curve.csv", index=False)
    hybrid_oof = calibration.copy()
    hybrid_oof["hybrid"] = weight * hybrid_oof["lgbm"] + (1 - weight) * hybrid_oof["nb"]

    test_all = test[["location_id", "month", "y_all", "site_type", "cluster"] + NB_FEATURES].copy()
    test_all = test_all.merge(
        coverage_v1[["location_id", "month", "predicted"]].rename(columns={"predicted": "coverage_v1"}),
        on=["location_id", "month"], how="left",
    ).merge(
        coverage_tuned[["location_id", "month", "predicted"]].rename(columns={"predicted": "coverage_tuned"}),
        on=["location_id", "month"], how="left",
    ).merge(
        nb_test[["location_id", "month", "predicted"]].rename(columns={"predicted": "nb"}),
        on=["location_id", "month"], how="left",
    )
    test_all["hybrid"] = weight * test_all["coverage_tuned"] + (1 - weight) * test_all["nb"]
    test_all.to_parquet(RESULTS / "test_predictions.parquet", index=False)

    training_y = train["y_all"].to_numpy(float)
    candidates = ["coverage_v1", "coverage_tuned", "nb", "hybrid"]
    test_metrics = {name: _candidate_metrics(test_all, name, training_y) for name in candidates}
    oof_frames = {
        "coverage_v1": pd.read_parquet(RESULTS.parent / "bottom_up/coverage_model/coverage_v1_oof_predictions.parquet").rename(columns={"predicted": "coverage_v1"}),
        "coverage_tuned": lgb_oof.rename(columns={"lgbm": "coverage_tuned"}),
        "nb": nb_oof.rename(columns={"nb": "nb"}),
        "hybrid": hybrid_oof.rename(columns={"hybrid": "hybrid"}),
    }
    validation_metrics = {}
    for name, frame_oof in oof_frames.items():
        frame_oof = frame_oof.dropna(subset=[name])
        validation_metrics[name] = {
            "n": len(frame_oof),
            "mae": regression_metrics(frame_oof["y_all"], frame_oof[name])["mae"],
            "poisson_deviance": regression_metrics(frame_oof["y_all"], frame_oof[name])["poisson_deviance"],
        }
    intervals = {}
    for name, frame_oof in oof_frames.items():
        cal = frame_oof[["y_all", "month"]].copy()
        cal["predicted"] = frame_oof[name]
        test_frame = test_all.copy()
        test_frame["predicted"] = test_frame[name]
        method = select_method(cal, alpha=0.1)
        interval, metrics = apply_conformal(cal, test_frame, method["selected"], alpha=0.1)
        intervals[name] = {"metrics": metrics, "test": test_frame}
        test_all[f"{name}_lower_90"] = interval.lower
        test_all[f"{name}_upper_90"] = interval.upper
    test_all.to_parquet(RESULTS / "test_predictions.parquet", index=False)

    # Geographic validation on the same 2 km block split used in the publication
    split = geographic_split(locations, holdout_fraction=0.20, buffer_m=1000, seed=20261024)
    geo_train, geo_val = geographic_frames(frame, split, "2024-08-01", "2024-09-01", "2025-08-01")
    geo = geo_val[["location_id", "month", "y_all", "site_type", "cluster"]].copy()
    geo["coverage_tuned"] = _lgbm_predict(geo_train, geo_val, tuned_metrics)
    geo_nb_model, geo_medians, geo_alpha, _ = _fit_nb(geo_train)
    geo["nb"] = _predict_nb(geo_nb_model, geo_medians, geo_val)
    geo["hybrid"] = weight * geo["coverage_tuned"] + (1 - weight) * geo["nb"]
    geo["coverage_v1"] = _lgbm_predict(geo_train, geo_val, v1_metrics)
    geo_metrics = {name: _candidate_metrics(geo, name, geo_train["y_all"].to_numpy(float)) for name in candidates}
    geo.to_parquet(RESULTS / "geographic_predictions.parquet", index=False)

    # Combined geographic + temporal holdout, retrospective on the test period.
    combined_train, combined_val = geographic_frames(frame, split, "2025-08-01", "2025-09-01", "2026-08-01")
    combined = combined_val[["location_id", "month", "y_all", "site_type", "cluster"]].copy()
    combined["coverage_tuned"] = _lgbm_predict(combined_train, combined_val, tuned_metrics)
    combined_nb, combined_medians, combined_alpha, _ = _fit_nb(combined_train)
    combined["nb"] = _predict_nb(combined_nb, combined_medians, combined_val)
    combined["hybrid"] = weight * combined["coverage_tuned"] + (1 - weight) * combined["nb"]
    combined["coverage_v1"] = _lgbm_predict(combined_train, combined_val, v1_metrics)
    combined_metrics = {name: _candidate_metrics(combined, name, combined_train["y_all"].to_numpy(float)) for name in candidates}
    combined.to_parquet(RESULTS / "combined_predictions.parquet", index=False)

    # Paired spatial-block uncertainty versus coverage-consistent V1.
    bootstrap = _bootstrapped_differences(test_all, locations)

    # Selection rule declared before the test: 1% validation deviance, MAE,
    # high-crash and geographic safeguards.
    reference_deviance = validation_metrics["coverage_v1"]["poisson_deviance"]
    reference_high_mae = test_metrics["coverage_v1"]["subgroup_errors"]["high_actual_top_decile"]["mae"]
    selection = {}
    for name in ("coverage_tuned", "nb", "hybrid"):
        improvement = 100 * (reference_deviance - validation_metrics[name]["poisson_deviance"]) / reference_deviance
        high_change = 100 * (test_metrics[name]["subgroup_errors"]["high_actual_top_decile"]["mae"] - reference_high_mae) / max(reference_high_mae, 1e-9)
        geo_change = 100 * (geo_metrics[name]["poisson_deviance"] - geo_metrics["coverage_v1"]["poisson_deviance"]) / geo_metrics["coverage_v1"]["poisson_deviance"]
        selection[name] = {
            "validation_deviance_improvement_pct": float(improvement),
            "high_crash_mae_change_pct": float(high_change),
            "geographic_deviance_change_pct": float(geo_change),
            "meets_1pct": bool(improvement >= 1.0),
            "meets_safeguards": bool(high_change <= 2.0 and geo_change <= 2.0),
        }

    summary = {
        "dataset": data_audit,
        "baseline": {
            "deployed_v1_original": json.loads((RESULTS.parent / "ml/v1/metrics.json").read_text())["test"]["test_metrics"],
            "coverage_v1": v1_metrics["test"],
            "coverage_tuned": tuned_metrics["test"],
        },
        "temporal_validation": validation_metrics,
        "test": test_metrics,
        "geographic": geo_metrics,
        "combined": combined_metrics,
        "hybrid": {
            "weight_lightgbm": weight,
            "weight_negative_binomial": 1 - weight,
            "oof_error_correlation": float(
                np.corrcoef(
                    calibration["y_all"] - calibration["lgbm"],
                    calibration["y_all"] - calibration["nb"],
                )[0, 1]
            ),
        },
        "intervals": {name: value["metrics"] for name, value in intervals.items()},
        "bootstrap": bootstrap,
        "selection": selection,
        "runtimes": {
            "nb_fit_and_predict_seconds": nb_seconds,
            "lightgbm_existing_training_seconds": tuned_metrics.get("training_seconds"),
            "nb_geographic_alpha": geo_alpha,
            "nb_combined_alpha": combined_alpha,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
        },
        "final_recommendation": "retain_v1" if not any(v["meets_1pct"] and v["meets_safeguards"] for v in selection.values()) else "review_best_candidate",
        "nb_vif": nb_test_info["vif"],
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2, default=str)[:4000])
