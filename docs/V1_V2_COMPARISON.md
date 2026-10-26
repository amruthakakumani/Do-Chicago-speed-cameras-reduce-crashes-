# V1 vs V2 comparison

**All V2 test numbers are retrospective**: the 2025-09 … 2026-08 period was
already examined in V1. The deployment decision uses only the pre-test temporal
folds specified in `docs/V2_BASELINE_AUDIT.md`.

---

## 1. Validation (pre-test, used for selection)

Three expanding six-month folds, 2024-03 … 2025-08.

| Metric | V1 | V2 | Change |
|---|---:|---:|---:|
| Mean Poisson deviance | 1.2227 | **1.2197** | −0.24% |
| Mean MAE | **1.5930** | 1.5933 | +0.02% |
| Fold 1 deviance | 1.2201 | **1.2125** | |
| Fold 2 deviance | **1.2260** | 1.2276 | |
| Fold 3 deviance | **1.2220** | 1.2191 | |

**Pre-declared rule:** ≥1% deviance improvement and ≤2% MAE regression.
**Result: not met. V1 remains the deployed model.**

## 2. Retrospective test comparison (same 16,260 rows)

| Metric | V1 | V2 | Change |
|---|---:|---:|---:|
| MAE | 1.5683 | **1.5642** | −0.26% |
| RMSE | 2.0931 | **2.0888** | −0.20% |
| Poisson deviance | 1.2233 | **1.2181** | −0.43% |
| Bias | +0.163 | **+0.144** | −0.019 |
| Relative bias | +4.33% | **+3.82%** | |
| R² | 0.6611 | **0.6624** | +0.0013 |
| Zero-crash MAE | 1.718 | **1.698** | −0.020 |
| High-crash (top decile) MAE | **2.897** | 2.922 | +0.025 |

V2 is slightly better overall and on zero-crash months, but slightly worse on the
highest-crash decile. The differences are small relative to uncertainty.

## 3. Geographic holdout

34 held-out 2 km blocks (294 locations: 24 camera, 270 comparison); training
locations within 1 km of any held-out location are removed.

| Evaluation | V1 features | V2 features |
|---|---:|---:|
| Geographic-only deviance | — | **1.2126** |
| Geographic-only MAE | — | **1.6112** |
| Combined temporal+geographic deviance | 1.2328 | **1.2189** |
| Combined temporal+geographic MAE | 1.6227 | **1.6147** |
| Combined bias | −3.74% | **−1.72%** |

The feature ablation uses the same training rows and the same evaluation rows;
V1 hyperparameters are used for the V1 feature set and V2 hyperparameters for the
V2 set. The new road/neighbourhood/citywide features improve geographic
generalization by about 1.1% in deviance and reduce bias.

## 4. Recursive 12-month diagnostic

Starting 2025-08, predicting 2025-09 … 2026-08 without observing intervening
crashes.

| Metric | V1 | V2 |
|---|---:|---:|
| MAE | 1.6452 | **1.6231** |
| RMSE | 2.1764 | **2.1548** |
| Poisson deviance | 1.2863 | **1.2663** |
| Bias | +0.448 (+11.9%) | **+0.382 (+10.1%)** |

Both models over-predict recursively; V2 is modestly better. Recursive errors are
not comparable to the one-step headline.

## 5. Prediction intervals

The deployed V1 model receives scale-normalised split-conformal intervals
calibrated on out-of-fold rolling predictions.

| Property | Value |
|---|---|
| Nominal coverage | 90% |
| Empirical coverage | **90.6%** |
| Mean width | 5.86 crashes |
| Camera-site coverage | 90.1% |
| Comparison-site coverage | 90.7% |
| Lowest risk decile coverage | 96.6% |
| Highest risk decile coverage | **84.8%** |
| Recursive-path coverage (one-step quantile applied) | 90.2% |

Conditional coverage at the highest-risk decile is below nominal; the intervals
are labelled approximate. See `docs/PREDICTION_INTERVALS.md`.

## 6. Coverage-consistent comparison

The V1/V2 target starts in 2017-01, eight months before citywide reporting. On
15,912 common coverage-consistent test rows:

| Model | MAE | Deviance |
|---|---:|---:|
| Deployed V1 | 1.5860 | 1.2242 |
| V2 candidate | 1.5820 | 1.2193 |
| Coverage V1 | 1.5799 | 1.2201 |
| Coverage full | 1.5786 | 1.2175 |
| Coverage tuned | 1.5775 | 1.2162 |

The coverage-tuned model improves deviance by 0.65% versus V1, below the 1%
promotion rule, and is slightly worse than coverage V1 on geographic holdout.
V1 remains deployed. The populations and target windows differ; raw MAE is not
a direct rank across reporting regimes.

## 7. Latency and memory

| Measure | V1 | V2 candidate |
|---|---:|---:|
| Inference, 157,180 rows | 1.33 s | 4.86 s |
| Peak RSS | 1.39 GB | 1.61 GB |
| Final refit | 13.2 s | 42.6 s |
| Optuna budget | 40 trials / 914 s | 11 trials / 922 s |

V2 is slower (84 features, more trees) and did not deliver enough validation
accuracy to justify replacing V1.

## 8. Decision

**Deployed model: V1.** V2 remains a research candidate with better geographic
and recursive accuracy and a richer interval/road/neighbourhood diagnostic
suite. The pre-declared 1% validation threshold was not met; changing the
threshold after seeing results would be goalpost-moving.

The dashboard therefore shows V1 point forecasts with V1 conformal intervals and
retains V2 candidate artifacts for inspection and future work.
