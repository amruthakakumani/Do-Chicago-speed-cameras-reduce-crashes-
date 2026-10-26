# Model card — LightGBM V2 crash-count forecaster

## 1. Model details

| Field | Value |
|---|---|
| Name | `lightgbm_poisson_crash_model_v2` |
| Version | 2.0.0 |
| Status | **Research candidate; V1 remains deployed** under the pre-declared rule |
| Type | Gradient-boosted decision trees, LightGBM Poisson objective |
| Framework | LightGBM 4.7 |
| Artifact | `models/v2_lightgbm_crash_model.txt` |
| Metadata | `models/v2_model_metadata.json` |
| Seed | 20261024 |
| Target | Reported crashes in a 250 m catchment during a calendar month |
| Unit | Location-month |
| Licence | No licence declared by the upstream repository |

## 2. Intended use

- One-month-ahead descriptive crash forecasting at crash-dense Chicago
  locations.
- Approximate 90% prediction intervals with calibrated marginal coverage.
- Geographic validation and sensitivity diagnostics.
- A map-based research interface that keeps prediction and causal inference
  separate.

## 3. Out-of-scope use

- **No causal interpretation.** Do not use the model to estimate the effect of
  installing, keeping or removing a camera.
- **No camera-specific causal effect.** The causal estimate is group-level.
- **No enforcement, insurance, policing or resource-allocation decisions**
  without independent validation, governance and legal/equity review.
- **No extrapolation beyond the training date range** or outside Chicago.
- **No interpretation of SHAP values as causal effects.**

## 4. Training data

| Item | Value |
|---|---|
| Crash source | Chicago Data Portal `85ca-t3if`, snapshot 2026-10-24 |
| Locations | 1,355 (129 camera catchments after overlap deduplication + 1,226 comparison sites) |
| Target window | 2017-01 … 2026-08; the first eight target months precede citywide reporting |
| Training rows | 140,920 (through 2025-08) |
| Test rows | 16,260 (2025-09 … 2026-08) |
| Features | 84 |
| Road data | US Census TIGER/Line 2024 EDGES, Cook + DuPage |
| Weather | Open-Meteo lagged observations + expanding climate normals |
| Traffic volume | **Not used** (AADT starts 2022-07; no valid 2017–2022 merge) |
| Missing targets | None; genuine zeros retained |

The location set is crash-dense and non-random; it is not a representative sample
of all Chicago streets. The target window begins eight months before the portal's
September 2017 citywide coverage start. A coverage-consistent model starting in
2018-01 improves validation deviance by 0.48%, below the 1% rule, and does not
improve geographic holdout; it is not deployed.

## 5. Features

V1's 44 features plus:

- **Temporal/regime (15):** lags 24/36, same-month 3-year mean, seasonal ratio,
  CV, six-month trend, shock features, months since last crash/injury,
  ever-crashed, citywide lag/3/12-month counts, city trend, city injury share,
  city shock, location-to-city ratio.
- **Neighbourhood (6):** Gaussian-kernel (750 m, 3 km cutoff) neighbouring crash
  lag/mean/std/zero/injury/share.
- **Road network (13):** TIGER density at 500 m/1 km, major/local density, road
  length, major-road share, segment count, intersection density/count, distance
  to major road and major intersection.
- **Weather normals (3):** expanding same-calendar-month temperature,
  precipitation and snowfall normals for recursive use.

All history features are shifted; all neighbourhood/city features use prior
months only. See `docs/V2_FEATURE_AVAILABILITY.md`.

## 6. Validation and selection

- Three expanding temporal folds, 2024-03 … 2025-08.
- Geographic holdout: 34 of 169 two-kilometre blocks, 294 locations, with a
  1 km training buffer; minimum train-to-holdout distance ≈ 1,005 m.
- Combined temporal + geographic holdout.
- Optuna: 11 trials in 922 s; best validation deviance 1.21935.
- **Pre-declared deployment rule:** V2 replaces V1 only if validation deviance
  improves by ≥1% and validation MAE does not regress by more than 2%.
- V2 improved validation deviance by **0.24%**, so **V1 remains deployed**.

## 7. V2 candidate test performance (retrospective)

| Metric | V1 | V2 candidate |
|---|---:|---:|
| MAE | 1.5683 | **1.5642** |
| RMSE | 2.0931 | **2.0888** |
| Poisson deviance | 1.2233 | **1.2181** |
| Bias | +0.163 | **+0.144** |
| R² | 0.6611 | **0.6624** |
| Zero-crash MAE | 1.718 | **1.698** |
| High-crash MAE | **2.897** | 2.922 |

The old test period has already been examined; these are retrospective
diagnostics, not a new untouched confirmation.

## 8. Geographic and recursive performance

| Evaluation | V1 | V2 |
|---|---:|---:|
| Geographic-only deviance | — | **1.2126** |
| Combined temporal + geographic deviance | 1.2328 (V1 features) | **1.2189** |
| Recursive 12-month MAE | 1.6452 | **1.6231** |
| Recursive 12-month deviance | 1.2863 | **1.2663** |

V2's new features improve geographic and recursive accuracy more clearly than
one-step temporal accuracy.

## 9. Prediction intervals

Scale-normalised split conformal intervals, calibrated on out-of-fold rolling
predictions. 90% empirical coverage is **0.906** with mean width 5.86. Coverage
is 0.901 for camera sites and 0.907 for comparison sites, but ranges from 0.97 to
0.85 across predicted-risk deciles, so conditional coverage is imperfect.
Intervals are approximate, not causal confidence intervals. The coverage-tuned
research artifact reaches 90.4% overall and 84.9% in the highest-risk decile; it
is not deployed. See `docs/PREDICTION_INTERVALS.md`.

## 10. Limitations and risks

1. Prediction is not causation; camera features are confounded by siting.
2. The location sample is crash-dense and non-random.
3. 2024 TIGER geometry is assumed time-invariant for 2017–2025 training.
4. No traffic-volume exposure; AADT excluded to avoid leakage.
5. Recursive forecasts over-predict by roughly 10%; one-step errors are lower.
6. High-crash months remain under-predicted; zero months over-predicted.
7. Intervals under-cover the highest-risk decile (≈85% at nominal 90%).
8. Trees cannot extrapolate beyond the training date range.
9. Reporting and camera-operation regime changes are absorbed by history, not
   modelled explicitly.

## 11. Ethical considerations

Crash data are aggregated to counts at locations; no individual driver data are
used. Speed cameras are a surveillance and enforcement technology. Predictive
mapping can stigmatise neighbourhoods; results should include uncertainty,
denominators and community context. Do not use predictions to justify enforcement
without causal evidence and equity review.

## 12. Reproduction

```bash
python scripts/train_lightgbm_v2.py --trials 40 --timeout 900
python scripts/evaluate_recursive_v2.py
streamlit run dashboard/app.py
```
