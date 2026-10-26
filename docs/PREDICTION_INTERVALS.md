# Prediction intervals

Prediction intervals are **approximate split-conformal intervals** around the
deployed LightGBM point forecast. They are not causal confidence intervals and
they do not represent a second prediction model.

---

## 1. Method

1. Fit the point model on the pre-test data.
2. Produce **out-of-fold** predictions from the three expanding rolling
   validation folds (2024-03 … 2025-08). These are the calibration rows; no test
   row and no in-sample fitted value is used.
3. Compute nonconformity scores. Two scores are evaluated:
   - **normalised:** `|y − μ| / sqrt(μ + 1)`, which scales with the predicted
     count;
   - **absolute:** `|y − μ|`, which gives constant-width intervals.
4. Choose the score **before the test period** using time-ordered
   cross-conformal coverage on the calibration months. Count outcomes are
   heteroskedastic; the normalised score is the a-priori choice provided its
   marginal calibration coverage is within 2 percentage points of nominal.
5. Apply the finite-sample-corrected quantile:
   `q = quantile(scores, min(1, (1−α)(1+1/n)))`.
6. Interval: `max(0, μ − q·scale)` to `μ + q·scale`, where `scale = sqrt(μ+1)`
   for the normalised score and `1` for the absolute score. Endpoints are
   non-negative.
7. Report 90% and 50% intervals, empirical coverage, width, and coverage by
   site type, geographic cluster and predicted-risk decile.

## 2. Why normalised intervals

On the test rows, the absolute-residual intervals achieve 90.3% marginal
coverage but collapse at high-risk sites: coverage falls from 99.3% in the lowest
predicted-risk decile to **67.0%** in the highest. The normalised intervals
achieve 90.6% marginal coverage and improve the highest decile to **84.8%**, with
a narrower mean width (5.86 vs 5.99). Absolute intervals are retained as a
reported comparator; the deployed intervals are normalised.

## 3. Verified coverage — deployed V1 model

Calibration: out-of-fold rolling predictions (18 months). Evaluation: the
retrospective test period 2025-09 … 2026-08 (16,260 rows).

| Interval | Nominal | Empirical | Mean width |
|---|---:|---:|---:|
| 90% | 90% | **90.6%** | 5.86 |
| 50% | 50% | **50.5%** | 2.73 |

### By location type

| Group | Rows | 90% coverage | Mean width |
|---|---:|---:|---:|
| Camera | 1,548 | 90.1% | 6.51 |
| Comparison site | 14,712 | 90.7% | 5.79 |

### By predicted-risk decile

| Decile | Coverage | Mean width |
|---:|---:|---:|
| 1 (lowest) | 96.6% | 3.12 |
| 2 | 94.3% | 3.64 |
| 3 | 96.2% | 4.20 |
| 4 | 93.5% | 4.82 |
| 5 | 90.8% | 5.52 |
| 6 | 89.0% | 6.01 |
| 7 | 87.5% | 6.50 |
| 8 | 88.0% | 7.06 |
| 9 | 85.4% | 7.94 |
| 10 (highest) | **84.8%** | 9.82 |

### By geographic cluster

90% coverage ranges from **87.4%** (cluster 1) to **94.2%** (cluster 3). Cluster
widths range from 4.64 to 7.30.

## 4. Recursive path

The same one-step quantile applied to the 12-month recursive forecast yields
**90.2%** empirical coverage with mean width 6.07. This is an approximation:
recursive errors compound and the calibration residuals are one-step residuals.
Recursive interval coverage is reported for transparency, not guaranteed.

## 5. Limitations

- **Exchangeability does not hold.** Crash observations are spatially and
  temporally dependent. The nominal marginal guarantee is asymptotic and does
  not survive distribution shift exactly.
- **Conditional coverage is imperfect.** The highest-risk decile is covered at
  about 85%, below the 90% nominal level. Low-risk deciles are over-covered.
- **One-step calibration.** Intervals for recursive or multi-month forecasts are
  approximate and likely too narrow in some periods.
- **No causal interpretation.** A prediction interval describes forecast
  uncertainty for observed crash counts; it says nothing about what would
  happen under a camera intervention.
- **Data drift.** A new reporting regime, camera policy change or traffic shock
  can invalidate the calibration; intervals should be recalibrated on a rolling
  basis.

## 6. Coverage-consistent and road-task sensitivity

The coverage-consistent camera-zone model (2018-01 targets) attains **90.4%**
marginal coverage and **84.9%** in the highest-risk decile. The separate road
task attains **90.3%** overall but only **60.2%** in its highest-risk decile.
Neither result fixes the conditional-coverage problem. The road task is a
different unit/population and is not comparable to camera-zone MAE.

## 7. Reproduction

```bash
python scripts/train_lightgbm_v2.py --reuse-tuning
python -m chicago_cameras.ml.coverage_dataset
python -m chicago_cameras.ml.reconstruction
python -m chicago_cameras.ml.tune_coverage
python -m chicago_cameras.ml.road_model
```

Intervals are stored in `results/ml_v2/selected_predictions.parquet`
(`lower_50`, `upper_50`, `lower_90`, `upper_90`), with full diagnostics in
`results/ml_v2/metrics.json → conformal`.
