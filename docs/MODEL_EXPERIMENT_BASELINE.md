# Model experiment baseline

**Branch:** `experiment/statistical-hybrid-models`
**Scope:** crash forecasting only. Causal results and publication files are untouched.

---

## 1. Dataset fingerprint and sample

All candidates use the coverage-consistent camera-zone dataset built for the
bottom-up reconstruction:

| Field | Value |
|---|---|
| File | `data/processed/ml_coverage_enriched.parquet` |
| SHA-256 | `978946729dcf7b5ec0f9df1b1fcb5134cfcebbc1a638e789ffaee59ab43cec8c` |
| Rows | 137,904 location-months |
| Locations | 1,326 |
| Camera catchments | 129 |
| Comparison sites | 1,197 |
| Target | Reported crashes in 250 m during a calendar month |
| Target window | 2018-01 … 2026-08 (104 months) |
| Test window | 2025-09 … 2026-08 (15,912 rows) |
| Training rows | 121,992 |
| Minimum location spacing | 500 m |
| Zero-crash share | 10.67% |
| Mean / variance | 3.881 / 13.341 |
| Variance-to-mean | 3.44 |
| Maximum monthly count | 53 |

The dataset starts in 2017-09 for history and 2018-01 for targets because
citywide reporting begins in September 2017. The camera-zone set is crash-dense
and non-random; it is not a citywide road census.

## 2. Baselines and references

The deployed V1 model was trained on the earlier 2017-01 target. Its original
retrospective test metrics are:

| Metric | Deployed V1, original sample |
|---|---:|
| MAE | 1.5683 |
| RMSE | 2.0931 |
| Poisson deviance | 1.2233 |
| R² | 0.6611 |
| Rows | 16,260 |

For fair candidate comparisons, the V1 reference is the **coverage-consistent
V1** trained on the same 137,904 rows and evaluated on the same 15,912 test rows:

| Metric | Coverage-consistent V1 |
|---|---:|
| MAE | 1.5799 |
| Poisson deviance | 1.2201 |
| R² | 0.6586 |
| Rows | 15,912 |

The optimized LightGBM candidate is the coverage-consistent tuned model from the
completed reconstruction: 78 features, 8 Optuna trials, validation deviance
1.21647, test deviance 1.21625, test MAE 1.57745.

## 3. Feature groups

LightGBM uses the full validated feature groups: history lags/rolling means,
calendar, weather, static geography, camera operation, TIGER road network,
neighbourhood history, citywide regime and long-history features. The compact
model retains 78 features.

The Negative Binomial uses 26 historically available numeric features plus
cluster indicators: short and long history, zero share, trend, seasonal ratio,
injury share, calendar, lagged weather, camera operation, static geography, road
density/class and neighbourhood history. It excludes target-derived columns and
does not use camera treatment as a causal variable.

## 4. Missingness and imputation

Missingness is confined to early history windows:

| Feature | Missing share |
|---|---:|
| `seasonal_ratio` | 7.69% |
| `roll_mean_12`, `roll_std_12`, `trend_12` | 3.85% |
| `zero_share_12`, `injury_share_12` | ~1.9% |
| neighbour history | 0.96% |
| all others | 0% |

The NB design imputes training medians; imputation is fit on training data only.
LightGBM handles missing values natively. Missingness is not treated as zero.

## 5. Validation protocols

- **Temporal:** three expanding six-month folds: 2024-01…06, 2024-07…12,
  2025-01…06; OOF rows 23,868.
- **Locked test:** 2025-09…2026-08, 15,912 rows; not used for tuning or weighting.
- **Geographic:** 34 held-out 2 km blocks, 290 locations, 1 km buffer; 3,480
  validation rows, 2024-09…2025-08.
- **Geographic + later period:** the same held-out blocks evaluated retrospectively
  on 2025-09…2026-08; clearly labelled retrospective.

All candidate metrics are computed on the same rows within each protocol.

## 6. Causal integrity

The experiment does not alter the published causal estimates, does not use
camera treatment as a causal variable, and does not interpret NB coefficients or
SHAP values as causal effects.

Artifacts: `results/statistical_hybrid/data_audit.json`,
`results/statistical_hybrid/feature_missing.csv`.
