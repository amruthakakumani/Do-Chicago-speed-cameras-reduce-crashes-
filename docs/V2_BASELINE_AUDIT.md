# V2 baseline audit (V1 reproduction and scope)

**Date:** 2026-10-25
**Branch:** `feature/lightgbm-v2` (from `feature/lightgbm-crash-prediction`)
**V1 artifacts preserved in:** `models/v1/` and `results/ml/v1/`
**V1 model remains the deployed fallback** until V2 passes the pre-declared
validation criterion in §8.

---

## 1. V1 artifact verification

`python scripts/evaluate_lightgbm.py` re-evaluates the saved V1 model on the
locked test rows and reproduces the published numbers:

| Metric | Published V1 | Re-evaluated |
|---|---:|---:|
| Test MAE | 1.5683 | 1.5683 |
| Test RMSE | 2.0931 | 2.0931 |
| Test Poisson deviance | 1.2233 | 1.2233 |
| Test R² | 0.6611 | 0.6611 |
| Test bias | +0.163 | +0.163 |
| Rows | 16,260 | 16,260 |
| Locations | 1,355 | 1,355 |
| Training rows | 140,920 | 140,920 |
| Test period | 2025-09 … 2026-08 | 2025-09 … 2026-08 |

V1 is reproducible from the saved model and the processed data snapshot.

## 2. V1 pipeline facts

- **Target:** reported crashes within 250 m of a location during a calendar
  month.
- **Locations:** 129 camera catchments (250 m catchments, overlapping camera
  catchments deduplicated) + 1,226 comparison sites = 1,355. Minimum
  nearest-neighbour distance is about **500 m**; camera–control catchments do
  not overlap by construction (controls are >600 m from every camera).
- **Features:** 44. History lags (1, 2, 3, 6, 12), rolling means/std (3, 6, 12),
  expanding mean/std, zero share, injury/severe shares, trend, calendar,
  holidays, lagged weather, static geography, camera operational context.
- **Training:** 2017-01 … 2025-08; 140,920 location-months.
- **Validation:** three expanding six-month folds ending 2025-08.
- **Test:** 2025-09 … 2026-08; 16,260 location-months.
- **Model:** one LightGBM Poisson model (924 rounds); Optuna 40 trials.
- **Causal analysis:** untouched; `results/estimates.json` is unchanged.

## 3. Feature availability and leakage

The V1 build enforces and tests:

1. `groupby(location).shift(k)` for lags, `k ≥ 1`.
2. Rolling statistics computed on `shift(1)`.
3. Injury/severe shares from shifted 12-month sums.
4. Weather lagged by at least one month; the target month's weather is never
   used.
5. No target-derived column in the feature list.
6. Chronological folds and a locked test period; months never shuffled.
7. A mechanical leakage test that perturbs future outcomes and verifies earlier
   features do not move.

The V1 test metrics are therefore **leakage-free for the scenario described in
§4**, but that scenario is not a 12-month deployment forecast.

## 4. Critical distinction: one-step-ahead vs recursive deployment

**V1 reports one-step-ahead forecasts with observed histories.** For every test
month, the lag and rolling features are computed from the *actual* crash counts
of preceding months, including months that fall inside the test period. This is a
"nowcast/one-step" evaluation: at each month, the model is given the true recent
history.

It is **not** a genuine 12-month-ahead deployment. A deployment that issues a
forecast in 2025-08 for 2026-08 cannot observe the intervening crash counts. If
those forecasts are produced recursively, each month's prediction feeds the next
month's lags, and errors compound. V1 has no recursive evaluation.

V2 adds a recursive benchmark and reports both scenarios separately
(`docs/LIGHTGBM_V2_METHODS.md`). One-step and recursive metrics are not
interchangeable and must not be presented as the same headline number.

## 5. What V1 does not do well

| Weakness | Evidence |
|---|---|
| Zero-crash months over-predicted | 1,865 zero-crash test rows, mean prediction 1.72, MAE 1.72 |
| High-crash months under-predicted | top actual decile: mean actual 10.99 vs mean predicted 9.02 |
| No uncertainty | point forecasts only |
| Geographic generalization untested | validation is temporal, not geographic |
| Marginal MAE vs historical mean | V1 MAE 1.568 vs expanding-mean 1.569 |
| Camera status barely used | `camera_operational` gain 0.001% — predictive redundancy, not causal evidence |

## 6. Data-source investigation for V2

| Source | Status | Decision |
|---|---|---|
| Chicago crash data `85ca-t3if` | Available, cached | Continue |
| Open-Meteo archive | Available, cached | Continue; lagged features + normals |
| US Census TIGER/Line 2024 roads (`tl_2024_17031_roads`, `tl_2024_17043_roads`) | Downloaded, SHA-256 recorded | Use for road density, class, distance, intersections |
| Chicago "Street Center Lines" `6imu-meau` | Map layer with no tabular fields | Not usable |
| Chicago "Average Daily Traffic Counts" `gc7y-n4xa` | Current snapshot only (2026); historical 2006 study separate | Exclude: current AADT would leak future exposure into 2017–2025 training; 2006 study is too old/sparse. Documented blocker. |
| Chicago speed limits | No current tabular dataset found | Use crash `posted_speed_limit` diagnostics only |
| OSM/Overpass | Available but heavier and less stable than TIGER | Not used for production; TIGER is public-domain and reproducible |

TIGER provenance (URL, vintage, SHA-256, row counts) is recorded in
`data/raw/tiger/provenance.json`.

## 7. V2 feature plan

- Temporal: months since last crash, same-month prior-year mean, seasonal ratio,
  coefficient of variation, six-month trend.
- Spatial: TIGER road density (500 m and 1 km), major/local road density,
  distance to nearest major road, intersection density, distance to nearest
  major intersection, major-road share.
- Neighbourhood: Gaussian-kernel weighted neighbouring crash lag and 12-month
  mean, and neighbouring injury share (strictly prior months).
- Weather: keep lagged observations for one-step; add static monthly climate
  normals for recursive forecasts.
- No traffic-volume feature (documented blocker).

## 8. Pre-declared V2 selection criterion

V2 replaces V1 as the deployed model only if, on the three **pre-test** rolling
folds (2024-03 … 2025-08), V2's mean Poisson deviance improves on V1's by at
least **1% relative** with the same fold construction, and V2 does not regress
on MAE by more than 2% relative. The decision is made from validation only.

The old test period (2025-09 … 2026-08) has already been examined; any V2 number
on it is a **retrospective diagnostic**, not a new untouched confirmation. A
genuinely untouched confirmation requires a later data snapshot or a new
geographic holdout that did not influence development.

## 9. Reproducibility

V1 is preserved exactly:

```bash
python scripts/evaluate_lightgbm.py   # reproduces V1 test metrics
```

V1 artifacts: `models/v1/lightgbm_crash_model.txt`,
`models/v1/model_metadata.json`, `results/ml/v1/`.
