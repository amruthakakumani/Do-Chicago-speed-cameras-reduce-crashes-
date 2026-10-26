# Geographic design comparison

**Purpose:** compare the historical 250 m camera-zone design with the new
prediction-oriented road-segment design. They answer different questions and
cannot be ranked by raw MAE.

---

## 1. Conceptual comparison

| Dimension | Camera-zone design | Road-segment design |
|---|---|---|
| Unit | 250 m catchment around a camera or comparison site | TIGER/Line road segment clipped to the city |
| Target | Reported crashes in the catchment-month | Crashes assigned to the segment-month |
| Population | Crash-dense camera and matched comparison locations | All major segments + seeded local sample |
| Treatment | Camera activation enters as an observational feature in forecasting; causal design is separate | No treatment model; predictive only |
| Exposure | None | None (historical AADT unavailable) |
| Zero interpretation | Observed zero catchment-months | Observed zero segment-months |
| Main use | Causal design and camera-zone forecasting | Road-level forecasting experiment and population-coverage contrast |
| Main weakness | Crash-dense, non-random location set; overlapping catchments | Assignment ambiguity; sparse road network history; no exposure |

The camera-zone task is the only one connected to the causal analysis. The road
task is deliberately separate: assigning crashes to segments and forecasting
segment counts does not identify a camera effect.

---

## 2. Empirical comparison

| Statistic | Camera-zone | Road-segment |
|---|---:|---:|
| Locations | 1,326 | 73,088 total; 15,126 modelling sample |
| Camera locations | 129 | — |
| Comparison/zero-crash locations | 1,197 controls | 2,440 sampled zero-crash segments |
| Months | 104 (2018-01…2026-08) | 108 (2017-09…2026-08) |
| Rows | 137,904 | 1,633,608 |
| Zero share | 10.7% | 89.4% |
| Mean crashes/month | 3.88 | 0.141 |
| Test rows | 15,912 | 181,512 |
| Coverage-tuned model MAE | 1.577 | 0.182 |
| Coverage-tuned model deviance | 1.216 | 0.395 |
| Test R² | 0.660 | 0.351 |
| 90% interval coverage | 90.4% | 90.3% |
| Highest-risk decile interval coverage | 84.9% | **60.2%** |
| Hotspot precision (top 10%, training threshold) | 0.72 | 0.48 |
| Hotspot recall | 0.55 | 0.46 |

**These MAE values are not comparable.** The road task has a different unit,
different population and a 90% zero share. A lower raw MAE is not evidence of a
better model.

---

## 3. Road network and assignment

- 73,088 TIGER 2024 road segments; 3,126 major (primary/secondary) and 69,962
  local.
- 6,659 km of road; median segment length 54.9 m.
- 98.9% of the 2024 Cook County city segments existed as 2017 TIGER TLIDs.
  However, 6,633 older segments are absent and 795 are new. The 2024 network is
  not treated as a perfect historical map.
- 982,255 in-city crashes in 2017-09…2026-08; 975,033 assigned within 30 m
  (99.26%).
- 7,222 unassigned; 331,234 (34.0%) ambiguous because a second segment lies
  within 5 m; 2,214 (0.23%) are >20 m from the selected segment.
- Every crash is assigned at most once. Ambiguous assignments are retained and
  flagged; they are not silently duplicated.

The ambiguity rate is a substantive limitation. Chicago has divided roads,
dense intersections and overlapping TIGER edges. A future version should
aggregate to named corridors or use a higher-quality local street network with
block-face identifiers.

---

## 4. Modelling sample

The full 73k-segment panel is too large for the available 8 GB environment, so
the modelling sample is:

- **all 3,126 major segments**;
- **12,000 local segments** drawn with seed 20261024 from all 69,962 local
  segments, not only those that ever crashed.

This yields 15,126 segments (20.7% of all segments), 1,633,608 segment-months,
2,440 zero-crash segments, and 23.6% of all assigned crashes. It is a
transparent stratified sample, not a citywide census. The full-network counts
and segment summaries are retained for population accounting.

---

## 5. Temporal and exposure constraints

- Camera-zone analysis is coverage-consistent from 2017-09, with target months
  starting 2018-01 so short lags exist.
- Road model uses 2017-09 counts and target months from 2018-01.
- The road model does **not** use AADT: only 2.5k segments and no pre-2022 data.
- The camera-zone model also does not use AADT. The count target is retained.

---

## 6. Interpretation

The camera-zone design supports a causal question because treatment timing and
comparison catchments are explicit. The road-segment design supports a
prediction question about a different population, but its assignment ambiguity
and lack of exposure make it a research experiment, not a replacement for the
camera-zone forecast or the causal estimate.

A road-segment prediction should never be presented as a camera-specific causal
effect. A camera-zone group estimate should never be presented as a road-segment
forecast. The dashboard keeps these sections separate.

---

## 7. Artifacts

- `data/processed/road_segments.parquet`
- `data/processed/crash_road_assignments.parquet`
- `data/processed/road_segment_month_counts.parquet`
- `data/processed/road_model_panel_features.parquet`
- `results/bottom_up/roads/segment_inventory.json`
- `results/bottom_up/roads/assignment_summary.json`
- `results/bottom_up/roads/sample_summary.json`
- `results/bottom_up/roads/vintage_comparison.json`
- `results/bottom_up/road_model/road_metrics.json`
- `results/bottom_up/figures/road_assignment.png`
- `results/bottom_up/figures/road_interval_coverage.png`
- `results/bottom_up/figures/road_sample_map.png`
