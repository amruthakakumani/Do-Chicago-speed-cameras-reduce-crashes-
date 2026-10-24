# Data dictionary

All source data come from the City of Chicago Data Portal (Socrata). Raw responses are preserved
verbatim as gzip CSV in `data/raw/`, together with `provenance.json` (URL, query, timestamps, row
counts, SHA-256). Processed tables are Parquet in `data/processed/`.

---

## 1. Sources

| Logical table | Dataset ID | Portal URL | Rows (2026-10-24) | Used for |
|---|---|---|---|---|
| `cameras` | `4i42-qv3h` | https://data.cityofchicago.org/d/4i42-qv3h | 209 | Camera coordinates and go-live dates |
| `crashes` | `85ca-t3if` | https://data.cityofchicago.org/d/85ca-t3if | 1,106,882 | Crash outcome |
| `violations` | `hhkd-xvj4` | https://data.cityofchicago.org/d/hhkd-xvj4 | 502,120 daily tickets | Mechanism check |

The crash download selects 12 columns and filters `crash_date >= 2013-03-01`; the violations table
is aggregated to camera-month in `cleaning.py`. Raw files are never modified.

---

## 2. `cameras` (processed)

One row per distinct physical camera location.

| Column | Type | Source / derivation | Notes |
|---|---|---|---|
| `id` | string | `id` | Portal row id |
| `location_id` | string | `location_id` | Camera identifier (e.g. `CHI096`); joins `violations.camera_id` |
| `address` | string | `address` | Street address |
| `first_approach`, `second_approach` | string | same | Enforcement direction(s) |
| `go_live` | datetime | `go_live_date` | Month of enforcement onset; the portal has no separate installation date |
| `lat`, `lon` | float | `latitude`, `longitude` | WGS84 |
| `x`, `y` | float | UTM 16N (EPSG:26916) | Metres; used for all distances |
| `go_month` | datetime | `go_live` floored to month | Aligns with the crash month grid |
| `go_live_year` | int | derived | For the go-live distribution |
| `camera_uid` | string | `location_id`, de-duplicated | Stable identifier used in stacks |
| `go_idx` | int | month index of `go_month` + treatment offset | Eligibility and panel alignment |
| `eligible_window` | bool | `go_idx ≥ pre_months` and `go_idx + post_months ≤ T` | 33 of 209 in the primary window |

**Cleaning:** rows missing coordinates or go-live dates are dropped (none after the portal schema);
exact duplicate coordinates are collapsed keeping the earliest go-live (none in the current
snapshot); coordinates outside the Chicago bounding box are dropped (none).

**Quality:** `go_live` distribution (2013: 39, 2014: 78, 2015: 6, 2018: 12, 2021: 1, 2023: 20,
2024: 3, 2025: 48, 2026: 2). All 209 coordinates are distinct.

---

## 3. `crashes` (processed)

One row per reported crash.

| Column | Type | Source / derivation | Notes |
|---|---|---|---|
| `crash_record_id` | string | same | Unique; duplicate ids are dropped (none in the current snapshot) |
| `date` | datetime | `crash_date` | Timestamp of the crash |
| `month` | datetime | `date` floored to month | Analysis time unit |
| `month_index` | int | index into the observed month grid | Position in the site × month matrix |
| `lat`, `lon` | float | same | WGS84 |
| `x`, `y` | float | UTM 16N | Metres |
| `injuries_total` | float | same | Total injuries |
| `injuries_fatal` | float | same | Fatal injuries |
| `injuries_incapacitating` | float | same | Incapacitating injuries |
| `injury_crash` | int8 | `injuries_total > 0` | Injury outcome |
| `severe_crash` | int8 | fatal > 0 or incapacitating > 0 | Severe outcome |
| `posted_speed_limit` | float | same; 0 → missing | 0 means "unknown" in this dataset |
| `crash_type` | string | same | e.g. injury/tow, property damage |
| `first_crash_type` | string | same | e.g. pedestrian, fixed object |
| `num_units` | float | same | Vehicles/units involved |
| `report_type` | string | same | On-scene vs desk report; relevant to the 2017 reporting change |

**Cleaning and exclusions (current snapshot):**

| Exclusion | Rows |
|---|---|
| Missing crash date | 0 |
| Missing coordinates | 8,673 |
| Coordinates outside the Chicago bbox | 75 |
| Outside the analysis month window (incl. reporting lag) | 16,660 |
| Months before reliable coverage | 476 |
| Duplicate crash ids | 0 |
| **Kept** | **1,080,998** |

**Coverage.** The portal table is effectively complete from 2015‑09. 2015‑01…08 contains only 476
records and 2013–14 contains 8; those months are treated as missing and excluded by
`cleaning.clean_crashes`. The last two calendar months are excluded because reports are still being
entered. Observed window: **2015‑09 … 2026‑08 (132 months)**, subject to the runtime snapshot.

**Zeros.** A site-month with no crash is an observed zero for months inside the window: the portal
table is a census of reported crashes, not a sample. Zeros outside the window are missing data and
are never used.

---

## 4. `violations` (processed)

One row per camera-month (sum of daily tickets).

| Column | Type | Source / derivation | Notes |
|---|---|---|---|
| `camera_id` | string | `camera_id` | Joins `cameras.location_id` |
| `month` | datetime | `violation_date` floored to month | |
| `violations` | float | sum of daily `violations` | Enforcement output, **not** speeding behaviour |
| `observed_days` | int | distinct ticket dates in the month | Data-quality aid |

---

## 5. Design tables (derived, in `results/`)

### `analysis_panel.parquet`

| Column | Meaning |
|---|---|
| `stack` | Camera-specific experiment (`S{camera_uid}`) |
| `camera_row`, `camera_uid` | Camera identity |
| `unit` | Site identifier (`T…` treated, `C…` comparison) |
| `treated` | 1 for the camera catchment |
| `rel` | Event month, −24…+35 (0 = go-live month) |
| `calendar_month` | Calendar month aligned to `rel` |
| `y` | Crashes in that site-month |
| `post` | `rel ≥ 0` |
| `tp` | `treated × post` |
| `quarter` | `floor(rel / 3)` |

### `sites.parquet`

Camera and comparison-site locations with `role`, `lon`, `lat`, `x`, `y`, `go_month`.

### Result tables

| File | Contents |
|---|---|
| `tables/main_estimates.csv` | Naive, original baseline, FE OLS, conditional Poisson, cohort 2×2, replication |
| `tables/replication_estimates.csv` | Original-sample replication with corrected projection |
| `tables/event_study.csv` | Quarterly coefficients and CIs |
| `tables/balance.csv` | Pre-period balance and differential pre-trend |
| `tables/subgroups.csv` | Region / baseline-risk / era estimates with BH-adjusted p-values |
| `tables/severity.csv` | Injury and severe/fatal outcomes |
| `tables/sensitivity.csv` | All robustness and falsification specifications |
| `tables/synthetic_control.csv` | SC fit, weights, pseudo p-values |
| `tables/sc_leave_one_out.csv` | Donor influence |
| `tables/sc_in_time_placebo.csv` | Fake treatment dates |
| `tables/placebo_geography.csv` | Geographic placebo draws |
| `tables/violations_curve.csv` | Mean violations by months since first ticket |
| `estimates.json` | Machine-readable bundle consumed by the dashboard |

---

## 6. Known data limitations

1. **Geocoding.** Crash coordinates are block-level; a 250 m circle may include adjacent streets and
   exclude the far side of an intersection. Radius sensitivity (150/400 m) partially addresses this.
2. **Reporting change.** 2016 has roughly one third of the 2018 crash volume. Reporting intensity
   changed, and if it changed differentially across neighbourhoods the DiD is biased. The 2019+
   cohort sensitivity and the era split address this only partially.
3. **Camera operation.** The portal does not report downtime, warning periods, or enforcement hours.
4. **Violations.** Ticket counts reflect enforcement policy and camera uptime, not speeding
   behaviour or exposure.
5. **No exposure denominator.** Crashes per catchment-month cannot be converted to crashes per VMT.
6. **Camera siting.** Cameras are placed where crashes were high; regression to the mean is a
   permanent threat, mitigated but not removed by matching on pre-period level and slope.
