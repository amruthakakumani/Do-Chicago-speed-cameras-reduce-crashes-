# Feature availability ledger

**Forecast origin:** end of calendar month `t−1`; target is month `t`.
**Forecast issue date:** first day of month `t` (before the month is observed).
**Rule:** every feature must be known at the origin, or be a deterministic
calendar/static attribute. No target-month crash, weather, exposure or camera
outcome is permitted.

---

## 1. Camera-zone model

| Group | Features | When known | Evidence / construction |
|---|---|---|---|
| Calendar | month index, year, month of year, sine/cosine, quarter, days in month, holiday count, COVID flag | Deterministic | Calendar from `month`; holidays from US calendar |
| Crash history | lags 1, 2, 3, 6, 12; rolling means/SDs 3/6/12; expanding mean/SD; zero share; injury/severe shares; trend | Prior months only | `shift(1)` or higher; `add_features`; leakage tests |
| Static geography | easting/northing, lat/lon, distance to centre, distance to nearest camera, cameras within 1 km, cluster, camera-site flag | Static | Projected coordinates; deterministic |
| Camera operation | operational flag, camera age, operational cameras within 1 km | Known at forecast origin from go-live date | `cameras.parquet`; observational only |
| Weather | temperature/precipitation/snow lag 1; 3-month means | Prior months only | Open-Meteo archive; lagged in `lagged_weather_features` |
| Road network | density/class/intersection features | Static built environment | TIGER 2024; 2017 overlap check |
| Neighbourhood | neighbour lag 1; 12-month mean/SD; zero/injury shares; shock | Prior months only | Gaussian kernel; prior-month matrices |
| Citywide | citywide lag 1; 3/12-month means; trend; injury share; location/city ratio; shock | Prior months only | Citywide aggregates shifted by at least one month |
| Long history | lag 24/36; same-month 3-year mean; seasonal ratio; CV; 6-month trend; months since crash/injury; ever crashed | Prior months only | Shifted history; expanding means |

**Excluded:** current/future AADT; target-month weather; target-month crashes;
any target-derived column. AADT is excluded because the current extract begins
2022-07 and has no historical coverage for 2017–2022.

---

## 2. Road-segment model

| Group | Features | When known |
|---|---|---|
| History | lag 1/2/3/6/12; rolling 3/6/12; expanding; zero share | Prior segment-months |
| Calendar | month index, month of year, sine/cosine, days, COVID | Deterministic |
| Static | length, major-road flag, road class, node degrees, intersection-edge flag, centroid coordinates | Static TIGER attributes |
| Excluded | AADT, target-month crashes, camera treatment | No valid historical exposure; predictive task only |

The road model is a **separate target**. It does not estimate a camera effect and
its MAE cannot be compared with the camera-zone MAE.

---

## 3. Leakage checks

- `validate_no_leakage` recomputes lags/rolling means and perturbs future
  outcomes; the build fails on mismatch.
- V2 neighbourhood features are recomputed from prior-month matrices.
- Rolling-origin validation keeps months ordered; test rows are never used for
  tuning.
- Conformal intervals are calibrated on out-of-fold rolling predictions, not
  test residuals.
- The test period 2025-09…2026-08 was already inspected in V1; all V2 and
  coverage results on it are retrospective.

---

## 4. Source-to-feature audit

| Source | Feature use | Time validity |
|---|---|---|
| Chicago crashes `85ca-t3if` | Target and lagged history | Target only at `t`; history ≤`t−1` |
| Camera locations `4i42-qv3h` | Camera status, age, geography | Go-live known; no future outcome |
| Open-Meteo archive | Lagged weather, climate normals | Prior month/long-run only |
| TIGER 2024 EDGES | Road density/class/intersections | Static; 2017 overlap documented |
| AADT `gc7y-n4xa` | **Not used** | Starts 2022-07; no valid 2017–2022 merge |
| Holidays | Calendar counts | Deterministic |

No values were imputed from future observations. Missing historical exposure is
documented, not manufactured.
