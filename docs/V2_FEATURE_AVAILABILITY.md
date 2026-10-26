# V2 feature availability report

Every feature is checked against the forecast issue date. Rows are
location-months; 157,180 rows, 1,355 locations, 84 features.

## 1. Availability matrix

| Feature group | Examples | Source | Available at issue date? | Missingness | Leakage control |
|---|---|---|---|---|---|
| Calendar | year, month, sin/cos, quarter, days, holidays | deterministic | Yes | 0% | calendar only |
| Crash lags | lag 1/2/3/6/12/24/36 | Chicago crashes | Yes (prior months) | lag 24: 6.9%; lag 36: 17.2% | grouped shift |
| Rolling history | mean/std 3/6/12, expanding, zero share | Chicago crashes | Yes (prior months) | <0.1% | shift before rolling |
| Severity history | injury/severe share 12m | Chicago crashes | Yes (prior months) | <0.1% | shifted sums |
| Seasonal history | same-month 3-year mean, seasonal ratio | Chicago crashes | Yes (prior years) | 17.2% for 3-year mean (early years) | lags 12/24/36 |
| Crash recency | months since last crash/injury | Chicago crashes | Yes (prior months) | crash 0.03%; injury 3.8% | last-hit before t |
| Citywide regime | city lag/mean/trend, injury share, location/city ratio | Chicago crashes | Yes (prior months) | <0.1% | shifted aggregates |
| Neighbourhood | weighted lag/12m mean/std/zero/injury/share | Chicago crashes + UTM distances | Yes (prior months) | <0.1% | Gaussian weights, prior slice |
| Weather lag | temp/precip/snow lag 1, 3-month mean | Open-Meteo | Yes (prior month) | 0% | shift by 1 |
| Climate normals | temp/precip/snow normals | Open-Meteo | Yes (prior years) | 0% | expanding same-month mean |
| Road density | total/major/local road density 500m/1km | TIGER 2024 EDGES | Yes (static) | 0% | static join |
| Intersections | count/density, major distance | TIGER 2024 EDGES | Yes (static) | 0% | topology nodes |
| Road distance | distance to nearest major road/intersection | TIGER 2024 EDGES | Yes (static) | 0% | static |
| Geography | easting/northing, lat/lon, centre/camera distance, cluster, site type | UTM + cameras | Yes (static) | 0% | static |
| Camera context | operational, age, cameras within 1 km | camera go-live | Yes (known schedule) | 0% | observational only |
| Traffic volume | AADT | Chicago portal | **No** | — | excluded (future exposure) |
| Speed limit | posted limit | crashes only | Partial | high | not used as a site feature |

## 2. Excluded or unavailable data

| Data | Status | Reason |
|---|---|---|
| Chicago AADT `gc7y-n4xa` | Actual extract starts 2022-07; 2,525 segments | No historical 2017–2021 denominator; excluded rather than backfilled |
| AADT 2006 historical study | Too old / sparse | Not representative of 2017–2026 |
| Chicago Street Center Lines `6imu-meau` | Map layer, no tabular attributes | Unusable for joins |
| OSM road network | Available but heavier/less stable | TIGER EDGES chosen for reproducibility |
| Speed limits by segment | No current tabular dataset found | Not fabricated |
| Future weather | Not used | Climate normals for recursive paths |
| Future crash counts | Not used | Recursive lags use predictions |

## 3. Missingness policy

- Missing values are **not** imputed. LightGBM handles missing splits.
- `lag_24`/`lag_36` are genuinely unavailable for the first years of the target
  window; missingness is higher in 2017–2019 and near zero by the test period.
- All-missing features are dropped from the schema rather than filled.
- The feature schema is saved in `models/v2_model_metadata.json`; inference
  validates names and order.

## 4. Road-data provenance

`data/raw/tiger/provenance.json` records the two TIGER EDGES archives, SHA-256,
vintage, download time and licence. 132,307 road edges and 46,340 junctions
(3,355 major) fall inside the Chicago bounds used for the 1,355 locations.
