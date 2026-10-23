# Do-Chicago-speed-cameras-reduce-crashes-
Compare crash rates near cameras against matched control areas using difference-in-differences and synthetic control.
A causal inference study using difference-in-differences, an event-study decay test, and synthetic control on public Chicago Data Portal data.

> **Status:** replace every `[bracketed]` value below with your own results after running the notebook on the real data.

## Question

Speed cameras are installed at dangerous locations, so a simple before/after comparison is misleading: dangerous spots tend to improve on their own, and city-wide events (for example COVID-19) change crashes everywhere at once. This project estimates the *causal* effect of cameras on crashes by comparing the change near cameras with the change at matched locations that never had a camera.

## Key findings

| Method | Estimated effect on crashes within 250 m | Uncertainty |
|---|---|---|
| Difference-in-differences (Poisson, matched controls) | `[X]%` | 95% CI `[lo, hi]` |
| Synthetic control (largest go-live cohort) | `[Y]%` | pseudo p-value `[p]` |
| Decay (change in effect per year) | `[+/- Z]` log-points | p = `[p]` |

- The effect `[held steady / faded to roughly zero after N months]`.
- The effect was `[larger / smaller]` in `[region]`.
- Drivers adapted: violations per camera `[fell / stayed flat]` over a camera's life.

## Data

All from the [Chicago Data Portal](https://data.cityofchicago.org):

| Dataset | ID | Used for |
|---|---|---|
| Speed Camera Locations | `4i42-qv3h` | Camera coordinates and go-live dates |
| Traffic Crashes - Crashes | `85ca-t3if` | Outcome: crashes by month and location |
| Speed Camera Violations | `hhkd-xvj4` | Mechanism check: do violations decline over time? |

The crash table is large (about 700 MB as CSV). The notebook expects a filtered copy at `data/crashes.csv` with columns `crash_record_id, crash_date, latitude, longitude, injuries_total, posted_speed_limit`, restricted to 2015 onward. A conversion cell for the portal's CSV export is included in the notebook.

## Method

1. **Outcome.** Count crashes within `NEAR_CAMERA` (250 m) of each camera, per month.
2. **Controls.** Lay a 500 m grid over the city. Keep points farther than `BUFFER_M` (600 m) from every camera (limits spillover) and with enough crashes to be real street areas.
3. **Matching (stacked design).** Treat each camera as its own mini-experiment. For each, find the 5 controls within `MAX_DISTANCE` (1000 m) whose pre-camera crash level and trend were closest. Align time to months since go-live (-24 to +35).
4. **Difference-in-differences.** OLS (crashes per month) and Poisson (percent change), standard errors clustered by camera.
5. **Event study.** Separate effect for each quarter around go-live, to check pre-trends and see whether the effect decays. A formal test uses an effect-by-years-since-go-live term.
6. **Heterogeneity.** K-means splits cameras into four geographic groups; the DiD is re-run within each.
7. **Synthetic control** ([Abadie, Diamond & Hainmueller, 2010](https://doi.org/10.1198/jasa.2009.ap08746)). Fit a weighted blend of no-camera sites to the average of the largest go-live cohort. Fitting is de-meaned, because camera sites are crash hotspots whose level exceeds any convex blend of ordinary sites. Inference is a placebo-in-space test (100 random pseudo-treated groups).
8. **Optional.** Bayesian Negative Binomial DiD in PyMC.

## Repository structure

```
.
|-- chicago_speed_cameras.ipynb   # full analysis, with plain-English notes above each cell
|-- data/                         # downloaded / converted data (not committed)
|   |-- cameras.csv
|   |-- crashes.csv
|   `-- violations_monthly.csv
`-- README.md
```

## How to run

```bash
pip install pandas numpy scipy statsmodels scikit-learn matplotlib requests
# optional: pip install pymc
jupyter notebook chicago_speed_cameras.ipynb
```

Run the cells top to bottom. Camera and violation data download automatically from the portal. Parameters are in the settings cell:

| Parameter | Default | Meaning |
|---|---|---|
| `NEAR_CAMERA` | 250 | Radius (m) defining "near a camera" |
| `MAX_DISTANCE` | 1000 | Controls must be within this distance of the camera they are matched to |
| `BUFFER_M` | 600 | Controls must be farther than this from any camera |
| `PRE_MONTHS` / `POST_MONTHS` | 24 / 36 | Months before / after go-live in the analysis window |
| `N_CONTROLS` | 5 | Matched controls per camera |
| `GRID_M` | 500 | Spacing of candidate control sites |

## Limitations

- **Non-random placement.** Cameras are sited where crashes were high. Matching on pre-trends reduces, but cannot remove, regression to the mean.
- **COVID-19.** Traffic and speeding changed sharply in 2020. Controls absorb city-wide shocks but not uneven impact across areas.
- **Reporting changes.** Chicago's crash-reporting process changed around 2017, which can shift counts.
- **Spillover.** Drivers may slow near a camera and speed up beyond it; the buffer reduces but does not eliminate this.
- **Go-live dates.** Assumes the listed date is when enforcement began; warning periods would blur the before/after split.
- **Camera types.** Park-zone and school-zone cameras are not distinguished.

## Robustness checks to try

- Compare city-wide controls with local controls (`MAX_DISTANCE`).
- Vary the radius (150 m, 400 m).
- Use injury crashes only as the outcome.
- Exclude windows overlapping 2020.

## Reference

Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program. *Journal of the American Statistical Association*, 105(490), 493-505.

## Tools

Python, pandas, NumPy, SciPy, statsmodels, scikit-learn, matplotlib (optional: PyMC)
