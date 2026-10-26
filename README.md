# Do Speed Cameras Make Chicago Safer?

### A Geospatial Causal Inference and Machine Learning Study

**Saketh Donepudi and Amrutha Kakumani**

*Working paper · October 2026 · Not peer reviewed*

[Paper (PDF)](paper/chicago_speed_cameras_research.pdf) ·
[Paper (Markdown)](paper/chicago_speed_cameras_research.md) ·
[Paper (DOCX)](paper/chicago_speed_cameras_research.docx) ·
[Final publication audit](docs/FINAL_PUBLICATION_AUDIT.md) ·
[Bottom-up reconstruction](paper/bottom_up_reconstruction.md) ·
[Reproducibility](docs/REPRODUCIBILITY.md)

---

## Research question

Do Chicago's automated speed cameras reduce crashes, and how sensitive is that
answer to geographic matching, treatment timing and research design?

Speed cameras are installed where crashes are already high. A simple before/after
comparison therefore confounds enforcement with regression to the mean, local
trends and city-wide shocks such as COVID-19. This project estimates the effect
using nearby matched comparison sites and reports how the answer changes when
defensible design choices change.

A second, deliberately separate analysis asks how accurately one LightGBM model
can forecast monthly crash counts at crash-dense locations, and where that model
fails. Forecasting is **not** used as causal evidence.

---

## Key findings

| Analysis | Estimate | Uncertainty | Sample |
|---|---:|---|---:|
| Original Poisson DiD (reproduced exactly) | **−7.3%** | 95% CI [−13.6%, −0.6%], p = 0.034 | 33 cameras |
| Revised conditional Poisson FE | **−2.6%** | 95% CI [−11.5%, +7.2%], p = 0.59 | 20 cameras |
| Cohort 2×2 | −0.095 crashes/site-month | 95% CI [−0.735, +0.520] | 20 cameras |
| Synthetic control (2023 cohort) | +4.2% | pre-fit R² 0.742; pseudo-p 0.495 | 7 cameras |
| Coverage-consistent 24/36 | **−8.6%** | 95% CI [−16.9%, +0.4%], p = 0.061; placebo p = 0.25 | 15 cameras |
| V1 forecast, held-out test | MAE 1.568 / deviance 1.223 | — | 16,260 location-months |
| V2 candidate, retrospective test | MAE 1.564 / deviance 1.218 | — | same rows |
| Coverage-tuned common-row deviance | 1.216 | 0.65% better than V1; below 1% rule | 15,912 rows |
| 90% conformal interval coverage | **90.6%** | 5.86 mean width; 84.8% in highest-risk decile | 16,260 rows |

**The central result is design sensitivity, not a new national effect size.**
The original −7.3% is reproducible, but it depends on citywide matching that the
documentation did not describe and on retaining overlapping camera catchments.
With local 2,000 m matching and overlap removal, the estimate is −2.6% with a
wide interval. A coverage-consistent design that requires 24 citywide pre-months
and 36 post-months leaves only 15 cameras and gives −8.6% [−16.9%, +0.4%], but it
is a different sample, not a direct replication. That interval includes both a meaningful reduction and a modest
increase.

> **A non-significant estimate is not proof of zero effect.** The data are
> insufficient to resolve the question at policy-relevant precision.

V1 remains the deployed forecasting model. V2 adds 40 features (including real
Census TIGER road-network data) and improves retrospective deviance by only
0.4%; it fails the pre-declared 1% validation promotion rule and is retained as
a research candidate. A coverage-consistent, tuned model improves deviance by
0.65% on common test rows—also below 1%—and is not deployed. Its geographic
holdout is slightly worse than coverage V1.

---

## Methodology

### Causal design

- **Catchment:** 250 m around each camera.
- **Comparison sites:** 500 m lattice points at least 600 m from every camera
  with ≥100 crashes and ≥40% active months.
- **Matching:** five nearby sites per camera on pre-period level and slope,
  evaluated in the camera's own pre-window; primary radius 2,000 m.
- **Overlap removal:** cameras whose catchment overlaps an already-live camera
  are dropped (13 of 33 eligible). Among all 209 cameras, 36.3% of attributed
  crashes fall inside two or more catchments.
- **Window:** −24 to +35 months around go-live. The publication panel starts in
  2015-09, but citywide reporting begins only in 2017-09. A coverage-consistent
  design requires 24 citywide pre-months and leaves 15 cameras.
- **Reporting boundary:** official E-Crash metadata; monthly and district
  diagnostics are in `docs/BOTTOM_UP_AUDIT.md`.
- **Preferred estimators:** conditional Poisson with camera-by-month fixed
  effects; per-camera cohort 2×2 with a camera bootstrap.
- **Inference:** camera-clustered standard errors, wild cluster bootstrap,
  geographic and date placebos, lattice-phase and design sensitivity.

### Forecasting

- **Unit:** location-month; V1/V2 use 1,355 locations (129 camera catchments +
  1,226 comparison sites), 157,180 rows, 2017-01 … 2026-08.
- **Coverage-consistent sensitivity:** 1,326 locations, 137,904 rows,
  2018-01 … 2026-08, trained and evaluated on common test rows.
- **Model:** one LightGBM Poisson model; no competing production model.
- **Features:** history lags/rolling windows, calendar, holidays, lagged weather,
  static geography; V2 adds road density/intersections, neighbourhood crash
  history, citywide regime features and climate normals (84 features total).
- **Validation:** three expanding temporal folds (2024-03 … 2025-08), a
  34-block geographic holdout with a 1 km buffer, and a combined holdout.
- **Road task:** separate 15,126-segment sample; raw MAE is not comparable to
  the camera-zone task and the model is not deployed.
- **Leakage:** all history features are shifted; weather is lagged;
  neighbourhood features use prior months; mechanical checks fail the build on
  any future-outcome leakage.
- **Uncertainty:** scale-normalised split-conformal intervals calibrated on
  out-of-fold predictions.

No traffic-volume feature is used: the public AADT extract begins in 2022-07,
covers 2,525 segments and cannot supply a 2017–2021 denominator. Using it would
leak future exposure or require fabricated backfills. No values were manufactured.

---

## Interactive research map

The Streamlit dashboard keeps a large Chicago map central: cameras, 250 m
treatment buffers, comparison sites, observed crash density, predictions,
predicted risk intensity, 90% interval width, held-out errors, and historical
hotspots, with year/month/camera/region/severity filters and a camera
drill-down. It reads precomputed validated artifacts and never retrains a model.

![Study map](paper/figures/fig1_study_map.png)

*Static preview. Run `streamlit run dashboard/app.py` for the interactive map.
No public deployment is claimed or provided.*

**Causal inference**, **crash forecasting**, and **methods/limitations** are
separate dashboard sections. The group-level camera effect is never shown as a
camera-specific estimate, and prediction intervals are never shown as causal
confidence intervals.

---

## Research paper

- [PDF](paper/chicago_speed_cameras_research.pdf)
- [Markdown](paper/chicago_speed_cameras_research.md)
- [DOCX](paper/chicago_speed_cameras_research.docx)

The paper reports the causal design, the exact replication, the stepwise design
changes, the event study, the synthetic control, the September 2017 reporting
boundary, the coverage-consistent sensitivity, the forecasting system, the
conformal intervals, the separate road task and the limitations. The PDF/DOCX
exports include a cover, author-information page, automatic table of contents,
running headers and page numbers. It is a working paper, not peer-reviewed, and
no institutional endorsement is claimed.

---

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
# macOS only: LightGBM needs OpenMP
brew install libomp
pip install -e ".[dev,dashboard,ml]"

python scripts/download_data.py
python -m chicago_cameras.pipeline --config config/analysis.yaml
python scripts/verify_replication.py

# forecasting
python scripts/train_lightgbm.py --trials 40 --timeout 900
python scripts/train_lightgbm_v2.py --trials 40 --timeout 900
python scripts/evaluate_recursive_v2.py

# bottom-up reconstruction
python -m chicago_cameras.coverage
python -m chicago_cameras.exposure
python -m chicago_cameras.road_network
python -m chicago_cameras.ml.coverage_dataset
python -m chicago_cameras.ml.reconstruction
python -m chicago_cameras.ml.tune_coverage
python -m chicago_cameras.ml.road_model

# formatted paper exports
python scripts/make_cover_graphic.py
python scripts/build_publication_docs.py

# tests and dashboard
python -m pytest -q
streamlit run dashboard/app.py
```

Runtime: data download ≈ 5 min; causal pipeline 4–20 min depending on placebo
count; V2 tuning/evaluation ≈ 25 min. A `--quick` flag reduces draws for
development.

**Verified status:** 63 tests pass; `ruff` clean; original notebook unchanged;
V1 and V2 artifacts preserved. The test period (2025-09 … 2026-08) was
already inspected in V1; V2 and coverage-consistent test numbers are
retrospective.

---

## Data provenance

| Dataset | Source | Use |
|---|---|---|
| Traffic Crashes – Crashes `85ca-t3if` | Chicago Data Portal | Outcome |
| Speed Camera Locations `4i42-qv3h` | Chicago Data Portal | Camera coordinates and go-live |
| Speed Camera Violations `hhkd-xvj4` | Chicago Data Portal | Mechanism check |
| Historical weather | Open-Meteo archive | Lagged weather and climate normals |
| AADT `gc7y-n4xa` | Chicago Data Portal | Audited, **not used**: starts 2022-07 |
| Community areas `igwz-8jzy` | Chicago Data Portal | City boundary for road clipping |
| Road network | US Census TIGER/Line 2024 EDGES (Cook, DuPage) | Road density, class, intersections |
| Holidays | `holidays` US calendar | Calendar features |

Raw files are preserved with URLs, query parameters, timestamps, row counts and
SHA-256 checksums in `data/raw/provenance.json` and
`data/raw/tiger/provenance.json`. Raw and processed data are not committed.

---

## Limitations

- **Non-random siting.** Cameras are placed at high-crash sites; matching
  reduces but cannot remove regression to the mean.
- **Small treated sample.** 20 cameras; wide intervals and fragile
  small-cluster inference.
- **Reporting changes.** Citywide reporting begins in 2017-09; the publication
  panel and V1/V2 target include earlier partial months. The coverage-consistent
  estimate uses a different, 15-camera sample.
- **No traffic exposure.** Crashes per catchment-month is not crashes per
  vehicle-mile. AADT starts in 2022-07 and covers only 1,454 distinct 2024 road
  segments within 50 m.
- **Spillover.** The buffer prevents catchment overlap, not crash displacement.
- **Timing.** Go-live dates are monthly; warning periods and downtime are not
  observed.
- **Synthetic control.** Moderate pre-fit and opposite sign; it does not
  corroborate the DiD.
- **Test reuse.** The test period is not a new untouched confirmation for V2.
- **Forecasting geography.** The location set is crash-dense and non-random.
- **High-risk calibration.** The model under-predicts high-crash months; 90%
  intervals cover 84.8% of the highest-risk decile.
- **V2 not deployed.** 0.24% validation improvement, below the 1% rule. The
  coverage-tuned model is also research-only (0.48% validation improvement).
- **Road-task ambiguity.** 34.0% of road assignments have a second segment
  within 5 m; road and camera-zone MAE are not comparable.

---

## Authors and contributions

**Saketh Donepudi** and **Amrutha Kakumani** are the co-authors of this release.
The repository builds on the original study, notebook and README created by
Amrutha Kakumani and earlier contributors; the original notebook is preserved
unchanged.

Affiliations are withheld by author decision for this release. Individual
contribution and CRediT wording is recorded in `docs/AUTHOR_CONTRIBUTIONS.md`
and approved for release.

---

## Citation

```text
Donepudi, S., & Kakumani, A. (2026). Do Speed Cameras Make Chicago Safer?
A Geospatial Causal Inference and Machine Learning Study [Working paper].
GitHub. https://github.com/amruthakakumani/Do-Chicago-speed-cameras-reduce-crashes-
```

A machine-readable citation is provided in [`CITATION.cff`](CITATION.cff).

---

## Repository structure

```text
.
├── paper/                          # manuscript, PDF/DOCX, 11 publication figures
├── src/chicago_cameras/
│   ├── ml/                         # V1/V2 forecasting, intervals, geography, recursion
│   ├── acquisition.py              # portal downloads + provenance
│   ├── cleaning.py                 # schema checks, coverage guard
│   ├── geospatial.py               # UTM metres, counting, overlap diagnostics
│   ├── matching.py                 # matched comparison groups
│   ├── design.py                   # catchments, treatment timing
│   ├── did.py / event_study.py     # causal estimators
│   ├── synthetic_control.py        # synthetic control
│   ├── robustness.py               # sensitivity and falsification
│   └── visualization.py            # causal figures
├── dashboard/app.py                # public research dashboard
├── scripts/                        # download, pipeline, replication, ML, figures
├── models/                         # V1 deployed model; V2 candidate; metadata
├── results/                        # causal and ML artefacts
├── tests/                          # 63 tests
├── docs/                           # audit, methods, data dictionary, readiness
└── Do_Chicago_speed_cameras_reduce_crashes_.ipynb   # original notebook
```

---

## Tests and reproducibility status

- `pytest -q`: **63 passed**.
- `ruff check`: clean.
- Original notebook: unchanged.
- Causal results: unchanged from the audited release.
- Dashboard: launches locally and reads precomputed artifacts; no live URL is
  claimed.
- Manuscript: Markdown, PDF and DOCX generated from the same source.
- Full status: [`docs/PUBLICATION_READINESS.md`](docs/PUBLICATION_READINESS.md).

---

## License

The upstream repository does not declare a license. No license is added here
because re-licensing requires the copyright holders' authorization.

## Acknowledgments

The original Chicago speed-camera study, notebook and README were created by
Amrutha Kakumani and contributors; see the git history. This release preserves
that work and does not claim authorship of it. The V1 forecasting system and V2
research candidate were developed in this repository with the methods and
limitations documented above.

No affiliation, credential, peer-review status, or institutional endorsement is
asserted.
