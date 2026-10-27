# Do Speed Cameras Make Chicago Safer?

### A geospatial causal inference and machine learning study

**Saketh Donepudi and Amrutha Kakumani**

*Working paper · October 2026 · Not peer reviewed*

[Research paper (PDF)](paper/Chicago_Speed_Cameras_Research.pdf) ·
[Reproducibility](docs/REPRODUCIBILITY.md) ·
[Documentation index](docs/README.md) ·
[Final release audit](docs/FINAL_RELEASE_AUDIT.md)

---

## Research overview

Chicago installs speed cameras where crashes are already high, so a simple
before/after comparison confounds enforcement with regression to the mean,
local trends and citywide shocks. This project estimates the effect using
nearby matched comparison sites, then tests how the answer changes under
defensible choices about matching, overlap, treatment timing and reporting
coverage.

A second, deliberately separate analysis asks how accurately a LightGBM model
can forecast monthly crash counts at crash-dense locations. Forecasting is
never used as causal evidence. The central finding is **design sensitivity**:
the original −7.3% estimate is reproducible, but the preferred local design
gives −2.6% with a confidence interval that includes zero and a modest
increase. The data cannot resolve the policy question at high precision.

---

## Key findings

| Analysis | Estimate | Uncertainty | Sample |
|---|---:|---|---:|
| Original Poisson DiD (reproduced exactly) | **−7.3%** | 95% CI [−13.6%, −0.6%], p = 0.034 | 33 cameras |
| Revised conditional Poisson FE | **−2.6%** | 95% CI [−11.5%, +7.2%], p = 0.59 | 20 cameras |
| Cohort 2×2 | −0.095 crashes/site-month | 95% CI [−0.735, +0.520] | 20 cameras |
| Synthetic control (2023 cohort) | +4.2% | pre-fit R² 0.742; pseudo-p 0.495 | 7 cameras |
| Coverage-consistent 24/36 | **−8.6%** | 95% CI [−16.9%, +0.4%], p = 0.061 | 15 cameras |
| V1 forecast, held-out test | MAE 1.568 / deviance 1.223 | — | 16,260 location-months |
| 90% conformal interval coverage | **90.6%** | 84.8% in highest-risk decile | 16,260 rows |

A non-significant estimate is not evidence of zero effect. The wide intervals
reflect a small treated sample, overlapping catchments and the difficulty of
constructing credible comparison sites in a densely monitored city.

---

## Predictive modeling

The deployed forecaster is **LightGBM V1**, a Poisson model for monthly crashes
at 1,355 camera and comparison locations. It uses lagged crash history,
calendar and holiday features, lagged weather and static geography. All history
features are shifted, weather is lagged, and temporal/geographic holdouts are
used to test leakage and generalisation.

V2 adds road-network, neighbourhood and climate features but improves
retrospective deviance by only about 0.4%, below the pre-declared 1% promotion
rule. A coverage-consistent optimized LightGBM, a Negative Binomial model and a
convex hybrid were also evaluated; none met the rule. V1 remains deployed. The
experiment is retained as a compact negative result, not as a new production
model.

---

## Research methodology

- **Data:** Chicago Data Portal crash, camera and violation datasets, with
  provenance, checksums and reporting-coverage diagnostics.
- **Geospatial design:** 250 m camera catchments, 500 m comparison lattice,
  nearest five matched sites, 2,000 m primary matching radius and overlap
  removal.
- **Causal inference:** conditional Poisson fixed effects, per-camera cohort
  2×2, event study, synthetic control, wild cluster bootstrap and placebo tests.
- **Machine learning:** LightGBM Poisson forecasting with rolling-origin tuning,
  temporal/geographic validation, conformal intervals and leakage checks.

The original notebook is preserved at the repository root. The paper separates
causal estimates from predictive associations and reports negative findings.

---

## Research paper

The final paper is a single PDF:

**[paper/Chicago_Speed_Cameras_Research.pdf](paper/Chicago_Speed_Cameras_Research.pdf)**

The reproducible manuscript source is kept internally at
`paper/source/chicago_speed_cameras_research.md`; the PDF is the public
manuscript. The paper reports the exact replication, stepwise design changes,
event study, synthetic control, reporting boundary, coverage-consistent
sensitivity, forecasting system, conformal intervals and limitations. It is a
working paper, not peer reviewed, and no institutional endorsement is claimed.

---

## Interactive Chicago dashboard

The Streamlit dashboard has seven pages covering the research overview, Chicago
map, causal inference, crash forecasting, robustness, coverage/reconstruction
and methods/limitations. It reads precomputed, validated artifacts and never
retrains a model.

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

No public dashboard URL is claimed or provided. Whenever the dashboard is
hosted, deployment must be treated as a separate task from publishing this
repository. See [`docs/DASHBOARD_DEPLOYMENT.md`](docs/DASHBOARD_DEPLOYMENT.md).

![Study map](paper/figures/fig1_study_map.png)

---

## Reproducibility

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
python scripts/evaluate_lightgbm.py

# formatted paper
python scripts/make_cover_graphic.py
python scripts/build_publication_docs.py

# tests and dashboard
python -m pytest -q
streamlit run dashboard/app.py
```

The raw and processed data are not committed. The pipeline uses the public
Chicago Data Portal and cached provenance; it never substitutes synthetic data.
See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for expected outputs,
runtime and environment notes.

---

## Data sources

| Dataset | Source | Use |
|---|---|---|
| Traffic Crashes – Crashes `85ca-t3if` | Chicago Data Portal | Outcome |
| Speed Camera Locations `4i42-qv3h` | Chicago Data Portal | Camera coordinates and go-live |
| Speed Camera Violations `hhkd-xvj4` | Chicago Data Portal | Descriptive mechanism check |
| Historical weather | Open-Meteo archive | Lagged weather and climate normals |
| AADT `gc7y-n4xa` | Chicago Data Portal | Audited, **not used**: starts 2022-07 |
| Community areas `igwz-8jzy` | Chicago Data Portal | City boundary for road clipping |
| Road network | US Census TIGER/Line 2024 EDGES (Cook, DuPage) | Road density, class, intersections |
| Holidays | `holidays` US calendar | Calendar features |

Raw files retain URLs, query parameters, timestamps, row counts and SHA-256
checksums in `data/raw/provenance.json`; raw and processed data are gitignored.

---

## Limitations

- Cameras are not randomly sited; matching cannot eliminate regression to the
  mean or local confounding.
- The preferred treated sample is small (20 cameras), so intervals are wide.
- Citywide crash reporting begins in 2017-09; earlier target months are partial.
- No valid historical traffic-exposure denominator exists for the full window.
- The synthetic control has moderate pre-fit and the opposite sign; it does not
  corroborate the DiD.
- The forecasting location set is crash-dense and non-random, not a citywide
  random sample.
- The test period was already inspected during V1 development; V2 and
  coverage-consistent test numbers are retrospective.
- Prediction intervals are not causal confidence intervals, and SHAP values are
  not causal effects.
- Road-segment assignments are ambiguous for about one third of crashes near
  segment boundaries.

---

## Citation

```text
Donepudi, S., & Kakumani, A. (2026). Do Speed Cameras Make Chicago Safer?
A Geospatial Causal Inference and Machine Learning Study [Working paper].
GitHub. https://github.com/amruthakakumani/Do-Chicago-speed-cameras-reduce-crashes-
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

---

## License

Repository code and research materials are released under the **MIT License**;
see [`LICENSE`](LICENSE). Third-party datasets remain subject to their source
terms. Copyright © 2026 Saketh Donepudi and Amrutha Kakumani.

The license terms should be confirmed by both co-authors before the public
GitHub release is finalized.

---

## Repository structure

```text
.
├── paper/                          # final PDF, source, templates, publication figures
├── src/chicago_cameras/            # causal pipeline, geospatial design, ML
├── dashboard/app.py                # seven-page research dashboard
├── scripts/                        # acquisition, analysis, model and figure commands
├── models/v1/                      # deployed V1 model and metadata
├── results/                        # causal, forecasting and validation artifacts
├── tests/                          # automated tests
├── docs/                           # methods, data dictionary, audits, model card
└── Do_Chicago_speed_cameras_reduce_crashes_.ipynb   # original notebook
```

---

## Authors and acknowledgments

**Saketh Donepudi** and **Amrutha Kakumani** are the co-authors of this release.
The repository builds on the original study, notebook and README created by
Amrutha Kakumani and earlier contributors; that work is preserved in Git
history. Affiliations are withheld by author decision. Contribution details are
in [`docs/AUTHOR_CONTRIBUTIONS.md`](docs/AUTHOR_CONTRIBUTIONS.md).

No affiliation, credential, peer-review status or institutional endorsement is
asserted.
