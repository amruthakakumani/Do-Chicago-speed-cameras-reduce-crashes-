# Release notes — draft (not published)

## Research release v1.1 — final review

This release packages a reproducible causal study, a separate crash forecasting
system, a coverage-aware reconstruction and a road-segment research task. It is a
working paper, not a peer-reviewed publication. Nothing here is released,
deployed or published.

### Causal analysis

- Exact reproduction of the original notebook's −7.3% Poisson estimate.
- Revised conditional Poisson FE estimate: −2.6% (95% CI −11.5% to +7.2%).
- Coverage-consistent 24/36 estimate: −8.6% (95% CI −16.9% to +0.4%), 15 cameras;
  explicit different-sample/estimand caveat.
- Cohort 2×2, event study, synthetic control, geographic placebos and a full
  sensitivity table.
- Original notebook preserved unchanged; V1/V2 artifacts preserved.

### Forecasting

- Deployed LightGBM V1: MAE 1.5683, RMSE 2.0931, Poisson deviance 1.2233.
- V2 candidate with 84 features: retrospective deviance 1.2181, but validation
  gain 0.243%, below the pre-declared 1% promotion rule; V1 remains deployed.
- Coverage-consistent tuned candidate: common-row deviance 1.2162, 0.65% better
  than V1 and below the 1% rule; not deployed.
- Scale-normalised conformal intervals: 90.6% marginal V1 coverage; 84.8% in the
  highest-risk decile. Coverage-tuned: 90.4% / 84.9%.
- Separate road-segment task: MAE 0.182, deviance 0.395, highest-risk interval
  coverage 60.2%; not comparable to camera-zone MAE.

### Data and exposure

- Official citywide reporting starts September 2017; V1/V2 target starts 2017-01.
- AADT actual extract starts 2022-07, not 2020-10; 2,525 segments; excluded from
  historical modelling.
- Road network: 73,088 TIGER 2024 segments; 34.0% ambiguous assignments.

### Deliverables

- `paper/chicago_speed_cameras_research.md` / `.pdf` (21 pages) / `.docx` (21 pages when rendered)
- Cover, author-information page, automatic TOC, running headers and body page numbers
- 11 figures in `paper/figures/` (PNG + SVG)
- `paper/bottom_up_reconstruction.md`
- Updated README, model cards, methodology documents and LinkedIn drafts
- Local Streamlit dashboard with seven pages
- `docs/FINAL_PUBLICATION_AUDIT.md`
- 63 tests; reproducible artifacts and provenance

### Caveats

- The 2025-09 … 2026-08 test period was previously inspected; V2 and
  coverage-consistent test results are retrospective.
- No live dashboard URL is claimed.
- No license is declared; no affiliation or peer-review status is asserted.
- No DOI, release, deployment or publication is created.

Author release approval recorded 2026-10-27. No license is added; affiliations
are withheld by author decision. Dashboard deployment and LinkedIn publication
remain separate.
