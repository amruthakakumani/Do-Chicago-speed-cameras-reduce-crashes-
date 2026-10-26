# Final publication audit

**Branch:** `publication/final-review`
**Date:** 2026-10-27
**Status:** final internal review complete; all changes uncommitted.
**Decision:** **READY FOR CO-AUTHOR REVIEW — NOT READY FOR PUBLIC RELEASE.**

This audit supersedes the earlier pre-bottom-up readiness record. The original
notebook, V1/V2 models, original results and authorship records are preserved.

---

## 1. Manuscript verification

| Check | Result |
|---|---|
| Markdown source | `paper/chicago_speed_cameras_research.md` |
| Body words excluding references/appendices | **4,997** |
| Total words including references/appendices | 5,613 |
| PDF | **22 pages** |
| DOCX | **11 inline images; 6 tables; 204 paragraphs** |
| Figures | **11 PNG + 11 SVG** |
| Figure generator | `scripts/make_publication_figures.py` |
| Tables | V1/V2, common model comparison, coverage regime, sensitivity, causal, etc. |
| Original 21-page paper | Preserved on `publication/research-release`; final-review version is a new file revision |
| Original notebook | Zero diff at repository root |
| V1/V2 artifacts | Preserved unchanged |

The final paper integrates:

- September 2017 citywide reporting boundary and the pre-citywide V1/V2 target.
- Original −7.3% (33 cameras), revised −2.6% (20 cameras), coverage-consistent
  −8.6% (15 cameras; 95% CI −16.9% to +0.4%; p = 0.061).
- Geographic matching and cohort sensitivity; 12/36 coverage design −6.9%.
- V1/V2, coverage-consistent and road-segment model results.
- High-risk calibration and interval undercoverage.
- Why AADT was excluded.
- Explicit sample, estimand and forecasting-population distinctions.

---

## 2. Verified statistics

| Claim | Verified value | Source |
|---|---:|---|
| Original Poisson DiD | −7.3% [−13.6%, −0.6%], p = 0.0342 | `results/estimates.json → replication.original_notebook` |
| Revised conditional Poisson FE | −2.59% [−11.47%, +7.19%], p = 0.5909 | `results/estimates.json` primary |
| Coverage 24/36 | −8.63% [−16.88%, +0.43%], p = 0.0612 | `results/bottom_up/coverage_causal/estimates.json` |
| Coverage 12/36 | −6.95% [−15.24%, +2.16%], p = 0.1307 | same |
| Coverage cohort 2×2 | −0.075 [−0.773, +0.605], p = 0.829 | same |
| Coverage pre-trend p | 0.855 | same |
| Coverage geographic placebo p | 0.25 | `results/bottom_up/coverage_causal/placebo_summary.json` |
| V1 test MAE / deviance | 1.5683 / 1.2233 | `results/ml/v1/metrics.json` |
| V2 test MAE / deviance | 1.5642 / 1.2181 | `results/ml_v2/metrics.json` |
| V2 validation gain | 0.243% | `results/ml_v2/deployment.json` |
| Coverage tuned validation deviance | 1.2165 | `results/bottom_up/coverage_model/coverage_tuned_metrics.json` |
| Coverage tuned common test MAE / deviance | 1.5775 / 1.2162 | `results/bottom_up/coverage_model/common_metrics.csv` |
| V1 common test MAE / deviance | 1.5860 / 1.2242 | same |
| Common test rows | 15,912 | same |
| V1 interval coverage | 90.6% overall; 84.8% high-risk | `results/ml_v2/metrics.json` |
| Coverage tuned interval coverage | 90.4% overall; 84.9% high-risk | coverage metrics |
| Road MAE / deviance / R² | 0.182 / 0.395 / 0.351 | `results/bottom_up/road_model/road_metrics.json` |
| Road highest-risk interval coverage | 60.2% | same |
| AADT actual first month | 2022-07 | `results/bottom_up/exposure/summary.json` |
| AADT segments / segment-months | 2,525 / 83,232 | same |
| Road assignment | 99.26%; 34.0% ambiguous | `results/bottom_up/roads/assignment_summary.json` |
| Snapshot / current portal rows | 1,106,882 / 1,107,461 | `results/bottom_up/coverage/summary.json` |
| Tests | **63 passed** | `pytest` |

No estimate was selected or altered by significance. The coverage estimate is
reported with its wide interval, small sample and placebo distribution.

---

## 3. Citation verification

- **13 DOI references** verified through Crossref:
  - 10.1002/14651858.CD004607.pub3
  - 10.1016/j.jeconom.2020.09.006
  - 10.1016/j.jeconom.2020.12.001
  - 10.1016/j.jeconom.2021.03.014
  - 10.1073/pnas.2520328122
  - 10.1093/ije/dyh299
  - 10.1136/bmj.38324.646574.AE
  - 10.1162/003355304772839588
  - 10.1162/rest.90.3.414
  - 10.1177/03611981221104808
  - 10.1198/jasa.2009.ap08746
  - 10.1257/aer.20181169
  - 10.1371/journal.pone.0221267
- External URLs checked: City of Chicago enforcement (200), Census TIGER 2017/2024
  (200), Chicago Data Portal AADT (200), community areas (200), UIC report (202,
  Cloudflare challenge), Open-Meteo documentation (200).
- No invented citations, affiliations, credentials or URLs.

---

## 4. Figure audit

All figures are generated from saved artifacts by
`scripts/make_publication_figures.py`; PNG is 300 dpi and SVG is vector.

| Figure | Purpose |
|---|---|
| 1 | Study map and comparison locations |
| 2 | Overlapping 250 m catchments |
| 3 | Original, revised and coverage-consistent causal estimates |
| 4 | Event study |
| 5 | Pre-treatment trends |
| 6 | Observed vs V1 forecast |
| 7 | Calibration and high-risk errors |
| 8 | V1 conformal interval coverage |
| 9 | September 2017 reporting break |
| 10 | Common-row model comparison |
| 11 | Road-task interval coverage |

The original eight figures were regenerated by the same script; Figure 3 now
includes coverage-consistent rows. Figure 9 is placed after Figure 8 to preserve
figure-number order; it is referenced in Section 3.

---

## 5. Dashboard verification

- Seven pages: Research overview, Chicago map, Causal inference, Crash forecasting,
  Robustness & sensitivity, Coverage & reconstruction, Methods & limitations.
- All seven pages render in Streamlit `AppTest` with zero exceptions.
- Reporting-era camera overlay works (gold pre-September 2017; purple post).
- Coverage page shows the reporting break, common-row model comparison, road-task
  warning and causal sensitivity.
- Camera-zone estimates, road predictions and causal findings remain separate.
- Local HTTP response: **200**. No public URL is claimed.
- Audit: `docs/DASHBOARD_AUDIT.md`.

---

## 6. Automated checks

| Check | Result |
|---|---|
| `pytest` | 63 collected/passed, exit 0 |
| `ruff check` | clean |
| Local Markdown links/images | no missing links |
| `CITATION.cff` | valid YAML |
| PDF | 22 pages, no build error |
| DOCX | opens; 11 images, 6 tables |
| Git diff whitespace | clean |
| Credentials/large raw data | none found; `data/` gitignored |
| Dashboard | seven pages + overlay; HTTP 200 |

---

## 7. Remaining scientific concerns

1. **Coverage-consistent causal sample is small.** Only 15 cameras, mostly the
   2023 cohort; the −8.6% estimate is imprecise and sits inside its geographic
   placebo distribution. It is not a replacement for the −2.6% estimate.
2. **V1/V2 target begins before citywide reporting.** The final paper discloses
   this; the models are retained because retraining does not meet the 1% rule.
3. **AADT is not historical exposure.** The actual extract starts in 2022-07,
   not 2020-10; only 1,454 distinct 2024 road segments match within 50 m.
4. **Road assignment ambiguity.** 34.0% of assignments have a second segment
   within 5 m; the road task is not comparable to camera-zone MAE.
5. **2024 road geometry.** 98.9% of 2024 Cook segments existed in 2017, but 6,633
   older segments are absent; block-face history is incomplete.
6. **Test-period reuse.** The 2025-09…2026-08 test period was already inspected in
   V1; all V2 and coverage-consistent test results are retrospective.
7. **High-risk intervals.** Coverage is 84.8–84.9% in the highest-risk camera
   decile and 60.2% for road segments at nominal 90%.
8. **No camera uptime/warning/enforcement data.** Treatment timing is coarse.

These are disclosed in the paper and do not support stronger causal or
deployment claims.

---

## 8. Author approvals required

- [ ] Saketh Donepudi and Amrutha Kakumani approve the final manuscript wording.
- [ ] Affiliations are verified and added.
- [ ] CRediT/individual contribution wording is confirmed in
      `docs/AUTHOR_CONTRIBUTIONS.md`.
- [ ] Both co-authors approve the LinkedIn article and launch-package wording.
- [ ] Both co-authors approve any dashboard deployment and verified URL.
- [ ] License decision is made; no license is currently declared.
- [ ] Explicit authorization is given before commit, push, merge, release or
      publication.

No affiliation, credential, peer-review status, DOI, release or live dashboard
URL is asserted.

---

## 9. Release readiness

**READY FOR CO-AUTHOR REVIEW. NOT READY FOR PUBLIC RELEASE.**

The scientific, software, manuscript, figure, citation and dashboard checks pass.
Publication is blocked only on author approvals, affiliation/contribution
confirmation, license decision and explicit release authorization.

**Repository state:** `publication/final-review`; all changes uncommitted. No
commit, push, merge, deployment, release or publication was performed.

---

## 10. Suggested pull request

> This PR integrates the bottom-up reconstruction into the final co-authored
> manuscript. It adds the September 2017 reporting boundary, the original −7.3%,
> revised −2.6% and coverage-consistent −8.6% estimates with explicit sample and
> estimand distinctions; adds the coverage-consistent and road-segment model
> results; documents the AADT exclusion and high-risk calibration limitations;
> regenerates the Markdown/PDF/DOCX manuscript with 11 figures; and updates the
> README, model cards, methodology documentation, LinkedIn drafts and dashboard.
> V1 remains deployed and V2 remains a research candidate because neither the V2
> nor coverage-tuned model meets the pre-declared 1% rule. No release, deployment,
> push or publication is performed without authorization.
