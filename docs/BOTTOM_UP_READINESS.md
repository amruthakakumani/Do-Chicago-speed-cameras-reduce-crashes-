# Bottom-up reconstruction readiness

**Branch:** `research/bottom-up-forecasting`
**Date:** 2026-10-27
**Decision:** **RESEARCH COMPLETE — READY FOR CO-AUTHOR REVIEW.**
No model is newly deployed; V1 remains deployed.

---

## 1. Acceptance criteria

| Criterion | Status | Evidence |
|---|---|---|
| Official coverage statement verified | PASS | Portal `85ca-t3if` metadata; `docs/BOTTOM_UP_AUDIT.md` |
| Raw record uniqueness checked | PASS | 1,106,882 rows / 1,106,882 unique IDs |
| Missing coordinates quantified | PASS | 8,673 missing; 75 zero; 8,748 outside bbox |
| Amendments / post-snapshot records audited | PASS | 240 AMENDED; 581 new; 2 absent; no version history |
| Reporting delay audited | PASS | Notification p95 2.53d, p99 9.17d; finalization not observable |
| Duplicate spatial attribution measured | PASS | 34,222 shared (36.3%); 107 pairs |
| Missing periods not treated as zeros | PASS | Coverage diagnostics; 2015-09 guard; citywide boundary |
| Coverage-consistent camera-zone dataset | PASS | 1,326 locations; 137,904 rows from 2018-01 |
| Citywide-consistent causal design | PASS | 15 cameras, 24/36; 12/36, 24/24, 12/24 sensitivities |
| Pre-trend and placebo diagnostics | PASS | Pre-trend p = 0.855; geographic placebo p = 0.25 |
| Road-segment geography | PASS | 73,088 segments; stable TLIDs; 2024 vs 2017 comparison |
| Crash-to-road assignment | PASS | 99.26% assigned; 34.0% ambiguous; 2,214 low confidence |
| Road modelling sample | PASS | 15,126 segments; 1,633,608 rows; 23.6% of crashes |
| Historical AADT assessed | PASS | Actual 2022-07 start; 2,525 segments; excluded from model |
| Exposure chronology never backfilled | PASS | `FEATURE_AVAILABILITY_LEDGER.md`; AADT not used |
| Feature ablation | PASS | 34→78 features; 0.67% deviance improvement vs minimal |
| LightGBM optimization | PASS | 8 Optuna trials, 600s cap |
| Temporal validation | PASS | Three rolling folds |
| Geographic holdout | PASS | 34 blocks, 290 locations, 1 km buffer |
| Calibration / zero / high-crash | PASS | Zero and top-decile errors reported |
| Hotspot precision/recall | PASS | Camera-zone 0.72/0.55; road 0.48/0.46 |
| Prediction interval coverage | PASS | Camera 90.4%; road 90.3%; high-risk 84.9%/60.2% |
| V1/V2 preserved | PASS | Original artifacts untouched |
| V2 decision not altered | PASS | Still research candidate; 0.243% validation gain |
| New model selection protocol | PASS | 1% validation deviance threshold declared before test |
| Final model selection | PASS | V1 retained; coverage-tuned fails threshold |
| Dashboard upgraded | PASS | Coverage page, overlay, separate road task |
| Tests | PASS | 63 collected/passed after addition |
| Original paper untouched | PASS | Proposed revision only |
| LinkedIn articles untouched | PASS | Publication drafts unchanged |
| No credential or raw-data commit | PASS | `data/` ignored |
| No release/DOI/deployment | PASS | None created |

---

## 2. Model-selection result

The **pre-declared rule** was: improve validation Poisson deviance by at least 1%,
do not regress test MAE by more than 2%, do not regress high-crash MAE by more
than 2%, and keep geographic/calibration acceptable.

| Model | Validation deviance | vs coverage V1 | Decision |
|---|---:|---:|---|
| V1 deployed | 1.2227 | — | Keep |
| V2 candidate | 1.2197 | — | Keep as candidate |
| Coverage V1 | 1.2224 | reference | Research reference |
| Coverage full | 1.2167 | +0.47% | Fail 1% |
| Coverage tuned | 1.2165 | +0.48% | Fail 1% |
| Road segment | 0.4043 | different task | Not comparable |

The coverage-tuned model improves common-row test deviance by 0.65% versus V1
(paired spatial-block bootstrap interval [−0.0114, −0.0043]), but it fails the
1% rule and is slightly worse than coverage V1 on the geographic holdout
(1.2267 vs 1.2257). It is retained as a research artifact, not deployed.

---

## 3. Key negative findings

1. The V1/V2 target includes eight months before citywide crash coverage.
2. The coverage-consistent causal design has only 15 cameras and an imprecise
   estimate: −8.6% [−16.9%, +0.4%], p = 0.061; placebo p = 0.25.
3. Road assignment is ambiguous for 34% of crashes; the highest-risk road
   interval coverage is 60.2%.
4. AADT is not a historical exposure series for this study: actual first month
   2022-07, 2,525 segments, 1,454 matching 2024 road segments.
5. More features and tuning improve deviance by less than 1%; the deployed V1 is
   retained.

---

## 4. Remaining limitations

- No historical traffic exposure before 2022-07.
- Camera uptime, warning periods and enforcement hours are unobserved.
- Coverage before September 2017 cannot be reconstructed at district/beat level.
- Amendments lack version history.
- Road-segment assignment ambiguity and 2024 geometry vintage.
- The test period 2025-09…2026-08 was already inspected in V1; all new test
  results are retrospective.
- Coverage-consistent causal sample is small and mostly the 2023 cohort.

---

## 5. Approval items

- [ ] Both co-authors approve the coverage-regime interpretation.
- [ ] Both co-authors approve the proposed revision in
      `paper/PROPOSED_REVISION.md`.
- [ ] Both co-authors decide whether the paper/README/dashboard text changes are
      applied to the publication branch.
- [ ] Both co-authors approve the road-task framing and ambiguity disclosure.
- [ ] Both co-authors approve any public dashboard deployment.
- [ ] Explicit authorization before commit, push, merge, release or publication.

No paper, article, DOI, release, deployment or LinkedIn post was changed or
published.

---

## 6. Suggested pull request description

> This PR adds a bottom-up reconstruction of the Chicago speed-camera study on a
> separate branch. It verifies the official September 2017 citywide reporting
> boundary, rebuilds a coverage-consistent camera-zone dataset, adds a separate
> road-segment prediction task, and audits historical AADT availability. The
> original publication, 21-page paper, V1/V2 artifacts and LinkedIn drafts are
> preserved. V1 remains deployed: the strongest coverage-tuned candidate improves
> validation deviance by 0.48% against a pre-declared 1% rule, and the road task
> is not comparable to the camera-zone task. No commit, push, deployment or
> publication is performed by this PR.
