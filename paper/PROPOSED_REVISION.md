# Proposed revision to the 21-page working paper

**Status:** APPLIED in `publication/final-review`; this file is retained as the
precise change log. The original paper remains preserved on
`publication/research-release`. Co-authors should review the applied wording
before release.

---

## 1. Why a revision is warranted

The bottom-up audit found a material definition issue: the Chicago portal is not
citywide until September 2017, while V1/V2 train from January 2017. The coverage-
consistent rebuild also produces a different causal sample and a separate road
task. These findings should be disclosed in the working paper even though they do
not justify replacing V1.

The central published finding still stands: the original −7.3% is reproducible
but not robust. The revision changes the interpretation of the forecasting
baseline and adds a second causal sensitivity; it does not turn the paper into a
new causal claim.

---

## 2. Proposed abstract edits

**Insert after the existing data sentence:**

> The portal is not citywide until September 2017. The deployed V1/V2 target
> starts in January 2017, so its first eight target months sit inside the partial-
> coverage regime. A coverage-consistent reconstruction starting in September
> 2017 does not materially improve held-out accuracy: on common test rows, the
> strongest coverage-tuned candidate improves Poisson deviance by 0.65% versus
> V1, below the pre-declared 1% promotion rule.

**Insert after the causal estimate sentence:**

> A stricter coverage-consistent design, which requires 24 citywide pre-months,
> leaves 15 cameras and yields −8.6% [−16.9%, +0.4%], p = 0.061, with a
> geographic placebo p-value of 0.25. This is a different sample, not a direct
> replication, and it remains compatible with zero.

**Do not change:** the existing −7.3% replication, −2.6% primary estimate,
V1 deployment decision, or the statement that the data cannot resolve the effect
at policy precision.

---

## 3. Proposed Section 3 (Data) edits

Replace the paragraph describing the analysis window with:

> The portal contains sparse records from 2013 and a coverage guard begins the
> publication panel in 2015-09. That guard removes near-empty early months, but
> it does not make the pre-September-2017 period citywide. The official portal
> description states that E-Crash data are available for some police districts in
> 2015 but citywide data are not available until September 2017. We therefore
> distinguish (a) the historical 2015-09 panel used by the publication, (b) the
> V1/V2 target window from 2017-01, and (c) the coverage-consistent target window
> from 2018-01. All three are reported; they are not interchangeable.

Add a coverage table:

| Period | Mean monthly crashes | Mean injury share |
|---|---:|---:|
| 2016-09…2017-08 | 5,382 | 7.7% |
| 2017-09…2018-08 | 9,817 | 14.1% |

Add a sentence: “All 22 districts have at least one record before the break, so
district presence is not a completeness measure.”

---

## 4. Proposed causal results edits

Add a subsection after the existing sensitivity table:

> **Coverage-consistent cohort.** Requiring 24 citywide pre-months and 36 post-
> months leaves 15 cameras (go-live 2021-07…2023-07). The conditional Poisson
> estimate is −8.6% [−16.9%, +0.4%], p = 0.061; the cohort 2×2 estimate is
> −0.075 [−0.773, +0.605] crashes/site-month; the pre-trend p-value is 0.855; the
> geographic placebo one-sided p-value is 0.25. This is a new sample and a new
> estimand, not a direct replication. It shows that the reporting regime is
> material to the estimate.

Add the 12/36 sensitivity: −6.9% [−15.2%, +2.2%], 16 cameras, p = 0.131.

Update the causal figure caption to show original, revised, coverage 24/36 and
coverage 12/36. Do not remove the original/revised estimates.

---

## 5. Proposed forecasting edits

Add after the V1/V2 comparison:

> The V1/V2 target starts in 2017-01, eight months before citywide coverage. A
> coverage-consistent dataset starting in 2018-01 contains 1,326 locations and
> 137,904 location-months. On the common 15,912 test rows, coverage-tuned
> deviance is 1.2162 versus 1.2242 for V1, a 0.65% difference with a paired
> spatial-block bootstrap interval of [−0.0114, −0.0043]. V2 is 0.41% better than
> V1 on the same rows. Neither meets the 1% rule. The coverage-tuned model’s
> geographic holdout (1.2267) is slightly worse than coverage V1 (1.2257).

Add a table of common-row metrics:

| Model | MAE | Deviance | Bias | R² |
|---|---:|---:|---:|---:|
| Deployed V1 | 1.5860 | 1.2242 | +0.164 | 0.657 |
| V2 candidate | 1.5820 | 1.2193 | +0.145 | 0.659 |
| Coverage V1 | 1.5799 | 1.2201 | +0.119 | 0.659 |
| Coverage full | 1.5786 | 1.2175 | +0.130 | 0.659 |
| Coverage tuned | 1.5775 | 1.2162 | +0.127 | 0.660 |

Keep V1 deployed. Add: “The coverage-tuned model is a research artifact, not a
deployment.”

---

## 6. Proposed interval edits

Add:

> Coverage-consistent intervals are 90.4–90.6% overall at nominal 90%, but
> 84.7–85.6% in the highest-risk decile. The high-risk undercoverage persists;
> additional features and tuning do not fix it.

Update the interval figure caption to use the coverage-consistent numbers only if
the figure is replaced; otherwise retain the V1 figure and add the sensitivity in
text.

---

## 7. Proposed limitations edits

Add:

1. **Reporting regime.** The publication’s V1/V2 target includes eight months
   before citywide coverage. The causal 2015-start panel also mixes partial and
   citywide reporting.
2. **Exposure.** AADT starts in 2022-07, not 2020-10, and covers only 1,454
   distinct 2024 road segments within 50 m. No historical exposure denominator is
   available for 2017–2021.
3. **Road-task ambiguity.** 34% of road assignments have a second segment within
   5 m; road-segment MAE is not comparable to camera-zone MAE.
4. **Road network vintage.** 98.9% of 2024 Cook segments existed in 2017, but
   6,633 older segments are absent; the 2024 network is not a perfect historical
   map.
5. **Causal power.** The strict coverage-consistent design has only 15 cameras;
   the estimate is imprecise and sits inside the geographic placebo distribution.

---

## 8. Proposed appendix additions

Add a supplementary appendix pointing to:

- `paper/bottom_up_reconstruction.md`
- `docs/BOTTOM_UP_AUDIT.md`
- `docs/GEOGRAPHIC_DESIGN_COMPARISON.md`
- `docs/FEATURE_AVAILABILITY_LEDGER.md`
- `results/bottom_up/`

Do not add a new DOI, release, deployment or peer-review claim.

---

## 9. What must not change

- The original notebook reproduction and its −7.3% estimate.
- The revised local-design −2.6% estimate.
- The V2 0.243% validation result and the decision **not** to deploy V2.
- The 21-page paper’s title, authors and working-paper status until both
  co-authors approve.
- The LinkedIn articles and publication branch.

---

## 10. Approval checklist

- [ ] Both co-authors approve the reporting-regime interpretation.
- [ ] Both co-authors approve adding the coverage-consistent causal sensitivity.
- [ ] Both co-authors approve the common-row forecasting table.
- [ ] Both co-authors approve the exposure and road-task limitations.
- [ ] Both co-authors decide whether to regenerate the paper PDF/DOCX.
- [ ] Both co-authors decide whether to update the README and dashboard text.
- [ ] No release, push, merge, deployment or publication before authorization.
