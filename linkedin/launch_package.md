# LinkedIn launch package

**Status:** drafts only. Nothing here has been published or scheduled. Both
co-authors must approve the final wording, contribution statement and
affiliations before release.

---

## 1. Articles

| File | Headline | Length |
|---|---|---|
| `linkedin/article_a_causal.md` | We Studied Chicago's Speed Cameras. The Most Interesting Finding Was How the Answer Changed. | ≈1,247 words |
| `linkedin/article_b_ml.md` | We Built a Chicago Crash Prediction Model. Adding 40 Features Barely Improved It. | ≈1,265 words |

Both articles name **Saketh Donepudi** and **Amrutha Kakumani** as co-authors and
link to the GitHub repository. Neither invents individual contributions.

---

## 2. Short launch post

> **Do speed cameras make Chicago safer? We rebuilt the analysis to find out
> whether the answer survives scrutiny.**
>
> Our original matched comparison suggested a 7.3% reduction in crashes near
> cameras. But the estimate depended on matching each camera to comparison sites
> anywhere in the city and keeping cameras whose catchments overlapped. When we
> restricted matching to nearby sites, removed overlap and used a proper metric
> projection, the estimate moved to −2.6% with a 95% interval of −11.5% to +7.2%.
> A stricter design using only citywide-reporting months leaves 15 cameras and
> gives −8.6% [−16.9%, +0.4%]—a different sample, not a stronger causal result.
>
> That is not proof cameras do nothing. It is evidence that the public data cannot
> resolve the effect at policy-relevant precision.
>
> We also built a separate LightGBM crash forecaster. It predicts monthly counts
> reasonably well—MAE 1.568—but adding 40 features, including real Census road
> data, improved retrospective deviance by only 0.4%. The simpler model remains
> deployed.
>
> Full paper, audit, code and figures: [GitHub link]
>
> Co-authored by Saketh Donepudi and Amrutha Kakumani. Working paper, not peer
> reviewed. No institutional endorsement claimed.

---

## 3. Technical follow-up post

> **What we learned engineering a Chicago crash forecasting system.**
>
> 1,355 locations. 157,180 location-months. One LightGBM Poisson model.
>
> V1: 44 features. MAE 1.568, Poisson deviance 1.223.
> V2: 84 features, including Census TIGER road density and intersections. MAE
> 1.564, deviance 1.218.
> Coverage-consistent tuned model: 78 features, common-row deviance 1.216,
> 0.65% better than V1—still below the 1% rule.
>
> That 0.4% gain did not meet our pre-declared 1% validation threshold, so V1 stays
> deployed. History features dominate; road features help geographic holdouts more
> than one-step accuracy.
>
> The uncomfortable part is calibration. 90% conformal intervals cover 90.6% of
> test rows overall—but only 84.8% of the highest-risk decile. High-crash months
> are under-predicted. More features did not fix the tails.
>
> Repo: [GitHub link]
> Paper: [GitHub paper link]
>
> Co-authored with Amrutha Kakumani. Working paper; not peer reviewed.

---

## 4. GitHub project description

> Reproducible study of Chicago speed cameras and crashes: matched
> difference-in-differences, conditional Poisson FE, event study, synthetic
> control and placebos, plus a LightGBM crash forecaster with conformal
> intervals. Working paper and interactive dashboard. Not peer reviewed.

---

## 5. Article-cover concepts

No image-generation tool is used here; these are visual briefs, not fake files.

**Article A (causal):**
- Split cover: left "−7.3%" in muted red; right "−2.6% [−11.5%, +7.2%]" in navy.
- Thin Chicago street-grid linework in the background at 15% opacity.
- Caption strip: "Same data. Different spatial design. Different answer."
- Avoid alarm imagery, police imagery or crash photographs.

**Article B (ML):**
- Two columns: "44 features" and "84 features" with nearly identical error bars.
- A small calibration curve with the 90% line and the high-risk decile highlighted.
- Caption strip: "More features. Almost the same forecast."
- Neutral academic palette (navy, muted red, sand).

---

## 6. Featured-section summaries

**For either co-author's LinkedIn Featured section:**

> Co-authored research release: *Do Speed Cameras Make Chicago Safer?* A
> reproducible causal audit and LightGBM crash forecasting study using public
> Chicago data. Finds that the original −7.3% camera estimate is reproducible but
> not robust to local matching and overlap removal; the revised estimate is −2.6%
> with a wide interval. Includes a working paper, eight figures, a local
> dashboard and full code.

Both co-authors should use the same project description; do not imply separate
studies.

---

## 7. Co-author acknowledgment text

> This is a co-authored research release by Saketh Donepudi and Amrutha Kakumani.
> The original Chicago study, notebook and README were created by Amrutha
> Kakumani and contributors. The causal audit, revised design, forecasting system
> and release were developed in the same repository. Affiliations and individual
> contribution roles are not asserted here; they require confirmation from both
> authors. The paper is a working paper, not peer reviewed, and no institutional
> endorsement is claimed.

---

## 8. Publishing sequence (proposal only)

1. Both co-authors approve the manuscript, author contributions, affiliations and
   LinkedIn wording.
2. Publish the GitHub repository (after the release is approved; not done here).
3. Publish Article A (causal) on a weekday morning.
4. Publish Article B (ML) 48–72 hours later.
5. Post the short launch post on the same day as Article A, linking to the
   repository.
6. Post the technical follow-up 24 hours after Article B.
7. Update both co-authors' Featured sections after the articles are live.

No post is scheduled or published automatically.

---

## 9. Editorial notes requiring approval

- **Affiliations:** not stated because they are unverified. Add only with
  author confirmation.
- **Individual contributions:** proposed CRediT roles in
  `docs/AUTHOR_CONTRIBUTIONS.md` are marked pending. Do not publish them as final.
- **Dashboard URL:** none is included because no deployment exists. Add only
  after a verified deployment.
- **License:** no license is declared by the upstream repository; do not add one
  without authorization.
- **Peer review:** the manuscript is a working paper. Do not describe it as
  peer reviewed or published.
- **Test period:** the 2025-09 … 2026-08 period was previously inspected; V2
  results are retrospective. Keep this caveat in both articles.
