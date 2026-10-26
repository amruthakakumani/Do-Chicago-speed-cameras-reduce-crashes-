# We Built a Chicago Crash Prediction Model. Adding 40 Features Barely Improved It.

*Co-authored by Saketh Donepudi and Amrutha Kakumani.*

Machine learning projects often assume that more features mean better
predictions. We tested that assumption on a real urban forecasting problem:
predicting monthly crash counts at crash-dense locations in Chicago. The result
was a useful lesson in restraint.

## The dataset

We used public Chicago crash records from 2015 to 2026, aggregated to
location-months. A location is either a 250-metre camera catchment or a matched
comparison site. The final analysis set contains 1,355 locations and 157,180
location-months. The target is the number of reported crashes within 250 metres
of the location during a calendar month.

The geography is not a random sample of Chicago streets. It is the crash-dense
geography used by the causal study. That matters: a model trained here should not
be assumed to work on a quiet residential block.

The original target window starts in January 2017. The Chicago portal is not
citywide until September 2017, so those first eight target months sit in a
partial-reporting regime. We therefore built a second, coverage-consistent
dataset starting in 2018-01: 1,326 locations and 137,904 location-months.

## The forecasting system

We trained a single LightGBM model with a Poisson objective. The Poisson
assumption matches a count outcome, and gradient-boosted trees handle nonlinear
interactions, missing values and high-cardinality geography without a bespoke
parametric form.

The first version, V1, used 44 features:

- calendar terms and US holidays;
- crash lags at 1, 2, 3, 6 and 12 months;
- rolling means and standard deviations over 3, 6 and 12 months;
- expanding mean and standard deviation;
- zero share, injury share, severe share and trend;
- lagged weather;
- static geography and camera operational context.

The engineering constraint was leakage. Every history feature is shifted by at
least one month. Rolling windows look backwards. Weather is lagged, never taken
from the forecast month. A build-time test perturbs future outcomes and verifies
that earlier features do not move. If the test fails, the build fails.

## V2: 84 features, real road data

For V2 we added 40 features:

- longer lags (24 and 36 months), same-month three-year means, seasonal ratios,
  coefficient of variation, months since the last crash and citywide regime
  features;
- Gaussian-kernel neighbourhood crash history, using only prior months;
- US Census TIGER/Line 2024 road-network features: road density at 500 m and
  1 km, major and local road density, major-road share, intersection density,
  distance to major roads and major intersections;
- expanding climate normals for recursive forecasts.

We did not add traffic volume. The public AADT extract actually begins in July
2022, not October 2020, and covers only 2,525 segments. Using it for 2017–2025
would leak future exposure or require fabricated backfills. We recorded the
limitation instead of manufacturing a feature.

The validation design was deliberately conservative. We used three expanding
six-month folds strictly before a locked test period, a geographic holdout of 34
two-kilometre blocks with a 1 km buffer, and a combined temporal-plus-geographic
holdout. The test period was 2025-09 to 2026-08.

## The result: 40 features bought about 0.4%

On the retrospective test period, V1 achieved MAE 1.5683, RMSE 2.0931 and
Poisson deviance 1.2233. V2 achieved MAE 1.5642, RMSE 2.0888 and deviance 1.2181.
That is a 0.4% reduction in deviance and a 0.3% reduction in MAE.

On the pre-test validation folds, the improvement was even smaller: 0.24% in
Poisson deviance. We had pre-declared a 1% promotion threshold. V2 did not meet
it. V1 remains deployed.

The coverage-consistent candidate adds real road, neighbourhood and citywide
features and 8 Optuna trials. On 15,912 common test rows it reaches deviance
1.216 versus 1.224 for V1—a 0.65% improvement, still below 1%. Its geographic
holdout is slightly worse than the simpler coverage model. It is a research
artifact, not a replacement.

This was not a case of a broken model. It was a case of diminishing returns. The
dominant signal is crash history: a location's recent 12-month average and its
ratio to the citywide average account for most of the model's gain. Road density
and intersection density help at unseen locations—V2 improves geographic-holdout
deviance by about 1.1%—but they do not transform one-step accuracy.

## Where the model fails

Two failures matter.

First, **high-crash months are under-predicted**. The top actual-crash decile has
a mean of 10.99 crashes and a mean prediction of about 9.02. The model shrinks
toward the mean, which is expected under a deviance loss but dangerous if the
forecast is used for high-risk decisions.

Second, **zero-crash months are over-predicted**. Zero months receive a mean
prediction of about 1.72. Again, this is the other side of shrinkage.

A richer feature set did not fix either problem. Throwing more variables at a
count model does not automatically improve the tails.

## Prediction intervals

Point forecasts are not enough for a research dashboard, so we added split
conformal intervals. The nonconformity score is scale-normalised:
`|y − μ| / sqrt(μ + 1)`. Calibration uses out-of-fold rolling predictions, not the
test period and not in-sample fitted values.

The 90% intervals achieve **90.6% marginal coverage** with a mean width of 5.86
crashes. That sounds good until you look at conditional coverage: the lowest-risk
decile is covered at 96.6%, while the highest-risk decile is covered at **84.8%**.
The intervals adapt with the predicted count, but they are still approximate.
They are not causal confidence intervals, and they should not be presented as
guarantees.

We also ran a genuine 12-month recursive forecast, feeding predictions into
subsequent lags and neighbourhood histories. Both models over-predict by roughly
10–12%; V2 is modestly better (MAE 1.623 vs 1.645). Recursive and one-step errors
are not interchangeable.

## The dashboard

The Streamlit dashboard keeps a large PyDeck map central. It shows cameras,
250 m treatment buffers, comparison sites, observed crash density, predictions,
predicted risk intensity, 90% interval width, held-out errors and historical
hotspots. Filters cover year, month, camera, region and severity. Selecting a
camera shows its history, the latest forecast, the interval, historical error and
SHAP explanations.

Causal estimates are in a separate section. The dashboard never presents the
study-wide camera effect as a camera-specific effect, and it never presents a
prediction interval as a causal confidence interval.

## Engineering lessons

1. **History dominates.** For monthly crash counts, simple lags and rolling
   averages are hard to beat.
2. **Real geospatial features help generalization more than one-step accuracy.**
   Road density and intersections improved the geographic holdout, not the
   headline test score.
3. **More features increase cost.** V2 has 84 features and is slower to train and
   score; the gain did not justify replacing V1.
4. **Calibration is conditional, not just marginal.** A 90.6% overall coverage can
   hide 84.8% coverage in the group that matters most.
5. **Test discipline matters.** We had already examined the 2025–2026 period in
   V1. V2 and coverage-consistent test numbers are retrospective, not a new
   untouched confirmation.
6. **The reporting regime is part of the data.** A model trained on partial
   coverage can look accurate for reasons unrelated to traffic risk.
6. **Prediction is not causation.** A model that predicts crashes well does not
   tell you whether a camera caused them to fall.

## About this work

This is a co-authored research release by **Saketh Donepudi** and **Amrutha
Kakumani**. The original Chicago study, notebook and README were created by
Amrutha Kakumani and contributors. The forecasting system, validation design,
dashboard and release were developed in the same repository. Affiliations and
individual contribution roles are intentionally not asserted; they require
confirmation from both authors.

Repository: https://github.com/amruthakakumani/Do-Chicago-speed-cameras-reduce-crashes-

The paper is a working paper, not peer reviewed. No institutional endorsement is
claimed.
