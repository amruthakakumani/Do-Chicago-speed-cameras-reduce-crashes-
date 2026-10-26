# We Studied Chicago's Speed Cameras. The Most Interesting Finding Was How the Answer Changed.

*Co-authored by Saketh Donepudi and Amrutha Kakumani.*

Speed cameras occupy an uncomfortable place in transportation policy. They are
installed because a location has a bad crash record. Then, when crashes fall
after installation, it is tempting to conclude that the camera caused the
improvement. That inference is often too quick.

We spent the past several months rebuilding a Chicago speed-camera study from
public data, auditing every statistical choice, and asking a narrower question:
**how much does the estimated effect depend on the research design?** The answer
turned out to be more interesting than the headline number.

## The initial result looked strong

The project's original notebook estimated that crashes near Chicago speed
cameras fell by about 7.3% relative to comparison locations. The result was
statistically significant at conventional levels. The design was a matched
difference-in-differences model: compare crash changes at camera catchments with
changes at locations that never received a camera.

That is a sensible starting point. But when we reproduced the notebook exactly,
we found that the code and the documentation did not quite agree. The README said
that comparison sites had to be within 1,000 metres of the camera they were
matched to. The code never applied that restriction. It selected the five
best-matching comparison sites anywhere in the city. The original analysis also
kept cameras whose 250-metre catchments overlapped a camera that was already
live, and it used a flat degrees-to-metres approximation rather than a projected
metric coordinate system.

None of this means the original result was fabricated. We reproduced its numbers
exactly: the same 33 cameras, the same 11,880 panel rows, the same −7.3% estimate
and the same p-value. The arithmetic was right. The question was whether the
design supported the claim.

## What changed when we fixed the design

We rebuilt the analysis with four corrections. We projected coordinates properly.
We restricted matching to nearby comparison sites. We removed cameras whose
catchments overlapped an already-live camera, because a crash inside two
catchments is not two independent observations. And we started the analysis
window when crash reporting was reliable, rather than treating sparse early
months as genuine zeros.

The revised estimate is **−2.6%**, with a 95% confidence interval from **−11.5%
to +7.2%** and a p-value of about 0.59, across **20 cameras**. The cohort-based
2×2 estimator, which is more robust to staggered adoption, gives a similar
picture: about −0.1 crashes per site-month, with an interval that includes zero.

That is not a null result in the ordinary sense. It is an imprecise result. The
data are consistent with a meaningful reduction, no effect, or a modest increase.
The honest conclusion is that the public data cannot resolve the question at the
precision a policy debate usually demands.

There is a second boundary that matters. The Chicago portal states that crash
data are not citywide until September 2017, but the published analysis and the
forecasting models begin earlier. When we require 24 citywide pre-months and 36
post-months, only 15 cameras remain. That design gives **−8.6%** with a 95%
interval from **−16.9% to +0.4%** and a p-value of 0.061. Its geographic placebo
p-value is 0.25. It is a different sample—mostly the 2023 cohort—not a replacement
for the −2.6% estimate. Reporting all three estimates is the honest approach.

## Why the answer moved

The largest design lever was geographic matching. When comparison sites are drawn
from the whole city, the estimate is around −7%. When they are restricted to
nearby locations, it is around −3%. This is not a statistical trick; it reflects a
different question. A citywide comparison absorbs some city-wide trends but not
local ones. A nearby comparison absorbs local trends but leaves fewer eligible
sites.

The second lever was overlap. Among all 209 cameras, 36% of attributed crashes
fell inside two or more 250-metre catchments. Keeping overlapping cameras makes
the sample look larger than it is and counts some crashes more than once. Removing
them costs treated units but produces a cleaner experiment.

A third lever was the arbitrary phase of the comparison lattice. Moving the 500 m
grid changes which sites are available. Under citywide matching, the estimate
ranged from roughly −5% to −8.5% across four grid phases. Under local matching,
the range narrowed to about −3.1% to −1.2%. That sensitivity is not a reason to
choose the most favourable phase; it is a reason to report the range.

## How this compares with published work

Chicago has been studied before. Tilahun (2023) used an empirical Bayes
before-after design at 101 camera locations and reported a 12% reduction in fatal
and injury crashes for the 2013–2014 rollout, with a 15% reduction in fatality
and severe-injury crashes. A 2022 University of Illinois Chicago report by Sutton
and Tilahun examined the equity and efficacy of the program.

New York City evidence is also relevant. Stagoff-Belfort, Ben-Menachem and Beck
(2025) exploited the staggered rollout of roughly 2,000 cameras between 2014 and
2023 and estimated reductions of about 5% per month for collisions and 2.5% per
month for injuries, cumulatively about 30% and 16% over the first seven months.

Our revised estimate is smaller. That does not disprove those studies. They
examine different programs, periods, outcomes and designs. Tilahun's outcome is
fatal and injury crashes; ours is all reported crashes. New York's program is
much larger and its design uses a different comparison structure. What our study
adds is a transparent demonstration that a Chicago estimate can move by several
percentage points when defensible design choices change. The coverage-consistent
estimate is more negative, but its 15-camera sample and placebo distribution do
not justify treating it as a stronger causal result.

## What we learned about causal inference

Three lessons stayed with us.

First, **reproducibility is not validity**. A result can reproduce exactly and
still depend on an undocumented design choice. The original notebook was
reproducible; the problem was that its implementation and its description had
drifted apart.

Second, **spatial design is a causal decision, not a preprocessing step**. The
choice of comparison sites changes the estimand. "Nearby" and "citywide" are not
interchangeable, and the difference should be reported rather than hidden in a
default.

Third, **an imprecise estimate is information**. The revised interval rules out
neither a meaningful benefit nor a modest harm. That is a legitimate finding. It
says the data are not strong enough for a confident claim, not that the policy
has no effect.

## What would change our mind

Better measurement would help more than another model. Traffic exposure,
camera uptime, warning periods, enforcement hours and a stable crash-reporting
regime would each improve identification. So would a larger independent treated
sample and a research design that can separate camera effects from local
transportation investments.

We are not arguing that cameras are ineffective. We are arguing that the current
public record does not support a precise causal claim in either direction.

## About this work

This is a co-authored research release by **Saketh Donepudi** and **Amrutha
Kakumani**. The original Chicago study, notebook and README were created by
Amrutha Kakumani and contributors. The causal audit, revised design, forecasting
system and release were developed in the same repository. Affiliations and
individual contribution roles are intentionally not asserted here; they require
confirmation from both authors.

The working paper, audit, figures and code are available in the repository:
https://github.com/amruthakakumani/Do-Chicago-speed-cameras-reduce-crashes-

The paper is a working paper, not peer reviewed. The analysis uses public data
from the Chicago Data Portal. No institutional endorsement is claimed.
