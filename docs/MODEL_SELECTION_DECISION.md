# Model selection decision

**Decision:** retain the deployed V1 forecasting model.
**Experimental candidates:** optimized LightGBM, Negative Binomial, convex hybrid.
**Status:** experiment only; no model was deployed.

---

## 1. Predeclared selection rule

Before evaluating the candidates, a candidate had to satisfy all of the following
relative to the coverage-consistent V1 reference:

1. At least **1%** improvement in temporal-validation Poisson deviance.
2. No greater than **2%** regression in overall test MAE.
3. No material deterioration in high-crash performance.
4. Acceptable geographic generalization.
5. No serious calibration regression.
6. Reasonable runtime and operational complexity.

The thresholds were not changed after seeing test results.

## 2. Results against the rule

| Candidate | Validation deviance improvement | High-crash MAE change | Geographic deviance change | Meets 1%? | Safeguards? |
|---|---:|---:|---:|---|---|
| Optimized LightGBM | **+0.483%** | −0.57% | +0.08% | No | Yes |
| Hybrid | **+0.491%** | −0.51% | +0.03% | No | Yes |
| Negative Binomial | −1.76% | +3.81% | +1.48% | No | No |

No candidate reaches the 1% improvement threshold. The hybrid is the best
validation candidate by a negligible margin; its test deviance is 1.21575 versus
1.21625 for optimized LightGBM, and its errors correlate 0.976 with LightGBM.

## 3. Why V1 remains deployed

- The 1% validation rule is not met by any candidate.
- The hybrid's gain over optimized LightGBM is 0.008% OOF and about 0.04% test;
  this is not operationally meaningful.
- NB is worse in overall and high-crash accuracy.
- Geographic validation is acceptable for LightGBM/hybrid but not evidence of a
  new production model.
- The hybrid requires two models and a stored weight; complexity is not justified.
- V1 is already deployed and its original test metrics are preserved.

The decision is robust to the small metric differences: the experiment succeeds
by showing that the simpler deployed model remains the defensible choice.

## 4. What remains experimental

- Optimized LightGBM (coverage-tuned) is retained as a research candidate.
- Negative Binomial is retained as a diagnostic/benchmark model.
- Hybrid is retained as an evaluated ensemble with weight 0.943952, but is not
  deployed.
- The causal manuscript and its estimates are unchanged by this experiment.

## 5. Deployment rule

No model is deployed automatically. A future deployment requires:

1. A new untouched test period or external validation.
2. A candidate meeting the 1% rule and high-risk/calibration safeguards.
3. Co-author approval.
4. A versioned model card and dashboard update.

Artifacts: `results/statistical_hybrid/summary.json`,
`results/statistical_hybrid/hybrid_weight_curve.csv`,
`results/statistical_hybrid/inference_timing.json`.


## 6. Publication implications

- The causal manuscript, README and causal estimates are unchanged by this
  experiment.
- The experiment is a supplementary negative result: the deployed V1 remains the
  defensible forecast; the optimized LightGBM/reconstructed hybrid do not meet
  the 1% rule.
- The experiment documents and diagnostic figures are retained as a compact
  research record without changing the causal paper.
- No model was deployed, released or published.
