# Documentation index

| Document | What it answers |
|---|---|
| [`METHODOLOGY.md`](METHODOLOGY.md) | Estimand, identifying assumptions, geospatial design and estimators. |
| [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) | Fields, derivations, exclusions and provenance. |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Intended use, training data, metrics, calibration, risks and limitations of the deployed V1 forecaster. |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Installation, data acquisition, exact commands, expected outputs and runtime. |
| [`AUDIT.md`](AUDIT.md) | Original-study audit: implementation, replication and the reason for each design correction. |
| [`BOTTOM_UP_AUDIT.md`](BOTTOM_UP_AUDIT.md) | Reporting-coverage, exposure, road-assignment and reconstruction findings. |
| [`FEATURE_AVAILABILITY_LEDGER.md`](FEATURE_AVAILABILITY_LEDGER.md) | When every forecasting feature becomes known; leakage rules and excluded AADT. |
| [`GEOGRAPHIC_DESIGN_COMPARISON.md`](GEOGRAPHIC_DESIGN_COMPARISON.md) | Camera-zone versus road-segment geography and why their errors are not comparable. |
| [`STATISTICAL_DIAGNOSTICS.md`](STATISTICAL_DIAGNOSTICS.md) | Count-model diagnostics, overdispersion, zero inflation and residual checks. |
| [`MODEL_EXPERIMENT_BASELINE.md`](MODEL_EXPERIMENT_BASELINE.md) | Coverage-consistent sample, common rows and experiment baselines. |
| [`MODEL_EXPERIMENT_RESULTS.md`](MODEL_EXPERIMENT_RESULTS.md) | Optimized LightGBM, Negative Binomial and hybrid comparison; negative result. |
| [`MODEL_SELECTION_DECISION.md`](MODEL_SELECTION_DECISION.md) | Predeclared 1% rule, selection outcome and why V1 remains deployed. |
| [`HYBRID_MODEL_METHODS.md`](HYBRID_MODEL_METHODS.md) | Convex LightGBM/Negative Binomial construction and validation protocol. |
| [`DASHBOARD_AUDIT.md`](DASHBOARD_AUDIT.md) | Dashboard page audit, accuracy checks, presentation and performance. |
| [`DASHBOARD_DEPLOYMENT.md`](DASHBOARD_DEPLOYMENT.md) | Local run instructions and deployment requirements; no live URL claimed. |
| [`AUTHOR_CONTRIBUTIONS.md`](AUTHOR_CONTRIBUTIONS.md) | Authorship and contribution record. |
| [`RELEASE_CLEANUP_AUDIT.md`](RELEASE_CLEANUP_AUDIT.md) | File classification and the cleanup performed for this release. |
| [`FINAL_RELEASE_AUDIT.md`](FINAL_RELEASE_AUDIT.md) | Final scientific, PDF, dashboard, test, dependency and repository audit. |

The original notebook is preserved unchanged at the repository root:
`Do_Chicago_speed_cameras_reduce_crashes_.ipynb`.
