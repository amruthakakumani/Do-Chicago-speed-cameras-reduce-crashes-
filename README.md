# Do Speed Cameras Make Chicago Safer?

### A Geospatial Causal Inference and Machine Learning Study

**Saketh Donepudi · Amrutha Kakumani**  
*Independent Research | October 2026*

An empirical investigation into the effectiveness of Chicago's automated speed enforcement program, combining causal inference, geospatial analytics, statistical modeling, and machine learning.

**[Read the Research Paper (PDF)](paper/Chicago_Speed_Cameras_Research.pdf)**

---

## Overview

Do speed cameras actually reduce traffic crashes, or do observed reductions reflect underlying trends, geographic differences, and statistical artifacts?

This study investigates that question using more than **1 million Chicago traffic crash records**, speed-camera locations, historical enforcement information, and geographic data.

We combine quasi-experimental causal inference methods with machine-learning crash forecasting to examine both the estimated impact of speed cameras and the predictability of traffic crashes.

The research emphasizes **methodological transparency, reproducibility, and robustness**, rather than statistical significance alone.

## Key Findings

### Causal Inference

| Research Design | Cameras | Estimated Effect | 95% Confidence Interval |
|---|---:|---:|---|
| Original Poisson DiD | 33 | −7.3% | [−13.6%, −0.6%] |
| Revised Geographic Design | 20 | −2.6% | [−11.5%, +7.2%] |
| Coverage-Consistent Design | 15 | −8.6% | [−16.9%, +0.4%] |

The original analysis suggested a statistically significant reduction in crashes.

However, correcting geographic matching, overlapping camera catchments, and reporting-coverage assumptions changed the estimated effects and their statistical uncertainty.

**The central finding is design sensitivity:** estimated effects depend substantially on geographic matching, reporting coverage, and treatment-cohort selection.

The three specifications use different eligible samples and should not be interpreted as interchangeable estimates of the same treatment effect.

Neither revised estimate establishes a statistically significant reduction at the conventional 5% threshold.

### Crash Forecasting

We developed a LightGBM model to forecast monthly crash counts using historical, temporal, environmental, and geographic features.

| Metric | LightGBM V1 |
|---|---:|
| R² | 0.6611 |
| Mean Absolute Error | 1.5683 |
| RMSE | 2.0931 |
| Mean Poisson Deviance | 1.2233 |

Additional feature engineering, Negative Binomial regression, and hybrid forecasting were evaluated.

Their improvements were insufficient to justify replacing the original LightGBM model under the predefined selection criteria.

**Final forecasting model: LightGBM V1.**

Forecasting performance does not establish the causal effectiveness of speed cameras.

---

## Research Methodology

### 1. Data Engineering

- Chicago Traffic Crashes dataset
- Speed-camera locations and enforcement records
- Temporal aggregation and data cleaning
- Historical reporting-coverage validation
- Geographic coordinate transformation

### 2. Geospatial Analysis

- 250-metre camera catchments
- Treatment and comparison-site matching
- Spatial overlap detection
- Road-network analysis
- Geographic sensitivity testing

### 3. Causal Inference

- Difference-in-Differences (DiD)
- Conditional Poisson Fixed Effects
- Cohort-Based Treatment Effects
- Event Studies
- Synthetic Control
- Placebo and Robustness Tests

### 4. Machine Learning

- LightGBM with Poisson objective
- Temporal feature engineering
- Hyperparameter optimization using Optuna
- Rolling-origin validation
- Geographic holdout validation
- SHAP model interpretation
- Conformal prediction intervals

---

## Interactive Research Dashboard

The repository includes a locally executable Streamlit research dashboard for exploring:

- Chicago speed-camera locations
- Geographic treatment and comparison areas
- Historical crash density
- Causal inference results
- LightGBM crash predictions
- Prediction uncertainty
- Reporting-coverage diagnostics

The dashboard is designed for **local research exploration** and is not publicly hosted.

### Run Locally

Clone the repository:

```bash
git clone https://github.com/amruthakakumani/chicago-speed-cameras-causal-inference.git

cd chicago-speed-cameras-causal-inference
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project:

```bash
pip install -e ".[dashboard]"
```

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

The application will normally open at:

`http://localhost:8501`

**Note:** Some analyses and dashboard components require generated datasets and model artifacts. Consult the repository's reproducibility documentation for the necessary data-processing steps.

---

## Research Paper

**Do Speed Cameras Make Chicago Safer? A Geospatial Causal Inference and Machine Learning Study**

Saketh Donepudi & Amrutha Kakumani (2026)

**[View Full Research Paper — PDF](paper/Chicago_Speed_Cameras_Research.pdf)**

The working paper includes:

- Research background and related literature
- Data sources and geographic methodology
- Causal identification strategy
- Statistical models and treatment-effect estimates
- Robustness and sensitivity analysis
- Machine-learning forecasting results
- Prediction uncertainty
- Limitations and conclusions
- References and reproducibility appendix

---

## Reproducibility

The repository provides Python implementations, statistical analysis scripts, configuration files, research figures, and automated tests.

Public datasets are obtained from the Chicago Data Portal and other documented sources.

Large source datasets are not stored directly in the repository.

Refer to:

**[Reproducibility Documentation](docs/REPRODUCIBILITY.md)**

**[Research Methodology](docs/METHODOLOGY.md)**

**[Model Documentation](docs/MODEL_CARD.md)**

---

## Limitations

This is an observational research study.

Camera placement is not random, and estimated causal effects remain sensitive to comparison-site selection, geographic assumptions, and reporting coverage.

Citywide crash reporting begins in September 2017, limiting the comparability of earlier observations.

Historical traffic-exposure measurements are insufficient for a consistent vehicle-adjusted crash-rate analysis.

The causal estimates have limited precision, and forecasting uncertainty is greater in high-risk locations.

These limitations are discussed in detail in the research paper.

---

## Authors

**Saketh Donepudi**  
Co-author

**Amrutha Kakumani**  
Co-author

*Independent Research, 2026*

---

## Citation

If you use this research, please cite the working paper and reference the repository's `CITATION.cff` file.

This research is an independent working paper and has not undergone formal peer review.

## License

MIT License, subject to final co-author approval of the repository's licensing terms.
