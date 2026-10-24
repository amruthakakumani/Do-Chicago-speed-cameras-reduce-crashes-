# Generated results

Everything in this directory is produced by the pipeline and should not be edited by hand:

```bash
python -m chicago_cameras.pipeline --config config/analysis.yaml
```

| Path | Contents |
|---|---|
| `estimates.json` | Machine-readable bundle used by the dashboard and notebook |
| `provenance.json` | Data-quality and acquisition record for this run |
| `analysis_panel.parquet` | Site-month panel used by the estimators |
| `sites.parquet` | Camera and comparison-site locations (WGS84) |
| `tables/main_estimates.csv` | Naive, original baseline, FE OLS, conditional Poisson, cohort 2×2, replication |
| `tables/replication_estimates.csv` | Original-sample replication with corrected projection |
| `tables/event_study.csv` | Quarterly event-study coefficients |
| `tables/cohort_event_effects.csv` | Per-event-month cohort 2×2 effects |
| `tables/balance.csv` | Pre-period balance |
| `tables/subgroups.csv` | Exploratory heterogeneity with BH-adjusted p-values |
| `tables/severity.csv` | Injury and severe/fatal outcomes |
| `tables/sensitivity.csv` | Every robustness and falsification specification |
| `tables/synthetic_control.csv` | Synthetic-control fit and pseudo p-values |
| `tables/sc_leave_one_out.csv` | Donor influence |
| `tables/sc_in_time_placebo.csv` | Fake treatment dates |
| `tables/sc_placebo_records.csv` | Placebo-in-space draws |
| `tables/placebo_geography.csv` | Random-location placebo draws |
| `tables/violations_curve.csv` | Mechanism check |
| `figures/*.png`, `figures/*.svg` | Ten publication figures |

Figures are written in both PNG (220 dpi) and SVG. The dashboard only reads these artefacts; it
never refits a model.
