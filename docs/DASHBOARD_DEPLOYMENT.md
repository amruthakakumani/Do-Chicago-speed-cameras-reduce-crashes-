# Dashboard deployment notes

The Streamlit dashboard is a reader of precomputed, validated artifacts. It does
not train models, call the network, or expose credentials.

## Local run

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

The app reads:

- `results/estimates.json` — causal estimates, diagnostics, placebos.
- `results/analysis_panel.parquet` — event-study/trends.
- `results/sites.parquet` — camera and comparison locations.
- `results/ml/v1/predictions.parquet` — deployed V1 forecasts.
- `results/ml/shap_values.parquet` — deployed V1 SHAP explanations.
- `results/ml_v2/deployment.json` — selected model and artifact paths.
- `results/ml_v2/selected_predictions.parquet` — test predictions and conformal intervals.
- `results/ml_v2/metrics.json` — V1/V2 comparison, interval coverage.
- `results/ml/crash_grid_monthly.parquet` — observed crash density.

## Streamlit Community Cloud (authorization required)

1. Push the repository to GitHub (requires author approval; not done in this
   release).
2. Create a Streamlit app pointing at `dashboard/app.py`.
3. Use Python 3.11 and install the committed `requirements.txt`.
4. Ensure the committed `results/` and `models/v1/` artifacts are available; do
   not commit raw data.
5. Set the app to read-only; no secrets are required.

No live URL is provided because the dashboard has not been deployed. Do not
publish a URL until the deployment is verified and both authors approve.

## Performance

- Data loaders are cached with `st.cache_data`.
- Observed crash density is pre-aggregated to a 500 m grid; individual crashes
  are not rendered.
- Prediction and SHAP tables are filtered before rendering.
- The map uses 1,355 points and aggregated columns, not millions of records.

## Accessibility and integrity

- Layers and filters have text labels; color is not the only encoding.
- Causal, predicted and observed quantities are in separate sections.
- Prediction intervals are labelled approximate and are not described as causal
  confidence intervals.
