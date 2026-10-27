# Dashboard audit

**Audit date:** 2026-10-27
**Scope:** `dashboard/app.py`, local Streamlit run, precomputed artifacts.
**Decision:** PASS for local research use. Public deployment requires authorization
and a verified URL.

---

## 1. Pages and purpose

| Sidebar page | Purpose | Primary data |
|---|---|---|
| Research overview | Research question, headline results, sample definition, caveats | `results/estimates.json`, `results/ml_v2/deployment.json` |
| Chicago map | Camera catchments, comparison sites and treatment radius | `results/sites.parquet` |
| Causal inference | Original versus revised estimates, event study, diagnostics | `results/estimates.json`, `results/tables/` |
| Crash forecasting | Map, predictions, intervals, errors, camera drill-down, SHAP, V1/V2 comparison | `results/ml/v1/`, `results/ml_v2/` |
| Robustness & sensitivity | Specification curve, placebos, lattice phase and era splits | `results/tables/sensitivity.csv`, `results/estimates.json` |
| Coverage & reconstruction | Reporting break, AADT audit, common-row models, road task and causal sensitivity | `results/bottom_up/` |
| Methods & limitations | Estimands, assumptions, model cards, provenance and known failure modes | `docs/` artifacts and JSON metadata |

---

## 2. Accuracy checks

| Check | Result | Status |
|---|---|---|
| Causal headline on dashboard matches `results/estimates.json` | −2.6% [−11.5%, +7.2%], 20 cameras | PASS |
| Original estimate is labelled as the original notebook result | −7.3% [−13.6%, −0.6%], 33 cameras | PASS |
| V1/V2 metrics match saved JSON | MAE 1.568 / 1.564; deviance 1.223 / 1.218 | PASS |
| Deployed model is V1 | `deployment.json` selects `v1` | PASS |
| V2 is labelled a research candidate | deployment note and comparison table | PASS |
| Test period is labelled retrospective where reused | note under V1/V2 table | PASS |
| Conformal coverage uses the selected interval method | normalized coverage 90.6% | PASS |
| Coverage page separates reporting, prediction and causal claims | explicit labels and warnings | PASS |
| Group-level causal effect is not shown per camera | explicit note on camera drill-down | PASS |
| Prediction intervals are not called causal intervals | map note, drill-down note, methods page | PASS |
| No data or model is retrained on reload | cached readers only; no training calls | PASS |
| Missing artifacts produce a clear message | guarded loaders and empty-state page | PASS |

---

## 3. Presentation checks

- The Chicago map remains the largest visual element on the crash-forecasting
  page; layers can be toggled instead of forcing every overlay at once.
- Observed crashes, predictions, intervals and errors have separate layers and
  separate colour keys. Blue/red error colours are explained in text.
- The map uses a 500 m aggregated crash grid, not individual crash points.
- Filters cover year, month, location type, region cluster, severity and camera.
- The camera drill-down shows camera metadata, the latest forecast, the 90%
  interval, descriptive camera-level error and SHAP contributions.
- The causal section is separate from the forecasting section; the dashboard
  does not imply that SHAP values are causal effects.
- The methods page states the estimand, key assumptions, model decision and
  limitations.

## 4. Performance checks

| Check | Result | Status |
|---|---|---|
| Data loaders cached with `st.cache_data` | yes | PASS |
| Observed density pre-aggregated | 500 m grid | PASS |
| Map point count | 1,355 locations | PASS |
| SHAP table filtered to selected camera/month before plotting | yes | PASS |
| Page render under AppTest | all seven pages complete | PASS |
| Local HTTP response | 200 | PASS |

## 5. Verification performed

```text
streamlit.testing.v1.AppTest: all seven pages, zero exceptions
streamlit run dashboard/app.py --server.headless true: HTTP 200
pytest: 67 collected, 66 passed, 1 skipped (V2 binary artefact intentionally removed)
ruff: clean
```

The `st.map` colour-by-role call was replaced with an explicit colour column so
the Chicago map renders correctly on the installed Streamlit version.

## 6. Known limitations

1. **No public deployment.** The dashboard runs locally. No URL is provided or
   claimed.
2. **Network basemap.** The forecasting map uses a CARTO basemap; an offline
   environment may show a blank basemap while the data layers remain available.
3. **No mobile-specific test.** Layout uses Streamlit columns and may stack
   differently on narrow screens.
4. **Descriptive camera errors.** Camera-level MAE is not a camera-specific
   causal effect.
5. **Approximate intervals.** Conditional coverage is imperfect, especially in
   the highest-risk decile.
6. **Artifact dependency.** The dashboard requires the committed `results/` and
   `models/` artifacts; it does not download data on load.

## 7. Decision

The dashboard is accurate, reproducible and appropriately caveated for local
research use. Public deployment is blocked on author authorization and a
verified deployment URL.
