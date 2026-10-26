# Reproducibility

This project is designed to be rebuilt from the public Chicago Data Portal with one command per
stage. No Colab, Google Drive, or private credentials are required.

---

## 1. Requirements

- Python **3.11+**
- ~2 GB free disk for raw and processed data
- Internet access for the first download and the Open-Meteo weather archive
- macOS only: LightGBM needs the OpenMP runtime (`brew install libomp`)

Dependencies are declared in `pyproject.toml` (runtime, `dev`, `dashboard` extras). They are not
pinned to exact versions; the reference environment used Python 3.11.9 with pandas 2.3, numpy 2.x,
scipy 1.17, statsmodels 0.15, scikit-learn 1.9, pyarrow, pyproj 3.7, shapely 2.2, matplotlib 3.10,
streamlit 1.66.

## 2. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,dashboard,ml]"
```

## 3. Get the data

```bash
python scripts/download_data.py --verbose
```

This writes `data/raw/{cameras,crashes,violations}.csv.gz` and `data/raw/provenance.json` with:

- the exact Socrata endpoint and query (`$select`, `$where`, `$order`, page size);
- acquisition and completion timestamps (UTC);
- rows per page and total rows;
- SHA-256 of the uncompressed payload;
- portal dataset id and human URL.

If the portal is unavailable, the script exits with an error. **It never substitutes synthetic
data.** The analysis will refuse to run without the three raw files.

## 4. Run the analysis

```bash
python -m chicago_cameras.pipeline --config config/analysis.yaml
```

Development run with fewer bootstrap/placebo draws (results are indicative, not final):

```bash
python -m chicago_cameras.pipeline --quick
```

Re-download and re-clean from scratch:

```bash
python -m chicago_cameras.pipeline --refresh
```

Runtime on a 2021 MacBook Pro (M1): download ≈ 5 min; cleaning/projection ≈ 40 s; causal analysis,
placebos and figures ≈ 4–20 min depending on placebo count; LightGBM tuning/evaluation ≈ 15–20 min. The processed Parquet cache is reused on subsequent runs unless
`--refresh` is passed or the data fingerprint changes.

### Stages and caches

| Stage | Output | Cache |
|---|---|---|
| Acquisition | `data/raw/*.csv.gz`, `provenance.json` | Reused unless `--refresh` |
| Cleaning / projection | `data/processed/*.parquet`, `quality_report.json` | Reused if config fingerprint matches |
| Design / estimation | `results/*.json`, `results/tables/*.csv` | Recomputed each run |
| Figures | `results/figures/*.png|svg` | Recomputed each run |

## 4b. Train and evaluate the LightGBM forecaster

The predictive layer is independent of the causal analysis and is optional for
reproducing the causal results.

```bash
python scripts/train_lightgbm.py --trials 40 --timeout 900
python scripts/evaluate_lightgbm.py
python scripts/build_prediction_report.py
```

- The first command builds the location-month dataset, runs Optuna on
  rolling-origin folds, refits on all pre-test months, evaluates once on the
  untouched test period, computes SHAP values and writes `models/` and
  `results/ml/`.
- Runtime on the reference machine: ≈ 15 minutes for tuning (900 s budget) plus
  ≈ 2 minutes for the final fit, predictions, SHAP and figures.
- `--reuse-tuning` refits with saved hyperparameters; `--quick` runs five trials
  for development. `--trials` and `--timeout` are explicit budgets.
- The weather archive is cached at `data/raw/weather_daily.csv.gz` with
  provenance. If it is unavailable, the weather features are dropped and the
  limitation is recorded; no weather values are invented.
- `scripts/evaluate_lightgbm.py` never retrains; it re-evaluates the saved model
  and can be used to verify reproducibility.
- `scripts/build_prediction_report.py` generates `docs/PREDICTION_RESULTS.md`
  and fills the metrics tables in the model card and README from
  `results/ml/metrics.json`.

## 4c. V2 prediction, intervals and geographic validation

```bash
python scripts/train_lightgbm_v2.py --trials 40 --timeout 900
python scripts/evaluate_recursive_v2.py
```

- Downloads US Census TIGER/Line 2024 EDGES for Cook and DuPage counties on
  first use; raw archives and SHA-256 are stored in `data/raw/tiger/`.
- Builds 84 features (V1 + road, neighbourhood, citywide, long-history and
  climate normals), tunes on pre-test rolling folds only, evaluates the
  retrospective test period, calibrates split-conformal intervals on out-of-fold
  predictions, runs geographic block holdouts and writes `results/ml_v2/`.
- Runtime on the reference machine: ≈ 25 minutes (tuning budget 900 s plus
  folds, intervals, geographic models and SHAP).
- `scripts/evaluate_recursive_v2.py` runs a genuine 12-month recursive path from
  2025-08 and writes `results/ml_v2/recursive_metrics.json`.
- The pre-declared selection rule is in `docs/V2_BASELINE_AUDIT.md`; V1 remains
  deployed if V2 does not meet it.

## 5. Verify the original study

```bash
python scripts/verify_replication.py
```

Expected output (current portal snapshot):

```
crashes kept          1,081,474
camera sample         33 of 209
panel rows            11,880
pre means             twins 4.298 | cameras 4.403
OLS      -0.3557 [-0.7110, -0.0005]  p=0.0497
Poisson  -7.3% [-13.6%, -0.6%]  p=0.0342
```

These match the notebook's saved outputs to four decimal places. The script uses the original flat
projection and citywide matching on purpose; it is an audit tool, not the upgraded design.

## 6. Tests

```bash
python -m pytest -q
```

63 tests cover distance/projection, counting zeros, duplicate attribution, overlap detection,
cleaning and the coverage guard, coverage-regime classification, panel completeness, matching
eligibility and the camera-specific pre-window regression test, estimator recovery of known effects,
synthetic-control exact recovery and poor-fit detection, an end-to-end synthetic design, exposure
chronology, road assignment uniqueness/ambiguity, feature-group separation, and the ML layer
(lag/rolling leakage, temporal splits, folds, schema checks, serialization, reproducible
non-negative predictions),
V2 temporal/road/neighbourhood features, geographic block isolation with a 1 km buffer, conformal
interval endpoints and coverage, recursive forecasts that ignore future actuals, model-version
compatibility and dashboard-artifact consistency.

CI (`.github/workflows/tests.yml`) installs the package and runs the tests on every push and pull
request. **CI does not download the crash dataset**, so it stays fast and offline-safe.

## 7. Determinism

- All stochastic steps take an explicit seed from `config/analysis.yaml`
  (`inference.seed = 20261024`): synthetic-control placebos, cohort bootstrap, wild cluster
  bootstrap, geographic placebos, k-means, and bootstrap CIs.
- Data processing is deterministic: no random sampling, no imputation, no row-order dependence.
- The lattice is anchored to the crash bounding box; its phase is a declared design parameter and is
  varied in sensitivity.
- Sorting is explicit before every group operation that affects results.

Minor numerical differences across BLAS versions are expected in the third or fourth decimal.

## 8. Data provenance and versioning

`data/raw/provenance.json` records the snapshot. To reproduce a specific analysis, archive the raw
gzip files (checksums are in provenance) together with `config/analysis.yaml`. The portal is a live
dataset; re-downloading later will change the crash counts slightly even though the study period is
fixed.

The raw crash file is ~106 MB gzip and is **not** committed. Processed Parquet is also ignored.

## 9. Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard only reads precomputed artefacts (`results/estimates.json`, `results/tables/*.csv`,
`results/figures/*`). It never refits a model. If results are absent it shows a "results not
generated" state with the command to run.

## 10. Reference notebook

`notebooks/research_analysis.ipynb` is the upgraded, clearly labelled notebook. It loads the
precomputed results and figures, and contains an optional cell to run the pipeline. The original
notebook is preserved untouched at the repository root.

## 11. Known environment caveats

- The `pymc` optional model from the original notebook is not installed by default; it is not part of
  the reported results.
- The dashboard requires the `dashboard` extra.
- On Linux CI, matplotlib uses the non-interactive `Agg` backend (set in `visualization.py`).
- The downloader honours HTTP 429 with `Retry-After`; a very aggressive schedule can still be
  throttled, in which case rerun after a few minutes. Partial raw files are never written: the file
  is written only after a complete download.


---

## Formatted paper build

The scientific Markdown source remains `paper/chicago_speed_cameras_research.md`.
The PDF/DOCX front matter is generated without editing that source:

```bash
python scripts/make_cover_graphic.py
python scripts/build_publication_docs.py
```

Requirements: Pandoc 3.12, Typst 0.15, LibreOffice/`soffice` for the DOCX preview,
`python-docx` and `pypdf`. The script creates a temporary `paper/build/` body,
applies the cover/author/TOC template, compiles the PDF, post-processes the DOCX
with a Word TOC field and sectioned page numbering, and validates the result.
