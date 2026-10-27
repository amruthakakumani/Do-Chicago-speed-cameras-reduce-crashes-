# Release cleanup audit

**Date:** 2026-10-28  
**Branch:** `release/final-cleanup`  
**Base:** `main` at `ebc8760` (unchanged; no push, merge, release or deployment)  
**Recovery snapshot:** `.context/pre_final_cleanup_20261028_000105.tar.gz` (119 MB, taken before destructive cleanup)

---

## 1. Classification

| Class | Files / areas | Decision |
|---|---|---|
| Essential | `README.md`, `CITATION.cff`, `LICENSE`, `pyproject.toml`, `requirements.txt`, `.gitignore`, original root notebook | Retained |
| Causal reproduction | `src/chicago_cameras/` causal modules, `config/`, `scripts/download_data.py`, `scripts/verify_replication.py`, causal tables/figures and `results/estimates.json` | Retained |
| Forecasting reproduction | V1 training/evaluation code, `models/v1/`, `results/ml/`, V2 code and evaluation metrics | Retained; redundant V2 binary checkpoint removed |
| Dashboard deployment | `dashboard/app.py`, `results/estimates.json`, `results/sites.parquet`, `results/ml/`, `results/ml_v2/deployment.json`, `results/ml_v2/selected_predictions.parquet`, `results/ml_v2/metrics.json`, `results/bottom_up/` coverage artifacts | Retained |
| Historical but scientifically important | `docs/AUDIT.md`, `docs/BOTTOM_UP_AUDIT.md`, `docs/GEOGRAPHIC_DESIGN_COMPARISON.md`, model-selection and negative-result documents, original notebook | Retained |
| Redundant | Duplicate manuscript exports, LinkedIn promotional copy, obsolete draft/handoff documents, root duplicate of V1 checkpoint, experimental bottom-up checkpoints | Removed |
| Obsolete | Old generated paper build, removed-document links, superseded prediction intermediates and experimental dashboard page | Removed |
| Safe to remove | Development caches, ignored local environment/data files, and `.context/` scratch space | Not committed |

---

## 2. Removed content

### Promotional and unsupported public material

- Entire `linkedin/` directory:
  - `article_a_causal.md`
  - `article_b_ml.md`
  - `launch_package.md`

### Paper duplicates and drafts

- `paper/chicago_speed_cameras_research.md`
- `paper/chicago_speed_cameras_research.docx`
- `paper/chicago_speed_cameras_research.pdf` (old path)
- `paper/PROPOSED_REVISION.md`
- `paper/bottom_up_reconstruction.md`, `.pdf`, `.docx`
- `paper/build/`

The approved PDF now lives only at
`paper/Chicago_Speed_Cameras_Research.pdf`; the source is retained internally at
`paper/source/chicago_speed_cameras_research.md`.

### Duplicate or superseded documentation

Removed: `BOTTOM_UP_READINESS.md`, `FINAL_PR_DRAFT.md`, `FINAL_PUBLICATION_AUDIT.md`,
`GITHUB_METADATA.md`, `LIGHTGBM_METHODOLOGY.md`, `LIGHTGBM_V2_METHODS.md`,
`ORIGINAL_README.md`, `PREDICTION_INTERVALS.md`, `PREDICTION_RESULTS.md`,
`PUBLICATION_AUDIT.md`, `PUBLICATION_READINESS.md`, `RELEASE_NOTES_DRAFT.md`,
`REPORT.md`, `V1_V2_COMPARISON.md`, `V2_BASELINE_AUDIT.md`,
`V2_FEATURE_AVAILABILITY.md`, and `V2_MODEL_CARD.md`.

### Redundant checkpoints and intermediates

- `models/bottom_up/` experimental checkpoints (10 files).
- Root duplicate V1 checkpoint and metadata; the canonical deployed artifact is
  now `models/v1/`.
- V2 binary checkpoint and metadata; V2 evaluation metrics remain.
- Large prediction/SHAP intermediates not needed by the final dashboard:
  statistical-experiment prediction files, bottom-up prediction Parquet files,
  V2 full prediction/SHAP/recursive Parquet files and V2 figure exports.
- `scripts/build_prediction_report.py`, which generated a removed documentation
  file and updated the old README.

### Dashboard page

Removed the experimental **Model experiment** Streamlit page. The scientific
negative-result record remains in `docs/MODEL_EXPERIMENT_BASELINE.md`,
`docs/MODEL_EXPERIMENT_RESULTS.md`, `docs/MODEL_SELECTION_DECISION.md`,
`docs/HYBRID_MODEL_METHODS.md`, `docs/STATISTICAL_DIAGNOSTICS.md` and
`results/statistical_hybrid/`.

The dashboard is back to seven pages.

---

## 3. Reference and dependency checks before removal

- Inspected every remaining reference to removed paths with repository-wide
  search.
- Removed the root V1 checkpoint only after updating the training, evaluation and
  model-card defaults to `models/v1/`.
- Verified that dashboard loaders require the retained V1 predictions/SHAP and
  the V1/V2 deployment/metrics files.
- Kept compact negative findings and model-selection justification rather than
  erasing experiments from the record.
- No Git history was rewritten; no `reset --hard`, `clean -fd` or force-push was
  used.
- Raw data, `.venv/`, and `.context/` remain ignored and are not part of the
  publishable payload.

---

## 4. Result

The final tree is a single-paper release with:

- one public PDF and one internal reproducible manuscript source;
- one deployed forecasting artifact (`models/v1/`);
- seven dashboard pages;
- causal, forecasting, coverage and negative-result documentation;
- an MIT `LICENSE` and updated citation metadata;
- no LinkedIn material or experimental dashboard page.

All final checks and outstanding approvals are recorded in
[`FINAL_RELEASE_AUDIT.md`](FINAL_RELEASE_AUDIT.md).
