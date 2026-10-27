# Final release audit

**Date:** 2026-10-28  
**Branch:** `release/final-cleanup`  
**Base:** `main` at `ebc8760`; no commit, push, merge, release or deployment has been performed  
**Cleanup record:** [`RELEASE_CLEANUP_AUDIT.md`](RELEASE_CLEANUP_AUDIT.md)

---

## 1. Files retained

- `README.md`, `CITATION.cff`, `LICENSE`, `requirements.txt`, `pyproject.toml`,
  `.gitignore`, `.gitattributes`, `.github/workflows/tests.yml`,
  `.streamlit/config.toml`.
- Original root notebook:
  `Do_Chicago_speed_cameras_reduce_crashes_.ipynb`.
- `paper/Chicago_Speed_Cameras_Research.pdf` (the only public paper).
- Internal source: `paper/source/chicago_speed_cameras_research.md`,
  `paper/templates/working_paper.typ`, and 11 publication figures plus the cover
  graphic under `paper/figures/`.
- `src/chicago_cameras/` causal, geospatial and ML modules; `scripts/`;
  `config/`; `dashboard/app.py`; `tests/`.
- Deployed V1 artifact at `models/v1/lightgbm_crash_model.txt` with metadata at
  `models/v1/model_metadata.json`.
- Causal results and tables in `results/`, V1 dashboard inputs in `results/ml/`,
  V1/V2 selection and interval inputs in `results/ml_v2/`, coverage outputs in
  `results/bottom_up/`, and the compact experiment record in
  `results/statistical_hybrid/`.
- Essential documentation in `docs/`, including methods, data dictionary,
  reproducibility, model card, audits, model-selection rationale and
  negative-result records.

## 2. Files removed

The full classification is in
[`RELEASE_CLEANUP_AUDIT.md`](RELEASE_CLEANUP_AUDIT.md). In summary:

- Entire `linkedin/` promotional directory.
- Duplicate Markdown/DOCX manuscript exports, old PDF path, paper build directory,
  proposed-revision draft and bottom-up duplicate report.
- Superseded publication/handoff drafts, duplicate reports and obsolete links.
- Experimental dashboard page; the dashboard now has seven pages.
- Redundant bottom-up checkpoints, root duplicate V1 checkpoint, V2 binary
  checkpoint and large prediction/SHAP intermediates not used by the final
  dashboard.
- `scripts/build_prediction_report.py`, which targeted removed documentation.

V2 code, metrics, deployment selection and negative-result documentation are
retained; only the redundant binary checkpoint was removed. The retraining
workflow remains available.

## 3. Scientific content preserved

- Causal estimates, tables and figures were not altered. The paper retains the
  original −7.3% replication, revised −2.6% estimate, coverage-consistent −8.6%
  sensitivity, event study, synthetic control, placebos and limitations.
- The original notebook remains at the repository root and in Git history.
- V1 forecasting artifacts, metrics, SHAP inputs and conformal interval
  predictions remain for the dashboard.
- The negative model experiment is retained in:
  `MODEL_EXPERIMENT_BASELINE.md`, `MODEL_EXPERIMENT_RESULTS.md`,
  `MODEL_SELECTION_DECISION.md`, `HYBRID_MODEL_METHODS.md`,
  `STATISTICAL_DIAGNOSTICS.md` and `results/statistical_hybrid/`.
- Co-author attribution remains in the README, paper, citation metadata and
  `AUTHOR_CONTRIBUTIONS.md`.

## 4. Model-selection justification

V1 remains the deployed forecaster. The coverage-consistent optimized LightGBM
and the hybrid improved temporal-validation Poisson deviance by about 0.48–0.49%,
below the predeclared 1% rule. The Negative Binomial model was materially worse.
The hybrid’s weight was 0.943952 on LightGBM and 0.056048 on Negative Binomial,
with out-of-fold error correlation 0.976, so its negligible gain does not justify
additional operational complexity. No candidate met the promotion rule.

## 5. PDF verification

| Check | Result |
|---|---|
| Public path | `paper/Chicago_Speed_Cameras_Research.pdf` |
| Pages | 21 |
| Publication figures | 11 (plus cover graphic) |
| Tables | 6 |
| File size | 2,694,967 bytes (~2.69 MB) |
| Body page numbers | 1–18; cover/author/TOC pages precede them |
| Source available | `paper/source/chicago_speed_cameras_research.md` |
| Build command | `python scripts/build_publication_docs.py` |
| Hidden build output | `paper/build/` is ignored and was removed after verification |

The PDF was rebuilt from the retained source. The text contains all 11 figure
captions and six numbered tables; the final body ends on page 18. No clipped or
missing section was found in the extracted text inspection.

## 6. Dashboard status

- Seven pages: Research overview, Chicago map, Causal inference, Crash
  forecasting, Robustness & sensitivity, Coverage & reconstruction, and Methods
  & limitations.
- `streamlit.testing.v1.AppTest`: all seven pages completed with zero exceptions.
- Local HTTP test: **200**.
- The dashboard reads saved artifacts, does not retrain, and has no public URL.
- Deployment remains separate from repository publication and is not authorized
  in this release.

## 7. Reproducibility status

- `pip check`: **No broken requirements found**.
- Model load check: V1 loaded with 44 features and 924 trees.
- CI workflow installs `.[dev,ml]`, runs the test suite and verifies imports
  without downloading data.
- Data acquisition, causal pipeline, replication check, V1 training/evaluation,
  coverage reconstruction and paper-build commands are documented in
  `docs/REPRODUCIBILITY.md`.
- Raw and processed datasets remain ignored; no synthetic data was substituted.
- Absolute local paths appear only inside ignored local data/provenance files;
  the current worktree’s ignored data and `.venv` are not part of the release.

## 8. Automated checks

| Check | Result |
|---|---|
| Pytest | 67 collected: **66 passed, 1 skipped** |
| Skipped test | V2 binary-artifact metadata test; V2 binary intentionally removed |
| Ruff | **All checks passed** (code/config; preserved notebooks have documented lint exemptions) |
| Model loading | PASS |
| Dashboard AppTest | PASS, seven pages |
| Dashboard HTTP | PASS, 200 |
| `pip check` | PASS |
| Local Markdown link validation | PASS after final audit files were added |
| Credential scan | No credentials found |
| PDF inspection | PASS: 21 pages, 11 figures, 6 tables |
| `git diff --check` | PASS |

The skipped V2 test is explicit and does not mask a failure: the V2 metrics and
retraining code remain, while the redundant binary checkpoint was removed.

## 9. Repository-size report

- Publishable payload (paper, source, models, results, docs, tests, scripts):
  **~51.4 MiB**.
- Git pack: **14.91 MiB**.
- `models/`: 5.3 MiB; `results/`: 37.6 MiB; `paper/`: 6.9 MiB after removing
  the ignored build directory.
- The 2.6 GB working-directory total includes ignored `data/` (~1.2 GB) and
  `.venv/` (~1.2 GB); neither is part of the public repository.

## 10. Outstanding approvals

The case-only PDF rename is staged so Git records the required uppercase path;
nothing has been committed or pushed.

1. **License:** MIT was added at the repository owner’s request, with copyright
   held by Saketh Donepudi and Amrutha Kakumani. Both co-authors should confirm
   the final license terms before public push/release.
2. **Git publication:** commit, push, pull request, merge and GitHub release
   require explicit authorization.
3. **Dashboard hosting:** no public deployment or URL is approved or claimed.
4. **Public metadata:** repository description, topics, release title and any
   institutional claims should be confirmed by both authors.

## 11. Suggested commit message

```text
Release final Chicago speed-camera research package

- retain the causal paper and reproducible V1 forecasting system
- keep compact negative results and model-selection rationale
- remove LinkedIn drafts, duplicate manuscripts and experimental dashboard page
- add MIT license, dashboard requirements and final release audits
- rebuild and verify the 21-page PDF
```

## 12. Suggested pull request description

> Final research release for the Chicago speed-camera study. This PR keeps the
> causal design and verified V1 forecaster, removes promotional and superseded
> development material, reduces the dashboard to seven research pages, makes the
> paper a single PDF with retained source, and adds an MIT license and
> requirements. The causal estimates are unchanged. V2 and the optimized
> LightGBM/Negative Binomial/hybrid experiment remain documented as negative
> results; no candidate met the predeclared 1% rule. Tests: 66 passed, 1 skipped
> (V2 binary artifact intentionally removed); Ruff clean; dashboard AppTest and
> HTTP checks pass. No deployment is included.

## 13. Recommended GitHub release notes

**Chicago Speed Cameras — final research release**

- Causal paper: one 21-page PDF, 11 figures and 6 tables.
- Deployed forecasting model: LightGBM V1; V2 and hybrid experiments retained as
  documented negative results, not deployed.
- Seven-page Streamlit research dashboard with local run instructions; no public
  URL.
- Reproducible causal, forecasting, coverage and validation workflows.
- MIT license and machine-readable citation metadata.
- Known limitations: small treated sample, reporting-coverage changes, no
  historical traffic-exposure denominator, ambiguous road assignments and
  imperfect high-risk interval coverage.

This release is ready for author review. It is **not committed, pushed, merged,
released or deployed**.
