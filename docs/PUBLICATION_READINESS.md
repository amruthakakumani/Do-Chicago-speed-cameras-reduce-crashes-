**Historical note:** this readiness record predates the bottom-up reconstruction. The final audit is `docs/FINAL_PUBLICATION_AUDIT.md`.

# Publication readiness

**Date:** 2026-10-26
**Branch:** `publication/research-release`
**Decision:** **READY FOR AUTHOR REVIEW — NOT READY FOR PUBLIC RELEASE.**
Publication requires explicit approval from both co-authors and resolution of
the items marked REQUIRES AUTHOR REVIEW.

---

## 1. Acceptance criteria

| Criterion | Status | Evidence |
|---|---|---|
| Every headline causal statistic verified from artifacts | PASS | `docs/PUBLICATION_AUDIT.md` §1 |
| Every headline forecasting statistic verified | PASS | audit §2 |
| Data snapshot and observation window verified | PASS | audit §3 (2026-10-24 snapshot; 2015-09 … 2026-08) |
| No future-dated observations | PASS | max crash date 2026-08-31; leading/trailing guards |
| Leakage checks pass | PASS | V1/V2 metadata; 63 tests |
| Root original notebook unchanged | PASS | zero diff; `notebooks/research_analysis.ipynb` is a separate research copy |
| V1 and V2 artifacts preserved | PASS | `models/v1/`, `models/v2_*`, `results/ml/v1/`, `results/ml_v2/` |
| V1 remains deployed under the pre-declared rule | PASS | `results/ml_v2/deployment.json` |
| V2 not described as a deployed improvement | PASS | paper, README, dashboard |
| Causal and predictive analyses separated | PASS | paper §§4–10; dashboard sections |
| Eight publication figures from saved data | PASS | `paper/figures/`, `scripts/make_publication_figures.py` |
| Figure units, periods, captions, sample sizes | PASS | `paper/figures/README.md`; figure captions |
| Manuscript 3,000–5,000 words excluding refs/appendices | PASS | 4,920 body words |
| Manuscript PDF and DOCX generated | PASS | `paper/*.pdf`, `paper/*.docx`; DOCX has 8 figures, 4 tables |
| References verified against Crossref/DOI metadata | PASS | 13 DOIs checked; publisher HEAD responses block bots, metadata confirmed |
| No invented citations or findings | PASS | audit |
| README links and images resolve locally | PASS | link check |
| No placeholder statistics | PASS | audit §5 |
| Dashboard launches and all pages render | PASS | AppTest all six pages; HTTP 200; `docs/DASHBOARD_AUDIT.md` |
| Dashboard uses precomputed artifacts only | PASS | cached loaders; no training |
| Group-level causal effect not shown per camera | PASS | dashboard caveat |
| Intervals not shown as causal confidence intervals | PASS | dashboard and paper |
| Tests | PASS | 63 passed, 0 failed; `ruff` clean |
| No credentials or private data | PASS | `.env` absent; data gitignored |
| No large raw datasets committed | PASS | `data/` gitignored |
| Reproducibility commands documented | PASS | README; `docs/REPRODUCIBILITY.md` |
| Test-period reuse disclosed | PASS | paper §12; audit §4 |
| License | REQUIRES AUTHOR REVIEW | upstream does not declare one; none added |
| Author affiliations | REQUIRES AUTHOR REVIEW | unverified; omitted |
| Individual contribution roles | REQUIRES AUTHOR REVIEW | `docs/AUTHOR_CONTRIBUTIONS.md` pending |
| Live dashboard URL | REQUIRES AUTHOR REVIEW | not deployed; no URL claimed |
| LinkedIn article approval | REQUIRES AUTHOR REVIEW | drafts in `linkedin/` |
| Release / push / publication | REQUIRES AUTHORIZATION | not performed |

---

## 2. Publication checklist

- [x] Read every headline number from saved artifacts.
- [x] Reproduce the original notebook's −7.3% estimate exactly.
- [x] Verify the revised −2.6% estimate and 20-camera sample.
- [x] Verify V1 and V2 forecasting metrics and the 0.24% validation gain.
- [x] Confirm V2 fails the 1% promotion rule and V1 stays deployed.
- [x] Verify 90.6% interval coverage and 84.8% high-risk coverage.
- [x] Generate eight figures from real data; no fabricated plots.
- [x] Write the 4,920-word manuscript and generate PDF/DOCX.
- [x] Verify 13 DOI references.
- [x] Run 63 tests and `ruff`.
- [x] Launch the dashboard and exercise every page.
- [x] Check README image and document links.
- [x] Confirm no credentials or raw data are committed.
- [ ] Co-authors approve affiliations and contributions.
- [ ] Co-authors approve LinkedIn wording.
- [ ] Authors authorize the GitHub release/push.
- [ ] Verify a deployment before publishing any dashboard URL.
- [ ] Add a license only with owner authorization.

---

## 3. Draft pull request description

> This PR packages the Chicago speed-camera project as a publication release:
> a verified causal audit, a separate LightGBM forecasting system, a 4,920-word
> paper with eight data-driven figures, a redesigned academic README, a local
> research dashboard and two LinkedIn drafts. The original notebook, causal
> results, V1/V2 artifacts and contributor history are preserved. V1 remains the
> deployed forecasting model because V2 improved validation deviance by only
> 0.24% against a pre-declared 1% rule; V2 is retained as a research candidate.
> The 2025-09 to 2026-08 test period is explicitly labeled retrospective. No
> deployment, release or publication is performed by this PR.

---

## 4. Remaining risks

1. **Co-author approval is outstanding.** Authorship, affiliations, contribution
   roles and LinkedIn wording are not final.
2. **Test-period reuse.** The test period is not a new untouched confirmation;
   this is disclosed in the paper, audit and dashboard.
3. **Causal precision.** The primary interval includes zero and a meaningful
   reduction; no stronger causal claim is supportable.
4. **High-risk calibration.** Intervals under-cover the highest-risk decile;
   forecasts should not drive enforcement or resource allocation.
5. **No external deployment.** The dashboard runs locally only.
6. **Citation formatting.** Publisher pages may block automated checks; metadata
   was verified through Crossref/DOI records.

---

## 5. Decision

The scientific and software deliverables are internally consistent, reproducible
and honest about uncertainty. They are ready for **author review**, not for
public release. Publication is blocked on co-author approval of authorship,
affiliations, contributions, LinkedIn wording, license and deployment.

**Repository state:** branch `publication/research-release`; all changes remain
uncommitted. No commit, push, release, deployment or publication was performed.
