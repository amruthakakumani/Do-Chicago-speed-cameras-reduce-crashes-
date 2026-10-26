# Draft pull request — final publication review

**Base:** `main` (or the co-authors' agreed publication base)
**Head:** `publication/final-review`
**Status:** draft only; not opened or pushed.

## Title

Final co-authored research release: Chicago speed cameras, causal sensitivity and crash forecasting

## Description

> This PR integrates the bottom-up reconstruction into the final co-authored
> manuscript. It adds the September 2017 citywide reporting boundary, the
> original −7.3% (33 cameras), revised −2.6% (20 cameras) and coverage-consistent
> −8.6% (15 cameras) estimates with explicit sample and estimand distinctions;
> adds coverage-consistent and road-segment model results; documents why AADT was
> excluded and why high-risk calibration remains imperfect; regenerates the
> Markdown/PDF/DOCX paper with 11 figures; and updates the README, model cards,
> methodology documentation, LinkedIn articles and dashboard. V1 remains
> deployed and V2 remains a research candidate because neither V2 nor the
> coverage-tuned model meets the pre-declared 1% validation rule. No release,
> deployment, push or publication is performed without explicit authorization.

## Review focus

1. Confirm the coverage-regime interpretation.
2. Confirm the original, revised and coverage-consistent estimates are presented
   as different samples, not competing significance claims.
3. Confirm the road-segment task is not ranked against the camera-zone task.
4. Confirm the AADT chronology and exclusion rationale.
5. Approve affiliations, contributions, license and LinkedIn wording.
6. Verify the dashboard before any public deployment.

## Checklist

- [x] Statistics checked against saved artifacts.
- [x] 13 DOI records checked through Crossref; external URLs checked.
- [x] 11 figures generated from saved data.
- [x] 21-page PDF and 21-page DOCX with cover, author page, TOC, headers and page numbers.
- [x] README, model cards, methodology and LinkedIn drafts aligned.
- [x] Seven dashboard pages and overlay tested.
- [x] 63 tests pass; `ruff` clean.
- [x] Local links checked.
- [ ] Co-author approvals.
- [ ] Explicit authorization to commit/push/release.
