# PHASE 0 — Repository Audit

**Project:** *Do Chicago speed cameras reduce crashes?*
**Repository:** `amruthakakumani/Do-Chicago-speed-cameras-reduce-crashes-`
**Audited commit:** `ff14045` ("Add files via upload")
**Audit date:** 2026-10-24
**Scope:** every tracked file (`README.md`, `Do_Chicago_speed_cameras_reduce_crashes_.ipynb` and its
saved outputs), the Chicago Data Portal endpoints it uses, and a full re-execution of the
original logic against the current portal snapshot.

The original notebook is preserved unchanged at the repository root. All corrections live in
`src/chicago_cameras/`, `notebooks/research_analysis.ipynb`, and the regenerated results.

---

## 1. Inventory

| Item | Finding |
|---|---|
| Tracked files | `README.md`, one notebook (23 cells, 556 KB) |
| Notebook environment | Google Colab; two cells mount Google Drive |
| Hard-coded paths | `/content/drive/MyDrive/chicago_car_crash/Traffic_Crashes_-_Crashes_20261009.csv`, `data/cameras.csv`, `data/crashes.csv`, `data/violations_monthly.csv` |
| Dependencies | Not pinned anywhere; README lists `pandas numpy scipy statsmodels scikit-learn matplotlib requests` (+ optional `pymc`) |
| Data sources | Socrata datasets `4i42-qv3h` (cameras), `85ca-t3if` (crashes), `hhkd-xvj4` (violations) |
| Tests | None |
| Provenance / checksums | None |
| Seeds | `KMeans(random_state=0)`, `np.random.default_rng(0)`, PyMC `random_seed=1`; no global seed |
| Data committed | No |

### Saved outputs versus source code

Every printed number in the saved notebook was re-derived from the source code and the current
portal snapshot (see §4). **The saved outputs are consistent with the code**: the original
pipeline reproduces its published estimates exactly, given the same inputs. The problems below
are methodological, not fabricated results.

---

## 2. README claims versus implemented code

| README claim | Implemented? | Evidence |
|---|---|---|
| 250 m "near a camera" radius | Yes | `RADIUS_M = 250`, `monthly_counts(..., radius=RADIUS_M)` |
| 500 m comparison lattice | Yes | `np.arange(..., GRID_M)` with `GRID_M = 500` |
| Controls > 600 m from every camera | Yes | `grid[cam_tree.query(grid)[0] > BUFFER_M]` |
| **Controls within 1,000 m of their camera** | **No** | `MAX_DISTANCE` is described in the README but never defined or applied. `dist` is computed over **all** comparison sites and the five smallest scores are taken citywide. |
| Five matched controls per camera | Yes, citywide | `np.argsort(dist)[:N_CONTROLS]` |
| −24 / +36 month event window | Yes | `rel = np.arange(-PRE_MONTHS, POST_MONTHS)` |
| Camera-clustered standard errors | Yes | `cov_type="cluster", cov_kwds={"groups": panel["stack"]}` |
| Event study / decay test | Yes | Quarterly Poisson interactions; `tp_t` trend term |
| Synthetic control, de-meaned fit | Yes | `run_synth` with `mu_y`, `mu_X` de-meaning |
| Placebo-in-space inference | Yes | 100 random pseudo-treated donor groups |
| Heterogeneity by area | Yes | k-means on camera coordinates, four groups |
| Violations mechanism check | Yes, descriptive only | Mean violations by months since first violation |
| Injury crashes as an outcome | **No** | `injuries_total` is downloaded but never used |
| Optional Bayesian NB DiD | Yes | PyMC Negative Binomial on aggregated cells (`n` as exposure) |
| Parallel trends assumption stated | Partially | Limitations mention mean reversion, COVID, reporting; no formal pre-trend test |

---

## 3. Findings

Priority: **H** = changes the headline estimate or its interpretation; **M** = affects a reported
number or its uncertainty; **L** = reproducibility / presentation.

| # | Existing methodology | Evidence of implementation | Identified issue or uncertainty | Scientific impact | Proposed correction | Priority | Verification test |
|---|---|---|---|---|---|---|---|
| 1 | "Matched twins" within 1,000 m | `MAX_DISTANCE` absent from notebook; `dist` computed over all `Y_ctl` | The documented local-matching design is **not implemented**. Controls may be anywhere in the city. The label "twins" overstates comparability. | Local shocks (construction, policing, land use) are not absorbed; estimates conflate camera effects with neighbourhood change. Also makes the design non-replicable from the README. | Explicit `max_match_distance_m` in the design. Primary local design = 2,000 m (1,000 m leaves only 11 of 33 cameras with five eligible controls); 1,000 m and citywide matching reported as sensitivity. | **H** | `tests/test_matching.py::test_distance_cap_excludes_far_controls`; sensitivity rows "Max match distance …" |
| 2 | Overlapping camera catchments | No overlap check; cameras deduplicated only on exact coordinates | 107 camera pairs lie < 500 m apart. Among all 209 cameras, 34,222 crashes (36.3% of those attributed) fall inside **two or more** 250 m catchments. Treated units are therefore neither independent nor cleanly untreated. | Double counting inflates precision; a later camera inside an already-treated catchment is not a new experiment. | Drop cameras whose catchment overlaps an **already-live** camera (13 of 33 eligible); `keep_all` as sensitivity. Report overlap diagnostics. | **H** | `test_design.py::test_build_design_end_to_end`; diagnostics `camera_attribution_*`, `overlap_pairs`, `dropped_for_overlap` |
| 3 | Zeros treated as observed | `monthly_counts` initialises a zero matrix and fills only crash months | The portal's crash table is effectively empty before 2015‑09 (476 records across 2015‑01…08; 8 records in 2013–14). Starting the window at 2015‑01 treats missing coverage as genuine zeros. | False precision; distorted pre-period levels and trends for the earliest cameras. | Coverage guard drops leading months below 10% of the median month (window starts 2015‑09, T = 132). Distinguish observed zeros from missing months in documentation. | **H** | `tests/test_cleaning.py::test_crash_coverage_guard_drops_sparse_leading_months`; `quality_report.json → coverage_start` |
| 4 | Pooled `treated + post + treated×post` with staggered go-live dates | Single OLS/Poisson on the stacked panel | The regression is a two-way-FE-style specification. With staggered adoption and heterogeneous effects it can place negative weights on some 2×2 comparisons (Goodman-Bacon; de Chaisemartin & D'Haultfœuille; Callaway & Sant'Anna). The stacked design mitigates but does not eliminate this. | The coefficient need not equal a convex weighted average of cohort-specific effects. | Keep the original as a baseline. Add (a) conditional Poisson with stack × event-month FE and (b) a transparent per-camera 2×2 ATT averaged across cameras with camera-level bootstrap. | **H** | `tests/test_did.py::test_conditional_poisson_recovers_rate_ratio`, `test_cohort_did_recovers_known_step`; main estimates table |
| 5 | Flat lon/lat → metres conversion | `x = (lon−LON0)·111320·cos(LAT0)`, `y = (lat−LAT0)·110574` | Degree-to-metre constants are approximate and anisotropic (latitude scale is ~0.5% off); the 250 m boundary is not exact. | Crashes near the boundary are misclassified; the effect on the estimate is not negligible at 33 cameras (see §4). | Project to EPSG:26916 (NAD83 / UTM 16N) with `pyproj`; validate against geodesic distance. | **H** | `test_geospatial.py::test_projection_matches_great_circle_distance` |
| 6 | Synthetic control on the largest cohort | `run_synth`, 100 donors, de-meaned | Pre-period fit is weak: R² = 0.74, pre-RMSPE = 0.71 crashes/month against a pre-mean of 3.41 (21%). The SC estimate is **+4.2%** (more crashes), opposite in sign to the DiD. No fit gate is applied. | The SC does not corroborate the DiD; presenting it as confirmation would be misleading. | Report pre-fit R²/RMSPE, refuse to label a weakly fitting SC as informative, and report the sign disagreement. Add in-time placebos and leave-one-out donor sensitivity. | **H** | `test_synthetic_control.py::test_poor_pre_fit_is_visible`; `results/tables/synthetic_control.csv` |
| 7 | "Drivers adapted: violations fell over a camera's life" (README) | Mean violations by months since first violation | Violations measure **enforcement output**, not speeding behaviour or exposure. Camera operation, ticketing policy, warning periods and fleet composition all change over time. | The README's causal mechanism claim is not supported by a descriptive curve. | Re-frame as a descriptive mechanism check with explicit caveats; estimate a within-camera slope with clustered SE, and state that it cannot establish behaviour change. | **H** | Report §Mechanism; `results/tables/violations_curve.csv` |
| 8 | Comparison lattice anchored to the crash bounding box | `np.arange(xmin, xmax, GRID_M)` | The 500 m lattice phase is arbitrary and data-dependent. With citywide matching, the estimate ranges from **−5.0% to −8.5%** across four phases; local matching narrows it to −1.2%…−3.1%. | A single lattice draw can overstate precision or shift the headline. | Report lattice-phase sensitivity for every design; prefer the local design whose phase spread is small; document that phase is not identified from the data. | **M** | `results/tables/sensitivity.csv` rows "Lattice phase …" |
| 9 | Cluster-robust SEs by camera stack | `cov_kwds={"groups": panel["stack"]}` | Comparison sites are **shared** across stacks (reused controls), so errors are correlated across clusters. Only 33 (original) or 20 (primary) clusters; asymptotic cluster SEs are downward-biased. | p-values may be too small; confidence intervals too narrow. | Report wild cluster bootstrap-t p-values (Rademacher, restricted null) alongside asymptotic SEs; note the small-cluster caveat. | **M** | `wild_cluster_bootstrap` in `results/estimates.json`; `test_did.py::test_wild_cluster_bootstrap_smoke` |
| 10 | Go-live month = first post month | `post = rel >= 0` | Go-live dates are monthly, mid-month; the go-live month is partially treated. Chicago also ran warning periods before enforcement. | Attenuates or blurs the immediate effect. | Estimate `treatment_month_offset` 0 (baseline, original) and 1 (first full month); document the warning-period assumption. | **M** | Sensitivity row "Go-live month excluded" |
| 11 | Crash-reporting change around 2017 | Acknowledged in README limitations only | 2016 has 44,097 crashes versus 83,477 in 2017 and 118,469 in 2018. The 2018 cohort's pre-period straddles this transition. | Differential reporting changes across neighbourhoods can masquerade as a camera effect. | Control for common shocks via stack × month FE; add a cohort-era sensitivity (2015–2018 vs 2019+). | **M** | Sensitivity rows "2015-2018 go-live cohorts only" / "2019+ go-live cohorts only" |
| 12 | Sample = cameras with 24 pre + 36 post months from 2015 | `eligible` mask | 176 of 209 cameras are excluded; the 2013–14 initial rollout (117 cameras) can never qualify because data begin in 2015. Estimates generalise only to 2018+ installations. | External validity, not internal validity, is limited: the earliest and often highest-risk sites are absent. | Document explicitly; report the go-live distribution of the estimation sample; avoid claiming a citywide average effect. | **M** | `estimates.json → sample`, `cameras.csv` year counts |
| 13 | Comparison sites are lattice cells | Grid + ≥ 100 crashes | Lattice points are not road-network nodes. The crash-volume screen makes them road-exposed in practice, but offset from intersections is arbitrary. | Control catchments may mix arterial and residential exposure. | Screen on crash volume **and** active-month share; report posted-speed-limit balance between camera and comparison catchments; keep radius/grid sensitivity. | **M** | `diagnostics.speed_limit_balance`, `control_overlap_pairs`; grid/radius sensitivity |
| 14 | 600 m exclusion buffer | `BUFFER_M = 600` | Buffer prevents **catchment** overlap (600 > 2×250) but not behavioural spillover or crash displacement beyond 250 m. | SUTVA may fail if drivers slow before a camera or crashes migrate. | State the assumption; vary buffer 500/800 m; note displacement as an untested threat. | **M** | Buffer sensitivity rows |
| 15 | Crash counts as the outcome | `y` = crashes per site-month | No exposure denominator; "crashes per site" is not "crashes per vehicle-mile". | Cannot interpret as risk per unit of travel. | Define the estimand as crashes per catchment-month; do not use an exposure offset that is unavailable. | **M** | METHODOLOGY.md estimand section |
| 16 | Injury data downloaded but unused | `INJURIES_TOTAL` in `keep` | Severity outcomes, which are policy-relevant, are never analysed. | Misses the welfare-relevant margin. | Add injury and severe/fatal outcomes with the same design. | **M** | `results/tables/severity.csv` |
| 17 | Subgroup comparisons by region | Four k-means groups, no adjustment | Nine exploratory comparisons; no multiple-testing control. | False positives likely if read as confirmatory. | BH-adjust p-values; label exploratory. | **M** | `subgroups.csv → pvalue_bh` |
| 18 | No exposure offset in the Bayesian model | `n = unit.nunique()` as exposure multiplier | `n` is the number of units, not traffic volume; it is a modelling convenience, not exposure. | The Bayesian "effect" inherits the same per-catchment interpretation. | Keep as an optional robustness model; document that `n` is a cell size, not VMT. | **L** | METHODOLOGY.md |
| 19 | Colab/Drive coupling | `drive.mount` twice; absolute Drive path | Cannot run outside Colab; no provenance; no pinned dependencies. | Not reproducible by a reviewer. | Package with `pyproject.toml`, config file, provenance JSON, checksums, CLI, tests, CI. | **L** | `REPRODUCIBILITY.md`; `.github/workflows/tests.yml` |
| 20 | README key-findings table | `[X]%`, `[lo, hi]`, `[p]` placeholders | The README never reports the notebook's actual results. | Readers cannot tell what was found. | Rewrite README with validated results, CIs, sample sizes and caveats. | **L** | README "Findings" |
| 21 | Cosmetic/robustness nits | `MIN_CRAHES` misspelling; unused `START`?; no schema validation | Typos do not affect results; missing schema validation can silently break on portal changes. | Low, but harms maintainability. | Rename, validate schemas explicitly, fail with actionable errors. | **L** | `test_cleaning.py`, `acquisition` schema checks |

### What is *not* wrong

To avoid over-correcting a defensible design:

- The stacked event-time panel, matched on pre-period level and slope, is a reasonable design for
  this question.
- Clustering at the camera level is the right *level* of clustering; the issue is the small number
  of clusters and shared controls, not the choice of cluster variable.
- The de-meaned synthetic-control fit is a sensible response to the level mismatch between hotspot
  catchments and ordinary sites.
- Quarterly event-study binning is appropriate given monthly noise.
- The 2-month reporting lag is a reasonable exclusion.
- The Cauchy-style robust check of dropping the buffer entirely was not attempted by the original;
  that is an omission, not an error.

---

## 4. Reproduction test

The original logic was re-implemented line-for-line (`.context/replication_check.py`, not part of
the deliverable) and run on the **current** portal snapshot.

| Quantity | Original saved output | Exact re-run of original logic | Status |
|---|---|---|---|
| Crashes kept (2015+) | 1,081,469 | 1,081,474 | 5-row portal update |
| Cameras eligible | 33 of 209 | 33 of 209 | Exact |
| Panel rows | 11,880 | 11,880 | Exact |
| Pre-period means (twins / cameras) | 4.30 / 4.40 | 4.298 / 4.403 | Exact |
| OLS ATT | −0.356 [−0.711, −0.000], p = 0.0497 | **−0.3557 [−0.7110, −0.0005], p = 0.0497** | Exact |
| Poisson ATT | −7.3% [−13.6%, −0.6%], p = 0.0342 | **−7.3% [−13.6%, −0.6%], p = 0.0342** | Exact |

**Conclusion:** the saved outputs are reproducible. The upgrade does not "correct" the arithmetic;
it changes the design in ways that matter:

| Specification | Sample | Poisson ATT | 95% CI | p |
|---|---|---|---|---|
| Original logic (flat projection, citywide matching, overlapping cameras kept) | 33 cameras | **−7.3%** | [−13.6, −0.6] | 0.034 |
| Corrected projection (UTM), same sample and design | 33 cameras | **−8.6%** | [−14.7, −2.1] | 0.011 |
| Primary local design (UTM, 2,000 m matching, overlap removal) | 20 cameras | **−2.6%** | [−11.5, +7.2] | 0.59 |
| Cohort 2×2 average (primary design) | 20 cameras | −0.10 crashes/site-month | [−0.70, +0.52] | 0.71 |

The headline finding of the upgrade is therefore **not** "the original was wrong"; it is that the
original's statistically significant −7% is **not robust** to two defensible design changes:
local comparison sites and removing contaminated (overlapping) camera catchments.

---

## 5. Defect found in the upgrade itself (disclosed)

During implementation, the first version of `matching.py` computed comparison-site pre-period
levels and slopes on a fixed 2015–2016 window instead of each camera's own pre-window. A regression
test (`test_matching_uses_each_camera_own_pre_window`) caught it. The defect changed the corrected-
projection replication from −8.6% to −2.4%. It is recorded here because scientific reports should
disclose implementation errors; the test prevents recurrence.

---

## 6. Decisions flagged for author review

1. **Primary matching radius.** 2,000 m is a judgement call: 1,000 m (the README's claim) leaves
   only 11 of 33 cameras with five eligible comparison sites. The upgrade reports all three
   (1,000 m, 2,000 m, citywide) and does not select the most favourable.
2. **Overlap rule.** Dropping cameras whose catchment overlaps an already-live camera costs 13 of
   33 eligible cameras. Keeping all cameras inflates precision and double-counts crashes. Both are
   reported; `drop_later` is the baseline.
3. **Coverage start.** The coverage guard moves the window from 2015‑01 to 2015‑09. If the authors
   believe the sparse early-2015 records are genuine, the window could start earlier and the
   estimates would change.
4. **Estimand.** The upgrade reports crashes per catchment-month, not crashes per VMT. Changing this
   would require an exposure measure the portal does not provide.
