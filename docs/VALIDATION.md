# Validation record

This document separates engineering verification from scientific validation.
Passing the checks below means that the public source release is internally
consistent and reproducible at its declared interfaces. It does not mean that
every promoted signal is a planet or that survey-level completeness and
false-positive rates have been calibrated.

## Release-candidate snapshot

Snapshot date: 2026-11-05.

| Layer | Command or assertion | Result |
|---|---|---:|
| Repository audit | `./scripts/check_release.sh` | Pass |
| Python public tests | `pytest -q Engine/avril` | 24 passed |
| Scientific-core regression | deterministic suite in `Engine/docs/REPRODUCIBILITY.md` | 93 passed |
| Python static analysis | `ruff check Engine/avril Engine/vlora Engine/desktop_bridge.py` | Pass |
| Native workstation | `swift build -c release` | Pass, zero warnings |
| App assembly | `AVRIL_NOVA_SKIP_RUNTIME=1 ./scripts/build_app.sh` | Pass |
| App integrity | `codesign --verify --deep --strict` | Pass |
| Local engine bridge | `/api/health` | Avril 3.0.0 ready |
| Frozen target catalog | `/api/desktop/targets` | `total=1,000`; 50-row default page |
| Research figures | `scripts/render_research_figures.py` | Rebuilt from frozen public metadata and queue |
| Git release payload | staged source tree | 317 files, no unstaged or untracked release inputs |

The source-build application uses ad-hoc signing for local verification. A
downloadable public binary still requires a relocatable embedded Python runtime,
Developer ID signing, Apple notarization, and Gatekeeper testing on a clean Mac.

## Target-prior validation

Five-fold evaluation is split at the star level using the frozen Kepler DR25
snapshot. Every planet around one star remains in the same fold.

| Frozen model view | ROC AUC | Average precision | Top-decile lift |
|---|---:|---:|---:|
| Ordinal catalog-yield prior | 0.7616 | 0.0681 | 4.4039× |
| Confirmed-clean prior | 0.7862 | 0.0642 | 4.8189× |
| Occurrence-grid reference | — | — | 3.7671× |

These values measure enrichment for frozen Kepler catalog labels. They are not
calibrated TESS probabilities, and they do not measure the sensitivity or
specificity of Avril V3 on light curves.

## Verified scientific invariants

The public tests and release documents assert that:

- target ranking is isolated from blind transit evidence;
- CPU and CUDA implementations share the declared search grids and thresholds;
- candidate count and locked period, epoch, and duration are preserved across
  post-search review;
- catalog truth is attached after blind discovery and ranking;
- gap-edge domination and incomplete-window support are explicit diagnostics;
- optional pixel localization cannot silently rewrite the discovered
  ephemeris;
- skipped work and rejected candidates remain represented in the audit ledger.

## Not yet claimed

The current release does not claim:

- a calibrated planet posterior;
- population-level completeness or reliability;
- a sealed TESS natural-yield improvement over an unbiased control;
- astrophysical validation of any individual new candidate;
- equivalence between target-prior enrichment on Kepler and realized TESS
  discovery yield.

## Publication-grade evaluation plan

Before making survey-level performance claims, add:

1. a new TIC-disjoint sealed natural-yield split;
2. injection/recovery in representative real TESS noise;
3. negative controls spanning eclipsing binaries, stellar rotation, red noise,
   system periods, phase shifts, gaps, and inverted light curves;
4. ablations for target prioritization and every evidence gate;
5. CPU/CUDA same-input comparisons on the declared reference set;
6. independent source localization and expert follow-up for promoted signals.

Machine-readable target-prior provenance and artifact hashes are stored in
`Engine/target_selection/metadata.json`. Engine-level release evidence is stored
in `Engine/validation/avril_v3_release_validation.json`.
