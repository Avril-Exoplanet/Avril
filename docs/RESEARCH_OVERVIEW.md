# Research overview

## Abstract

Avril is a local-first system for prioritizing, searching, and auditing transit
signals in public TESS photometry. Its design separates two questions that are
often conflated: **which stars deserve finite search compute first**, and
**which signals survive an evidence-preserving transit search**. A frozen
stellar-state prior routes compute before any TESS light curve is opened. The
blind search then evaluates temporal consistency, gap integrity, event
direction, harmonics, physical diagnostics, red noise, and exact phase-null
trials while retaining every candidate and skip reason in an audit ledger.

The current release is a research preview. It produces candidates for expert
review, not validated planets, and makes no claim of population-level
completeness or calibrated false-positive probability.

## Problem formulation

Let a target star have pre-search metadata \(x_*\), public TESS observing
opportunity \(o_*\), and an unknown set of transit signals. Avril uses a frozen
target prior \(f(x_*)\) and transparent opportunity rules \(g(o_*)\) only to
choose the order in which targets receive compute:

```text
queue priority = frozen stellar prior + transparent observing opportunity
```

Neither term enters the transit detection statistic. Once a target is opened,
the candidate search uses only the frozen light-curve manifest and the declared
Avril configuration. The separation prevents a promising catalog profile from
turning a weak signal into a high-confidence candidate.

## Method

### 1. Pre-search target intelligence

The public prior is trained from a frozen Kepler DR25 snapshot using six
stellar-state features: effective temperature, surface gravity, metallicity,
mass, radius, and J−K color. Known planet count, KOI/TCE count, brightness,
noise, observing duration, and all TESS light-curve evidence are excluded.

Five-fold validation is split by star, keeping every planet around a star in
the same fold. The default ordinal prior achieves a 4.40× top-decile enrichment
for the frozen high-reliability host label; the confirmed-only profile reaches
4.82×, and the occurrence-grid reference reaches 3.77×. These are enrichment
measurements on Kepler labels, not calibrated probabilities for TESS.

### 2. Frozen input acquisition

Avril records the exact public SPOC/TESS-SPOC products or raw TESSCut-derived
self-photometry used in a run. One preferred official light curve is selected
per sector. Pixel products are follow-up evidence and remain disabled by
default.

### 3. Candidate-preserving search

The engine searches complete temporal blocks using fixed grids and deterministic
cross-sector votes, then performs local period, epoch, and duration refits.
Locked ephemerides and discovery rank remain immutable. Later diagnostics can
change review priority but cannot silently delete the candidate or rewrite its
discovery history.

### 4. Evidence gates

The review path distinguishes hard failures from manual-review warnings. It
tests event direction, harmonic support, catalog context, incomplete-window and
gap-edge domination, red-noise robustness, physical/reference consistency, and
4,096 exact phase-null trials where applicable. Sparse two- and three-event
signals remain reviewable instead of being forced into a false binary decision.

### 5. Scientific review

The native workstation exposes the light curve, candidate fold, event table,
gap diagnostics, source and pixel context, run timeline, configuration, and
exportable JSON/PDF evidence record. A high-priority row remains a candidate
until independent validation is complete.

## Reproducibility contract

- CPU and CUDA backends share the same scientific grids and thresholds.
- Backend changes may alter throughput, not candidate count or locked
  ephemerides.
- Every released model and catalog artifact has a frozen manifest and hash.
- Catalog truth is attached after blind discovery and ranking.
- Scientific-path changes require deterministic tests and a public regression
  target.
- Opened targets and Avril-derived labels cannot enter a claimed blind fold.

## Evaluation status

Current evidence includes deterministic unit tests, known-system regressions,
cross-backend invariants, and out-of-fold target-prior enrichment. These checks
verify implementation and ranking behavior; they do not establish survey
completeness.

A publication-grade yield evaluation should add:

1. a sealed TIC-disjoint natural-yield set;
2. injection/recovery in representative real TESS noise;
3. negative controls for binaries, rotation, red noise, gaps, system periods,
   phase shifts, and inverted light curves;
4. ablations for the target prior and every evidence gate;
5. independent source localization and follow-up review for promoted signals.

## Responsible interpretation

Avril is designed to make candidate discovery faster and more auditable. It is
not a replacement for astrophysical validation, peer review, or follow-up
observations. Institutional names and NASA imagery in the source distribution
do not imply endorsement.
