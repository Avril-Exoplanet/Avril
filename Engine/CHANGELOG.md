# Changelog

## Avril V3.0.0 — 2026-10-31

- Added BLS-first Scout routing with TLS fallback.
- Added post-search known-catalog and event-proven alias short-circuiting.
- Added pre-vetting gap-edge domination checks.
- Preserved every candidate and locked ephemeris in the audit ledger while
  restricting expensive vetting to meaningful unexplained candidates.
- Reused the detailed upstream event ledger instead of measuring every event a
  second time; a real 304-candidate regression preserved every scientific
  decision while reducing finalization wall time from 15.036 s to 7.941 s.

## Avril V2.0.1 — 2026-10-31

- Added locked-event gap-integrity diagnostics using transit coverage, distance
  to long-gap edges and left/right baseline symmetry.
- Prevented a single gap-compromised event from promoting an otherwise weak
  event train to automatic high priority; all candidates and ephemerides remain
  in the audit ledger.
- Preserved high priority when at least two uncompromised events independently
  exceed 3 sigma or the uncompromised combined MES-like statistic remains above
  the 7.1 reference.

## Avril V2.0 — 2026-10-30

- Preserved the complete-grid search, exact refit, candidate count and locked
  ephemerides while compiling event and physical-review batches.
- Added deterministic pre-significance gates so candidates that cannot become
  automatic high priority remain auditable without wasting 4,096 exact null
  trials.
- Required at least four supported events for automatic high priority; sparse
  two- and three-event signals remain explicit manual-review candidates.
- Added direct harmonic-winner verification and exact fallback, preventing a
  failed short-period alias from suppressing a viable longer-period signal.
- Reduced TOI-125 science finalization from 304.721 to 29.492 seconds while
  preserving all 503 candidates and identical CPU/CUDA review decisions.
- Added release-sealed known-target, FFI Core, sparse-sector Pro, web, archive,
  wheel-build and one-click package checks.

## Avril V1.0 — 2026-10-29

- Froze one scientific contract for Apple Silicon CPU and NVIDIA CUDA.
- Added complete-grid CUDA search and deterministic CUDA ephemeris refitting.
- Added five deterministic vote seeds, event-direction repeatability and
  integer/rational harmonic-family arbitration.
- Added immutable discovery/science ranks and separate algorithm/web archives.
