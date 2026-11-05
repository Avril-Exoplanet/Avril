# Changelog

All notable changes to Avril will be documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and release versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) where the
scientific contract permits it.

## [Unreleased]

### Fixed

- Keep installer wheel caches on the native system disk by default, preventing
  `fbpca` and `memoization` wheel-discovery failures when an ambient pip cache
  points to ExFAT or a network volume.

### Changed

- Make the repository-level macOS bootstrap select a supported Python 3.11,
  3.12, or 3.13 interpreter instead of silently accepting Apple's older system
  Python.

### Added

- Native macOS workstation with target selection, local execution, scientific
  light-curve plots, optional pixel validation, run records, and PDF export.
- Avril V3 CPU/CUDA search engine with deterministic evidence gates and a
  candidate-preserving audit ledger.
- Frozen 1,000-target discovery queue and leakage-controlled target-prior model
  card.
- Reproducible source-build scripts, CI, release audit, governance documents,
  asset provenance, and architecture figures.

### Scientific status

- Research preview. Candidate output is not planet validation.
- Population-level completeness and false-positive calibration remain future
  work and require sealed TIC-disjoint evaluation plus injection/recovery.

[Unreleased]: https://github.com/Avril-Exoplanet/Avril/compare/HEAD...HEAD
