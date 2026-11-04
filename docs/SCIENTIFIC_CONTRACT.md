# Scientific contract

Avril is a candidate-search and triage system. It maximizes auditable recovery of recurring dimming events while preserving common false-alarm and false-positive evidence.

## Terms

- **Target priority**: a queue ordering used before downloading or searching a target. It is not light-curve evidence.
- **Candidate**: a periodic or event-family hypothesis retained in the run ledger.
- **High-priority follow-up candidate**: a candidate that passes the current automated review and is worth localization and independent follow-up.
- **Validated or confirmed planet**: an external scientific status that Avril does not assign.

## What Avril does not establish

Avril does not, by itself:

- validate or confirm an exoplanet;
- provide a calibrated posterior planet probability;
- prove that a dimming signal is on the target star;
- replace spectroscopy, high-resolution imaging, or expert review;
- guarantee completeness outside a measured injection/recovery domain;
- make an unmatched catalog query equivalent to novelty.

## Required evaluation sequence

1. Regression on known systems without reading truth during execution.
2. Injection/recovery in real TESS noise across depth, duration, period, event count, gaps, and harmonic interference.
3. Negative controls covering quiet stars, eclipsing binaries, rotation, red noise, instrument periods, phase shifts, and inverted curves.
4. A new TIC-disjoint blind set whose labels remain sealed until predictions and artifact hashes are frozen.
5. Natural large-sample runs followed by pixel localization and external catalog review for final priorities.

Previously opened blind data becomes regression data. It must not be reused to support a new blind-performance claim.

## Discovery rank and science rank

`discovery_rank` is produced by search, deterministic voting, exact refitting, event repeatability, and harmonic arbitration. It is immutable.

`science_rank` is the public order after available red-noise, physical, reference, and pixel review. It places automated-review passes first, incomplete cases second, and material false-alarm or false-positive risks last. Within a tier, immutable discovery order is preserved.

## Gap integrity

A strong event is not accepted merely because its fitted depth is large. When a candidate's apparent significance is dominated by an event at a long missing-cadence boundary, has poor transit-window coverage, or lacks clean supporting events away from the gap, the candidate is downgraded while the original measurements remain auditable.

## Reporting

Population-level reports should include recall stratified by depth and event count, alerts per 1,000 searched targets, abstention/defer rate, runtime, failure count, and the exact frozen manifests and hashes used for the run.
