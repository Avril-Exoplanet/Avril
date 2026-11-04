# Pitfalls and failure modes

This file records issues encountered while developing the production path.

## Numerical precision

- Sending flux near 1.0 directly through a float32 GPU path loses shallow
  transit precision during subtraction. Centre flux in float64 before GPU
  conversion and compare against the float64 reference.
- CPU and GPU can disagree at equal-power boundary cells by the last floating
  bit. Require final candidate coverage, ephemeris tolerance, and identical
  review priority; do not require raw local-maxima bit identity.
- Cuvarbase's fast duration model and output contract differ from Astropy BLS.
  It is not the Avril production kernel.

## Parallelism

- Python threads do not make arbitrary Python loops parallel. Avril relies on
  compiled kernels plus bounded processes where work is candidate-independent.
- `workers x threads` can silently oversubscribe a machine. More workers can be
  slower than a conservative profile.
- Starting several GPU workers recompiles contexts and produces bursty, low
  utilization. One persistent GPU worker is the default.
- Candidate preparation and physical vetting remain CPU work in CUDA mode.
  Exact phase-null trials use the validated CUDA backend only for candidates
  that survive deterministic gates; short GPU bursts alone are not evidence of
  poor end-to-end throughput.
- Forgetting to install `optimization_v4` silently selects the Python fallback
  and can turn a seconds-scale exact null test into minutes. Production entry
  points install it before computation and record the selected backend.

## Candidate and harmonic handling

- Raw peak count is not refit workload. Do not call a target
  candidate-explosive until voting, locked-ephemeris validation, and exact
  family construction are complete.
- Expanding integer harmonic order alone can connect unrelated candidates into
  a giant family. Require predicted low-point/event overlap.
- Copying one representative refit result across a family changes candidate
  grids and can change pass/fail states. Shared scans must preserve each
  candidate's original grid mask.
- A two-of-seven or similar weak support count may remain in the audit ledger;
  that does not mean it passed final review.
- Pairwise harmonic loss is not enough to defer a candidate. A direct winner
  must itself pass event, physical and reference gates plus the exact phase-null
  test. Otherwise the loser must be tested as a fallback. This prevents a noisy
  short-period alias from suppressing a real longer-period event train.
- Two- and three-event signals are not deleted, but they cannot receive
  automatic high priority in V2; they remain explicit manual-review rows.

## Long-period recovery

- A coarse period error accumulates over years and can move later transits out
  of the validation window. Exact local period/epoch refitting is necessary.
- Long-period candidates with only two or three events need explicit protection
  from short-period majority rules.
- Fixed-width one-day detrending can attenuate very long-duration transits.
  Report this selection effect in injection/recovery tests.
- A high combined statistic can be created by one event immediately beside a
  long cadence gap, especially when the transit window is incomplete and the
  local baseline exists on only one side. Avril audits event-window coverage,
  long-gap distance and left/right baseline symmetry. It retains the candidate
  but blocks automatic high priority only when that compromised event dominates
  and the remaining complete events are weak. A gap-adjacent event is not, by
  itself, a veto when independent complete events remain significant.

## Data and catalogues

- Search all naturally available official sectors unless the experiment freezes
  a different rule before looking at labels.
- Do not infer completeness from the number of cached files; cadence selection
  and archive availability vary by target.
- A failed TOI catalogue match is not evidence for a new planet. Catalogue
  snapshots can be stale or unavailable.
- Light curves are sufficient for periodic search. Pixel files are expensive
  and should be downloaded for final high-priority source localization, not for
  every initial search.
- CROWDSAP and moment centroids are not substitutes for a TPF difference image.

## Scientific language

- Rank score is not planet probability.
- Automated candidate discovery is not validation or confirmation.
- An unmatched high-priority candidate still needs pixel localization,
  neighbour analysis, independent data, and expert review.
