# Performance and device tuning

## Default profiles

`avril run --backend auto` chooses a profile from available logical CPUs and
CUDA readiness. Defaults are deliberately bounded to avoid nested
process/thread oversubscription.

On a 10-core Mac, the CPU profile uses three temporal search workers with three
threads each, up to ten refit workers, four significance processes sharing ten
threads, and ten candidate-vetting processes.

On the validated 12-vCPU CUDA host, Avril keeps one persistent GPU search
worker, uses the complete CUDA fine/refit grids, then runs candidate preparation
and vetting in bounded CPU batches. Multiple simultaneous GPU search workers
usually waste memory and repeatedly compile contexts.

Use `--cpu-workers N` to cap all defaults for shared machines. Environment
variables such as `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
`VECLIB_MAXIMUM_THREADS` should not be set above the profile's CPU limit.

## Where time is spent

Runtime depends more on promoted refit families and total exact period trials
than on raw peak count or sector count alone. A target with 28,000 raw peaks may
collapse to a few hundred refit families; another with fewer raw peaks may still
be expensive if many independent event trains survive voting.

The scheduler therefore gates on:

1. exact refit-eligible family count;
2. estimated exact period-grid trial count.

It does not stop a target solely because the initial search has many peaks.

## Verified V2 baseline regressions

Cached complete 2-minute SPOC inputs, excluding network and optional pixel
follow-up:

| Target | Sectors | End-to-end compute | Regression result |
|---|---:|---:|---|
| L98-59 | 27 | 261.94 s | 2.253126, 3.690682 and 7.450741 d are the three automatic highs |
| TOI-786 | 43 | 218.11 s | 12.669431 and refit-rescued 38.553883 d rank 1-2 |
| TOI-125 | 7 | 311.53 s staged | 4.651669, 9.155244 and 19.981423 d rank 1-3 |
| TOI-1269 | 42 | 384.18 s | 4.252996 and 9.237794 d rank 1-2 |

The release-sealed L98-59 V2 science-finalization stage takes 80.67 seconds after
the reusable 181.27-second discovery/refit stages. TOI-786 science finalization takes 27.73
seconds. The V2 gate avoids exact phase-null work only after a deterministic
proof that the candidate cannot become automatic high priority; it never
removes the candidate or changes its locked ephemeris.

The GPU kernel itself can be much faster than CPU BLS, but downloads, event
validation, harmonic arbitration, red-noise trials, and physical review limit
end-to-end speedup. Do not advertise kernel-only speedup as pipeline speedup.

## Safe optimization rules

- Batch candidates or period cells; do not remove cells.
- Reuse sector views and compiled kernels.
- Use event-overlap proof before sharing harmonic-family work.
- Keep each candidate restricted to its original period-grid mask.
- Keep CPU and GPU preparation overlapped only when deterministic result order
  is restored before writing output.
- Install the compiled optimization layer before timing. Missing
  `optimization_v4.install()` silently falls back to a much slower Python loop.
- Pipeline targets across a batch: download target N+2, search/refit N+1, and
  CPU-vet N. This improves GPU occupancy without changing per-target results.
- Compare final ephemerides and review decisions, not just the top period.

Any approximate or reduced-grid mode must use a different release identifier
and cannot be reported as Avril V3.0.
