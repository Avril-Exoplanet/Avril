# Architecture and scientific contract

Avril V3.0 has one pipeline and two numerical backends. Backend selection may
change throughput, never candidate rules or scientific thresholds.

```text
official SPOC manifest
        |
resumable concurrent download
        |
sector cleaning and detrending
        |
three complete temporal-block searches
        |
five deterministic resampling votes
        |
locked-ephemeris validation
        |
exact period/epoch refit for promoted families
        |
event-direction and harmonic-family arbitration
        |
post-search catalogue + event-proven alias + gap-integrity funnel
        |
physical and lightweight reference vetting for meaningful candidates
        |
4096 exact phase-null/red-noise trials for still-viable candidates
        |
candidate ledger + immutable receipts
```

## CPU path

Apple Silicon and ordinary x86 machines run vectorized/compiled NumPy, Numba,
and the native exact-BLS kernel. Independent temporal searches use bounded
process/thread groups. Candidate significance and physical vetting run across
candidates with multiple processes.

## CUDA path

CUDA uses a reference Astropy coarse search, evaluates every configured fine
grid cell with the exact CUDA BLS implementation, and runs the complete
deterministic global refit grid on the GPU. It then returns to the same CPU
multi-process significance and physical-vetting implementation used by Mac.

The production path does not use the old cuvarbase shortcut. That shortcut has
a different duration parameterization and does not return the complete
epoch/duration solution required by Avril.

## Invariants

- Five deterministic vote seeds and 4,096 phase-null trials are fixed defaults.
- Every base peak is retained in audit artifacts.
- Harmonic grouping requires event-timing evidence; integer divisibility alone
  is insufficient.
- Exact refit grids are not shortened to claim a speedup.
- Two- and three-event trains remain auditable manual-review candidates;
  automatic high priority requires at least four supported events.
- A harmonic loser can skip an exact null test only when its direct event-overlap
  winner passes the deterministic gates and its own exact null test. If that
  winner fails, the loser automatically receives an exact fallback test.
- Workload safety gates defer a target rather than delete candidates.
- Catalogue truth is never read during search and ranking.
- Detailed per-event measurements are stored once and reused by the V3 funnel;
  reuse cannot change a locked ephemeris or decision.
- A high-priority candidate is not a confirmed planet.

## Source map

- `avril/`: public CLI, manifest builder, and hardware profiles.
- `run_avril_v3_threshold_round.py`: workload-aware production scheduler.
- `run_vlora_*_experiment.py`: CPU/CUDA discovery, voting, and exact refit.
- `run_avril_v3_science_finalize.py`: meaningful-candidate funnel, compiled
  event/gap gates, selective exact phase-null testing, harmonic fallback,
  physical review and final science rank.
- `multiplanet_trial/live_web/`: production search and vetting kernels. The
  historical directory name is retained only to avoid an unscientific code
  rewrite after validation.
- `multiplanet_trial/v03`, `v05_eval`, `v07_eval`: required archive, manifest,
  and multibranch components. Public users should call `avril`, not these
  internal modules.

The separately distributed web package consumes result JSON. It is not imported
by the algorithm package.
