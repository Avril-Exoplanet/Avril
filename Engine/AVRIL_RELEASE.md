# Avril V3.0 release contract

`multiplanet_trial/live_web/vlora_release.py` is the single source of truth for
the release identifier and fixed scientific defaults. The historical module
name is retained to keep the validated import graph stable.

The V3.0 production CUDA path is:

1. reference Astropy coarse seeds;
2. every configured fine-grid cell on CUDA (`complete-grid`);
3. complete deterministic CUDA global-period refitting;
4. compiled, multi-process physical/reference hard gates;
5. exact 4,096-trial CUDA phase-null significance for every candidate still
   eligible for automatic high priority;
6. exact fallback evaluation whenever a proposed direct harmonic winner fails.

Before physical/reference and exact phase-null work, V3 attaches post-search
catalogue metadata, requires actual event-time overlap before calling a signal
an alias, and checks whether the evidence is dominated by events at missing-data
edges. Every short-circuited row remains in the candidate ledger with its locked
ephemeris and reason. Detailed upstream event measurements are reused rather
than recomputed.

The CPU path runs the same search grid and review contract through the native
exact CPU kernels. Approximate, top-k, reduced-grid, and cuvarbase modes are not
Avril V3.0. Two- and three-event signals remain auditable manual-review
candidates; they are not silently deleted or advertised as automatic
high-priority discoveries.

The release is accepted only after the same manifest and FITS products pass
cross-platform comparison. The contract is equality of high-priority periods,
locked ephemerides, and scientific review decisions, with at least 98% Mac
candidate coverage. Raw floating-point boundary maxima need not be bit-identical
across ARM, x86, and CUDA.

`scripts/build_avril_v3_release.py` creates separate immutable algorithm and web
archives. It records per-file and archive SHA-256 hashes and refuses to
overwrite an existing release directory.
