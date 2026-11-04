# Contributing

Changes to Avril's scientific path need more than a unit test.

1. Open an issue describing the scientific failure mode and intended invariant.
2. Add deterministic unit tests and a real public TESS regression target.
3. Do not tune against an opened blind set.
4. Run CPU tests and, for numerical-kernel changes, the same-input CPU/CUDA
   comparison in `docs/REPRODUCIBILITY.md`.
5. Record runtime, candidate coverage, priority disagreements, and known-signal
   recovery. A speed change must disclose any reduction in computation.
6. Never describe an unmatched periodic signal as a new planet without pixel
   localization and independent scientific follow-up.

Generated data, FITS files, environments, caches, and result directories must
not be committed. Keep large data outside the repository and bind it to runs by
manifest and SHA-256.
