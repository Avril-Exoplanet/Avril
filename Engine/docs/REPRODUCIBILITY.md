# Reproducibility and release verification

## Freeze inputs before computation

```bash
avril manifest "TOI-786" --output manifests/toi786.json
shasum -a 256 manifests/toi786.json
```

Run the same manifest on both systems. Do not independently query MAST on each
machine because archive contents can change.

```bash
AVRIL_DATA_DIR=/data/avril avril run manifests/toi786.json --backend cpu
AVRIL_DATA_DIR=/data/avril avril run manifests/toi786.json --backend cuda
```

The run directory contains `avril_run_receipt.json`, the manifest hash,
execution profile, full logs, stage results, and final candidate ledgers.

## Cross-platform acceptance

Use the bundled comparator on the two final `result.json` files:

```bash
python scripts/compare_vlora_cross_platform.py \
  --mac /path/to/mac/result.json \
  --cuda /path/to/cuda/result.json \
  --known-period 12.669431052787182 \
  --known-period 38.553883319802566 \
  --output validation.json
```

V3.0 requires:

- at least 98% Mac candidate coverage;
- zero priority disagreements among matched candidates;
- identical high-priority candidate count and ephemerides within tolerance;
- recovery of declared regression periods on both systems;
- faster full-pipeline CUDA wall time for the acceptance target.

Known periods are used only after both label-free runs are sealed. They must not
enter search, ranking, or parameter selection.

## Tests

```bash
pytest -q avril \
  multiplanet_trial/live_web/test_cuda_hybrid_search.py \
  multiplanet_trial/live_web/test_cuda_science_acceleration.py \
  multiplanet_trial/live_web/test_exact_bls_acceleration.py \
  multiplanet_trial/live_web/test_detection_significance.py \
  multiplanet_trial/live_web/test_post_discovery_vetting.py \
  multiplanet_trial/live_web/test_rational_aliases.py \
  multiplanet_trial/live_web/test_harmonic_families.py \
  multiplanet_trial/live_web/test_event_coherence.py \
  multiplanet_trial/live_web/test_science_review.py \
  multiplanet_trial/live_web/test_fast_science_finalize.py \
  multiplanet_trial/live_web/test_meaningful_candidate_gate.py
```

## Release archives

`scripts/build_avril_v3_release.py` creates immutable algorithm and web archives.
It refuses to overwrite an existing output directory and records SHA-256 for
every file plus the final archives.

Verify either archive without installing Avril:

```bash
python scripts/verify_avril_release.py /path/to/Avril-V3.0-Algorithm.zip
python scripts/verify_avril_release.py /path/to/Avril-V3.0-Web.zip
```
