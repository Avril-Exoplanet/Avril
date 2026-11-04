# Releasing Avril

Source publication and binary distribution have different gates.

## Source release gate

Run:

```bash
./scripts/check_release.sh
```

The check rejects machine-specific paths, common credentials, generated FITS/log/cache files, Swift build failures, Python syntax errors, and missing governance documents.

Before tagging a release:

1. Verify every asset and vendored component against `THIRD_PARTY_NOTICES.md`.
2. Freeze target/model artifacts and update SHA-256 hashes.
3. Run deterministic CPU tests and the documented CPU/CUDA same-input comparison for numerical-kernel changes.
4. Record candidate-count, locked-ephemeris, priority, and runtime comparisons.
5. Update `CITATION.cff`, `CHANGELOG.md`, and the research-preview status.
6. Generate the release from a clean checkout.

GitHub repository setup should also use `docs/assets/github-social-preview.png`
as the social preview image and enable private vulnerability reporting. The
preview image is publication artwork; it is not a screenshot or a scientific
result.

## macOS binary gate

A downloadable `.app` or `.dmg` must additionally:

- embed a relocatable Python runtime containing the pinned engine dependencies;
- package the engine from the same commit as the UI;
- build for the declared Apple Silicon architecture(s);
- be signed with the release identity and notarized by Apple;
- pass Gatekeeper verification on a clean Mac account;
- start with networking disabled and show a useful engine/readiness error;
- complete one public non-sensitive smoke target and export JSON/PDF successfully;
- contain no developer paths, caches, FITS data, logs, tokens, or unpublished results.

Pass the runtime directory explicitly:

```bash
AVRIL_NOVA_PYTHON_RUNTIME=/path/to/relocatable-python \
AVRIL_NOVA_BUILD_ROOT=/path/to/release-build \
./scripts/build_app.sh
```

Ad-hoc signing performed by the development script is for local testing only.
