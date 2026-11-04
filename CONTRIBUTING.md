# Contributing to Avril

Avril accepts interface, documentation, performance, and scientific-path contributions. Scientific correctness takes precedence over benchmark speed or visual simplicity.

## Before opening a change

Open an issue for changes that alter candidate generation, detrending, grids, voting, refitting, event/gap logic, physical vetting, phase-null evaluation, ranking, target priors, or artifact schemas. State the failure mode and the invariant the change is intended to preserve.

Small documentation, accessibility, and isolated UI fixes may go directly to a pull request.

## Development setup

```bash
./scripts/bootstrap_engine.sh cpu
swift build
source .venv/bin/activate
pytest -q Engine/avril Engine/multiplanet_trial/live_web
```

Run the public release audit before requesting review:

```bash
./scripts/check_release.sh
```

## Scientific-path requirements

A scientific-path pull request must:

1. add deterministic unit tests for the failure mode;
2. include at least one public TESS regression target when the behavior cannot be proven synthetically;
3. avoid tuning against an opened blind set;
4. report candidate count, locked ephemeris differences, priority disagreements, failures, and runtime;
5. run the documented same-input CPU/CUDA comparison for numerical-kernel changes;
6. disclose every reduction in grid density, trial count, event coverage, or computation;
7. preserve skipped/deferred candidates and explicit reasons in the audit ledger.

A speedup may not be described as equivalent if it silently removes scientific work.

## Target-prior requirements

Every model update needs a new frozen data manifest, SHA-256 hashes, feature list, label definition, star-level split rule, out-of-fold metrics, and domain-shift statement. All planets around one star must remain in one fold. Avril outputs and hand-selected successes must not leak into a claimed blind validation split.

## User-interface requirements

- Keep the workstation native; do not add HTML/WebView runtime dependencies.
- Preserve keyboard access, readable minimum type sizes, stable pane geometry, and native macOS window behavior.
- Scientific plots must show units, data provenance, masks/quality state, and the distinction between raw, detrended, and phase-folded views.
- UI labels must not turn ranking priors or candidate tiers into planet probabilities.
- Long work must remain cancellable and must not block the main actor.

## Data hygiene

Never commit FITS products, run outputs, unpublished target lists, credentials, engine logs, virtual environments, caches, or machine-specific paths. Public regressions should be reproducible from a manifest and public archive identifiers.

## Pull requests

Use a focused title and include:

- problem and scientific/UI impact;
- implementation summary;
- validation commands and results;
- screenshots for visible UI changes;
- performance and memory changes;
- artifact/schema compatibility notes;
- any new third-party code, model, font, image, or license.
