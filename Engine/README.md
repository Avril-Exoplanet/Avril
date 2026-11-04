# Avril V3.0

Avril is an evidence-preserving transit-candidate search engine for public TESS
data. It supports official SPOC light curves and a separate raw-FFI
self-photometry path for targets without sufficient SPOC coverage.

Avril reports **candidates**, not confirmed planets. Its ranking is not a planet
probability. Catalogue matching, target-pixel localization, neighbour checks,
independent follow-up, and scientific review remain necessary.

## What changed in V3.0

V3 keeps V2's complete-grid discovery and exact refit contract, then adds a
meaningful-candidate funnel before expensive follow-up:

- BLS-first Scout can skip quiet targets; TLS is a fallback router only;
- known catalogue periods are attached strictly after blind discovery;
- harmonic and side-band aliases are removed from deep follow-up only when
  their actual supported event times prove the relationship;
- event-incoherent and gap-edge-dominated signals remain in the audit ledger
  but do not run expensive red-noise, physical, or pixel vetting;
- only unexplained, repeated, non-gap-dominated candidates enter full vetting;
- event repeatability is compiled and evaluated in batches;
- physical/reference vetting uses bandwidth-aware multi-process batches;
- exact 4,096-trial phase-null evaluation runs only after deterministic event,
  physical and reference gates;
- a resolved harmonic loser is deferred only when its **direct** winner is
  itself viable and passes exact phase-null significance; otherwise the loser
  is evaluated as a fallback;
- two- and three-event signals remain in the ledger for manual review, while
  automatic high priority requires at least four supported events;
- every candidate, locked period, epoch, duration and skip reason remains in
  the audit output.

## Production pipeline

The production path is fixed and auditable:

1. select one preferred official SPOC/TESS-SPOC light curve per sector;
2. clean and detrend each sector without using catalogue truth;
3. search three complete temporal blocks and preserve every base peak;
4. vote across five deterministic sector-resampling seeds;
5. group near-duplicate and integer/rational harmonic event trains;
6. run exact local period/epoch refitting for promoted families;
7. measure event-direction repeatability and reject mixed-sign event trains;
8. attach post-search catalogue matches and test every supported event for
   missing-cadence/gap-edge domination;
9. annotate odd/even depth, secondary eclipse, binary shape, duration, and
   stellar-parameter consistency only for meaningful unexplained candidates;
10. run 4,096 exact phase-shift/red-noise trials only for candidates still able
   to reach automatic high priority;
11. retain immutable discovery ranks, then publish a science rank that places
    automated-review passes ahead of false-alarm-risk candidates;
12. retain a complete candidate ledger, skip reason, and timing receipt.

For FFI targets, Avril first downloads all TESSCut sectors and performs
label-free multi-aperture photometry. The globally selected aperture then runs
through the same optimized Avril Core pipeline by default; raw pixels are never
pretended to be official SPOC PDCSAP products.

Avril Pro is an explicit sparse-sector option and is accepted only when the
actual FFI coverage has three or four sectors. It is not the old multiview
engine. It keeps the current Avril search, exact refit, red-noise and physical
review, but splits nearby period hypotheses when their clocks would drift apart
over the full observing baseline. No hypothesis is deleted by this change.

The CPU and CUDA paths use the same thresholds and final CPU multi-process
vetting. CUDA changes numerical execution only; it does not truncate the period
grid.

## Quick start

### One-click desktop start

Extract `Avril-V3.0-Algorithm.zip` and `Avril-V3.0-Web.zip` side by side.

- macOS: double-click **Install Avril.command** once, then double-click
  **Start Avril.command**.
- Windows CPU: double-click **Install-Avril.bat** once, then
  **Start-Avril.bat**. WSL2 remains the recommended CUDA path.
- Linux: run `./start_avril.sh`; it installs the CPU environment on first use.

The start scripts serve the bundled, prebuilt static interface on loopback and
open it in the default browser. Computation and FITS data stay on the local
machine.

### Terminal setup

Python 3.11 or 3.12 is recommended. From a source checkout:

```bash
./setup_avril.sh cpu
source .venv/bin/activate
avril doctor --backend cpu
avril run "TOI-786" --backend cpu
```

On Linux/WSL2 with an NVIDIA driver and CUDA 12 toolkit (including `nvcc`):

```bash
./setup_avril.sh cuda
source .venv/bin/activate
avril doctor --backend cuda
avril run "TOI-786" --backend cuda
```

`--backend auto` selects CUDA only when `nvidia-smi`, `nvcc`, CuPy, and PyCUDA
are all available. Otherwise it runs the complete CPU path.

Source and pixel policy are explicit:

```bash
# Preferred official light curves
avril run "TOI-786" --source spoc --pixels none

# Raw FFI cutouts -> self-photometry -> optimized Avril Core (default)
avril run "TIC 123456789" --source ffi

# Optional only when the archive actually contains fewer than 5 sectors
avril run "TIC 123456789" --source ffi --ffi-engine pro

# Try complete SPOC first; use FFI only if coverage is insufficient.
# Pixel files are localized only after a candidate passes science review.
avril run "TIC 123456789" --source auto --pixels followup
```

Pixel localization is off by default. FFI still requires raw TESSCut pixels to
construct its light curves; that input step is distinct from optional
post-discovery localization.

By default data is stored in `$AVRIL_DATA_DIR`. When that variable is unset,
Avril uses the OS-native application-data location: `~/Library/Application
Support/Avril` on macOS, `%LOCALAPPDATA%\\Avril` on Windows, and
`$XDG_DATA_HOME/avril` (or `~/.local/share/avril`) on Linux.

## Batch and reproducible runs

Freeze a label-free manifest first:

```bash
avril manifest "TIC 375059587" --output manifests/toi786.json
avril run manifests/toi786.json --backend auto
```

Every run writes the exact manifest and hash, selected hardware profile, all
scientific parameters, stage logs and timings, complete/deferred target lists,
and an unabridged candidate ledger.

Targets whose exact refit workload exceeds the default safety limit are marked
`deferred_explosive`; their candidates are not discarded. Increase
`--max-refit-families` and `--max-refit-period-trials` to resume them.

## V3 validation status

V3 is currently a release candidate. Its meaningful-candidate funnel has unit
and real-target regression coverage, but the production Scout gate remains
disabled until a larger blind routing-recall calibration is complete.

On TIC 67630845, the native V3 stage retained all 304 locked candidates, routed
283 unexplained event-coherent candidates to physical/reference vetting, and
short-circuited 21 audit-retained rows (one direct catalogue period, eight
event-proven catalogue aliases, and twelve incoherent event trains). Reusing
the upstream per-event ledger reduced funnel time from 5.009 s to 0.154 s and
science-finalization wall time from 15.036 s to 7.941 s. All 304 scientific
decisions, three final high-priority candidates, ranks, and locked ephemerides
were identical. This target contained no signal that met the strict
gap-dominated criterion; synthetic gap routing is covered by unit tests.

## V2 baseline snapshot

The release gate used cached, complete 2-minute SPOC products. Network download
and optional target-pixel follow-up are excluded from compute time.

| Target | Sectors | V2 compute | Required recovery |
|---|---:|---:|---|
| L98-59 | 27 | 261.94 s | 2.253126, 3.690682, 7.450741 d; exactly these three automatic high-priority |
| TOI-786 | 43 | 218.11 s | 12.669431 and refit-rescued 38.553883 d, ranks 1-2 |
| TOI-125 | 7 | 311.53 s staged; lower when light-curve loading is reused | 4.651669, 9.155244, 19.981423 d, ranks 1-3 |
| TOI-1269 | 42 | 384.18 s | 4.252996 and 9.237794 d, ranks 1-2 |

Candidate count and locked ephemerides are invariants. These are engineering
and known-target regressions, not population-level occurrence-rate or
false-positive calibration.

## Documentation

- [Architecture and scientific contract](docs/ARCHITECTURE.md)
- [Installation and device profiles](docs/INSTALLATION.md)
- [Performance tuning](docs/PERFORMANCE.md)
- [Pitfalls and failure modes](docs/PITFALLS.md)
- [Reproducibility and release verification](docs/REPRODUCIBILITY.md)
- [Scientific scope and evaluation rules](docs/SCIENTIFIC_SCOPE.md)
- [Web/engine interface](docs/WEB_INTERFACE.md)
- [SPOC, FFI self-photometry, and pixel policy](docs/SOURCES_AND_PIXELS.md)

## Development

```bash
python -m pip install -e '.[dev]'
pytest -q
ruff check avril vlora/web_server.py run_avril_v3_science_batch.py \
  run_avril_v3_threshold_round.py run_avril_v3_science_finalize.py
```

The public release is split into two archives. `Avril-V3.0-Algorithm` contains
the full scientific engine and no web dependencies. `Avril-V3.0-Web` contains
the interface and Cloudflare deployment project and no scientific engine.

## Licence and citation

Avril is released under the Apache License 2.0. See [LICENSE](LICENSE),
[CITATION.cff](CITATION.cff), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
