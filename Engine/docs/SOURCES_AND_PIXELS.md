# Photometry sources and pixel policy

Avril has two scientifically distinct input pipelines. They share downstream
candidate principles, but they are not interchangeable file formats and their
provenance is recorded separately.

## Official SPOC

`--source spoc` queries MAST for official SPOC or TESS-SPOC light-curve
products, selects one preferred cadence per sector, and downloads all selected
sectors with up to 80 concurrent file transfers. It searches the official
PDCSAP light curve with the Avril deterministic vote, exact-refit, red-noise,
and physical-review pipeline. This is the preferred path when at least three
official sectors exist.

```bash
avril run "TOI-786" --source spoc --backend auto
```

## Raw FFI self-photometry

`--source ffi` is for targets without enough official light curves, including
fainter targets. It does **not** treat a TESSCut file as a SPOC light curve.
Instead it:

1. resolves a TIC identity and sky coordinate;
2. discovers every available TESSCut sector;
3. downloads resumable 15 x 15 calibrated FFI cutouts (8 files in parallel,
   16 HTTP ranges per file by default: at most 128 streams);
4. estimates the local background per cadence;
5. measures fixed apertures of 1.5, 2.0, 2.5 and 3.0 pixels;
6. selects one global core aperture using label-free point-to-point scatter;
7. retains all aperture columns for audit while selecting one global aperture;
8. runs that light curve through the optimized Avril Core search, exact refit,
   red-noise tests, physical review and complete candidate ledger.

No period, epoch, duration, catalogue label, or injected truth is used to pick
the aperture. `core` is the default for every FFI target and uses the same
current CPU/CUDA execution selection as official-light-curve runs.

```bash
avril run "TIC 123456789" --source ffi
```

### Sparse-sector Avril Pro

Avril Pro is opt-in and the executor permits it only when the actual prepared FFI
input contains three or four sectors:

```bash
avril run "TIC 123456789" --source ffi --ffi-engine pro
```

This is not the legacy cross-view implementation. It starts from the same
optimized Avril code as Core. The only scientific variation is a stricter
full-baseline test before nearby period hypotheses may share a family: clocks
that drift by more than half a transit duration remain separate and receive
their own exact validation/refit. All raw peaks remain in the audit. At five or
more sectors, use Core; an explicit Pro request is rejected before the
large TESSCut download begins.

`--source auto` first tries the complete official SPOC inventory and falls back
to FFI only when fewer than three official sectors are available. A saved
public manifest always means the frozen SPOC path and never silently changes
source.

## Two meanings of “pixels”

Raw FFI cutouts are mandatory input to FFI self-photometry. They cannot be
disabled because the light curve is measured from them. This is different from
post-discovery pixel localization.

Post-discovery localization is off by default for both sources:

```bash
avril run "TOI-786" --pixels none
```

With `--pixels followup`, Avril first finishes discovery, exact refitting,
event repeatability, red-noise significance, and physical checks. It then:

- downloads official target-pixel files only if the SPOC result already has a
  high-priority follow-up candidate; or
- reuses the existing TESSCut cutouts for final FFI engine candidates.

Pixel follow-up never deletes candidates or changes period, epoch, duration,
or immutable `discovery_rank`. It may change review priority and therefore
recomputes the public `science_rank`. The original no-pixel result stays
immutable beside a separate pixel-follow-up result.

## Storage

FFI cutouts can be much larger than light curves. Set `AVRIL_DATA_DIR` to a
large disk before starting. On the development Mac, the default is
the OS-native Avril application-data directory when `$AVRIL_DATA_DIR` is unset.
