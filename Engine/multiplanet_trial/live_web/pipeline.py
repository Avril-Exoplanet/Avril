"""The active live pipeline: three searches, one ledger, one validation pass.

The orchestration in this module is intentionally small and non-destructive:

1. split the observations into three disjoint chronological blocks;
2. search every block independently with BLS and event-first;
3. retain every raw hypothesis in an append-only audit ledger;
4. form candidates only from period+phase matches between different blocks;
5. measure every matched ephemeris unchanged in all three blocks;
6. within each shared-evidence uncertainty family, select the ephemeris that
   maximises the weakest fold S/N; and
7. report per-event repeatability, then compress proven integer aliases.

No fold receives its own epoch adjustment, and per-event hit fractions do not
tune or reject the chosen ephemeris.  The previous experimental orchestration lives in
``legacy_pipeline.py`` solely so old frozen receipts remain reproducible.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import re
import time
import urllib.parse
import urllib.request
import warnings
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path

import lightkurve as lk
import numpy as np
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares

from multiplanet_trial.live_web.sealed_validation import event_repeatability_metrics
from multiplanet_trial.live_web.candidate_tiers import apply_candidate_tiers
from multiplanet_trial.live_web.detection_significance import (
    annotate_detection_significance,
)
from multiplanet_trial.live_web.post_discovery_vetting import (
    annotate_locked_candidates,
)
from multiplanet_trial.live_web.temporal_consensus import (
    candidate_match_details,
    select_temporal_blocks,
)
from multiplanet_trial.v03.archive import (
    _download_one_resumable,
    configure_lightkurve_cache,
    discover_products,
    serializable_inventory,
)
from multiplanet_trial.v03.lightkurve_evidence import measure_fixed_ephemeris
from multiplanet_trial.v07_eval.multibranch_search import (
    SearchBranch,
    _prepare_lightcurve,
    event_first_candidates,
    search_branch_bls,
)


Progress = Callable[[str, int, str, dict | None], None]
DEFAULT_ROOT = Path.home() / ".local" / "share" / "avril"
STRONG_SNR = 6.0
WEAK_SNR = 3.0
DEFAULT_DOWNLOAD_WORKERS = 80
STATUS_ZH = {
    "CP": "Confirmed planet",
    "KP": "Known planet or object",
    "PC": "Planet candidate",
    "FP": "Known false positive",
    "FA": "Known false alarm",
}


def _progress(
    callback: Progress | None,
    stage: str,
    percent: int,
    message: str,
    **detail,
) -> None:
    if callback:
        callback(stage, percent, message, detail or None)


def _json_number(value, digits: int = 7):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "target"


def _segment_metadata(path: Path) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(path, memmap=True) as hdul:
            primary = hdul[0].header
            table_header = hdul[1].header
            table = hdul[1].data
            segment = next(
                (
                    int(primary[key])
                    for key in ("SECTOR", "QUARTER", "CAMPAIGN")
                    if primary.get(key) is not None
                ),
                -1,
            )
            names = set(table.names)
            if "TIME" in names:
                time_column = "TIME"
            elif "TMID_BJD" in names:
                # T16 stores barycentric time as BJD - 2457000, the same
                # numerical zero-point used by the active TESS pipeline.
                time_column = "TMID_BJD"
            else:
                raise ValueError(f"No supported time column in {path.name}")
            time_values = np.asarray(table[time_column], dtype=float)
            good = np.isfinite(time_values)
            if "QUALITY" in names:
                good &= np.asarray(table["QUALITY"], dtype=np.int64) == 0
            clean = time_values[good]
            if not clean.size:
                raise ValueError(f"No finite cadences in {path.name}")
            tic = primary.get("TICID") or table_header.get("TICID")
            return {
                "segment": segment,
                "tic_id": int(tic) if tic is not None else None,
                "object": str(primary.get("OBJECT") or table_header.get("OBJECT") or ""),
                "start": float(np.nanmin(clean)),
                "end": float(np.nanmax(clean)),
                "time_column": time_column,
            }


def _product_priority(product: Mapping, *, deep: bool) -> tuple:
    author_priority = {
        "SPOC": 0,
        "TESS-SPOC": 1,
        "Kepler": 0,
        "K2": 0,
    }.get(str(product.get("author") or ""), 5)
    try:
        exposure = float(product.get("exptime") or 1e12)
    except (TypeError, ValueError):
        exposure = 1e12
    cadence_priority = 0 if deep or 60 <= exposure <= 600 else 1
    return (
        cadence_priority,
        author_priority,
        exposure if deep else abs(exposure - 120.0),
        str(product.get("productFilename") or ""),
    )


def _lightcurve_product_priority(product: Mapping) -> tuple:
    """Prefer native two-minute photometry without discarding any sector.

    A native 120-second product avoids both the extra I/O of 20-second files
    and a scientifically unnecessary pre-processing pass.  When it is not
    available, faster products are retained and exposure-binned to 120 seconds
    before detrending.  Slower products remain a final, native-cadence
    fallback; they are never upsampled.
    """
    author_priority = {
        "SPOC": 0,
        "TESS-SPOC": 1,
        "Kepler": 0,
        "K2": 0,
    }.get(str(product.get("author") or ""), 5)
    try:
        exposure = float(product.get("exptime") or 1e12)
    except (TypeError, ValueError):
        exposure = 1e12
    if 90.0 <= exposure <= 150.0:
        cadence_class = 0
        cadence_distance = abs(exposure - 120.0)
    elif exposure < 90.0:
        cadence_class = 1
        cadence_distance = abs(exposure - 120.0)
    elif np.isfinite(exposure) and exposure < 1e11:
        cadence_class = 2
        cadence_distance = abs(exposure - 120.0)
    else:
        cadence_class = 3
        cadence_distance = exposure
    return (
        cadence_class,
        author_priority,
        cadence_distance,
        str(product.get("productFilename") or ""),
    )


def _select_lightcurves(
    products: Iterable[Mapping],
    max_segments: int | None,
    *,
    deep: bool = False,
) -> list[dict]:
    """Select one deterministic light curve per observing segment."""
    grouped: dict[str, list[dict]] = {}
    for source in products:
        if source.get("product_kind") != "lightcurve":
            continue
        row = dict(source)
        author = str(row.get("author") or "").upper()
        try:
            exposure = float(row.get("exptime") or float("nan"))
        except (TypeError, ValueError):
            exposure = float("nan")
        # Standard TESS SPOC mode accepts native products through 600 s.
        # Thirty-minute FFIs belong to the explicit FFI source path, where
        # aperture extraction and background handling are different.
        if author in {"SPOC", "TESS-SPOC"} and np.isfinite(exposure) and exposure > 600.0:
            continue
        sequence = row.get("sequence_number")
        key = str(sequence if sequence is not None else row.get("mission") or row.get("productFilename"))
        grouped.setdefault(key, []).append(row)
    selected = [
        min(rows, key=_lightcurve_product_priority)
        for rows in grouped.values()
    ]
    selected.sort(
        key=lambda row: (
            float(row.get("sequence_number"))
            if row.get("sequence_number") is not None
            else 1e12,
            str(row.get("mission") or ""),
        )
    )
    if max_segments is None or max_segments <= 0 or len(selected) <= max_segments:
        return selected
    indices = sorted(
        {
            int(round(value))
            for value in np.linspace(0, len(selected) - 1, max_segments)
        }
    )
    return [selected[index] for index in indices]


def _select_target_pixels(
    products: Iterable[Mapping],
    selected_segments: Iterable[int],
    *,
    deep: bool = False,
) -> list[dict]:
    """Select one deterministic target-pixel product per searched segment."""
    wanted = {int(value) for value in selected_segments}
    grouped: dict[int, list[dict]] = {}
    for source in products:
        if source.get("product_kind") != "target_pixel":
            continue
        sequence = source.get("sequence_number")
        if sequence is None:
            continue
        try:
            segment = int(sequence)
        except (TypeError, ValueError):
            continue
        if segment not in wanted:
            continue
        grouped.setdefault(segment, []).append(dict(source))
    selected = [
        min(rows, key=lambda row: _product_priority(row, deep=deep))
        for _segment, rows in sorted(grouped.items())
    ]
    selected.sort(key=lambda row: int(row["sequence_number"]))
    return selected


def _discover(target: str, mission: str, *, lightcurve_source: str = "spoc") -> dict:
    if lightcurve_source == "t16":
        if mission != "TESS":
            raise ValueError("T16 light curves are available only for TESS")
        return discover_products(
            target,
            missions=("TESS",),
            author=("T16",),
            exptime=None,
            include_lightcurves=True,
            include_target_pixels=False,
            limit=None,
        )
    author = ("SPOC", "TESS-SPOC") if mission == "TESS" else mission
    inventory = discover_products(
        target,
        missions=(mission,),
        author=author,
        exptime=None,
        include_lightcurves=True,
        include_target_pixels=False,
        limit=None,
    )
    if inventory["counts"]["lightcurves"]:
        return inventory
    return discover_products(
        target,
        missions=(mission,),
        author=None,
        exptime=None,
        include_lightcurves=True,
        include_target_pixels=False,
        limit=None,
    )


def _discover_target_pixels(target: str, mission: str) -> dict:
    """Discover pixel products separately so an archive failure cannot stop search."""
    author = ("SPOC", "TESS-SPOC") if mission == "TESS" else mission
    try:
        inventory = discover_products(
            target,
            missions=(mission,),
            author=author,
            exptime=None,
            include_lightcurves=False,
            include_target_pixels=True,
            limit=None,
        )
        if inventory["counts"]["target_pixels"]:
            return inventory
        return discover_products(
            target,
            missions=(mission,),
            author=None,
            exptime=None,
            include_lightcurves=False,
            include_target_pixels=True,
            limit=None,
        )
    except Exception as exc:
        return {
            "target": str(target),
            "missions_requested": [mission],
            "products": [],
            "counts": {"lightcurves": 0, "target_pixels": 0, "total": 0},
            "diagnostic": {
                "stage": "target_pixel_discovery",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            },
        }


def _download_products(
    products: list[dict],
    cache_root: Path,
    progress: Progress | None,
    *,
    max_workers: int = DEFAULT_DOWNLOAD_WORKERS,
) -> list[Path]:
    """Download light curves concurrently while preserving segment order."""
    destination = configure_lightkurve_cache(cache_root) / "archive-products"
    if not products:
        return []
    paths: list[Path | None] = [None] * len(products)
    workers = max(1, min(int(max_workers), len(products)))
    targets = []
    for product in products:
        mission_name = str(product.get("mission") or "unknown").split()[0].lower()
        targets.append(
            destination
            / mission_name
            / "lightcurve"
            / str(product["productFilename"])
        )

    def transfer_metrics(verified: int) -> dict:
        downloaded = 0
        total = 0
        active = 0
        for product, target in zip(products, targets):
            expected = int(product.get("size") or 0)
            partial = target.with_suffix(target.suffix + ".part")
            complete = target.exists()
            local_size = target.stat().st_size if complete else (
                partial.stat().st_size if partial.exists() else 0
            )
            # MAST can report an unpacked/estimated product size that is much
            # larger than the transferred FITS file.  Once a resumable fetch
            # has atomically completed, its on-disk size is the authoritative
            # transfer total; pending files continue to use the inventory size.
            total += local_size if complete else expected
            if partial.exists() and local_size > 0:
                active += 1
            downloaded += local_size if complete else (
                min(local_size, expected) if expected else local_size
            )
        return {
            "files_verified": verified,
            "files_total": len(products),
            "files_active": active,
            "bytes_downloaded": downloaded,
            "bytes_total": total or None,
            "download_percent": (
                round(100.0 * downloaded / total, 1) if total else None
            ),
            "parallel_workers": workers,
        }

    initial_metrics = transfer_metrics(0)
    _progress(
        progress,
        "downloading",
        8,
        f"Downloading and validating {len(products)} light-curve segments with {workers} concurrent workers",
        title="Preparing Stage 1 light curves",
        command=f"archive fetch --resume --parallel {workers} --kind lightcurve",
        metrics=initial_metrics,
    )

    def download_one(product: Mapping) -> Path:
        mission_name = str(product.get("mission") or "unknown").split()[0].lower()
        return _download_one_resumable(
            product, destination / mission_name / "lightcurve"
        )

    completed = 0
    last_reported_bytes = int(initial_metrics["bytes_downloaded"])
    last_reported_at = time.monotonic()
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="lightcurve-download"
    ) as pool:
        futures = {
            pool.submit(download_one, product): (index, product)
            for index, product in enumerate(products)
        }
        pending = set(futures)
        while pending:
            done, pending = wait(
                pending, timeout=1.0, return_when=FIRST_COMPLETED
            )
            last_product = None
            for future in done:
                index, product = futures[future]
                paths[index] = future.result()
                completed += 1
                last_product = product
            metrics = transfer_metrics(completed)
            now = time.monotonic()
            byte_step = max(
                1_000_000,
                int((metrics.get("bytes_total") or 0) / 100),
            )
            should_report = bool(done) or (
                int(metrics["bytes_downloaded"]) - last_reported_bytes >= byte_step
            ) or now - last_reported_at >= 5.0
            if not should_report:
                continue
            fraction = (
                float(metrics["bytes_downloaded"])
                / float(metrics["bytes_total"])
                if metrics.get("bytes_total")
                else completed / max(1, len(products))
            )
            _progress(
                progress,
                "downloading",
                8 + int(12 * min(1.0, fraction)),
                (
                    f"Light curves: {completed}/{len(products)} verified; "
                    f"{metrics['download_percent'] or 0:.1f}% of bytes ready"
                ),
                title=f"Light-curve transfer {completed}/{len(products)} verified",
                command=(
                    f"archive fetch --resume --product {last_product.get('productFilename')}"
                    if last_product is not None
                    else "archive fetch --resume --monitor-bytes"
                ),
                metrics=metrics,
            )
            last_reported_bytes = int(metrics["bytes_downloaded"])
            last_reported_at = now
    return [path for path in paths if path is not None]


def _download_target_pixels(
    products: list[dict],
    cache_root: Path,
    progress: Progress | None,
    *,
    max_workers: int = DEFAULT_DOWNLOAD_WORKERS,
    progress_stage: str = "pixel_prefetch",
    progress_percent: int = 24,
) -> tuple[list[Path], list[dict]]:
    """Download selected TPFs concurrently without dropping candidates on failure."""
    destination = configure_lightkurve_cache(cache_root) / "archive-products"
    if not products:
        return [], []
    paths: list[Path | None] = [None] * len(products)
    diagnostics: list[dict] = []
    workers = max(1, min(int(max_workers), len(products)))
    targets = []
    for product in products:
        mission_name = str(product.get("mission") or "unknown").split()[0].lower()
        targets.append(
            destination
            / mission_name
            / "target_pixel"
            / str(product["productFilename"])
        )

    finished_indices: set[int] = set()

    def transfer_metrics(verified: int, completed: int, failed: int) -> dict:
        downloaded = 0
        total = 0
        active = 0
        for index, (product, target) in enumerate(zip(products, targets)):
            expected = int(product.get("size") or 0)
            partial = target.with_suffix(target.suffix + ".part")
            complete = target.exists()
            local_size = target.stat().st_size if complete else (
                partial.stat().st_size if partial.exists() else 0
            )
            total += local_size if complete else expected
            if partial.exists() and local_size > 0 and index not in finished_indices:
                active += 1
            downloaded += local_size if complete else (
                min(local_size, expected) if expected else local_size
            )
        return {
            "files_verified": verified,
            "files_completed": completed,
            "files_failed": failed,
            "files_total": len(products),
            "files_active": active,
            "bytes_downloaded": downloaded,
            "bytes_total": total or None,
            "download_percent": (
                round(100.0 * downloaded / total, 1) if total else None
            ),
            "parallel_workers": workers,
        }

    initial_metrics = transfer_metrics(0, 0, 0)
    _progress(
        progress,
        progress_stage,
        progress_percent,
        f"Prefetching {len(products)} pixel segments with {workers} concurrent workers during Stage 1",
        title="Prefetching Stage 2 pixel data",
        command=f"archive fetch --resume --parallel {workers} --kind target_pixel",
        metrics=initial_metrics,
    )

    def download_one(product: Mapping) -> Path:
        mission_name = str(product.get("mission") or "unknown").split()[0].lower()
        return _download_one_resumable(
            product, destination / mission_name / "target_pixel"
        )

    completed = 0
    verified = 0
    failed = 0
    last_reported_bytes = int(initial_metrics["bytes_downloaded"])
    last_reported_at = time.monotonic()
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="target-pixel-download"
    ) as pool:
        futures = {
            pool.submit(download_one, product): (index, product)
            for index, product in enumerate(products)
        }
        pending = set(futures)
        while pending:
            done, pending = wait(
                pending, timeout=1.0, return_when=FIRST_COMPLETED
            )
            last_product = None
            for future in done:
                index, product = futures[future]
                last_product = product
                try:
                    paths[index] = future.result()
                    verified += 1
                except Exception as exc:
                    failed += 1
                    diagnostics.append(
                        {
                            "segment": product.get("sequence_number"),
                            "file": product.get("productFilename"),
                            "stage": "target_pixel_download",
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                completed += 1
                finished_indices.add(index)
            metrics = transfer_metrics(verified, completed, failed)
            now = time.monotonic()
            byte_step = max(
                2_000_000,
                int((metrics.get("bytes_total") or 0) / 100),
            )
            should_report = bool(done) or (
                int(metrics["bytes_downloaded"]) - last_reported_bytes >= byte_step
            ) or now - last_reported_at >= 5.0
            if not should_report:
                continue
            _progress(
                progress,
                progress_stage,
                progress_percent,
                (
                    f"Pixel files: {verified}/{len(products)} verified; "
                    f"{completed}/{len(products)} completed; {failed} failed; "
                    f"{metrics['download_percent'] or 0:.1f}% of bytes ready"
                ),
                title=(
                    f"Pixel transfer {verified}/{len(products)} verified"
                    + (f" · {failed} failed" if failed else "")
                ),
                command=(
                    f"archive fetch --resume --product {last_product.get('productFilename')}"
                    if last_product is not None
                    else "archive fetch --resume --monitor-bytes"
                ),
                metrics=metrics,
            )
            last_reported_bytes = int(metrics["bytes_downloaded"])
            last_reported_at = now
    diagnostics.sort(key=lambda row: (row.get("segment") is None, row.get("segment")))
    return [path for path in paths if path is not None], diagnostics


def _prefetch_target_pixels(
    target: str,
    mission: str,
    segments: Iterable[int],
    *,
    deep: bool,
    cache_root: Path,
    progress: Progress | None,
    download_workers: int,
) -> tuple[dict, list[dict], list[Path], list[dict]]:
    """Discover and fetch stage-two pixels while stage one searches light curves."""
    pixel_inventory = _discover_target_pixels(target, mission)
    selected = _select_target_pixels(
        pixel_inventory.get("products", []), segments, deep=deep
    )
    paths, diagnostics = _download_target_pixels(
        selected,
        cache_root,
        progress,
        max_workers=download_workers,
    )
    return pixel_inventory, selected, paths, diagnostics


def _submit_background_pixel_prefetch(
    prefetch: Callable,
    *args,
    **kwargs,
):
    """Submit pixel I/O without waiting, so period search can start immediately."""
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pixel-prefetch")
    return pool, pool.submit(prefetch, *args, **kwargs)


def _median_cadence_seconds(time_values: np.ndarray) -> float:
    values = np.asarray(time_values, dtype=np.float64)
    values = np.sort(values[np.isfinite(values)])
    steps = np.diff(values)
    steps = steps[(steps > 0.0) & (steps < 0.02)]
    return float(np.median(steps) * 86400.0) if steps.size else float("nan")


def _prepare_search_cadence(lightcurve, target_seconds: float):
    """Exposure-bin sub-two-minute data before expensive detrending.

    Only finite time/flux pairs participate in the linear mean.  The later
    normalization, one-day flattening and outlier policy remain unchanged.
    Native target-cadence and slower light curves are returned unchanged.
    """
    time_values = np.asarray(lightcurve.time.value, dtype=np.float64)
    flux_values = np.asarray(lightcurve.flux.value, dtype=np.float64)
    good = np.isfinite(time_values) & np.isfinite(flux_values)
    time_values = time_values[good]
    flux_values = flux_values[good]
    order = np.argsort(time_values)
    time_values = time_values[order]
    flux_values = flux_values[order]
    input_cadence_seconds = _median_cadence_seconds(time_values)
    input_cadences = int(time_values.size)
    should_bin = bool(
        np.isfinite(input_cadence_seconds)
        and input_cadence_seconds < 0.75 * float(target_seconds)
    )
    if not should_bin or not input_cadences:
        return lightcurve, {
            "input_cadences": input_cadences,
            "pre_detrend_cadences": input_cadences,
            "median_input_cadence_seconds": input_cadence_seconds,
            "pre_detrend_cadence_seconds": input_cadence_seconds,
            "binned_to_seconds": None,
        }
    target_days = float(target_seconds) / 86400.0
    bin_id = np.floor((time_values - float(time_values[0])) / target_days).astype(
        np.int64
    )
    counts = np.bincount(bin_id).astype(np.float64)
    occupied = counts > 0.0
    binned_time = np.bincount(bin_id, weights=time_values)[occupied] / counts[occupied]
    binned_flux = np.bincount(bin_id, weights=flux_values)[occupied] / counts[occupied]
    prepared = lk.LightCurve(time=binned_time, flux=binned_flux)
    return prepared, {
        "input_cadences": input_cadences,
        "pre_detrend_cadences": int(binned_time.size),
        "median_input_cadence_seconds": input_cadence_seconds,
        "pre_detrend_cadence_seconds": _median_cadence_seconds(binned_time),
        "binned_to_seconds": float(target_seconds),
    }


def _load_pdcsap_branch(paths: list[Path]) -> tuple[SearchBranch, list[dict]]:
    """Load one photometry view and normalize prepared cadences to two minutes.

    Products materially faster than 120 seconds are exposure-binned before the
    unchanged one-day Lightkurve flattening. Native two-minute and slower
    products are preserved sample-for-sample. This prevents sub-two-minute
    products from exploding the workload without fabricating samples in slower
    products.
    """
    pieces: list[tuple[np.ndarray, np.ndarray, int]] = []
    diagnostics: list[dict] = []
    target_seconds = float(os.environ.get("VLORA_SURVEY_CADENCE_SECONDS", "120"))
    for path in paths:
        metadata = _segment_metadata(path)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lightcurve = lk.read(path, flux_column="pdcsap_flux")
            lightcurve, cadence_row = _prepare_search_cadence(
                lightcurve, target_seconds
            )
            time_values, flux_values = _prepare_lightcurve(lightcurve, 1.0)
            order = np.argsort(time_values)
            time_values = np.asarray(time_values, dtype=np.float64)[order]
            flux_values = np.asarray(flux_values, dtype=np.float64)[order]
            pieces.append((time_values, flux_values, int(metadata["segment"])))
            diagnostics.append(
                {
                    "file": path.name,
                    "segment": int(metadata["segment"]),
                    "status": "ready",
                    "cadences": int(time_values.size),
                    **cadence_row,
                    "analysis_cadences": int(time_values.size),
                    "analysis_cadence_seconds": _median_cadence_seconds(time_values),
                    "cadence_preparation_order": "exposure-bin then detrend",
                }
            )
        except Exception as exc:
            diagnostics.append(
                {
                    "file": path.name,
                    "segment": int(metadata["segment"]),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if not pieces:
        raise RuntimeError("All PDCSAP light curves failed to load")
    time_values = np.concatenate([row[0] for row in pieces])
    flux_values = np.concatenate([row[1] for row in pieces])
    segment_ids = np.concatenate(
        [np.full(row[0].size, row[2], dtype=int) for row in pieces]
    )
    order = np.argsort(time_values)
    return (
        SearchBranch(
            key="pdcsap",
            label="PDCSAP",
            processing="normalize + Lightkurve flatten(1 d)",
            photometry_family="spoc_pipeline_aperture",
            method_family="bls+event_first",
            time=time_values[order],
            flux=flux_values[order],
            segment_id=segment_ids[order],
            metadata={
                "target_cadence_seconds": target_seconds,
                "cadence_policy": (
                    "mean-bin only prepared cadences faster than target; "
                    "native target and slower cadences unchanged"
                ),
            },
        ),
        diagnostics,
    )


def _robust_scatter(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 10:
        return math.inf
    median = float(np.nanmedian(finite))
    scatter = 1.4826 * float(np.nanmedian(np.abs(finite - median)))
    return scatter if math.isfinite(scatter) and scatter > 0 else math.inf


def _select_t16_aperture(paths: list[Path]) -> tuple[int, dict[int, float]]:
    """Choose one T16 aperture globally using label-free robust scatter.

    A single aperture is used for every sector so sector-to-sector changes in
    aperture do not masquerade as astrophysical variability. No catalogue
    period, epoch, or transit mask enters this choice.
    """
    per_aperture: dict[int, list[float]] = {1: [], 2: [], 3: []}
    for path in paths:
        with fits.open(path, memmap=True) as hdul:
            table = hdul[1].data
            names = set(table.names)
            for aperture in per_aperture:
                column = f"TFA{aperture}"
                if column in names:
                    scatter = _robust_scatter(np.asarray(table[column], dtype=float))
                    if math.isfinite(scatter):
                        per_aperture[aperture].append(scatter)
    scores = {
        aperture: float(np.nanmedian(values)) if values else math.inf
        for aperture, values in per_aperture.items()
    }
    selected = min(scores, key=scores.get)
    if not math.isfinite(scores[selected]):
        raise ValueError("No usable T16 TFA aperture columns were found")
    return int(selected), scores


def _t16_magnitude_to_relative_flux(magnitude: np.ndarray) -> np.ndarray:
    """Convert a T16 magnitude series to dimensionless relative flux."""
    values = np.asarray(magnitude, dtype=float)
    reference = float(np.nanmedian(values[np.isfinite(values)]))
    return np.power(10.0, -0.4 * (values - reference))


def _prepare_t16_lightcurve(
    time_values: np.ndarray, magnitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the T16 paper's one-pass 7-sigma preparation without re-flattening."""
    time_values = np.asarray(time_values, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    finite = np.isfinite(time_values) & np.isfinite(magnitude)
    if finite.sum() < 10:
        raise ValueError("fewer than 10 finite T16 cadences")
    time_values = time_values[finite]
    relative_flux = _t16_magnitude_to_relative_flux(magnitude[finite])
    center = float(np.nanmedian(relative_flux))
    relative_flux = relative_flux / center
    scatter = _robust_scatter(relative_flux)
    if math.isfinite(scatter) and scatter > 0:
        keep = np.abs(relative_flux - 1.0) <= 7.0 * scatter
        time_values = time_values[keep]
        relative_flux = relative_flux[keep]
    order = np.argsort(time_values)
    return time_values[order], relative_flux[order]


def _load_t16_branch(paths: list[Path]) -> tuple[SearchBranch, list[dict]]:
    """Load T16 difference-image TFA photometry into the active search shape."""
    aperture, aperture_scores = _select_t16_aperture(paths)
    flux_column = f"TFA{aperture}"
    pieces: list[tuple[np.ndarray, np.ndarray, int]] = []
    diagnostics: list[dict] = []
    for path in paths:
        metadata = _segment_metadata(path)
        try:
            with fits.open(path, memmap=True) as hdul:
                table = hdul[1].data
                time_values = np.asarray(table["TMID_BJD"], dtype=float)
                magnitude = np.asarray(table[flux_column], dtype=float)
            clean_time, clean_flux = _prepare_t16_lightcurve(
                time_values, magnitude
            )
            pieces.append((clean_time, clean_flux, int(metadata["segment"])))
            diagnostics.append(
                {
                    "file": path.name,
                    "segment": int(metadata["segment"]),
                    "status": "ready",
                    "cadences": int(clean_time.size),
                    "photometry_source": "T16",
                    "magnitude_column": flux_column,
                }
            )
        except Exception as exc:
            diagnostics.append(
                {
                    "file": path.name,
                    "segment": int(metadata["segment"]),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if not pieces:
        raise RuntimeError("All T16 light curves failed to load")
    time_values = np.concatenate([row[0] for row in pieces])
    flux_values = np.concatenate([row[1] for row in pieces])
    segment_ids = np.concatenate(
        [np.full(row[0].size, row[2], dtype=int) for row in pieces]
    )
    order = np.argsort(time_values)
    return (
        SearchBranch(
            key="t16_tfa",
            label=f"T16 TFA aperture {aperture}",
            processing=(
                f"{flux_column} magnitude to relative flux + normalize + "
                "single 7-sigma clipping; no second flatten"
            ),
            photometry_family="t16_difference_image_tfa",
            method_family="bls+event_first",
            time=time_values[order],
            flux=flux_values[order],
            segment_id=segment_ids[order],
            metadata={
                "photometry_source": "T16",
                "magnitude_column": flux_column,
                "aperture": aperture,
                "aperture_scores_magnitude_mad_sigma": aperture_scores,
                "aperture_selection_used_catalog_signal": False,
            },
        ),
        diagnostics,
    )


def _subset_branch(branch: SearchBranch, segments: Iterable[int]) -> SearchBranch:
    selected = {int(value) for value in segments}
    mask = np.isin(branch.segment_id, list(selected))
    return SearchBranch(
        key=branch.key,
        label=branch.label,
        processing=branch.processing,
        photometry_family=branch.photometry_family,
        method_family=branch.method_family,
        time=branch.time[mask],
        flux=branch.flux[mask],
        segment_id=branch.segment_id[mask],
        can_seed_candidates=branch.can_seed_candidates,
        metadata={**dict(branch.metadata), "temporal_segments": sorted(selected)},
    )


def _strength(candidate: Mapping) -> float:
    metrics = candidate.get("search_metrics", {})
    return float(metrics.get("snr") or 0.0) + 0.35 * float(
        candidate.get("robust_sde") or 0.0
    )


def _search_block(
    branch: SearchBranch,
    *,
    block_index: int,
    segments: list[int],
    period_max: float,
) -> tuple[list[dict], dict]:
    """Search one untouched block and return every accepted hypothesis."""
    local = _subset_branch(branch, segments)
    durations = (0.035, 0.05, 0.07, 0.09, 0.12, 0.16, 0.21, 0.28, 0.36, 0.45)
    bls, bls_diagnostics = search_branch_bls(
        local,
        holdout_segment=None,
        period_min=0.5,
        period_max=float(period_max),
        n_periods=18_000,
        max_candidates=None,
        sde_threshold=6.0,
        snr_threshold=5.0,
        min_events=2,
        durations_days=durations,
        max_seed_periods=None,
        suppress_harmonics=False,
        retain_harmonic_hypotheses=False,
        max_harmonic_hypotheses_per_round=None,
        max_independent_hypotheses_before_masking=None,
        preserve_all_local_peaks=True,
    )
    events, event_diagnostics = event_first_candidates(
        local,
        holdout_segment=None,
        period_min=0.5,
        period_max=float(period_max),
        min_event_snr=3.0,
        exhaustive_hypotheses=True,
    )
    ledger: list[dict] = []
    for method, rows in (("bls", bls), ("event_first", events)):
        for row in rows:
            ledger.append(
                {
                    "raw_id": f"B{block_index + 1}-{method.upper()}-{len(ledger) + 1:04d}",
                    "source_block": block_index + 1,
                    "source_segments": list(segments),
                    "method": method,
                    "period_days": float(row["period_days"]),
                    "epoch_btjd": float(row["epoch_btjd"]),
                    "duration_hours": float(row["duration_hours"]),
                    "robust_sde": _json_number(row.get("robust_sde"), 6),
                    "search_metrics": dict(row.get("search_metrics", {})),
                    "source_strength": _strength(row),
                }
            )
    return ledger, {
        "block": block_index + 1,
        "segments": list(segments),
        "cadences": int(local.time.size),
        "bls_hypotheses": len(bls),
        "event_first_hypotheses": len(events),
        "bls_attempts": len(bls_diagnostics),
        "event_first": event_diagnostics,
    }


def _pair_proposal(first: Mapping, second: Mapping, proof: Mapping) -> dict:
    bridge = bool(proof.get("cycle_connected_within_both_uncertainties"))
    if bridge:
        period = float(proof["cycle_connected_period_days"])
    else:
        period = 0.5 * (float(first["period_days"]) + float(second["period_days"]))
    reference = max((first, second), key=_strength)
    return {
        "period_days": period,
        "epoch_btjd": float(reference["epoch_btjd"]),
        "duration_hours": float(
            np.median([first["duration_hours"], second["duration_hours"]])
        ),
        "source_raw_ids": [first["raw_id"], second["raw_id"]],
        "discovered_blocks": sorted(
            {int(first["source_block"]), int(second["source_block"])}
        ),
        "source_methods": sorted({str(first["method"]), str(second["method"])}),
        "source_strength": max(_strength(first), _strength(second)),
        "match_proofs": [dict(proof)],
    }


def _build_cross_block_proposals(
    ledger: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Create every cross-block match without pre-validation pruning.

    Nearby periods that describe the same apparent event train are intentionally
    kept as separate proposals here.  Their fixed-ephemeris measurements can be
    materially different in another temporal block, so equivalence is decided
    only after every proposal has completed sealed validation.
    """
    matches: list[dict] = []
    raw_proposals: list[dict] = []
    for left_index, first in enumerate(ledger):
        for second in ledger[left_index + 1 :]:
            if first["source_block"] == second["source_block"]:
                continue
            proof = candidate_match_details(first, second)
            if proof is None:
                continue
            proposal = _pair_proposal(first, second, proof)
            raw_proposals.append(proposal)
            matches.append(
                {
                    "first_raw_id": first["raw_id"],
                    "second_raw_id": second["raw_id"],
                    **dict(proof),
                }
            )

    proposals = sorted(
        raw_proposals, key=lambda row: row["source_strength"], reverse=True
    )
    for index, proposal in enumerate(proposals, start=1):
        proposal["proposal_id"] = f"MATCH-{index:04d}"
    return proposals, matches


def _metric_outcome(
    metrics: Mapping, *, independently_discovered: bool = True
) -> str:
    if metrics.get("status") == "insufficient_coverage":
        return "untestable"
    snr = metrics.get("snr")
    depth = metrics.get("depth_ppm")
    if snr is None or depth is None:
        return "untestable"
    if float(depth) > 0 and float(snr) >= STRONG_SNR:
        return "strong"
    if float(depth) > 0 and float(snr) >= WEAK_SNR:
        return "weak"
    # A fold that did not discover the period independently can contain just
    # one scheduled transit.  A weak/non-detection of that single event is not
    # enough evidence to contradict two independent discovery folds.
    if not independently_discovered and int(metrics.get("events") or 0) < 2:
        return "untestable"
    if float(depth) <= 0 or float(snr) < WEAK_SNR:
        return "contradiction"
    return "untestable"


def _measure_all_blocks(
    candidate: Mapping,
    blocks: list[list[int]],
    branch: SearchBranch,
) -> list[dict]:
    rows: list[dict] = []
    discovered_blocks = {
        int(value) for value in candidate.get("discovered_blocks", [])
    }
    if candidate.get("source_block") is not None:
        discovered_blocks.add(int(candidate["source_block"]))
    for block_index, segments in enumerate(blocks, start=1):
        local = _subset_branch(branch, segments)
        metrics = measure_fixed_ephemeris(
            local.time,
            local.flux,
            period_days=float(candidate["period_days"]),
            epoch_btjd=float(candidate["epoch_btjd"]),
            duration_hours=float(candidate["duration_hours"]),
        )
        metrics = dict(metrics)
        metrics["timing_offset_days"] = 0.0
        metrics["timing_tolerance_applied"] = False
        rows.append(
            {
                "block": block_index,
                "segments": list(segments),
                "independently_discovered": block_index in discovered_blocks,
                "outcome": _metric_outcome(
                    metrics,
                    independently_discovered=block_index in discovered_blocks,
                ),
                **metrics,
            }
        )
    return rows


def _validation_decision(measurements: list[dict]) -> dict:
    strong = [row for row in measurements if row["outcome"] == "strong"]
    weak = [row for row in measurements if row["outcome"] == "weak"]
    contradictions = [
        row for row in measurements if row["outcome"] == "contradiction"
    ]
    positive = [*strong, *weak]
    combined_snr = float(
        np.sqrt(sum(float(row.get("snr") or 0.0) ** 2 for row in positive))
    )
    passed = (
        not contradictions
        and len(positive) >= 2
        and (len(strong) >= 1 or combined_snr >= STRONG_SNR)
    )
    return {
        "passed": passed,
        "outcome": "passed" if passed else "failed",
        "code": (
            "fixed_ephemeris_reproduced"
            if passed
            else "covered_block_contradiction"
            if contradictions
            else "fewer_than_two_supporting_blocks"
        ),
        "block_measurements": measurements,
        "strong_blocks": [int(row["block"]) for row in strong],
        "weak_blocks": [int(row["block"]) for row in weak],
        "untestable_blocks": [
            int(row["block"])
            for row in measurements
            if row["outcome"] == "untestable"
        ],
        "contradicting_blocks": [int(row["block"]) for row in contradictions],
        "sealed_segments_with_strong_support": len(strong),
        "combined_positive_snr": combined_snr,
        "thresholds": {
            "strong_snr": STRONG_SNR,
            "weak_snr": WEAK_SNR,
            "minimum_positive_blocks": 2,
            "covered_contradictions_allowed": 0,
        },
    }


def _validate_raw_ledger(
    ledger: list[dict], blocks: list[list[int]], branch: SearchBranch
) -> list[dict]:
    """Measure every raw hypothesis; none is removed from this ledger."""
    output: list[dict] = []
    for row in ledger:
        measurements = _measure_all_blocks(row, blocks, branch)
        output.append(
            {
                **row,
                "fixed_block_validation": _validation_decision(measurements),
            }
        )
    return output


def _validate_proposals(
    proposals: list[dict], blocks: list[list[int]], branch: SearchBranch
) -> list[dict]:
    output: list[dict] = []
    for proposal in proposals:
        measurements = _measure_all_blocks(proposal, blocks, branch)
        verification = _validation_decision(measurements)
        # Event-by-event diagnostics are evaluated after proposals sharing a
        # raw fold hypothesis have been reduced to their maximin-S/N family
        # representative.  Family selection does not use this diagnostic, so
        # deferring it avoids rescanning all sectors for thousands of
        # alternatives that cannot reach ranking.
        verification["event_repeatability_role"] = (
            "deferred_until_family_selection"
            if verification.get("passed") is True
            else "skipped_after_failed_fixed_ephemeris"
        )
        output.append({**proposal, "holdout_verification": verification})
    return output


def _attach_event_repeatability(
    candidate: Mapping, verification: dict, branch: SearchBranch
) -> None:
    repeatability = event_repeatability_metrics(
        branch,
        holdout_segment=None,
        period_days=float(candidate["period_days"]),
        epoch_btjd=float(candidate["epoch_btjd"]),
        duration_hours=float(candidate["duration_hours"]),
    )
    verification["event_repeatability"] = repeatability
    verification["event_repeatability_role"] = "diagnostic_after_ephemeris_selection"


def _apply_fold_event_direction_test(candidate: dict, blocks: list[list[int]]) -> None:
    """Test a selected ephemeris without tuning it from individual events."""
    verification = candidate["holdout_verification"]
    event_rows = verification.get("event_repeatability", {}).get(
        "event_support", []
    )
    block_by_segment = {
        int(segment): block_index
        for block_index, segments in enumerate(blocks, start=1)
        for segment in segments
    }
    summaries: list[dict] = []
    for block_index in range(1, len(blocks) + 1):
        depths = [
            float(row["depth_ppm"])
            for row in event_rows
            if block_by_segment.get(int(row["segment"])) == block_index
        ]
        if not depths:
            summaries.append(
                {
                    "block": block_index,
                    "testable_events": 0,
                    "outcome": "untestable",
                }
            )
            continue
        count = len(depths)
        positive = sum(depth > 0 for depth in depths)
        # Under no coherent dip, positive and negative event depths are equally
        # likely.  This exact tail probability is reported as evidence; the
        # direction decision itself is simply whether the robust median is a dip.
        sign_probability = sum(
            math.comb(count, value) for value in range(positive, count + 1)
        ) / (2**count)
        median_depth = float(np.median(depths))
        # A single predicted event cannot establish the direction of an
        # entire time block.  Treat it as useful audit evidence, but never as
        # a hard contradiction.  This matters for long-period planets: a
        # third block can contain only one noisy transit while two earlier
        # blocks contain a repeated, coherent event train.
        direction_testable = count >= 2
        summaries.append(
            {
                "block": block_index,
                "testable_events": count,
                "events_with_positive_depth": positive,
                "median_depth_ppm": median_depth,
                "one_sided_sign_probability": sign_probability,
                "outcome": (
                    "dimming"
                    if direction_testable and median_depth > 0
                    else "not_dimming"
                    if direction_testable
                    else "insufficient_event_count"
                ),
            }
        )

    testable = [
        row
        for row in summaries
        if row["outcome"] in {"dimming", "not_dimming"}
    ]
    # This is a veto applied after the candidate already passed fixed-ephemeris
    # validation; it is not another promotion gate.  Absence of two-event
    # direction evidence must therefore remain neutral.  Reject only when a
    # reliably testable block is explicitly non-dimming.
    stable = bool(all(row["outcome"] == "dimming" for row in testable))
    verification["fold_event_direction_test"] = {
        "passed": stable,
        "rule": (
            "positive_median_event_depth_in_every_fold_with_at_least_two_events"
        ),
        "minimum_events_per_direction_fold": 2,
        "decision_role": "veto_only_insufficient_evidence_is_neutral",
        "blocks": summaries,
        "used_to_tune_ephemeris": False,
    }
    if verification.get("passed") is True and not stable:
        verification["passed"] = False
        verification["outcome"] = "failed"
        verification["code"] = "event_depth_direction_not_stable_across_folds"


def _build_single_block_arbitrations(
    raw_registry: list[dict],
    matched_raw_ids: set[str],
    branch: SearchBranch,
) -> list[dict]:
    """Retain a one-block period when another block verifies it unchanged.

    A long-period signal may show two transits in one discovery block but only
    one (or none) in each other block.  Those other blocks cannot independently
    estimate a period, yet a covered single transit can still test the locked
    ephemeris.  This is the sole arbiter path; it performs no new search and no
    period adjustment.
    """
    output: list[dict] = []
    for raw in raw_registry:
        raw_id = str(raw["raw_id"])
        if raw_id in matched_raw_ids:
            continue
        fixed = raw["fixed_block_validation"]
        if fixed.get("passed") is not True:
            continue
        verification = dict(fixed)
        candidate = {
            "proposal_id": f"ARBITER-{raw_id}",
            "period_days": float(raw["period_days"]),
            "epoch_btjd": float(raw["epoch_btjd"]),
            "duration_hours": float(raw["duration_hours"]),
            "source_raw_ids": [raw_id],
            "discovered_blocks": [int(raw["source_block"])],
            "source_methods": [str(raw["method"])],
            "source_strength": float(raw.get("source_strength") or 0.0),
            "match_proofs": [],
            "candidate_source": "single_block_discovery_fixed_in_other_blocks",
        }
        _attach_event_repeatability(candidate, verification, branch)
        output.append({**candidate, "holdout_verification": verification})
    return output


def _validated_same_signal(first: Mapping, second: Mapping) -> bool:
    """Identify duplicate solutions from their validated predicted event train."""
    period_a = float(first["period_days"])
    period_b = float(second["period_days"])
    if abs(period_a - period_b) / min(period_a, period_b) > 0.001:
        return False

    def predicted_events(candidate: Mapping) -> list[Mapping]:
        return list(
            candidate.get("holdout_verification", {})
            .get("event_repeatability", {})
            .get("event_support", [])
        )

    events_a = predicted_events(first)
    events_b = predicted_events(second)
    if not events_a or not events_b:
        return False
    tolerance = max(
        float(first.get("duration_hours") or 0.0),
        float(second.get("duration_hours") or 0.0),
    ) / 24.0
    # Same predicate as the former nested ``any`` loop, indexed by sector and
    # sorted center.  A candidate-explosion target otherwise repeats millions
    # of comparisons merely to prove two already-locked event trains equal.
    from bisect import bisect_left, bisect_right

    cache_key = "_validated_event_centers_by_segment"
    indexed = second.get(cache_key)
    if indexed is None:
        indexed = {}
        for event in events_b:
            indexed.setdefault(int(event["segment"]), []).append(
                float(event["center_btjd"])
            )
        for centers in indexed.values():
            centers.sort()
        if isinstance(second, dict):
            second[cache_key] = indexed
    matched = 0
    for event in events_a:
        centers = indexed.get(int(event["segment"]), [])
        center = float(event["center_btjd"])
        if bisect_left(centers, center - tolerance) < bisect_right(
            centers, center + tolerance
        ):
            matched += 1
    return matched / min(len(events_a), len(events_b)) >= 0.80


def _collapse_duplicate_passes(
    candidates: list[dict],
    branch: SearchBranch | None = None,
) -> tuple[list[dict], list[dict]]:
    """Choose one maximin-S/N ephemeris per evidence family, then deduplicate."""
    if not candidates:
        return [], []

    duplicates: list[dict] = []
    ordered = sorted(
        candidates,
        key=_validated_solution_quality,
        reverse=True,
    )

    # Proposals sharing an independently discovered fold candidate are
    # alternative cross-fold connections inside one period-uncertainty family.
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    raw_owner: dict[str, int] = {}
    for index, candidate in enumerate(ordered):
        for raw_id in candidate.get("source_raw_ids", []):
            raw_id = str(raw_id)
            if raw_id in raw_owner:
                union(index, raw_owner[raw_id])
            else:
                raw_owner[raw_id] = index

    families: dict[int, list[dict]] = {}
    for index, candidate in enumerate(ordered):
        families.setdefault(find(index), []).append(candidate)

    family_representatives: list[dict] = []
    for family in families.values():
        family.sort(key=_validated_solution_quality, reverse=True)
        representative = family[0]
        representative["equivalent_proposal_ids"] = [
            row["proposal_id"] for row in family
        ]
        representative["balanced_family_proposal_ids"] = list(
            representative["equivalent_proposal_ids"]
        )
        family_periods = [float(row["period_days"]) for row in family]
        representative["balanced_family_period_bounds_days"] = [
            min(family_periods),
            max(family_periods),
        ]
        representative["balanced_family_duration_hours"] = float(
            np.median([float(row["duration_hours"]) for row in family])
        )
        representative["source_raw_ids"] = sorted(
            {
                str(raw_id)
                for row in family
                for raw_id in row.get("source_raw_ids", [])
            }
        )
        representative["discovered_blocks"] = sorted(
            {
                int(block)
                for row in family
                for block in row.get("discovered_blocks", [])
            }
        )
        family_representatives.append(representative)
        duplicates.extend(
            {
                **row,
                "registry_state": "validated_family_alternative",
                "represented_by_proposal_id": representative["proposal_id"],
            }
            for row in family[1:]
        )

    # The representative is chosen exclusively from locked-ephemeris fold
    # measurements above.  Attach the unchanged per-event diagnostic only now;
    # it is still present for every candidate that can enter cross-family
    # event-train deduplication, balancing, alias checks, and ranking.
    if branch is not None:
        for representative in family_representatives:
            verification = representative["holdout_verification"]
            if "event_repeatability" not in verification:
                _attach_event_repeatability(representative, verification, branch)

    display: list[dict] = []
    # Period-sorted index of the already accepted display representatives.
    # The original algorithm considered them in quality/insertion order; the
    # index narrows the exact predicate to the only periods that can pass and
    # then restores that order before selecting ``next(...)``.
    from bisect import bisect_left, bisect_right, insort

    display_period_index: list[tuple[float, int, dict]] = []
    for candidate in sorted(
        family_representatives, key=_validated_solution_quality, reverse=True
    ):
        period = float(candidate["period_days"])
        lower = period / 1.001
        upper = period * 1.001
        first = bisect_left(display_period_index, (lower, -1, {}))
        last = bisect_right(
            display_period_index,
            (upper, len(display) + 1, {}),
        )
        possible = sorted(
            display_period_index[first:last], key=lambda row: row[1]
        )
        representative = next(
            (
                row
                for _period, _display_index, row in possible
                if _validated_same_signal(row, candidate)
            ),
            None,
        )
        if representative is None:
            display_index = len(display)
            display.append(candidate)
            insort(
                display_period_index,
                (period, display_index, candidate),
            )
            continue
        representative["equivalent_proposal_ids"].extend(
            candidate["equivalent_proposal_ids"]
        )
        representative["balanced_family_proposal_ids"].extend(
            candidate.get("balanced_family_proposal_ids", [])
        )
        first_bounds = representative["balanced_family_period_bounds_days"]
        second_bounds = candidate["balanced_family_period_bounds_days"]
        representative["balanced_family_period_bounds_days"] = [
            min(float(first_bounds[0]), float(second_bounds[0])),
            max(float(first_bounds[1]), float(second_bounds[1])),
        ]
        representative["source_raw_ids"] = sorted(
            set(representative["source_raw_ids"]) | set(candidate["source_raw_ids"])
        )
        representative["discovered_blocks"] = sorted(
            set(representative["discovered_blocks"])
            | set(candidate["discovered_blocks"])
        )
        duplicates.append(
            {
                **candidate,
                "registry_state": "validated_equivalent",
                "represented_by_proposal_id": representative["proposal_id"],
            }
        )
    return display, duplicates


def _balance_family_ephemeris(
    candidate: dict, blocks: list[list[int]], branch: SearchBranch
) -> None:
    """Fit one global ephemeris by maximising the weakest fold S/N."""
    bounds = candidate.get("balanced_family_period_bounds_days")
    if not bounds or len(bounds) != 2:
        return
    lower, upper = map(float, bounds)
    if not (np.isfinite(lower) and np.isfinite(upper) and upper > lower > 0):
        return

    duration_hours = float(
        candidate.get("balanced_family_duration_hours")
        or candidate["duration_hours"]
    )
    duration_days = duration_hours / 24.0
    baseline_days = float(np.nanmax(branch.time) - np.nanmin(branch.time))
    center_period = 0.5 * (lower + upper)
    # Resolve one tenth of a transit duration after extrapolation across the
    # complete baseline.  The data determine both the range and the spacing.
    period_resolution = max(
        np.finfo(float).eps,
        duration_days * center_period / max(baseline_days, center_period) / 10.0,
    )
    grid_size = int(
        np.clip(math.ceil((upper - lower) / period_resolution) + 1, 101, 2001)
    )
    periods = np.linspace(lower, upper, grid_size)
    bls = BoxLeastSquares(branch.time, branch.flux)

    def evaluate_grid(period_grid: np.ndarray) -> tuple[dict, tuple]:
        power = bls.power(
            period_grid,
            duration_days,
            objective="snr",
            oversample=10,
        )
        power_values = np.asarray(power.power, dtype=float)
        epochs = np.asarray(power.transit_time, dtype=float)
        if period_grid.size <= 401:
            indices = set(range(period_grid.size))
        else:
            indices = set(
                np.linspace(0, period_grid.size - 1, 401, dtype=int).tolist()
            )
        indices.update(
            np.argsort(np.nan_to_num(power_values, nan=-np.inf))[-100:].tolist()
        )
        best_row: dict | None = None
        best_score: tuple | None = None
        for index in sorted(indices):
            trial = {
                **candidate,
                "period_days": float(period_grid[index]),
                "epoch_btjd": float(epochs[index]),
                "duration_hours": duration_hours,
            }
            measurements = _measure_all_blocks(trial, blocks, branch)
            covered = [
                row
                for row in measurements
                if row["outcome"] != "untestable"
            ]
            signed_snrs = [
                (
                    float(row.get("snr") or 0.0)
                    if float(row.get("depth_ppm") or 0.0) > 0
                    else -abs(float(row.get("snr") or 0.0))
                )
                for row in covered
            ]
            score = (
                len([value for value in signed_snrs if value > 0]),
                min(signed_snrs, default=-np.inf),
                sum(signed_snrs),
                float(power_values[index]),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_row = {
                    "period_days": trial["period_days"],
                    "epoch_btjd": trial["epoch_btjd"],
                    "measurements": measurements,
                    "score": score,
                }
        assert best_row is not None and best_score is not None
        return best_row, best_score

    coarse, _ = evaluate_grid(periods)
    coarse_step = float(periods[1] - periods[0])
    fine_periods = np.linspace(
        max(lower, float(coarse["period_days"]) - coarse_step),
        min(upper, float(coarse["period_days"]) + coarse_step),
        101,
    )
    best, score = evaluate_grid(fine_periods)
    original_period = float(candidate["period_days"])
    original_epoch = float(candidate["epoch_btjd"])
    candidate["period_days"] = float(best["period_days"])
    candidate["epoch_btjd"] = float(best["epoch_btjd"])
    candidate["duration_hours"] = duration_hours
    candidate["holdout_verification"] = _validation_decision(best["measurements"])
    _attach_event_repeatability(
        candidate, candidate["holdout_verification"], branch
    )
    candidate["ephemeris_balance"] = {
        "method": "global_bls_then_maximise_weakest_fold_snr",
        "period_bounds_days": [lower, upper],
        "coarse_grid_size": grid_size,
        "fine_grid_size": 101,
        "original_period_days": original_period,
        "original_epoch_btjd": original_epoch,
        "selected_period_days": float(candidate["period_days"]),
        "selected_epoch_btjd": float(candidate["epoch_btjd"]),
        "weakest_fold_snr": float(score[1]),
        "used_catalog_period": False,
    }


def _validated_solution_quality(candidate: dict) -> tuple:
    """Rank equivalent, already-passing ephemerides by cross-block stability.

    This is deliberately based on sealed validation rather than catalogue values
    or a hand-picked period.  Independent discovery is preferred first; among
    equally independent solutions we maximise the weakest covered block.  The
    later per-event report is deliberately not fed back into period selection.
    """
    verification = candidate.get("holdout_verification", {})
    measurements = [
        row
        for row in verification.get("block_measurements", [])
        if row.get("outcome") in {"strong", "weak"}
    ]
    snrs = [float(row.get("snr") or 0.0) for row in measurements]
    return (
        len(candidate.get("discovered_blocks", [])),
        min(snrs, default=0.0),
        len(verification.get("strong_blocks", [])),
        float(verification.get("combined_positive_snr") or 0.0),
        float(candidate.get("source_strength") or 0.0),
    )


def _residue_support(candidate: Mapping, order: int) -> list[dict]:
    events = (
        candidate.get("holdout_verification", {})
        .get("event_repeatability", {})
        .get("event_support", [])
    )
    # One pass is mathematically identical to rebuilding and filtering the
    # complete event list once per residue.  The former O(order * events)
    # implementation dominated high-order harmonic families.
    testable_counts = np.zeros(order, dtype=np.int64)
    supported_counts = np.zeros(order, dtype=np.int64)
    for row in events:
        residue = int(row["cycle"]) % order
        testable_counts[residue] += 1
        supported_counts[residue] += int(bool(row.get("supported")))
    output: list[dict] = []
    for residue in range(order):
        testable = int(testable_counts[residue])
        supported = int(supported_counts[residue])
        output.append(
            {
                "residue": residue,
                "testable": testable,
                "supported": supported,
                "fraction": supported / testable if testable else None,
            }
        )
    return output


def _integer_alias_decision(first: Mapping, second: Mapping) -> dict | None:
    period_a = float(first["period_days"])
    period_b = float(second["period_days"])
    shorter, longer = (first, second) if period_a < period_b else (second, first)
    ratio = float(longer["period_days"]) / float(shorter["period_days"])
    order = int(round(ratio))
    if order < 2 or order > 200 or abs(ratio - order) / order > 0.001:
        return None
    phase_error = abs(
        (
            float(longer["epoch_btjd"])
            - float(shorter["epoch_btjd"])
            + 0.5 * float(shorter["period_days"])
        )
        % float(shorter["period_days"])
        - 0.5 * float(shorter["period_days"])
    )
    tolerance = max(
        float(shorter["duration_hours"]), float(longer["duration_hours"])
    ) / 24.0
    if phase_error > tolerance:
        return None
    residues = _residue_support(shorter, order)
    testable = [row for row in residues if row["testable"] > 0]
    if len(testable) < 2:
        return None
    fractions = [float(row["fraction"] or 0.0) for row in testable]
    supported_counts = [int(row["supported"]) for row in testable]
    total_supported = sum(supported_counts)
    well_populated_residues = sum(value >= 2 for value in supported_counts)
    all_residues_recur = (
        len(testable) == order
        and total_supported >= max(6, 2 * order)
        and all(value >= 1 for value in supported_counts)
        and well_populated_residues >= math.ceil(order / 2)
        and min(supported_counts)
        >= max(1, math.ceil(0.15 * max(supported_counts)))
    )
    ordered_counts = sorted(supported_counts, reverse=True)
    dominant_supported = ordered_counts[0] if ordered_counts else 0
    second_supported = ordered_counts[1] if len(ordered_counts) > 1 else 0
    one_residue_dominates = (
        dominant_supported >= 3
        and total_supported > 0
        and dominant_supported / total_supported >= 0.75
        and second_supported <= max(1, math.floor(0.20 * dominant_supported))
        and max(fractions, default=0.0) >= 0.15
    )
    # A shorter period is established only when *every* residue class is
    # actually observed and repeats.  Treating two observed classes out of a
    # possible three as sufficient wrongly collapsed a real 49.824 d signal
    # into an unrelated 16.608 d candidate when the third residue had no data.
    if (
        len(testable) == order
        and all(value >= 0.5 for value in fractions)
    ) or all_residues_recur:
        winner, loser, reason = shorter, longer, "all_short_period_residues_repeat"
    elif (
        max(fractions) >= 0.6
        and sorted(fractions, reverse=True)[1] <= 0.3
    ) or one_residue_dominates:
        winner, loser, reason = longer, shorter, "only_one_short_period_residue_repeats"
    else:
        return None
    return {
        "winner_proposal_id": winner["proposal_id"],
        "loser_proposal_id": loser["proposal_id"],
        "order": order,
        "ratio": ratio,
        "phase_error_days": phase_error,
        "residue_support": residues,
        "reason": reason,
    }


def _compress_proven_aliases(
    candidates: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Move only evidence-proven integer aliases out of the display list.

    All pair decisions are frozen before any candidate is removed.  This makes
    the result invariant to candidate input order and prevents an early local
    deletion from changing which later harmonic relations are considered.
    """

    by_id = {str(row["proposal_id"]): row for row in candidates}
    decisions: list[dict] = []
    for left_index, first in enumerate(candidates):
        for second in candidates[left_index + 1 :]:
            decision = _integer_alias_decision(first, second)
            if decision is not None:
                decisions.append(decision)

    winners_by_loser: dict[str, list[str]] = {}
    decision_by_edge: dict[tuple[str, str], dict] = {}
    for decision in decisions:
        winner_id = str(decision["winner_proposal_id"])
        loser_id = str(decision["loser_proposal_id"])
        winners_by_loser.setdefault(loser_id, []).append(winner_id)
        decision_by_edge[(winner_id, loser_id)] = decision

    def terminal_winners(start_id: str) -> tuple[set[str], set[str]]:
        frontier = [start_id]
        visited: set[str] = set()
        terminals: set[str] = set()
        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            parents = winners_by_loser.get(current, [])
            if not parents:
                terminals.add(current)
            else:
                frontier.extend(parents)
        return terminals, visited

    representative_by_loser: dict[str, str] = {}
    cycle_survivors: set[str] = set()
    for loser_id in winners_by_loser:
        terminals, visited = terminal_winners(loser_id)
        if terminals:
            representative_by_loser[loser_id] = max(
                terminals, key=lambda value: _validated_solution_quality(by_id[value])
            )
            continue
        # Defensive cycle handling: retain exactly one globally strongest
        # solution rather than letting a contradictory edge set erase all of
        # them.  Such a cycle remains visible in the decision audit.
        survivor = max(
            visited, key=lambda value: _validated_solution_quality(by_id[value])
        )
        cycle_survivors.add(survivor)
        if loser_id != survivor:
            representative_by_loser[loser_id] = survivor

    removed_ids = set(representative_by_loser) - cycle_survivors
    active = {
        proposal_id: row
        for proposal_id, row in by_id.items()
        if proposal_id not in removed_ids
    }
    aliases: list[dict] = []
    for loser_id in sorted(
        removed_ids,
        key=lambda value: _validated_solution_quality(by_id[value]),
        reverse=True,
    ):
        winner_id = representative_by_loser[loser_id]
        winner = active[winner_id]
        loser = by_id[loser_id]
        direct_winners = winners_by_loser[loser_id]
        direct_winner_id = max(
            direct_winners,
            key=lambda value: _validated_solution_quality(by_id[value]),
        )
        decision = decision_by_edge[(direct_winner_id, loser_id)]
        winner.setdefault("alternate_periods", []).append(
            {
                "period_days": float(loser["period_days"]),
                "proposal_id": loser_id,
                "holdout_verification": loser["holdout_verification"],
                "alias_decision": decision,
            }
        )
        aliases.append(
            {
                **loser,
                "registry_state": "validated_integer_alias",
                "represented_by_proposal_id": winner_id,
                "alias_decision": decision,
            }
        )
    return list(active.values()), aliases, decisions


def _fold_summary(
    branch: SearchBranch,
    period: float,
    epoch: float,
    allowed_segments: Iterable[int],
) -> dict:
    local = _subset_branch(branch, allowed_segments)
    phase = ((local.time - epoch + 0.5 * period) % period) / period - 0.5
    ppm = (local.flux - 1.0) * 1_000_000.0
    finite = np.isfinite(phase) & np.isfinite(ppm)
    phase, ppm = phase[finite], ppm[finite]
    if not phase.size:
        return {"bins": [], "points": [], "point_count": 0}
    edges = np.linspace(-0.5, 0.5, 161)
    bins = []
    for left, right in zip(edges[:-1], edges[1:]):
        values = ppm[(phase >= left) & (phase < right)]
        if values.size:
            bins.append(
                [
                    round(float((left + right) / 2), 5),
                    round(float(np.nanmedian(values)), 2),
                    round(float(np.nanpercentile(values, 25)), 2),
                    round(float(np.nanpercentile(values, 75)), 2),
                    int(values.size),
                ]
            )
    order = np.argsort(phase)
    sample = order[:: max(1, len(order) // 560)][:560]
    points = [
        [round(float(phase[index]), 5), round(float(ppm[index]), 2)]
        for index in sample
    ]
    return {"bins": bins, "points": points, "point_count": int(phase.size)}


def _candidate_events(candidate: Mapping, segments: list[dict]) -> list[dict]:
    period = float(candidate["period_days"])
    epoch = float(candidate["epoch_btjd"])
    output: list[dict] = []
    for segment in segments:
        first = math.ceil((float(segment["start"]) - epoch) / period)
        last = math.floor((float(segment["end"]) - epoch) / period)
        for cycle in range(first, last + 1):
            output.append(
                {
                    "segment": int(segment["segment"]),
                    "cycle": cycle,
                    "time": _json_number(epoch + cycle * period, 5),
                    "kind": segment["kind"],
                }
            )
    return output


def _query_toi_catalog(tic_id: int | None) -> list[dict]:
    if tic_id is None:
        return []
    query = (
        "select toi,pl_orbper,pl_tranmid,tfopwg_disp from toi "
        f"where tid={int(tic_id)} and pl_orbper is not null order by pl_orbper"
    )
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?" + urllib.parse.urlencode(
        {"query": query, "format": "json"}
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            rows = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    output = []
    for row in rows:
        period = _json_number(row.get("pl_orbper"), 7)
        if period is None:
            continue
        disposition = str(row.get("tfopwg_disp") or "")
        output.append(
            {
                "name": f"TOI-{row.get('toi')}",
                "period": period,
                "epochBtjd": (
                    _json_number(float(row["pl_tranmid"]) - 2457000.0, 6)
                    if row.get("pl_tranmid")
                    else None
                ),
                "disposition": disposition,
                "status": STATUS_ZH.get(disposition, disposition or "Known catalog signal"),
            }
        )
    return output


def _attach_catalog_answers(candidates: list[dict], catalog: list[dict]) -> list[dict]:
    """Attach truth only after the complete result is locked."""
    for candidate in candidates:
        hypotheses = [
            (float(candidate["period_days"]), False),
            *[
                (float(row["period_days"]), True)
                for row in candidate.get("alternate_periods", [])
            ],
        ]
        viable = [
            (abs(period - float(answer["period"])) / float(answer["period"]), answer, period, alternate)
            for answer in catalog
            for period, alternate in hypotheses
            if abs(period - float(answer["period"])) / float(answer["period"]) <= 0.01
        ]
        if viable:
            _error, answer, period, alternate = min(viable, key=lambda row: row[0])
            candidate["known_match"] = {
                **answer,
                "periodErrorMinutes": _json_number(
                    abs(period - float(answer["period"])) * 1440.0, 2
                ),
                "matchedPeriod": _json_number(period, 7),
                "matchedAsAlternate": alternate,
            }
        else:
            candidate["known_match"] = None
    for answer in catalog:
        matches = []
        public_matches = []
        for candidate in candidates:
            for period, alternate in [
                (float(candidate["period_days"]), False),
                *[
                    (float(row["period_days"]), True)
                    for row in candidate.get("alternate_periods", [])
                ],
            ]:
                error = abs(period - float(answer["period"])) / float(answer["period"])
                if error <= 0.01:
                    row = (int(candidate["rank"]), period, alternate)
                    matches.append(row)
                    decision = candidate.get("candidate_decision") or {}
                    if decision.get("public_candidate", True):
                        public_matches.append(row)
        generated = min(matches) if matches else None
        match = min(public_matches) if public_matches else None
        answer["matchedRank"] = match[0] if match else None
        answer["generatedRank"] = generated[0] if generated else None
        answer["algorithmPeriod"] = (
            _json_number((match or generated)[1], 7) if (match or generated) else None
        )
        answer["matchedAsAlternate"] = bool((match or generated) and (match or generated)[2])
    return catalog


def _public_candidate(
    row: dict,
    *,
    rank: int,
    blocks: list[list[int]],
    branch: SearchBranch,
    timeline: list[dict],
) -> dict:
    verification = row["holdout_verification"]
    positive_blocks = [*verification["strong_blocks"], *verification["weak_blocks"]]
    measurements = verification["block_measurements"]
    positive_snrs = [
        float(item["snr"])
        for item in measurements
        if item["outcome"] in {"strong", "weak"} and item.get("snr") is not None
    ]
    positive_depths = [
        float(item["depth_ppm"])
        for item in measurements
        if item["outcome"] in {"strong", "weak"}
        and item.get("depth_ppm") is not None
    ]
    discovery_segments = sorted(
        {segment for block in row["discovered_blocks"] for segment in blocks[block - 1]}
    )
    validation_segments = sorted(
        {segment for block, group in enumerate(blocks, start=1) if block not in row["discovered_blocks"] for segment in group}
    )
    if not validation_segments:
        validation_segments = sorted({segment for group in blocks for segment in group})
    score = (
        10.0 * len(row["discovered_blocks"])
        + 4.0 * len(verification["strong_blocks"])
        + min(10.0, min(positive_snrs) if positive_snrs else 0.0)
    )
    candidate = {
        **row,
        "candidate_id": f"CLEAN-{rank:03d}",
        "rank": rank,
        "candidate_score": score,
        "candidate_source": row.get(
            "candidate_source", "independent_temporal_intersection"
        ),
        "supporting_branches": [
            f"temporal_block_{value}" for value in row["discovered_blocks"]
        ],
        "holdout_supporting_branches": [
            f"temporal_block_{value}" for value in positive_blocks
        ],
        "features": {
            "median_search_snr": float(np.median(positive_snrs)) if positive_snrs else None,
            "primary_depth_ppm": float(np.median(positive_depths)) if positive_depths else None,
        },
        "reasons_zh": [
            (
                f"{len(row['discovered_blocks'])} disjoint time blocks independently recovered the same period and phase"
                if len(row["discovered_blocks"]) >= 2
                else "One time block independently determined the period; another reproduced the dimming at the locked epoch"
            ),
            "The locked period was retested in all three time blocks without a sufficiently covered counterexample",
        ],
        "cross_validation": {
            "passed": True,
            "matched_fold_numbers": list(row["discovered_blocks"]),
            "folds_measured": 3,
        },
        "fold": {
            "discovery": _fold_summary(
                branch,
                float(row["period_days"]),
                float(row["epoch_btjd"]),
                discovery_segments,
            ),
            "holdout": _fold_summary(
                branch,
                float(row["period_days"]),
                float(row["epoch_btjd"]),
                validation_segments,
            ),
        },
        "events": _candidate_events(row, timeline),
        "alias_group_size": 1 + len(row.get("alternate_periods", [])),
        "physical_vetting": {},
        "pixel_status": "not_run",
        "hard_rejected": False,
    }
    return candidate


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_live_target(
    target: str,
    mission: str,
    *,
    run_dir: Path,
    cache_root: Path = DEFAULT_ROOT / "cache",
    max_segments: int | None = None,
    period_max: float = 100.0,
    deep: bool = False,
    progress: Progress | None = None,
    dual_seal: bool = True,
    run_pixel_vetting: bool = True,
    significance_phase_trials: int = 4096,
    download_workers: int = DEFAULT_DOWNLOAD_WORKERS,
    frozen_lightcurve_products: Iterable[Mapping] | None = None,
    query_catalog: bool = True,
    lightcurve_source: str = "spoc",
) -> dict:
    """Run discovery, lock every candidate, then annotate physical evidence.

    Post-discovery vetting is strictly non-destructive: it cannot remove or
    reorder candidates and cannot change a locked period, epoch, or duration.
    """
    del dual_seal  # Kept only for command-line compatibility; three blocks are mandatory.
    started = time.monotonic()
    mission = mission if mission in {"TESS", "Kepler", "K2"} else "TESS"
    lightcurve_source = str(lightcurve_source).lower()
    if lightcurve_source not in {"spoc", "t16"}:
        raise ValueError("lightcurve_source must be 'spoc' or 't16'")
    if lightcurve_source == "t16" and mission != "TESS":
        raise ValueError("T16 light curves are available only for TESS")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if frozen_lightcurve_products is None:
        _progress(
            progress,
            "discovering",
            3,
            f"Locating {mission} light curves for {target}",
            title="Building the observation manifest",
            command=f"archive discover --target {target!r} --mission {mission}",
        )
        inventory = _discover(
            target, mission, lightcurve_source=lightcurve_source
        )
        selected = _select_lightcurves(
            inventory["products"], max_segments=max_segments, deep=deep
        )
        product_policy = f"live_archive_discovery:{lightcurve_source}"
    else:
        selected = [dict(product) for product in frozen_lightcurve_products]
        if max_segments is not None and max_segments > 0:
            selected = selected[:max_segments]
        inventory = {
            "target": target,
            "missions_requested": [mission],
            "products": selected,
            "counts": {
                "lightcurves": len(selected),
                "target_pixels": 0,
                "total": len(selected),
            },
            "diagnostic": {
                "stage": "lightcurve_discovery",
                "status": "frozen_manifest_supplied",
            },
        }
        product_policy = "frozen_label_free_manifest"
        _progress(
            progress,
            "discovering",
            3,
            f"Using {len(selected)} light-curve products from the sealed manifest",
            title="Input products locked",
            command="archive use-frozen-manifest --no-live-discovery",
            metrics={"products": len(selected)},
        )
    if len(selected) < 3:
        raise RuntimeError("At least three observing segments are required to construct three disjoint time blocks")
    download_workers = max(1, int(download_workers))
    paths = _download_products(
        selected,
        Path(cache_root),
        progress,
        max_workers=download_workers,
    )
    _progress(
        progress,
        "downloading",
        20,
        f"All {len(paths)} light-curve segments are ready; starting the period-search channels",
        title="Light-curve acquisition complete",
        command="pipeline advance --from lightcurve-fetch --to period-search",
        metrics={
            "lightcurve_files_ready": len(paths),
            "lightcurve_download_workers": min(download_workers, len(selected)),
            "next_stage": "period_search",
        },
    )
    metadata = [_segment_metadata(path) for path in paths]
    paired = sorted(zip(metadata, paths), key=lambda row: row[0]["segment"])
    metadata = [row[0] for row in paired]
    paths = [row[1] for row in paired]
    segments = sorted({int(row["segment"]) for row in metadata})
    blocks = select_temporal_blocks(segments)
    if len(blocks) != 3 or any(not block for block in blocks):
        raise RuntimeError("Unable to construct three disjoint time blocks")
    _progress(
        progress,
        "preparing",
        22,
        "Three disjoint time blocks are locked; their assignment will remain fixed",
        title="A/B/C time blocks locked",
        command="temporal split --independent --three-blocks",
        metrics={"blocks": blocks},
    )

    # Pixel discovery and downloads are independent of candidate selection.
    # Start them now so network I/O overlaps the three CPU-heavy fold searches;
    # candidate periods remain unavailable to and unaffected by this worker.
    pixel_prefetch_pool: ThreadPoolExecutor | None = None
    pixel_prefetch_future = None
    if run_pixel_vetting:
        _progress(
            progress,
            "pixel_prefetch",
            23,
            "Light curves are ready: period searches are running while pixel files download concurrently",
            title="Dual-channel pipeline started",
            command=(
                f"pipeline overlap --foreground period-search "
                f"--background target-pixel-fetch --parallel {download_workers} "
                f"--segments {','.join(map(str, segments))}"
            ),
            metrics={
                "segments": len(segments),
                "pixel_download_workers": min(download_workers, len(segments)),
                "lightcurve_files_ready": len(paths),
                "foreground_lane": "period_search",
                "background_lane": "target_pixel_download",
            },
        )
        pixel_prefetch_pool, pixel_prefetch_future = _submit_background_pixel_prefetch(
            _prefetch_target_pixels,
            target,
            mission,
            segments,
            deep=deep,
            cache_root=Path(cache_root),
            progress=progress,
            download_workers=download_workers,
        )
    if lightcurve_source == "t16":
        branch, load_diagnostics = _load_t16_branch(paths)
    else:
        branch, load_diagnostics = _load_pdcsap_branch(paths)

    ledger: list[dict] = []
    search_diagnostics: list[dict] = []

    def announce_block_start(block_index: int, block: list[int]) -> None:
        start_percent = 25 + block_index * 18
        _progress(
            progress,
            "searching",
            start_percent,
            f"Running an independent full search of time block {block_index + 1}/3",
            title=f"Independent search: block {block_index + 1}",
            command=f"search block-{block_index + 1} --uncapped --segments {','.join(map(str, block))}",
            metrics={"segments": block, "candidate_cap": None},
        )
    def announce_block_finish(
        block_index: int, rows: list[dict]
    ) -> None:
        start_percent = 25 + block_index * 18
        _progress(
            progress,
            "searching",
            start_percent + 16,
            f"Search of block {block_index + 1} is complete; all {len(rows)} raw hypotheses were recorded",
            title=f"Block {block_index + 1} hypotheses recorded",
            metrics={"raw_hypotheses": len(rows)},
        )

    # The blocks are independent by construction.  Running them concurrently
    # changes no search or validation decision and avoids making wall time the
    # sum of three large BLS matrices.
    completed_blocks: list[tuple[int, list[dict], dict]] = []
    for block_index, block in enumerate(blocks):
        announce_block_start(block_index, block)
    # Event-first contains substantial Python work, so threads would merely
    # take turns behind the GIL.  Spawned workers give each independent fold a
    # real CPU without sharing mutable search state.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=3, mp_context=context) as pool:
        futures = {
            pool.submit(
                _search_block,
                branch,
                block_index=block_index,
                segments=block,
                period_max=float(period_max),
            ): block_index
            for block_index, block in enumerate(blocks)
        }
        for future in as_completed(futures):
            block_index = futures[future]
            rows, diagnostics = future.result()
            completed_blocks.append((block_index, rows, diagnostics))
            announce_block_finish(block_index, rows)
    for _block_index, rows, diagnostics in sorted(completed_blocks):
        ledger.extend(rows)
        search_diagnostics.append(diagnostics)

    proposals, match_edges = _build_cross_block_proposals(ledger)
    _progress(
        progress,
        "matching",
        81,

        f"Cross-block matching complete: all {len(match_edges)} period-and-phase matches advanced to validation",
        title="Comparing the three-block intersection",
        command="match --identity period+phase+event-train --minimum-blocks 2",
        metrics={"match_edges": len(match_edges), "proposals_to_validate": len(proposals)},
    )

    # High-coverage targets can produce well over one hundred thousand exact
    # cross-fold matches.  Validation is scientifically independent per row,
    # so process the unchanged ledger in bounded batches and report real
    # completion counts instead of leaving the public UI motionless at 81%.
    validation_batch_size = 2_048
    validation_total = len(ledger) + len(proposals)
    validation_completed = 0

    def announce_validation_progress(kind: str) -> None:
        fraction = (
            validation_completed / validation_total if validation_total else 1.0
        )
        _progress(
            progress,
            "validating",
            min(91, 81 + int(10 * fraction)),
            (
                f"Sealed validation {validation_completed}/{validation_total}: "
                f"completed {kind}"
            ),
            title="Validating cross-block signals",
            command=(
                "validate --locked-ephemeris "
                f"--completed {validation_completed} --total {validation_total}"
            ),
            metrics={
                "hypotheses_completed": validation_completed,
                "hypotheses_total": validation_total,
                "raw_hypotheses": len(ledger),
                "matched_proposals": len(proposals),
                "batch_size": validation_batch_size,
            },
        )

    raw_registry: list[dict] = []
    for start in range(0, len(ledger), validation_batch_size):
        stop = min(start + validation_batch_size, len(ledger))
        raw_registry.extend(_validate_raw_ledger(ledger[start:stop], blocks, branch))
        validation_completed += stop - start
        announce_validation_progress("raw hypothesis")

    proposal_registry: list[dict] = []
    for start in range(0, len(proposals), validation_batch_size):
        stop = min(start + validation_batch_size, len(proposals))
        proposal_registry.extend(
            _validate_proposals(proposals[start:stop], blocks, branch)
        )
        validation_completed += stop - start
        announce_validation_progress("matched proposal")
    matched_raw_ids = {
        raw_id
        for proposal in proposal_registry
        if proposal["holdout_verification"].get("passed") is True
        for raw_id in proposal["source_raw_ids"]
    }
    arbiter_registry = _build_single_block_arbitrations(
        raw_registry, matched_raw_ids, branch
    )
    proposal_registry.extend(arbiter_registry)
    preliminary_passed = [
        row
        for row in proposal_registry
        if row["holdout_verification"]["passed"] is True
    ]
    balanced_families, equivalents = _collapse_duplicate_passes(
        preliminary_passed, branch
    )
    family_total = len(balanced_families)
    family_progress_interval = max(1, family_total // 50)
    for family_index, candidate in enumerate(balanced_families, start=1):
        _balance_family_ephemeris(candidate, blocks, branch)
        _apply_fold_event_direction_test(candidate, blocks)
        if (
            family_index == family_total
            or family_index % family_progress_interval == 0
        ):
            _progress(
                progress,
                "validating",
                91,
                f"Candidate-family ephemeris balancing {family_index}/{family_total}",
                title="Balancing candidate-family ephemerides",
                command=(
                    "balance-families --completed "
                    f"{family_index} --total {family_total}"
                ),
                metrics={
                    "families_completed": family_index,
                    "families_total": family_total,
                },
            )
    stable_families = [
        row
        for row in balanced_families
        if row["holdout_verification"]["passed"] is True
    ]
    canonical, aliases, alias_decisions = _compress_proven_aliases(stable_families)
    canonical.sort(
        key=lambda row: (
            len(row["discovered_blocks"]),
            len(row["holdout_verification"]["strong_blocks"]),
            min(
                [
                    float(item.get("snr") or 0.0)
                    for item in row["holdout_verification"]["block_measurements"]
                    if item["outcome"] in {"strong", "weak"}
                ]
                or [0.0]
            ),
            float(row.get("source_strength") or 0.0),
        ),
        reverse=True,
    )
    _progress(
        progress,
        "validating",
        92,

        f"Unified-ephemeris cross-block validation complete: {len(stable_families)} stable dimming families and {len(canonical)} unique primary periods",
        title="Sealed-period validation complete",
        command="validate --global-ephemeris-balance --no-per-fold-retuning",
        metrics={
            "raw_validated": len(raw_registry),
            "matched_validated": len(proposal_registry),
            "single_block_arbitrations": len(arbiter_registry),
            "folded_validation_passed": len(preliminary_passed),
            "balanced_families": len(balanced_families),
            "stable_event_families": len(stable_families),
            "proven_aliases": len(aliases),
        },
    )

    block_by_segment = {
        segment: block_index
        for block_index, block in enumerate(blocks, start=1)
        for segment in block
    }
    timeline = [
        {
            **row,
            "kind": "search" if block_by_segment[int(row["segment"])] == 1 else "holdout",
            "temporal_block": block_by_segment[int(row["segment"])],
        }
        for row in metadata
    ]
    candidates = [
        _public_candidate(
            row,
            rank=index,
            blocks=blocks,
            branch=branch,
            timeline=timeline,
        )
        for index, row in enumerate(canonical, start=1)
    ]

    # This is a downstream, locked-ephemeris measurement.  It deliberately
    # cannot feed back into candidate discovery, ranking, or period selection.
    search_trial_factor = len(blocks) * 18_000 * 10 + len(ledger)
    candidates, detection_significance = annotate_detection_significance(
        candidates,
        branch,
        blocks,
        phase_trials=int(significance_phase_trials),
        search_trial_factor=search_trial_factor,
        progress=progress,
    )

    # Candidate ephemerides and their ordering are final at this point.  Save
    # them before joining the potentially slow TPF prefetch so a completed
    # discovery result is never hidden behind stage-two network latency.
    discovery_checkpoint = {
        "status": "discovery_locked_before_physical_vetting",
        "target": target,
        "object": metadata[0].get("object") or target,
        "tic_id": metadata[0].get("tic_id"),
        "mission": mission,
        "lightcurve_source": lightcurve_source,
        "photometry_branch": {
            "key": branch.key,
            "label": branch.label,
            "processing": branch.processing,
            "photometry_family": branch.photometry_family,
            "metadata": dict(branch.metadata),
        },
        "engine_version": "clean-temporal-intersection-v4-red-noise-significance",
        "temporal_blocks": [
            {"block": index + 1, "segments": block}
            for index, block in enumerate(blocks)
        ],
        "candidate_pool_total": len(candidates),
        "candidates": candidates,
        "detection_significance": detection_significance,
        "truth_catalog_used": False,
        "physical_vetting_complete": False,
        "pixel_vetting_complete": False,
    }
    _write_json(run_dir / "discovery_checkpoint.json", discovery_checkpoint)
    _progress(
        progress,
        "discovery_checkpoint",
        92,
        f"Blind-search results were saved first: {len(candidates)} primary periods; later pixel annotations cannot change the ranking",
        title="Locked candidates saved",
        command="checkpoint discovery --before-pixel-join",
        metrics={"candidates": len(candidates)},
    )

    pixel_inventory = {
        "target": target,
        "missions_requested": [mission],
        "products": [],
        "counts": {"lightcurves": 0, "target_pixels": 0, "total": 0},
        "diagnostic": {
            "stage": "target_pixel_discovery",
            "status": "disabled",
        },
    }
    selected_target_pixels: list[dict] = []
    target_pixel_paths: list[Path] = []
    pixel_download_diagnostics: list[dict] = []
    pixel_prefetch_report = {
        "enabled": bool(run_pixel_vetting),
        "started_during_stage_one": bool(pixel_prefetch_future is not None),
        "lightcurves_ready_before_search": True,
        "started_before_period_search": bool(pixel_prefetch_future is not None),
        "overlapped_with_period_search": bool(pixel_prefetch_future is not None),
        "download_strategy": "background_high_concurrency",
        "parallel_workers": download_workers,
        "ready_before_pixel_vetting": False,
        "wait_seconds_at_pixel_stage": 0.0,
    }
    if pixel_prefetch_future is not None:
        pixel_prefetch_report["ready_before_pixel_vetting"] = bool(
            pixel_prefetch_future.done()
        )
        pixel_wait_started = time.monotonic()
        try:
            (
                pixel_inventory,
                selected_target_pixels,
                target_pixel_paths,
                pixel_download_diagnostics,
            ) = pixel_prefetch_future.result()
        except Exception as exc:
            pixel_download_diagnostics.append(
                {
                    "stage": "target_pixel_prefetch",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            pixel_inventory["diagnostic"] = pixel_download_diagnostics[-1]
        finally:
            pixel_prefetch_report["wait_seconds_at_pixel_stage"] = round(
                time.monotonic() - pixel_wait_started, 3
            )
            assert pixel_prefetch_pool is not None
            pixel_prefetch_pool.shutdown(wait=True)
    _progress(
        progress,
        "pixel_prefetch",
        93,
        (
            f"Pixel prefetch complete: {len(target_pixel_paths)}/{len(selected_target_pixels)} observing segments are ready"
            if run_pixel_vetting
            else "Pixel checks were disabled by the run configuration"
        ),
        title="Stage 2 pixel preparation complete",
        command="pixel prefetch --join",
        metrics={
            "ready": len(target_pixel_paths),
            "requested": len(selected_target_pixels),
            "wait_seconds": pixel_prefetch_report["wait_seconds_at_pixel_stage"],
        },
    )
    candidates, post_discovery_vetting = annotate_locked_candidates(
        paths,
        candidates,
        target_pixel_paths=target_pixel_paths,
        progress=progress,
        t16_aperture=(
            int(branch.metadata["aperture"])
            if lightcurve_source == "t16"
            else None
        ),
    )
    post_discovery_vetting["pixel_vetting_requested"] = bool(run_pixel_vetting)
    post_discovery_vetting["pixel_download_diagnostics"] = pixel_download_diagnostics
    post_discovery_vetting["pixel_prefetch"] = pixel_prefetch_report

    tiered_candidates = apply_candidate_tiers(candidates)
    candidate_audit = tiered_candidates["audit"]
    final_candidates = tiered_candidates["final"]
    review_candidates = tiered_candidates["review"]
    rejected_candidates = tiered_candidates["rejected"]
    post_discovery_vetting["candidate_tiers"] = {
        "final_candidates": len(final_candidates),
        "review_required": len(review_candidates),
        "rejected_signals": len(rejected_candidates),
        "audit_pool": len(candidate_audit),
    }

    if query_catalog:
        _progress(
            progress,
            "catalog",
            98,
            "Candidate rankings and physical annotations are locked; known-object catalogs are being queried only now",
            title="Catalog answers unsealed after label-free analysis",
            command=f"catalog query --tic {metadata[0].get('tic_id')}",
        )
        catalog = (
            _query_toi_catalog(metadata[0].get("tic_id"))
            if mission == "TESS"
            else []
        )
        catalog_answers = _attach_catalog_answers(candidate_audit, catalog)
        catalog_policy = (
            "queried only after search, matching, fixed validation, alias decisions, "
            "and label-free vetting annotations were locked"
        )
    else:
        _progress(
            progress,
            "catalog",
            98,
            "Catalog queries are disabled during formal blind evaluation; candidates remain label-free and sealed",
            title="Official catalog answers remain sealed",
            command="catalog disabled --blind-run",
        )
        catalog_answers = []
        catalog_policy = (
            "disabled for the sealed blind run; catalog matching is permitted only "
            "after all predictions are hashed"
        )

    rejected = [
        {
            **row,
            "registry_state": "fixed_validation_failed",
        }
        for row in proposal_registry
        if row["holdout_verification"]["passed"] is not True
    ]
    result = {
        "target": target,
        "object": metadata[0].get("object") or target,
        "tic_id": metadata[0].get("tic_id"),
        "mission": mission,
        "lightcurve_source": lightcurve_source,
        "photometry_branch": {
            "key": branch.key,
            "label": branch.label,
            "processing": branch.processing,
            "photometry_family": branch.photometry_family,
            "metadata": dict(branch.metadata),
        },
        "engine_version": "clean-temporal-intersection-v4-red-noise-significance",
        "input_product_policy": product_policy,
        "algorithm": {
            "steps": [
                "three_disjoint_independent_searches",
                "append_only_raw_candidate_ledger",
                "period_phase_event_train_intersection",
                "fixed_ephemeris_measurement_in_all_blocks",
                "shared_evidence_family_maximin_snr_selection",
                "post_selection_event_repeatability_diagnostic",
                "post_selection_fold_event_direction_test",
                "post_validation_integer_alias_compression",
                "locked_ephemeris_red_noise_mes_like_and_phase_null",
                "locked_candidate_physical_vetting_annotation",
                "locked_candidate_pixel_neighbor_annotation",
                "final_candidate_gate_after_all_label_free_checks",
            ],
            "removed_from_active_path": [
                "multibranch_fusion",
                "promote_back",
                "residual_promotion",
                "greedy_cross_fold_merge",
            ],
            "candidate_cap": None,
            "truth_catalog_used_during_search": False,
            "post_discovery_vetting_can_delete_audit_records": False,
            "post_discovery_vetting_can_retune_ephemeris": False,
            "detection_significance_can_delete_audit_records_or_rerank": False,
            "detection_significance_is_official_spoc_mes": False,
            "download_workers": download_workers,
            "target_pixels_prefetched_during_stage_one": bool(run_pixel_vetting),
        },
        "temporal_blocks": [
            {"block": index + 1, "segments": block}
            for index, block in enumerate(blocks)
        ],
        "search_segments": segments,
        "holdout_segments": segments,
        "holdout_segment": None,
        "strict_holdout": False,
        "independent_discovery_folds": True,
        "post_intersection_global_ephemeris_fit": True,
        "detection_significance": detection_significance,
        "timeline_segments": timeline,
        "candidates": final_candidates,
        "final_candidates": final_candidates,
        "review_candidates": review_candidates,
        "rejected_candidates": rejected_candidates,
        "candidate_audit": candidate_audit,
        "candidate_pool_total": len(candidate_audit),
        "final_candidate_count": len(final_candidates),
        "candidate_tier_policy": (
            "the audit pool is append-only; the public candidate list contains only "
            "signals that pass event-train repeatability, multi-time-block red-noise "
            "significance, harmonic independence, physical and lightweight reference "
            "false-positive checks, and pixel localization"
        ),
        "post_discovery_vetting": post_discovery_vetting,
        "candidate_registry": [
            *[
                {**row, "registry_state": "ranked"}
                for row in canonical
            ],
            *equivalents,
            *aliases,
            *rejected,
        ],
        "raw_candidate_ledger": raw_registry,
        "raw_hypothesis_count": len(raw_registry),
        "cross_block_match_edges": match_edges,
        "alias_decisions": alias_decisions,
        "cross_validation": {
            "folds_run": 3,
            "policy": "independent A/B/C search; all cross-block ephemerides measured unchanged; shared-evidence families select the maximum weakest-fold S/N; event repeatability is diagnostic only",
            "matched_proposals": len(proposal_registry),
            "single_block_arbitrations": len(arbiter_registry),
            "passed_before_family_balance": len(preliminary_passed),
            "balanced_families": len(balanced_families),
            "stable_event_families": len(stable_families),
        },
        "catalog_answers": catalog_answers,
        "catalog_policy": catalog_policy,
        "catalog_query_enabled": bool(query_catalog),
        "archive_inventory": serializable_inventory(inventory),
        "target_pixel_inventory": serializable_inventory(pixel_inventory),
        "selected_products": selected,
        "selected_target_pixel_products": selected_target_pixels,
        "target_pixel_files": [str(path) for path in target_pixel_paths],
        "load_diagnostics": load_diagnostics,
        "search_diagnostics": search_diagnostics,
        "runtime_seconds": round(time.monotonic() - started, 2),
    }
    _write_json(run_dir / "candidate_ledger.json", raw_registry)
    _write_json(run_dir / "result.json", result)
    _progress(
        progress,
        "complete",
        100,
        (
            f"Complete: {len(final_candidates)} final candidates passed the full review; "
            f"{len(candidate_audit)} locked signals remain in the audit ledger"
        ),
        title="Auditable pipeline run complete",
        metrics={
            "final_candidates": len(final_candidates),
            "review_candidates": len(review_candidates),
            "rejected_signals": len(rejected_candidates),
            "audit_pool": len(candidate_audit),
            "raw_hypotheses": len(raw_registry),
        },
    )
    return result


__all__ = [
    "DEFAULT_ROOT",
    "run_live_target",
    "_build_cross_block_proposals",
    "_compress_proven_aliases",
    "_validation_decision",
]
