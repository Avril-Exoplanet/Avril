"""Fast TESSCut acquisition and label-free multi-aperture FFI photometry."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from avril.source_config import SourceConfig, safe_target_name
from multiplanet_trial.ffi_self_photometry import (
    DEFAULT_RADII,
    extract_sector,
    select_global_radius,
    write_spoc_shaped_lightcurve,
)

from .io import write_json
from .models import PreparedInput, TargetIdentity

Progress = Callable[[str, int, str, dict | None], None]
SEGMENTED_DOWNLOADER = Path(__file__).resolve().parent / "assets" / "download_tesscut_segmented.sh"


def _emit(progress: Progress | None, stage: str, percent: int, message: str, detail=None) -> None:
    if progress:
        progress(stage, percent, message, detail)


def disk_report(destination: Path) -> dict:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths = [Path("/"), destination]
    report = {}
    for path in paths:
        usage = shutil.disk_usage(path)
        report[str(path)] = {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }
    if shutil.disk_usage(destination).free < 2 * 1024**3:
        raise OSError(f"Less than 2 GiB is free at {destination}")
    return report


def resolve_target_identity(config: SourceConfig) -> TargetIdentity:
    if config.tic_id is not None and config.ra_deg is not None and config.dec_deg is not None:
        return TargetIdentity(
            target=config.target,
            tic_id=int(config.tic_id),
            ra_deg=float(config.ra_deg),
            dec_deg=float(config.dec_deg),
        )

    from astroquery.mast import Catalogs

    rows = Catalogs.query_object(config.target, radius=0.001 * u.deg, catalog="TIC")
    if len(rows) == 0:
        raise LookupError(f"No TIC source found for {config.target}")
    requested = re.search(r"(?i)TIC\s*([0-9]+)", config.target)
    selected = None
    if requested:
        wanted = int(requested.group(1))
        selected = next((row for row in rows if int(row["ID"]) == wanted), None)
    if selected is None:
        selected = rows[0]
    return TargetIdentity(
        target=config.target,
        tic_id=int(selected["ID"]),
        ra_deg=float(selected["ra"]),
        dec_deg=float(selected["dec"]),
    )


def discover_ffi_sectors(identity: TargetIdentity) -> list[int]:
    from astroquery.mast import Tesscut

    coordinate = SkyCoord(identity.ra_deg * u.deg, identity.dec_deg * u.deg)
    table = Tesscut.get_sectors(coordinates=coordinate)
    sectors = sorted({int(value) for value in table["sector"]})
    if len(sectors) < 3:
        raise RuntimeError(
            f"TESSCut found only {len(sectors)} sectors; VLora needs at least three"
        )
    return sectors


def tesscut_transfer_metrics(
    destination: Path,
    sectors: list[int],
    config: SourceConfig,
) -> dict:
    """Measure resumable transfers without dropping completed files from total."""

    downloaded = 0
    total = 0
    complete = 0
    active = 0
    every_size_known = True
    for sector in sectors:
        prefix = destination / f"sector-{sector:04d}"
        final = prefix.with_suffix(".zip")
        assembled = prefix.with_suffix(".assembled")
        header = prefix.with_suffix(".headers")
        expected = 0
        if header.exists():
            matches = re.findall(
                r"(?im)^content-range:\s*bytes\s+\d+-\d+/(\d+)",
                header.read_text(encoding="utf-8", errors="ignore"),
            )
            if matches:
                expected = int(matches[-1])
        if final.exists():
            local = final.stat().st_size
            complete += 1
            # The segmented downloader removes temporary headers after a file
            # has been verified.  Its final size is then the authoritative
            # expected size and must remain in the denominator.
            expected = local
        elif assembled.exists():
            local = assembled.stat().st_size
            active += 1
        else:
            pieces = set(destination.glob(f"sector-{sector:04d}.part-[0-9]*"))
            local = sum(path.stat().st_size for path in pieces if path.is_file())
            active += int(local > 0)
        downloaded += min(local, expected) if expected else local
        if expected:
            total += expected
        else:
            every_size_known = False
    known_total = total if every_size_known and total else None
    return {
        "files_verified": complete,
        "files_total": len(sectors),
        "files_active": active,
        "bytes_downloaded": downloaded,
        "bytes_total": known_total,
        "download_percent": (
            round(min(100.0, 100.0 * downloaded / total), 1)
            if known_total
            else None
        ),
        "parallel_files": config.tesscut_file_workers,
        "http_streams": (
            config.tesscut_connections_per_file * config.tesscut_file_workers
        ),
    }


def download_tesscut_archives(
    identity: TargetIdentity,
    sectors: list[int],
    destination: Path,
    config: SourceConfig,
    *,
    progress: Progress | None = None,
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["VLORA_TESSCUT_FILE_WORKERS"] = str(config.tesscut_file_workers)
    command = [
        "/bin/bash",
        str(SEGMENTED_DOWNLOADER),
        str(destination),
        str(identity.ra_deg),
        str(identity.dec_deg),
        str(config.tesscut_size_pixels),
        str(config.tesscut_connections_per_file),
        *(str(sector) for sector in sectors),
    ]
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    last_bytes = -1
    last_emit = 0.0
    while process.poll() is None:
        metrics = tesscut_transfer_metrics(destination, sectors, config)
        now = time.monotonic()
        if (
            int(metrics["bytes_downloaded"]) != last_bytes
            or now - last_emit >= 5.0
        ):
            _emit(
                progress,
                "ffi_downloading",
                5,
                (
                    f"TESSCut: {metrics['files_verified']}/{len(sectors)} verified; "
                    + (
                        f"{metrics['download_percent']:.1f}% of known bytes ready"
                        if metrics["download_percent"] is not None
                        else f"{metrics['bytes_downloaded']} bytes received"
                    )
                ),
                {
                    "title": "TESSCut transfer in progress",
                    "command": "tesscut fetch --segmented --resume",
                    "metrics": metrics,
                },
            )
            last_bytes = int(metrics["bytes_downloaded"])
            last_emit = now
        time.sleep(1.0)
    output, _ = process.communicate()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, command, output=output
        )
    archives = [destination / f"sector-{sector:04d}.zip" for sector in sectors]
    missing = [path for path in archives if not path.exists()]
    if missing:
        raise OSError(f"Missing {len(missing)} validated TESSCut archives")
    metrics = tesscut_transfer_metrics(destination, sectors, config)
    _emit(
        progress,
        "ffi_downloading",
        12,
        f"TESSCut transfer complete: {len(archives)}/{len(archives)} archives verified",
        {
            "title": "All TESSCut archives verified",
            "command": "tesscut verify --all-sectors",
            "metrics": metrics,
        },
    )
    return archives


def _safe_member_name(member: zipfile.ZipInfo) -> str:
    name = Path(member.filename)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError(f"Unsafe TESSCut archive member: {member.filename}")
    return name.name


def extract_tesscut_archives(archives: list[Path], destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for archive in archives:
        with zipfile.ZipFile(archive) as handle:
            members = [row for row in handle.infolist() if row.filename.lower().endswith(".fits")]
            if len(members) != 1:
                raise ValueError(f"Expected one FITS file in {archive}, found {len(members)}")
            member = members[0]
            output = destination / _safe_member_name(member)
            valid = False
            if output.exists():
                try:
                    with fits.open(output, memmap=True) as hdul:
                        valid = len(hdul) >= 2 and "FLUX" in hdul[1].data.names
                except Exception:
                    valid = False
            if not valid:
                temporary = output.with_suffix(output.suffix + ".tmp")
                with handle.open(member) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                os.replace(temporary, output)
            outputs.append(output.resolve())
    return outputs


def make_selfmade_lightcurves(
    identity: TargetIdentity,
    cutouts: list[Path],
    output_dir: Path,
    *,
    workers: int,
) -> tuple[list[Path], dict]:
    coordinate = SkyCoord(identity.ra_deg * u.deg, identity.dec_deg * u.deg)
    with ThreadPoolExecutor(max_workers=min(max(1, workers), len(cutouts))) as pool:
        sectors = list(
            pool.map(
                lambda path: extract_sector(path, coordinate=coordinate),
                cutouts,
            )
        )
    sectors.sort(key=lambda row: row.sector)
    selected_radius, diagnostics = select_global_radius(sectors, DEFAULT_RADII)
    outputs = [
        write_spoc_shaped_lightcurve(
            sector,
            output_dir=output_dir,
            target=identity.target,
            tic_id=identity.tic_id,
            ra=identity.ra_deg,
            dec=identity.dec_deg,
            selected_radius=selected_radius,
            radii=DEFAULT_RADII,
        ).resolve()
        for sector in sectors
    ]
    manifest = {
        "schema_version": "vlora-selfmade-ffi-multiview-v1",
        "target": identity.target,
        "tic_id": identity.tic_id,
        "coordinate_deg": {"ra": identity.ra_deg, "dec": identity.dec_deg},
        "truth_free_photometry": True,
        "period_epoch_duration_used": False,
        "source": "MAST TESSCut calibrated FFI pixels",
        "available_aperture_radii_pixels": list(DEFAULT_RADII),
        "selected_core_radius_pixels": selected_radius,
        "search_photometry": (
            "one globally selected label-free aperture written as a SPOC-shaped "
            "light curve for the optimized Avril pipeline"
        ),
        "all_apertures_retained_in_fits": True,
        "aperture_diagnostics": {str(key): value for key, value in diagnostics.items()},
        "sectors": [row.sector for row in sectors],
        "inputs": [str(path) for path in cutouts],
        "outputs": [str(path) for path in outputs],
    }
    write_json(output_dir / "photometry_manifest.json", manifest)
    return outputs, manifest


def _products(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with fits.open(path, memmap=True) as hdul:
            sector = int(hdul[0].header["SECTOR"])
        rows.append(
            {
                "product_kind": "lightcurve",
                "mission": "TESS",
                "author": "VLORA-SELFMADE-FFI",
                "sequence_number": sector,
                "productFilename": path.name,
                "local_path": str(path),
            }
        )
    return sorted(rows, key=lambda row: int(row["sequence_number"]))


def prepare_ffi_input(
    config: SourceConfig,
    *,
    progress: Progress | None = None,
) -> PreparedInput:
    root = config.cache_root / "ffi" / safe_target_name(config.target)
    report = disk_report(root)
    _emit(
        progress,
        "ffi_capacity",
        3,
        f"FFI data will be stored under {root}",
        {"disk": report},
    )
    identity = resolve_target_identity(config)
    sectors = discover_ffi_sectors(identity)
    _emit(
        progress,
        "ffi_discovery",
        4,
        f"TESSCut covers {len(sectors)} sectors",
        {"sectors": sectors},
    )
    archives = download_tesscut_archives(
        identity,
        sectors,
        root / "tesscut-archives",
        config,
        progress=progress,
    )
    _emit(
        progress,
        "ffi_extracting",
        13,
        f"Extracting {len(archives)} validated TESSCut archives",
        {"metrics": {"files": len(archives)}},
    )
    cutouts = extract_tesscut_archives(archives, root / "cutout-fits")
    _emit(
        progress,
        "ffi_photometry",
        15,
        f"Building label-free aperture photometry for {len(cutouts)} sectors",
        {"metrics": {"sectors": len(cutouts), "workers": config.photometry_workers}},
    )
    lightcurves, manifest = make_selfmade_lightcurves(
        identity,
        cutouts,
        root / "lightcurves",
        workers=config.photometry_workers,
    )
    return PreparedInput(
        source_kind="ffi-selfmade",
        target=config.target,
        products=_products(lightcurves),
        lightcurve_paths=lightcurves,
        pixel_paths=cutouts,
        identity=identity,
        receipt={
            "selection": "official_spoc_unavailable_then_all_tesscut_sectors",
            "fallback_used": True,
            "disk_before_download": report,
            "tesscut": {
                "sectors": sectors,
                "size_pixels": config.tesscut_size_pixels,
                "connections_per_file": config.tesscut_connections_per_file,
                "parallel_files": config.tesscut_file_workers,
                "maximum_http_streams": (
                    config.tesscut_connections_per_file * config.tesscut_file_workers
                ),
                "archives": [str(path) for path in archives],
            },
            "photometry": manifest,
        },
    )
