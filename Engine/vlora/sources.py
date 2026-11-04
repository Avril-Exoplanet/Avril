"""Data-source discovery and deterministic input preparation."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from astropy.io import fits

from avril.source_config import SourceConfig, SourceMode
from multiplanet_trial.live_web import pipeline as core_pipeline
from multiplanet_trial.v03.archive import discover_products, serializable_inventory

from .ffi import prepare_ffi_input
from .models import PreparedInput, TargetIdentity

Progress = Callable[[str, int, str, dict | None], None]


def _emit(progress: Progress | None, stage: str, percent: int, message: str, detail=None) -> None:
    if progress:
        progress(stage, percent, message, detail)


def select_spoc_products(products: list[dict], *, deep: bool = True) -> list[dict]:
    """Select only official SPOC/TESS-SPOC rows, one per sector."""
    allowed = {"SPOC", "TESS-SPOC"}
    official = [
        dict(row)
        for row in products
        if str(row.get("author") or "").upper() in allowed
        and row.get("product_kind") == "lightcurve"
    ]
    return core_pipeline._select_lightcurves(official, max_segments=None, deep=deep)


def discover_spoc(target: str, *, deep: bool = True) -> tuple[list[dict], dict]:
    inventory = discover_products(
        target,
        missions=("TESS",),
        author=("SPOC", "TESS-SPOC"),
        exptime=None,
        include_lightcurves=True,
        include_target_pixels=False,
        limit=None,
    )
    selected = select_spoc_products(inventory.get("products", []), deep=deep)
    return selected, serializable_inventory(inventory)


def _local_products(paths: list[Path]) -> list[dict]:
    products: list[dict] = []
    for path in paths:
        metadata = core_pipeline._segment_metadata(path)
        with fits.open(path, memmap=True) as hdul:
            table = hdul[1].data
            time_name = "TIME" if "TIME" in table.names else "TMID_BJD"
            values = table[time_name]
            finite = sorted(float(value) for value in values if value == value)
            exposure = (
                (finite[len(finite) // 2] - finite[len(finite) // 2 - 1]) * 86400.0
                if len(finite) > 2
                else None
            )
        products.append(
            {
                "product_kind": "lightcurve",
                "mission": "TESS",
                "author": "VLORA-SELFMADE-FFI",
                "sequence_number": int(metadata["segment"]),
                "productFilename": path.name,
                "exptime": exposure,
                "local_path": str(path),
            }
        )
    return sorted(products, key=lambda row: int(row["sequence_number"]))


def load_local_ffi_input(config: SourceConfig) -> PreparedInput:
    directory = config.local_lightcurve_dir
    assert directory is not None
    paths = sorted(
        path.resolve()
        for path in directory.glob("*-lc.fits")
        if not path.name.startswith("._")
    )
    if len(paths) < 3:
        raise RuntimeError("Local FFI input needs at least three light-curve sectors")
    manifest_path = directory / "photometry_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    identity = None
    coordinate = manifest.get("coordinate_deg") or {}
    if manifest.get("tic_id") is not None and coordinate:
        identity = TargetIdentity(
            target=str(manifest.get("target") or config.target),
            tic_id=int(manifest["tic_id"]),
            ra_deg=float(coordinate["ra"]),
            dec_deg=float(coordinate["dec"]),
        )
    pixel_paths: list[Path] = []
    if config.local_pixel_dir is not None:
        pixel_paths = sorted(
            path.resolve()
            for path in config.local_pixel_dir.glob("*.fits")
            if not path.name.startswith("._")
        )
    return PreparedInput(
        source_kind="ffi-selfmade-local",
        target=config.target,
        products=_local_products(paths),
        lightcurve_paths=paths,
        pixel_paths=pixel_paths,
        identity=identity,
        receipt={
            "selection": "explicit_local_lightcurve_directory",
            "lightcurve_directory": str(directory),
            "pixel_directory": str(config.local_pixel_dir) if config.local_pixel_dir else None,
            "photometry_manifest": manifest,
        },
    )


def prepare_input(config: SourceConfig, progress: Progress | None = None) -> PreparedInput:
    if config.local_lightcurve_dir is not None:
        _emit(progress, "source", 1, "Using explicit local FFI photometry")
        return load_local_ffi_input(config)

    if config.source in {SourceMode.AUTO, SourceMode.SPOC}:
        _emit(progress, "source", 1, "Checking official SPOC coverage")
        selected, inventory = discover_spoc(
            config.target, deep=config.deep_cadence_selection
        )
        if len(selected) >= 3:
            _emit(
                progress,
                "source",
                2,
                f"Selected {len(selected)} official SPOC sectors",
                {"segments": [row.get("sequence_number") for row in selected]},
            )
            return PreparedInput(
                source_kind="spoc-official",
                target=config.target,
                products=selected,
                receipt={
                    "selection": "official_spoc_or_tess_spoc_only",
                    "fallback_used": False,
                    "selected_products": selected,
                    "archive_inventory": inventory,
                },
            )
        if config.source == SourceMode.SPOC:
            raise RuntimeError(
                f"Only {len(selected)} official SPOC sectors were found; at least three are required"
            )
        _emit(
            progress,
            "source",
            2,
            "Official SPOC coverage is insufficient; switching to self-made FFI photometry",
            {"official_spoc_sectors": len(selected)},
        )

    if config.mission != "TESS":
        raise RuntimeError("Automatic FFI fallback is currently available only for TESS")
    return prepare_ffi_input(config, progress=progress)
