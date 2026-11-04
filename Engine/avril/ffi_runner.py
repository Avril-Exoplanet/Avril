"""Route raw TESS FFI self-photometry through the optimized Avril pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from multiplanet_trial.live_web.vlora_release import RELEASE_ID
from multiplanet_trial.v05_eval.integrity import atomic_json, sha256_file
from vlora.sources import prepare_input

from .hardware import execution_profile, select_backend
from .source_config import SourceConfig, SourceMode


def _progress(stage: str, percent: int, message: str, detail: dict | None) -> None:
    print(
        json.dumps(
            {
                "stage": stage,
                "percent": int(percent),
                "message": message,
                "detail": detail,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _safe(value: str) -> str:
    return "".join(
        character if character.isalnum() else "_" for character in value
    ).strip("_").lower() or "target"


def _preflight_pro_sector_count(args, cache: Path) -> None:
    """Reject ineligible Avril Pro requests before downloading FFI archives."""

    if args.ffi_engine != "pro":
        return
    if args.local_lightcurves is not None:
        sectors = list(Path(args.local_lightcurves).expanduser().glob("*-lc.fits"))
        count = len([path for path in sectors if not path.name.startswith("._")])
    else:
        from vlora.ffi import discover_ffi_sectors, resolve_target_identity

        preflight = SourceConfig(
            target=args.input,
            source=SourceMode.FFI,
            cache_root=cache,
            tic_id=args.tic_id,
            ra_deg=args.ra,
            dec_deg=args.dec,
        )
        count = len(discover_ffi_sectors(resolve_target_identity(preflight)))
    if count >= 5:
        raise ValueError(
            "Avril Pro is available only when the actual FFI coverage has "
            f"fewer than 5 sectors; this target has {count}. Use --ffi-engine core."
        )


def _public_ffi_manifest(args, prepared, output: Path) -> Path:
    paths = {path.name: path.resolve() for path in prepared.lightcurve_paths}
    products: list[dict] = []
    for row in prepared.products:
        filename = str(row["productFilename"])
        path = paths.get(filename)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"missing self-made FFI light curve: {filename}")
        sector = int(row["sequence_number"])
        products.append(
            {
                "segment": sector,
                "segment_kind": "sector",
                "product_id": filename,
                "filename": filename,
                "data_uri": f"local:ffi-self-photometry/{filename}",
                "url": "local://ffi-self-photometry",
                "author": "AVRIL-SELFMADE-FFI",
                "cadence": f"{float(row.get('exptime') or 1800.0):g}s",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "evaluation_role": "search",
            }
        )
    products.sort(key=lambda row: int(row["segment"]))
    identity = prepared.identity
    target_id = f"TIC {identity.tic_id}" if identity is not None else args.input
    payload = {
        "schema_version": "evaluation-public-v1",
        "dataset_id": f"avril-v3-ffi-{_safe(target_id)}",
        "split": "development",
        "policy": (
            "label-free raw TESS FFI self-photometry; selected aperture is "
            "searched by the optimized Avril pipeline"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "targets": [
            {
                "physical_id": target_id,
                "target_id": target_id,
                "mission": "TESS",
                "products": products,
            }
        ],
    }
    path = output / "ffi_public_manifest.json"
    atomic_json(path, payload)
    return path


def run_ffi(args, root: Path) -> int:
    """Run FFI Core, or sparse-sector Pro, on the current Avril engine."""

    safe_target = _safe(args.input)
    output = (
        args.output_root
        or (root / "results" / f"ffi-{safe_target}-{args.ffi_engine}")
    ).expanduser().resolve()
    cache = (args.cache_root or (root / "cache")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    _preflight_pro_sector_count(args, cache)
    config = SourceConfig(
        target=args.input,
        source=SourceMode.FFI,
        output_dir=output,
        cache_root=cache,
        period_max_days=args.period_max,
        significance_phase_trials=args.phase_trials,
        download_workers=args.download_workers,
        run_pixel_vetting=False,
        query_catalog_after_lock=False,
        tic_id=args.tic_id,
        ra_deg=args.ra,
        dec_deg=args.dec,
        tesscut_size_pixels=args.tesscut_size,
        tesscut_connections_per_file=args.tesscut_connections,
        tesscut_file_workers=args.tesscut_file_workers,
        photometry_workers=args.photometry_workers,
        fixed_ephemeris_threads=min(args.cpu_workers or (os.cpu_count() or 1), 16),
        significance_threads=min(args.cpu_workers or (os.cpu_count() or 1), 16),
        local_lightcurve_dir=args.local_lightcurves,
        local_pixel_dir=args.local_pixels,
    )
    prepared = prepare_input(config, progress=_progress)
    sector_count = len(prepared.products)
    if args.ffi_engine == "pro" and sector_count >= 5:
        raise ValueError(
            "Avril Pro is available only for fewer than 5 sectors; "
            f"the prepared input contains {sector_count}."
        )
    if sector_count < 3:
        raise RuntimeError("Avril requires at least three FFI sectors")
    lightcurve_parents = {path.resolve().parent for path in prepared.lightcurve_paths}
    if len(lightcurve_parents) != 1:
        raise ValueError("prepared FFI light curves must share one cache directory")
    science_cache = next(iter(lightcurve_parents))

    family_mode = (
        "sparse-sector-pro" if args.ffi_engine == "pro" else "core"
    )
    selected_backend = select_backend(args.backend)
    profile = execution_profile(selected_backend, args.cpu_workers)
    manifest_path = _public_ffi_manifest(args, prepared, output)
    source_receipt = {
        "schema_version": "avril-v3-ffi-source-receipt",
        "source_kind": prepared.source_kind,
        "target": prepared.target,
        "sector_count": sector_count,
        "products": prepared.products,
        "lightcurve_paths": [str(path.resolve()) for path in prepared.lightcurve_paths],
        "pixel_paths": [str(path.resolve()) for path in prepared.pixel_paths],
        "identity": (
            {
                "tic_id": prepared.identity.tic_id,
                "ra_deg": prepared.identity.ra_deg,
                "dec_deg": prepared.identity.dec_deg,
            }
            if prepared.identity is not None
            else None
        ),
        "preparation": prepared.receipt,
    }
    atomic_json(output / "ffi_source_receipt.json", source_receipt)
    run_receipt = {
        "schema_version": "avril-v3-run-receipt",
        "release_id": RELEASE_ID,
        "source": "ffi",
        "source_pipeline": "tesscut-self-photometry-optimized-avril",
        "ffi_engine": args.ffi_engine,
        "candidate_family_mode": family_mode,
        "pro_sector_policy": "explicit_only_and_actual_sector_count_less_than_5",
        "actual_sector_count": sector_count,
        "legacy_multiview_pipeline_used": False,
        "raw_pixels_required_for_photometry": True,
        "post_discovery_pixel_localization": args.pixels,
        "backend_requested": args.backend,
        "backend_selected": selected_backend,
        "execution_profile": profile.as_dict(),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
    }
    atomic_json(output / "avril_run_receipt.json", run_receipt)

    scheduler = Path(__file__).resolve().parents[1] / "run_avril_v3_threshold_round.py"
    command = [
        sys.executable,
        str(scheduler),
        "--public",
        str(manifest_path),
        "--cache-root",
        str(science_cache),
        "--output-root",
        str(output),
        "--period-max",
        str(args.period_max),
        "--phase-trials",
        str(args.phase_trials),
        "--vote-rounds",
        str(args.vote_rounds),
        "--max-refit-families",
        str(args.max_refit_families),
        "--max-refit-period-trials",
        str(args.max_refit_period_trials),
        "--precision-rescue-budget",
        str(args.precision_rescue_budget),
        "--candidate-family-mode",
        family_mode,
        "--significance-backend",
        selected_backend,
        *profile.scheduler_arguments(),
    ]
    print(
        json.dumps(
            {"stage": "launch", "command": command, "receipt": run_receipt},
            ensure_ascii=False,
        ),
        flush=True,
    )
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return int(completed.returncode)

    if args.pixels == "followup":
        from .pixel_followup import followup_ffi_result

        for final_result in sorted(output.glob("targets/*/final-science/result.json")):
            report = followup_ffi_result(
                final_result,
                prepared.pixel_paths,
                progress=_progress,
            )
            print(
                json.dumps(
                    {"stage": "pixel_followup_complete", "report": report},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "status": "complete",
                "source": prepared.source_kind,
                "ffi_engine": args.ffi_engine,
                "candidate_family_mode": family_mode,
                "result": str((output / "threshold_round_summary.json").resolve()),
                "pixel_policy": args.pixels,
            }
        ),
        flush=True,
    )
    return 0
