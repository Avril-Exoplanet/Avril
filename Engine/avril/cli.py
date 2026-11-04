"""Single supported command-line interface for Avril V3.0."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

from multiplanet_trial.live_web.vlora_release import PHASE_NULL_TRIALS, RELEASE_ID, VOTE_ROUNDS
from multiplanet_trial.v05_eval.dataset_protocol import load_json, validate_public_manifest
from multiplanet_trial.v05_eval.integrity import atomic_json, sha256_file

from . import __version__
from .hardware import execution_profile, select_backend, system_report
from .paths import default_data_root


def data_root() -> Path:
    return default_data_root()


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avril",
        description="Avril V3.0: fast discovery followed by meaningful-candidate vetting.",
    )
    parser.add_argument("--version", action="version", version=f"Avril {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="network-free CPU/CUDA readiness report")
    doctor.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")

    manifest = subparsers.add_parser("manifest", help="freeze all official SPOC sectors")
    manifest.add_argument("target", help='for example "TOI-786" or "TIC 375059587"')
    manifest.add_argument("--output", type=Path)

    run = subparsers.add_parser("run", help="run the complete production pipeline")
    run.add_argument("input", help="target name or a label-free public manifest JSON")
    run.add_argument(
        "--source",
        choices=("auto", "spoc", "ffi"),
        default="auto",
        help="official SPOC light curves, raw TESS FFI self-photometry, or automatic fallback",
    )
    run.add_argument(
        "--pixels",
        choices=("none", "followup"),
        default="none",
        help="post-discovery localization; followup runs only for final high-priority candidates",
    )
    run.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument("--data-root", type=Path)
    run.add_argument("--cache-root", type=Path)
    run.add_argument("--output-root", type=Path)
    run.add_argument("--cpu-workers", type=int)
    run.add_argument("--download-workers", type=int, default=80)
    run.add_argument("--period-max", type=float, default=400.0)
    run.add_argument("--phase-trials", type=int, default=PHASE_NULL_TRIALS)
    run.add_argument("--vote-rounds", type=int, default=VOTE_ROUNDS)
    run.add_argument("--max-refit-families", type=int, default=2_000)
    run.add_argument("--max-refit-period-trials", type=int, default=3_000_000)
    run.add_argument(
        "--precision-rescue-budget",
        type=int,
        default=128,
        help=(
            "Exact post-ranking rescues per target; all other candidates "
            "remain in the audit"
        ),
    )
    run.add_argument("--limit", type=int)
    run.add_argument("--skip-download", action="store_true")
    run.add_argument("--refresh-manifest", action="store_true")
    run.add_argument(
        "--ffi-engine",
        choices=("core", "pro"),
        default="core",
        help=(
            "FFI uses optimized Core by default; Pro is opt-in and only "
            "valid when the actual target coverage has fewer than five sectors"
        ),
    )
    run.add_argument("--tic-id", type=int)
    run.add_argument("--ra", type=float)
    run.add_argument("--dec", type=float)
    run.add_argument("--tesscut-size", type=int, default=15)
    run.add_argument("--tesscut-connections", type=int, default=16)
    run.add_argument("--tesscut-file-workers", type=int, default=8)
    run.add_argument("--photometry-workers", type=int, default=4)
    run.add_argument("--local-lightcurves", type=Path)
    run.add_argument("--local-pixels", type=Path)

    serve = subparsers.add_parser("serve", help="run the loopback-only web bridge")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--web-root", type=Path)
    serve.add_argument("--open-browser", action="store_true")
    serve.add_argument("--allow-origin", action="append", default=[])
    return parser


def _manifest_for_input(value: str, root: Path, *, refresh: bool = False) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        payload = load_json(candidate)
        validate_public_manifest(payload)
        return candidate.resolve()
    target_key = _safe(value).lower() or "target"
    output = root / "manifests" / f"{target_key}.json"
    if output.is_file() and not refresh:
        validate_public_manifest(load_json(output))
        return output.resolve()
    from .manifest import build_target_manifest

    build_target_manifest(value, output)
    return output.resolve()


def _doctor(requested_backend: str) -> int:
    required = ("numpy", "scipy", "astropy", "pandas", "lightkurve", "numba", "exovetter")
    modules = {}
    for name in required:
        try:
            module = importlib.import_module(name)
            modules[name] = {"ok": True, "version": str(getattr(module, "__version__", "installed"))}
        except Exception as exc:
            modules[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        backend = select_backend(requested_backend)
        backend_error = None
    except RuntimeError as exc:
        backend = requested_backend
        backend_error = str(exc)
    report = {
        "release_id": RELEASE_ID,
        "python": sys.version.split()[0],
        "dependencies": modules,
        "system": system_report(),
        "selected_backend": backend,
        "backend_error": backend_error,
    }
    ready = all(row["ok"] for row in modules.values()) and backend_error is None
    report["status"] = "ready" if ready else "not_ready"
    print(json.dumps(report, indent=2))
    return 0 if ready else 1


def _run(args: argparse.Namespace) -> int:
    root = (args.data_root or data_root()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input).expanduser()
    if args.source == "ffi" and input_path.is_file():
        raise ValueError("--source ffi accepts a target name, not a SPOC manifest")
    if args.source == "ffi":
        from .ffi_runner import run_ffi

        return run_ffi(args, root)
    try:
        manifest_path = _manifest_for_input(
            args.input, root, refresh=args.refresh_manifest
        )
    except RuntimeError as exc:
        if args.source != "auto" or input_path.is_file():
            raise
        print(
            json.dumps(
                {
                    "stage": "source_fallback",
                    "source": "ffi",
                    "reason": str(exc),
                }
            ),
            flush=True,
        )
        from .ffi_runner import run_ffi

        return run_ffi(args, root)
    manifest = load_json(manifest_path)
    summary = validate_public_manifest(manifest)
    selected_backend = select_backend(args.backend)
    profile = execution_profile(selected_backend, args.cpu_workers)
    cache_root = (args.cache_root or (root / "cache" / "spoc")).expanduser().resolve()
    run_key = f"{_safe(str(summary.get('dataset_id') or manifest_path.stem))}-{sha256_file(manifest_path)[:12]}"
    output_root = (args.output_root or (root / "results" / run_key)).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_receipt = {
        "schema_version": "avril-v3-run-receipt",
        "release_id": RELEASE_ID,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source": "spoc",
        "post_discovery_pixel_localization": args.pixels,
        "backend_requested": args.backend,
        "backend_selected": selected_backend,
        "execution_profile": profile.as_dict(),
        "scientific_options": {
            "period_max_days": args.period_max,
            "phase_null_trials": args.phase_trials,
            "vote_rounds": args.vote_rounds,
            "max_refit_families_before_defer": args.max_refit_families,
            "max_refit_period_trials_before_defer": args.max_refit_period_trials,
            "precision_rescue_budget": args.precision_rescue_budget,
        },
    }
    atomic_json(output_root / "avril_run_receipt.json", run_receipt)

    targets = list(manifest["targets"][: args.limit] if args.limit else manifest["targets"])
    if not args.skip_download:
        batch = importlib.import_module("run_avril_v3_science_batch")
        prefetch = batch.prefetch
        download_receipt = prefetch(targets, cache_root, args.download_workers)
        atomic_json(output_root / "download_receipt.json", download_receipt)

    scheduler = importlib.import_module("run_avril_v3_threshold_round")
    command = [
        sys.executable,
        str(Path(scheduler.__file__).resolve()),
        "--public",
        str(manifest_path),
        "--cache-root",
        str(cache_root),
        "--output-root",
        str(output_root),
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
        "--significance-backend",
        selected_backend,
        *profile.scheduler_arguments(),
    ]
    if args.limit:
        command.extend(("--limit", str(args.limit)))
    print(json.dumps({"stage": "launch", "command": command, "receipt": run_receipt}), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return int(completed.returncode)
    if args.pixels == "followup":
        from .pixel_followup import followup_spoc_result

        for final_result in sorted(
            output_root.glob("targets/*/final-science/result.json")
        ):
            target_root = final_result.parent.parent
            receipt_path = target_root / "source_receipt.json"
            receipt = load_json(receipt_path)
            report = followup_spoc_result(
                final_result,
                receipt_path,
                cache_root,
                target=str(receipt.get("target_id") or receipt.get("physical_id")),
                download_workers=args.download_workers,
            )
            print(
                json.dumps(
                    {
                        "stage": "pixel_followup_complete",
                        "target": receipt.get("target_id"),
                        "report": report,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return 0


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        raise SystemExit(_doctor(args.backend))
    if args.command == "manifest":
        from .manifest import build_target_manifest

        root = data_root()
        output = args.output or (root / "manifests" / f"{_safe(args.target).lower()}.json")
        payload = build_target_manifest(args.target, output)
        print(json.dumps({"manifest": str(output.resolve()), "sectors": len(payload["targets"][0]["products"])}))
        return
    if args.command == "serve":
        from vlora.web_server import DEFAULT_ALLOWED_ORIGINS, serve

        serve(
            host=args.host,
            port=args.port,
            web_root=args.web_root,
            allowed_origins=set(DEFAULT_ALLOWED_ORIGINS) | set(args.allow_origin),
            open_browser=args.open_browser,
        )
        return
    raise SystemExit(_run(args))
