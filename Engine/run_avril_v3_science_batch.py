#!/usr/bin/env python3
"""Resumable label-free batch for the complete Avril V3 pipeline.

The input manifest freezes target identities and every naturally available
SPOC light curve.  Catalog truth is never read here.  The stage-1 backend is
explicit and defaults to the exact CPU path for the Mac port.  Light curves
are prefetched with high I/O concurrency before science processing starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from multiplanet_trial.clean_eval.run_blind_batch import normalise_public_product
from multiplanet_trial.live_web.vlora_release import (
    PHASE_NULL_TRIALS,
    RELEASE_ID,
    SIGNIFICANCE_BACKEND,
    VOTE_ROUNDS,
)
from multiplanet_trial.v03.archive import _download_one_resumable
from multiplanet_trial.v05_eval.dataset_protocol import (
    load_json,
    validate_public_manifest,
)
from multiplanet_trial.v05_eval.integrity import atomic_json, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parent
ENGINE_FILES = (
    "avril/__init__.py",
    "avril/cli.py",
    "avril/hardware.py",
    "avril/manifest.py",
    "run_avril_v3_science_batch.py",
    "run_avril_v3_threshold_round.py",
    "run_avril_v3_science_finalize.py",
    "run_vlora_cuda_vote_experiment.py",
    "run_vlora_vote_experiment.py",
    "run_vlora_vote_harmonic_finalize.py",
    "run_vlora_vote_rational_finalize.py",
    "multiplanet_trial/candidate_physical_features_t16_multiview.py",
    "multiplanet_trial/exovetter_features.py",
    "multiplanet_trial/ffi_self_photometry.py",
    "multiplanet_trial/vetting.py",
    "multiplanet_trial/clean_eval/run_blind_batch.py",
    "multiplanet_trial/v03/archive.py",
    "multiplanet_trial/v03/event_search.py",
    "multiplanet_trial/v03/hybrid.py",
    "multiplanet_trial/v03/lightkurve_evidence.py",
    "multiplanet_trial/v03/multibranch_search.py",
    "multiplanet_trial/v03/pixel_vetting.py",
    "multiplanet_trial/v05_eval/dataset_protocol.py",
    "multiplanet_trial/v05_eval/integrity.py",
    "multiplanet_trial/v07_eval/multibranch_search.py",
    "multiplanet_trial/v08_eval/multibranch_search.py",
    "multiplanet_trial/live_web/candidate_tiers.py",
    "multiplanet_trial/live_web/compiled_event_repeatability.py",
    "multiplanet_trial/live_web/compiled_red_noise_gate.py",
    "multiplanet_trial/live_web/cpp_exact_bls.py",
    "multiplanet_trial/live_web/cuda_astropy_bls.py",
    "multiplanet_trial/live_web/cuda_fixed_ephemeris.py",
    "multiplanet_trial/live_web/cupy_astropy_bls.py",
    "multiplanet_trial/live_web/cuda_hybrid_search.py",
    "multiplanet_trial/live_web/cuda_phase_null.py",
    "multiplanet_trial/live_web/cuda_science_acceleration.py",
    "multiplanet_trial/live_web/detection_significance.py",
    "multiplanet_trial/live_web/ensemble_intersection.py",
    "multiplanet_trial/live_web/event_coherence.py",
    "multiplanet_trial/live_web/event_train_funnel.py",
    "multiplanet_trial/live_web/exact_bls_acceleration.py",
    "multiplanet_trial/live_web/fixed_ephemeris_acceleration.py",
    "multiplanet_trial/live_web/fast_science_finalize.py",
    "multiplanet_trial/live_web/fast_survey_vetting.py",
    "multiplanet_trial/live_web/harmonic_families.py",
    "multiplanet_trial/live_web/optimization_v1.py",
    "multiplanet_trial/live_web/optimization_v3.py",
    "multiplanet_trial/live_web/optimization_v4.py",
    "multiplanet_trial/live_web/pipeline.py",
    "multiplanet_trial/live_web/post_discovery_vetting.py",
    "multiplanet_trial/live_web/rational_aliases.py",
    "multiplanet_trial/live_web/result_storage_v1.py",
    "multiplanet_trial/live_web/science_review.py",
    "multiplanet_trial/live_web/sealed_validation.py",
    "multiplanet_trial/live_web/significance_acceleration.py",
    "multiplanet_trial/live_web/temporal_consensus.py",
    "multiplanet_trial/live_web/vlora_release.py",
    "multiplanet_trial/live_web/native/exact_bls_kernel.cpp",
    "scripts/compare_vlora_cross_platform.py",
)


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _engine_manifest() -> dict[str, str]:
    return {
        name: sha256_file(PROJECT_ROOT / name)
        for name in ENGINE_FILES
    }


def _canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _target_seed(physical_id: str, fallback: int) -> int:
    digest = hashlib.sha256(
        f"avril-v3.0|{fallback}|{physical_id}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _products(target: dict) -> list[dict]:
    products = [normalise_public_product(row) for row in target["products"]]
    products.sort(key=lambda row: int(row["sequence_number"]))
    return products


def prefetch(targets: list[dict], cache_root: Path, workers: int) -> dict:
    cache_root.mkdir(parents=True, exist_ok=True)
    unique: dict[str, dict] = {}
    for target in targets:
        for product in _products(target):
            unique.setdefault(str(product["productFilename"]), product)
    started = time.monotonic()
    failures: list[dict] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {
            pool.submit(_download_one_resumable, product, cache_root): product
            for product in unique.values()
        }
        for future in as_completed(futures):
            product = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(
                    {
                        "product": product["productFilename"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            completed += 1
            if completed % 20 == 0 or completed == len(futures):
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    json.dumps(
                        {
                            "stage": "download",
                            "files_complete": completed,
                            "files_total": len(futures),
                            "failures": len(failures),
                            "files_per_second": round(completed / elapsed, 3),
                        }
                    ),
                    flush=True,
                )
    receipt = {
        "products": len(unique),
        "workers": max(1, int(workers)),
        "runtime_seconds": round(time.monotonic() - started, 3),
        "cache_root": str(cache_root.resolve()),
        "failures": failures,
    }
    if failures:
        raise RuntimeError(f"{len(failures)} light curves failed to download")
    return receipt


def _run(command: list[str], log_path: Path, label: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": label, "command": command}), flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\nCOMMAND " + " ".join(command) + "\n")
        completed = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert completed.stdout is not None
        for line in completed.stdout:
            handle.write(line)
            handle.flush()
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payload.setdefault("pipeline_stage", label)
                print(json.dumps(payload, ensure_ascii=False), flush=True)
        return_code = completed.wait()
    if return_code != 0:
        raise RuntimeError(f"{label} failed with exit code {return_code}")


def _target_run(
    target: dict,
    *,
    cache_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict:
    physical_id = str(target["physical_id"])
    target_root = output_root / "targets" / _safe_id(physical_id)
    final_result = target_root / "final-science" / "result.json"
    receipt_path = target_root / "source_receipt.json"
    products = _products(target)
    if len(products) < 3:
        raise ValueError(f"{physical_id} has fewer than three light-curve sectors")
    atomic_json(
        receipt_path,
        {
            "schema_version": "vlora-natural-sector-receipt-v1",
            "physical_id": physical_id,
            "target_id": target["target_id"],
            "mission": target.get("mission") or "TESS",
            "products": products,
            "sector_count": len(products),
            "sector_count_used_for_selection": False,
        },
    )
    stage1 = target_root / "stage1-cuda-vote"
    stage3 = target_root / "stage3-event-integer"
    stage4 = target_root / "stage4-harmonic-families"
    final = final_result.parent
    log = target_root / "run.log"
    if not (stage1 / "result.json").is_file():
        stage1_program = (
            "run_vlora_cuda_vote_experiment.py"
            if args.stage1_engine == "cuda"
            else "run_vlora_vote_experiment.py"
        )
        stage1_command = [
                sys.executable,
                str(PROJECT_ROOT / stage1_program),
                "--source-receipt", str(receipt_path),
                "--cache-root", str(cache_root),
                "--output", str(stage1),
                "--period-max", str(args.period_max),
                "--vote-rounds", str(args.vote_rounds),
                "--search-workers", str(args.search_workers),
                "--bls-threads", str(args.bls_threads),
                "--event-threads", str(args.event_threads),
                "--event-processes", str(args.event_processes),
                "--fixed-ephemeris-threads", str(args.fixed_ephemeris_threads),
                "--search-fixed-ephemeris-threads", str(args.search_fixed_ephemeris_threads),
                "--refit-workers", str(args.refit_workers),
                "--refit-mode", args.refit_mode,
                "--precision-rescue-budget", str(args.precision_rescue_budget),
                "--candidate-family-mode", args.candidate_family_mode,
                "--seed", str(_target_seed(physical_id, args.seed)),
        ]
        if args.stage1_engine == "cuda":
            stage1_command.extend(("--cuda-search", "complete-grid"))
        _run(
            stage1_command,
            log,
            f"{physical_id}:{args.stage1_engine}-search-vote-refit",
        )
    if not (stage3 / "result.json").is_file():
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "run_vlora_vote_harmonic_finalize.py"),
                "--source-result", str(stage1 / "result.json"),
                "--source-validated", str(stage1 / "validated_families.json.gz"),
                "--source-receipt", str(receipt_path),
                "--cache-root", str(cache_root),
                "--output", str(stage3),
                "--workers", str(args.event_workers),
            ],
            log,
            f"{physical_id}:event-repeatability-integer-aliases",
        )
    if not (stage4 / "result.json").is_file():
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "run_vlora_vote_rational_finalize.py"),
                "--source-result", str(stage3 / "result.json"),
                "--event-families", str(stage3 / "event_repeatability_families.json.gz"),
                "--output", str(stage4),
            ],
            log,
            f"{physical_id}:harmonic-family-arbitration",
        )
    if not final_result.is_file():
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "run_avril_v3_science_finalize.py"),
                "--source-result", str(stage4 / "result.json"),
                "--source-receipt", str(receipt_path),
                "--cache-root", str(cache_root),
                "--output", str(final),
                "--phase-trials", str(args.phase_trials),
                "--significance-threads", str(args.significance_threads),
                "--significance-processes", str(args.significance_processes),
                "--significance-backend", args.significance_backend,
                "--vetting-threads", str(args.vetting_threads),
                "--vetting-processes", str(args.vetting_processes),
            ],
            log,
            f"{physical_id}:red-noise-physical-review",
        )
    result = load_json(final_result)
    candidates = list(result.get("ranked_candidates", []))
    priorities: dict[str, int] = {}
    for candidate in candidates:
        priority = str((candidate.get("science_review") or {}).get("priority"))
        priorities[priority] = priorities.get(priority, 0) + 1
    summary = {
        "physical_id": physical_id,
        "target_id": target["target_id"],
        "sector_count": len(products),
        "status": "ok",
        "result": str(final_result.resolve()),
        "result_sha256": sha256_file(final_result),
        "candidate_count": len(candidates),
        "priority_counts": priorities,
        "timing_seconds": result.get("timing_seconds"),
    }
    atomic_json(target_root / "receipt.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--download-workers", type=int, default=80)
    parser.add_argument("--skip-prefetch", action="store_true")
    parser.add_argument("--prefetch-only", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--period-max", type=float, default=400.0)
    parser.add_argument("--vote-rounds", type=int, default=VOTE_ROUNDS)
    parser.add_argument("--stage1-engine", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--search-workers", type=int, default=3)
    parser.add_argument("--bls-threads", type=int, default=3)
    parser.add_argument("--event-threads", type=int, default=3)
    parser.add_argument("--event-processes", type=int, default=3)
    parser.add_argument("--fixed-ephemeris-threads", type=int, default=10)
    parser.add_argument("--search-fixed-ephemeris-threads", type=int, default=3)
    parser.add_argument("--refit-workers", type=int, default=10)
    parser.add_argument(
        "--refit-mode",
        choices=(
            "targeted",
            "adaptive",
            "native-selective",
            "native-shared",
            "native-exact",
            "batched-exact",
            "exact",
            "shared",
        ),
        default="targeted",
    )
    parser.add_argument("--precision-rescue-budget", type=int, default=128)
    parser.add_argument("--event-workers", type=int, default=10)
    parser.add_argument(
        "--candidate-family-mode",
        choices=("core", "sparse-sector-pro"),
        default="core",
    )
    parser.add_argument("--phase-trials", type=int, default=PHASE_NULL_TRIALS)
    parser.add_argument("--significance-threads", type=int, default=10)
    parser.add_argument("--significance-processes", type=int, default=4)
    parser.add_argument("--vetting-threads", type=int, default=10)
    parser.add_argument("--vetting-processes", type=int, default=10)
    parser.add_argument(
        "--significance-backend",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    if args.precision_rescue_budget < 1:
        parser.error("--precision-rescue-budget must be at least 1")

    public = load_json(args.public)
    validate_public_manifest(public)
    all_targets = list(public["targets"])
    stop = None if args.limit is None else args.start_index + args.limit
    targets = all_targets[args.start_index:stop]
    if not targets:
        raise ValueError("selected target slice is empty")
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": "avril-v3-science-batch-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "public_manifest": str(args.public.resolve()),
        "public_manifest_sha256": sha256_file(args.public),
        "truth_opened": False,
        "release_id": RELEASE_ID,
        "production_significance_backend": SIGNIFICANCE_BACKEND,
        "catalog_query_enabled": False,
        "target_slice": {"start": args.start_index, "count": len(targets)},
        "natural_sector_counts": [len(row["products"]) for row in targets],
        "sector_count_used_for_selection": False,
        "options": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
        "engine_files": _engine_manifest(),
    }
    config["engine_manifest_sha256"] = _canonical_hash(config["engine_files"])
    atomic_json(args.output_root / "run_config.json", config)

    if not args.skip_prefetch:
        receipt = prefetch(targets, args.cache_root, args.download_workers)
        atomic_json(args.output_root / "download_receipt.json", receipt)
    if args.prefetch_only:
        return

    summaries: list[dict] = []
    failures: list[dict] = []
    for index, target in enumerate(targets, start=1):
        started = time.monotonic()
        try:
            summary = _target_run(
                target,
                cache_root=args.cache_root,
                output_root=args.output_root,
                args=args,
            )
            summaries.append(summary)
        except Exception as exc:
            failures.append(
                {
                    "physical_id": target["physical_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        atomic_json(
            args.output_root / "progress.json",
            {
                "targets_requested": len(targets),
                "targets_complete": len(summaries),
                "targets_failed": failures,
                "truth_opened": False,
            },
        )
        print(
            json.dumps(
                {
                    "stage": "target_complete",
                    "index": index,
                    "total": len(targets),
                    "physical_id": target["physical_id"],
                    "seconds": round(time.monotonic() - started, 3),
                    "failures": len(failures),
                }
            ),
            flush=True,
        )
    predictions = {
        "schema_version": "avril-v3-science-predictions-v1",
        "truth_opened": False,
        "catalog_query_enabled": False,
        "run_config_sha256": sha256_file(args.output_root / "run_config.json"),
        "targets": summaries,
        "failures": failures,
    }
    atomic_json(args.output_root / "predictions.json", predictions)
    atomic_json(
        args.output_root / "predictions_seal.json",
        {
            "predictions_sha256": sha256_file(args.output_root / "predictions.json"),
            "targets_complete": len(summaries),
            "targets_failed": len(failures),
            "truth_opened": False,
        },
    )
    if failures:
        raise RuntimeError(f"batch completed with {len(failures)} target failures")


if __name__ == "__main__":
    main()
