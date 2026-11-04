"""Run the clean engine on a label-free blind manifest without opening truth.

The runner deliberately separates three phases:

1. globally prefetch every frozen light-curve product with high I/O concurrency;
2. run a small number of targets concurrently (each target already uses three
   CPU processes for its independent temporal folds); and
3. assemble label-free predictions only when every target has a valid receipt.

Catalog queries and target-pixel downloads are disabled here.  They are a
post-seal audit and are not allowed to influence discovery or ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Mapping
from zoneinfo import ZoneInfo

from multiplanet_trial.v03.archive import (
    _download_one_resumable,
    configure_lightkurve_cache,
)
from multiplanet_trial.v05_eval.dataset_protocol import (
    load_json,
    validate_public_manifest,
)
from multiplanet_trial.v05_eval.integrity import atomic_json, sha256_file


ENGINE_VERSION = "clean-temporal-intersection-v4-red-noise-significance"


def normalise_public_product(product: Mapping) -> dict:
    """Convert an evaluation-manifest product to the archive downloader schema."""
    filename = str(product.get("filename") or product.get("product_id") or "")
    if not filename.endswith(".fits"):
        raise ValueError(f"Invalid frozen FITS product: {filename!r}")
    segment = int(product["segment"])
    data_uri = str(product.get("data_uri") or f"mast:TESS/product/{filename}")
    cadence = str(product.get("cadence") or "120s").removesuffix("s")
    try:
        exptime = float(cadence)
    except ValueError:
        exptime = 120.0
    return {
        "productFilename": filename,
        "dataURI": data_uri,
        "mission": f"TESS Sector {segment}",
        "product_kind": "lightcurve",
        "sequence_number": segment,
        "author": str(product.get("author") or "SPOC"),
        "exptime": exptime,
        "size": product.get("size_bytes"),
        "evaluation_role": product.get("evaluation_role"),
    }


def _target_key(target: Mapping) -> str:
    return str(target["physical_id"]).replace(":", "_").replace(" ", "_")


def _options_payload(args: argparse.Namespace) -> dict:
    return {
        "engine_version": ENGINE_VERSION,
        "period_max_days": float(args.period_max),
        "significance_phase_trials": int(args.significance_phase_trials),
        "target_workers": int(args.target_workers),
        "per_target_download_workers": int(args.per_target_download_workers),
        "global_download_workers": int(args.download_workers),
        "pixel_vetting": False,
        "catalog_query": False,
        "candidate_cap": None,
    }


def _run_key(target_input: Path, release_manifest: Path, options: Mapping) -> str:
    payload = {
        "target_input_sha256": sha256_file(target_input),
        "release_manifest_sha256": sha256_file(release_manifest),
        "options": options,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _receipt_current(receipt: Path, result: Path, expected_key: str) -> bool:
    if not receipt.is_file() or not result.is_file():
        return False
    try:
        payload = load_json(receipt)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "ok"
        and payload.get("run_key") == expected_key
        and payload.get("result_sha256") == sha256_file(result)
    )


def _prefetch_products(targets: list[dict], cache_root: Path, workers: int) -> dict:
    destination = configure_lightkurve_cache(cache_root) / "archive-products/tess/lightcurve"
    products: dict[str, dict] = {}
    for target in targets:
        for raw in target["products"]:
            product = normalise_public_product(raw)
            products.setdefault(product["productFilename"], product)
    started = time.monotonic()
    failures = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {
            pool.submit(_download_one_resumable, product, destination): product
            for product in products.values()
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
                    f"download {completed}/{len(futures)} "
                    f"({completed / elapsed:.2f} files/s; failures={len(failures)})",
                    flush=True,
                )
    if failures:
        raise RuntimeError(f"{len(failures)} frozen products failed: {failures[:3]}")
    return {
        "products": len(products),
        "workers": max(1, int(workers)),
        "runtime_seconds": round(time.monotonic() - started, 3),
        "destination": str(destination.resolve()),
    }


def _compact_candidate(candidate: Mapping) -> dict:
    keys = (
        "candidate_id",
        "rank",
        "period_days",
        "epoch_btjd",
        "duration_hours",
        "candidate_score",
        "candidate_source",
        "discovered_blocks",
        "features",
        "holdout_verification",
        "detection_significance",
        "physical_vetting",
        "pixel_status",
        "hard_rejected",
        "alias_group_size",
        "alternate_periods",
    )
    output = {key: candidate.get(key) for key in keys if key in candidate}
    output["score_is_probability"] = False
    return output


def _run_one(
    target: dict,
    target_input: Path,
    target_dir: Path,
    log_path: Path,
    receipt_path: Path,
    run_key: str,
    args: argparse.Namespace,
) -> dict:
    command = [
        sys.executable,
        "-m",
        "multiplanet_trial.live_web.run_clean_target",
        str(target["target_id"]),
        "--mission",
        str(target.get("mission") or "TESS"),
        "--output",
        str(target_dir),
        "--cache-root",
        str(args.cache_root),
        "--period-max",
        str(float(args.period_max)),
        "--significance-phase-trials",
        str(int(args.significance_phase_trials)),
        "--download-workers",
        str(int(args.per_target_download_workers)),
        "--products-json",
        str(target_input),
        "--skip-pixels",
        "--skip-catalog",
    ]
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[name] = "1"
    started = time.monotonic()
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        log.write(("\nCOMMAND " + " ".join(command) + "\n").encode("utf-8"))
        completed = subprocess.run(
            command,
            cwd=args.project_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result_path = target_dir / "result.json"
    if completed.returncode != 0 or not result_path.is_file():
        return {
            "physical_id": target["physical_id"],
            "status": "error",
            "returncode": completed.returncode,
            "runtime_seconds": round(time.monotonic() - started, 3),
            "log": str(log_path.resolve()),
        }
    result = load_json(result_path)
    if result.get("catalog_query_enabled") is not False or result.get("catalog_answers"):
        raise RuntimeError(f"Blind isolation failed for {target['physical_id']}")
    receipt = {
        "schema_version": "clean-blind-target-receipt-v1",
        "physical_id": target["physical_id"],
        "status": "ok",
        "run_key": run_key,
        "result": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path),
        "candidate_count": len(result.get("candidates", [])),
        "catalog_query_enabled": False,
        "pixel_vetting_requested": False,
        "runtime_seconds": round(time.monotonic() - started, 3),
    }
    atomic_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one sealed, label-free clean blind batch")
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--download-workers", type=int, default=24)
    parser.add_argument("--target-workers", type=int, default=2)
    parser.add_argument("--per-target-download-workers", type=int, default=12)
    parser.add_argument("--period-max", type=float, default=100.0)
    parser.add_argument("--significance-phase-trials", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-prefetch", action="store_true")
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.public = args.public.resolve()
    args.release_manifest = args.release_manifest.resolve()
    args.run_root = args.run_root.resolve()
    args.cache_root = args.cache_root.resolve()

    public = load_json(args.public)
    summary = validate_public_manifest(public)
    if summary["split"] != "blind":
        raise ValueError("This entry point accepts only the reserved blind split")
    targets = list(public["targets"][: args.limit] if args.limit else public["targets"])
    options = _options_payload(args)
    args.run_root.mkdir(parents=True, exist_ok=True)
    for child in ("inputs", "targets", "logs", "receipts"):
        (args.run_root / child).mkdir(parents=True, exist_ok=True)
    run_config = {
        "schema_version": "clean-blind-run-v1",
        "created_at_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "dataset_id": public["dataset_id"],
        "split": public["split"],
        "public_manifest": str(args.public),
        "public_manifest_sha256": sha256_file(args.public),
        "release_manifest": str(args.release_manifest),
        "release_manifest_sha256": sha256_file(args.release_manifest),
        "truth_opened": False,
        "catalog_query_enabled": False,
        "options": options,
        "targets_requested": len(targets),
    }
    atomic_json(args.run_root / "run_config.json", run_config)

    if not args.skip_prefetch:
        prefetch = _prefetch_products(targets, args.cache_root, args.download_workers)
        atomic_json(args.run_root / "download_receipt.json", prefetch)

    jobs = []
    for target in targets:
        key = _target_key(target)
        target_input = args.run_root / "inputs" / f"{key}.json"
        products = [normalise_public_product(row) for row in target["products"]]
        atomic_json(
            target_input,
            {
                "physical_id": target["physical_id"],
                "target_id": target["target_id"],
                "products": products,
            },
        )
        target_dir = args.run_root / "targets" / key
        log_path = args.run_root / "logs" / f"{key}.log"
        receipt_path = args.run_root / "receipts" / f"{key}.json"
        key_hash = _run_key(target_input, args.release_manifest, options)
        if not _receipt_current(receipt_path, target_dir / "result.json", key_hash):
            jobs.append((target, target_input, target_dir, log_path, receipt_path, key_hash))

    started = time.monotonic()
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.target_workers))) as pool:
        futures = {
            pool.submit(_run_one, *job, args): job[0]["physical_id"] for job in jobs
        }
        completed = 0
        for future in as_completed(futures):
            physical_id = futures[future]
            try:
                receipt = future.result()
            except Exception as exc:
                receipt = {
                    "physical_id": physical_id,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if receipt.get("status") != "ok":
                failures.append(receipt)
            completed += 1
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"search {completed}/{len(jobs)} ({completed / elapsed:.3f} targets/s; "
                f"failures={len(failures)})",
                flush=True,
            )

    receipts = []
    target_outputs = []
    for target in targets:
        key = _target_key(target)
        receipt_path = args.run_root / "receipts" / f"{key}.json"
        if not receipt_path.is_file():
            continue
        receipt = load_json(receipt_path)
        receipts.append(receipt)
        result = load_json(receipt["result"])
        target_outputs.append(
            {
                "physical_id": target["physical_id"],
                "status": "ok",
                "candidate_pool_total": int(result.get("candidate_pool_total", 0)),
                "candidates": [_compact_candidate(row) for row in result.get("candidates", [])],
                "result_sha256": receipt["result_sha256"],
            }
        )
    progress = {
        "dataset_id": public["dataset_id"],
        "targets_total": len(targets),
        "targets_complete": len(target_outputs),
        "targets_failed_this_invocation": failures,
        "truth_opened": False,
    }
    atomic_json(args.run_root / "progress.json", progress)
    if len(target_outputs) != len(targets):
        raise RuntimeError(
            f"Blind batch incomplete: {len(target_outputs)}/{len(targets)}; rerun resumes checkpoints"
        )
    predictions = {
        "schema_version": "candidate-predictions-clean-v1",
        "dataset_id": public["dataset_id"],
        "split": public["split"],
        "score_name": "candidate_score",
        "score_is_probability": False,
        "public_manifest": str(args.public),
        "public_manifest_sha256": sha256_file(args.public),
        "release_manifest": str(args.release_manifest),
        "release_manifest_sha256": sha256_file(args.release_manifest),
        "truth_opened": False,
        "catalog_query_enabled": False,
        "search_options": options,
        "targets": target_outputs,
        "summary": {
            "targets": len(target_outputs),
            "candidates": sum(len(row["candidates"]) for row in target_outputs),
        },
    }
    predictions_path = args.run_root / "blind_predictions.json"
    atomic_json(predictions_path, predictions)
    atomic_json(
        args.run_root / "blind_predictions_seal.json",
        {
            "schema_version": "clean-blind-prediction-seal-v1",
            "sealed_at_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "truth_opened": False,
            "predictions": str(predictions_path),
            "predictions_sha256": sha256_file(predictions_path),
            "public_manifest_sha256": sha256_file(args.public),
            "release_manifest_sha256": sha256_file(args.release_manifest),
            "targets": len(target_outputs),
            "candidates": predictions["summary"]["candidates"],
        },
    )
    print(json.dumps(predictions["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
