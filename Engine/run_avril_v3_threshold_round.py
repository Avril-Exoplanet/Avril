#!/usr/bin/env python3
"""Run a blind Avril round with self-managed targeted precision rescue.

Raw peaks and promoted validation families are not dense-refit workload.  The
default targeted mode validates every promoted locked ephemeris, selects a
bounded catalogue-blind rescue board, and retains every unselected family in
the audit.  The legacy workload gate remains available only for explicitly
requested non-self-managed refit modes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from multiplanet_trial.v05_eval.dataset_protocol import load_json, validate_public_manifest
from multiplanet_trial.v05_eval.integrity import atomic_json
from run_avril_v3_science_batch import (
    PROJECT_ROOT,
    _products,
    _safe_id,
    _target_run,
    _target_seed,
)


def _stage1_command(args: argparse.Namespace, receipt: Path, output: Path, seed: int) -> list[str]:
    command = [
        sys.executable,
        str(
            PROJECT_ROOT
            / (
                "run_vlora_cuda_vote_experiment.py"
                if args.stage1_engine == "cuda"
                else "run_vlora_vote_experiment.py"
            )
        ),
        "--source-receipt", str(receipt),
        "--cache-root", str(args.cache_root),
        "--output", str(output),
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
        "--seed", str(seed),
    ]
    if args.stage1_engine == "cuda":
        command.extend(("--cuda-search", "complete-grid"))
    return command


def _json_event(line: str) -> dict | None:
    try:
        value = json.loads(line)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _defer(target_root: Path, reason: str, metrics: dict) -> dict:
    record = {
        "status": "deferred_explosive",
        "reason": reason,
        "metrics": metrics,
        "scientific_pipeline_changed": False,
        "resume_supported": True,
    }
    atomic_json(target_root / "deferred.json", record)
    print(json.dumps({"stage": "target_deferred", **record}), flush=True)
    return record


def _run_stage1_with_gates(
    target: dict,
    args: argparse.Namespace,
    target_root: Path,
    receipt_path: Path,
) -> dict:
    stage1 = target_root / "stage1-cuda-vote"
    result_path = stage1 / "result.json"
    deferred_path = target_root / "deferred.json"
    self_managed_refit = args.refit_mode in {"adaptive", "targeted"}
    if deferred_path.is_file() and not self_managed_refit:
        previous = load_json(deferred_path)
        previous_metrics = previous.get("metrics") or {}
        if (
            previous.get("reason") == "refit_workload_limit"
            and (
                int(previous_metrics.get("refit_eligible_families") or 0)
                > args.max_refit_families
                or int(previous_metrics.get("estimated_exact_period_trials") or 0)
                > args.max_refit_period_trials
            )
        ):
            return _defer(target_root, "refit_workload_limit", {
                "refit_eligible_families": int(
                    previous_metrics.get("refit_eligible_families") or 0
                ),
                "max_refit_families": args.max_refit_families,
                "estimated_exact_period_trials": int(
                    previous_metrics.get("estimated_exact_period_trials") or 0
                ),
                "max_refit_period_trials": args.max_refit_period_trials,
                "source": "preserved_threshold_audit",
            })
    if result_path.is_file():
        result = load_json(result_path)
        raw = int(result.get("raw_hypothesis_count") or 0)
        promoted = int(result.get("promoted_family_count") or 0)
        estimate = result.get("refit_workload_estimate") or {}
        refit_families = int(
            estimate.get("eligible_families")
            or (result.get("ephemeris_refit_diagnostics") or {}).get(
                "eligible_families"
            )
            or 0
        )
        period_trials = int(estimate.get("total_period_trials") or 0)
        if (
            not self_managed_refit
            and (
                refit_families > args.max_refit_families
                or period_trials > args.max_refit_period_trials
            )
        ):
            return _defer(target_root, "refit_workload_limit", {
                "refit_eligible_families": refit_families,
                "max_refit_families": args.max_refit_families,
                "estimated_exact_period_trials": period_trials,
                "max_refit_period_trials": args.max_refit_period_trials,
            })
        return {
            "status": "accepted",
            "raw_hypotheses": raw,
            "promoted_families": promoted,
            "refit_eligible_families": refit_families,
            "estimated_exact_period_trials": period_trials,
        }

    log_path = target_root / "run.log"
    command = _stage1_command(
        args,
        receipt_path,
        stage1,
        _target_seed(str(target["physical_id"]), args.seed),
    )
    abort: tuple[str, dict] | None = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            event = _json_event(line)
            if event is not None:
                event["physical_id"] = str(target["physical_id"])
                # Stage 1 is streamed directly here before the science-batch
                # wrapper sees it. Preserve the same namespace that wrapper
                # would add so an engine-level `complete` event cannot be
                # mistaken for completion of the whole workstation job.
                event.setdefault(
                    "pipeline_stage",
                    f"{target['physical_id']}:{args.stage1_engine}-search-vote-refit",
                )
                print(json.dumps(event, ensure_ascii=False), flush=True)
            else:
                print(f"[{target['physical_id']}] {line}", end="", flush=True)
            if not event:
                continue
            metrics = event.get("metrics") or {}
            if event.get("stage") == "refitting":
                refit_families = int(
                    metrics.get("refit_eligible_families") or 0
                )
                period_trials = int(
                    metrics.get("estimated_exact_period_trials") or 0
                )
                if (
                    not self_managed_refit
                    and (
                        refit_families > args.max_refit_families
                        or period_trials > args.max_refit_period_trials
                    )
                ):
                    abort = ("refit_workload_limit", {
                        "refit_eligible_families": refit_families,
                        "max_refit_families": args.max_refit_families,
                        "estimated_exact_period_trials": period_trials,
                        "max_refit_period_trials": args.max_refit_period_trials,
                    })
            if abort is not None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
        return_code = process.wait()
    if abort is not None:
        return _defer(target_root, abort[0], abort[1])
    if return_code != 0:
        raise RuntimeError(f"stage 1 failed with exit code {return_code}")
    result = load_json(result_path)
    estimate = result.get("refit_workload_estimate") or {}
    return {
        "status": "accepted",
        "raw_hypotheses": int(result.get("raw_hypothesis_count") or 0),
        "promoted_families": int(result.get("promoted_family_count") or 0),
        "refit_eligible_families": int(estimate.get("eligible_families") or 0),
        "estimated_exact_period_trials": int(
            estimate.get("total_period_trials") or 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stage1-engine", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--max-refit-families", type=int, default=2_000)
    parser.add_argument("--max-refit-period-trials", type=int, default=3_000_000)
    parser.add_argument("--period-max", type=float, default=400.0)
    parser.add_argument("--vote-rounds", type=int, default=5)
    parser.add_argument("--search-workers", type=int, default=1)
    parser.add_argument("--bls-threads", type=int, default=12)
    parser.add_argument("--event-threads", type=int, default=12)
    parser.add_argument("--event-processes", type=int, default=3)
    parser.add_argument("--fixed-ephemeris-threads", type=int, default=12)
    parser.add_argument("--search-fixed-ephemeris-threads", type=int, default=12)
    parser.add_argument("--refit-workers", type=int, default=12)
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
    parser.add_argument(
        "--candidate-family-mode",
        choices=("core", "sparse-sector-pro"),
        default="core",
    )
    parser.add_argument("--event-workers", type=int, default=12)
    parser.add_argument("--phase-trials", type=int, default=4096)
    parser.add_argument("--significance-threads", type=int, default=12)
    parser.add_argument("--significance-processes", type=int, default=12)
    parser.add_argument("--vetting-threads", type=int, default=1)
    # The full-light-curve physical/reference diagnostics still contain
    # Python orchestration around compiled array kernels.  Four forked Linux
    # workers share the read-only light curve through copy-on-write and avoid
    # the GIL: on the 503-candidate TOI-125 regression this took 12.70 s versus
    # 41.54 s with four threads, with identical physical/reference decisions.
    # Keep an explicit CLI override for platforms where ``fork`` is absent.
    parser.add_argument("--vetting-processes", type=int, default=4)
    parser.add_argument(
        "--significance-backend",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    if args.precision_rescue_budget < 1:
        parser.error("--precision-rescue-budget must be at least 1")

    manifest = load_json(args.public)
    if manifest.get("schema_version") == "clean-injection-public-v1":
        if manifest.get("split") != "injection_recovery":
            raise ValueError("Clean injection manifest has the wrong split")
        if not isinstance(manifest.get("targets"), list) or not manifest["targets"]:
            raise ValueError("Clean injection manifest has no targets")
        for target in manifest["targets"]:
            if not target.get("physical_id") or not target.get("target_id"):
                raise ValueError("Clean injection target identity is incomplete")
            if len(target.get("products") or []) < 3:
                raise ValueError("Clean injection target has fewer than three products")
    else:
        validate_public_manifest(manifest)
    targets = list(manifest["targets"][: args.limit] if args.limit else manifest["targets"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    deferred: list[dict] = []
    failures: list[dict] = []

    for index, target in enumerate(targets):
        physical_id = str(target["physical_id"])
        target_root = args.output_root / "targets" / _safe_id(physical_id)
        target_root.mkdir(parents=True, exist_ok=True)
        products = _products(target)
        receipt_path = target_root / "source_receipt.json"
        atomic_json(receipt_path, {
            "schema_version": "vlora-natural-sector-receipt-v1",
            "physical_id": physical_id,
            "target_id": target["target_id"],
            "mission": target.get("mission") or "TESS",
            "products": products,
            "sector_count": len(products),
            "sector_count_used_for_selection": False,
        })
        print(json.dumps({
            "stage": "target_start",
            "index": index + 1,
            "total": len(targets),
            "physical_id": physical_id,
        }), flush=True)
        try:
            gate = _run_stage1_with_gates(target, args, target_root, receipt_path)
            if gate["status"] != "accepted":
                deferred.append({"physical_id": physical_id, **gate})
                continue
            target_args = args
            if args.significance_backend == "auto":
                target_args = argparse.Namespace(**vars(args))
                # Exact 4096-trial CPU and CUDA results are bit-identical in
                # the sparse-sector regression.  For short light curves the
                # 12-process compiled CPU path is faster because device launch
                # and transfer overhead dominates; long light curves amortize
                # that overhead and benefit from the exact CUDA phase-null
                # kernel.  The threshold is a scheduling choice only.
                target_args.significance_backend = (
                    "cpu" if len(products) <= 8 else "cuda"
                )
                print(
                    json.dumps(
                        {
                            "stage": "significance_backend_selection",
                            "physical_id": physical_id,
                            "sector_count": len(products),
                            "backend": target_args.significance_backend,
                            "phase_trials_each": int(args.phase_trials),
                            "scientific_computation_changed": False,
                        }
                    ),
                    flush=True,
                )
            summary = _target_run(
                target,
                cache_root=args.cache_root,
                output_root=args.output_root,
                args=target_args,
            )
            summary["significance_backend"] = target_args.significance_backend
            summaries.append(summary)
        except Exception as exc:
            failures.append({
                "physical_id": physical_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
        finally:
            atomic_json(args.output_root / "threshold_progress.json", {
                "targets_total": len(targets),
                "targets_complete": len(summaries),
                "targets_deferred": len(deferred),
                "targets_failed": len(failures),
                "complete": summaries,
                "deferred": deferred,
                "failures": failures,
            })

    atomic_json(args.output_root / "threshold_round_summary.json", {
        "schema_version": "avril-v3-threshold-round-v1",
        "refit_mode": args.refit_mode,
        "precision_rescue_budget": args.precision_rescue_budget,
        "legacy_workload_gate_active": args.refit_mode not in {"adaptive", "targeted"},
        "legacy_refit_family_limit": args.max_refit_families,
        "legacy_refit_period_trial_limit": args.max_refit_period_trials,
        "targets_total": len(targets),
        "complete": summaries,
        "deferred": deferred,
        "failures": failures,
        "truth_opened": False,
    })


if __name__ == "__main__":
    main()
