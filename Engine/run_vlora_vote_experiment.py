#!/usr/bin/env python3
"""Run the optimized VLora multi-group vote experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from multiplanet_trial.live_web import fixed_ephemeris_acceleration, optimization_v3
from multiplanet_trial.live_web.ensemble_intersection import (
    DEFAULT_REFIT_MODE,
    DEFAULT_VOTE_SEED_COUNT,
    FAMILY_MODES,
    REFIT_MODES,
    TARGETED_REFIT_MAX_FAMILIES,
    run_vote_intersection,
)
from multiplanet_trial.v03 import event_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--period-max", type=float, default=400.0)
    parser.add_argument("--vote-rounds", type=int, default=5)
    parser.add_argument(
        "--vote-seeds",
        type=int,
        default=DEFAULT_VOTE_SEED_COUNT,
        help="Deterministic sector-resampling seeds; search still runs only once",
    )
    parser.add_argument("--search-workers", type=int, default=3)
    parser.add_argument("--bls-threads", type=int, default=4)
    parser.add_argument(
        "--bls-min-periods",
        type=int,
        default=optimization_v3.DEFAULT_MIN_PERIODS,
    )
    parser.add_argument("--event-threads", type=int, default=4)
    parser.add_argument(
        "--event-processes",
        type=int,
        default=max(1, min(3, (os.cpu_count() or 1) // 4)),
        help=(
            "Processes per temporal search block for exhaustive event-train "
            "fitting; results are collected in deterministic seed order"
        ),
    )
    parser.add_argument(
        "--search-fixed-ephemeris-threads",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 1) // 3)),
    )
    parser.add_argument(
        "--fixed-ephemeris-threads",
        type=int,
        default=min(10, os.cpu_count() or 1),
    )
    parser.add_argument("--refit-workers", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument(
        "--refit-mode",
        choices=REFIT_MODES,
        default=DEFAULT_REFIT_MODE,
        help=(
            "targeted ranks unresolved families without catalogue truth and "
            "runs dense exact rescue only on the precision shortlist"
        ),
    )
    parser.add_argument(
        "--precision-rescue-budget",
        type=int,
        default=TARGETED_REFIT_MAX_FAMILIES,
        help=(
            "Maximum unresolved families receiving dense exact refit in "
            "targeted mode; all other families remain in the audit"
        ),
    )
    parser.add_argument("--seed", type=int, default=307210830)
    parser.add_argument(
        "--candidate-family-mode",
        choices=FAMILY_MODES,
        default="core",
    )
    args = parser.parse_args()

    if args.bls_threads < 1:
        parser.error("--bls-threads must be at least 1")
    if args.vote_seeds < 1:
        parser.error("--vote-seeds must be at least 1")
    if args.bls_min_periods < 1:
        parser.error("--bls-min-periods must be at least 1")
    if args.event_threads < 1:
        parser.error("--event-threads must be at least 1")
    if args.event_processes < 1:
        parser.error("--event-processes must be at least 1")
    if args.fixed_ephemeris_threads < 1:
        parser.error("--fixed-ephemeris-threads must be at least 1")
    if args.search_fixed_ephemeris_threads < 1:
        parser.error("--search-fixed-ephemeris-threads must be at least 1")
    if args.precision_rescue_budget < 1:
        parser.error("--precision-rescue-budget must be at least 1")
    os.environ[optimization_v3.THREAD_ENV] = str(args.bls_threads)
    os.environ[optimization_v3.MIN_PERIODS_ENV] = str(args.bls_min_periods)
    os.environ[event_search.THREAD_ENV] = str(args.event_threads)
    os.environ[event_search.PROCESS_ENV] = str(args.event_processes)
    os.environ[fixed_ephemeris_acceleration.THREAD_ENV] = str(
        args.fixed_ephemeris_threads
    )
    os.environ[optimization_v3.SEARCH_FIXED_THREADS_ENV] = str(
        args.search_fixed_ephemeris_threads
    )
    os.environ["HOUXING_TARGETED_REFIT_MAX_FAMILIES"] = str(
        args.precision_rescue_budget
    )

    receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    paths = [
        args.cache_root / str(row["productFilename"])
        for row in receipt["products"]
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing cached light curves: {missing}")

    def progress(stage: str, message: str, metrics: dict) -> None:
        print(
            json.dumps(
                {"stage": stage, "message": message, "metrics": metrics},
                ensure_ascii=False,
            ),
            flush=True,
        )

    result = run_vote_intersection(
        paths,
        output_dir=args.output,
        period_max=args.period_max,
        vote_rounds=args.vote_rounds,
        vote_seed_count=args.vote_seeds,
        search_workers=args.search_workers,
        refit_workers=args.refit_workers,
        refit_mode=args.refit_mode,
        seed=args.seed,
        progress=progress,
        candidate_family_mode=args.candidate_family_mode,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "result": str((args.output / "result.json").resolve()),
                "raw_hypotheses": result["raw_hypothesis_count"],
                "promoted_families": result["promoted_family_count"],
                "stable_families": result["stable_family_count"],
                "search_workers": args.search_workers,
                "vote_seed_count": args.vote_seeds,
                "bls_threads_per_search_worker": args.bls_threads,
                "maximum_concurrent_bls_threads": (
                    args.search_workers * args.bls_threads
                ),
                "event_threads_per_search_worker": args.event_threads,
                "maximum_concurrent_event_threads": (
                    args.search_workers * args.event_threads
                ),
                "event_processes_per_search_worker": args.event_processes,
                "maximum_concurrent_event_processes": (
                    args.search_workers * args.event_processes
                ),
                "fixed_ephemeris_threads": args.fixed_ephemeris_threads,
                "search_fixed_ephemeris_threads_per_worker": (
                    args.search_fixed_ephemeris_threads
                ),
                "refit_workers": args.refit_workers,
                "refit_mode": args.refit_mode,
                "precision_rescue_budget": args.precision_rescue_budget,
                "candidate_family_mode": args.candidate_family_mode,
                "timing_seconds": result["timing_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
