#!/usr/bin/env python3
"""Run opt-in VLora CUDA discovery, voting, and global ephemeris rescue."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from multiplanet_trial.live_web import fixed_ephemeris_acceleration, optimization_v3
from multiplanet_trial.live_web.cuda_hybrid_search import (
    search_block_cuda_complete_grid,
    search_block_cuda_full,
    search_block_cuda_hybrid,
)
from multiplanet_trial.live_web.cuda_science_acceleration import (
    refit_failed_vote_families_cuda,
)
from multiplanet_trial.live_web.ensemble_intersection import (
    DEFAULT_VOTE_SEED_COUNT,
    FAMILY_MODES,
    REFIT_MODES,
    TARGETED_REFIT_MAX_FAMILIES,
    run_vote_intersection,
)
from multiplanet_trial.live_web.vlora_release import (
    CUDA_REFIT_BACKEND,
    CUDA_SEARCH_BACKEND,
    RELEASE_ID,
)
from multiplanet_trial.v03 import event_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--period-max", type=float, default=400.0)
    parser.add_argument("--vote-rounds", type=int, default=5)
    parser.add_argument("--vote-seeds", type=int, default=DEFAULT_VOTE_SEED_COUNT)
    parser.add_argument(
        "--cuda-search",
        choices=("complete-grid", "exact", "cuvarbase", "coarse-only"),
        default="complete-grid",
        help=(
            "complete-grid is the frozen production path: reference Astropy "
            "coarse seeds plus every configured fine-grid cell on CUDA; "
            "exact is a deprecated alias for complete-grid; "
            "cuvarbase uses its different fast BLS definition; "
            "coarse-only retains the older Astropy CPU fine grid"
        ),
    )
    parser.add_argument(
        "--cuda-refit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run complete deterministic global period grids on CUDA.",
    )
    parser.add_argument(
        "--search-workers",
        type=int,
        default=1,
        help="Use one persistent GPU worker so the CUDA context is compiled once.",
    )
    parser.add_argument("--bls-threads", type=int, default=12)
    parser.add_argument("--event-threads", type=int, default=12)
    parser.add_argument("--event-processes", type=int, default=3)
    parser.add_argument("--fixed-ephemeris-threads", type=int, default=12)
    parser.add_argument("--search-fixed-ephemeris-threads", type=int, default=12)
    parser.add_argument("--refit-workers", type=int, default=12)
    parser.add_argument(
        "--refit-mode",
        choices=REFIT_MODES,
        default="targeted",
        help=(
            "targeted builds a catalogue-blind preliminary candidate board "
            "and runs exact CUDA rescue only for its small precision shortlist"
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
    for name in (
        "search_workers",
        "bls_threads",
        "event_threads",
        "event_processes",
        "fixed_ephemeris_threads",
        "search_fixed_ephemeris_threads",
        "refit_workers",
        "vote_seeds",
        "precision_rescue_budget",
    ):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")

    os.environ[optimization_v3.THREAD_ENV] = str(args.bls_threads)
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
    os.environ.setdefault("VLORA_CUDA_FIXED_EPHEMERIS", "auto")

    receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    paths = [
        args.cache_root / str(row["productFilename"])
        for row in receipt["products"]
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing cached light curves: {missing[:5]}")

    def progress(stage: str, message: str, metrics: dict) -> None:
        print(
            json.dumps(
                {"stage": stage, "message": message, "metrics": metrics},
                ensure_ascii=False,
            ),
            flush=True,
        )

    complete_grid_cuda_search = args.cuda_search in ("complete-grid", "exact")
    cuvarbase_full_search = args.cuda_search == "cuvarbase"
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
        search_block_worker=(
            search_block_cuda_complete_grid
            if complete_grid_cuda_search
            else (
                search_block_cuda_full
                if cuvarbase_full_search
                else search_block_cuda_hybrid
            )
        ),
        search_backend=(
            CUDA_SEARCH_BACKEND
            if complete_grid_cuda_search
            else (
                "cuda-cuvarbase-complete-broad-and-fine-grid"
                if cuvarbase_full_search
                else "cuda-cuvarbase-coarse-plus-cpu-float64-refinement"
            )
        ),
        # Targeted mode still performs exact complete-grid CUDA rescue, but
        # only after the catalogue-blind preliminary evidence board has
        # selected a small, period-diverse shortlist.  Every unselected row
        # remains unresolved in the compressed audit.
        refit_runner=(
            refit_failed_vote_families_cuda
            if args.cuda_refit and args.refit_mode != "adaptive"
            else None
        ),
        candidate_family_mode=args.candidate_family_mode,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "release_id": RELEASE_ID,
                "engine": CUDA_SEARCH_BACKEND,
                "refit_engine": (
                    "adaptive-native-exact-or-selective"
                    if args.refit_mode == "adaptive"
                    else CUDA_REFIT_BACKEND
                    if args.cuda_refit
                    else "compiled-cpu"
                ),
                "candidate_family_mode": args.candidate_family_mode,
                "refit_mode": args.refit_mode,
                "precision_rescue_budget": args.precision_rescue_budget,
                "result": str((args.output / "result.json").resolve()),
                "raw_hypotheses": result["raw_hypothesis_count"],
                "promoted_families": result["promoted_family_count"],
                "stable_families": result["stable_family_count"],
                "timing_seconds": result["timing_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
