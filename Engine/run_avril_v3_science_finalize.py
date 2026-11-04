#!/usr/bin/env python3
"""Finish locked Avril V3 candidates through the meaningful-signal funnel."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from pathlib import Path

from multiplanet_trial.live_web.vlora_release import (
    NUMBA_CACHE_TAG,
    RELEASE_ID,
    SCHEMA_VERSION,
    SIGNIFICANCE_BACKEND,
)

os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path.home() / ".cache" / "avril" / "numba" / NUMBA_CACHE_TAG),
)

from multiplanet_trial.live_web import optimization_v4
from multiplanet_trial.live_web import pipeline as core
from multiplanet_trial.live_web.detection_significance import (
    annotate_detection_significance,
    annotate_detection_significance_cuda_pipeline,
    annotate_detection_significance_parallel,
)
from multiplanet_trial.live_web.ensemble_intersection import (
    _write_gzip_json,
    _write_json,
)
from multiplanet_trial.live_web.fast_science_finalize import (
    candidate_key,
    direct_alias_winners,
    eligible_for_phase_null,
    merge_candidates,
    refresh_harmonic_decision,
    skipped_significance,
    skipped_vetting,
)
from multiplanet_trial.live_web.meaningful_candidate_gate import (
    apply_meaningful_candidate_gate,
)
from multiplanet_trial.live_web.post_discovery_vetting import (
    annotate_locked_candidates,
)
from multiplanet_trial.live_web.science_review import (
    rank_science_candidates,
    science_review_annotation,
)
from run_vlora_vote_harmonic_finalize import _compact_candidate


def _read_json_document(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _reuse_event_measurements(
    candidates: list[dict],
    source: dict,
    source_result: Path,
) -> int:
    audit_name = (source.get("audit_files") or {}).get(
        "meaningful_gate_candidates"
    )
    if not audit_name:
        return 0
    audit_path = Path(str(audit_name))
    if not audit_path.is_absolute():
        audit_path = source_result.parent / audit_path
    if not audit_path.is_file():
        return 0
    detailed = _read_json_document(audit_path)
    by_key = {candidate_key(row): row for row in detailed}
    reused = 0
    for candidate in candidates:
        measured = by_key.get(candidate_key(candidate))
        if measured is None:
            continue
        repeatability = (
            measured.get("holdout_verification", {}).get(
                "event_repeatability", {}
            )
        )
        if not isinstance(repeatability.get("event_support"), list):
            continue
        verification = dict(candidate.get("holdout_verification") or {})
        verification["event_repeatability"] = repeatability
        candidate["holdout_verification"] = verification
        reused += 1
    return reused


def _compact_science_candidate(candidate: dict) -> dict:
    compact = _compact_candidate(candidate)
    compact["candidate_id"] = candidate.get("candidate_id")
    compact["discovery_rank"] = candidate.get("discovery_rank")
    compact["science_rank"] = candidate.get("science_rank")
    compact["rank"] = candidate.get("rank")
    compact["direct_harmonic_alias"] = candidate.get("direct_harmonic_alias")
    meaningful = dict(candidate.get("meaningful_candidate_gate") or {})
    gap = dict(meaningful.get("event_gap_diagnostics") or {})
    gap.pop("events", None)
    if gap:
        meaningful["event_gap_diagnostics"] = gap
    compact["meaningful_candidate_gate"] = meaningful or None
    significance = dict(compact.get("detection_significance") or {})
    significance.pop("events", None)
    significance.pop("duration_noise_models", None)
    compact["detection_significance"] = significance
    pixel = dict(compact.get("pixel_vetting") or {})
    pixel.pop("per_sector", None)
    compact["pixel_vetting"] = pixel
    return compact


def _cuda_available(required: bool) -> bool:
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception as exc:
        if required:
            raise RuntimeError(
                "CUDA significance was requested but is unavailable"
            ) from exc
        return False


def _progress(stage: str, percent: int, message: str, details=None) -> None:
    print(
        json.dumps(
            {
                "stage": stage,
                "percent": int(percent),
                "message": message,
                "details": details or {},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-trials", type=int, default=4096)
    parser.add_argument("--significance-threads", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--significance-processes", type=int, default=4)
    parser.add_argument("--vetting-threads", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--vetting-processes", type=int, default=4)
    parser.add_argument(
        "--significance-backend",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--target-pixel-path", action="append", type=Path, default=[])
    parser.add_argument("--harmonic-relations", type=Path)
    args = parser.parse_args()
    if args.phase_trials < 0:
        parser.error("--phase-trials must be non-negative")
    if min(
        args.significance_threads,
        args.significance_processes,
        args.vetting_threads,
        args.vetting_processes,
    ) < 1:
        parser.error("thread/process counts must be at least one")

    cuda_significance = (
        args.significance_backend != "cpu"
        and _cuda_available(args.significance_backend == "cuda")
    )
    os.environ["VLORA_CUDA_PHASE_NULL"] = "on" if cuda_significance else "off"
    os.environ["HOUXING_SIGNIFICANCE_THREADS"] = str(args.significance_threads)
    os.environ.setdefault(
        "VLORA_SIGNIFICANCE_PREPARATION_WORKERS",
        str(min(4, args.significance_threads)),
    )
    os.environ.setdefault("VLORA_SIGNIFICANCE_PREPARATION_AHEAD", "8")
    optimization_v4.install()

    started = time.monotonic()
    source = json.loads(args.source_result.read_text(encoding="utf-8"))
    relations_path = args.harmonic_relations
    if relations_path is None:
        relation_name = (source.get("audit_files") or {}).get(
            "harmonic_relations"
        )
        if relation_name:
            relations_path = args.source_result.parent / str(relation_name)
    harmonic_relations: list[dict] = []
    if relations_path is not None and relations_path.is_file():
        opener = gzip.open if relations_path.suffix == ".gz" else open
        with opener(relations_path, "rt", encoding="utf-8") as handle:
            harmonic_relations = json.load(handle)
    receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    paths = [
        args.cache_root / str(row["productFilename"])
        for row in receipt["products"]
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing cached light curves: {missing}")
    metadata = [core._segment_metadata(path) for path in paths]
    paths = [
        row[1]
        for row in sorted(
            zip(metadata, paths, strict=True), key=lambda row: row[0]["segment"]
        )
    ]
    branch, load_diagnostics = core._load_pdcsap_branch(paths)
    blocks = [list(map(int, block)) for block in source["base_search_blocks"]]
    candidates = [dict(row) for row in source["ranked_candidates"]]
    for rank, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = candidate.get(
            "candidate_id"
        ) or candidate.get("proposal_id")
        candidate["discovery_rank"] = int(candidate.get("discovery_rank") or rank)
        candidate["rank"] = rank
        candidate["ensemble_rank"] = int(candidate.get("ensemble_rank") or rank)
    detailed_event_measurements_reused = _reuse_event_measurements(
        candidates,
        source,
        args.source_result,
    )
    if len({candidate_key(row) for row in candidates}) != len(candidates):
        raise ValueError("source candidate identities are not unique")

    tic_id = metadata[0].get("tic_id") if metadata else None
    catalog_cache = args.output.parent / "toi_catalog.json"
    if catalog_cache.is_file():
        catalog = json.loads(catalog_cache.read_text(encoding="utf-8"))
    else:
        catalog = core._query_toi_catalog(tic_id)
        _write_json(catalog_cache, catalog)
    funnel_started = time.monotonic()
    candidates, coherent, funnel_report = apply_meaningful_candidate_gate(
        candidates,
        branch,
        catalog=catalog,
        threads=args.vetting_threads,
    )
    funnel_seconds = time.monotonic() - funnel_started
    meaningful_ids = {candidate_key(row) for row in coherent}
    skipped_event = [
        row for row in candidates if candidate_key(row) not in meaningful_ids
    ]
    print(
        json.dumps(
            {
                "stage": "physical_reference_gate",
                "all_candidates": len(candidates),
                "eligible": len(coherent),
                "short_circuited_but_retained": len(skipped_event),
                "detailed_event_measurements_reused": (
                    detailed_event_measurements_reused
                ),
                "meaningful_candidate_funnel": funnel_report,
            }
        ),
        flush=True,
    )
    physical_started = time.monotonic()
    vetted, vetting_report = annotate_locked_candidates(
        paths,
        coherent,
        target_pixel_paths=args.target_pixel_path,
        workers=args.vetting_threads,
        processes=args.vetting_processes,
        progress=_progress,
    )
    physical_seconds = time.monotonic() - physical_started
    merged_vetting = merge_candidates(
        candidates,
        [
            *vetted,
            *(
                skipped_vetting(
                    row,
                    reason=(
                        "skipped_after_"
                        + str(
                            (row.get("meaningful_candidate_gate") or {}).get(
                                "disposition"
                            )
                            or "meaningful_candidate_gate"
                        )
                    ),
                )
                for row in skipped_event
            ),
        ],
    )

    phase_candidates = [row for row in merged_vetting if eligible_for_phase_null(row)]
    alias_winners = direct_alias_winners(phase_candidates, harmonic_relations)
    first_pass_candidates = [
        row for row in phase_candidates if candidate_key(row) not in alias_winners
    ]
    # Conflicting resolved edges can theoretically form a cycle.  Running one
    # deterministic member breaks that cycle without deleting or assuming any
    # result; every remaining member falls back to exact evaluation if it does
    # not obtain a significant direct winner.
    if phase_candidates and not first_pass_candidates:
        first_pass_candidates = [
            min(
                phase_candidates,
                key=lambda row: (
                    int(row.get("discovery_rank") or 10**9),
                    candidate_key(row),
                ),
            )
        ]
    print(
        json.dumps(
            {
                "stage": "exact_phase_null_gate",
                "all_candidates": len(merged_vetting),
                "eligible_before_direct_alias_deferral": len(phase_candidates),
                "first_pass": len(first_pass_candidates),
                "deferred_direct_aliases": len(phase_candidates)
                - len(first_pass_candidates),
                "phase_trials_each": int(args.phase_trials),
                "backend": "cuda-exact" if cuda_significance else "cpu-compiled",
            }
        ),
        flush=True,
    )
    significance_started = time.monotonic()

    def run_significance(rows: list[dict]) -> tuple[list[dict], dict]:
        if not rows:
            return [], {
                "schema_version": "locked-red-noise-significance-stage-v1",
                "candidate_count": 0,
                "phase_trials_each": int(args.phase_trials),
                "passes_7p1_reference": 0,
                "locked_ephemerides_verified": True,
            }
        if cuda_significance:
            return annotate_detection_significance_cuda_pipeline(
                rows,
                branch,
                blocks,
                phase_trials=args.phase_trials,
                preparation_workers=min(4, args.significance_threads),
                preparation_ahead=8,
                progress=_progress,
            )
        if args.significance_processes > 1:
            return annotate_detection_significance_parallel(
                rows,
                branch,
                blocks,
                phase_trials=args.phase_trials,
                processes=args.significance_processes,
                threads_per_process=max(
                    1,
                    args.significance_threads // args.significance_processes,
                ),
                progress=_progress,
            )
        return annotate_detection_significance(
            rows,
            branch,
            blocks,
            phase_trials=args.phase_trials,
            progress=_progress,
        )

    first_significant, first_report = run_significance(first_pass_candidates)
    provisional = merge_candidates(merged_vetting, first_significant)
    first_ids = {candidate_key(row) for row in first_significant}
    for candidate in provisional:
        if candidate_key(candidate) in first_ids:
            refresh_harmonic_decision(candidate, provisional)
    first_by_id = {candidate_key(row): row for row in provisional}
    passing_first = {
        key
        for key in first_ids
        if (first_by_id[key].get("detection_significance") or {}).get(
            "passes_7p1_reference"
        )
        is True
    }
    verified_alias_winners = {
        loser: sorted(winners & passing_first)
        for loser, winners in alias_winners.items()
        if winners & passing_first
    }
    fallback_candidates = [
        row
        for row in phase_candidates
        if candidate_key(row) in alias_winners
        and candidate_key(row) not in verified_alias_winners
    ]
    fallback_significant, fallback_report = run_significance(fallback_candidates)
    significant = [*first_significant, *fallback_significant]
    significance_report = dict(first_report)
    significance_report["candidate_count"] = len(significant)
    significance_report["passes_7p1_reference"] = sum(
        bool((row.get("detection_significance") or {}).get("passes_7p1_reference"))
        for row in significant
    )
    significance_report["first_pass_candidate_count"] = len(first_significant)
    significance_report["fallback_candidate_count"] = len(fallback_significant)
    significance_report["direct_alias_short_circuit_count"] = len(
        verified_alias_winners
    )
    significance_report["fallback_report"] = fallback_report
    significance_seconds = time.monotonic() - significance_started
    all_after_significance = merge_candidates(merged_vetting, significant)
    measured_ids = {candidate_key(row) for row in significant}
    for candidate in all_after_significance:
        key = candidate_key(candidate)
        if key in measured_ids:
            refresh_harmonic_decision(candidate, all_after_significance)
        elif key in verified_alias_winners:
            candidate["direct_harmonic_alias"] = {
                "winner_proposal_ids": verified_alias_winners[key],
                "winner_passed_exact_phase_null": True,
                "relation_scope": "direct_resolved_event_train_only",
                "transitive_family_membership_used": False,
            }
            candidate["detection_significance"] = skipped_significance(
                candidate,
                requested_trials=args.phase_trials,
                reason="skipped_after_significant_direct_harmonic_winner",
            )
        else:
            candidate["detection_significance"] = skipped_significance(
                candidate,
                requested_trials=args.phase_trials,
            )
        candidate["science_review"] = science_review_annotation(candidate)
    significance_report["passes_7p1_reference"] = sum(
        bool((row.get("detection_significance") or {}).get("passes_7p1_reference"))
        for row in all_after_significance
    )
    ranked = rank_science_candidates(all_after_significance)

    if len(ranked) != len(candidates):
        raise AssertionError("survey short-circuit changed candidate count")
    input_ephemerides = {
        candidate_key(row): (
            float(row["period_days"]),
            float(row["epoch_btjd"]),
            float(row["duration_hours"]),
        )
        for row in candidates
    }
    output_ephemerides = {
        candidate_key(row): (
            float(row["period_days"]),
            float(row["epoch_btjd"]),
            float(row["duration_hours"]),
        )
        for row in ranked
    }
    if output_ephemerides != input_ephemerides:
        raise AssertionError("survey short-circuit changed a locked ephemeris")

    elapsed = time.monotonic() - started
    priority_counts: dict[str, int] = {}
    for candidate in ranked:
        priority = str(candidate["science_review"]["priority"])
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    significance_report.update(
        {
            "backend": "cuda-exact-phase-null" if cuda_significance else "cpu-compiled",
            "eligible_candidate_count_before_direct_alias_deferral": len(
                phase_candidates
            ),
            "exactly_evaluated_candidate_count": len(significant),
            "short_circuited_candidate_count": len(candidates) - len(significant),
            "wall_seconds": round(significance_seconds, 6),
            "positive_decision_preserving_short_circuit": True,
        }
    )
    if cuda_significance:
        from multiplanet_trial.live_web.cuda_phase_null import get_cuda_phase_null_measurer

        significance_report["cuda"] = get_cuda_phase_null_measurer().diagnostics()

    args.output.mkdir(parents=True, exist_ok=True)
    _write_gzip_json(args.output / "science_candidates.json.gz", ranked)
    result = {
        **source,
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "engine_version": "avril-v3-meaningful-candidate-funnel",
        "production_significance_backend": SIGNIFICANCE_BACKEND,
        "ranked_candidates": [_compact_science_candidate(row) for row in ranked],
        "science_annotation": {
            "policy": (
                "all locked candidates retained; deterministic failed gates short-circuit "
                "only downstream work that cannot change a positive decision"
            ),
            "candidate_count": len(ranked),
            "candidate_count_preserved": True,
            "locked_ephemerides_preserved": True,
            "event_gate_eligible": len(coherent),
            "meaningful_candidate_funnel": funnel_report,
            "catalog": catalog,
            "exact_phase_null_viable_before_alias_deferral": len(phase_candidates),
            "exact_phase_null_evaluated": len(significant),
            "direct_alias_short_circuited": len(verified_alias_winners),
            "phase_trials_each_eligible_candidate": int(args.phase_trials),
            "priority_counts": priority_counts,
            "detection_significance": significance_report,
            "post_discovery_vetting": vetting_report,
            "load_diagnostics": load_diagnostics,
        },
        "audit_files": {
            **source.get("audit_files", {}),
            "period_ranking_source": str(args.source_result.resolve()),
            "science_candidates": "science_candidates.json.gz",
        },
        "scientific_status": (
            "non-destructive candidate ranking with exact locked-ephemeris phase-null "
            "trials for every candidate that can still pass all earlier deterministic gates"
        ),
    }
    timing = dict(source.get("timing_seconds") or {})
    timing.update(
        {
            "physical_reference_gate_wall": round(physical_seconds, 3),
            "meaningful_candidate_funnel_wall": round(funnel_seconds, 3),
            "exact_phase_null_gate_wall": round(significance_seconds, 3),
            "science_annotation_wall": round(elapsed, 3),
        }
    )
    result["timing_seconds"] = timing
    _write_json(args.output / "result.json", result)
    print(
        json.dumps(
            {
                "status": "complete",
                "result": str((args.output / "result.json").resolve()),
                "candidates": len(ranked),
                "event_gate_eligible": len(coherent),
                "phase_null_eligible": len(phase_candidates),
                "phase_null_evaluated": len(significant),
                "direct_alias_short_circuited": len(verified_alias_winners),
                "priority_counts": priority_counts,
                "seconds": round(elapsed, 3),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
