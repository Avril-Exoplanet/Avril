#!/usr/bin/env python3
"""Attach per-event evidence and compress only proven integer aliases."""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from multiplanet_trial.live_web import pipeline as core
from multiplanet_trial.live_web.ensemble_intersection import (
    _write_gzip_json,
    _write_json,
    attach_vote_channel,
    score_ledger_on_vote_groups,
)
from multiplanet_trial.live_web.event_coherence import (
    annotate_event_coherence,
    event_coherence_quality,
)
from multiplanet_trial.live_web.compiled_event_repeatability import (
    attach_event_repeatability_batch,
)
from multiplanet_trial.live_web.sealed_validation import (
    prepare_event_repeatability_views,
)


_WORKER_BRANCH = None
_WORKER_BLOCKS = None


def _initialize_worker(branch, blocks: list[list[int]]) -> None:
    global _WORKER_BRANCH, _WORKER_BLOCKS
    _WORKER_BRANCH = branch
    _WORKER_BLOCKS = blocks


def _event_batch_worker(batch: list[tuple[int, dict]]) -> list[tuple[int, dict]]:
    if _WORKER_BRANCH is None or _WORKER_BLOCKS is None:
        raise RuntimeError("event-repeatability worker was not initialized")
    output: list[tuple[int, dict]] = []
    for index, source in batch:
        candidate = dict(source)
        verification = dict(candidate["holdout_verification"])
        candidate["holdout_verification"] = verification
        if not verification.get("event_repeatability"):
            core._attach_event_repeatability(
                candidate, verification, _WORKER_BRANCH
            )
        core._apply_fold_event_direction_test(candidate, _WORKER_BLOCKS)
        annotate_event_coherence(candidate, _WORKER_BLOCKS)
        output.append((index, candidate))
    return output


def _ranking_quality(candidate: dict) -> tuple:
    vote = candidate["ensemble_vote"]
    verification = candidate["holdout_verification"]
    measurements = [
        row
        for row in verification.get("block_measurements", [])
        if row.get("outcome") in {"strong", "weak"}
    ]
    snrs = [float(row.get("snr") or 0.0) for row in measurements]
    repeatability = verification.get("event_repeatability", {})
    coherence = event_coherence_quality(candidate)
    seed_consensus = candidate.get("vote_seed_consensus") or {}
    multi_seed_quality: tuple = ()
    if int(seed_consensus.get("seed_count") or 1) > 1:
        multi_seed_quality = (
            float(seed_consensus.get("promotion_support_rate") or 0.0),
            float(seed_consensus.get("refit_support_rate") or 0.0),
            -float(seed_consensus.get("mean_rank") or np.inf),
            -float(seed_consensus.get("rank_variance") or np.inf),
        )
    return (
        coherence[0],
        coherence[1],
        *multi_seed_quality,
        int(vote["positive_round_count"]),
        int(vote["positive_groups"]),
        int(vote["strong_groups"]),
        -int(vote["contradicting_groups"]),
        len(candidate.get("discovered_blocks", [])),
        coherence[4],
        coherence[2],
        coherence[3],
        min(snrs, default=0.0),
        float(repeatability.get("support_fraction") or 0.0),
        float(verification.get("combined_positive_snr") or 0.0),
        float(candidate.get("source_strength") or 0.0),
    )


def _compact_candidate(candidate: dict) -> dict:
    vote = dict(candidate.get("ensemble_vote") or {})
    vote.pop("measurements", None)
    long_vote = dict(candidate.get("long_period_vote") or {})
    long_vote.pop("measurements", None)
    verification = dict(candidate.get("holdout_verification") or {})
    repeatability = dict(verification.get("event_repeatability") or {})
    event_count = len(repeatability.pop("event_support", []))
    repeatability.pop("modulo", None)
    repeatability["event_support_rows_in_audit"] = event_count
    verification["event_repeatability"] = repeatability
    return {
        "proposal_id": candidate.get("proposal_id"),
        "period_days": candidate.get("period_days"),
        "epoch_btjd": candidate.get("epoch_btjd"),
        "duration_hours": candidate.get("duration_hours"),
        "source_methods": candidate.get("source_methods"),
        "source_strength": candidate.get("source_strength"),
        "ensemble_score": candidate.get("ensemble_score"),
        "ensemble_vote": vote,
        "long_period_vote": long_vote or None,
        "vote_seed_consensus": candidate.get("vote_seed_consensus"),
        "ensemble": candidate.get("ensemble"),
        "discovered_blocks": candidate.get("discovered_blocks"),
        "holdout_verification": verification,
        "harmonic_family": candidate.get("harmonic_family"),
        "event_coherence": candidate.get("event_coherence"),
        "ephemeris_balance": candidate.get("ephemeris_balance"),
        "ephemeris_refit": candidate.get("ephemeris_refit"),
        "ensemble_rank": candidate.get("ensemble_rank"),
        "detection_significance": candidate.get("detection_significance"),
        "physical_vetting": candidate.get("physical_vetting"),
        "reference_vetting": candidate.get("reference_vetting"),
        "pixel_vetting": candidate.get("pixel_vetting"),
        "vetting_summary": candidate.get("vetting_summary"),
        "science_review": candidate.get("science_review"),
        "equivalent_proposal_ids": candidate.get("equivalent_proposal_ids", []),
        "alternate_periods": [
            {
                "period_days": row.get("period_days"),
                "proposal_id": row.get("proposal_id"),
                "alias_decision": row.get("alias_decision"),
                "harmonic_family_relation": row.get(
                    "harmonic_family_relation"
                ),
            }
            for row in candidate.get("alternate_periods", [])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-validated", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    args = parser.parse_args()

    started = time.monotonic()
    source = json.loads(args.source_result.read_text(encoding="utf-8"))
    with gzip.open(args.source_validated, "rt", encoding="utf-8") as handle:
        validated = json.load(handle)
    stable = [
        row
        for row in validated
        if row["holdout_verification"].get("passed") is True
    ]

    receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    paths = [
        args.cache_root / str(row["productFilename"])
        for row in receipt["products"]
    ]
    metadata = [core._segment_metadata(path) for path in paths]
    paired = sorted(zip(metadata, paths), key=lambda row: row[0]["segment"])
    paths = [row[1] for row in paired]
    branch, _load_diagnostics = core._load_pdcsap_branch(paths)
    blocks = [list(map(int, block)) for block in source["base_search_blocks"]]
    stable = score_ledger_on_vote_groups(
        stable,
        branch,
        source["vote_groups"],
    )
    long_period_vote_groups = source.get("long_period_vote_groups") or []
    long_indices = [
        index
        for index, row in enumerate(stable)
        if float(row.get("period_days") or 0.0) >= 20.0
    ]
    if long_period_vote_groups and long_indices:
        rescored = attach_vote_channel(
            [stable[index] for index in long_indices],
            branch,
            long_period_vote_groups,
            field="long_period_vote",
        )
        for index, row in zip(long_indices, rescored):
            stable[index] = row

    workers = max(1, min(int(args.workers), os.cpu_count() or 1, len(stable)))
    print(
        json.dumps(
            {
                "stage": "event_repeatability",
                "message": (
                    "Measuring every predicted event for every stable family "
                    "with the exact compiled batch kernel"
                ),
                "families": len(stable),
                "workers": workers,
                "backend": "compiled-batch-exact",
            }
        ),
        flush=True,
    )
    prepared_views = prepare_event_repeatability_views(branch)
    measured_stable = attach_event_repeatability_batch(
        stable,
        prepared_views=prepared_views,
        holdout_segment=None,
        threads=workers,
    )
    event_backend = "compiled-batch-exact"
    if measured_stable is None:
        # Minimal installations without Numba retain the exact reference
        # implementation.  This fallback is deliberately complete, not a
        # reduced scientific path.
        event_backend = "reference-exact"
        measured_stable = []
        for candidate in stable:
            verification = candidate["holdout_verification"]
            if not verification.get("event_repeatability"):
                core._attach_event_repeatability(candidate, verification, branch)
            measured_stable.append(candidate)
    for candidate in measured_stable:
        core._apply_fold_event_direction_test(candidate, blocks)
        annotate_event_coherence(candidate, blocks)
    print(
        json.dumps(
            {
                "stage": "event_repeatability",
                "message": f"Per-event diagnostics {len(measured_stable)}/{len(stable)}",
                "families_completed": len(measured_stable),
                "families_total": len(stable),
                "backend": event_backend,
            }
        ),
        flush=True,
    )
    event_seconds = time.monotonic() - started

    direction_passed = [
        row
        for row in measured_stable
        if row["holdout_verification"].get("passed") is True
    ]
    print(
        json.dumps(
            {
                "stage": "deduplicating",
                "message": "Collapsing only directly equivalent validated event trains",
                "direction_passed": len(direction_passed),
            }
        ),
        flush=True,
    )
    dedup_started = time.monotonic()
    deduplicated, equivalents = core._collapse_duplicate_passes(
        direction_passed, branch=None
    )
    dedup_seconds = time.monotonic() - dedup_started

    print(
        json.dumps(
            {
                "stage": "harmonics",
                "message": "Compressing only integer aliases proven by event residues",
                "deduplicated_families": len(deduplicated),
            }
        ),
        flush=True,
    )
    harmonic_started = time.monotonic()
    canonical, aliases, alias_decisions = core._compress_proven_aliases(
        deduplicated
    )
    harmonic_seconds = time.monotonic() - harmonic_started
    canonical.sort(key=_ranking_quality, reverse=True)
    for rank, candidate in enumerate(canonical, start=1):
        candidate["ensemble_rank"] = rank

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    _write_gzip_json(output / "event_repeatability_families.json.gz", measured_stable)
    _write_gzip_json(output / "equivalent_event_trains.json.gz", equivalents)
    _write_gzip_json(output / "proven_integer_aliases.json.gz", aliases)
    _write_gzip_json(output / "alias_decisions.json.gz", alias_decisions)

    inherited = source["timing_seconds"]
    inherited_compute = float(
        inherited.get("reusable_pipeline_compute_total")
        or inherited.get("total")
    )
    source_audit = source["audit_files"]
    raw_scored_ledger_source = source_audit.get("scored_raw_ledger_source")
    if raw_scored_ledger_source is None:
        raw_scored_ledger_source = str(
            (
                args.source_result.parent
                / source_audit["scored_raw_ledger"]
            ).resolve()
        )
    result = {
        "schema_version": "vlora-vote-intersection-experiment-v3",
        "engine_version": "vlora-core-multigroup-vote-v3-event-residue-aliases",
        "target": source["target"],
        "tic_id": source["tic_id"],
        "candidate_cap": None,
        "all_local_peaks_retained": True,
        "segments": source["segments"],
        "base_search_blocks": blocks,
        "vote_groups": source["vote_groups"],
        "long_period_vote_groups": source.get(
            "long_period_vote_groups", []
        ),
        "vote_seed_count": int(source.get("vote_seed_count") or 1),
        "vote_seed_values": source.get("vote_seed_values", []),
        "raw_hypothesis_count": source["raw_hypothesis_count"],
        "globally_stable_family_count": len(stable),
        "event_direction_passed_count": len(direction_passed),
        "equivalent_event_train_count": len(equivalents),
        "deduplicated_family_count": len(deduplicated),
        "proven_integer_alias_count": len(aliases),
        "canonical_candidate_count": len(canonical),
        "timing_seconds": {
            "inherited_search_vote_match_validation": inherited_compute,
            "parallel_event_repeatability_including_load": round(event_seconds, 3),
            "event_train_deduplication": round(dedup_seconds, 3),
            "proven_harmonic_compression": round(harmonic_seconds, 3),
            "reusable_pipeline_compute_total": round(
                inherited_compute + event_seconds + dedup_seconds + harmonic_seconds,
                3,
            ),
            "resume_wall_total": round(time.monotonic() - started, 3),
        },
        "ranked_candidates": [_compact_candidate(row) for row in canonical],
        "audit_files": {
            "raw_scored_ledger_source": raw_scored_ledger_source,
            "direct_anchor_family_audit_source": str(
                (args.source_result.parent / source["audit_files"]["family_audit"]).resolve()
            ),
            "event_repeatability_families": "event_repeatability_families.json.gz",
            "equivalent_event_trains": "equivalent_event_trains.json.gz",
            "proven_integer_aliases": "proven_integer_aliases.json.gz",
            "alias_decisions": "alias_decisions.json.gz",
        },
        "scientific_status": (
            "upgrade experiment with all local peaks retained, multi-group voting, "
            "direct-anchor identity, per-event direction checks, and residue-proven "
            "integer-alias compression; red-noise significance and physical/pixel "
            "vetting are not yet applied"
        ),
    }
    _write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "status": "complete",
                "result": str((output / "result.json").resolve()),
                "canonical_candidates": len(canonical),
                "proven_integer_aliases": len(aliases),
                "timing_seconds": result["timing_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
