#!/usr/bin/env python3
"""Add evidence-proven non-integer p/q alias compression to vote results."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np

from multiplanet_trial.live_web import pipeline as core
from multiplanet_trial.live_web.ensemble_intersection import _write_gzip_json, _write_json
from multiplanet_trial.live_web.event_coherence import (
    annotate_event_coherence,
    event_coherence_quality,
)
from multiplanet_trial.live_web.harmonic_families import (
    arbitrate_harmonic_families,
)
from run_vlora_vote_harmonic_finalize import _compact_candidate


def _quality(candidate: dict) -> tuple:
    vote = candidate["ensemble_vote"]
    verification = candidate["holdout_verification"]
    snrs = [
        float(row.get("snr") or 0.0)
        for row in verification.get("block_measurements", [])
        if row.get("outcome") in {"strong", "weak"}
    ]
    minimum_snr = min(snrs, default=0.0)
    repeatability = verification.get("event_repeatability", {})
    support_fraction = float(repeatability.get("support_fraction") or 0.0)
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
        minimum_snr * support_fraction,
        support_fraction,
        minimum_snr,
        int(repeatability.get("supported_events") or 0),
        float(candidate.get("source_strength") or 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--event-families", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    source = json.loads(args.source_result.read_text(encoding="utf-8"))
    with gzip.open(args.event_families, "rt", encoding="utf-8") as handle:
        measured = json.load(handle)
    blocks = [list(map(int, block)) for block in source["base_search_blocks"]]
    for candidate in measured:
        annotate_event_coherence(candidate, blocks)
    direction_passed = [
        row
        for row in measured
        if row["holdout_verification"].get("passed") is True
    ]
    deduplicated, equivalents = core._collapse_duplicate_passes(
        direction_passed, branch=None
    )
    print(
        json.dumps(
            {
                "stage": "harmonic_families",
                "message": (
                    "Building integer and low-order rational harmonic families, "
                    "then selecting one global primary per resolved family"
                ),
                "deduplicated_candidates": len(deduplicated),
            }
        ),
        flush=True,
    )
    family_started = time.monotonic()
    canonical, harmonic_aliases, harmonic_relations, harmonic_families = (
        arbitrate_harmonic_families(deduplicated, quality=_quality)
    )
    family_seconds = time.monotonic() - family_started
    canonical.sort(key=_quality, reverse=True)
    for rank, candidate in enumerate(canonical, start=1):
        candidate["ensemble_rank"] = rank

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    _write_gzip_json(output / "equivalent_event_trains.json.gz", equivalents)
    integer_relations = [
        row
        for row in harmonic_relations
        if row["relation_type"] == "integer_harmonic"
    ]
    rational_relations = [
        row
        for row in harmonic_relations
        if row["relation_type"] == "low_order_rational_harmonic"
    ]
    resolved_families = [row for row in harmonic_families if row["resolved"]]
    partial_families = [
        row for row in harmonic_families if row.get("status") == "partially_resolved"
    ]
    unresolved_families = [
        row for row in harmonic_families if row.get("status") == "unresolved"
    ]
    _write_gzip_json(output / "harmonic_family_aliases.json.gz", harmonic_aliases)
    _write_gzip_json(output / "harmonic_relations.json.gz", harmonic_relations)
    _write_gzip_json(output / "harmonic_families.json.gz", harmonic_families)
    # Preserve the detailed per-event measurements already computed upstream.
    # The public ranked list stays compact, while Avril V3 can reuse this audit
    # ledger for catalogue-alias and data-gap routing without measuring the
    # same events a second time.
    _write_gzip_json(
        output / "meaningful_gate_candidates.json.gz",
        canonical,
    )

    prior_compute = float(source["timing_seconds"]["reusable_pipeline_compute_total"])
    result = {
        "schema_version": "vlora-vote-intersection-experiment-v6",
        "engine_version": "vlora-core-multigroup-vote-v6-global-harmonic-families",
        "target": source["target"],
        "tic_id": source["tic_id"],
        "candidate_cap": None,
        "all_local_peaks_retained": True,
        "segments": source["segments"],
        "base_search_blocks": source["base_search_blocks"],
        "vote_groups": source["vote_groups"],
        "long_period_vote_groups": source.get(
            "long_period_vote_groups", []
        ),
        "vote_seed_count": int(source.get("vote_seed_count") or 1),
        "vote_seed_values": source.get("vote_seed_values", []),
        "raw_hypothesis_count": source["raw_hypothesis_count"],
        "globally_stable_family_count": source["globally_stable_family_count"],
        "event_direction_passed_count": len(direction_passed),
        "equivalent_event_train_count": len(equivalents),
        "deduplicated_family_count": len(deduplicated),
        "harmonic_family_count": len(harmonic_families),
        "resolved_harmonic_family_count": len(resolved_families),
        "partially_resolved_harmonic_family_count": len(partial_families),
        "unresolved_harmonic_family_count": len(unresolved_families),
        "harmonic_family_alias_count": len(harmonic_aliases),
        "integer_harmonic_relation_count": len(integer_relations),
        "rational_harmonic_relation_count": len(rational_relations),
        "canonical_candidate_count": len(canonical),
        "timing_seconds": {
            "inherited_through_integer_aliases": prior_compute,
            "global_harmonic_family_arbitration": round(family_seconds, 3),
            "reusable_pipeline_compute_total": round(prior_compute + family_seconds, 3),
            "resume_wall_total": round(time.monotonic() - started, 3),
        },
        "ranked_candidates": [_compact_candidate(row) for row in canonical],
        "audit_files": {
            **source["audit_files"],
            "equivalent_event_trains_v4": "equivalent_event_trains.json.gz",
            "harmonic_family_aliases": "harmonic_family_aliases.json.gz",
            "harmonic_relations": "harmonic_relations.json.gz",
            "harmonic_families": "harmonic_families.json.gz",
            "meaningful_gate_candidates": "meaningful_gate_candidates.json.gz",
        },
        "scientific_status": (
            "period-candidate ranking with all local peaks retained, multi-scale "
            "group voting, robust event-coherence ranking, direct-anchor identity, "
            "and global integer plus low-order rational harmonic-family arbitration; "
            "red-noise significance and physical/pixel vetting are applied by the "
            "downstream science-finalize stage"
        ),
    }
    _write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "status": "complete",
                "result": str((output / "result.json").resolve()),
                "canonical_candidates": len(canonical),
                "harmonic_families": len(harmonic_families),
                "resolved_harmonic_families": len(resolved_families),
                "partially_resolved_harmonic_families": len(partial_families),
                "unresolved_harmonic_families": len(unresolved_families),
                "harmonic_family_aliases": len(harmonic_aliases),
                "timing_seconds": result["timing_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
