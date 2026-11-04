"""Experimental VLora multi-group temporal-intersection engine.

The experiment keeps every local BLS and event-first peak in an append-only
ledger.  Speed comes from collapsing the exact cross-group match graph before
the expensive all-sector fixed-ephemeris measurement; no local peak is removed
or capped during discovery.
"""

from __future__ import annotations

import gzip
import json
import math
import multiprocessing
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np
from astropy.timeseries import BoxLeastSquares

from . import (
    exact_bls_acceleration,
    harmonic_families,
    optimization_v1,
    optimization_v3,
    optimization_v4,
)
from . import pipeline as core
from .event_coherence import annotate_event_coherence
from .temporal_consensus import candidate_match_details, select_temporal_blocks

EPHEMERIS_REFIT_RELATIVE_HALF_WIDTH = 0.005
EPHEMERIS_REFIT_MIN_POSITIVE_GROUPS = 5
EPHEMERIS_REFIT_MIN_STRONG_GROUPS = 2
LONG_PERIOD_RESCUE_MIN_DAYS = 20.0
DEFAULT_VOTE_SEED_COUNT = 5
VOTE_SEED_STRIDE = 2_000_033
MULTI_SEED_MIN_SUPPORT_FRACTION = 0.60
REFIT_MODES = (
    "targeted",
    "adaptive",
    "native-selective",
    "native-shared",
    "native-exact",
    "batched-exact",
    "exact",
    "shared",
)
DEFAULT_REFIT_MODE = "targeted"
ADAPTIVE_EXPLOSION_MIN_COARSE_POINTS = 3_000_000
ADAPTIVE_STRICT_FALLBACK_FRACTION = 0.60
ADAPTIVE_LONG_PERIOD_PROTECTION_DAYS = 100.0
TARGETED_REFIT_MAX_FAMILIES = 128
TARGETED_REFIT_RELATIVE_PERIOD_SEPARATION = 0.005
TARGETED_REFIT_PERIOD_BANDS = (
    (0.0, 5.0, 0.15),
    (5.0, 20.0, 0.40),
    (20.0, 100.0, 0.25),
    (100.0, math.inf, 0.20),
)

_REFIT_BRANCH = None
_REFIT_BLOCKS = None
_REFIT_PREPARED_BLOCKS = None
_REFIT_BLS = None
_REFIT_BASELINE_DAYS = None
_REFIT_NATIVE_EXACT_DIAGNOSTICS = None
_REFIT_NATIVE_EXACT_PREPARED = None
_REFIT_NATIVE_EXACT_SOLUTION_CACHE = None


def _scaled_vote_thresholds(vote: Mapping) -> dict[str, int]:
    """Keep the evidence gate equally strict when vote rounds change.

    The reference configuration is nine groups across three rounds.  Scaling
    prevents a future 15-group run from passing candidates merely because the
    old absolute counts were left unchanged.
    """

    groups = max(1, int(vote.get("groups_measured") or 9))
    rounds = max(
        1,
        int(vote.get("rounds_measured") or 0),
        len(vote.get("positive_groups_by_round") or {}),
        3 if not vote.get("positive_groups_by_round") else 1,
    )
    return {
        "promotion_positive_groups": math.ceil(groups * 4 / 9),
        "promotion_positive_rounds": math.ceil(rounds * 2 / 3),
        "refit_positive_groups": math.ceil(groups * 5 / 9),
        "refit_strong_groups": math.ceil(groups * 2 / 9),
        "refit_positive_rounds": math.ceil(rounds * 2 / 3),
    }


def build_balanced_discovery_groups(
    segments: Iterable[int],
    *,
    rounds: int = 2,
    groups_per_round: int = 3,
    seed: int = 307210830,
) -> list[dict]:
    """Build deterministic stratified partitions with complete round coverage.

    Each round partitions every sector exactly once.  Every group draws from
    early, middle, and late observing epochs, so a group is not just one short
    contiguous visit.  Across ``rounds`` every sector is searched exactly that
    many times.
    """

    ordered = sorted({int(value) for value in segments})
    if len(ordered) < groups_per_round * 3:
        raise ValueError("at least three sectors per discovery group are required")
    if rounds < 1 or groups_per_round < 2:
        raise ValueError("rounds >= 1 and groups_per_round >= 2 are required")

    strata = [list(map(int, row)) for row in np.array_split(ordered, 3)]
    groups: list[dict] = []
    for round_index in range(rounds):
        buckets = [[] for _ in range(groups_per_round)]
        for stratum_index, stratum in enumerate(strata):
            rng = np.random.default_rng(int(seed) + 10_007 * round_index + 101 * stratum_index)
            permuted = list(np.asarray(stratum, dtype=int)[rng.permutation(len(stratum))])
            offset = (round_index + stratum_index) % groups_per_round
            for item_index, segment in enumerate(permuted):
                buckets[(item_index + offset) % groups_per_round].append(int(segment))
        for within_round, bucket in enumerate(buckets):
            groups.append(
                {
                    "group": len(groups) + 1,
                    "round": round_index + 1,
                    "within_round": within_round + 1,
                    "segments": sorted(bucket),
                }
            )
    return groups


def build_natural_sector_vote_groups(
    segments: Iterable[int],
    *,
    rounds: int = 5,
    preferred_groups_per_round: int = 3,
    seed: int = 307210830,
) -> tuple[list[dict], dict]:
    """Use the strongest honest vote plan allowed by the available sectors.

    Targets are never selected by sector count.  Nine or more sectors retain
    the normal three-group design; six to eight use two groups with at least
    three sectors each.  Three to five sectors use every unique
    leave-one-sector-out view once.  The latter is explicitly labelled as
    overlapping limited-data evidence, so repeated random samples cannot
    masquerade as independent votes.
    """

    ordered = sorted({int(value) for value in segments})
    if len(ordered) < 3:
        raise ValueError("at least three sectors are required")
    if len(ordered) >= preferred_groups_per_round * 3:
        groups_per_round = int(preferred_groups_per_round)
        groups = build_balanced_discovery_groups(
            ordered,
            rounds=rounds,
            groups_per_round=groups_per_round,
            seed=seed,
        )
        mode = f"stratified_disjoint_{groups_per_round}_groups_per_round"
    elif len(ordered) >= 6:
        groups_per_round = 2
        groups = build_balanced_discovery_groups(
            ordered,
            rounds=rounds,
            groups_per_round=groups_per_round,
            seed=seed,
        )
        mode = "stratified_disjoint_2_groups_per_round"
    else:
        groups_per_round = 1
        groups = [
            {
                "group": index + 1,
                "round": index + 1,
                "within_round": 1,
                "segments": [value for value in ordered if value != omitted],
                "overlapping_limited_sector_evidence": True,
                "omitted_segment": omitted,
            }
            for index, omitted in enumerate(ordered)
        ]
        mode = "unique_leave_one_sector_out_limited_data"
    return groups, {
        "mode": mode,
        "available_sector_count": len(ordered),
        "rounds_requested": int(rounds),
        "effective_rounds": len({int(row["round"]) for row in groups}),
        "groups_per_round": groups_per_round,
        "vote_group_count": len(groups),
        "groups_disjoint_within_round": len(ordered) >= 6,
        "limited_sector_evidence": len(ordered) < 6,
        "sector_count_used_for_target_selection": False,
    }


def _search_group_worker(
    branch: core.SearchBranch,
    group_index: int,
    segments: list[int],
    period_max: float,
) -> tuple[list[dict], dict]:
    """Spawn-safe worker using the unchanged uncapped Core search kernel."""

    return optimization_v3.search_block_exact(
        branch,
        block_index=group_index,
        segments=segments,
        period_max=period_max,
    )


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        parent = self.parent
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _proposal_quality(proposal: Mapping) -> tuple:
    proof = (proposal.get("match_proofs") or [{}])[0]
    return (
        bool(proof.get("cycle_connected_within_both_uncertainties")),
        float(proposal.get("source_strength") or 0.0),
        -abs(float(proof.get("period_delta_days") or np.inf)),
    )


def build_consensus_families(
    ledger: list[dict], discovery_groups: list[dict]
) -> tuple[list[dict], dict, list[dict]]:
    """Collapse exact cross-group matches before global validation.

    All ledger rows are returned with a family annotation.  Only the family
    representative is sent to the expensive full-baseline measurement, while
    every alternative remains available in the compressed audit ledger.
    """

    union_find = _UnionFind(len(ledger))
    best_edge: list[dict | None] = [None] * len(ledger)
    accepted_edges = 0
    disjoint_edges = 0
    pair_diagnostics: dict = {}
    group_segments = {int(row["group"]): set(map(int, row["segments"])) for row in discovery_groups}

    for first_index, second_index in optimization_v1._indexed_pair_candidates(
        ledger, pair_diagnostics
    ):
        first = ledger[first_index]
        second = ledger[second_index]
        proof = candidate_match_details(first, second)
        if proof is None:
            continue
        accepted_edges += 1
        union_find.union(first_index, second_index)
        proposal = core._pair_proposal(first, second, proof)
        proposal["edge_indices"] = [first_index, second_index]
        proposal["groups_disjoint"] = group_segments[int(first["source_block"])].isdisjoint(
            group_segments[int(second["source_block"])]
        )
        if proposal["groups_disjoint"]:
            disjoint_edges += 1
        for index in (first_index, second_index):
            current = best_edge[index]
            if current is None or _proposal_quality(proposal) > _proposal_quality(current):
                best_edge[index] = proposal

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(ledger)):
        members_by_root[union_find.find(index)].append(index)

    groups_by_id = {int(row["group"]): row for row in discovery_groups}
    families: list[dict] = []
    ledger_annotations: list[dict] = [{} for _ in ledger]
    matched_components = 0
    for _component_index, members in enumerate(members_by_root.values(), start=1):
        support_groups = sorted({int(ledger[index]["source_block"]) for index in members})
        if len(support_groups) < 2:
            for index in members:
                ledger_annotations[index] = {
                    "ensemble_family_id": None,
                    "ensemble_support_groups": 1,
                    "ensemble_support_rounds": 1,
                    "ensemble_disjoint_group_pairs": 0,
                }
            continue

        matched_components += 1
        family_id = f"ENS-{matched_components:05d}"
        support_rounds = sorted({int(groups_by_id[group]["round"]) for group in support_groups})
        disjoint_pairs = sum(
            group_segments[left].isdisjoint(group_segments[right])
            for left_pos, left in enumerate(support_groups)
            for right in support_groups[left_pos + 1 :]
        )
        candidate_edges = [best_edge[index] for index in members if best_edge[index]]
        representative = max(candidate_edges, key=_proposal_quality)
        raw_ids = sorted({str(ledger[index]["raw_id"]) for index in members})
        methods = sorted({str(ledger[index]["method"]) for index in members})
        source_strength = max(
            float(ledger[index].get("source_strength") or 0.0) for index in members
        )
        proposal = {
            **{key: value for key, value in representative.items() if key != "edge_indices"},
            "proposal_id": family_id,
            "source_raw_ids": raw_ids,
            "source_methods": methods,
            "source_strength": source_strength,
            # The global validation uses three disjoint chronological blocks.
            # Every stratified discovery group spans those epochs.
            "discovered_blocks": [1, 2, 3],
            "ensemble": {
                "support_groups": support_groups,
                "support_group_count": len(support_groups),
                "support_rounds": support_rounds,
                "support_round_count": len(support_rounds),
                "disjoint_group_pairs": int(disjoint_pairs),
                "raw_member_count": len(members),
                "all_local_peaks_retained": True,
            },
        }
        families.append(proposal)
        annotation = {
            "ensemble_family_id": family_id,
            "ensemble_support_groups": len(support_groups),
            "ensemble_support_rounds": len(support_rounds),
            "ensemble_disjoint_group_pairs": int(disjoint_pairs),
        }
        for index in members:
            ledger_annotations[index] = annotation

    families.sort(
        key=lambda row: (
            int(row["ensemble"]["support_group_count"]),
            int(row["ensemble"]["support_round_count"]),
            int(row["ensemble"]["disjoint_group_pairs"]),
            float(row.get("source_strength") or 0.0),
        ),
        reverse=True,
    )
    for index, family in enumerate(families, start=1):
        family["proposal_id"] = f"ENS-{index:05d}"

    pair_diagnostics.update(
        {
            "accepted_matches": accepted_edges,
            "accepted_disjoint_matches": disjoint_edges,
            "connected_components": len(members_by_root),
            "multi_group_families": len(families),
        }
    )
    return families, pair_diagnostics, ledger_annotations


def _json_safe(value):
    """Return a strict-JSON copy without changing the scientific result.

    Some optional diagnostics use NaN/Infinity to mean "not measurable".  A
    target must not be discarded merely because one such diagnostic cannot be
    represented by strict JSON, so non-finite scalars are persisted as null.
    Finite values, candidate ordering, and all in-memory calculations are left
    unchanged.
    """

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_gzip_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def run_ensemble_intersection(
    lightcurve_paths: Iterable[Path],
    *,
    output_dir: Path,
    period_max: float = 400.0,
    rounds: int = 2,
    groups_per_round: int = 3,
    search_workers: int = 3,
    seed: int = 307210830,
    progress=None,
) -> dict:
    """Run the upgrade experiment on already-downloaded SPOC light curves."""

    started = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(path) for path in lightcurve_paths]
    metadata = [core._segment_metadata(path) for path in paths]
    paired = sorted(zip(metadata, paths, strict=False), key=lambda row: row[0]["segment"])
    metadata = [row[0] for row in paired]
    paths = [row[1] for row in paired]
    segments = sorted({int(row["segment"]) for row in metadata})
    branch, load_diagnostics = core._load_pdcsap_branch(paths)
    groups = build_balanced_discovery_groups(
        segments,
        rounds=rounds,
        groups_per_round=groups_per_round,
        seed=seed,
    )
    _write_json(output_dir / "discovery_groups.json", groups)

    def emit(stage: str, message: str, metrics: dict) -> None:
        if progress:
            progress(stage, message, metrics)

    emit(
        "searching",
        f"Starting {len(groups)} uncapped stratified discovery groups",
        {"groups": groups, "all_local_peaks_retained": True},
    )
    completed: list[tuple[int, list[dict], dict]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max(1, min(int(search_workers), len(groups))),
        mp_context=context,
    ) as pool:
        futures = {
            pool.submit(
                _search_group_worker,
                branch,
                int(group["group"]) - 1,
                list(group["segments"]),
                float(period_max),
            ): group
            for group in groups
        }
        for future in as_completed(futures):
            group = futures[future]
            rows, diagnostics = future.result()
            completed.append((int(group["group"]), rows, diagnostics))
            emit(
                "searching",
                f"Discovery group {group['group']}/{len(groups)} complete",
                {
                    "group": group["group"],
                    "round": group["round"],
                    "segments": group["segments"],
                    "raw_hypotheses": len(rows),
                },
            )

    ledger: list[dict] = []
    search_diagnostics: list[dict] = []
    for _group, rows, diagnostics in sorted(completed):
        ledger.extend(rows)
        search_diagnostics.append(diagnostics)
    search_seconds = time.monotonic() - started
    emit(
        "matching",
        "Building exact cross-group period+phase intersection graph",
        {"raw_hypotheses": len(ledger)},
    )
    families, match_diagnostics, annotations = build_consensus_families(ledger, groups)
    matching_seconds = time.monotonic() - started - search_seconds

    validation_blocks = select_temporal_blocks(segments)
    optimization_v4.install()
    emit(
        "validating",
        "Vectorized full-sector fixed-ephemeris measurement of consensus families",
        {
            "families": len(families),
            "validation_blocks": validation_blocks,
        },
    )
    validated = optimization_v4.validate_proposals_vectorized(families, validation_blocks, branch)
    stable = [row for row in validated if row["holdout_verification"].get("passed") is True]
    for row in stable:
        measurements = row["holdout_verification"]["block_measurements"]
        positive_snrs = [
            float(item.get("snr") or 0.0)
            for item in measurements
            if item.get("outcome") in {"strong", "weak"}
        ]
        ensemble = row["ensemble"]
        row["ensemble_score"] = round(
            100.0 * int(ensemble["support_group_count"])
            + 25.0 * int(ensemble["support_round_count"])
            + 5.0 * int(ensemble["disjoint_group_pairs"])
            + min(25.0, min(positive_snrs) if positive_snrs else 0.0),
            6,
        )
    stable.sort(
        key=lambda row: (
            float(row["ensemble_score"]),
            float(row.get("source_strength") or 0.0),
        ),
        reverse=True,
    )
    validation_seconds = time.monotonic() - started - search_seconds - matching_seconds

    annotated_ledger = [
        {**row, "ensemble_consensus": annotations[index]} for index, row in enumerate(ledger)
    ]
    _write_gzip_json(output_dir / "raw_candidate_ledger.json.gz", annotated_ledger)
    _write_gzip_json(output_dir / "validated_families.json.gz", validated)
    result = {
        "schema_version": "vlora-ensemble-intersection-experiment-v1",
        "engine_version": "vlora-core-ensemble-intersection-v1",
        "target": metadata[0].get("object") or "TIC 307210830",
        "tic_id": metadata[0].get("tic_id"),
        "period_max_days": float(period_max),
        "candidate_cap": None,
        "all_local_peaks_retained": True,
        "segments": segments,
        "discovery_groups": groups,
        "validation_blocks": validation_blocks,
        "raw_hypothesis_count": len(ledger),
        "consensus_family_count": len(families),
        "stable_family_count": len(stable),
        "match_diagnostics": match_diagnostics,
        "search_diagnostics": search_diagnostics,
        "load_diagnostics": load_diagnostics,
        "timing_seconds": {
            "search": round(search_seconds, 3),
            "matching": round(matching_seconds, 3),
            "vectorized_global_validation": round(validation_seconds, 3),
            "total": round(time.monotonic() - started, 3),
        },
        "stable_families": stable,
        "audit_files": {
            "raw_candidate_ledger": "raw_candidate_ledger.json.gz",
            "validated_families": "validated_families.json.gz",
        },
        "scientific_status": (
            "upgrade experiment; locked global validation complete, red-noise "
            "significance and physical/pixel vetting intentionally not yet run"
        ),
    }
    _write_json(output_dir / "result.json", result)
    emit(
        "complete",
        "Ensemble intersection experiment complete",
        {
            "raw_hypotheses": len(ledger),
            "consensus_families": len(families),
            "stable_families": len(stable),
            "timing_seconds": result["timing_seconds"],
        },
    )
    return result


def score_ledger_on_vote_groups(
    ledger: list[dict],
    branch: core.SearchBranch,
    vote_groups: list[dict],
    *,
    batch_size: int = 4_096,
    progress=None,
) -> list[dict]:
    """Measure every local peak on every vote group in compiled batches."""

    prepared = [core._subset_branch(branch, list(group["segments"])) for group in vote_groups]
    output: list[dict] = []
    total = len(ledger)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        batch = ledger[start:stop]
        measurements_by_candidate: list[list[dict]] = [[] for _ in batch]
        for group, local in zip(vote_groups, prepared, strict=False):
            metric_rows = optimization_v4._measure_fixed_ephemerides_exact(
                local.time, local.flux, batch
            )
            for index, metrics in enumerate(metric_rows):
                metrics = dict(metrics)
                measurements_by_candidate[index].append(
                    {
                        "group": int(group["group"]),
                        "round": int(group["round"]),
                        "segments": list(group["segments"]),
                        "outcome": core._metric_outcome(metrics, independently_discovered=False),
                        **metrics,
                    }
                )
        for row, measurements in zip(batch, measurements_by_candidate, strict=False):
            positive = [item for item in measurements if item["outcome"] in {"strong", "weak"}]
            strong = [item for item in measurements if item["outcome"] == "strong"]
            contradictions = [item for item in measurements if item["outcome"] == "contradiction"]
            positive_rounds = sorted({int(item["round"]) for item in positive})
            positive_by_round = {
                str(round_index): sum(int(item["round"]) == round_index for item in positive)
                for round_index in sorted({int(group["round"]) for group in vote_groups})
            }
            positive_snrs = [float(item.get("snr") or 0.0) for item in positive]
            vote = {
                "groups_measured": len(vote_groups),
                "rounds_measured": len({int(group["round"]) for group in vote_groups}),
                "positive_groups": len(positive),
                "strong_groups": len(strong),
                "contradicting_groups": len(contradictions),
                "untestable_groups": sum(item["outcome"] == "untestable" for item in measurements),
                "positive_rounds": positive_rounds,
                "positive_round_count": len(positive_rounds),
                "positive_groups_by_round": positive_by_round,
                "weakest_positive_snr": (min(positive_snrs) if positive_snrs else None),
                "combined_positive_snr": float(
                    np.sqrt(sum(value * value for value in positive_snrs))
                ),
                "measurements": measurements,
            }
            output.append({**row, "ensemble_vote": vote})
        if progress:
            progress(
                "voting",
                f"Vectorized multi-group scoring {stop}/{total}",
                {
                    "hypotheses_completed": stop,
                    "hypotheses_total": total,
                    "vote_groups": len(vote_groups),
                    "batch_size": batch_size,
                },
            )
    return output


def attach_vote_channel(
    ledger: list[dict],
    branch: core.SearchBranch,
    vote_groups: list[dict],
    *,
    field: str,
    progress=None,
    scorer: Callable | None = None,
) -> list[dict]:
    """Score a second vote geometry without overwriting the main channel."""

    if not ledger:
        return []
    score_function = scorer or score_ledger_on_vote_groups
    scored = score_function(ledger, branch, vote_groups, progress=progress)
    return [
        {
            **source,
            field: measured["ensemble_vote"],
        }
        for source, measured in zip(ledger, scored, strict=False)
    ]


def _vote_quality(candidate: Mapping) -> tuple:
    vote = candidate["ensemble_vote"]
    weakest = vote.get("weakest_positive_snr")
    ordinary = (
        int(vote["positive_round_count"]),
        int(vote["positive_groups"]),
        int(vote["strong_groups"]),
        -int(vote["contradicting_groups"]),
        float(weakest) if weakest is not None else -np.inf,
        float(candidate.get("source_strength") or 0.0),
    )
    consensus = candidate.get("vote_seed_consensus") or {}
    if int(consensus.get("seed_count") or 1) <= 1:
        return ordinary
    return (
        float(consensus.get("promotion_support_rate") or 0.0),
        float(consensus.get("refit_support_rate") or 0.0),
        -float(consensus.get("mean_rank") or np.inf),
        -float(consensus.get("rank_variance") or np.inf),
        *ordinary,
    )


def deterministic_vote_seeds(seed: int, count: int) -> list[int]:
    """Return reproducible, non-overlapping seeds for sector resampling."""

    if count < 1:
        raise ValueError("vote seed count must be at least one")
    return [int(seed) + VOTE_SEED_STRIDE * index for index in range(int(count))]


def _ordinary_vote_support(vote: Mapping, *, refit: bool) -> bool:
    thresholds = _scaled_vote_thresholds(vote)
    return bool(
        int(vote.get("positive_groups") or 0)
        >= thresholds["refit_positive_groups" if refit else "promotion_positive_groups"]
        and (not refit or int(vote.get("strong_groups") or 0) >= thresholds["refit_strong_groups"])
        and int(vote.get("positive_round_count") or 0)
        >= thresholds["refit_positive_rounds" if refit else "promotion_positive_rounds"]
    )


def _long_vote_support(vote: Mapping, period_days: float) -> bool:
    if not vote or float(period_days) < LONG_PERIOD_RESCUE_MIN_DAYS:
        return False
    thresholds = _scaled_vote_thresholds(vote)
    return bool(
        int(vote.get("positive_groups") or 0) >= thresholds["promotion_positive_groups"]
        and int(vote.get("strong_groups") or 0) >= thresholds["refit_strong_groups"]
        and int(vote.get("positive_round_count") or 0) >= thresholds["promotion_positive_rounds"]
    )


def _compact_vote_counts(vote: Mapping) -> dict:
    return {
        key: vote.get(key)
        for key in (
            "groups_measured",
            "rounds_measured",
            "positive_groups",
            "strong_groups",
            "contradicting_groups",
            "untestable_groups",
            "positive_round_count",
            "weakest_positive_snr",
            "combined_positive_snr",
        )
    }


def score_ledger_on_vote_seed_ensemble(
    ledger: list[dict],
    branch: core.SearchBranch,
    seed_geometries: list[dict],
    *,
    scorer: Callable | None = None,
    progress=None,
) -> list[dict]:
    """Score one search ledger on several deterministic sector partitions.

    The expensive local-period search is not repeated.  Each seed only runs
    the vectorized fixed-ephemeris vote measurements.  Full measurements for
    the first seed remain in the ordinary audit fields; additional seeds are
    stored as compact counts plus ranks so large ledgers do not balloon.
    """

    if not ledger:
        return []
    score_function = scorer or score_ledger_on_vote_groups
    seed_rows: list[tuple[dict, list[dict], list[int]]] = []
    long_indices = [
        index
        for index, row in enumerate(ledger)
        if float(row.get("period_days") or 0.0) >= LONG_PERIOD_RESCUE_MIN_DAYS
    ]
    for seed_index, geometry in enumerate(seed_geometries, start=1):
        measured = score_function(
            ledger,
            branch,
            geometry["vote_groups"],
            progress=None,
        )
        if long_indices:
            long_rows = attach_vote_channel(
                [measured[index] for index in long_indices],
                branch,
                geometry["long_period_vote_groups"],
                field="long_period_vote",
                progress=None,
                scorer=score_function,
            )
            for index, row in zip(long_indices, long_rows, strict=False):
                measured[index] = row
        order = sorted(
            range(len(measured)),
            key=lambda index: _vote_quality(measured[index]),
            reverse=True,
        )
        ranks = [0] * len(measured)
        for rank, index in enumerate(order, start=1):
            ranks[index] = rank
        seed_rows.append((geometry, measured, ranks))
        if progress:
            progress(
                "multi_seed_voting",
                f"Deterministic vote seed {seed_index}/{len(seed_geometries)} complete",
                {
                    "seed_index": seed_index,
                    "seed_count": len(seed_geometries),
                    "seed": int(geometry["seed"]),
                    "hypotheses": len(ledger),
                },
            )

    seed_count = len(seed_rows)
    required = max(
        1,
        math.ceil(seed_count * MULTI_SEED_MIN_SUPPORT_FRACTION),
    )
    population = len(ledger)
    output: list[dict] = []
    for index in range(population):
        per_seed: list[dict] = []
        ranks: list[int] = []
        promotion_count = 0
        refit_count = 0
        ordinary_count = 0
        long_count = 0
        for geometry, measured, seed_ranks in seed_rows:
            candidate = measured[index]
            vote = candidate["ensemble_vote"]
            long_vote = candidate.get("long_period_vote") or {}
            period = float(candidate.get("period_days") or 0.0)
            ordinary = _ordinary_vote_support(vote, refit=False)
            refit = _ordinary_vote_support(vote, refit=True)
            long_support = _long_vote_support(long_vote, period)
            promotion = ordinary or long_support
            refit = refit or long_support
            rank = int(seed_ranks[index])
            percentile = 1.0 if population == 1 else 1.0 - (rank - 1) / (population - 1)
            ranks.append(rank)
            promotion_count += int(promotion)
            refit_count += int(refit)
            ordinary_count += int(ordinary)
            long_count += int(long_support)
            per_seed.append(
                {
                    "seed": int(geometry["seed"]),
                    "rank": rank,
                    "rank_percentile": percentile,
                    "promotion_supported": promotion,
                    "refit_supported": refit,
                    "ordinary_vote_supported": ordinary,
                    "long_period_vote_supported": long_support,
                    "vote": _compact_vote_counts(vote),
                    "long_period_vote": (_compact_vote_counts(long_vote) if long_vote else None),
                }
            )
        rank_array = np.asarray(ranks, dtype=np.float64)
        mean_rank = float(np.mean(rank_array))
        rank_variance = float(np.var(rank_array))
        rank_stddev = float(np.sqrt(rank_variance))
        mean_percentile = 1.0 if population == 1 else 1.0 - (mean_rank - 1.0) / (population - 1.0)
        normalized_stddev = rank_stddev / max(1.0, population - 1.0)
        promotion_rate = promotion_count / seed_count
        refit_rate = refit_count / seed_count
        stability_score = (
            100.0 * promotion_rate
            + 50.0 * refit_rate
            + 25.0 * mean_percentile
            - 25.0 * normalized_stddev
        )
        base = dict(seed_rows[0][1][index])
        base["vote_seed_consensus"] = {
            "schema_version": "vlora-deterministic-vote-seeds-v1",
            "seed_count": seed_count,
            "seeds": [int(row[0]["seed"]) for row in seed_rows],
            "minimum_support_fraction": MULTI_SEED_MIN_SUPPORT_FRACTION,
            "minimum_supported_seeds": required,
            "promotion_supported_seeds": promotion_count,
            "promotion_support_rate": promotion_rate,
            "refit_supported_seeds": refit_count,
            "refit_support_rate": refit_rate,
            "ordinary_vote_supported_seeds": ordinary_count,
            "long_period_vote_supported_seeds": long_count,
            "ranking_population": population,
            "mean_rank": mean_rank,
            "rank_variance": rank_variance,
            "rank_stddev": rank_stddev,
            "mean_rank_percentile": mean_percentile,
            "normalized_rank_stddev": normalized_stddev,
            "stability_score": stability_score,
            "per_seed": per_seed,
        }
        output.append(base)
    return output


def rescore_changed_ephemerides_with_cached_population(
    ledger: list[dict],
    changed_indices: list[int],
    branch: core.SearchBranch,
    seed_geometries: list[dict],
    *,
    scorer: Callable | None = None,
    progress=None,
) -> tuple[list[dict], dict]:
    """Re-measure changed clocks and rank them in the complete population.

    Every row already carries compact per-seed vote measurements from the
    first discovery vote. A targeted refit changes only a small set of
    ephemerides, so only those rows need another expensive fixed-ephemeris
    measurement. The cached measurements for unchanged rows and the fresh
    measurements for changed rows are then ranked together. This reproduces
    the full-population percentile contract without re-reading the same light
    curve windows for every unchanged or unresolved audit row.
    """

    if not ledger:
        return [], {
            "remeasured_families": 0,
            "ranking_population": 0,
            "cached_measurement_families": 0,
            "candidate_rows_deleted": False,
        }
    changed = sorted(set(int(index) for index in changed_indices))
    if any(index < 0 or index >= len(ledger) for index in changed):
        raise IndexError("changed vote-family index is out of range")
    changed_set = set(changed)
    expected_seeds = [int(geometry["seed"]) for geometry in seed_geometries]
    for index, row in enumerate(ledger):
        if index in changed_set:
            continue
        consensus = row.get("vote_seed_consensus") or {}
        cached = list(consensus.get("per_seed") or [])
        cached_seeds = [int(item.get("seed")) for item in cached]
        if cached_seeds != expected_seeds:
            raise ValueError("unchanged family lacks the complete cached vote-seed record")

    score_function = scorer or score_ledger_on_vote_groups
    base_rows = [dict(row) for row in ledger]
    per_candidate: list[list[dict]] = [[] for _row in ledger]
    active_rows = [ledger[index] for index in changed]
    active_long_positions = [
        position
        for position, row in enumerate(active_rows)
        if float(row.get("period_days") or 0.0) >= LONG_PERIOD_RESCUE_MIN_DAYS
    ]

    for seed_index, geometry in enumerate(seed_geometries):
        measured_active = score_function(
            active_rows,
            branch,
            geometry["vote_groups"],
            progress=None,
        )
        if active_long_positions:
            long_rows = attach_vote_channel(
                [measured_active[position] for position in active_long_positions],
                branch,
                geometry["long_period_vote_groups"],
                field="long_period_vote",
                progress=None,
                scorer=score_function,
            )
            for position, row in zip(
                active_long_positions,
                long_rows,
                strict=False,
            ):
                measured_active[position] = row
        measured_by_index = {
            index: measured for index, measured in zip(changed, measured_active, strict=True)
        }
        ranking_rows: list[dict] = []
        for index, row in enumerate(ledger):
            if index in measured_by_index:
                candidate = measured_by_index[index]
            else:
                cached = row["vote_seed_consensus"]["per_seed"][seed_index]
                candidate = {
                    **row,
                    "ensemble_vote": dict(cached["vote"]),
                }
                if cached.get("long_period_vote") is not None:
                    candidate["long_period_vote"] = dict(cached["long_period_vote"])
            ranking_rows.append(candidate)
        order = sorted(
            range(len(ranking_rows)),
            key=lambda index: _vote_quality(ranking_rows[index]),
            reverse=True,
        )
        ranks = [0] * len(ranking_rows)
        for rank, index in enumerate(order, start=1):
            ranks[index] = rank
        population = len(ranking_rows)
        for index, candidate in enumerate(ranking_rows):
            vote = candidate["ensemble_vote"]
            long_vote = candidate.get("long_period_vote") or {}
            period = float(candidate.get("period_days") or 0.0)
            ordinary = _ordinary_vote_support(vote, refit=False)
            refit = _ordinary_vote_support(vote, refit=True)
            long_support = _long_vote_support(long_vote, period)
            rank = int(ranks[index])
            percentile = 1.0 if population == 1 else 1.0 - (rank - 1) / (population - 1)
            per_candidate[index].append(
                {
                    "seed": int(geometry["seed"]),
                    "rank": rank,
                    "rank_percentile": percentile,
                    "promotion_supported": ordinary or long_support,
                    "refit_supported": refit or long_support,
                    "ordinary_vote_supported": ordinary,
                    "long_period_vote_supported": long_support,
                    "vote": _compact_vote_counts(vote),
                    "long_period_vote": (_compact_vote_counts(long_vote) if long_vote else None),
                }
            )
        if seed_index == 0:
            for index, candidate in measured_by_index.items():
                base_rows[index] = dict(candidate)
        if progress:
            progress(
                "multi_seed_voting",
                (f"Changed-ephemeris vote seed {seed_index + 1}/{len(seed_geometries)} complete"),
                {
                    "seed_index": seed_index + 1,
                    "seed_count": len(seed_geometries),
                    "seed": int(geometry["seed"]),
                    "families_remeasured": len(changed),
                    "ranking_population": len(ledger),
                },
            )

    seed_count = len(seed_geometries)
    required = max(1, math.ceil(seed_count * MULTI_SEED_MIN_SUPPORT_FRACTION))
    population = len(ledger)
    output: list[dict] = []
    for base, per_seed in zip(base_rows, per_candidate, strict=True):
        ranks = np.asarray(
            [int(item["rank"]) for item in per_seed],
            dtype=np.float64,
        )
        mean_rank = float(np.mean(ranks))
        rank_variance = float(np.var(ranks))
        rank_stddev = float(np.sqrt(rank_variance))
        mean_percentile = 1.0 if population == 1 else 1.0 - (mean_rank - 1.0) / (population - 1.0)
        normalized_stddev = rank_stddev / max(1.0, population - 1.0)
        promotion_count = sum(bool(item["promotion_supported"]) for item in per_seed)
        refit_count = sum(bool(item["refit_supported"]) for item in per_seed)
        promotion_rate = promotion_count / seed_count
        refit_rate = refit_count / seed_count
        result = dict(base)
        result["vote_seed_consensus"] = {
            "schema_version": "vlora-deterministic-vote-seeds-v1",
            "seed_count": seed_count,
            "seeds": expected_seeds,
            "minimum_support_fraction": MULTI_SEED_MIN_SUPPORT_FRACTION,
            "minimum_supported_seeds": required,
            "promotion_supported_seeds": promotion_count,
            "promotion_support_rate": promotion_rate,
            "refit_supported_seeds": refit_count,
            "refit_support_rate": refit_rate,
            "ordinary_vote_supported_seeds": sum(
                bool(item["ordinary_vote_supported"]) for item in per_seed
            ),
            "long_period_vote_supported_seeds": sum(
                bool(item["long_period_vote_supported"]) for item in per_seed
            ),
            "ranking_population": population,
            "mean_rank": mean_rank,
            "rank_variance": rank_variance,
            "rank_stddev": rank_stddev,
            "mean_rank_percentile": mean_percentile,
            "normalized_rank_stddev": normalized_stddev,
            "stability_score": (
                100.0 * promotion_rate
                + 50.0 * refit_rate
                + 25.0 * mean_percentile
                - 25.0 * normalized_stddev
            ),
            "per_seed": per_seed,
        }
        output.append(result)
    return output, {
        "remeasured_families": len(changed),
        "ranking_population": len(ledger),
        "cached_measurement_families": len(ledger) - len(changed),
        "cached_measurements_reused": ((len(ledger) - len(changed)) * len(seed_geometries)),
        "candidate_rows_deleted": False,
    }


FAMILY_MODES = ("core", "sparse-sector-pro")


def _pro_neighbour_merge_is_safe(
    first: Mapping,
    second: Mapping,
    *,
    baseline_days: float,
) -> bool:
    """Test whether nearby hypotheses stay aligned over the full baseline.

    Core uses the established local period+phase relation. Sparse-sector
    Avril Pro is deliberately more conservative: a neighbour edge is retained
    only when the clocks drift by no more than half of the shorter transit
    duration over the complete observing baseline. A rejected edge deletes
    nothing; it gives each hypothesis its own validation and exact refit.
    """

    first_period = float(first["period_days"])
    second_period = float(second["period_days"])
    if first_period <= 0.0 or second_period <= 0.0:
        return False
    drift_days = (
        float(baseline_days) * abs(first_period - second_period) / max(first_period, second_period)
    )
    duration_days = min(float(first["duration_hours"]), float(second["duration_hours"])) / 24.0
    return drift_days <= 0.5 * duration_days


def build_vote_consensus_families(
    scored_ledger: list[dict],
    *,
    family_mode: str = "core",
    baseline_days: float | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Form direct-anchor families after every local peak has received votes.

    Candidate identity is deliberately not transitive.  If A matches B and B
    matches C, but A does not match C, the three hypotheses must not collapse
    into one family.  A union-find connected component violates that rule and
    can bridge a real period into a nearby harmonic through a chain of small
    local-period steps.  Each family below is therefore a star: every member
    has passed the unchanged exact period+phase predicate directly against its
    anchor.  Every ledger row belongs to exactly one anchored family.
    """

    if family_mode not in FAMILY_MODES:
        raise ValueError(f"unsupported candidate family mode: {family_mode}")
    if family_mode == "sparse-sector-pro" and (
        baseline_days is None
        or not np.isfinite(float(baseline_days))
        or float(baseline_days) <= 0.0
    ):
        raise ValueError("sparse-sector-pro requires a positive full baseline")

    diagnostics: dict = {}
    accepted_edges = 0
    pro_split_edges = 0
    neighbours: list[list[int]] = [[] for _ in scored_ledger]
    for first_index, second_index in optimization_v1._indexed_pair_candidates(
        scored_ledger, diagnostics
    ):
        proof = candidate_match_details(scored_ledger[first_index], scored_ledger[second_index])
        if proof is None:
            continue
        if family_mode == "sparse-sector-pro" and not _pro_neighbour_merge_is_safe(
            scored_ledger[first_index],
            scored_ledger[second_index],
            baseline_days=float(baseline_days),
        ):
            pro_split_edges += 1
            continue
        accepted_edges += 1
        neighbours[first_index].append(second_index)
        neighbours[second_index].append(first_index)

    # Strong, broadly reproduced peaks become anchors first.  Ties retain the
    # append-only ledger order, making the partition deterministic.
    anchor_order = sorted(
        range(len(scored_ledger)),
        key=lambda index: _vote_quality(scored_ledger[index]),
        reverse=True,
    )
    unassigned = np.ones(len(scored_ledger), dtype=np.bool_)
    anchored_members: list[tuple[int, list[int]]] = []
    for anchor_index in anchor_order:
        if not unassigned[anchor_index]:
            continue
        members = [anchor_index]
        members.extend(index for index in neighbours[anchor_index] if unassigned[index])
        # Preserve deterministic ledger order inside the audit family.
        members = sorted(set(members))
        unassigned[members] = False
        anchored_members.append((anchor_index, members))

    promoted: list[dict] = []
    audit: list[dict] = []
    for family_index, (anchor_index, members) in enumerate(anchored_members, start=1):
        family_id = f"VOTE-{family_index:05d}"
        source_blocks = sorted({int(scored_ledger[index]["source_block"]) for index in members})
        representative = scored_ledger[anchor_index]
        vote = representative["ensemble_vote"]
        thresholds = _scaled_vote_thresholds(vote)
        member_periods = [float(scored_ledger[index]["period_days"]) for index in members]
        member_durations = [float(scored_ledger[index]["duration_hours"]) for index in members]
        member_ephemerides = [
            {
                "raw_id": str(scored_ledger[index]["raw_id"]),
                "source_block": int(scored_ledger[index]["source_block"]),
                "source_segments": [
                    int(value) for value in scored_ledger[index].get("source_segments", [])
                ],
                "period_days": float(scored_ledger[index]["period_days"]),
                "epoch_btjd": float(scored_ledger[index]["epoch_btjd"]),
                "duration_hours": float(scored_ledger[index]["duration_hours"]),
                "source_strength": float(scored_ledger[index].get("source_strength") or 0.0),
                "robust_sde": float(scored_ledger[index].get("robust_sde") or 0.0),
            }
            for index in members
        ]
        representative_period = float(representative["period_days"])
        relative_lower = representative_period * (1.0 - EPHEMERIS_REFIT_RELATIVE_HALF_WIDTH)
        relative_upper = representative_period * (1.0 + EPHEMERIS_REFIT_RELATIVE_HALF_WIDTH)
        refit_bounds = [
            min(relative_lower, min(member_periods)),
            max(relative_upper, max(member_periods)),
        ]
        # This is an evidence rule rather than a count cap: a family advances
        # if independently discovered in two base blocks, or if a single-block
        # small peak reproduces in at least four resampled groups over at least
        # two independently randomized rounds.
        promotion_reasons: list[str] = []
        science_review_reasons: list[str] = []
        if len(source_blocks) >= 2:
            promotion_reasons.append("independent_base_block_intersection")
        seed_consensus = representative.get("vote_seed_consensus") or {}
        multi_seed = int(seed_consensus.get("seed_count") or 1) > 1
        minimum_supported_seeds = int(seed_consensus.get("minimum_supported_seeds") or 1)
        if multi_seed:
            ordinary_supported_seeds = int(seed_consensus.get("ordinary_vote_supported_seeds") or 0)
            if ordinary_supported_seeds >= minimum_supported_seeds:
                promotion_reasons.append("deterministic_multi_seed_forced_ephemeris_recurrence")
            elif ordinary_supported_seeds:
                science_review_reasons.append("deterministic_vote_seed_carry_for_science_review")
        elif (
            int(vote["positive_groups"]) >= thresholds["promotion_positive_groups"]
            and int(vote["positive_round_count"]) >= thresholds["promotion_positive_rounds"]
        ):
            promotion_reasons.append("multi_group_forced_ephemeris_recurrence")
        long_vote = representative.get("long_period_vote") or {}
        if multi_seed:
            long_supported_seeds = int(seed_consensus.get("long_period_vote_supported_seeds") or 0)
            if long_supported_seeds >= minimum_supported_seeds:
                promotion_reasons.append("deterministic_multi_seed_long_period_recurrence_rescue")
            elif long_supported_seeds:
                science_review_reasons.append(
                    "deterministic_long_period_seed_carry_for_science_review"
                )
        elif long_vote and representative_period >= LONG_PERIOD_RESCUE_MIN_DAYS:
            long_thresholds = _scaled_vote_thresholds(long_vote)
            if (
                int(long_vote.get("positive_groups") or 0)
                >= long_thresholds["promotion_positive_groups"]
                and int(long_vote.get("strong_groups") or 0)
                >= long_thresholds["refit_strong_groups"]
                and int(long_vote.get("positive_round_count") or 0)
                >= long_thresholds["promotion_positive_rounds"]
            ):
                promotion_reasons.append("large_group_long_period_recurrence_rescue")
        promoted_to_global = bool(promotion_reasons)
        family_audit = {
            "family_id": family_id,
            "member_count": len(members),
            "source_blocks": source_blocks,
            "representative_raw_id": representative["raw_id"],
            "representative_vote": vote,
            "representative_long_period_vote": long_vote or None,
            "representative_vote_seed_consensus": seed_consensus or None,
            "scaled_vote_thresholds": thresholds,
            "promoted_to_global_validation": promoted_to_global,
            "promotion_reasons": promotion_reasons,
            "science_review_reasons": science_review_reasons,
            "ephemeris_refit_period_bounds_days": refit_bounds,
            "ephemeris_refit_duration_hours": float(np.median(member_durations)),
            "member_ephemerides": member_ephemerides,
            "all_member_raw_ids": [str(scored_ledger[index]["raw_id"]) for index in members],
        }
        audit.append(family_audit)
        if not promoted_to_global:
            continue
        promoted.append(
            {
                **representative,
                "proposal_id": family_id,
                "source_raw_ids": family_audit["all_member_raw_ids"],
                "source_methods": sorted(
                    {str(scored_ledger[index]["method"]) for index in members}
                ),
                # Preserve the distinction between active discovery and a
                # forced locked-ephemeris check.  In particular, a signal
                # found in two blocks must be allowed to test the third block
                # without falsely claiming it was independently discovered
                # there.
                "discovered_blocks": source_blocks,
                "balanced_family_period_bounds_days": refit_bounds,
                "balanced_family_duration_hours": float(np.median(member_durations)),
                # Compact local-block solutions are the event-time anchors for
                # the matrix ephemeris prefit.  They contain no catalogue or
                # truth information and every original ledger row remains in
                # the append-only compressed audit.
                "member_ephemerides": member_ephemerides,
                "ensemble": {
                    "member_count": len(members),
                    "source_blocks": source_blocks,
                    "promotion_reasons": promotion_reasons,
                    "all_local_peaks_retained": True,
                    "vote_seed_consensus": seed_consensus or None,
                },
            }
        )

    promoted.sort(key=_vote_quality, reverse=True)
    diagnostics.update(
        {
            "accepted_matches": accepted_edges,
            "family_method": "direct_anchor_no_transitive_closure",
            "candidate_family_mode": family_mode,
            "full_baseline_days": (float(baseline_days) if baseline_days is not None else None),
            "pro_neighbour_edges_split_for_full_baseline_drift": pro_split_edges,
            "candidates_deleted_by_family_mode": 0,
            "anchored_families": len(anchored_members),
            "promoted_families": len(promoted),
            "unpromoted_families": len(anchored_members) - len(promoted),
        }
    )
    return promoted, audit, diagnostics


def ephemeris_refit_eligible(candidate: Mapping) -> bool:
    """Return whether a failed vote family has enough evidence for a refit.

    A cross-block family is always eligible.  A single-block seed is eligible
    only when at least two randomized groups measured it strongly and at least
    five groups across two rounds measured positive depth.  This is an
    evidence gate, not a rank or candidate-count cap.
    """

    verification = candidate.get("holdout_verification") or {}
    if verification.get("passed") is True:
        return False
    discovered_blocks = {int(value) for value in candidate.get("discovered_blocks", [])}
    if len(discovered_blocks) >= 2:
        return True
    seed_consensus = candidate.get("vote_seed_consensus") or {}
    if int(seed_consensus.get("seed_count") or 1) > 1:
        # Under-supported rows remain in the append-only family audit, but the
        # expensive global refit requires the declared deterministic consensus
        # threshold.  Otherwise a single favourable partition can multiply the
        # refit queue by an order of magnitude while pretending that 1/5 is the
        # configured 60% support rule.
        return bool(
            int(seed_consensus.get("refit_supported_seeds") or 0)
            >= int(seed_consensus.get("minimum_supported_seeds") or 1)
        )
    vote = candidate.get("ensemble_vote") or {}
    thresholds = _scaled_vote_thresholds(vote)
    ordinary_vote_eligible = (
        int(vote.get("positive_groups") or 0) >= thresholds["refit_positive_groups"]
        and int(vote.get("strong_groups") or 0) >= thresholds["refit_strong_groups"]
        and int(vote.get("positive_round_count") or 0) >= thresholds["refit_positive_rounds"]
    )
    long_vote = candidate.get("long_period_vote") or {}
    long_vote_eligible = False
    if long_vote and float(candidate.get("period_days") or 0.0) >= LONG_PERIOD_RESCUE_MIN_DAYS:
        long_thresholds = _scaled_vote_thresholds(long_vote)
        long_vote_eligible = (
            int(long_vote.get("positive_groups") or 0)
            >= long_thresholds["promotion_positive_groups"]
            and int(long_vote.get("strong_groups") or 0) >= long_thresholds["refit_strong_groups"]
            and int(long_vote.get("positive_round_count") or 0)
            >= long_thresholds["promotion_positive_rounds"]
        )
    return ordinary_vote_eligible or long_vote_eligible


def _initialize_refit_worker(
    branch: core.SearchBranch,
    blocks: list[list[int]],
    native_exact_threads: int | None = None,
) -> None:
    global _REFIT_BRANCH, _REFIT_BLOCKS, _REFIT_PREPARED_BLOCKS
    global _REFIT_BLS, _REFIT_BASELINE_DAYS, _REFIT_NATIVE_EXACT_DIAGNOSTICS
    global _REFIT_NATIVE_EXACT_PREPARED, _REFIT_NATIVE_EXACT_SOLUTION_CACHE
    _REFIT_BRANCH = branch
    _REFIT_BLOCKS = blocks
    _REFIT_PREPARED_BLOCKS = optimization_v4._prepare_blocks(blocks, branch)
    _REFIT_BLS = BoxLeastSquares(branch.time, branch.flux)
    _REFIT_BASELINE_DAYS = float(np.nanmax(branch.time) - np.nanmin(branch.time))
    # The legacy path is candidate/process-parallel, so it keeps one compiled
    # validation thread per worker.  Native-exact normally uses one process;
    # give its fixed-ephemeris matrix the same bounded thread budget as BLS.
    fixed_threads = 1 if native_exact_threads is None else max(1, int(native_exact_threads))
    os.environ["HOUXING_FIXED_EPHEMERIS_THREADS"] = str(fixed_threads)
    # Keep the production default serial, but preserve an explicit override
    # for exact A/B benchmarks.  Unconditionally assigning ``1`` here made a
    # requested block-parallel experiment silently run the serial baseline.
    os.environ.setdefault(optimization_v4.FIXED_BLOCK_WORKERS_ENV, "1")
    os.environ["HOUXING_BLS_THREADS"] = "1"
    if native_exact_threads is not None:
        os.environ[exact_bls_acceleration.THREAD_ENV] = str(max(1, int(native_exact_threads)))
        _REFIT_NATIVE_EXACT_PREPARED = exact_bls_acceleration.prepare_exact_bls(
            branch.time,
            branch.flux,
        )
    else:
        _REFIT_NATIVE_EXACT_PREPARED = None
    _REFIT_NATIVE_EXACT_SOLUTION_CACHE = {}
    _REFIT_NATIVE_EXACT_DIAGNOSTICS = {
        "calls": 0,
        "periods_evaluated": 0,
        "period_trials_skipped": 0,
        "threads": 0,
        "tile_size": 0,
        "backend": None,
        "simd": None,
        "strategy": None,
        "seconds": 0.0,
        "strategy_calls": {},
        "prefetch_requests": 0,
        "prefetch_superbatches": 0,
        "cache_hits": 0,
    }


def _native_solution_cache_key(
    periods: np.ndarray,
    duration: float,
) -> tuple[bytes, int, bytes]:
    """Return an exact, collision-free key for one immutable period request."""

    requested = np.ascontiguousarray(periods, dtype=np.float64)
    return (
        np.asarray([duration], dtype=np.float64).tobytes(),
        int(requested.size),
        requested.tobytes(),
    )


def _native_exact_solution_uncached(*, periods, durations) -> Mapping:
    """Run one complete exact request and record its physical kernel work."""

    global _REFIT_NATIVE_EXACT_DIAGNOSTICS
    if _REFIT_BRANCH is None:
        raise RuntimeError("native exact refit worker was not initialized")
    if _REFIT_NATIVE_EXACT_PREPARED is None:
        raise RuntimeError("native exact BLS branch cache was not initialized")
    requested_periods = np.ascontiguousarray(periods, dtype=np.float64)
    started = time.perf_counter()
    solved = exact_bls_acceleration.exact_bls_solutions(
        _REFIT_BRANCH.time,
        _REFIT_BRANCH.flux,
        periods=requested_periods,
        durations=np.asarray(durations, dtype=np.float64),
        oversample=10,
        prepared=_REFIT_NATIVE_EXACT_PREPARED,
    )
    elapsed = time.perf_counter() - started
    if solved is None:
        raise RuntimeError("native exact BLS requires Numba")
    evaluated = np.asarray(solved["periods_evaluated"], dtype=np.float64)
    if not np.array_equal(evaluated, requested_periods):
        raise RuntimeError("native exact BLS changed or omitted period trials")
    skipped = int(np.asarray(solved["period_trials_skipped"]).reshape(-1)[0])
    if skipped != 0:
        raise RuntimeError("native exact BLS reported omitted period trials")
    if _REFIT_NATIVE_EXACT_DIAGNOSTICS is None:
        _REFIT_NATIVE_EXACT_DIAGNOSTICS = {}
    diagnostics = _REFIT_NATIVE_EXACT_DIAGNOSTICS
    diagnostics["calls"] = int(diagnostics.get("calls") or 0) + 1
    diagnostics["periods_evaluated"] = (
        int(diagnostics.get("periods_evaluated") or 0) + requested_periods.size
    )
    diagnostics["period_trials_skipped"] = (
        int(diagnostics.get("period_trials_skipped") or 0) + skipped
    )
    diagnostics["threads"] = int(np.asarray(solved["threads"]).reshape(-1)[0])
    diagnostics["tile_size"] = int(np.asarray(solved["tile_size"]).reshape(-1)[0])
    diagnostics["backend"] = str(solved.get("backend") or "unknown")
    diagnostics["simd"] = str(solved.get("simd") or "unknown")
    diagnostics["strategy"] = str(solved.get("strategy") or "unknown")
    strategy_calls = dict(diagnostics.get("strategy_calls") or {})
    strategy = str(solved.get("strategy") or "unknown")
    strategy_calls[strategy] = int(strategy_calls.get(strategy) or 0) + 1
    diagnostics["strategy_calls"] = strategy_calls
    diagnostics["seconds"] = float(diagnostics.get("seconds") or 0.0) + elapsed
    return solved


def _native_exact_solution_provider(*, periods, durations) -> Mapping:
    """Return an exact solution, reusing only byte-identical prefetched grids."""

    global _REFIT_NATIVE_EXACT_DIAGNOSTICS
    global _REFIT_NATIVE_EXACT_SOLUTION_CACHE
    requested_periods = np.ascontiguousarray(periods, dtype=np.float64)
    duration_array = np.asarray(durations, dtype=np.float64).reshape(-1)
    if duration_array.size != 1:
        raise ValueError("native exact BLS requires one duration per request")
    key = _native_solution_cache_key(requested_periods, float(duration_array[0]))
    cache = _REFIT_NATIVE_EXACT_SOLUTION_CACHE
    if cache is not None and key in cache:
        cached = cache[key]
        evaluated = np.asarray(cached["periods_evaluated"], dtype=np.float64)
        if not np.array_equal(evaluated, requested_periods):
            raise RuntimeError("native exact cache returned a different period grid")
        if _REFIT_NATIVE_EXACT_DIAGNOSTICS is not None:
            diagnostics = _REFIT_NATIVE_EXACT_DIAGNOSTICS
            diagnostics["cache_hits"] = int(diagnostics.get("cache_hits") or 0) + 1
        return cached
    solved = _native_exact_solution_uncached(
        periods=requested_periods,
        durations=duration_array,
    )
    if cache is not None:
        cache[key] = solved
    return solved


def _native_exact_solution_prefetch(
    requests: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Super-batch byte-exact grids by duration before cluster evaluation.

    Concatenation only changes dispatch: every period remains present and the
    returned arrays are split back at their original offsets before use.
    """

    global _REFIT_NATIVE_EXACT_DIAGNOSTICS
    global _REFIT_NATIVE_EXACT_SOLUTION_CACHE
    if not requests:
        return
    if _REFIT_NATIVE_EXACT_SOLUTION_CACHE is None:
        _REFIT_NATIVE_EXACT_SOLUTION_CACHE = {}
    cache = _REFIT_NATIVE_EXACT_SOLUTION_CACHE
    unique: dict[tuple[bytes, int, bytes], tuple[np.ndarray, float]] = {}
    for periods, durations in requests:
        requested = np.ascontiguousarray(periods, dtype=np.float64)
        duration_array = np.asarray(durations, dtype=np.float64).reshape(-1)
        if duration_array.size != 1:
            raise ValueError("native exact prefetch requires one duration")
        duration = float(duration_array[0])
        key = _native_solution_cache_key(requested, duration)
        if key not in cache:
            unique.setdefault(key, (requested, duration))
    if _REFIT_NATIVE_EXACT_DIAGNOSTICS is not None:
        diagnostics = _REFIT_NATIVE_EXACT_DIAGNOSTICS
        diagnostics["prefetch_requests"] = int(diagnostics.get("prefetch_requests") or 0) + len(
            requests
        )
    by_duration: dict[float, list[tuple[tuple[bytes, int, bytes], np.ndarray]]] = defaultdict(list)
    for key, (requested, duration) in unique.items():
        by_duration[duration].append((key, requested))
    for duration, entries in by_duration.items():
        sizes = np.asarray([periods.size for _key, periods in entries], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
        combined = np.concatenate([periods for _key, periods in entries])
        solved = _native_exact_solution_uncached(
            periods=combined,
            durations=np.asarray([duration], dtype=np.float64),
        )
        for index, (key, requested) in enumerate(entries):
            start = int(offsets[index])
            stop = int(offsets[index + 1])
            cache[key] = {
                "power": np.asarray(solved["power"])[start:stop],
                "epoch_btjd": np.asarray(solved["epoch_btjd"])[start:stop],
                "periods_evaluated": requested,
                "period_trials_skipped": np.asarray([0], dtype=np.int64),
                "threads": solved["threads"],
                "tile_size": solved["tile_size"],
                "backend": solved["backend"],
                "simd": solved["simd"],
                "strategy": solved["strategy"],
            }
        if _REFIT_NATIVE_EXACT_DIAGNOSTICS is not None:
            diagnostics = _REFIT_NATIVE_EXACT_DIAGNOSTICS
            diagnostics["prefetch_superbatches"] = (
                int(diagnostics.get("prefetch_superbatches") or 0) + 1
            )


def _refit_worker(item: tuple[int, dict]) -> tuple[int, dict]:
    if _REFIT_BRANCH is None or _REFIT_BLOCKS is None:
        raise RuntimeError("ephemeris-refit worker was not initialized")
    index, source = item
    candidate = deepcopy(source)
    before = deepcopy(candidate.get("holdout_verification") or {})
    optimization_v4.balance_family_ephemeris_vectorized(
        candidate,
        _REFIT_BLOCKS,
        _REFIT_BRANCH,
        attach_event_repeatability=False,
        prepared_blocks=_REFIT_PREPARED_BLOCKS,
        bls=_REFIT_BLS,
        baseline_days=_REFIT_BASELINE_DAYS,
    )
    return _finish_refit_candidate(index, source, candidate, before)


def _finish_refit_candidate(
    index: int,
    source: dict,
    candidate: dict,
    before: dict,
    *,
    execution: str = "candidate_exact_grid",
    shared_grid_diagnostics: dict | None = None,
    branch: core.SearchBranch | None = None,
    blocks: list[list[int]] | None = None,
) -> tuple[int, dict]:
    """Apply the unchanged post-refit scientific checks and audit payload."""

    active_branch = branch if branch is not None else _REFIT_BRANCH
    active_blocks = blocks if blocks is not None else _REFIT_BLOCKS
    if active_branch is None or active_blocks is None:
        raise RuntimeError("ephemeris-refit finalizer has no light-curve context")

    original_ephemeris = {
        "period_days": float(source["period_days"]),
        "epoch_btjd": float(source["epoch_btjd"]),
        "duration_hours": float(source["duration_hours"]),
    }
    verification = candidate.get("holdout_verification") or {}
    if verification.get("passed") is True:
        core._attach_event_repeatability(candidate, verification, active_branch)
        candidate["holdout_verification"] = verification
        core._apply_fold_event_direction_test(candidate, active_blocks)
        annotate_event_coherence(candidate, active_blocks)
        verification = candidate.get("holdout_verification") or {}
    verification["event_repeatability_role"] = (
        "measured_before_refit_rescue_acceptance"
        if verification.get("passed") is True
        else "skipped_after_failed_global_ephemeris_refit"
    )
    candidate["holdout_verification"] = verification
    candidate["ephemeris_refit"] = {
        "attempted": True,
        "eligible_without_catalog": True,
        "original_ephemeris": original_ephemeris,
        "selected_ephemeris": {
            "period_days": float(candidate["period_days"]),
            "epoch_btjd": float(candidate["epoch_btjd"]),
            "duration_hours": float(candidate["duration_hours"]),
        },
        "selection_policy": (
            "original ephemeris and pre-refit verification retained; selected "
            "ephemeris must pass fixed-block and event-direction validation"
        ),
        "rescued_failed_fixed_ephemeris": bool(
            before.get("passed") is not True and verification.get("passed") is True
        ),
        "pre_refit_verification": before,
        "execution": execution,
    }
    if shared_grid_diagnostics is not None:
        candidate["ephemeris_refit"]["shared_grid_diagnostics"] = shared_grid_diagnostics
    return index, candidate


def _refit_worker_batch(
    items: list[tuple[int, dict]],
) -> list[tuple[int, dict]]:
    """Process a small dynamic batch to amortize spawn-queue serialization."""

    return [_refit_worker(item) for item in items]


def _exact_batched_refit_worker_batch(
    items: list[tuple[int, dict]],
) -> tuple[list[tuple[int, dict]], dict]:
    """Evaluate every original candidate grid, batched only for execution.

    Unlike the optional shared-lattice experiment, this path concatenates the
    exact per-family period arrays and splits the results back at their exact
    offsets.  No period, epoch solution, validation trial, candidate, or
    threshold is omitted.  Harmonic members may be adjacent in a batch but
    each retains its complete independent grid and result.
    """

    if _REFIT_BRANCH is None or _REFIT_BLOCKS is None:
        raise RuntimeError("ephemeris-refit worker was not initialized")
    candidates = [deepcopy(source) for _index, source in items]
    coarse_points = 0
    fine_points = 0
    for candidate in candidates:
        spec = optimization_v4._family_balance_spec(
            candidate,
            _REFIT_BRANCH,
            baseline_days=_REFIT_BASELINE_DAYS,
        )
        if spec is not None:
            coarse_points += int(spec["grid_size"])
            fine_points += 101
    optimization_v4.balance_family_ephemerides_vectorized(
        candidates,
        _REFIT_BLOCKS,
        _REFIT_BRANCH,
        attach_event_repeatability=False,
        batch_size=max(1, len(candidates)),
        prepared_blocks=_REFIT_PREPARED_BLOCKS,
        bls=_REFIT_BLS,
        baseline_days=_REFIT_BASELINE_DAYS,
    )
    output: list[tuple[int, dict]] = []
    scalar_fallbacks = 0
    for (index, source), candidate in zip(items, candidates, strict=False):
        if "holdout_verification" not in candidate:
            output.append(_refit_worker((index, source)))
            scalar_fallbacks += 1
            continue
        output.append(
            _finish_refit_candidate(
                index,
                source,
                candidate,
                deepcopy(source.get("holdout_verification") or {}),
                execution="exact_candidate_grids_batched_without_omission",
            )
        )
    return output, {
        "families": len(items),
        "exact_original_coarse_points": coarse_points,
        "exact_original_fine_points": fine_points,
        "candidate_grids_skipped": 0,
        "period_trials_skipped": 0,
        "scalar_fallbacks": scalar_fallbacks,
    }


def _native_exact_refit_worker_batch(
    items: list[tuple[int, dict]],
) -> tuple[list[tuple[int, dict]], dict]:
    """Use the exact tiled CPU kernel without changing any candidate grid."""

    global _REFIT_NATIVE_EXACT_DIAGNOSTICS
    global _REFIT_NATIVE_EXACT_SOLUTION_CACHE
    if _REFIT_BRANCH is None or _REFIT_BLOCKS is None:
        raise RuntimeError("ephemeris-refit worker was not initialized")
    _REFIT_NATIVE_EXACT_DIAGNOSTICS = {
        "calls": 0,
        "periods_evaluated": 0,
        "period_trials_skipped": 0,
        "threads": 0,
        "tile_size": 0,
        "backend": None,
        "simd": None,
        "strategy": None,
        "seconds": 0.0,
        "strategy_calls": {},
        "prefetch_requests": 0,
        "prefetch_superbatches": 0,
        "cache_hits": 0,
    }
    _REFIT_NATIVE_EXACT_SOLUTION_CACHE = {}
    candidates = [deepcopy(source) for _index, source in items]
    coarse_points = 0
    fine_points = 0
    for candidate in candidates:
        spec = optimization_v4._family_balance_spec(
            candidate,
            _REFIT_BRANCH,
            baseline_days=_REFIT_BASELINE_DAYS,
        )
        if spec is not None:
            coarse_points += int(spec["grid_size"])
            fine_points += 101
    optimization_v4.balance_family_ephemerides_vectorized(
        candidates,
        _REFIT_BLOCKS,
        _REFIT_BRANCH,
        attach_event_repeatability=False,
        batch_size=max(1, len(candidates)),
        prepared_blocks=_REFIT_PREPARED_BLOCKS,
        bls=_REFIT_BLS,
        baseline_days=_REFIT_BASELINE_DAYS,
        solution_provider=_native_exact_solution_provider,
    )
    output: list[tuple[int, dict]] = []
    scalar_fallbacks = 0
    for (index, source), candidate in zip(items, candidates, strict=False):
        if "holdout_verification" not in candidate:
            # A missing scientific result is never silently accepted.  The
            # unchanged Astropy scalar path remains the exact fallback.
            output.append(_refit_worker((index, source)))
            scalar_fallbacks += 1
            continue
        output.append(
            _finish_refit_candidate(
                index,
                source,
                candidate,
                deepcopy(source.get("holdout_verification") or {}),
                execution=("native_cpp_or_numba_exact_candidate_grids_without_omission"),
            )
        )
    native = dict(_REFIT_NATIVE_EXACT_DIAGNOSTICS or {})
    return output, {
        "families": len(items),
        "exact_original_coarse_points": coarse_points,
        "exact_original_fine_points": fine_points,
        "candidate_grids_skipped": 0,
        "period_trials_skipped": int(native.get("period_trials_skipped") or 0),
        "scalar_fallbacks": scalar_fallbacks,
        "native_exact": native,
    }


def _shared_refit_worker_batch(
    items: list[tuple[int, dict]],
) -> tuple[list[tuple[int, dict]], dict]:
    """Share overlapping BLS windows, with exact per-candidate fallback."""

    if _REFIT_BRANCH is None or _REFIT_BLOCKS is None:
        raise RuntimeError("ephemeris-refit worker was not initialized")
    candidates = [deepcopy(source) for _index, source in items]
    diagnostics = optimization_v4.balance_family_ephemerides_shared(
        candidates,
        _REFIT_BLOCKS,
        _REFIT_BRANCH,
        prepared_blocks=_REFIT_PREPARED_BLOCKS,
        bls=_REFIT_BLS,
        baseline_days=_REFIT_BASELINE_DAYS,
        max_candidates=len(candidates),
    )
    output: list[tuple[int, dict]] = []
    fallbacks = 0
    for (index, source), candidate in zip(items, candidates, strict=False):
        before = deepcopy(source.get("holdout_verification") or {})
        completed = _finish_refit_candidate(
            index,
            source,
            candidate,
            before,
            execution="shared_no_coarser_period_grid",
            shared_grid_diagnostics=diagnostics,
        )
        # A shared-grid result that does not survive every unchanged event
        # check is never allowed to replace the reference calculation.
        if completed[1]["holdout_verification"].get("passed") is not True:
            completed = _refit_worker((index, source))
            completed[1]["ephemeris_refit"]["shared_grid_attempt"] = {
                **diagnostics,
                "fell_back_to_exact_candidate_grid": True,
            }
            fallbacks += 1
        output.append(completed)
    return output, {**diagnostics, "exact_fallbacks": fallbacks}


def _merge_native_exact_diagnostics(*rows: Mapping | None) -> dict:
    """Combine independent native-kernel accounting without hiding work."""

    available = [dict(row) for row in rows if row]
    strategy_calls: dict[str, int] = defaultdict(int)
    for row in available:
        for strategy, count in dict(row.get("strategy_calls") or {}).items():
            strategy_calls[str(strategy)] += int(count)
    strategies = [key for key, count in strategy_calls.items() if count > 0]
    return {
        "calls": sum(int(row.get("calls") or 0) for row in available),
        "periods_evaluated": sum(int(row.get("periods_evaluated") or 0) for row in available),
        "period_trials_skipped": sum(
            int(row.get("period_trials_skipped") or 0) for row in available
        ),
        "threads": max((int(row.get("threads") or 0) for row in available), default=0),
        "tile_size": max((int(row.get("tile_size") or 0) for row in available), default=0),
        "backend": next(
            (row.get("backend") for row in reversed(available) if row.get("backend")),
            None,
        ),
        "simd": next(
            (row.get("simd") for row in reversed(available) if row.get("simd")),
            None,
        ),
        "strategy": (strategies[0] if len(strategies) == 1 else "hybrid" if strategies else None),
        "seconds": sum(float(row.get("seconds") or 0.0) for row in available),
        "strategy_calls": dict(strategy_calls),
        "prefetch_requests": sum(int(row.get("prefetch_requests") or 0) for row in available),
        "prefetch_superbatches": sum(
            int(row.get("prefetch_superbatches") or 0) for row in available
        ),
        "cache_hits": sum(int(row.get("cache_hits") or 0) for row in available),
    }


def _adaptive_fallback_quality(candidate: Mapping) -> tuple:
    """Rank ambiguous shared-grid failures without catalogue information."""

    vote = candidate.get("ensemble_vote") or {}
    verification = candidate.get("holdout_verification") or {}
    measurements = list(verification.get("block_measurements") or [])
    positive = [row for row in measurements if row.get("outcome") in {"strong", "weak"}]
    signed_snrs = [
        float(row.get("snr") or 0.0) * (1.0 if float(row.get("depth_ppm") or 0.0) > 0.0 else -1.0)
        for row in measurements
    ]
    discovered_blocks = {int(value) for value in candidate.get("discovered_blocks", [])}
    return (
        len(discovered_blocks) >= 2,
        float(candidate.get("period_days") or 0.0) >= 20.0,
        len(positive),
        min(signed_snrs, default=-np.inf),
        sum(max(0.0, value) ** 2 for value in signed_snrs) ** 0.5,
        int(vote.get("positive_round_count") or 0),
        int(vote.get("positive_groups") or 0),
        int(vote.get("strong_groups") or 0),
        -int(vote.get("contradicting_groups") or 0),
        float(candidate.get("source_strength") or 0.0),
    )


def _adaptive_long_period_protected(candidate: Mapping) -> bool:
    """Protect sparse long-period event trains from budget truncation."""

    if float(candidate.get("period_days") or 0.0) < float(
        os.environ.get(
            "HOUXING_ADAPTIVE_LONG_PERIOD_DAYS",
            ADAPTIVE_LONG_PERIOD_PROTECTION_DAYS,
        )
    ):
        return False
    vote = candidate.get("ensemble_vote") or {}
    rounds = max(
        1,
        int(vote.get("rounds_measured") or 0),
        len(vote.get("positive_rounds") or []),
    )
    minimum_rounds = max(2, int(math.ceil(0.8 * rounds)))
    positive_rounds = int(vote.get("positive_round_count") or 0)
    strong_groups = int(vote.get("strong_groups") or 0)
    contradicting_groups = int(vote.get("contradicting_groups") or 0)
    full_round_low_contradiction = (
        positive_rounds >= rounds and strong_groups >= 2 and contradicting_groups <= 2
    )
    return (
        positive_rounds >= minimum_rounds
        and (strong_groups >= 3 or full_round_low_contradiction)
        and (contradicting_groups <= 5 or float(candidate.get("source_strength") or 0.0) >= 24.0)
    )


def _select_adaptive_exact_fallbacks(
    unresolved: list[tuple[int, dict, dict]],
    *,
    fraction: float,
) -> tuple[list[tuple[int, dict]], dict]:
    """Select a deterministic strict fallback budget and retain every row."""

    ordered = sorted(
        unresolved,
        key=lambda row: _adaptive_fallback_quality(row[2]),
        reverse=True,
    )
    requested = float(np.clip(fraction, 0.0, 1.0))
    budget = min(len(ordered), int(math.ceil(requested * len(ordered))))
    selected_indices = {index for index, _source, _candidate in ordered[:budget]}
    protected_indices = {
        index
        for index, _source, candidate in ordered[budget:]
        if _adaptive_long_period_protected(candidate)
    }
    selected_indices.update(protected_indices)
    selected = [
        (index, source) for index, source, _candidate in unresolved if index in selected_indices
    ]
    return selected, {
        "requested_fraction": requested,
        "ranked_budget_families": budget,
        "long_period_protected_families": len(protected_indices),
        "strict_fallback_families": len(selected),
        "deferred_families": len(unresolved) - len(selected),
        "candidate_rows_deleted": False,
        "truth_catalog_used": False,
    }


def _targeted_rescue_quality(candidate: Mapping) -> tuple:
    """Rank likely clock-drift failures before any expensive dense refit.

    The strongest signature of an imprecise ephemeris is support in two
    complete temporal blocks with the remaining block failing.  Deterministic
    resampling support then ranks otherwise similar rows.  Catalogue periods,
    target names, and previously published dispositions are deliberately not
    inputs.
    """

    def finite(value, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float(default)
        return result if math.isfinite(result) else float(default)

    vote = candidate.get("ensemble_vote") or {}
    seed = candidate.get("vote_seed_consensus") or {}
    verification = candidate.get("holdout_verification") or {}
    measurements = list(verification.get("block_measurements") or [])
    positive = [row for row in measurements if row.get("outcome") in {"strong", "weak"}]
    strong = [row for row in measurements if row.get("outcome") == "strong"]
    positive_snrs = [finite(row.get("snr")) for row in positive]
    positive_depth_blocks = sum(float(row.get("depth_ppm") or 0.0) > 0.0 for row in measurements)
    return (
        len(positive),
        len(strong),
        min(positive_snrs, default=-1.0e300),
        int(seed.get("refit_supported_seeds") or 0),
        finite(seed.get("refit_support_rate")),
        int(vote.get("positive_round_count") or 0),
        int(vote.get("strong_groups") or 0),
        int(vote.get("positive_groups") or 0),
        -int(vote.get("contradicting_groups") or 0),
        positive_depth_blocks,
        finite(seed.get("mean_rank_percentile")),
        finite(candidate.get("source_strength")),
        finite(candidate.get("robust_sde")),
    )


def _targeted_period_band(period_days: float) -> int:
    for index, (lower, upper, _weight) in enumerate(TARGETED_REFIT_PERIOD_BANDS):
        if lower <= period_days < upper:
            return index
    return len(TARGETED_REFIT_PERIOD_BANDS) - 1


def _select_targeted_precision_rescues(
    eligible: list[tuple[int, dict]],
    *,
    max_families: int | None = None,
    relative_period_separation: float | None = None,
) -> tuple[list[tuple[int, dict]], list[tuple[int, dict]], dict]:
    """Build a small, period-diverse, catalogue-blind precision-rescue board.

    Every input row remains in the audit.  The board only controls which
    unresolved rows receive the expensive dense period/epoch calculation in
    this run.  Period-neighbour suppression is therefore scheduling, not a
    scientific merge or candidate deletion.
    """

    maximum = max(
        1,
        int(
            max_families
            if max_families is not None
            else os.environ.get(
                "HOUXING_TARGETED_REFIT_MAX_FAMILIES",
                TARGETED_REFIT_MAX_FAMILIES,
            )
        ),
    )
    separation = float(
        relative_period_separation
        if relative_period_separation is not None
        else os.environ.get(
            "HOUXING_TARGETED_REFIT_RELATIVE_PERIOD_SEPARATION",
            TARGETED_REFIT_RELATIVE_PERIOD_SEPARATION,
        )
    )
    separation = max(0.0, separation)
    ordered = sorted(
        eligible,
        key=lambda item: (
            _targeted_rescue_quality(item[1]),
            str(item[1].get("proposal_id") or item[1].get("raw_id") or ""),
        ),
        reverse=True,
    )
    weights = [float(row[2]) for row in TARGETED_REFIT_PERIOD_BANDS]
    weight_total = sum(weights) or 1.0
    quotas = [int(math.floor(maximum * weight / weight_total)) for weight in weights]
    for index in range(maximum - sum(quotas)):
        quotas[index % len(quotas)] += 1

    selected: list[tuple[int, dict]] = []
    selected_indices: set[int] = set()
    selected_periods_by_band: dict[int, list[float]] = defaultdict(list)

    def separated(period: float, band: int) -> bool:
        return all(
            abs(period - previous) / max(np.finfo(float).eps, min(period, previous)) > separation
            for previous in selected_periods_by_band[band]
        )

    for band, quota in enumerate(quotas):
        if quota <= 0:
            continue
        for index, candidate in ordered:
            period = float(candidate.get("period_days") or 0.0)
            if _targeted_period_band(period) != band or not separated(period, band):
                continue
            selected.append((index, candidate))
            selected_indices.add(index)
            selected_periods_by_band[band].append(period)
            if len(selected_periods_by_band[band]) >= quota:
                break

    # If a band contains fewer useful rows than its quota, spend the remainder
    # on the globally strongest still-separated rows rather than wasting it.
    if len(selected) < maximum:
        for index, candidate in ordered:
            if index in selected_indices:
                continue
            period = float(candidate.get("period_days") or 0.0)
            band = _targeted_period_band(period)
            if not separated(period, band):
                continue
            selected.append((index, candidate))
            selected_indices.add(index)
            selected_periods_by_band[band].append(period)
            if len(selected) >= maximum:
                break

    selected.sort(key=lambda item: item[0])
    deferred = [item for item in eligible if item[0] not in selected_indices]
    quality_rank = {index: rank for rank, (index, _candidate) in enumerate(ordered, start=1)}
    board = [
        {
            "proposal_id": candidate.get("proposal_id"),
            "period_days": float(candidate.get("period_days") or 0.0),
            "period_band": _targeted_period_band(float(candidate.get("period_days") or 0.0)),
            "preliminary_evidence_rank": int(quality_rank[index]),
            "quality": list(_targeted_rescue_quality(candidate)),
        }
        for index, candidate in sorted(
            selected,
            key=lambda item: quality_rank[item[0]],
        )
    ]
    return (
        selected,
        deferred,
        {
            "policy": "post_initial_validation_targeted_precision_rescue",
            "input_unresolved_families": len(eligible),
            "maximum_dense_refit_families": maximum,
            "selected_dense_refit_families": len(selected),
            "deferred_but_retained_families": len(deferred),
            "relative_period_separation": separation,
            "period_bands_days": [
                [lower, None if math.isinf(upper) else upper]
                for lower, upper, _weight in TARGETED_REFIT_PERIOD_BANDS
            ],
            "period_band_quotas": quotas,
            "selected_by_period_band": {
                str(index): len(periods) for index, periods in selected_periods_by_band.items()
            },
            "preliminary_rescue_board": board,
            "candidate_rows_deleted": False,
            "truth_catalog_used": False,
        },
    )


def _schedule_targeted_refits_before_compute(
    eligible: list[tuple[int, dict]],
    output: list[dict],
) -> tuple[list[tuple[int, dict]], dict]:
    """Select and annotate the one shared targeted rescue plan.

    CPU and CUDA refit backends both call this helper before *any* linear or
    dense refit calculation.  Keeping the scheduling and audit annotation in
    one function prevents the two execution backends from silently drifting.
    """

    selected, deferred, decision = _select_targeted_precision_rescues(eligible)
    quality_order = sorted(
        eligible,
        key=lambda item: (
            _targeted_rescue_quality(item[1]),
            str(item[1].get("proposal_id") or item[1].get("raw_id") or ""),
        ),
        reverse=True,
    )
    rank_by_index = {
        index: rank
        for rank, (index, _candidate) in enumerate(quality_order, start=1)
    }
    for index, source in deferred:
        retained = deepcopy(source)
        retained.setdefault("ephemeris_refit", {})["targeted_rescue_deferred"] = {
            "reason": "outside_post_ranking_precision_rescue_board",
            "preliminary_evidence_rank": int(rank_by_index[index]),
            "candidate_retained_for_audit": True,
            "scientific_status": "unresolved",
            "truth_catalog_used": False,
        }
        output[index] = retained
    decision = {
        **decision,
        "selected_before_any_refit_compute": True,
        "linear_or_dense_attempted_families": len(selected),
    }
    return selected, decision


def _plan_post_refit_revote(
    validated: list[dict],
    *,
    refit_mode: str,
) -> tuple[list[tuple[int, dict]], list[tuple[int, dict]], dict]:
    """Separate scientific revotes from append-only audit rows.

    Targeted rescue deliberately leaves unresolved families in ``validated``
    so the audit is complete. Those rows failed the unchanged locked-
    ephemeris holdout and were not selected or rescued; re-scoring them across
    every seed cannot promote them because the final stable set still requires
    that same holdout to pass. Families that passed before rescue already have
    complete deterministic votes and their ephemerides did not change. Only a
    successfully rescued family needs a new vote measurement.

    Non-targeted modes retain the historical full re-vote contract. No row is
    deleted in either case.
    """

    indexed = list(enumerate(validated))
    if str(refit_mode) != "targeted":
        active = indexed
        audit_only: list[tuple[int, dict]] = []
        unchanged_pass: list[tuple[int, dict]] = []
    else:
        active = [
            item
            for item in indexed
            if item[1].get("holdout_verification", {}).get("passed") is True
            and item[1].get("ephemeris_refit", {}).get("rescued_failed_fixed_ephemeris") is True
        ]
        active_indices = {index for index, _row in active}
        audit_only = [
            item
            for item in indexed
            if item[1].get("holdout_verification", {}).get("passed") is not True
        ]
        unchanged_pass = [
            item
            for item in indexed
            if item[1].get("holdout_verification", {}).get("passed") is True
            and item[0] not in active_indices
        ]
    return (
        active,
        audit_only,
        {
            "policy": (
                "locked_holdout_pass_only"
                if str(refit_mode) == "targeted"
                else "all_validated_families"
            ),
            "post_refit_revote_families": len(active),
            "post_refit_unchanged_pass_families": len(unchanged_pass),
            "post_refit_deferred_audit_only_families": len(audit_only),
            "candidate_rows_retained": len(validated),
            "candidate_rows_deleted": False,
        },
    )


def _adaptive_refit_decision(
    items: list[tuple[int, dict]],
    branch: core.SearchBranch,
) -> dict:
    """Choose lossless exact mode unless the measured grid truly explodes."""

    baseline = float(np.nanmax(branch.time) - np.nanmin(branch.time))
    coarse_points = 0
    valid_specs = 0
    for _index, candidate in items:
        spec = optimization_v4._family_balance_spec(
            candidate,
            branch,
            baseline_days=baseline,
        )
        if spec is None:
            continue
        valid_specs += 1
        coarse_points += int(spec["grid_size"])
    threshold = max(
        1,
        int(
            os.environ.get(
                "HOUXING_ADAPTIVE_EXPLOSION_MIN_COARSE_POINTS",
                ADAPTIVE_EXPLOSION_MIN_COARSE_POINTS,
            )
        ),
    )
    execution_mode = "native-selective" if coarse_points >= threshold else "native-exact"
    return {
        "execution_mode": execution_mode,
        "dense_families": len(items),
        "families_with_balance_grid": valid_specs,
        "estimated_original_coarse_points": coarse_points,
        "explosion_threshold_coarse_points": threshold,
        "approximation_enabled": execution_mode == "native-selective",
        "normal_targets_remain_lossless": execution_mode == "native-exact",
        "decision_uses_truth_catalog": False,
    }


def _near_period_event_train_duplicate(
    representative: Mapping,
    candidate: Mapping,
) -> dict | None:
    """Prove that two near-period solutions use the same measured dips.

    Period proximity alone is deliberately insufficient.  The selected epochs
    must align, and supported low points must overlap one-to-one in at least
    two sectors.  This relation is used only to avoid a duplicate strict-grid
    fallback; both original candidate rows remain in the audit trail.
    """

    first_period = float(representative.get("period_days") or 0.0)
    second_period = float(candidate.get("period_days") or 0.0)
    if first_period <= 0.0 or second_period <= 0.0:
        return None
    relative_period_error = abs(first_period - second_period) / min(first_period, second_period)
    if relative_period_error > 0.01:
        return None
    tolerance_days = (
        max(
            float(representative.get("duration_hours") or 0.0),
            float(candidate.get("duration_hours") or 0.0),
        )
        / 24.0
    )
    if tolerance_days <= 0.0:
        return None
    reference_period = min(first_period, second_period)
    phase_error = abs(
        (
            float(candidate.get("epoch_btjd") or 0.0)
            - float(representative.get("epoch_btjd") or 0.0)
            + 0.5 * reference_period
        )
        % reference_period
        - 0.5 * reference_period
    )
    if phase_error > tolerance_days:
        return None
    first_support = (
        (representative.get("holdout_verification") or {})
        .get("event_repeatability", {})
        .get("event_support")
    )
    second_support = (
        (candidate.get("holdout_verification") or {})
        .get("event_repeatability", {})
        .get("event_support")
    )
    if not first_support or not second_support:
        return None
    first_events = harmonic_families._supported_events(representative)
    second_events = harmonic_families._supported_events(candidate)
    if not first_events or not second_events:
        return None
    matched, segments, containment = harmonic_families._matched_events(
        first_events,
        second_events,
        tolerance_days=tolerance_days,
    )
    if matched < 3 or len(segments) < 2 or containment < 0.80:
        return None
    return {
        "representative_proposal_id": str(representative["proposal_id"]),
        "duplicate_proposal_id": str(candidate["proposal_id"]),
        "relation_type": "near_period_same_measured_event_train",
        "relative_period_error": relative_period_error,
        "phase_error_days": phase_error,
        "event_center_tolerance_days": tolerance_days,
        "matched_supported_events": matched,
        "matched_segments": segments,
        "supported_event_containment": containment,
        "representative_supported_events": len(first_events),
        "duplicate_supported_events": len(second_events),
        "candidate_removed_from_audit": False,
        "period_arithmetic_alone_was_sufficient": False,
        "truth_catalog_used": False,
    }


def _native_shared_refit_worker_batch(
    items: list[tuple[int, dict]],
    *,
    max_candidates: int | None = None,
) -> tuple[list[tuple[int, dict]], dict]:
    """Run exact-membership shared grids in C++, then fallback failures.

    The execution lattice is the exact union of the original family grids;
    every family ranks only its own membership indices.  A family that does
    not pass every unchanged post-fit check is still recomputed by the strict
    native-exact path as a defensive regression check.
    """

    global _REFIT_NATIVE_EXACT_DIAGNOSTICS
    global _REFIT_NATIVE_EXACT_SOLUTION_CACHE
    if _REFIT_BRANCH is None or _REFIT_BLOCKS is None:
        raise RuntimeError("ephemeris-refit worker was not initialized")
    _REFIT_NATIVE_EXACT_DIAGNOSTICS = {
        "calls": 0,
        "periods_evaluated": 0,
        "period_trials_skipped": 0,
        "threads": 0,
        "tile_size": 0,
        "backend": None,
        "simd": None,
        "strategy": None,
        "seconds": 0.0,
        "strategy_calls": {},
        "prefetch_requests": 0,
        "prefetch_superbatches": 0,
        "cache_hits": 0,
    }
    _REFIT_NATIVE_EXACT_SOLUTION_CACHE = {}
    candidates = [deepcopy(source) for _index, source in items]
    diagnostics = optimization_v4.balance_family_ephemerides_shared(
        candidates,
        _REFIT_BLOCKS,
        _REFIT_BRANCH,
        prepared_blocks=_REFIT_PREPARED_BLOCKS,
        bls=_REFIT_BLS,
        baseline_days=_REFIT_BASELINE_DAYS,
        max_candidates=(
            max(1, int(max_candidates)) if max_candidates is not None else len(candidates)
        ),
        solution_provider=_native_exact_solution_provider,
        power_prefetch=_native_exact_solution_prefetch,
    )
    shared_native = dict(_REFIT_NATIVE_EXACT_DIAGNOSTICS or {})
    output_by_index: dict[int, dict] = {}
    unresolved: list[tuple[int, dict, dict]] = []
    for (index, source), candidate in zip(items, candidates, strict=False):
        before = deepcopy(source.get("holdout_verification") or {})
        completed_index, completed = _finish_refit_candidate(
            index,
            source,
            candidate,
            before,
            execution="native_shared_no_coarser_period_grid",
            shared_grid_diagnostics=diagnostics,
        )
        if completed["holdout_verification"].get("passed") is True:
            output_by_index[completed_index] = completed
        else:
            unresolved.append((index, source, completed))

    # Every shared-grid failure is recomputed on its complete original grid.
    # Event-train equivalence remains useful later for harmonic annotation,
    # but must never be used to skip a scientific refit in the default path.
    fallback_items = [(index, source) for index, source, _candidate in unresolved]
    event_train_aliases = 0

    fallback_native = None
    exact_original_coarse_points = 0
    exact_original_fine_points = 0
    if fallback_items:
        fallback_rows, fallback_diagnostics = _native_exact_refit_worker_batch(fallback_items)
        fallback_native = fallback_diagnostics.get("native_exact") or {}
        exact_original_coarse_points = int(
            fallback_diagnostics.get("exact_original_coarse_points") or 0
        )
        exact_original_fine_points = int(
            fallback_diagnostics.get("exact_original_fine_points") or 0
        )
        for index, candidate in fallback_rows:
            candidate["ephemeris_refit"]["shared_grid_attempt"] = {
                **diagnostics,
                "fell_back_to_exact_candidate_grid": True,
            }
            output_by_index[index] = candidate

    native = _merge_native_exact_diagnostics(shared_native, fallback_native)
    if int(native["period_trials_skipped"]) != 0:
        raise RuntimeError("native shared refit omitted period trials")
    return [(index, output_by_index[index]) for index, _source in items], {
        **diagnostics,
        "exact_fallbacks": len(fallback_items),
        "event_train_duplicate_fallbacks_skipped": event_train_aliases,
        "all_shared_failures_received_strict_exact_fallback": True,
        "fallback_exact_original_coarse_points": exact_original_coarse_points,
        "fallback_exact_original_fine_points": exact_original_fine_points,
        "native_exact": native,
    }


def _native_selective_refit_worker_batch(
    items: list[tuple[int, dict]],
    *,
    max_candidates: int | None = None,
) -> tuple[list[tuple[int, dict]], dict]:
    """Shared-first refit for true explosions with an audited exact budget."""

    global _REFIT_NATIVE_EXACT_DIAGNOSTICS
    global _REFIT_NATIVE_EXACT_SOLUTION_CACHE
    if _REFIT_BRANCH is None or _REFIT_BLOCKS is None:
        raise RuntimeError("ephemeris-refit worker was not initialized")
    _REFIT_NATIVE_EXACT_DIAGNOSTICS = {
        "calls": 0,
        "periods_evaluated": 0,
        "period_trials_skipped": 0,
        "threads": 0,
        "tile_size": 0,
        "backend": None,
        "simd": None,
        "strategy": None,
        "seconds": 0.0,
        "strategy_calls": {},
        "prefetch_requests": 0,
        "prefetch_superbatches": 0,
        "cache_hits": 0,
    }
    _REFIT_NATIVE_EXACT_SOLUTION_CACHE = {}
    candidates = [deepcopy(source) for _index, source in items]
    diagnostics = optimization_v4.balance_family_ephemerides_shared(
        candidates,
        _REFIT_BLOCKS,
        _REFIT_BRANCH,
        prepared_blocks=_REFIT_PREPARED_BLOCKS,
        bls=_REFIT_BLS,
        baseline_days=_REFIT_BASELINE_DAYS,
        max_candidates=(
            max(1, int(max_candidates)) if max_candidates is not None else len(candidates)
        ),
        solution_provider=_native_exact_solution_provider,
        power_prefetch=_native_exact_solution_prefetch,
    )
    shared_native = dict(_REFIT_NATIVE_EXACT_DIAGNOSTICS or {})
    output_by_index: dict[int, dict] = {}
    unresolved: list[tuple[int, dict, dict]] = []
    for (index, source), candidate in zip(items, candidates, strict=False):
        completed_index, completed = _finish_refit_candidate(
            index,
            source,
            candidate,
            deepcopy(source.get("holdout_verification") or {}),
            execution="native_adaptive_shared_screen",
            shared_grid_diagnostics=diagnostics,
        )
        if completed["holdout_verification"].get("passed") is True:
            output_by_index[completed_index] = completed
        else:
            unresolved.append((index, source, completed))

    fallback_fraction = float(
        os.environ.get(
            "HOUXING_ADAPTIVE_STRICT_FALLBACK_FRACTION",
            ADAPTIVE_STRICT_FALLBACK_FRACTION,
        )
    )
    fallback_items, selection = _select_adaptive_exact_fallbacks(
        unresolved,
        fraction=fallback_fraction,
    )
    selected_indices = {index for index, _source in fallback_items}
    for index, _source, completed in unresolved:
        if index in selected_indices:
            continue
        completed["ephemeris_refit"]["adaptive_deferred"] = {
            "reason": "outside_strict_fallback_budget_after_shared_failure",
            "candidate_retained_for_audit": True,
            "scientific_status": "unresolved",
            "truth_catalog_used": False,
        }
        output_by_index[index] = completed

    fallback_native = None
    exact_original_coarse_points = 0
    exact_original_fine_points = 0
    if fallback_items:
        fallback_rows, fallback_diagnostics = _native_exact_refit_worker_batch(fallback_items)
        fallback_native = fallback_diagnostics.get("native_exact") or {}
        exact_original_coarse_points = int(
            fallback_diagnostics.get("exact_original_coarse_points") or 0
        )
        exact_original_fine_points = int(
            fallback_diagnostics.get("exact_original_fine_points") or 0
        )
        for index, candidate in fallback_rows:
            candidate["ephemeris_refit"]["shared_grid_attempt"] = {
                **diagnostics,
                "fell_back_to_exact_candidate_grid": True,
            }
            output_by_index[index] = candidate

    native = _merge_native_exact_diagnostics(shared_native, fallback_native)
    if int(native["period_trials_skipped"]) != 0:
        raise RuntimeError("native adaptive refit omitted an executed period trial")
    return [(index, output_by_index[index]) for index, _source in items], {
        **diagnostics,
        **selection,
        "exact_fallbacks": len(fallback_items),
        "all_shared_failures_received_strict_exact_fallback": (
            int(selection.get("deferred_families") or 0) == 0
        ),
        "fallback_exact_original_coarse_points": exact_original_coarse_points,
        "fallback_exact_original_fine_points": exact_original_fine_points,
        "native_exact": native,
    }


def _plan_shared_refit_batches(
    eligible: list[tuple[int, dict]],
    branch: core.SearchBranch,
    *,
    max_candidates: int,
) -> list[list[tuple[int, dict]]]:
    """Plan bounded, duration-identical overlap batches in the parent."""

    baseline = float(np.nanmax(branch.time) - np.nanmin(branch.time))
    specs: list[dict] = []
    singleton_items: list[tuple[int, dict]] = []
    for item in eligible:
        spec = optimization_v4._family_balance_spec(item[1], branch, baseline_days=baseline)
        if spec is None:
            singleton_items.append(item)
            continue
        spec["refit_item"] = item
        specs.append(spec)
    clusters = optimization_v4.build_shared_period_clusters(
        specs,
        max_candidates=max_candidates,
    )
    planned: list[tuple[float, list[tuple[int, dict]]]] = []
    data_points = float(branch.time.size)
    for cluster in clusters:
        periods = np.asarray(cluster["periods"], dtype=float)
        duration_days = float(cluster["duration_days"])
        phase_bins = np.ceil(periods / (duration_days / 10.0)) + 10.0
        # Same shape as the Astropy fast-BLS cost model.  Long, dense tasks go
        # first so the process pool cannot finish with one enormous straggler.
        estimated_cost = float(np.sum(data_points + 3.0 * phase_bins, dtype=np.float64))
        planned.append(
            (
                estimated_cost,
                [spec["refit_item"] for spec in cluster["specs"]],
            )
        )
    planned.extend((0.0, [item]) for item in singleton_items)
    planned.sort(key=lambda row: row[0], reverse=True)
    return [batch for _cost, batch in planned]


def _pre_refit_integer_harmonic_relation(
    first: Mapping,
    second: Mapping,
) -> dict | None:
    """Return a conservative scheduling-only integer-harmonic relation.

    This predicate never removes a candidate.  It merely lets one directly
    phase-aligned member run before its aliases.  Actual deferral requires the
    downstream event-backed harmonic arbiter to resolve the same pair after
    the representative has passed every unchanged fold/event check.
    """

    first_period = float(first.get("period_days") or 0.0)
    second_period = float(second.get("period_days") or 0.0)
    if not (
        np.isfinite(first_period)
        and np.isfinite(second_period)
        and first_period > 0
        and second_period > 0
    ):
        return None
    shorter, longer = (first, second) if first_period < second_period else (second, first)
    short_period = float(shorter["period_days"])
    long_period = float(longer["period_days"])
    ratio = long_period / short_period
    order = int(round(ratio))
    if not (2 <= order <= harmonic_families.MAX_INTEGER_HARMONIC_ORDER):
        return None
    relative_error = abs(ratio - order) / order
    if relative_error > harmonic_families.INTEGER_RATIO_RELATIVE_TOLERANCE:
        return None
    tolerance_days = (
        max(
            float(shorter.get("duration_hours") or 0.0),
            float(longer.get("duration_hours") or 0.0),
        )
        / 24.0
    )
    if tolerance_days <= 0:
        return None
    phase_error = abs(
        (float(longer["epoch_btjd"]) - float(shorter["epoch_btjd"]) + 0.5 * short_period)
        % short_period
        - 0.5 * short_period
    )
    if phase_error > tolerance_days:
        return None
    return {
        "shorter_proposal_id": str(shorter["proposal_id"]),
        "longer_proposal_id": str(longer["proposal_id"]),
        "integer_order": order,
        "period_ratio": ratio,
        "relative_error": relative_error,
        "phase_error_days": phase_error,
        "scheduling_only": True,
        "candidate_removed": False,
    }


def _plan_pre_refit_harmonic_groups(
    eligible: list[tuple[int, dict]],
    *,
    maximum_group_size: int = 64,
    branch: core.SearchBranch | None = None,
    require_event_proof: bool = False,
) -> tuple[list[list[tuple[int, dict]]], list[dict]]:
    """Build direct-anchor groups without transitive harmonic chaining.

    With ``require_event_proof`` period arithmetic is only a cheap index.  A
    scheduling edge exists only after both locked hypotheses have measured
    the same downward event windows in multiple sectors.  Unproved candidates
    remain independent and still receive their complete original refit grid.
    """

    if not eligible:
        return [], []
    candidates = [row for _index, row in eligible]
    arithmetic_pairs: list[tuple[int, int, dict]] = []
    for left, right in harmonic_families._commensurate_candidate_pairs(candidates):
        relation = _pre_refit_integer_harmonic_relation(candidates[left], candidates[right])
        if relation is not None:
            arithmetic_pairs.append((left, right, relation))

    event_candidates: dict[int, dict] = {}
    if require_event_proof:
        if branch is None:
            raise ValueError("event-backed pre-refit grouping requires a branch")
        from .compiled_event_repeatability import (
            attach_event_repeatability_batch,
        )
        from .sealed_validation import (
            event_repeatability_metrics,
            prepare_event_repeatability_views,
        )

        involved = sorted(
            {index for left, right, _relation in arithmetic_pairs for index in (left, right)}
        )
        prepared_views = prepare_event_repeatability_views(branch)

        # Per-event medians/MADs are scientifically independent across
        # candidates.  Run the unchanged statistic in one compiled parallel
        # batch so the GPU search is not followed by a long Python thread-pool
        # sawtooth.  Minimal installations retain the exact reference path.
        compiled_candidates = [deepcopy(candidates[index]) for index in involved]
        compiled = attach_event_repeatability_batch(
            compiled_candidates,
            prepared_views=prepared_views,
            holdout_segment=None,
        )
        if compiled is not None:
            event_candidates.update(
                {index: candidate for index, candidate in zip(involved, compiled, strict=True)}
            )

        def attach(index: int) -> tuple[int, dict]:
            candidate = deepcopy(candidates[index])
            verification = deepcopy(candidate.get("holdout_verification") or {})
            repeatability = verification.get("event_repeatability")
            if not repeatability:
                repeatability = event_repeatability_metrics(
                    branch,
                    holdout_segment=None,
                    period_days=float(candidate["period_days"]),
                    epoch_btjd=float(candidate["epoch_btjd"]),
                    duration_hours=float(candidate["duration_hours"]),
                    prepared_views=prepared_views,
                )
                verification["event_repeatability"] = repeatability
            candidate["holdout_verification"] = verification
            return index, candidate

        if not event_candidates:
            event_workers = min(
                max(1, int(os.environ.get("HOUXING_PRE_REFIT_EVENT_THREADS", "8"))),
                max(1, len(involved)),
            )
            with ThreadPoolExecutor(max_workers=event_workers) as pool:
                for index, candidate in pool.map(attach, involved):
                    event_candidates[index] = candidate

    neighbours: list[list[tuple[int, dict]]] = [[] for _ in candidates]
    relations: list[dict] = []
    for left, right, relation in arithmetic_pairs:
        if require_event_proof:
            evidence = harmonic_families.harmonic_relation(
                event_candidates[left], event_candidates[right]
            )
            if evidence is None:
                continue
            relation = {
                **relation,
                "event_backed": True,
                "matched_supported_events": int(evidence["matched_supported_events"]),
                "matched_segments": list(evidence["matched_segments"]),
                "supported_event_containment": float(evidence["supported_event_containment"]),
                "family_computation_shared": True,
                "candidate_grid_membership_preserved": True,
            }
        relations.append(relation)
        neighbours[left].append(eligible[right])
        neighbours[right].append(eligible[left])

    item_position = {index: position for position, (index, _row) in enumerate(eligible)}
    unassigned = {index for index, _row in eligible}
    ordered = sorted(
        eligible,
        key=lambda item: _vote_quality(item[1]),
        reverse=True,
    )
    groups: list[list[tuple[int, dict]]] = []
    for anchor in ordered:
        anchor_index = anchor[0]
        if anchor_index not in unassigned:
            continue
        direct = [
            item
            for item in neighbours[item_position[anchor_index]]
            if item[0] in unassigned and item[0] != anchor_index
        ]
        direct.sort(key=lambda item: _vote_quality(item[1]), reverse=True)
        members = [anchor, *direct[: max(0, int(maximum_group_size) - 1)]]
        for index, _row in members:
            unassigned.discard(index)
        groups.append(members)
    return groups, relations


def _plan_exact_batched_refit_batches(
    harmonic_groups: list[list[tuple[int, dict]]],
    branch: core.SearchBranch,
    *,
    max_candidates: int,
) -> list[list[tuple[int, dict]]]:
    """Batch every original grid while preserving harmonic-first ordering."""

    max_candidates = max(1, int(max_candidates))
    ordered_items = [item for group in harmonic_groups for item in group]
    baseline = float(np.nanmax(branch.time) - np.nanmin(branch.time))
    by_duration: dict[float, list[tuple[int, dict]]] = defaultdict(list)
    for item in ordered_items:
        spec = optimization_v4._family_balance_spec(item[1], branch, baseline_days=baseline)
        duration = (
            float(spec["duration_days"])
            if spec is not None
            else float(item[1].get("duration_hours") or 0.0) / 24.0
        )
        by_duration[duration].append(item)

    planned: list[tuple[float, list[tuple[int, dict]]]] = []
    for duration_items in by_duration.values():
        for start in range(0, len(duration_items), max_candidates):
            batch = duration_items[start : start + max_candidates]
            cost = 0.0
            for _index, candidate in batch:
                spec = optimization_v4._family_balance_spec(
                    candidate, branch, baseline_days=baseline
                )
                cost += float(spec["grid_size"] if spec is not None else 0)
            planned.append((cost, batch))
    planned.sort(key=lambda row: row[0], reverse=True)
    return [batch for _cost, batch in planned]


def _execute_native_exact_inline(
    items: list[tuple[int, dict]],
    blocks: list[list[int]],
    branch: core.SearchBranch,
    *,
    compute_threads: int,
    completed_before: int,
    total_eligible: int | None,
    rescued_before: int,
    progress,
    preplanned_harmonic_groups: list[list[tuple[int, dict]]] | None,
) -> tuple[list[tuple[int, dict]], dict]:
    """Run one native thread pool in-process, avoiding spawn and data copies."""

    maximum = max(1, int(os.environ.get("HOUXING_EXACT_REFIT_BATCH_SIZE", "24")))
    harmonic_groups = preplanned_harmonic_groups
    if harmonic_groups is None:
        harmonic_groups, _relations = _plan_pre_refit_harmonic_groups(items)
    batches = _plan_exact_batched_refit_batches(
        harmonic_groups,
        branch,
        max_candidates=maximum,
    )
    _initialize_refit_worker(branch, blocks, compute_threads)
    total = int(total_eligible if total_eligible is not None else len(items))
    progress_interval = max(1, total // 50)
    next_progress = ((completed_before // progress_interval) + 1) * progress_interval
    completed = 0
    rescued = 0
    exact_original_coarse_points = 0
    exact_original_fine_points = 0
    exact_period_trials_skipped = 0
    native_exact_calls = 0
    native_exact_periods = 0
    native_exact_threads = 0
    native_exact_tile_size = 0
    native_exact_backend = None
    native_exact_simd = None
    native_exact_strategy = None
    native_exact_seconds = 0.0
    results: list[tuple[int, dict]] = []
    for batch in batches:
        completed_batch, exact_diagnostics = _native_exact_refit_worker_batch(batch)
        results.extend(completed_batch)
        exact_original_coarse_points += int(exact_diagnostics["exact_original_coarse_points"])
        exact_original_fine_points += int(exact_diagnostics["exact_original_fine_points"])
        exact_period_trials_skipped += int(exact_diagnostics["period_trials_skipped"])
        native = exact_diagnostics.get("native_exact") or {}
        native_exact_calls += int(native.get("calls") or 0)
        native_exact_periods += int(native.get("periods_evaluated") or 0)
        native_exact_threads = max(native_exact_threads, int(native.get("threads") or 0))
        native_exact_tile_size = max(native_exact_tile_size, int(native.get("tile_size") or 0))
        native_exact_backend = native.get("backend") or native_exact_backend
        native_exact_simd = native.get("simd") or native_exact_simd
        native_exact_strategy = native.get("strategy") or native_exact_strategy
        native_exact_seconds += float(native.get("seconds") or 0.0)
        for _index, candidate in completed_batch:
            rescued += int(candidate["ephemeris_refit"]["rescued_failed_fixed_ephemeris"])
        completed += len(completed_batch)
        global_completed = completed_before + completed
        if progress and (global_completed >= total or global_completed >= next_progress):
            progress(
                "refitting",
                f"Global period/epoch refit {global_completed}/{total}",
                {
                    "families_completed": global_completed,
                    "families_total": total,
                    "families_rescued": rescued_before + rescued,
                    "workers": 1,
                    "execution": "inline_native_exact_refit",
                    "compute_threads": compute_threads,
                    "threads_per_process": compute_threads,
                },
            )
            while next_progress <= global_completed:
                next_progress += progress_interval
    return results, {
        "attempted_families": len(items),
        "rescued_families": rescued,
        "workers": 1,
        "compute_threads": compute_threads,
        "threads_per_process": compute_threads,
        "shared_grid_individual_coarse_points": 0,
        "shared_grid_evaluated_coarse_points": 0,
        "shared_grid_exact_fallbacks": 0,
        "exact_original_coarse_points": exact_original_coarse_points,
        "exact_original_fine_points": exact_original_fine_points,
        "exact_period_trials_skipped": exact_period_trials_skipped,
        "native_exact_calls": native_exact_calls,
        "native_exact_periods_evaluated": native_exact_periods,
        "native_exact_threads": native_exact_threads,
        "native_exact_tile_size": native_exact_tile_size,
        "native_exact_backend": native_exact_backend,
        "native_exact_simd": native_exact_simd,
        "native_exact_strategy": native_exact_strategy,
        "native_exact_seconds": native_exact_seconds,
    }


def _execute_native_shared_inline(
    items: list[tuple[int, dict]],
    blocks: list[list[int]],
    branch: core.SearchBranch,
    *,
    compute_threads: int,
    completed_before: int,
    total_eligible: int | None,
    rescued_before: int,
    progress,
) -> tuple[list[tuple[int, dict]], dict]:
    """Run C++ exact-membership shared grids in one resident process."""

    maximum = max(
        1,
        int(os.environ.get("HOUXING_SHARED_REFIT_MAX_CANDIDATES", "64")),
    )
    _initialize_refit_worker(branch, blocks, compute_threads)
    total = int(total_eligible if total_eligible is not None else len(items))
    if progress:
        progress(
            "refitting",
            f"Super-batching exact shared grids for {len(items)} families",
            {
                "families_completed": completed_before,
                "families_total": total,
                "workers": 1,
                "execution": "inline_native_shared_duration_superbatch",
                "compute_threads": compute_threads,
                "threads_per_process": compute_threads,
            },
        )
    results, diagnostics = _native_shared_refit_worker_batch(
        items,
        max_candidates=maximum,
    )
    rescued = sum(
        int(candidate["ephemeris_refit"]["rescued_failed_fixed_ephemeris"])
        for _index, candidate in results
    )
    individual_coarse_points = int(diagnostics.get("individual_coarse_points") or 0)
    shared_coarse_points = int(diagnostics.get("shared_coarse_points") or 0)
    individual_fine_points = int(diagnostics.get("individual_fine_points") or 0)
    shared_fine_points = int(diagnostics.get("shared_fine_points") or 0)
    exact_fallbacks = int(diagnostics.get("exact_fallbacks") or 0)
    event_train_duplicate_fallbacks_skipped = int(
        diagnostics.get("event_train_duplicate_fallbacks_skipped") or 0
    )
    all_shared_failures_received_strict_exact_fallback = bool(
        diagnostics.get("all_shared_failures_received_strict_exact_fallback")
    )
    fallback_exact_coarse_points = int(
        diagnostics.get("fallback_exact_original_coarse_points") or 0
    )
    fallback_exact_fine_points = int(diagnostics.get("fallback_exact_original_fine_points") or 0)
    native = dict(diagnostics.get("native_exact") or {})
    if progress:
        progress(
            "refitting",
            f"Global period/epoch refit {completed_before + len(items)}/{total}",
            {
                "families_completed": completed_before + len(items),
                "families_total": total,
                "families_rescued": rescued_before + rescued,
                "workers": 1,
                "execution": "inline_native_shared_duration_superbatch",
                "compute_threads": compute_threads,
                "threads_per_process": compute_threads,
                "shared_coarse_point_reuse": (
                    individual_coarse_points / max(1, shared_coarse_points)
                ),
                "strict_exact_fallbacks": exact_fallbacks,
                "native_kernel_calls": int(native.get("calls") or 0),
            },
        )
    return results, {
        "attempted_families": len(items),
        "rescued_families": rescued,
        "workers": 1,
        "compute_threads": compute_threads,
        "threads_per_process": compute_threads,
        "shared_grid_individual_coarse_points": individual_coarse_points,
        "shared_grid_evaluated_coarse_points": shared_coarse_points,
        "shared_grid_individual_fine_points": individual_fine_points,
        "shared_grid_evaluated_fine_points": shared_fine_points,
        "shared_grid_exact_fallbacks": exact_fallbacks,
        "event_train_duplicate_fallbacks_skipped": (event_train_duplicate_fallbacks_skipped),
        "all_shared_failures_received_strict_exact_fallback": (
            all_shared_failures_received_strict_exact_fallback
        ),
        "exact_original_coarse_points": fallback_exact_coarse_points,
        "exact_original_fine_points": fallback_exact_fine_points,
        "exact_period_trials_skipped": int(native.get("period_trials_skipped") or 0),
        "native_exact_calls": int(native.get("calls") or 0),
        "native_exact_periods_evaluated": int(native.get("periods_evaluated") or 0),
        "native_exact_threads": int(native.get("threads") or 0),
        "native_exact_tile_size": int(native.get("tile_size") or 0),
        "native_exact_backend": native.get("backend"),
        "native_exact_simd": native.get("simd"),
        "native_exact_strategy": native.get("strategy"),
        "native_exact_seconds": float(native.get("seconds") or 0.0),
        "native_exact_strategy_calls": dict(native.get("strategy_calls") or {}),
        "native_exact_prefetch_requests": int(native.get("prefetch_requests") or 0),
        "native_exact_prefetch_superbatches": int(native.get("prefetch_superbatches") or 0),
        "native_exact_cache_hits": int(native.get("cache_hits") or 0),
    }


def _execute_native_selective_inline(
    items: list[tuple[int, dict]],
    blocks: list[list[int]],
    branch: core.SearchBranch,
    *,
    compute_threads: int,
    completed_before: int,
    total_eligible: int | None,
    rescued_before: int,
    progress,
) -> tuple[list[tuple[int, dict]], dict]:
    """Run the explosion-only shared screen and audited fallback budget."""

    maximum = max(
        1,
        int(os.environ.get("HOUXING_SHARED_REFIT_MAX_CANDIDATES", "64")),
    )
    _initialize_refit_worker(branch, blocks, compute_threads)
    total = int(total_eligible if total_eligible is not None else len(items))
    if progress:
        progress(
            "refitting",
            f"Adaptive explosion refit for {len(items)} families",
            {
                "families_completed": completed_before,
                "families_total": total,
                "workers": 1,
                "execution": "inline_native_adaptive_explosion_refit",
                "compute_threads": compute_threads,
                "threads_per_process": compute_threads,
            },
        )
    results, diagnostics = _native_selective_refit_worker_batch(
        items,
        max_candidates=maximum,
    )
    rescued = sum(
        int(candidate["ephemeris_refit"]["rescued_failed_fixed_ephemeris"])
        for _index, candidate in results
    )
    individual_coarse_points = int(diagnostics.get("individual_coarse_points") or 0)
    shared_coarse_points = int(diagnostics.get("shared_coarse_points") or 0)
    native = dict(diagnostics.get("native_exact") or {})
    if progress:
        progress(
            "refitting",
            f"Global period/epoch refit {completed_before + len(items)}/{total}",
            {
                "families_completed": completed_before + len(items),
                "families_total": total,
                "families_rescued": rescued_before + rescued,
                "workers": 1,
                "execution": "inline_native_adaptive_explosion_refit",
                "compute_threads": compute_threads,
                "threads_per_process": compute_threads,
                "shared_coarse_point_reuse": (
                    individual_coarse_points / max(1, shared_coarse_points)
                ),
                "strict_exact_fallbacks": int(diagnostics.get("strict_fallback_families") or 0),
                "deferred_families": int(diagnostics.get("deferred_families") or 0),
            },
        )
    return results, {
        "attempted_families": len(items),
        "rescued_families": rescued,
        "workers": 1,
        "compute_threads": compute_threads,
        "threads_per_process": compute_threads,
        "shared_grid_individual_coarse_points": individual_coarse_points,
        "shared_grid_evaluated_coarse_points": shared_coarse_points,
        "shared_grid_individual_fine_points": int(diagnostics.get("individual_fine_points") or 0),
        "shared_grid_evaluated_fine_points": int(diagnostics.get("shared_fine_points") or 0),
        "shared_grid_exact_fallbacks": int(diagnostics.get("strict_fallback_families") or 0),
        "adaptive_ranked_budget_families": int(diagnostics.get("ranked_budget_families") or 0),
        "adaptive_long_period_protected_families": int(
            diagnostics.get("long_period_protected_families") or 0
        ),
        "adaptive_deferred_families": int(diagnostics.get("deferred_families") or 0),
        "candidate_rows_deleted": False,
        "all_shared_failures_received_strict_exact_fallback": bool(
            diagnostics.get("all_shared_failures_received_strict_exact_fallback")
        ),
        "exact_original_coarse_points": int(
            diagnostics.get("fallback_exact_original_coarse_points") or 0
        ),
        "exact_original_fine_points": int(
            diagnostics.get("fallback_exact_original_fine_points") or 0
        ),
        "exact_period_trials_skipped": int(native.get("period_trials_skipped") or 0),
        "native_exact_calls": int(native.get("calls") or 0),
        "native_exact_periods_evaluated": int(native.get("periods_evaluated") or 0),
        "native_exact_threads": int(native.get("threads") or 0),
        "native_exact_tile_size": int(native.get("tile_size") or 0),
        "native_exact_backend": native.get("backend"),
        "native_exact_simd": native.get("simd"),
        "native_exact_strategy": native.get("strategy"),
        "native_exact_seconds": float(native.get("seconds") or 0.0),
        "native_exact_strategy_calls": dict(native.get("strategy_calls") or {}),
        "native_exact_prefetch_requests": int(native.get("prefetch_requests") or 0),
        "native_exact_prefetch_superbatches": int(native.get("prefetch_superbatches") or 0),
        "native_exact_cache_hits": int(native.get("cache_hits") or 0),
    }


def _execute_dense_refit_items(
    items: list[tuple[int, dict]],
    blocks: list[list[int]],
    branch: core.SearchBranch,
    *,
    workers: int,
    mode: str,
    completed_before: int = 0,
    total_eligible: int | None = None,
    rescued_before: int = 0,
    progress=None,
    preplanned_harmonic_groups: list[list[tuple[int, dict]]] | None = None,
) -> tuple[list[tuple[int, dict]], dict]:
    """Run the existing dense-grid implementation on an explicit item set."""

    if not items:
        return [], {
            "attempted_families": 0,
            "rescued_families": 0,
            "shared_grid_individual_coarse_points": 0,
            "shared_grid_evaluated_coarse_points": 0,
            "shared_grid_exact_fallbacks": 0,
        }
    compute_threads = max(1, min(int(workers), os.cpu_count() or 1))
    requested_native_processes = max(1, int(os.environ.get("HOUXING_NATIVE_EXACT_PROCESSES", "1")))
    if mode == "native-shared":
        return _execute_native_shared_inline(
            items,
            blocks,
            branch,
            compute_threads=compute_threads,
            completed_before=completed_before,
            total_eligible=total_eligible,
            rescued_before=rescued_before,
            progress=progress,
        )
    if mode == "native-selective":
        return _execute_native_selective_inline(
            items,
            blocks,
            branch,
            compute_threads=compute_threads,
            completed_before=completed_before,
            total_eligible=total_eligible,
            rescued_before=rescued_before,
            progress=progress,
        )
    if mode == "native-exact" and requested_native_processes == 1:
        return _execute_native_exact_inline(
            items,
            blocks,
            branch,
            compute_threads=compute_threads,
            completed_before=completed_before,
            total_eligible=total_eligible,
            rescued_before=rescued_before,
            progress=progress,
            preplanned_harmonic_groups=preplanned_harmonic_groups,
        )
    worker_count = max(1, min(compute_threads, len(items)))
    native_batches = None
    native_threads_per_process = 0
    if mode == "native-exact":
        maximum = max(1, int(os.environ.get("HOUXING_EXACT_REFIT_BATCH_SIZE", "24")))
        harmonic_groups = preplanned_harmonic_groups
        if harmonic_groups is None:
            harmonic_groups, _relations = _plan_pre_refit_harmonic_groups(items)
        native_batches = _plan_exact_batched_refit_batches(
            harmonic_groups,
            branch,
            max_candidates=maximum,
        )
        requested_processes = requested_native_processes
        # Independent duration batches can run in separate processes while
        # every process still tiles its complete, unchanged period grids.  The
        # total OpenMP thread budget never exceeds ``workers``.
        process_count = max(
            1,
            min(
                requested_processes,
                len(native_batches),
                compute_threads,
            ),
        )
        native_threads_per_process = max(1, compute_threads // process_count)
    else:
        process_count = worker_count
    context = multiprocessing.get_context("spawn")
    completed = 0
    rescued = 0
    total = int(total_eligible if total_eligible is not None else len(items))
    progress_interval = max(1, total // 50)
    next_progress = ((completed_before // progress_interval) + 1) * progress_interval
    shared_individual_points = 0
    shared_grid_points = 0
    exact_fallbacks = 0
    exact_original_coarse_points = 0
    exact_original_fine_points = 0
    exact_period_trials_skipped = 0
    native_exact_calls = 0
    native_exact_periods = 0
    native_exact_threads = 0
    native_exact_tile_size = 0
    native_exact_backend = None
    native_exact_simd = None
    native_exact_strategy = None
    native_exact_seconds = 0.0
    results: list[tuple[int, dict]] = []
    initializer_args = (
        (branch, blocks, native_threads_per_process) if mode == "native-exact" else (branch, blocks)
    )
    with ProcessPoolExecutor(
        max_workers=process_count,
        mp_context=context,
        initializer=_initialize_refit_worker,
        initargs=initializer_args,
    ) as pool:
        if mode in {"batched-exact", "native-exact"}:
            maximum = max(1, int(os.environ.get("HOUXING_EXACT_REFIT_BATCH_SIZE", "24")))
            if mode == "native-exact":
                batches = native_batches
            else:
                harmonic_groups = preplanned_harmonic_groups
                if harmonic_groups is None:
                    harmonic_groups, _relations = _plan_pre_refit_harmonic_groups(items)
                batches = _plan_exact_batched_refit_batches(
                    harmonic_groups,
                    branch,
                    max_candidates=maximum,
                )
            worker = (
                _native_exact_refit_worker_batch
                if mode == "native-exact"
                else _exact_batched_refit_worker_batch
            )
            futures = [pool.submit(worker, batch) for batch in batches]
        elif mode == "shared":
            maximum = max(1, int(os.environ.get("HOUXING_SHARED_REFIT_MAX_CANDIDATES", "24")))
            batches = _plan_shared_refit_batches(items, branch, max_candidates=maximum)
            futures = [pool.submit(_shared_refit_worker_batch, batch) for batch in batches]
        else:
            batches = [items[start : start + 4] for start in range(0, len(items), 4)]
            futures = [pool.submit(_refit_worker_batch, batch) for batch in batches]
        for future in as_completed(futures):
            future_result = future.result()
            if mode == "shared":
                completed_batch, shared_diagnostics = future_result
                shared_individual_points += int(shared_diagnostics["individual_coarse_points"])
                shared_grid_points += int(shared_diagnostics["shared_coarse_points"])
                exact_fallbacks += int(shared_diagnostics["exact_fallbacks"])
            elif mode in {"batched-exact", "native-exact"}:
                completed_batch, exact_diagnostics = future_result
                exact_original_coarse_points += int(
                    exact_diagnostics["exact_original_coarse_points"]
                )
                exact_original_fine_points += int(exact_diagnostics["exact_original_fine_points"])
                exact_period_trials_skipped += int(exact_diagnostics["period_trials_skipped"])
                if mode == "native-exact":
                    native = exact_diagnostics.get("native_exact") or {}
                    native_exact_calls += int(native.get("calls") or 0)
                    native_exact_periods += int(native.get("periods_evaluated") or 0)
                    native_exact_threads = max(
                        native_exact_threads, int(native.get("threads") or 0)
                    )
                    native_exact_tile_size = max(
                        native_exact_tile_size,
                        int(native.get("tile_size") or 0),
                    )
                    native_exact_backend = native.get("backend") or native_exact_backend
                    native_exact_simd = native.get("simd") or native_exact_simd
                    native_exact_strategy = native.get("strategy") or native_exact_strategy
                    native_exact_seconds += float(native.get("seconds") or 0.0)
            else:
                completed_batch = future_result
            results.extend(completed_batch)
            for _index, candidate in completed_batch:
                rescued += int(candidate["ephemeris_refit"]["rescued_failed_fixed_ephemeris"])
            completed += len(completed_batch)
            global_completed = completed_before + completed
            if progress and (global_completed >= total or global_completed >= next_progress):
                progress(
                    "refitting",
                    f"Global period/epoch refit {global_completed}/{total}",
                    {
                        "families_completed": global_completed,
                        "families_total": total,
                        "families_rescued": rescued_before + rescued,
                        "workers": worker_count,
                        "execution": (
                            "harmonic_scheduled_native_tiled_exact_refit"
                            if mode == "native-exact"
                            else "harmonic_scheduled_exact_grid_batch_refit"
                            if mode == "batched-exact"
                            else "harmonic_scheduled_process_parallel_shared_refit"
                            if mode == "shared"
                            else "harmonic_scheduled_process_parallel_exact_refit"
                        ),
                        "compute_threads": (
                            process_count * native_threads_per_process
                            if mode == "native-exact"
                            else worker_count
                        ),
                        "threads_per_process": (
                            native_threads_per_process if mode == "native-exact" else 1
                        ),
                    },
                )
                while next_progress <= global_completed:
                    next_progress += progress_interval
    return results, {
        "attempted_families": len(items),
        "rescued_families": rescued,
        "workers": process_count,
        "compute_threads": (
            process_count * native_threads_per_process if mode == "native-exact" else worker_count
        ),
        "threads_per_process": (native_threads_per_process if mode == "native-exact" else 1),
        "shared_grid_individual_coarse_points": shared_individual_points,
        "shared_grid_evaluated_coarse_points": shared_grid_points,
        "shared_grid_exact_fallbacks": exact_fallbacks,
        "exact_original_coarse_points": exact_original_coarse_points,
        "exact_original_fine_points": exact_original_fine_points,
        "exact_period_trials_skipped": exact_period_trials_skipped,
        "native_exact_calls": native_exact_calls,
        "native_exact_periods_evaluated": native_exact_periods,
        "native_exact_threads": native_exact_threads,
        "native_exact_tile_size": native_exact_tile_size,
        "native_exact_backend": native_exact_backend,
        "native_exact_simd": native_exact_simd,
        "native_exact_strategy": native_exact_strategy,
        "native_exact_seconds": native_exact_seconds,
    }


def refit_failed_vote_families(
    validated: list[dict],
    blocks: list[list[int]],
    branch: core.SearchBranch,
    *,
    workers: int = 10,
    mode: str = DEFAULT_REFIT_MODE,
    progress=None,
) -> tuple[list[dict], dict]:
    """Globally refit evidence-backed failures before the final fold decision.

    Targeted mode first builds the catalogue-blind precision-rescue board and
    only then performs any refit work.  This is important: the linear event-
    anchor solve is cheaper than a dense BLS grid, but applying it to every
    unresolved family still scales with a candidate explosion.  Deferred rows
    remain explicit and unresolved in the append-only audit.

    For the selected board, a cheap weighted linear event-anchor solve runs
    before the exact dense fallback.  A candidate leaves that fallback only
    after the proposed period/epoch passes the unchanged fixed-block, event-
    direction, and event-coherence checks.  Non-targeted modes retain their
    historical complete behavior.
    """

    eligible = [
        (index, row) for index, row in enumerate(validated) if ephemeris_refit_eligible(row)
    ]
    output = list(validated)
    if not eligible:
        return output, {
            "eligible_families": 0,
            "attempted_families": 0,
            "rescued_families": 0,
            "workers": 0,
        }

    if mode not in REFIT_MODES:
        raise ValueError(f"refit mode must be one of {REFIT_MODES!r}")
    requested_mode = mode
    adaptive_decision = None
    targeted_decision = None

    rescue_eligible = eligible
    if requested_mode == "targeted":
        rescue_eligible, targeted_decision = (
            _schedule_targeted_refits_before_compute(eligible, output)
        )
        mode = "native-exact"

    # Most failed fixed ephemerides are only slightly imprecise after being
    # measured on separated temporal blocks.  Solve the two-parameter linear
    # clock (epoch + cycle * period) from those event anchors before launching
    # thousands of dense BLS trials.  Work on deep copies so an unsuccessful
    # shortcut cannot contaminate the strict dense fallback.
    linear_candidates = [deepcopy(row) for _index, row in rescue_eligible]
    linear_diagnostics = optimization_v4.try_linear_event_anchor_rescue_batch(
        linear_candidates,
        blocks,
        branch,
    )
    linear_accepted_indices: set[int] = set()
    dense_eligible: list[tuple[int, dict]] = []
    fixed_block_linear_rescues = 0
    for (index, source), candidate in zip(rescue_eligible, linear_candidates, strict=False):
        if candidate.get("holdout_verification", {}).get("passed") is not True:
            dense_eligible.append((index, source))
            continue
        fixed_block_linear_rescues += 1
        completed_index, completed = _finish_refit_candidate(
            index,
            source,
            candidate,
            deepcopy(source.get("holdout_verification") or {}),
            execution="batched_weighted_linear_event_anchor_seed",
            branch=branch,
            blocks=blocks,
        )
        if completed.get("holdout_verification", {}).get("passed") is True:
            output[completed_index] = completed
            linear_accepted_indices.add(completed_index)
        else:
            # The cheap seed passed the fixed-block depth check but failed an
            # unchanged post-fit event check.  Refit the untouched source on
            # its complete dense grid; never use a failed shortcut as input.
            dense_eligible.append((index, source))

    dense_eligible_after_linear_prefit = len(dense_eligible)
    if requested_mode == "adaptive":
        adaptive_decision = _adaptive_refit_decision(dense_eligible, branch)
        mode = str(adaptive_decision["execution_mode"])

    harmonic_groups, scheduling_relations = _plan_pre_refit_harmonic_groups(
        dense_eligible,
        branch=branch,
        require_event_proof=True,
    )
    scheduled = [item for group in harmonic_groups for item in group]
    scheduled_indices = [index for index, _row in scheduled]
    if (
        len(scheduled) != len(dense_eligible)
        or len(set(scheduled_indices)) != len(dense_eligible)
        or set(scheduled_indices) != {index for index, _row in dense_eligible}
    ):
        raise RuntimeError("harmonic scheduler did not preserve every refit family")

    # Harmonics alter execution order only.  Every shortlisted family,
    # including every alias, retains its complete original coarse grid, fine
    # grid, epoch solve, three-block measurements, and event-direction check.
    if scheduled:
        dense_results, dense_diagnostics = _execute_dense_refit_items(
            scheduled,
            blocks,
            branch,
            workers=workers,
            mode=mode,
            completed_before=len(linear_accepted_indices),
            total_eligible=len(rescue_eligible),
            rescued_before=len(linear_accepted_indices),
            progress=progress,
            preplanned_harmonic_groups=harmonic_groups,
        )
    else:
        dense_results = []
        dense_diagnostics = {
            "attempted_families": 0,
            "rescued_families": 0,
            "workers": 0,
            "shared_grid_individual_coarse_points": 0,
            "shared_grid_evaluated_coarse_points": 0,
            "shared_grid_exact_fallbacks": 0,
            "exact_period_trials_skipped": 0,
        }
    for index, candidate in dense_results:
        output[index] = candidate

    dense_rescued = sum(
        candidate.get("holdout_verification", {}).get("passed") is True
        for _index, candidate in dense_results
    )
    shared_individual_points = int(
        dense_diagnostics.get("shared_grid_individual_coarse_points") or 0
    )
    shared_grid_points = int(dense_diagnostics.get("shared_grid_evaluated_coarse_points") or 0)
    exact_fallbacks = int(dense_diagnostics.get("shared_grid_exact_fallbacks") or 0)
    worker_count = int(dense_diagnostics.get("workers") or 0)
    return output, {
        "eligible_families": len(eligible),
        "attempted_families": len(rescue_eligible),
        "rescued_families": len(linear_accepted_indices) + dense_rescued,
        "workers": worker_count,
        "requested_mode": requested_mode,
        "resolved_mode": mode,
        "adaptive_decision": adaptive_decision,
        "targeted_rescue_decision": targeted_decision,
        "execution": (
            "post_ranking_targeted_precision_rescue_with_native_exact_grids"
            if requested_mode == "targeted"
            else "harmonic_first_ordering_plus_native_exact_grid_without_omission"
            if mode == "native-exact"
            else "harmonic_first_ordering_plus_native_adaptive_explosion_"
            "shared_grid_and_audited_selective_exact_fallback"
            if mode == "native-selective"
            else "harmonic_first_ordering_plus_native_exact_membership_shared_"
            "grid_with_strict_exact_fallback"
            if mode == "native-shared"
            else "harmonic_first_ordering_plus_exact_candidate_grid_batching_without_omission"
            if mode in {"batched-exact", "native-exact"}
            else "legacy_shared_grid_mode"
            if mode == "shared"
            else "legacy_scalar_exact_mode"
        ),
        "process_batch_size": (
            int(os.environ.get("HOUXING_EXACT_REFIT_BATCH_SIZE", "24"))
            if mode in {"batched-exact", "native-exact"}
            else int(
                os.environ.get(
                    "HOUXING_SHARED_REFIT_MAX_CANDIDATES",
                    "64" if mode in {"native-shared", "native-selective"} else "24",
                )
            )
            if mode in {"shared", "native-shared", "native-selective"}
            else 4
        ),
        "shared_grid_individual_coarse_points": (
            shared_individual_points
            if mode in {"shared", "native-shared", "native-selective"}
            else None
        ),
        "shared_grid_evaluated_coarse_points": (
            shared_grid_points if mode in {"shared", "native-shared", "native-selective"} else None
        ),
        "shared_grid_coarse_point_reuse": (
            shared_individual_points / max(1, shared_grid_points)
            if mode in {"shared", "native-shared", "native-selective"}
            else None
        ),
        "shared_grid_exact_fallbacks": (
            exact_fallbacks if mode in {"shared", "native-shared", "native-selective"} else None
        ),
        "event_train_duplicate_fallbacks_skipped": (
            int(dense_diagnostics.get("event_train_duplicate_fallbacks_skipped") or 0)
            if mode in {"native-shared", "native-selective"}
            else None
        ),
        "all_shared_failures_received_strict_exact_fallback": (
            bool(dense_diagnostics.get("all_shared_failures_received_strict_exact_fallback"))
            if mode in {"native-shared", "native-selective"}
            else None
        ),
        "adaptive_ranked_budget_families": (
            int(dense_diagnostics.get("adaptive_ranked_budget_families") or 0)
            if mode == "native-selective"
            else None
        ),
        "adaptive_long_period_protected_families": (
            int(dense_diagnostics.get("adaptive_long_period_protected_families") or 0)
            if mode == "native-selective"
            else None
        ),
        "adaptive_deferred_families": (
            int(dense_diagnostics.get("adaptive_deferred_families") or 0)
            if mode == "native-selective"
            else 0
        ),
        "targeted_shortlisted_families": (
            int(targeted_decision.get("selected_dense_refit_families") or 0)
            if targeted_decision
            else None
        ),
        "targeted_deferred_families": (
            int(targeted_decision.get("deferred_but_retained_families") or 0)
            if targeted_decision
            else 0
        ),
        "candidate_rows_deleted": False,
        "native_exact": (
            {
                "calls": int(dense_diagnostics.get("native_exact_calls") or 0),
                "periods_evaluated": int(
                    dense_diagnostics.get("native_exact_periods_evaluated") or 0
                ),
                "period_trials_skipped": int(
                    dense_diagnostics.get("exact_period_trials_skipped") or 0
                ),
                "threads": int(dense_diagnostics.get("native_exact_threads") or 0),
                "tile_size": int(dense_diagnostics.get("native_exact_tile_size") or 0),
                "backend": dense_diagnostics.get("native_exact_backend"),
                "simd": dense_diagnostics.get("native_exact_simd"),
                "strategy": dense_diagnostics.get("native_exact_strategy"),
                "seconds": float(dense_diagnostics.get("native_exact_seconds") or 0.0),
                "strategy_calls": dict(dense_diagnostics.get("native_exact_strategy_calls") or {}),
                "prefetch_requests": int(
                    dense_diagnostics.get("native_exact_prefetch_requests") or 0
                ),
                "prefetch_superbatches": int(
                    dense_diagnostics.get("native_exact_prefetch_superbatches") or 0
                ),
                "cache_hits": int(dense_diagnostics.get("native_exact_cache_hits") or 0),
            }
            if mode in {"native-exact", "native-shared", "native-selective"}
            else None
        ),
        "pre_refit_harmonic_scheduling": {
            "groups": len(harmonic_groups),
            "multi_member_groups": sum(len(group) > 1 for group in harmonic_groups),
            "scheduling_relations": len(scheduling_relations),
            "families_scheduled": len(scheduled),
            "families_skipped": 0,
            "aliases_skipped": 0,
            "candidate_removed_from_audit": False,
            "scheduling_changes_scientific_coverage": False,
        },
        "linear_event_anchor_prefit": {
            **linear_diagnostics,
            "input_eligible_families": len(eligible),
            "attempted_families": len(rescue_eligible),
            "fixed_block_rescues": fixed_block_linear_rescues,
            "accepted_after_event_checks": len(linear_accepted_indices),
            "postcheck_failures_sent_to_dense_grid": (
                fixed_block_linear_rescues - len(linear_accepted_indices)
            ),
            "dense_fallback_families": len(dense_eligible),
            "dense_fallback_families_after_targeted_board": (
                dense_eligible_after_linear_prefit
            ),
            "candidate_removed_without_validation": False,
            "failed_prefit_inputs_restored_from_original": True,
        },
        "dense_grid_attempted_families": len(dense_results),
        "exact_original_coarse_points": int(
            dense_diagnostics.get("exact_original_coarse_points") or 0
        ),
        "exact_original_fine_points": int(dense_diagnostics.get("exact_original_fine_points") or 0),
        "exact_period_trials_skipped": int(
            dense_diagnostics.get("exact_period_trials_skipped") or 0
        ),
        "selection_rule": {
            "cross_block_family": True,
            "single_block_thresholds_scale_from_reference": {
                "reference_groups": 9,
                "reference_rounds": 3,
                "min_positive_groups": EPHEMERIS_REFIT_MIN_POSITIVE_GROUPS,
                "min_strong_groups": EPHEMERIS_REFIT_MIN_STRONG_GROUPS,
                "min_positive_rounds": 2,
            },
        },
        "relative_period_half_width": EPHEMERIS_REFIT_RELATIVE_HALF_WIDTH,
        "used_catalog_period": False,
    }


def _compact_stable_family(row: Mapping) -> dict:
    """Keep every stable family in result.json without duplicating its audit."""

    vote = dict(row.get("ensemble_vote") or {})
    vote.pop("measurements", None)
    long_vote = dict(row.get("long_period_vote") or {})
    long_vote.pop("measurements", None)
    verification = dict(row.get("holdout_verification") or {})
    return {
        "proposal_id": row.get("proposal_id"),
        "representative_raw_id": row.get("raw_id"),
        "period_days": row.get("period_days"),
        "epoch_btjd": row.get("epoch_btjd"),
        "duration_hours": row.get("duration_hours"),
        "method": row.get("method"),
        "source_methods": row.get("source_methods"),
        "source_strength": row.get("source_strength"),
        "robust_sde": row.get("robust_sde"),
        "ensemble_score": row.get("ensemble_score"),
        "vote_seed_consensus": row.get("vote_seed_consensus"),
        "ensemble_vote": vote,
        "long_period_vote": long_vote or None,
        "ensemble": row.get("ensemble"),
        "holdout_verification": verification,
        "ephemeris_balance": row.get("ephemeris_balance"),
        "ephemeris_refit": row.get("ephemeris_refit"),
        "event_coherence": row.get("event_coherence"),
    }


def run_vote_intersection(
    lightcurve_paths: Iterable[Path],
    *,
    output_dir: Path,
    period_max: float = 400.0,
    vote_rounds: int = 5,
    search_workers: int = 3,
    refit_workers: int = min(10, os.cpu_count() or 1),
    refit_mode: str = DEFAULT_REFIT_MODE,
    seed: int = 307210830,
    vote_seed_count: int = DEFAULT_VOTE_SEED_COUNT,
    progress=None,
    search_block_worker=None,
    search_backend: str = "cpu-exact",
    vote_scorer: Callable | None = None,
    proposal_validator: Callable | None = None,
    refit_runner: Callable | None = None,
    candidate_family_mode: str = "core",
    search_scheduler: Callable | None = None,
    audit_codec: str = "gzip",
) -> dict:
    """Search local blocks once, then score every peak on many sector groups."""

    started = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if audit_codec not in {"gzip", "zstd"}:
        raise ValueError("audit_codec must be 'gzip' or 'zstd'")

    def write_audit(stem: str, payload) -> str:
        if audit_codec == "gzip":
            name = f"{stem}.json.gz"
            _write_gzip_json(output_dir / name, payload)
            return name
        try:
            import orjson
            import zstandard
        except ImportError as exc:
            raise RuntimeError("zstd audit output requires orjson and zstandard") from exc
        name = f"{stem}.json.zst"
        serialized = orjson.dumps(payload)
        compressor = zstandard.ZstdCompressor(level=1, threads=-1)
        (output_dir / name).write_bytes(compressor.compress(serialized))
        return name

    paths = [Path(path) for path in lightcurve_paths]
    metadata = [core._segment_metadata(path) for path in paths]
    paired = sorted(zip(metadata, paths, strict=False), key=lambda row: row[0]["segment"])
    metadata = [row[0] for row in paired]
    paths = [row[1] for row in paired]
    segments = sorted({int(row["segment"]) for row in metadata})
    branch, load_diagnostics = core._load_pdcsap_branch(paths)
    base_blocks = select_temporal_blocks(segments)
    vote_seed_values = deterministic_vote_seeds(seed, vote_seed_count)
    vote_seed_geometries: list[dict] = []
    for vote_seed in vote_seed_values:
        seed_vote_groups, seed_vote_plan = build_natural_sector_vote_groups(
            segments,
            rounds=vote_rounds,
            preferred_groups_per_round=3,
            seed=vote_seed,
        )
        seed_long_groups, seed_long_plan = build_natural_sector_vote_groups(
            segments,
            rounds=vote_rounds,
            preferred_groups_per_round=2,
            seed=vote_seed + 1_000_003,
        )
        vote_seed_geometries.append(
            {
                "seed": int(vote_seed),
                "vote_groups": seed_vote_groups,
                "vote_plan": seed_vote_plan,
                "long_period_vote_groups": seed_long_groups,
                "long_period_vote_plan": seed_long_plan,
            }
        )
    vote_groups = vote_seed_geometries[0]["vote_groups"]
    vote_plan = vote_seed_geometries[0]["vote_plan"]
    long_period_vote_groups = vote_seed_geometries[0]["long_period_vote_groups"]
    long_period_vote_plan = vote_seed_geometries[0]["long_period_vote_plan"]
    _write_json(
        output_dir / "group_plan.json",
        {
            "base_search_blocks": base_blocks,
            "vote_groups": vote_groups,
            "vote_plan": vote_plan,
            "long_period_vote_groups": long_period_vote_groups,
            "long_period_vote_plan": long_period_vote_plan,
            "vote_seed_count": len(vote_seed_geometries),
            "vote_seed_values": vote_seed_values,
            "vote_seed_geometries": vote_seed_geometries,
        },
    )

    def emit(stage: str, message: str, metrics: dict) -> None:
        if progress:
            progress(stage, message, metrics)

    emit(
        "searching",
        "Searching three local temporal blocks with the uncapped Core kernel",
        {
            "base_blocks": base_blocks,
            "candidate_cap": None,
            "all_local_peaks_retained": True,
        },
    )
    if search_scheduler is not None:
        ledger, search_diagnostics = search_scheduler(
            branch, base_blocks, float(period_max), emit=emit
        )
    else:
        completed: list[tuple[int, list[dict], dict]] = []
        worker = search_block_worker or _search_group_worker
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max(1, min(int(search_workers), 3)), mp_context=context
        ) as pool:
            futures = {
                pool.submit(
                    worker,
                    branch,
                    block_index,
                    list(block),
                    float(period_max),
                ): block_index
                for block_index, block in enumerate(base_blocks)
            }
            for future in as_completed(futures):
                block_index = futures[future]
                rows, diagnostics = future.result()
                completed.append((block_index, rows, diagnostics))
                emit(
                    "searching",
                    f"Base block {block_index + 1}/3 complete",
                    {"block": block_index + 1, "raw_hypotheses": len(rows)},
                )
        ledger = []
        search_diagnostics = []
        for _block, rows, diagnostics in sorted(completed):
            ledger.extend(rows)
            search_diagnostics.append(diagnostics)
    search_seconds = time.monotonic() - started

    optimization_v4.install()
    emit(
        "voting",
        "Scoring every local peak across deterministic sector resamplings",
        {
            "raw_hypotheses": len(ledger),
            "vote_groups_per_seed": len(vote_groups),
            "vote_seed_count": len(vote_seed_geometries),
        },
    )
    score_function = vote_scorer or score_ledger_on_vote_groups
    scored = score_ledger_on_vote_seed_ensemble(
        ledger,
        branch,
        vote_seed_geometries,
        scorer=score_function,
        progress=progress,
    )
    vote_seconds = time.monotonic() - started - search_seconds
    scored_audit_file = write_audit("scored_raw_ledger", scored)

    emit(
        "matching",
        "Forming exact period+phase families from the fully scored ledger",
        {"raw_hypotheses": len(scored)},
    )
    family_baseline_days = float(np.nanmax(branch.time) - np.nanmin(branch.time))
    promoted, family_audit, match_diagnostics = build_vote_consensus_families(
        scored,
        family_mode=candidate_family_mode,
        baseline_days=family_baseline_days,
    )
    matching_seconds = time.monotonic() - started - search_seconds - vote_seconds
    family_audit_file = write_audit("family_audit", family_audit)

    emit(
        "validating",
        "Initial fixed-ephemeris validation of evidence-promoted families",
        {"promoted_families": len(promoted), "blocks": base_blocks},
    )
    validation_started = time.monotonic()
    validate_function = proposal_validator or optimization_v4.validate_proposals_vectorized
    validated = validate_function(promoted, base_blocks, branch)
    initial_stable_count = sum(
        row["holdout_verification"].get("passed") is True for row in validated
    )
    validation_seconds = time.monotonic() - validation_started

    refit_eligible_rows = [row for row in validated if ephemeris_refit_eligible(row)]
    refit_baseline_days = float(np.nanmax(branch.time) - np.nanmin(branch.time))
    estimated_refit_coarse_points = 0
    estimated_refit_fine_points = 0
    for row in refit_eligible_rows:
        spec = optimization_v4._family_balance_spec(
            row,
            branch,
            baseline_days=refit_baseline_days,
        )
        if spec is None:
            continue
        estimated_refit_coarse_points += int(spec["grid_size"])
        estimated_refit_fine_points += 101
    refit_workload_estimate = {
        "eligible_families": len(refit_eligible_rows),
        "coarse_period_trials": estimated_refit_coarse_points,
        "fine_period_trials": estimated_refit_fine_points,
        "total_period_trials": (estimated_refit_coarse_points + estimated_refit_fine_points),
        "estimate_is_pre_linear_rescue_upper_bound": True,
        "candidate_rows_deleted": False,
    }

    emit(
        "refitting",
        "Globally refitting evidence-backed period/epoch failures",
        {
            "initial_stable_families": initial_stable_count,
            "promoted_families": len(promoted),
            "refit_eligible_families": len(refit_eligible_rows),
            "estimated_exact_period_trials": refit_workload_estimate["total_period_trials"],
            "used_catalog_period": False,
        },
    )
    refit_started = time.monotonic()
    if refit_runner is None:
        validated, refit_diagnostics = refit_failed_vote_families(
            validated,
            base_blocks,
            branch,
            workers=refit_workers,
            mode=refit_mode,
            progress=progress,
        )
    else:
        validated, refit_diagnostics = refit_runner(
            validated,
            base_blocks,
            branch,
            workers=refit_workers,
            mode=refit_mode,
            progress=progress,
        )
    refit_seconds = time.monotonic() - refit_started
    ephemerides_changed = refit_diagnostics.get("ephemerides_changed")
    revote_items, revote_audit_only, revote_diagnostics = _plan_post_refit_revote(
        validated, refit_mode=refit_mode
    )
    refit_diagnostics = dict(refit_diagnostics)
    refit_diagnostics["post_refit_revote"] = revote_diagnostics
    if ephemerides_changed is False:
        emit(
            "revoting",
            "Re-vote skipped because no ephemeris changed",
            {
                "families": len(validated),
                "scientific_result_changed": False,
            },
        )
        revote_seconds = 0.0
    elif not revote_items:
        emit(
            "revoting",
            "Re-vote skipped because no family passed locked holdout validation",
            revote_diagnostics,
        )
        revote_seconds = 0.0
    else:
        emit(
            "revoting",
            (
                "Re-scoring holdout-passing refined ephemerides across every "
                "deterministic vote seed"
                if refit_mode == "targeted"
                else "Re-scoring refined ephemerides across every deterministic vote seed"
            ),
            {
                "families": len(revote_items),
                "audit_only_families": len(revote_audit_only),
                "vote_groups_per_seed": len(vote_groups),
                "vote_seed_count": len(vote_seed_geometries),
                "candidate_rows_deleted": False,
            },
        )
        revote_started = time.monotonic()
        if refit_mode == "targeted":
            validated, cached_revote_diagnostics = (
                rescore_changed_ephemerides_with_cached_population(
                    validated,
                    [index for index, _row in revote_items],
                    branch,
                    vote_seed_geometries,
                    scorer=score_function,
                    progress=progress,
                )
            )
            refit_diagnostics["cached_population_revote"] = cached_revote_diagnostics
        else:
            validated = score_ledger_on_vote_seed_ensemble(
                validated,
                branch,
                vote_seed_geometries,
                scorer=score_function,
                progress=progress,
            )
        revote_seconds = time.monotonic() - revote_started
    stable = [row for row in validated if row["holdout_verification"].get("passed") is True]
    for row in stable:
        vote = row["ensemble_vote"]
        measurements = row["holdout_verification"]["block_measurements"]
        positive_snrs = [
            float(item.get("snr") or 0.0)
            for item in measurements
            if item["outcome"] in {"strong", "weak"}
        ]
        base_score = (
            100.0 * int(vote["positive_round_count"])
            + 20.0 * int(vote["positive_groups"])
            + 10.0 * int(vote["strong_groups"])
            - 10.0 * int(vote["contradicting_groups"])
            + min(25.0, min(positive_snrs) if positive_snrs else 0.0)
        )
        stability_score = float(
            (row.get("vote_seed_consensus") or {}).get("stability_score") or 0.0
        )
        row["ensemble_score"] = round(base_score + stability_score, 6)
    stable.sort(
        key=lambda row: (
            float(row["ensemble_score"]),
            *_vote_quality(row),
        ),
        reverse=True,
    )
    validated_audit_file = write_audit("validated_families", validated)
    result = {
        "schema_version": "vlora-vote-intersection-experiment-v3",
        "engine_version": ("vlora-core-multigroup-vote-v3-global-ephemeris-refit"),
        "target": metadata[0].get("object") or "TIC 307210830",
        "tic_id": metadata[0].get("tic_id"),
        "period_max_days": float(period_max),
        "candidate_cap": None,
        "all_local_peaks_retained": True,
        "candidate_family_mode": candidate_family_mode,
        "refit_mode": refit_mode,
        "search_backend": str(search_backend),
        "segments": segments,
        "base_search_blocks": base_blocks,
        "vote_groups": vote_groups,
        "vote_plan": vote_plan,
        "long_period_vote_groups": long_period_vote_groups,
        "long_period_vote_plan": long_period_vote_plan,
        "vote_seed_count": len(vote_seed_geometries),
        "vote_seed_values": vote_seed_values,
        "vote_seed_geometries": vote_seed_geometries,
        "raw_hypothesis_count": len(ledger),
        "promoted_family_count": len(promoted),
        "initial_stable_family_count": initial_stable_count,
        "stable_family_count": len(stable),
        "refit_workload_estimate": refit_workload_estimate,
        "ephemeris_refit_diagnostics": refit_diagnostics,
        "post_refit_revote_diagnostics": revote_diagnostics,
        "match_diagnostics": match_diagnostics,
        "search_diagnostics": search_diagnostics,
        "load_diagnostics": load_diagnostics,
        "timing_seconds": {
            "search": round(search_seconds, 3),
            "vectorized_multigroup_voting": round(vote_seconds, 3),
            "matching_and_family_reduction": round(matching_seconds, 3),
            "initial_vectorized_global_validation": round(validation_seconds, 3),
            "parallel_global_period_epoch_refit": round(refit_seconds, 3),
            "post_refit_vectorized_revoting": round(revote_seconds, 3),
            "total": round(time.monotonic() - started, 3),
        },
        # No family is omitted here.  Per-peak/per-vote details live in the
        # compressed audit files so the human-readable summary stays usable.
        "stable_families": [_compact_stable_family(row) for row in stable],
        "audit_files": {
            "codec": audit_codec,
            "scored_raw_ledger": scored_audit_file,
            "family_audit": family_audit_file,
            "validated_families": validated_audit_file,
        },
        "scientific_status": (
            "upgrade experiment; all local peaks received multi-group votes and "
            "deterministic cross-seed stability scores; "
            "evidence-backed fixed-ephemeris failures received the configured "
            "catalog-blind period/epoch rescue before locked validation; "
            "any explosion-budget deferrals remain unresolved in the audit; red-noise, "
            "physical, and pixel vetting are not yet applied"
        ),
    }
    _write_json(output_dir / "result.json", result)
    emit(
        "complete",
        "Multi-group vote-intersection experiment complete",
        {
            "raw_hypotheses": len(ledger),
            "promoted_families": len(promoted),
            "stable_families": len(stable),
            "timing_seconds": result["timing_seconds"],
        },
    )
    return result


__all__ = [
    "DEFAULT_REFIT_MODE",
    "DEFAULT_VOTE_SEED_COUNT",
    "FAMILY_MODES",
    "REFIT_MODES",
    "TARGETED_REFIT_MAX_FAMILIES",
    "build_balanced_discovery_groups",
    "build_natural_sector_vote_groups",
    "build_consensus_families",
    "run_ensemble_intersection",
    "score_ledger_on_vote_groups",
    "score_ledger_on_vote_seed_ensemble",
    "deterministic_vote_seeds",
    "build_vote_consensus_families",
    "ephemeris_refit_eligible",
    "refit_failed_vote_families",
    "run_vote_intersection",
]
