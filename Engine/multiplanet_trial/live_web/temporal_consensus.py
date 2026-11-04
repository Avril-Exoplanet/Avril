"""Three-block temporal consensus without destructive candidate pruning.

Two primary time blocks are searched independently.  A third block is an
independent arbiter for hypotheses found in only one primary block.  Candidate
identity includes period *and* phase; every raw fold hypothesis remains in the
audit ledger even when it does not enter the ranked consensus set.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np


def select_temporal_blocks(segments: list[int]) -> list[list[int]]:
    """Split ordered observing segments into three disjoint, stable blocks."""
    ordered = sorted({int(value) for value in segments})
    if len(ordered) < 3:
        return []
    quotient, remainder = divmod(len(ordered), 3)
    sizes = [quotient + (index < remainder) for index in range(3)]
    blocks: list[list[int]] = []
    cursor = 0
    for size in sizes:
        blocks.append(ordered[cursor : cursor + size])
        cursor += size
    return blocks


def _finite_positive(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def _period_uncertainty(candidate: Mapping) -> float:
    period = float(candidate["period_days"])
    duration_days = float(candidate.get("duration_hours") or 2.0) / 24.0
    verification = candidate.get("holdout_verification", {})
    reported = _finite_positive(verification.get("period_uncertainty_days"))
    if reported is not None:
        return reported
    # The fallback is deliberately based on transit duration rather than a
    # fixed percentage.  Three observed cycles constrain a period to roughly
    # one third of a transit duration; retain the same 0.005--0.3% safety bounds
    # used by sealed validation.
    return float(np.clip(duration_days / 3.0, period * 5e-5, period * 0.003))


def _wrapped_phase_error(delta_days: float, period_days: float) -> float:
    return abs(
        (delta_days + 0.5 * period_days) % period_days - 0.5 * period_days
    )


def candidate_match_details(first: Mapping, second: Mapping) -> dict | None:
    """Return a phase-aware match proof, or ``None`` for distinct hypotheses.

    Both period estimates must overlap within their discovery-derived
    uncertainty and each ephemeris must predict the other block's epoch within
    a duration-scaled tolerance.  This prevents a deeper, wrong phase at almost
    the same period from erasing the real event train.
    """
    period_a = float(first["period_days"])
    period_b = float(second["period_days"])
    if period_a <= 0 or period_b <= 0:
        return None
    uncertainty_a = _period_uncertainty(first)
    uncertainty_b = _period_uncertainty(second)
    period_delta = abs(period_a - period_b)
    if period_delta > uncertainty_a + uncertainty_b:
        return None

    epoch_a = float(first["epoch_btjd"])
    epoch_b = float(second["epoch_btjd"])
    epoch_delta = epoch_b - epoch_a
    phase_error_a = _wrapped_phase_error(epoch_delta, period_a)
    phase_error_b = _wrapped_phase_error(epoch_delta, period_b)
    duration_days = max(
        float(first.get("duration_hours") or 2.0),
        float(second.get("duration_hours") or 2.0),
    ) / 24.0
    phase_tolerance = min(
        0.15 * 0.5 * (period_a + period_b),
        max(2.5 * duration_days, 0.005 * 0.5 * (period_a + period_b)),
    )
    cycle_count = int(round(abs(epoch_delta) / (0.5 * (period_a + period_b))))
    connected_period = (
        abs(epoch_delta) / cycle_count if cycle_count > 0 else 0.5 * (period_a + period_b)
    )
    # Widely separated blocks have an integer-cycle ambiguity: each block can
    # estimate a slightly different local period even though their event trains
    # connect exactly at one period between them.  Accept that connection only
    # when the bridge period lies inside *both* predeclared uncertainty windows.
    # Nearby epochs have no such lever arm and must agree directly in phase.
    bridge_inside_both = (
        cycle_count > 0
        and abs(connected_period - period_a) <= uncertainty_a
        and abs(connected_period - period_b) <= uncertainty_b
    )
    direct_phase_match = max(phase_error_a, phase_error_b) <= phase_tolerance
    if not (bridge_inside_both or direct_phase_match):
        return None
    return {
        "period_delta_days": period_delta,
        "period_uncertainty_a_days": uncertainty_a,
        "period_uncertainty_b_days": uncertainty_b,
        "phase_error_a_days": phase_error_a,
        "phase_error_b_days": phase_error_b,
        "phase_tolerance_days": phase_tolerance,
        "integer_cycles_between_epochs": cycle_count,
        "cycle_connected_period_days": connected_period,
        "cycle_connected_within_both_uncertainties": bridge_inside_both,
        "identity_fields": ["period", "epoch_phase", "event_train"],
    }


def _outcome(candidate: Mapping) -> str:
    verification = candidate.get("holdout_verification", {})
    if verification.get("passed") is True:
        return "passed"
    if verification.get("outcome") in {"inconclusive", "untestable"}:
        return str(verification["outcome"])
    return "failed"


def _quality(candidate: Mapping) -> tuple:
    verification = candidate.get("holdout_verification", {})
    priority = {"failed": 0, "untestable": 1, "inconclusive": 2, "passed": 3}
    search_snr = _finite_positive(verification.get("search_metrics", {}).get("snr")) or 0.0
    sealed_snr = _finite_positive(verification.get("sealed_metrics", {}).get("snr")) or 0.0
    return (
        priority[_outcome(candidate)],
        min(search_snr, sealed_snr),
        float(candidate.get("candidate_score") or 0.0),
        search_snr,
    )


def _candidate_audit(candidate: Mapping, fold_index: int, candidate_index: int) -> dict:
    return {
        "source_temporal_fold": fold_index + 1,
        "source_candidate_index": candidate_index,
        "candidate_id": candidate.get("candidate_id"),
        "period_days": float(candidate["period_days"]),
        "epoch_btjd": float(candidate["epoch_btjd"]),
        "duration_hours": float(candidate.get("duration_hours") or 0.0),
        "candidate_score": float(candidate.get("candidate_score") or 0.0),
        "fold_outcome": _outcome(candidate),
        "holdout_verification": dict(candidate.get("holdout_verification", {})),
    }


def merge_temporal_consensus_results(
    results: list[dict], *, min_sealed_snr: float = 6.0
) -> dict:
    """Merge three independent searches with a two-of-three discovery rule."""
    if len(results) != 3:
        raise ValueError("temporal consensus requires exactly three fold results")
    pools = [
        list(result.get("candidate_registry") or result.get("candidates", []))
        for result in results
    ]
    nodes = [
        (fold_index, candidate_index)
        for fold_index, pool in enumerate(pools)
        for candidate_index in range(len(pool))
    ]
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    match_proofs: dict[tuple[tuple[int, int], tuple[int, int]], dict] = {}
    for left_fold in range(3):
        for right_fold in range(left_fold + 1, 3):
            for left_index, left in enumerate(pools[left_fold]):
                for right_index, right in enumerate(pools[right_fold]):
                    proof = candidate_match_details(left, right)
                    if proof is None:
                        continue
                    left_node = (left_fold, left_index)
                    right_node = (right_fold, right_index)
                    adjacency[left_node].add(right_node)
                    adjacency[right_node].add(left_node)
                    match_proofs[(left_node, right_node)] = proof

    components: list[list[tuple[int, int]]] = []
    unseen = set(nodes)
    while unseen:
        start = unseen.pop()
        component = [start]
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in adjacency.get(current, set()):
                if neighbor not in unseen:
                    continue
                unseen.remove(neighbor)
                component.append(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))

    ranked: list[dict] = []
    retained: list[dict] = []
    rejected: list[dict] = []
    unmatched: list[dict] = []
    all_audit = [
        _candidate_audit(candidate, fold_index, candidate_index)
        for fold_index, pool in enumerate(pools)
        for candidate_index, candidate in enumerate(pool)
    ]

    for component in components:
        fold_nodes: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for node in component:
            fold_nodes[node[0]].append(node)
        if len(fold_nodes) < 2:
            fold_index, candidate_index = component[0]
            unmatched.append(
                {
                    **dict(pools[fold_index][candidate_index]),
                    "rank": None,
                    "display_rank": None,
                    "registry_state": "temporal_unmatched",
                    "source_temporal_fold": fold_index + 1,
                    "temporal_exclusion_reason": "not_reproduced_in_either_other_block",
                }
            )
            continue

        chosen_nodes = [
            max(fold_members, key=lambda node: _quality(pools[node[0]][node[1]]))
            for _fold, fold_members in sorted(fold_nodes.items())
        ]
        members = [pools[fold][index] for fold, index in chosen_nodes]
        representative = max(members, key=_quality)
        candidate = dict(representative)
        outcomes = [_outcome(member) for member in members]
        weak_members = [
            member
            for member, outcome in zip(members, outcomes)
            if outcome == "inconclusive"
            and member.get("holdout_verification", {}).get(
                "weak_support_eligible_for_combination", True
            )
        ]
        weak_snrs = [
            _finite_positive(
                member.get("holdout_verification", {})
                .get("sealed_metrics", {})
                .get("snr")
            )
            or 0.0
            for member in weak_members
        ]
        combined_weak_snr = float(np.sqrt(sum(value**2 for value in weak_snrs)))
        weak_passed = len(weak_members) >= 2 and combined_weak_snr >= min_sealed_snr
        folds_failed = outcomes.count("failed")
        consensus_passed = folds_failed == 0 and (
            outcomes.count("passed") >= 1 or weak_passed
        )
        consensus_inconclusive = (
            folds_failed == 0
            and not consensus_passed
            and outcomes.count("inconclusive") >= 1
        )

        component_edges = []
        for edge, proof in match_proofs.items():
            if edge[0] in component and edge[1] in component:
                component_edges.append(
                    {
                        "left_fold": edge[0][0] + 1,
                        "left_candidate_index": edge[0][1],
                        "right_fold": edge[1][0] + 1,
                        "right_candidate_index": edge[1][1],
                        **proof,
                    }
                )
        fold_evidence = [
            {
                "fold": fold + 1,
                "discovery_segments": list(results[fold].get("search_segments", [])),
                "sealed_segments": list(results[fold].get("holdout_segments", [])),
                "period_days": float(member["period_days"]),
                "epoch_btjd": float(member["epoch_btjd"]),
                "duration_hours": float(member.get("duration_hours") or 0.0),
                "holdout_verification": dict(member.get("holdout_verification", {})),
            }
            for (fold, _index), member in zip(chosen_nodes, members)
        ]
        candidate["candidate_score"] = min(
            float(member.get("candidate_score") or 0.0) for member in members
        )
        candidate["temporal_consensus"] = {
            "passed": consensus_passed,
            "discovery_folds_required": 2,
            "discovery_folds_matched": len(fold_nodes),
            "matched_fold_numbers": [fold + 1 for fold in sorted(fold_nodes)],
            "primary_fold_intersection": 0 in fold_nodes and 1 in fold_nodes,
            "arbiter_used": 2 in fold_nodes and not (0 in fold_nodes and 1 in fold_nodes),
            "folds_passed": outcomes.count("passed"),
            "folds_inconclusive": outcomes.count("inconclusive"),
            "folds_untestable": outcomes.count("untestable"),
            "folds_failed": folds_failed,
            "combined_weak_sealed_snr": combined_weak_snr,
            "weak_consensus_passed": weak_passed,
            "fold_evidence": fold_evidence,
            "match_proofs": component_edges,
            "all_component_hypotheses": [
                _candidate_audit(pools[fold][index], fold, index)
                for fold, index in component
            ],
            "truth_catalog_used": False,
        }
        proof = dict(candidate.get("holdout_verification", {}))
        proof["passed"] = consensus_passed
        proof["outcome"] = (
            "passed"
            if consensus_passed
            else ("inconclusive" if consensus_inconclusive else "failed")
        )
        proof["status"] = (
            "temporal_two_of_three_passed"
            if consensus_passed
            else (
                "temporal_consensus_inconclusive"
                if consensus_inconclusive
                else "temporal_consensus_failed"
            )
        )
        proof["cross_validation_folds"] = fold_evidence
        candidate["holdout_verification"] = proof
        candidate["alternate_periods"] = []
        if consensus_passed:
            candidate["registry_state"] = "temporal_validated"
            candidate["conclusion"] = "two_of_three_temporal_consensus"
            ranked.append(candidate)
        elif consensus_inconclusive:
            candidate["registry_state"] = "temporal_inconclusive"
            retained.append(candidate)
        else:
            candidate["registry_state"] = "temporal_failed"
            rejected.append(candidate)

    ranked.sort(key=lambda row: float(row.get("candidate_score") or 0.0), reverse=True)
    retained.sort(key=lambda row: float(row.get("candidate_score") or 0.0), reverse=True)
    rejected.sort(key=lambda row: float(row.get("candidate_score") or 0.0), reverse=True)
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
        candidate["display_rank"] = rank
    registry = [*ranked, *retained, *rejected, *unmatched]
    for registry_rank, candidate in enumerate(registry, start=1):
        candidate["rank"] = registry_rank
        if candidate.get("registry_state") != "temporal_validated":
            candidate["display_rank"] = None

    output = dict(results[0])
    output["schema_version"] = "houxing-live-result-3-temporal-consensus"
    output["engine_version"] = f"{output.get('engine_version', 'unknown')}+temporal-2of3"
    output["search_segments"] = sorted(
        {segment for result in results for segment in result.get("search_segments", [])}
    )
    output["holdout_segment"] = None
    output["holdout_segments"] = []
    output["candidates"] = ranked
    output["untestable_candidates"] = retained
    output["candidate_registry"] = registry
    output["candidate_pool_total"] = len(ranked)
    output["temporal_candidate_instances"] = all_audit
    output["temporal_consensus"] = {
        "folds_run": 3,
        "blocks": [list(result.get("search_segments", [])) for result in results],
        "primary_blocks": [
            list(results[0].get("search_segments", [])),
            list(results[1].get("search_segments", [])),
        ],
        "arbiter_block": list(results[2].get("search_segments", [])),
        "required_independent_discoveries": 2,
        "validated_candidates": len(ranked),
        "inconclusive_retained_candidates": len(retained),
        "failed_candidates": len(rejected),
        "unmatched_candidate_instances": len(unmatched),
        "raw_candidate_instances": len(all_audit),
        "policy": (
            "intersect the first two independent searches; use the third block "
            "to arbitrate primary-only hypotheses; require period-and-phase "
            "identity in at least two blocks; preserve every raw hypothesis"
        ),
        "truth_catalog_used": False,
    }
    output["strict_holdout_policy"] = {
        "rank_requires_two_independent_discoveries": True,
        "rank_requires_no_explicit_covered_contradiction": True,
        "untestable_fold_is_failure": False,
        "candidate_identity": "period_plus_epoch_phase_plus_event_train",
        "raw_candidates_preserved": True,
        "truth_catalog_used": False,
    }
    output["runtime_seconds"] = round(
        sum(float(result.get("runtime_seconds") or 0.0) for result in results), 2
    )
    return output
