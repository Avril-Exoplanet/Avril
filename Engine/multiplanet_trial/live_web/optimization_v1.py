"""Exact, drop-in speedups for the clean temporal-consensus pipeline.

This module deliberately leaves :mod:`multiplanet_trial.live_web.pipeline`
untouched.  Every optimization preserves the reference candidate ledger and
calls the reference scientific predicates and measurement kernels unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from . import pipeline as reference
from .temporal_consensus import _period_uncertainty, candidate_match_details


def _finite_period_uncertainty(candidate: Mapping) -> tuple[float, float] | None:
    try:
        period = float(candidate["period_days"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(period) or period <= 0:
        return None
    try:
        uncertainty = float(_period_uncertainty(candidate))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(uncertainty) or uncertainty < 0:
        return None
    return period, uncertainty


def _indexed_pair_candidates(
    ledger: list[dict], diagnostics: dict | None = None
):
    """Yield a conservative superset of cross-block pairs in reference order.

    A true match must satisfy ``abs(period_a-period_b) <= ua+ub``.  For each
    other block we search the wider interval ``ua + max(ub)`` and then pass
    every returned pair to the original exact predicate.  ``nextafter`` widens
    both bounds by one representable float so boundary rounding cannot exclude
    a pair accepted by the reference implementation.
    """

    blocks: dict[object, list[int]] = {}
    parsed: list[tuple[float, float] | None] = []
    for index, candidate in enumerate(ledger):
        blocks.setdefault(candidate.get("source_block"), []).append(index)
        parsed.append(_finite_period_uncertainty(candidate))

    indices: dict[object, dict] = {}
    for block, members in blocks.items():
        valid = [index for index in members if parsed[index] is not None]
        invalid = [index for index in members if parsed[index] is None]
        valid.sort(key=lambda index: (parsed[index][0], index))
        periods = np.asarray([parsed[index][0] for index in valid], dtype=np.float64)
        max_uncertainty = max(
            (parsed[index][1] for index in valid),
            default=0.0,
        )
        indices[block] = {
            "members": members,
            "valid": valid,
            "invalid": invalid,
            "periods": periods,
            "max_uncertainty": float(max_uncertainty),
        }

    possible = sum(
        len(blocks[left]) * len(blocks[right])
        for left_index, left in enumerate(blocks)
        for right in list(blocks)[left_index + 1 :]
    )
    exact_calls = 0
    for first_index, first in enumerate(ledger):
        parsed_first = parsed[first_index]
        possible_seconds: list[int] = []
        for block, block_index in indices.items():
            if block == first.get("source_block"):
                continue
            if parsed_first is None:
                possible_seconds.extend(block_index["members"])
            else:
                period, uncertainty = parsed_first
                radius = uncertainty + block_index["max_uncertainty"]
                lower = np.nextafter(period - radius, -np.inf)
                upper = np.nextafter(period + radius, np.inf)
                periods = block_index["periods"]
                start = int(np.searchsorted(periods, lower, side="left"))
                stop = int(np.searchsorted(periods, upper, side="right"))
                possible_seconds.extend(block_index["valid"][start:stop])
                # Invalid/non-finite inputs take the exact reference path.
                possible_seconds.extend(block_index["invalid"])

        for second_index in sorted(set(possible_seconds)):
            if second_index <= first_index:
                continue
            exact_calls += 1
            yield first_index, second_index

    if diagnostics is not None:
        diagnostics.update(
            {
                "all_cross_block_pairs": int(possible),
                "exact_predicate_calls": int(exact_calls),
                "index_rejected_pairs": int(possible - exact_calls),
            }
        )


def build_cross_block_proposals_indexed(
    ledger: list[dict],
    *,
    diagnostics: dict | None = None,
    pair_builder=None,
    append_only_order: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Exact replacement for ``_build_cross_block_proposals``.

    The index only rejects pairs that cannot pass the reference period-overlap
    inequality.  Every remaining pair is decided by ``candidate_match_details``
    and emitted in the same nested-loop order as the reference implementation.
    """

    matches: list[dict] = []
    raw_proposals: list[dict] = []
    for first_index, second_index in _indexed_pair_candidates(ledger, diagnostics):
        first = ledger[first_index]
        second = ledger[second_index]
        proof = candidate_match_details(first, second)
        if proof is None:
            continue
        builder = pair_builder or reference._pair_proposal
        raw_proposals.append(builder(first, second, proof))
        matches.append(
            {
                "first_raw_id": first["raw_id"],
                "second_raw_id": second["raw_id"],
                **dict(proof),
            }
        )

    if append_only_order:
        proposals = sorted(
            raw_proposals,
            key=lambda row: (
                tuple(str(value) for value in row["source_raw_ids"]),
                float(row["period_days"]),
                float(row["epoch_btjd"]),
            ),
        )
    else:
        proposals = sorted(
            raw_proposals,
            key=lambda row: row["source_strength"],
            reverse=True,
        )
    for index, proposal in enumerate(proposals, start=1):
        proposal["proposal_id"] = f"MATCH-{index:04d}"
    if diagnostics is not None:
        diagnostics["accepted_matches"] = len(matches)
    return proposals, matches


def _prepare_blocks(blocks: list[list[int]], branch: reference.SearchBranch):
    return [reference._subset_branch(branch, segments) for segments in blocks]


def _measure_prepared_blocks(
    candidate: Mapping,
    blocks: list[list[int]],
    prepared_blocks: list[reference.SearchBranch],
) -> list[dict]:
    """Run the unchanged measurement kernel on pre-sliced fold arrays."""

    rows: list[dict] = []
    discovered_blocks = {
        int(value) for value in candidate.get("discovered_blocks", [])
    }
    if candidate.get("source_block") is not None:
        discovered_blocks.add(int(candidate["source_block"]))
    for block_index, (segments, local) in enumerate(
        zip(blocks, prepared_blocks), start=1
    ):
        metrics = reference.measure_fixed_ephemeris(
            local.time,
            local.flux,
            period_days=float(candidate["period_days"]),
            epoch_btjd=float(candidate["epoch_btjd"]),
            duration_hours=float(candidate["duration_hours"]),
        )
        metrics = dict(metrics)
        metrics["timing_offset_days"] = 0.0
        metrics["timing_tolerance_applied"] = False
        rows.append(
            {
                "block": block_index,
                "segments": list(segments),
                "independently_discovered": block_index in discovered_blocks,
                "outcome": reference._metric_outcome(
                    metrics,
                    independently_discovered=block_index in discovered_blocks,
                ),
                **metrics,
            }
        )
    return rows


def validate_raw_ledger_cached_blocks(
    ledger: list[dict],
    blocks: list[list[int]],
    branch: reference.SearchBranch,
) -> list[dict]:
    """Validate every raw hypothesis after slicing each fold exactly once."""

    prepared = _prepare_blocks(blocks, branch)
    output: list[dict] = []
    for row in ledger:
        measurements = _measure_prepared_blocks(row, blocks, prepared)
        output.append(
            {
                **row,
                "fixed_block_validation": reference._validation_decision(measurements),
            }
        )
    return output


def validate_proposals_cached_blocks(
    proposals: list[dict],
    blocks: list[list[int]],
    branch: reference.SearchBranch,
) -> list[dict]:
    """Validate every proposal unchanged after slicing each fold once."""

    prepared = _prepare_blocks(blocks, branch)
    output: list[dict] = []
    for proposal in proposals:
        cached = proposal.get("multiphase_prevalidation")
        if isinstance(cached, Mapping):
            verification = dict(cached)
            verification["fixed_ephemeris_measurement_reused"] = True
        else:
            measurements = _measure_prepared_blocks(proposal, blocks, prepared)
            verification = reference._validation_decision(measurements)
        verification["event_repeatability_role"] = (
            "deferred_until_family_selection"
            if verification.get("passed") is True
            else "skipped_after_failed_fixed_ephemeris"
        )
        output.append({**proposal, "holdout_verification": verification})
    return output


def install() -> None:
    """Install only the exact replacements into an explicitly opted-in process."""

    reference._build_cross_block_proposals = build_cross_block_proposals_indexed
    reference._validate_raw_ledger = validate_raw_ledger_cached_blocks
    reference._validate_proposals = validate_proposals_cached_blocks


def pair_reduction_summary(ledger: list[dict]) -> dict:
    """Return index efficiency without changing or validating the ledger."""

    diagnostics: dict = {}
    for _pair in _indexed_pair_candidates(ledger, diagnostics):
        pass
    return diagnostics
