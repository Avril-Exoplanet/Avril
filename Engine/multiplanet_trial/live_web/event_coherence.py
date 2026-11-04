"""Robust, non-destructive event-train coherence annotations.

The ordinary ``support_fraction`` counts only individually >=3-sigma events.
That statistic is useful for deep transits but is badly calibrated for shallow
multi-event planets: many real events can have the correct sign without being
individually significant.  This module therefore combines exact sign tests in
independent temporal blocks with a winsorised signed-S/N summary.

The result is a ranking/review annotation.  It never deletes a candidate or
changes its locked ephemeris.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np


COHERENCE_LEVEL_RANK = {
    "unavailable": -1,
    "incoherent_event_train": 0,
    "mixed_event_train": 1,
    "sparse_coherent_event_train": 2,
    "repeated_coherent_event_train": 3,
}


def one_sided_sign_probability(positive: int, total: int) -> float | None:
    """Exact P[X >= positive] for X~Binomial(total, 0.5)."""

    positive = int(positive)
    total = int(total)
    if total <= 0 or positive < 0 or positive > total:
        return None
    return float(
        sum(math.comb(total, value) for value in range(positive, total + 1))
        / (2**total)
    )


def _block_map(blocks: Iterable[Iterable[int]]) -> dict[int, int]:
    return {
        int(segment): block_index
        for block_index, segments in enumerate(blocks, start=1)
        for segment in segments
    }


def _event_summary(events: list[Mapping]) -> dict:
    depths = np.asarray(
        [float(row.get("depth_ppm") or 0.0) for row in events], dtype=float
    )
    signed_snrs = np.asarray(
        [float(row.get("snr") or 0.0) for row in events], dtype=float
    )
    finite = np.isfinite(depths) & np.isfinite(signed_snrs)
    depths = depths[finite]
    signed_snrs = signed_snrs[finite]
    total = int(depths.size)
    if total == 0:
        return {
            "testable_events": 0,
            "positive_depth_events": 0,
            "positive_depth_fraction": None,
            "median_depth_ppm": None,
            "one_sided_sign_probability": None,
            "winsorized_signed_snr": None,
        }
    positive = int(np.count_nonzero(depths > 0))
    clipped = np.clip(signed_snrs, -3.0, 3.0)
    robust_signed_snr = float(np.sum(clipped) / math.sqrt(total))
    return {
        "testable_events": total,
        "positive_depth_events": positive,
        "positive_depth_fraction": float(positive / total),
        "median_depth_ppm": float(np.median(depths)),
        "one_sided_sign_probability": one_sided_sign_probability(
            positive, total
        ),
        "winsorized_signed_snr": robust_signed_snr,
        "winsorization": "per-event signed S/N clipped to [-3,+3]",
    }


def event_coherence_metrics(
    candidate: Mapping, blocks: Iterable[Iterable[int]]
) -> dict:
    """Summarise event direction/repeatability without a destructive gate."""

    blocks = [list(map(int, block)) for block in blocks]
    verification = candidate.get("holdout_verification") or {}
    repeatability = verification.get("event_repeatability") or {}
    events = list(repeatability.get("event_support") or [])
    segment_to_block = _block_map(blocks)
    per_block: list[dict] = []
    for block_index in range(1, len(blocks) + 1):
        selected = [
            row
            for row in events
            if segment_to_block.get(int(row.get("segment") or -1))
            == block_index
        ]
        per_block.append(
            {"block": block_index, **_event_summary(selected)}
        )

    combined = _event_summary(events)
    testable = int(combined["testable_events"])
    significant_blocks = [
        row
        for row in per_block
        if int(row["testable_events"]) >= 5
        and float(row.get("median_depth_ppm") or 0.0) > 0
        and row.get("one_sided_sign_probability") is not None
        # Six consistently positive events already give p=1/64.  A 0.025
        # per-block threshold recognises that exact small-N case while the
        # combined train must still pass the stricter 0.01 condition below.
        and float(row["one_sided_sign_probability"]) <= 0.025
    ]
    supporting_blocks = [
        row
        for row in per_block
        if int(row["testable_events"]) >= 2
        and float(row.get("median_depth_ppm") or 0.0) > 0
        and row.get("one_sided_sign_probability") is not None
        and float(row["one_sided_sign_probability"]) <= 0.10
    ]
    positive_blocks = [
        row
        for row in per_block
        if int(row["testable_events"]) > 0
        and float(row.get("median_depth_ppm") or 0.0) > 0
    ]
    overall_probability = combined.get("one_sided_sign_probability")
    robust_snr = float(combined.get("winsorized_signed_snr") or 0.0)
    positive_events = int(combined.get("positive_depth_events") or 0)

    if testable == 0:
        level = "unavailable"
        reasons = ["no_testable_events"]
    elif (
        len(significant_blocks) >= 2
        and overall_probability is not None
        and float(overall_probability) <= 0.01
        and robust_snr > 0
    ):
        level = "repeated_coherent_event_train"
        reasons = [
            "significant_positive_direction_in_multiple_temporal_blocks",
            "winsorized_event_signal_is_positive",
        ]
    elif (
        testable <= 8
        and positive_events >= 2
        and len(positive_blocks) >= min(2, len(per_block))
        and robust_snr > 0
    ) or (
        testable <= 40
        and len(supporting_blocks) >= 2
        and overall_probability is not None
        and float(overall_probability) <= 0.05
        and robust_snr > 0
    ):
        level = "sparse_coherent_event_train"
        reasons = [
            "limited_event_count_but_direction_is_repeated",
            "retain_for_long_period_review",
        ]
    elif (
        testable >= 20
        and not significant_blocks
        and (
            overall_probability is None
            or float(overall_probability) > 0.05
            or robust_snr <= 0
        )
    ):
        level = "incoherent_event_train"
        reasons = [
            "many_testable_events_without_significant_directional_repetition"
        ]
    else:
        level = "mixed_event_train"
        reasons = ["event_evidence_is_not_decisive"]

    probability_score = (
        min(300.0, -math.log10(max(float(overall_probability), 1e-300)))
        if overall_probability is not None
        else 0.0
    )
    return {
        "schema_version": "vlora-event-coherence-v1",
        "policy": "soft ranking and review evidence; never delete or retune",
        "level": level,
        "level_rank": COHERENCE_LEVEL_RANK[level],
        "reasons": reasons,
        "combined": combined,
        "blocks": per_block,
        "significant_direction_blocks": len(significant_blocks),
        "supporting_direction_blocks": len(supporting_blocks),
        "positive_median_blocks": len(positive_blocks),
        "sign_probability_score": probability_score,
        "candidate_retained": True,
        "ephemeris_unchanged": True,
    }


def annotate_event_coherence(
    candidate: dict, blocks: Iterable[Iterable[int]]
) -> dict:
    """Attach and return the event-coherence annotation."""

    materialized_blocks = [list(map(int, block)) for block in blocks]
    report = event_coherence_metrics(candidate, materialized_blocks)
    candidate["event_coherence"] = report
    return report


def event_coherence_quality(candidate: Mapping) -> tuple:
    """Stable sort prefix used before aggregate fold/vote statistics."""

    report = candidate.get("event_coherence") or {}
    combined = report.get("combined") or {}
    return (
        int(report.get("level_rank", COHERENCE_LEVEL_RANK["unavailable"])),
        int(report.get("significant_direction_blocks") or 0),
        float(report.get("sign_probability_score") or 0.0),
        float(combined.get("winsorized_signed_snr") or 0.0),
        int(report.get("supporting_direction_blocks") or 0),
    )


__all__ = [
    "COHERENCE_LEVEL_RANK",
    "annotate_event_coherence",
    "event_coherence_metrics",
    "event_coherence_quality",
    "one_sided_sign_probability",
]
