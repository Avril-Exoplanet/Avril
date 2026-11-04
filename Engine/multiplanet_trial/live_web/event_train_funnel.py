"""Direct, event-proven compute funnel for expensive downstream null tests.

Every candidate remains in the audit.  The funnel only avoids running the same
4096 phase-shift workflow for many aliases that predict and measure essentially
the same downward event train.  Membership is direct to the selected anchor;
relations are never chained transitively.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations


SCALES_DAYS = (0.035, 0.05, 0.07, 0.09, 0.12, 0.16, 0.21, 0.28, 0.36, 0.45)
MIN_MATCHED_POSITIVE_EVENTS = 3
MIN_MATCHED_SEGMENTS = 2
MIN_POSITIVE_EVENT_CONTAINMENT = 0.80
MAX_ANCHOR_EVENTS = 16


def _positive_events(candidate: Mapping) -> list[dict]:
    rows = [
        dict(row)
        for row in (
            candidate.get("holdout_verification", {})
            .get("event_repeatability", {})
            .get("event_support", [])
        )
        if math.isfinite(float(row.get("depth_ppm") or 0.0))
        and float(row.get("depth_ppm") or 0.0) > 0.0
    ]
    rows.sort(key=lambda row: (int(row["segment"]), float(row["center_btjd"])))
    return rows


def _anchor_events(rows: list[dict]) -> list[dict]:
    if len(rows) <= MAX_ANCHOR_EVENTS:
        return rows
    strongest = sorted(
        range(len(rows)),
        key=lambda index: float(rows[index].get("snr") or 0.0),
        reverse=True,
    )[: MAX_ANCHOR_EVENTS // 2]
    spread = [
        int(round(index * (len(rows) - 1) / (MAX_ANCHOR_EVENTS // 2 - 1)))
        for index in range(MAX_ANCHOR_EVENTS // 2)
    ]
    return [rows[index] for index in sorted(set(strongest + spread))]


def _proposal_pairs(candidates: Sequence[Mapping], events: list[list[dict]]):
    """LSH proposes pairs; the exact event proof below makes every decision."""

    buckets: dict[tuple[int, int, int, int], set[int]] = defaultdict(set)
    for candidate_index, (candidate, rows) in enumerate(zip(candidates, events)):
        duration = float(candidate.get("duration_hours") or 0.0) / 24.0
        for row in _anchor_events(rows):
            segment = int(row["segment"])
            center = float(row["center_btjd"])
            for scale_index, scale in enumerate(SCALES_DAYS):
                if scale + 1e-15 < duration:
                    continue
                width = 2.0 * scale
                # Two offset grids guarantee that centers separated by at
                # most ``scale`` share at least one bucket.
                buckets[
                    (scale_index, 0, segment, int(math.floor(center / width)))
                ].add(candidate_index)
                buckets[
                    (
                        scale_index,
                        1,
                        segment,
                        int(math.floor((center + scale) / width)),
                    )
                ].add(candidate_index)
    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        ordered = sorted(members)
        pairs.update(combinations(ordered, 2))
    return pairs


def _matched_positive_events(
    first: list[dict], second: list[dict], tolerance: float
) -> tuple[int, list[int], float]:
    minimum = min(len(first), len(second))
    if minimum == 0:
        return 0, [], 0.0
    first_by_segment: dict[int, list[float]] = defaultdict(list)
    second_by_segment: dict[int, list[float]] = defaultdict(list)
    for row in first:
        first_by_segment[int(row["segment"])].append(float(row["center_btjd"]))
    for row in second:
        second_by_segment[int(row["segment"])].append(float(row["center_btjd"]))
    matched = 0
    segments: list[int] = []
    for segment in sorted(set(first_by_segment) & set(second_by_segment)):
        left_values = first_by_segment[segment]
        right_values = second_by_segment[segment]
        left = right = local = 0
        while left < len(left_values) and right < len(right_values):
            delta = left_values[left] - right_values[right]
            if abs(delta) <= tolerance:
                local += 1
                left += 1
                right += 1
            elif delta < 0.0:
                left += 1
            else:
                right += 1
        if local:
            matched += local
            segments.append(segment)
    return matched, segments, matched / minimum


def direct_event_train_groups(
    candidates: Sequence[dict],
    *,
    quality: Callable[[dict], tuple],
) -> tuple[list[list[int]], list[dict]]:
    """Return direct-anchor compute groups and complete relation evidence."""

    if not candidates:
        return [], []
    events = [_positive_events(candidate) for candidate in candidates]
    proposed = _proposal_pairs(candidates, events)
    neighbours: dict[int, set[int]] = defaultdict(set)
    relation_by_pair: dict[tuple[int, int], dict] = {}
    for left, right in proposed:
        tolerance = max(
            float(candidates[left].get("duration_hours") or 0.0),
            float(candidates[right].get("duration_hours") or 0.0),
        ) / 24.0
        matched, segments, containment = _matched_positive_events(
            events[left], events[right], tolerance
        )
        if (
            matched < MIN_MATCHED_POSITIVE_EVENTS
            or len(segments) < MIN_MATCHED_SEGMENTS
            or containment < MIN_POSITIVE_EVENT_CONTAINMENT
        ):
            continue
        relation = {
            "first_index": left,
            "second_index": right,
            "first_proposal_id": candidates[left].get("proposal_id"),
            "second_proposal_id": candidates[right].get("proposal_id"),
            "matched_positive_events": matched,
            "matched_segments": segments,
            "positive_event_containment": containment,
            "period_arithmetic_required": False,
            "direct_event_proof": True,
            "candidate_removed": False,
        }
        relation_by_pair[(left, right)] = relation
        neighbours[left].add(right)
        neighbours[right].add(left)

    ordered = sorted(range(len(candidates)), key=lambda index: quality(candidates[index]), reverse=True)
    unassigned = set(ordered)
    groups: list[list[int]] = []
    used_relations: list[dict] = []
    for anchor in ordered:
        if anchor not in unassigned:
            continue
        members = [anchor]
        related = sorted(
            neighbours.get(anchor, set()) & unassigned,
            key=lambda index: quality(candidates[index]),
            reverse=True,
        )
        for member in related:
            if member == anchor:
                continue
            members.append(member)
            pair = (min(anchor, member), max(anchor, member))
            used_relations.append(relation_by_pair[pair])
        for member in members:
            unassigned.discard(member)
        groups.append(members)
    return groups, used_relations


__all__ = ["direct_event_train_groups"]
