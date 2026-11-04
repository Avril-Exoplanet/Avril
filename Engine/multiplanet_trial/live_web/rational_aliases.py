"""Evidence-gated compression of non-integer rational period aliases."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from fractions import Fraction


MAX_DENOMINATOR = 16
# Only low-order p/q aliases are structurally plausible BLS fold aliases.
# Large numerators make accidental subset overlap almost inevitable for a
# many-sector target (for example a 143/2 coincidence matched five events and
# incorrectly hid a real 2.253 d signal in the L 98-59 stress test).
MAX_NUMERATOR = 16
RATIO_RELATIVE_TOLERANCE = 1e-4
MIN_SHARED_SUPPORTED_EVENTS = 3
MIN_SHARED_SEGMENTS = 2
MIN_SUPPORTED_EVENT_CONTAINMENT = 0.50
MIN_WINNER_SUPPORT_FRACTION = 0.15
MIN_SUPPORT_FRACTION_ADVANTAGE = 0.10


def _repeatability(candidate: Mapping) -> Mapping:
    return (
        candidate.get("holdout_verification", {})
        .get("event_repeatability", {})
    )


def _supported_events(candidate: Mapping) -> list[dict]:
    return sorted(
        [
            dict(row)
            for row in _repeatability(candidate).get("event_support", [])
            if row.get("supported") is True
        ],
        key=lambda row: (int(row["segment"]), float(row["center_btjd"])),
    )


def _matched_supported_events(
    first_events: list[dict], second_events: list[dict], tolerance_days: float
) -> tuple[int, list[int]]:
    """Greedily count one-to-one supported events aligned in the same sector."""

    first_by_segment: dict[int, list[float]] = {}
    second_by_segment: dict[int, list[float]] = {}
    for row in first_events:
        first_by_segment.setdefault(int(row["segment"]), []).append(
            float(row["center_btjd"])
        )
    for row in second_events:
        second_by_segment.setdefault(int(row["segment"]), []).append(
            float(row["center_btjd"])
        )

    matched = 0
    matched_segments: list[int] = []
    for segment in sorted(set(first_by_segment) & set(second_by_segment)):
        first = first_by_segment[segment]
        second = second_by_segment[segment]
        left = right = local_matches = 0
        while left < len(first) and right < len(second):
            delta = first[left] - second[right]
            if abs(delta) <= tolerance_days:
                local_matches += 1
                left += 1
                right += 1
            elif delta < 0:
                left += 1
            else:
                right += 1
        if local_matches:
            matched += local_matches
            matched_segments.append(segment)
    return matched, matched_segments


def rational_alias_decision(first: Mapping, second: Mapping) -> dict | None:
    """Resolve a p/q ambiguity only when supported events prove dependence.

    Period commensurability alone is never enough: real multi-planet systems
    can be resonant.  The two solutions must share supported event centers in
    multiple sectors, and one solution must predict those events materially
    more coherently (or explain far more supported events at equal coherence).
    """

    first_period = float(first["period_days"])
    second_period = float(second["period_days"])
    ratio = max(first_period, second_period) / min(first_period, second_period)
    fraction = Fraction(ratio).limit_denominator(MAX_DENOMINATOR)
    if fraction.denominator == 1:
        return None
    if fraction.numerator > MAX_NUMERATOR:
        return None
    approximated = float(fraction)
    relative_error = abs(ratio - approximated) / approximated
    if relative_error > RATIO_RELATIVE_TOLERANCE:
        return None

    first_events = _supported_events(first)
    second_events = _supported_events(second)
    minimum_events = min(len(first_events), len(second_events))
    if minimum_events < MIN_SHARED_SUPPORTED_EVENTS:
        return None
    tolerance_days = max(
        float(first.get("duration_hours") or 0.0),
        float(second.get("duration_hours") or 0.0),
    ) / 24.0
    if tolerance_days <= 0:
        return None
    matched, segments = _matched_supported_events(
        first_events, second_events, tolerance_days
    )
    containment = matched / minimum_events
    if (
        matched < MIN_SHARED_SUPPORTED_EVENTS
        or len(segments) < MIN_SHARED_SEGMENTS
        or containment < MIN_SUPPORTED_EVENT_CONTAINMENT
    ):
        return None

    first_fraction = float(_repeatability(first).get("support_fraction") or 0.0)
    second_fraction = float(_repeatability(second).get("support_fraction") or 0.0)
    winner: Mapping | None = None
    loser: Mapping | None = None
    reason: str | None = None
    if (
        max(first_fraction, second_fraction) >= MIN_WINNER_SUPPORT_FRACTION
        and abs(first_fraction - second_fraction)
        >= MIN_SUPPORT_FRACTION_ADVANTAGE
    ):
        winner, loser = (
            (first, second)
            if first_fraction > second_fraction
            else (second, first)
        )
        reason = "higher_supported_event_fraction"
    elif (
        min(first_fraction, second_fraction) >= MIN_WINNER_SUPPORT_FRACTION
        and max(len(first_events), len(second_events))
        >= 2 * min(len(first_events), len(second_events))
    ):
        winner, loser = (
            (first, second)
            if len(first_events) > len(second_events)
            else (second, first)
        )
        reason = "equal_coherence_more_supported_events"
    if winner is None or loser is None or reason is None:
        return None

    return {
        "winner_proposal_id": str(winner["proposal_id"]),
        "loser_proposal_id": str(loser["proposal_id"]),
        "period_ratio": ratio,
        "rational_numerator": int(fraction.numerator),
        "rational_denominator": int(fraction.denominator),
        "period_ratio_relative_error": relative_error,
        "matched_supported_events": matched,
        "matched_segments": segments,
        "supported_event_containment": containment,
        "first_support_fraction": first_fraction,
        "second_support_fraction": second_fraction,
        "first_supported_events": len(first_events),
        "second_supported_events": len(second_events),
        "event_center_tolerance_days": tolerance_days,
        "reason": reason,
        "truth_catalog_used": False,
    }


def compress_proven_rational_aliases(
    candidates: list[dict],
    *,
    quality: Callable[[dict], tuple] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Move only directly proven p/q aliases out of the canonical list."""

    ordered = sorted(candidates, key=quality, reverse=True) if quality else list(candidates)
    active = {str(row["proposal_id"]): row for row in ordered}
    aliases: list[dict] = []
    decisions: list[dict] = []
    for first_index, first in enumerate(ordered):
        first_id = str(first["proposal_id"])
        if first_id not in active:
            continue
        for second in ordered[first_index + 1 :]:
            second_id = str(second["proposal_id"])
            if first_id not in active:
                break
            if second_id not in active:
                continue
            decision = rational_alias_decision(first, second)
            if decision is None:
                continue
            winner_id = decision["winner_proposal_id"]
            loser_id = decision["loser_proposal_id"]
            if winner_id not in active or loser_id not in active:
                continue
            winner = active[winner_id]
            loser = active.pop(loser_id)
            winner.setdefault("alternate_periods", []).append(
                {
                    "period_days": float(loser["period_days"]),
                    "proposal_id": loser_id,
                    "holdout_verification": loser["holdout_verification"],
                    "alias_decision": decision,
                }
            )
            aliases.append(
                {
                    **loser,
                    "registry_state": "validated_rational_alias",
                    "represented_by_proposal_id": winner_id,
                    "alias_decision": decision,
                }
            )
            decisions.append(decision)
    return list(active.values()), aliases, decisions


__all__ = [
    "rational_alias_decision",
    "compress_proven_rational_aliases",
]
