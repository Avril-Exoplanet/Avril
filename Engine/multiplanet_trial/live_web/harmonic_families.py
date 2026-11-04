"""Order-independent harmonic-family arbitration for locked candidates.

Candidates are never deleted from the audit trail.  Direct, event-backed
integer and low-order rational relations first form families.  A family is
collapsed only when one undefeated member has a directed evidence path to
every other member.  Otherwise every member remains visible and the family is
marked unresolved.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Callable, Mapping
from fractions import Fraction

from . import pipeline as core
from .rational_aliases import (
    MAX_DENOMINATOR,
    MAX_NUMERATOR,
    RATIO_RELATIVE_TOLERANCE,
    rational_alias_decision,
)

MIN_SHARED_SUPPORTED_EVENTS = 3
MIN_SHARED_SEGMENTS = 2
MIN_UNRESOLVED_CONTAINMENT = 0.50
MAX_INTEGER_HARMONIC_ORDER = 200
INTEGER_RATIO_RELATIVE_TOLERANCE = 0.001
HIGH_ORDER_MIN_POSITIVE_EVENT_CONTAINMENT = 0.80
HIGH_ORDER_MIN_SHORT_POSITIVE_FRACTION = 0.80
HIGH_ORDER_MIN_POSITIVE_EVENTS_PER_RESIDUE = 2


def _commensurate_candidate_pairs(
    candidates: list[Mapping],
) -> list[tuple[int, int]]:
    """Return every pair that can pass ``_commensurability``.

    The former all-pairs loop spent quadratic time proving that almost every
    unrelated period was not a harmonic.  A sorted-period index searches only
    the exact integer and low-order rational tolerance windows used by the
    unchanged final predicate.  The returned original-index order matches the
    former nested loop, making this an execution-only optimization.
    """

    if len(candidates) < 2:
        return []
    ratio_windows: dict[float, float] = {
        float(order): INTEGER_RATIO_RELATIVE_TOLERANCE
        for order in range(2, MAX_INTEGER_HARMONIC_ORDER + 1)
    }
    for numerator in range(2, MAX_NUMERATOR + 1):
        for denominator in range(1, MAX_DENOMINATOR + 1):
            fraction = Fraction(numerator, denominator)
            if fraction <= 1 or fraction.numerator > MAX_NUMERATOR:
                continue
            ratio = float(fraction)
            ratio_windows.setdefault(ratio, RATIO_RELATIVE_TOLERANCE)

    ordered = sorted(
        (float(candidate["period_days"]), index)
        for index, candidate in enumerate(candidates)
    )
    periods = [row[0] for row in ordered]
    possible: set[tuple[int, int]] = set()
    for short_position, (short_period, short_index) in enumerate(ordered[:-1]):
        if not math.isfinite(short_period) or short_period <= 0:
            continue
        for ratio, tolerance in ratio_windows.items():
            lower = short_period * ratio * (1.0 - tolerance)
            upper = short_period * ratio * (1.0 + tolerance)
            first = bisect_left(periods, lower, short_position + 1)
            last = bisect_right(periods, upper, first)
            for long_position in range(first, last):
                long_index = ordered[long_position][1]
                possible.add(
                    (min(short_index, long_index), max(short_index, long_index))
                )
    return sorted(possible)


def _repeatability(candidate: Mapping) -> Mapping:
    return (
        candidate.get("holdout_verification", {})
        .get("event_repeatability", {})
    )


def _supported_events(candidate: Mapping) -> list[dict]:
    cached = candidate.get("_harmonic_supported_events_cache")
    if cached is not None:
        return cached
    rows = sorted(
        [
            dict(row)
            for row in _repeatability(candidate).get("event_support", [])
            if row.get("supported") is True
        ],
        key=lambda row: (int(row["segment"]), float(row["center_btjd"])),
    )
    if isinstance(candidate, dict):
        candidate["_harmonic_supported_events_cache"] = rows
    return rows


def _positive_events(candidate: Mapping) -> list[dict]:
    """Return testable epochs whose measured direction is a flux decrease."""

    cached = candidate.get("_harmonic_positive_events_cache")
    if cached is not None:
        return cached
    rows = sorted(
        [
            dict(row)
            for row in _repeatability(candidate).get("event_support", [])
            if math.isfinite(float(row.get("depth_ppm") or 0.0))
            and float(row.get("depth_ppm") or 0.0) > 0.0
        ],
        key=lambda row: (int(row["segment"]), float(row["center_btjd"])),
    )
    if isinstance(candidate, dict):
        candidate["_harmonic_positive_events_cache"] = rows
    return rows


def _matched_events(
    first_events: list[dict],
    second_events: list[dict],
    *,
    tolerance_days: float,
) -> tuple[int, list[int], float]:
    minimum = min(len(first_events), len(second_events))
    if minimum == 0:
        return 0, [], 0.0
    first_by_segment: dict[int, list[float]] = defaultdict(list)
    second_by_segment: dict[int, list[float]] = defaultdict(list)
    for row in first_events:
        first_by_segment[int(row["segment"])].append(float(row["center_btjd"]))
    for row in second_events:
        second_by_segment[int(row["segment"])].append(float(row["center_btjd"]))
    matched = 0
    matched_segments: list[int] = []
    for segment in sorted(set(first_by_segment) & set(second_by_segment)):
        left_values = first_by_segment[segment]
        right_values = second_by_segment[segment]
        left = right = local = 0
        while left < len(left_values) and right < len(right_values):
            delta = left_values[left] - right_values[right]
            if abs(delta) <= tolerance_days:
                local += 1
                left += 1
                right += 1
            elif delta < 0:
                left += 1
            else:
                right += 1
        if local:
            matched += local
            matched_segments.append(segment)
    return matched, matched_segments, matched / minimum


def _matched_supported_events(
    first: Mapping, second: Mapping
) -> tuple[int, list[int], float]:
    tolerance = max(
        float(first.get("duration_hours") or 0.0),
        float(second.get("duration_hours") or 0.0),
    ) / 24.0
    return _matched_events(
        _supported_events(first),
        _supported_events(second),
        tolerance_days=tolerance,
    )


def _commensurability(first: Mapping, second: Mapping) -> dict | None:
    first_period = float(first["period_days"])
    second_period = float(second["period_days"])
    ratio = max(first_period, second_period) / min(first_period, second_period)
    integer_order = int(round(ratio))
    if integer_order >= 2:
        integer_relative_error = abs(ratio - integer_order) / integer_order
        if (
            integer_order <= MAX_INTEGER_HARMONIC_ORDER
            and integer_relative_error <= INTEGER_RATIO_RELATIVE_TOLERANCE
        ):
            return {
                "period_ratio": ratio,
                "numerator": integer_order,
                "denominator": 1,
                "relative_error": integer_relative_error,
                "relation_type": "integer_harmonic",
            }

    # Non-integer p/q relations remain deliberately low order.  High-order
    # rational coincidences are common in many-sector searches and are not
    # sufficient evidence that two real planets are aliases of one another.
    fraction = Fraction(ratio).limit_denominator(MAX_DENOMINATOR)
    if fraction.numerator > MAX_NUMERATOR:
        return None
    approximated = float(fraction)
    relative_error = abs(ratio - approximated) / approximated
    tolerance = RATIO_RELATIVE_TOLERANCE
    if fraction.numerator < 2 or relative_error > tolerance:
        return None
    return {
        "period_ratio": ratio,
        "numerator": int(fraction.numerator),
        "denominator": int(fraction.denominator),
        "relative_error": relative_error,
        "relation_type": (
            "integer_harmonic"
            if fraction.denominator == 1
            else "low_order_rational_harmonic"
        ),
    }


def _high_order_integer_alias_decision(
    first: Mapping,
    second: Mapping,
    *,
    order: int,
    matched_positive_events: int,
    positive_segments: list[int],
    positive_containment: float,
) -> dict | None:
    """Resolve a high-order subset only with strong event-train evidence.

    Period commensurability alone is intentionally insufficient.  The shorter
    solution must be a densely repeating train, the longer solution must share
    a majority of its supported events across multiple sectors, and the locked
    epochs must align.  This catches sparse 17x/37x/83x subset folds without
    merging unrelated planets that merely sit near a high-order ratio.
    """

    first_period = float(first["period_days"])
    second_period = float(second["period_days"])
    shorter, longer = (first, second) if first_period < second_period else (second, first)
    shorter_period = float(shorter["period_days"])
    ratio = float(longer["period_days"]) / shorter_period
    phase_error = abs(
        (
            float(longer["epoch_btjd"])
            - float(shorter["epoch_btjd"])
            + 0.5 * shorter_period
        )
        % shorter_period
        - 0.5 * shorter_period
    )
    tolerance = max(
        float(shorter.get("duration_hours") or 0.0),
        float(longer.get("duration_hours") or 0.0),
    ) / 24.0
    shorter_repeatability = _repeatability(shorter)
    shorter_positive = len(_positive_events(shorter))
    longer_positive = len(_positive_events(longer))
    shorter_testable = int(shorter_repeatability.get("testable_events") or 0)
    shorter_positive_fraction = (
        shorter_positive / shorter_testable if shorter_testable else 0.0
    )
    minimum_shorter_positive = HIGH_ORDER_MIN_POSITIVE_EVENTS_PER_RESIDUE * order
    if (
        tolerance <= 0
        or phase_error > tolerance
        or matched_positive_events < MIN_SHARED_SUPPORTED_EVENTS
        or len(positive_segments) < MIN_SHARED_SEGMENTS
        or positive_containment < HIGH_ORDER_MIN_POSITIVE_EVENT_CONTAINMENT
        or shorter_positive_fraction < HIGH_ORDER_MIN_SHORT_POSITIVE_FRACTION
        or shorter_positive < minimum_shorter_positive
        or longer_positive < MIN_SHARED_SUPPORTED_EVENTS
    ):
        return None
    return {
        "winner_proposal_id": str(shorter["proposal_id"]),
        "loser_proposal_id": str(longer["proposal_id"]),
        "order": order,
        "ratio": ratio,
        "phase_error_days": phase_error,
        "matched_positive_events": matched_positive_events,
        "matched_positive_segments": positive_segments,
        "positive_event_containment": positive_containment,
        "shorter_positive_events": shorter_positive,
        "longer_positive_events": longer_positive,
        "shorter_positive_fraction": shorter_positive_fraction,
        "minimum_shorter_positive_events": minimum_shorter_positive,
        "reason": "high_order_low_point_subset_of_repeating_shorter_period",
    }


def harmonic_relation(first: Mapping, second: Mapping) -> dict | None:
    """Return a direct event-backed family edge, resolved when possible."""

    commensurate = _commensurability(first, second)
    if commensurate is None:
        return None
    first_id = str(first["proposal_id"])
    second_id = str(second["proposal_id"])
    integer = core._integer_alias_decision(first, second)
    rational = None if integer is not None else rational_alias_decision(first, second)
    matched, segments, containment = _matched_supported_events(first, second)

    decision = integer or rational
    high_order_integer = (
        int(commensurate["denominator"]) == 1
        and int(commensurate["numerator"]) > MAX_NUMERATOR
    )
    if decision is None and high_order_integer:
        positive_tolerance = max(
            float(first.get("duration_hours") or 0.0),
            float(second.get("duration_hours") or 0.0),
        ) / 24.0
        positive_matched, positive_segments, positive_containment = _matched_events(
            _positive_events(first),
            _positive_events(second),
            tolerance_days=positive_tolerance,
        )
        decision = _high_order_integer_alias_decision(
            first,
            second,
            order=int(commensurate["numerator"]),
            matched_positive_events=positive_matched,
            positive_segments=positive_segments,
            positive_containment=positive_containment,
        )
    # Do not create an unresolved graph edge from high-order period arithmetic
    # alone.  Such edges can transitively join unrelated real planets into one
    # enormous family.  High-order edges exist only when the event train proves
    # the dependency above.
    if high_order_integer and decision is None:
        return None
    if decision is None and not (
        matched >= MIN_SHARED_SUPPORTED_EVENTS
        and len(segments) >= MIN_SHARED_SEGMENTS
        and containment >= MIN_UNRESOLVED_CONTAINMENT
    ):
        return None
    relation = {
        "first_proposal_id": first_id,
        "second_proposal_id": second_id,
        **commensurate,
        "matched_supported_events": matched,
        "matched_segments": segments,
        "supported_event_containment": containment,
        "resolved": decision is not None,
        "winner_proposal_id": (
            str(decision["winner_proposal_id"]) if decision else None
        ),
        "loser_proposal_id": (
            str(decision["loser_proposal_id"]) if decision else None
        ),
        "decision": decision,
        "truth_catalog_used": False,
    }
    return relation


def _event_consensus_scores(candidates: list[dict]) -> dict[str, dict]:
    """Score prediction precision and family-event recall for every member."""

    maximum_duration = max(
        (float(row.get("duration_hours") or 0.0) for row in candidates),
        default=0.0,
    ) / 24.0
    events_by_segment: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for candidate in candidates:
        proposal_id = str(candidate["proposal_id"])
        for event in _supported_events(candidate):
            events_by_segment[int(event["segment"])].append(
                (float(event["center_btjd"]), proposal_id)
            )

    consensus_clusters: list[dict] = []
    for segment, events in events_by_segment.items():
        events.sort()
        local: list[list[tuple[float, str]]] = []
        for center, proposal_id in events:
            if not local or center - local[-1][-1][0] > maximum_duration:
                local.append([(center, proposal_id)])
            else:
                local[-1].append((center, proposal_id))
        for cluster in local:
            supporters = {proposal_id for _center, proposal_id in cluster}
            if len(supporters) >= 2:
                consensus_clusters.append(
                    {
                        "segment": segment,
                        "center_btjd": sum(value for value, _ in cluster) / len(cluster),
                        "supporters": supporters,
                    }
                )
    if not consensus_clusters:
        for segment, events in events_by_segment.items():
            for center, proposal_id in events:
                consensus_clusters.append(
                    {
                        "segment": segment,
                        "center_btjd": center,
                        "supporters": {proposal_id},
                    }
                )

    output: dict[str, dict] = {}
    for candidate in candidates:
        proposal_id = str(candidate["proposal_id"])
        repeatability = _repeatability(candidate)
        testable = int(repeatability.get("testable_events") or 0)
        supported = int(repeatability.get("supported_events") or 0)
        precision = supported / testable if testable else 0.0
        supported_events = _supported_events(candidate)
        explained = 0
        for cluster in consensus_clusters:
            if any(
                int(event["segment"]) == int(cluster["segment"])
                and abs(
                    float(event["center_btjd"])
                    - float(cluster["center_btjd"])
                )
                <= maximum_duration
                for event in supported_events
            ):
                explained += 1
        recall = explained / len(consensus_clusters) if consensus_clusters else 0.0
        beta_squared = 4.0
        f2 = (
            (1.0 + beta_squared) * precision * recall
            / (beta_squared * precision + recall)
            if precision > 0 and recall > 0
            else 0.0
        )
        output[proposal_id] = {
            "testable_events": testable,
            "supported_events": supported,
            "predicted_empty_events": max(0, testable - supported),
            "event_precision": precision,
            "family_event_recall": recall,
            "event_f2": f2,
            "consensus_event_clusters": len(consensus_clusters),
        }
    return output


def _reachable(start: str, directed: Mapping[str, set[str]]) -> set[str]:
    visited: set[str] = set()
    frontier = [start]
    while frontier:
        current = frontier.pop()
        for child in sorted(directed.get(current, set())):
            if child not in visited:
                visited.add(child)
                frontier.append(child)
    return visited


def arbitrate_harmonic_families(
    candidates: list[dict],
    *,
    quality: Callable[[dict], tuple],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Group direct harmonic relations and select one defensible primary.

    Returns canonical candidates, retained alias audit rows, relation audit,
    and family audit.  Unresolved family members remain canonical.
    """

    if not candidates:
        return [], [], [], []
    relations: list[dict] = []
    neighbours: dict[str, set[str]] = defaultdict(set)
    directed: dict[str, set[str]] = defaultdict(set)
    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    by_id = {str(row["proposal_id"]): row for row in candidates}
    for left_index, right_index in _commensurate_candidate_pairs(candidates):
        left = candidates[left_index]
        right = candidates[right_index]
        if str(right["proposal_id"]) < str(left["proposal_id"]):
            left, right = right, left
        relation = harmonic_relation(left, right)
        if relation is None:
            continue
        relations.append(relation)
        left_id = str(relation["first_proposal_id"])
        right_id = str(relation["second_proposal_id"])
        neighbours[left_id].add(right_id)
        neighbours[right_id].add(left_id)
        if relation["resolved"]:
            winner = str(relation["winner_proposal_id"])
            loser = str(relation["loser_proposal_id"])
            directed[winner].add(loser)
            wins[winner] += 1
            losses[loser] += 1

    components: list[list[str]] = []
    visited: set[str] = set()
    for proposal_id in sorted(by_id):
        if proposal_id in visited:
            continue
        component: list[str] = []
        frontier = [proposal_id]
        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            frontier.extend(
                reversed(sorted(neighbours.get(current, set()) - visited))
            )
        component.sort()
        components.append(component)

    relations.sort(
        key=lambda row: (
            str(row["first_proposal_id"]),
            str(row["second_proposal_id"]),
            str(row.get("relation_type") or ""),
        )
    )

    canonical: list[dict] = []
    aliases: list[dict] = []
    families: list[dict] = []
    relation_by_pair = {
        frozenset(
            [str(row["first_proposal_id"]), str(row["second_proposal_id"])]
        ): row
        for row in relations
    }
    family_index = 0
    for component in components:
        members = [by_id[proposal_id] for proposal_id in component]
        if len(component) == 1:
            canonical.append(members[0])
            continue
        family_index += 1
        family_id = f"HARM-{family_index:05d}"
        event_scores = _event_consensus_scores(members)

        def selection_key(
            proposal_id: str,
            event_scores: Mapping[str, Mapping] = event_scores,
        ) -> tuple:
            score = event_scores[proposal_id]
            return (
                -int(losses[proposal_id]),
                int(wins[proposal_id]),
                float(score["family_event_recall"]),
                float(score["event_f2"]),
                *quality(by_id[proposal_id]),
                str(proposal_id),
            )

        suggested_primary = max(component, key=selection_key)
        undefeated = [value for value in component if losses[value] == 0]
        reachable = _reachable(suggested_primary, directed)
        fully_resolved = (
            len(undefeated) == 1
            and undefeated[0] == suggested_primary
            and set(component) - {suggested_primary} <= reachable
        )
        undefeated_by_wins = sorted(
            undefeated,
            key=lambda value: (wins[value], selection_key(value)),
            reverse=True,
        )
        second_undefeated_wins = (
            int(wins[undefeated_by_wins[1]])
            if len(undefeated_by_wins) > 1
            else -1
        )
        dominant_primary = (
            int(losses[suggested_primary]) == 0
            and int(wins[suggested_primary]) >= 2
            and int(wins[suggested_primary]) >= second_undefeated_wins + 2
            and len(reachable) >= 2
        )
        resolved_members = (
            set(component)
            if fully_resolved
            else ({suggested_primary} | reachable if dominant_primary else set())
        )
        unresolved_members = set(component) - resolved_members
        status = (
            "resolved"
            if fully_resolved
            else "partially_resolved"
            if resolved_members
            else "unresolved"
        )
        member_audit = []
        for proposal_id in sorted(component, key=selection_key, reverse=True):
            direct_relation = relation_by_pair.get(
                frozenset([suggested_primary, proposal_id])
            )
            member_audit.append(
                {
                    "proposal_id": proposal_id,
                    "period_days": float(by_id[proposal_id]["period_days"]),
                    "pairwise_wins": int(wins[proposal_id]),
                    "pairwise_losses": int(losses[proposal_id]),
                    "directly_related_to_primary": direct_relation is not None,
                    "selection_metrics": event_scores[proposal_id],
                }
            )
        family = {
            "family_id": family_id,
            "status": status,
            "resolved": fully_resolved,
            "primary_proposal_id": (
                suggested_primary if resolved_members else None
            ),
            "suggested_primary_proposal_id": suggested_primary,
            "member_count": len(component),
            "resolved_member_count": len(resolved_members),
            "unresolved_member_count": len(unresolved_members),
            "members": member_audit,
            "policy": (
                "collapse only members reached from a dominant undefeated "
                "primary by directed event-evidence paths; weakly connected "
                "fringe members remain visible"
            ),
            "truth_catalog_used": False,
        }
        families.append(family)
        for proposal_id in sorted(component, key=selection_key, reverse=True):
            candidate = by_id[proposal_id]
            candidate["harmonic_family"] = {
                "family_id": family_id,
                "status": status,
                "resolved": fully_resolved,
                "primary_proposal_id": family["primary_proposal_id"],
                "suggested_primary_proposal_id": suggested_primary,
                "member_count": len(component),
                "selection_metrics": event_scores[proposal_id],
                "pairwise_wins": int(wins[proposal_id]),
                "pairwise_losses": int(losses[proposal_id]),
            }
        if not resolved_members:
            canonical.extend(members)
            continue

        primary = by_id[suggested_primary]
        primary.setdefault("alternate_periods", [])
        canonical.append(primary)
        for proposal_id in sorted(component, key=selection_key, reverse=True):
            if proposal_id == suggested_primary:
                continue
            if proposal_id in unresolved_members:
                canonical.append(by_id[proposal_id])
                continue
            loser = by_id[proposal_id]
            direct_relation = relation_by_pair.get(
                frozenset([suggested_primary, proposal_id])
            )
            alias_relation = direct_relation or {
                "relation_type": "transitive_harmonic_family_member",
                "resolved": True,
                "winner_proposal_id": suggested_primary,
                "loser_proposal_id": proposal_id,
            }
            primary["alternate_periods"].append(
                {
                    "period_days": float(loser["period_days"]),
                    "proposal_id": proposal_id,
                    "holdout_verification": loser["holdout_verification"],
                    "harmonic_family_relation": alias_relation,
                }
            )
            aliases.append(
                {
                    **loser,
                    "registry_state": "validated_harmonic_family_alias",
                    "represented_by_proposal_id": suggested_primary,
                    "harmonic_family_relation": alias_relation,
                }
            )
    canonical.sort(
        key=lambda candidate: (
            *quality(candidate),
            str(candidate.get("proposal_id") or ""),
        ),
        reverse=True,
    )
    return canonical, aliases, relations, families


__all__ = ["arbitrate_harmonic_families", "harmonic_relation"]
