"""Merge frozen-BLS output with the experimental event-first channel."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Mapping

from .event_search import (
    DipEvent,
    EventFirstResult,
    PeriodicEventCandidate,
    measure_candidate_timings,
)


def _plain(candidate: Any) -> dict:
    if isinstance(candidate, Mapping):
        return dict(candidate)
    if is_dataclass(candidate):
        return asdict(candidate)
    if hasattr(candidate, "to_dict"):
        return dict(candidate.to_dict())
    raise TypeError("BLS candidates must be mappings or dataclass-like objects")


def event_result_from_dict(payload: Mapping[str, Any]) -> EventFirstResult:
    """Restore a saved event-first result for later evidence processing."""
    result = payload.get("result", payload)
    periodic = []
    for raw in result.get("periodic", []):
        item = dict(raw)
        item["ttv_minutes"] = tuple(item.get("ttv_minutes", ()))
        item["matched_event_indices"] = tuple(item.get("matched_event_indices", ()))
        periodic.append(PeriodicEventCandidate(**item))
    return EventFirstResult(
        periodic=tuple(periodic),
        doubles=tuple(dict(item) for item in result.get("doubles", [])),
        singles=tuple(DipEvent(**item) for item in result.get("singles", [])),
        all_events=tuple(DipEvent(**item) for item in result.get("all_events", [])),
        method=str(result.get("method", "independent-event-first-v0.3-experimental")),
        limitations=tuple(result.get("limitations", ())),
    )


def merge_candidate_channels(
    bls_candidates: Iterable[Any],
    event_result: EventFirstResult,
    *,
    period_tolerance_fraction: float = 0.01,
) -> dict:
    """Create a game/backend queue without turning diagnostics into hard vetoes.

    A signal found independently by BLS and the event-first pass receives a
    cross-method flag.  Event-only periodic, double, and single events remain in
    review/waiting states rather than being promoted to planets.
    """
    merged: list[dict] = []
    matched_event_periodic: set[int] = set()
    events = list(event_result.all_events)

    for raw in bls_candidates:
        candidate = _plain(raw)
        period = float(candidate["period_days"])
        epoch = float(candidate["epoch_btjd"])
        duration = float(candidate["duration_hours"])
        matches = [
            (index, item)
            for index, item in enumerate(event_result.periodic)
            if abs(item.period_days - period) / period <= period_tolerance_fraction
        ]
        matches.sort(key=lambda pair: abs(pair[1].period_days - period))
        event_match = matches[0] if matches else None
        if event_match is not None:
            matched_event_periodic.add(event_match[0])
        timing = measure_candidate_timings(
            events,
            period_days=period,
            epoch_btjd=epoch,
            duration_hours=duration,
        )
        merged.append(
            {
                "kind": "periodic",
                "status": "cross_method_priority" if event_match else "player_pool",
                "detection_sources": ["bls", "event_first"] if event_match else ["bls"],
                "signal": candidate,
                "event_first_match": event_match[1].to_dict() if event_match else None,
                "event_timing": timing,
                "hard_rejected": False,
            }
        )

    for index, item in enumerate(event_result.periodic):
        if index in matched_event_periodic:
            continue
        merged.append(
            {
                "kind": "periodic",
                "status": "player_pool_event_first",
                "detection_sources": ["event_first"],
                "signal": item.to_dict(),
                "hard_rejected": False,
            }
        )
    for item in event_result.doubles:
        merged.append(
            {
                "kind": "double_event",
                "status": "waiting_for_more_data",
                "detection_sources": ["event_first"],
                "signal": dict(item),
                "hard_rejected": False,
            }
        )
    for item in event_result.singles:
        merged.append(
            {
                "kind": "single_event",
                "status": "waiting_for_more_data",
                "detection_sources": ["event_first"],
                "signal": item.to_dict(),
                "hard_rejected": False,
            }
        )
    return {
        "version": "v0.3-experimental",
        "policy": "merge channels and expose evidence; do not hard-veto event-only signals",
        "candidates": merged,
    }
