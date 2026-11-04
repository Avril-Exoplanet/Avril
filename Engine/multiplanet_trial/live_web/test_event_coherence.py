from __future__ import annotations

from .event_coherence import event_coherence_metrics, event_coherence_quality


BLOCKS = [[1], [2], [3]]


def _candidate(events: list[dict]) -> dict:
    return {
        "holdout_verification": {
            "event_repeatability": {"event_support": events}
        }
    }


def _events(segment: int, depths: list[float]) -> list[dict]:
    return [
        {
            "segment": segment,
            "depth_ppm": depth,
            "snr": depth / 100.0,
        }
        for depth in depths
    ]


def test_many_aggregate_events_do_not_beat_repeated_event_direction() -> None:
    incoherent = _candidate(
        _events(1, [100, -100] * 20 + [5000])
        + _events(2, [100, -100] * 20 + [5000])
        + _events(3, [100, -100] * 20 + [5000])
    )
    coherent = _candidate(
        _events(1, [100, 120, 80, 90, 110, 130, 95, 105])
        + _events(2, [90, 115, 75, 100, 125, 85, 105, 95])
        + _events(3, [80, 90, 100, 110, 120, 130, 140, 150])
    )
    incoherent["event_coherence"] = event_coherence_metrics(incoherent, BLOCKS)
    coherent["event_coherence"] = event_coherence_metrics(coherent, BLOCKS)
    assert incoherent["event_coherence"]["level"] == "incoherent_event_train"
    assert coherent["event_coherence"]["level"] == "repeated_coherent_event_train"
    assert event_coherence_quality(coherent) > event_coherence_quality(incoherent)


def test_sparse_long_period_events_are_retained_as_review_evidence() -> None:
    candidate = _candidate(
        _events(1, [350.0]) + _events(2, [420.0]) + _events(3, [280.0])
    )
    report = event_coherence_metrics(candidate, BLOCKS)
    assert report["level"] == "sparse_coherent_event_train"
    assert report["candidate_retained"] is True
