from __future__ import annotations

from multiplanet_trial.live_web.rational_aliases import rational_alias_decision


def _candidate(proposal_id: str, period: float, centers: list[tuple[int, float, bool]]) -> dict:
    supported = sum(row[2] for row in centers)
    return {
        "proposal_id": proposal_id,
        "period_days": period,
        "epoch_btjd": 0.0,
        "duration_hours": 2.0,
        "holdout_verification": {
            "event_repeatability": {
                "testable_events": len(centers),
                "supported_events": supported,
                "support_fraction": supported / len(centers),
                "event_support": [
                    {
                        "segment": segment,
                        "center_btjd": center,
                        "supported": is_supported,
                    }
                    for segment, center, is_supported in centers
                ],
            }
        },
    }


def test_rational_alias_requires_shared_supported_events_across_sectors() -> None:
    real = _candidate(
        "real",
        3.0,
        [
            (1, 0.0, True),
            (1, 3.0, True),
            (2, 6.0, True),
            (2, 9.0, True),
            (2, 12.0, True),
            (2, 15.0, True),
            (2, 18.0, True),
        ],
    )
    alias = _candidate(
        "alias",
        4.5,
        [
            (1, 0.0, True),
            (1, 4.5, False),
            (2, 9.0, True),
            (2, 13.5, False),
            (2, 18.0, True),
        ],
    )
    decision = rational_alias_decision(real, alias)
    assert decision is not None
    assert decision["rational_numerator"] == 3
    assert decision["rational_denominator"] == 2
    assert decision["winner_proposal_id"] == "real"
    assert decision["truth_catalog_used"] is False


def test_resonant_periods_with_different_event_centers_are_not_aliases() -> None:
    first = _candidate(
        "first",
        3.0,
        [(1, 0.0, True), (1, 3.0, True), (2, 6.0, True), (2, 9.0, True)],
    )
    second = _candidate(
        "second",
        4.5,
        [(1, 1.0, True), (1, 5.5, True), (2, 10.0, True), (2, 14.5, True)],
    )
    assert rational_alias_decision(first, second) is None


def test_high_order_subset_coincidence_is_not_a_rational_alias() -> None:
    short = _candidate(
        "short",
        2.0,
        [
            (1, 0.0, True),
            (1, 2.0, True),
            (2, 4.0, True),
            (2, 6.0, True),
            (3, 8.0, True),
            (3, 10.0, True),
        ],
    )
    high_order = _candidate(
        "high-order",
        143.0,
        [(1, 0.0, True), (2, 286.0, True), (3, 572.0, True)],
    )
    assert rational_alias_decision(short, high_order) is None


def test_rational_alias_can_be_resolved_with_realistic_sparse_support() -> None:
    real = _candidate(
        "real",
        12.0,
        [
            (segment, float(segment * 12), segment <= 13)
            for segment in range(1, 41)
        ],
    )
    alias = _candidate(
        "alias",
        8.0,
        [
            (segment, float(segment * 12), segment <= 11)
            for segment in range(1, 60)
        ],
    )
    decision = rational_alias_decision(real, alias)
    assert decision is not None
    assert decision["winner_proposal_id"] == "real"
