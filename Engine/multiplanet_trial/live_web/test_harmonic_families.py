from __future__ import annotations

import copy
import json

import numpy as np

from multiplanet_trial.live_web.harmonic_families import (
    _commensurability,
    _commensurate_candidate_pairs,
    arbitrate_harmonic_families,
    harmonic_relation,
)


def _candidate(
    proposal_id: str,
    period: float,
    events: list[tuple[int, int, float, bool]],
) -> dict:
    supported = sum(bool(row[3]) for row in events)
    return {
        "proposal_id": proposal_id,
        "period_days": period,
        "epoch_btjd": 0.0,
        "duration_hours": 3.0,
        "source_strength": 10.0,
        "discovered_blocks": [1, 2, 3],
        "ensemble_vote": {
            "positive_round_count": 3,
            "positive_groups": 9,
            "strong_groups": 6,
            "contradicting_groups": 0,
        },
        "holdout_verification": {
            "strong_blocks": [1, 2, 3],
            "combined_positive_snr": 15.0,
            "block_measurements": [
                {"outcome": "strong", "snr": 8.0} for _ in range(3)
            ],
            "event_repeatability": {
                "testable_events": len(events),
                "supported_events": supported,
                "support_fraction": supported / len(events),
                "event_support": [
                    {
                        "cycle": cycle,
                        "segment": segment,
                        "center_btjd": center,
                        "depth_ppm": 1000.0 if is_supported else -100.0,
                        "supported": is_supported,
                    }
                    for cycle, segment, center, is_supported in events
                ],
            },
        },
    }


def _quality(candidate: dict) -> tuple:
    repeatability = candidate["holdout_verification"]["event_repeatability"]
    return (
        float(repeatability["support_fraction"]),
        int(repeatability["supported_events"]),
    )


def test_family_selects_period_that_avoids_empty_and_missing_events() -> None:
    half = _candidate(
        "half",
        6.0,
        [
            (cycle, cycle // 4 + 1, cycle * 6.0, cycle % 2 == 0)
            for cycle in range(40)
        ],
    )
    primary = _candidate(
        "primary",
        12.0,
        [
            (cycle, cycle // 2 + 1, cycle * 12.0, True)
            for cycle in range(20)
        ],
    )
    double = _candidate(
        "double",
        24.0,
        [
            (cycle, cycle + 1, cycle * 24.0, True)
            for cycle in range(10)
        ],
    )
    canonical, aliases, relations, families = arbitrate_harmonic_families(
        [half, primary, double], quality=_quality
    )
    assert [row["proposal_id"] for row in canonical] == ["primary"]
    assert {row["proposal_id"] for row in aliases} == {"half", "double"}
    assert len(relations) == 3
    assert len(families) == 1
    assert families[0]["resolved"] is True
    assert families[0]["primary_proposal_id"] == "primary"


def test_family_arbitration_is_input_order_invariant() -> None:
    half = _candidate(
        "half",
        6.0,
        [(cycle, 1, cycle * 6.0, cycle % 2 == 0) for cycle in range(20)],
    )
    primary = _candidate(
        "primary",
        12.0,
        [(cycle, 1, cycle * 12.0, True) for cycle in range(10)],
    )
    first = arbitrate_harmonic_families([half, primary], quality=_quality)
    second = arbitrate_harmonic_families([primary, half], quality=_quality)
    assert [row["proposal_id"] for row in first[0]] == ["primary"]
    assert [row["proposal_id"] for row in second[0]] == ["primary"]


def test_family_audit_and_alternate_periods_are_byte_stable_across_input_order() -> None:
    half = _candidate(
        "half",
        6.0,
        [(cycle, cycle // 4 + 1, cycle * 6.0, cycle % 2 == 0) for cycle in range(40)],
    )
    primary = _candidate(
        "primary",
        12.0,
        [(cycle, cycle // 2 + 1, cycle * 12.0, True) for cycle in range(20)],
    )
    double = _candidate(
        "double",
        24.0,
        [(cycle, cycle + 1, cycle * 24.0, True) for cycle in range(10)],
    )

    serializations = []
    for rows in (
        [half, primary, double],
        [double, half, primary],
        [primary, double, half],
    ):
        result = arbitrate_harmonic_families(copy.deepcopy(rows), quality=_quality)
        serializations.append(
            json.dumps(result, sort_keys=True, separators=(",", ":"))
        )

    assert len(set(serializations)) == 1


def test_unresolved_family_keeps_every_member_visible() -> None:
    short = _candidate(
        "short",
        10.0,
        [
            (0, 1, 0.0, True),
            (1, 2, 10.0, True),
            (2, 3, 20.0, True),
            (3, 4, 30.0, False),
            (4, 5, 40.0, False),
            (5, 6, 50.0, False),
        ],
    )
    long = _candidate(
        "long",
        20.0,
        [
            (0, 1, 0.0, True),
            (1, 2, 10.0, True),
            (2, 3, 20.0, True),
        ],
    )
    relation = harmonic_relation(short, long)
    assert relation is not None
    assert relation["resolved"] is False
    canonical, aliases, _relations, families = arbitrate_harmonic_families(
        [short, long], quality=_quality
    )
    assert {row["proposal_id"] for row in canonical} == {"short", "long"}
    assert aliases == []
    assert families[0]["resolved"] is False


def test_high_order_integer_alias_uses_events_and_epoch_alignment() -> None:
    order = 17
    short = _candidate(
        "short",
        4.0,
        [
            (cycle, cycle // order + 1, cycle * 4.0, True)
            for cycle in range(3 * order)
            if cycle % order != order - 1
        ],
    )
    long = _candidate(
        "long",
        4.0 * order,
        [
            (cycle, cycle + 1, cycle * 4.0 * order, True)
            for cycle in range(3)
        ],
    )
    relation = harmonic_relation(short, long)
    assert relation is not None
    assert relation["relation_type"] == "integer_harmonic"
    assert relation["numerator"] == order
    assert relation["resolved"] is True
    assert relation["winner_proposal_id"] == "short"
    assert (
        relation["decision"]["reason"]
        == "high_order_low_point_subset_of_repeating_shorter_period"
    )

    canonical, aliases, _relations, families = arbitrate_harmonic_families(
        [long, short], quality=_quality
    )
    assert [row["proposal_id"] for row in canonical] == ["short"]
    assert [row["proposal_id"] for row in aliases] == ["long"]
    assert families[0]["resolved"] is True


def test_nearby_high_order_period_without_commensurability_stays_separate() -> None:
    short = _candidate(
        "short",
        4.0,
        [(cycle, cycle // 17 + 1, cycle * 4.0, True) for cycle in range(51)],
    )
    nearby = _candidate(
        "nearby",
        68.08,
        [(cycle, cycle + 1, cycle * 68.08, True) for cycle in range(3)],
    )
    assert harmonic_relation(short, nearby) is None


def test_period_index_keeps_every_pair_the_quadratic_scan_would_test() -> None:
    rng = np.random.default_rng(20261024)
    periods = rng.uniform(0.5, 400.0, 300).tolist()
    periods.extend(
        [
            3.0,
            3.0 * 17.0 * (1.0 - 0.001),
            5.0,
            5.0 * 7.0 / 3.0 * (1.0 + 0.0001),
            11.0,
            11.0 * 200.0 * (1.0 - 0.001),
        ]
    )
    candidates = [
        {"proposal_id": str(index), "period_days": period}
        for index, period in enumerate(periods)
    ]
    expected = {
        (left, right)
        for left in range(len(candidates))
        for right in range(left + 1, len(candidates))
        if _commensurability(candidates[left], candidates[right]) is not None
    }
    actual = set(_commensurate_candidate_pairs(candidates))
    assert actual == expected
