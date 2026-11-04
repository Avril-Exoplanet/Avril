from __future__ import annotations

from .candidate_tiers import candidate_tier
from .science_review import rank_science_candidates, science_review_annotation


def _fully_passing_candidate(pixel_status: str, *, sectors_requested: int) -> dict:
    return {
        "event_coherence": {"level": "repeated_coherent_event_train"},
        "holdout_verification": {
            "event_repeatability": {
                "supported_events": 8,
                "testable_events": 8,
                "support_fraction": 0.875,
            }
        },
        "detection_significance": {
            "passes_7p1_reference": True,
            "status": "meets_7p1_mes_like_reference",
            "harmonic_warnings": [],
            "block_metrics": [
                {"block": 1, "mes_like": 8.0},
                {"block": 2, "mes_like": 7.0},
            ],
        },
        "physical_vetting": {"passed": True, "vetoes": []},
        "reference_vetting": {
            "available": True,
            "status": "complete",
            "warnings": [],
        },
        "pixel_vetting": {
            "aggregate_status": pixel_status,
            "sectors_requested": sectors_requested,
        },
    }


def test_good_diagnostics_create_follow_up_priority_not_planet_claim() -> None:
    result = science_review_annotation(
        {
            "event_coherence": {
                "level": "repeated_coherent_event_train"
            },
            "detection_significance": {
                "passes_7p1_reference": True,
                "status": "meets_7p1_mes_like_reference",
                "harmonic_warnings": [],
                "block_metrics": [
                    {"block": 1, "mes_like": 8.0},
                    {"block": 2, "mes_like": 7.0},
                    {"block": 3, "mes_like": 2.0},
                ],
            },
            "physical_vetting": {"passed": True, "vetoes": []},
            "reference_vetting": {
                "available": True,
                "status": "complete",
                "warnings": [],
            },
            "pixel_vetting": {"aggregate_status": "no_pixel_localization"},
        }
    )
    assert result["priority"] == "high_priority_follow_up_candidate"
    assert result["not_planet_probability"] is True
    assert result["confirmation_required"] is True


def test_reference_warning_lowers_priority_without_deleting_candidate() -> None:
    result = science_review_annotation(
        {
            "event_coherence": {"level": "repeated_coherent_event_train"},
            "detection_significance": {
                "passes_7p1_reference": True,
                "status": "meets_7p1_mes_like_reference",
                "harmonic_warnings": [],
                "block_metrics": [
                    {"block": 1, "mes_like": 8.0},
                    {"block": 2, "mes_like": 7.0},
                    {"block": 3, "mes_like": 2.0},
                ],
            },
            "physical_vetting": {"vetoes": []},
            "reference_vetting": {
                "warnings": ["sinusoidal_variability_at_period"]
            },
        }
    )
    assert result["priority"] == "low_priority_or_false_alarm_risk"
    assert result["candidate_retained"] is True
    assert "reference:sinusoidal_variability_at_period" in result["risk_flags"]


def test_measured_off_target_pixel_signal_lowers_priority_without_deleting() -> None:
    result = science_review_annotation(
        {
            "event_coherence": {"level": "repeated_coherent_event_train"},
            "detection_significance": {
                "passes_7p1_reference": True,
                "status": "meets_7p1_mes_like_reference",
                "harmonic_warnings": [],
                "block_metrics": [
                    {"block": 1, "mes_like": 8.0},
                    {"block": 2, "mes_like": 7.0},
                ],
            },
            "physical_vetting": {"vetoes": []},
            "reference_vetting": {"warnings": []},
            "pixel_vetting": {
                "aggregate_status": "possible_off_target_in_at_least_one_sector"
            },
        }
    )
    assert result["priority"] == "low_priority_or_false_alarm_risk"
    assert result["candidate_retained"] is True
    assert "pixel:possible_off_target_in_at_least_one_sector" in result["risk_flags"]


def test_cross_sector_on_target_consensus_is_a_clean_pixel_result() -> None:
    candidate = _fully_passing_candidate(
        "on_target_consensus_across_sectors", sectors_requested=35
    )
    review = science_review_annotation(candidate)
    tier = candidate_tier(candidate)
    assert review["priority"] == "high_priority_follow_up_candidate"
    assert not any(flag.startswith("pixel:") for flag in review["risk_flags"])
    assert tier["tier"] == "final_candidate"


def test_cross_sector_off_target_consensus_is_rejected() -> None:
    candidate = _fully_passing_candidate(
        "possible_off_target_consensus_across_sectors", sectors_requested=12
    )
    review = science_review_annotation(candidate)
    tier = candidate_tier(candidate)
    assert review["priority"] == "low_priority_or_false_alarm_risk"
    assert tier["tier"] == "rejected_signal"
    assert "pixel_source_possible_off_target" in tier["rejection_reasons"]


def test_attempted_but_mixed_pixel_followup_requires_manual_review() -> None:
    candidate = _fully_passing_candidate(
        "pixel_localization_mixed_or_unresolved", sectors_requested=42
    )
    review = science_review_annotation(candidate)
    tier = candidate_tier(candidate)
    assert review["priority"] == "manual_review_required"
    assert tier["tier"] == "review_required"
    assert "pixel:pixel_localization_mixed_or_unresolved" in tier["review_reasons"]


def test_single_time_block_cannot_be_high_priority() -> None:
    result = science_review_annotation(
        {
            "event_coherence": {"level": "sparse_coherent_event_train"},
            "detection_significance": {
                "passes_7p1_reference": True,
                "status": "meets_7p1_mes_like_reference",
                "harmonic_warnings": [],
                "block_metrics": [
                    {"block": 1, "mes_like": 1.0},
                    {"block": 2, "mes_like": 60.0},
                    {"block": 3, "mes_like": 2.0},
                ],
            },
            "physical_vetting": {"vetoes": []},
            "reference_vetting": {"warnings": []},
        }
    )
    assert result["priority"] == "low_priority_or_false_alarm_risk"
    assert "red_noise:single_time_block_dominated" in result["risk_flags"]


def test_gap_dominated_single_event_is_retained_but_low_priority() -> None:
    candidate = _fully_passing_candidate(
        "no_pixel_localization", sectors_requested=0
    )
    candidate["detection_significance"]["status"] = (
        "gap_adjacent_single_event_dominated"
    )
    candidate["detection_significance"]["passes_7p1_reference"] = False
    candidate["detection_significance"]["event_gap_diagnostics"] = {
        "gap_dominated_weak_remainder": True,
        "dominant_event_segment": 104,
        "dominant_event_ses_like": 18.46,
        "combined_without_gap_compromised_events": {"mes_like": 2.09},
    }

    result = science_review_annotation(candidate)

    assert result["priority"] == "low_priority_or_false_alarm_risk"
    assert result["candidate_retained"] is True
    assert (
        "event_integrity:gap_adjacent_single_event_dominated"
        in result["risk_flags"]
    )


def test_sparse_three_event_signal_is_manual_not_deleted() -> None:
    result = science_review_annotation(
        {
            "event_coherence": {"level": "sparse_coherent_event_train"},
            "holdout_verification": {
                "event_repeatability": {"supported_events": 3}
            },
            "detection_significance": {
                "passes_7p1_reference": True,
                "status": "meets_7p1_mes_like_reference",
                "harmonic_warnings": [],
                "block_metrics": [
                    {"block": 1, "mes_like": 8.0},
                    {"block": 2, "mes_like": 7.0},
                ],
            },
            "physical_vetting": {"passed": True, "vetoes": []},
            "reference_vetting": {
                "available": True,
                "status": "complete",
                "warnings": [],
            },
        }
    )
    assert result["priority"] == "manual_review_required"
    assert result["candidate_retained"] is True


def test_incoherent_event_train_is_retained_but_low_priority() -> None:
    result = science_review_annotation(
        {
            "event_coherence": {"level": "incoherent_event_train"},
            "detection_significance": {
                "passes_7p1_reference": True,
                "status": "meets_7p1_mes_like_reference",
            },
            "physical_vetting": {"vetoes": []},
        }
    )
    assert result["priority"] == "low_priority_or_false_alarm_risk"
    assert result["candidate_retained"] is True


def test_direct_event_proven_harmonic_loser_cannot_be_high_priority() -> None:
    result = science_review_annotation(
        {
            "event_coherence": {"level": "repeated_coherent_event_train"},
            "detection_significance": {
                "passes_7p1_reference": True,
                "status": "meets_7p1_mes_like_reference",
                "harmonic_warnings": [],
                "block_metrics": [
                    {"block": 1, "mes_like": 9.0},
                    {"block": 2, "mes_like": 8.0},
                ],
            },
            "physical_vetting": {"passed": True, "vetoes": []},
            "reference_vetting": {
                "available": True,
                "status": "complete",
                "warnings": [],
            },
            "direct_harmonic_alias": {
                "winner_proposal_ids": ["short-period-winner"],
                "winner_passed_exact_phase_null": True,
            },
        }
    )
    assert result["priority"] == "low_priority_or_false_alarm_risk"
    assert "harmonic:direct_alias_of_significant_winner" in result["risk_flags"]


def test_science_rank_promotes_review_passes_and_preserves_discovery_rank() -> None:
    priorities = {
        1: "high_priority_follow_up_candidate",
        2: "low_priority_or_false_alarm_risk",
        3: "low_priority_or_false_alarm_risk",
        4: "low_priority_or_false_alarm_risk",
        5: "low_priority_or_false_alarm_risk",
        6: "low_priority_or_false_alarm_risk",
        7: "high_priority_follow_up_candidate",
        8: "high_priority_follow_up_candidate",
    }
    candidates = [
        {
            "candidate_id": f"C{rank}",
            "rank": rank,
            "ensemble_rank": rank,
            "period_days": float(rank),
            "epoch_btjd": 100.0 + rank,
            "duration_hours": 2.0,
            "science_review": {"priority": priority},
        }
        for rank, priority in priorities.items()
    ]

    ranked = rank_science_candidates(candidates)

    assert [row["candidate_id"] for row in ranked[:3]] == ["C1", "C7", "C8"]
    assert [row["science_rank"] for row in ranked] == list(range(1, 9))
    assert {row["candidate_id"]: row["discovery_rank"] for row in ranked} == {
        f"C{rank}": rank for rank in priorities
    }
    assert len(ranked) == len(candidates)
    assert [
        (row["period_days"], row["epoch_btjd"], row["duration_hours"])
        for row in ranked
    ] == [
        (1.0, 101.0, 2.0),
        (7.0, 107.0, 2.0),
        (8.0, 108.0, 2.0),
        (2.0, 102.0, 2.0),
        (3.0, 103.0, 2.0),
        (4.0, 104.0, 2.0),
        (5.0, 105.0, 2.0),
        (6.0, 106.0, 2.0),
    ]
