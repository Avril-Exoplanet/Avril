from __future__ import annotations

from .fast_science_finalize import (
    eligible_for_phase_null,
    eligible_for_physical_vetting,
    direct_alias_winners,
    merge_candidates,
    refresh_harmonic_decision,
    skipped_significance,
)


def _candidate(identifier: str, period: float = 5.0) -> dict:
    return {
        "candidate_id": identifier,
        "proposal_id": identifier,
        "rank": 1,
        "period_days": period,
        "epoch_btjd": 100.0,
        "duration_hours": 2.0,
        "event_coherence": {"level": "repeated_coherent_event_train"},
        "holdout_verification": {
            "event_repeatability": {"supported_events": 8}
        },
        "physical_vetting": {"passed": True, "vetoes": []},
        "reference_vetting": {
            "available": True,
            "status": "complete",
            "warnings": [],
        },
        "pixel_vetting": {"aggregate_status": "no_pixel_localization"},
    }


def test_only_deterministically_viable_candidates_reach_phase_null() -> None:
    good = _candidate("good")
    incoherent = _candidate("incoherent")
    incoherent["event_coherence"]["level"] = "incoherent_event_train"
    vetoed = _candidate("vetoed")
    vetoed["physical_vetting"] = {"passed": False, "vetoes": ["odd_even"]}
    warned = _candidate("warned")
    warned["reference_vetting"]["warnings"] = ["model_shift_secondary"]

    assert eligible_for_physical_vetting(good) is True
    assert eligible_for_phase_null(good) is True
    assert eligible_for_phase_null(incoherent) is False
    assert eligible_for_phase_null(vetoed) is False
    assert eligible_for_phase_null(warned) is False


def test_new_and_legacy_off_target_consensus_both_stop_phase_null() -> None:
    for status in (
        "possible_off_target_consensus_across_sectors",
        "possible_off_target_in_at_least_one_sector",
    ):
        candidate = _candidate(status)
        candidate["pixel_vetting"]["aggregate_status"] = status
        assert eligible_for_phase_null(candidate) is False
        result = skipped_significance(candidate, requested_trials=4096)
        assert result["status"] == "skipped_after_off_target_pixel_veto"


def test_sparse_two_or_three_event_signal_is_retained_for_manual_review() -> None:
    sparse = _candidate("sparse")
    sparse["event_coherence"]["level"] = "sparse_coherent_event_train"
    sparse["holdout_verification"]["event_repeatability"]["supported_events"] = 3
    assert eligible_for_physical_vetting(sparse) is True
    assert eligible_for_phase_null(sparse) is False
    result = skipped_significance(sparse, requested_trials=4096)
    assert result["status"] == (
        "skipped_insufficient_events_for_automatic_high_priority"
    )


def test_alias_loss_only_defers_when_the_direct_winner_is_still_viable() -> None:
    loser = _candidate("real-long")
    viable_winner = _candidate("viable-short")
    failed_winner = _candidate("failed-short")
    failed_winner["event_coherence"]["level"] = "mixed_event_train"
    relations = [
        {
            "resolved": True,
            "relation_type": "integer_harmonic",
            "winner_proposal_id": "viable-short",
            "loser_proposal_id": "real-long",
        },
        {
            "resolved": True,
            "relation_type": "integer_harmonic",
            "winner_proposal_id": "failed-short",
            "loser_proposal_id": "real-long",
        },
    ]
    assert direct_alias_winners(
        [loser, viable_winner, failed_winner], relations
    ) == {"real-long": {"viable-short"}}


def test_rational_resonance_never_short_circuits_independent_phase_null() -> None:
    real = _candidate("real-9d", period=9.15)
    sparse_long = _candidate("sparse-41d", period=41.175)
    relations = [
        {
            "resolved": True,
            "relation_type": "low_order_rational_harmonic",
            "numerator": 9,
            "denominator": 2,
            "winner_proposal_id": "sparse-41d",
            "loser_proposal_id": "real-9d",
        }
    ]
    assert direct_alias_winners([real, sparse_long], relations) == {}


def test_short_circuit_retains_identity_and_declares_zero_executed_trials() -> None:
    candidate = _candidate("vetoed")
    candidate["physical_vetting"] = {"passed": False, "vetoes": ["odd_even"]}
    result = skipped_significance(candidate, requested_trials=4096)
    assert result["status"] == "skipped_after_physical_hard_veto"
    assert result["phase_shift_null"]["requested_trials"] == 4096
    assert result["phase_shift_null"]["executed_trials"] == 0
    assert result["passes_7p1_reference"] is False
    assert result["candidate_retained"] is True


def test_merge_preserves_original_order_and_candidate_count() -> None:
    originals = [_candidate("A"), _candidate("B"), _candidate("C")]
    annotated = [{**_candidate("C"), "marker": 3}, {**_candidate("A"), "marker": 1}]
    merged = merge_candidates(originals, annotated)
    assert [row["candidate_id"] for row in merged] == ["A", "B", "C"]
    assert [row.get("marker") for row in merged] == [1, None, 3]


def test_full_ledger_harmonic_context_can_only_remove_independence() -> None:
    short = _candidate("short", period=5.0)
    long = _candidate("long", period=10.0)
    long["detection_significance"] = {
        "combined": {"mes_like": 20.0},
        "null_calibrated_mes_like": 9.0,
        "look_elsewhere_fap_estimate": 1e-4,
        "passes_7p1_reference": True,
        "flags": [],
    }
    short["holdout_verification"] = {
        "event_repeatability": {"support_fraction": 0.9}
    }
    long["holdout_verification"] = {
        "event_repeatability": {"support_fraction": 0.9}
    }
    refresh_harmonic_decision(long, [short, long])
    assert long["detection_significance"]["passes_7p1_reference"] is False
    assert long["detection_significance"]["status"] == "integer_harmonic_not_independent"
    assert long["detection_significance"]["harmonic_warnings"]
