from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from multiplanet_trial.live_web import ensemble_intersection as ensemble_module
from multiplanet_trial.live_web.ensemble_intersection import (
    DEFAULT_REFIT_MODE,
    DEFAULT_VOTE_SEED_COUNT,
    REFIT_MODES,
    _adaptive_long_period_protected,
    _near_period_event_train_duplicate,
    _plan_post_refit_revote,
    _plan_pre_refit_harmonic_groups,
    _pre_refit_integer_harmonic_relation,
    _scaled_vote_thresholds,
    _select_adaptive_exact_fallbacks,
    _select_targeted_precision_rescues,
    build_balanced_discovery_groups,
    build_consensus_families,
    build_natural_sector_vote_groups,
    build_vote_consensus_families,
    deterministic_vote_seeds,
    ephemeris_refit_eligible,
    refit_failed_vote_families,
    rescore_changed_ephemerides_with_cached_population,
    score_ledger_on_vote_seed_ensemble,
)


def test_production_refit_default_is_post_ranking_targeted() -> None:
    assert DEFAULT_REFIT_MODE == "targeted"
    assert "targeted" in REFIT_MODES
    assert "adaptive" in REFIT_MODES
    assert "native-selective" in REFIT_MODES
    assert "native-shared" in REFIT_MODES


def test_targeted_post_refit_revote_keeps_failures_as_audit_only() -> None:
    rows = [
        {
            "proposal_id": "INITIAL-PASS",
            "holdout_verification": {"passed": True},
        },
        {
            "proposal_id": "REFIT-PASS",
            "holdout_verification": {"passed": True},
            "ephemeris_refit": {"rescued_failed_fixed_ephemeris": True},
        },
        {
            "proposal_id": "TARGETED-DEFERRED",
            "holdout_verification": {"passed": False},
            "ephemeris_refit": {"targeted_rescue_deferred": {}},
        },
        {
            "proposal_id": "DENSE-FAILED",
            "holdout_verification": {"passed": False},
        },
    ]
    active, audit_only, diagnostics = _plan_post_refit_revote(
        rows,
        refit_mode="targeted",
    )
    assert [index for index, _row in active] == [1]
    assert [index for index, _row in audit_only] == [2, 3]
    assert diagnostics["post_refit_revote_families"] == 1
    assert diagnostics["post_refit_unchanged_pass_families"] == 1
    assert diagnostics["post_refit_deferred_audit_only_families"] == 2
    assert diagnostics["candidate_rows_retained"] == 4
    assert diagnostics["candidate_rows_deleted"] is False
    assert [row["proposal_id"] for row in rows] == [
        "INITIAL-PASS",
        "REFIT-PASS",
        "TARGETED-DEFERRED",
        "DENSE-FAILED",
    ]


def test_non_targeted_post_refit_revote_preserves_full_historical_scope() -> None:
    rows = [
        {"holdout_verification": {"passed": True}},
        {"holdout_verification": {"passed": False}},
    ]
    active, audit_only, diagnostics = _plan_post_refit_revote(
        rows,
        refit_mode="native-exact",
    )
    assert [index for index, _row in active] == [0, 1]
    assert audit_only == []
    assert diagnostics["candidate_rows_deleted"] is False


def test_sparse_ephemeris_remeasurement_matches_full_population_revote() -> None:
    geometries = [
        {
            "seed": seed,
            "vote_groups": [{"score_offset": offset}],
            "long_period_vote_groups": [],
        }
        for seed, offset in ((101, 0.1), (202, 0.4), (303, 0.8))
    ]

    def scorer(rows, _branch, vote_groups, *, progress=None):
        del progress
        offset = float(vote_groups[0]["score_offset"])
        output = []
        for row in rows:
            period = float(row["period_days"])
            positive = 5 + int(round(10.0 * (period + offset))) % 4
            strong = min(positive, 2 + int(round(period + offset)) % 5)
            vote = {
                "groups_measured": 9,
                "rounds_measured": 3,
                "positive_groups": positive,
                "strong_groups": strong,
                "contradicting_groups": 9 - positive,
                "untestable_groups": 0,
                "positive_rounds": [1, 2, 3],
                "positive_round_count": 3,
                "positive_groups_by_round": {
                    "1": positive // 3,
                    "2": positive // 3,
                    "3": positive - 2 * (positive // 3),
                },
                "weakest_positive_snr": period + offset,
                "combined_positive_snr": 2.0 * period + offset,
                "measurements": [],
            }
            output.append({**row, "ensemble_vote": vote})
        return output

    rows = [
        {
            "proposal_id": f"ROW-{index}",
            "period_days": period,
            "source_strength": strength,
        }
        for index, (period, strength) in enumerate(((2.0, 5.0), (3.0, 8.0), (4.0, 3.0), (5.0, 7.0)))
    ]
    initial = score_ledger_on_vote_seed_ensemble(
        rows,
        SimpleNamespace(),
        geometries,
        scorer=scorer,
    )
    changed = [dict(row) for row in initial]
    changed[1]["period_days"] = 3.25
    full = score_ledger_on_vote_seed_ensemble(
        changed,
        SimpleNamespace(),
        geometries,
        scorer=scorer,
    )
    sparse, diagnostics = rescore_changed_ephemerides_with_cached_population(
        changed,
        [1],
        SimpleNamespace(),
        geometries,
        scorer=scorer,
    )
    assert sparse == full
    assert diagnostics == {
        "remeasured_families": 1,
        "ranking_population": 4,
        "cached_measurement_families": 3,
        "cached_measurements_reused": 9,
        "candidate_rows_deleted": False,
    }


def test_targeted_rescue_board_is_small_diverse_and_auditable() -> None:
    def candidate(
        proposal_id: str,
        period: float,
        *,
        positive_blocks: int,
        strength: float,
    ) -> dict:
        measurements = []
        for block in range(3):
            supported = block < positive_blocks
            measurements.append(
                {
                    "block": block + 1,
                    "outcome": "strong" if supported else "contradiction",
                    "snr": 8.0 if supported else 1.0,
                    "depth_ppm": 100.0 if supported else 10.0,
                }
            )
        return {
            "proposal_id": proposal_id,
            "period_days": period,
            "source_strength": strength,
            "robust_sde": strength,
            "ensemble_vote": {
                "positive_round_count": 3,
                "positive_groups": 7,
                "strong_groups": 4,
                "contradicting_groups": 2,
            },
            "vote_seed_consensus": {
                "refit_supported_seeds": 3,
                "refit_support_rate": 1.0,
                "mean_rank_percentile": 0.95,
            },
            "holdout_verification": {
                "passed": False,
                "block_measurements": measurements,
            },
        }

    rows = [
        candidate("SHORT", 2.0, positive_blocks=1, strength=8.0),
        candidate("MID-BEST", 10.0, positive_blocks=2, strength=12.0),
        candidate("MID-DUPLICATE", 10.01, positive_blocks=1, strength=50.0),
        candidate("LONG", 50.0, positive_blocks=1, strength=9.0),
        candidate("VERY-LONG", 150.0, positive_blocks=1, strength=10.0),
        candidate("EXTRA", 250.0, positive_blocks=1, strength=7.0),
    ]
    selected, deferred, diagnostics = _select_targeted_precision_rescues(
        list(enumerate(rows)),
        max_families=4,
        relative_period_separation=0.005,
    )
    selected_ids = {row["proposal_id"] for _index, row in selected}
    assert len(selected) == 4
    assert len(deferred) == 2
    assert "MID-BEST" in selected_ids
    assert "MID-DUPLICATE" not in selected_ids
    assert diagnostics["selected_dense_refit_families"] == 4
    assert diagnostics["deferred_but_retained_families"] == 2
    assert diagnostics["candidate_rows_deleted"] is False
    assert diagnostics["truth_catalog_used"] is False
    assert len(diagnostics["preliminary_rescue_board"]) == 4


def test_targeted_rescue_board_is_strict_json_serializable_without_positive_blocks() -> None:
    candidate = {
        "proposal_id": "NO-POSITIVE-BLOCK",
        "period_days": 25.0,
        "source_strength": float("nan"),
        "robust_sde": float("inf"),
        "ensemble_vote": {},
        "vote_seed_consensus": {"mean_rank_percentile": float("nan")},
        "holdout_verification": {
            "passed": False,
            "block_measurements": [{"outcome": "contradiction", "snr": -2.0, "depth_ppm": -20.0}],
        },
    }
    _selected, _deferred, diagnostics = _select_targeted_precision_rescues(
        [(0, candidate)],
        max_families=1,
    )
    json.dumps(diagnostics, allow_nan=False)


def test_targeted_board_runs_before_any_linear_refit(monkeypatch) -> None:
    rows = []
    for index, period in enumerate((2.0, 7.0, 12.0, 30.0, 120.0, 220.0)):
        rows.append(
            {
                "proposal_id": f"ROW-{index}",
                "period_days": period,
                "discovered_blocks": [1, 2],
                "source_strength": float(20 - index),
                "robust_sde": float(20 - index),
                "ensemble_vote": {
                    "positive_round_count": 3,
                    "positive_groups": 7,
                    "strong_groups": 4,
                    "contradicting_groups": 2,
                },
                "vote_seed_consensus": {
                    "refit_supported_seeds": 5,
                    "refit_support_rate": 1.0,
                    "mean_rank_percentile": 0.9,
                },
                "holdout_verification": {
                    "passed": False,
                    "block_measurements": [
                        {"outcome": "strong", "snr": 8.0, "depth_ppm": 100.0},
                        {"outcome": "strong", "snr": 7.0, "depth_ppm": 90.0},
                        {"outcome": "contradiction", "snr": 1.0, "depth_ppm": -5.0},
                    ],
                },
            }
        )

    attempted_ids = []

    def fake_linear(candidates, _blocks, _branch):
        attempted_ids.extend(row["proposal_id"] for row in candidates)
        for row in candidates:
            row["holdout_verification"] = {
                **row["holdout_verification"],
                "passed": True,
            }
        return {"backend": "fake-linear"}

    def fake_finish(index, _source, candidate, _initial, **_kwargs):
        completed = dict(candidate)
        completed["ephemeris_refit"] = {"rescued_failed_fixed_ephemeris": True}
        return index, completed

    monkeypatch.setenv("HOUXING_TARGETED_REFIT_MAX_FAMILIES", "2")
    monkeypatch.setattr(
        ensemble_module.optimization_v4,
        "try_linear_event_anchor_rescue_batch",
        fake_linear,
    )
    monkeypatch.setattr(ensemble_module, "_finish_refit_candidate", fake_finish)

    output, diagnostics = refit_failed_vote_families(
        rows,
        [[1], [2], [3]],
        SimpleNamespace(),
        mode="targeted",
    )

    assert len(attempted_ids) == 2
    assert diagnostics["eligible_families"] == 6
    assert diagnostics["attempted_families"] == 2
    assert diagnostics["targeted_shortlisted_families"] == 2
    assert diagnostics["targeted_deferred_families"] == 4
    assert diagnostics["linear_event_anchor_prefit"]["attempted_families"] == 2
    assert (
        sum(
            "targeted_rescue_deferred" in row.get("ephemeris_refit", {})
            for row in output
        )
        == 4
    )


def test_adaptive_budget_keeps_long_period_protection_and_audit_rows() -> None:
    short = {
        "proposal_id": "SHORT",
        "period_days": 12.0,
        "source_strength": 30.0,
        "ensemble_vote": {
            "rounds_measured": 5,
            "positive_round_count": 5,
            "positive_groups": 12,
            "strong_groups": 8,
            "contradicting_groups": 0,
        },
        "holdout_verification": {"block_measurements": []},
    }
    protected = {
        "proposal_id": "LONG",
        "period_days": 121.0,
        "source_strength": 25.0,
        "ensemble_vote": {
            "rounds_measured": 5,
            "positive_round_count": 4,
            "positive_groups": 8,
            "strong_groups": 3,
            "contradicting_groups": 6,
        },
        "holdout_verification": {"block_measurements": []},
    }
    full_round_sparse = {
        "proposal_id": "LONG-FULL-ROUND",
        "period_days": 195.0,
        "source_strength": 12.0,
        "ensemble_vote": {
            "rounds_measured": 5,
            "positive_round_count": 5,
            "positive_groups": 5,
            "strong_groups": 2,
            "contradicting_groups": 2,
        },
        "holdout_verification": {"block_measurements": []},
    }
    assert _adaptive_long_period_protected(protected) is True
    assert _adaptive_long_period_protected(full_round_sparse) is True
    selected, diagnostics = _select_adaptive_exact_fallbacks(
        [
            (0, short, short),
            (1, protected, protected),
            (2, full_round_sparse, full_round_sparse),
        ],
        fraction=0.0,
    )
    assert [index for index, _source in selected] == [1, 2]
    assert diagnostics["long_period_protected_families"] == 2
    assert diagnostics["deferred_families"] == 1
    assert diagnostics["candidate_rows_deleted"] is False
    assert diagnostics["truth_catalog_used"] is False


def test_vote_seed_schedule_is_deterministic_and_distinct() -> None:
    first = deterministic_vote_seeds(229449838, DEFAULT_VOTE_SEED_COUNT)
    second = deterministic_vote_seeds(229449838, DEFAULT_VOTE_SEED_COUNT)
    assert first == second
    assert len(first) == DEFAULT_VOTE_SEED_COUNT
    assert len(set(first)) == DEFAULT_VOTE_SEED_COUNT
    assert first[0] == 229449838


def test_natural_sector_vote_plan_adapts_without_target_filtering() -> None:
    large, large_plan = build_natural_sector_vote_groups(range(1, 13), rounds=5, seed=11)
    assert len(large) == 15
    assert large_plan["groups_per_round"] == 3
    assert large_plan["sector_count_used_for_target_selection"] is False

    medium, medium_plan = build_natural_sector_vote_groups(range(1, 8), rounds=5, seed=11)
    assert len(medium) == 10
    assert medium_plan["groups_per_round"] == 2
    assert all(len(row["segments"]) >= 3 for row in medium)

    small, small_plan = build_natural_sector_vote_groups([4, 8, 12, 20], rounds=5, seed=11)
    assert len(small) == 4
    assert small_plan["limited_sector_evidence"] is True
    assert {row["omitted_segment"] for row in small} == {4, 8, 12, 20}
    assert all(len(row["segments"]) == 3 for row in small)


def _row(raw_id: str, group: int, period: float, epoch: float) -> dict:
    return {
        "raw_id": raw_id,
        "source_block": group,
        "source_segments": [group],
        "method": "bls",
        "period_days": period,
        "epoch_btjd": epoch,
        "duration_hours": 2.0,
        "robust_sde": 8.0,
        "search_metrics": {"snr": 9.0},
        "source_strength": 11.8,
    }


def test_balanced_groups_cover_every_sector_once_per_round() -> None:
    segments = list(range(1, 28))
    groups = build_balanced_discovery_groups(segments, rounds=2, seed=7)
    assert len(groups) == 6
    for round_index in (1, 2):
        round_groups = [row for row in groups if row["round"] == round_index]
        flattened = [value for row in round_groups for value in row["segments"]]
        assert sorted(flattened) == segments
        for left_index, left in enumerate(round_groups):
            for right in round_groups[left_index + 1 :]:
                assert set(left["segments"]).isdisjoint(right["segments"])


def test_consensus_retains_every_local_peak_in_annotations() -> None:
    groups = [
        {"group": 1, "round": 1, "segments": [1, 4, 7]},
        {"group": 2, "round": 1, "segments": [2, 5, 8]},
        {"group": 3, "round": 1, "segments": [3, 6, 9]},
    ]
    ledger = [
        _row("A", 1, 3.69067, 100.0),
        _row("B", 2, 3.69068, 100.0),
        _row("SMALL", 3, 17.123, 101.0),
    ]
    families, diagnostics, annotations = build_consensus_families(ledger, groups)
    assert len(annotations) == len(ledger)
    assert annotations[2]["ensemble_family_id"] is None
    assert annotations[2]["ensemble_support_groups"] == 1
    assert len(families) == 1
    assert families[0]["ensemble"]["all_local_peaks_retained"] is True
    assert diagnostics["accepted_matches"] == 1


def _with_vote(
    row: dict,
    *,
    positive_groups: int,
    positive_round_count: int,
) -> dict:
    return {
        **row,
        "ensemble_vote": {
            "groups_measured": 9,
            "positive_groups": positive_groups,
            "strong_groups": positive_groups,
            "contradicting_groups": 0,
            "untestable_groups": 9 - positive_groups,
            "positive_rounds": list(range(1, positive_round_count + 1)),
            "positive_round_count": positive_round_count,
            "positive_groups_by_round": {},
            "weakest_positive_snr": 7.0 if positive_groups else None,
            "combined_positive_snr": 12.0 if positive_groups else 0.0,
            "measurements": [],
        },
    }


def test_vote_consensus_promotes_cross_block_match_without_hiding_singleton() -> None:
    ledger = [
        _with_vote(_row("A", 1, 3.69067, 100.0), positive_groups=3, positive_round_count=1),
        _with_vote(_row("B", 2, 3.69068, 100.0), positive_groups=3, positive_round_count=1),
        _with_vote(_row("SMALL", 3, 17.123, 101.0), positive_groups=1, positive_round_count=1),
    ]
    promoted, audit, diagnostics = build_vote_consensus_families(ledger)
    assert len(audit) == 2
    assert len(promoted) == 1
    assert promoted[0]["discovered_blocks"] == [1, 2]
    assert promoted[0]["ensemble"]["all_local_peaks_retained"] is True
    lower, upper = promoted[0]["balanced_family_period_bounds_days"]
    assert lower < 3.69067 < 3.69068 < upper
    assert diagnostics["unpromoted_families"] == 1


def test_sparse_sector_pro_splits_neighbours_that_drift_over_full_baseline() -> None:
    ledger = [
        _with_vote(_row("A", 1, 10.0, 100.0), positive_groups=4, positive_round_count=2),
        _with_vote(_row("B", 2, 10.04, 100.0), positive_groups=4, positive_round_count=2),
    ]
    core_promoted, core_audit, core_diagnostics = build_vote_consensus_families(ledger)
    pro_promoted, pro_audit, pro_diagnostics = build_vote_consensus_families(
        ledger,
        family_mode="sparse-sector-pro",
        baseline_days=1_000.0,
    )
    assert len(core_audit) == 1
    assert len(core_promoted) == 1
    assert core_diagnostics["candidate_family_mode"] == "core"
    assert len(pro_audit) == 2
    assert len(pro_promoted) == 2
    assert pro_diagnostics["pro_neighbour_edges_split_for_full_baseline_drift"] == 1
    assert pro_diagnostics["candidates_deleted_by_family_mode"] == 0


def test_sparse_sector_pro_still_merges_clocks_aligned_over_full_baseline() -> None:
    ledger = [
        _with_vote(_row("A", 1, 10.0, 100.0), positive_groups=4, positive_round_count=2),
        _with_vote(_row("B", 2, 10.0001, 100.0), positive_groups=4, positive_round_count=2),
    ]
    promoted, audit, diagnostics = build_vote_consensus_families(
        ledger,
        family_mode="sparse-sector-pro",
        baseline_days=1_000.0,
    )
    assert len(audit) == 1
    assert len(promoted) == 1
    assert diagnostics["accepted_matches"] == 1


def test_vote_consensus_promotes_recurrent_small_single_block_peak() -> None:
    ledger = [
        _with_vote(
            _row("SMALL", 3, 17.123, 101.0),
            positive_groups=4,
            positive_round_count=2,
        )
    ]
    promoted, audit, diagnostics = build_vote_consensus_families(ledger)
    assert len(promoted) == 1
    assert promoted[0]["discovered_blocks"] == [3]
    assert promoted[0]["ensemble"]["promotion_reasons"] == [
        "multi_group_forced_ephemeris_recurrence"
    ]
    assert audit[0]["promoted_to_global_validation"] is True
    assert diagnostics["promoted_families"] == 1


def test_multi_seed_support_retains_audit_but_gates_global_validation() -> None:
    accepted = _with_vote(
        _row("ACCEPTED", 1, 17.123, 101.0),
        positive_groups=1,
        positive_round_count=1,
    )
    rejected = _with_vote(
        _row("REJECTED", 2, 31.456, 102.0),
        positive_groups=9,
        positive_round_count=3,
    )
    accepted["vote_seed_consensus"] = {
        "seed_count": 5,
        "minimum_supported_seeds": 3,
        "ordinary_vote_supported_seeds": 3,
        "long_period_vote_supported_seeds": 0,
        "promotion_support_rate": 0.6,
        "refit_support_rate": 0.6,
        "mean_rank": 2.0,
        "rank_variance": 0.4,
    }
    rejected["vote_seed_consensus"] = {
        "seed_count": 5,
        "minimum_supported_seeds": 3,
        "ordinary_vote_supported_seeds": 2,
        "long_period_vote_supported_seeds": 0,
        "promotion_support_rate": 0.4,
        "refit_support_rate": 0.4,
        "mean_rank": 1.0,
        "rank_variance": 0.0,
    }
    promoted, audit, diagnostics = build_vote_consensus_families([accepted, rejected])
    assert [row["raw_id"] for row in promoted] == ["ACCEPTED"]
    assert promoted[0]["vote_seed_consensus"]["promotion_support_rate"] == 0.6
    assert len(audit) == 2
    assert {row["representative_raw_id"] for row in audit} == {
        "ACCEPTED",
        "REJECTED",
    }
    assert diagnostics["promoted_families"] == 1
    rejected_audit = next(row for row in audit if row["representative_raw_id"] == "REJECTED")
    assert rejected_audit["promoted_to_global_validation"] is False
    assert (
        "deterministic_vote_seed_carry_for_science_review"
        in (rejected_audit["science_review_reasons"])
    )


def test_multi_seed_refit_requires_declared_consensus_support() -> None:
    candidate = _with_vote(
        _row("A", 1, 10.97, 100.0),
        positive_groups=9,
        positive_round_count=3,
    )
    candidate["discovered_blocks"] = [1]
    candidate["holdout_verification"] = {"passed": False}
    candidate["vote_seed_consensus"] = {
        "seed_count": 5,
        "minimum_supported_seeds": 3,
        "refit_supported_seeds": 3,
    }
    assert ephemeris_refit_eligible(candidate) is True
    candidate["vote_seed_consensus"]["refit_supported_seeds"] = 2
    assert ephemeris_refit_eligible(candidate) is False
    candidate["vote_seed_consensus"]["refit_supported_seeds"] = 0
    assert ephemeris_refit_eligible(candidate) is False


def test_vote_consensus_does_not_transitively_bridge_distinct_periods() -> None:
    # A matches B and B matches C under the duration-derived uncertainty, but
    # A and C do not match.  They must form two anchored families, not one
    # union-find component spanning the whole chain.
    ledger = [
        _with_vote(_row("A", 1, 7.38, 100.0), positive_groups=9, positive_round_count=3),
        _with_vote(_row("B", 2, 7.42, 100.0), positive_groups=9, positive_round_count=3),
        _with_vote(_row("C", 3, 7.46, 100.0), positive_groups=9, positive_round_count=3),
    ]
    promoted, audit, diagnostics = build_vote_consensus_families(ledger)
    assert len(audit) == 2
    assert sorted(row["member_count"] for row in audit) == [1, 2]
    assert diagnostics["family_method"] == "direct_anchor_no_transitive_closure"
    assert diagnostics["anchored_families"] == 2
    assert len(promoted) == 2


def test_failed_cross_block_family_is_eligible_for_global_ephemeris_refit() -> None:
    candidate = _with_vote(
        _row("A", 1, 38.551, 100.0),
        positive_groups=8,
        positive_round_count=3,
    )
    candidate["discovered_blocks"] = [1, 2]
    candidate["holdout_verification"] = {"passed": False}
    assert ephemeris_refit_eligible(candidate) is True


def test_single_block_refit_requires_repeated_strong_vote_evidence() -> None:
    candidate = _with_vote(
        _row("A", 1, 10.97, 100.0),
        positive_groups=5,
        positive_round_count=3,
    )
    candidate["discovered_blocks"] = [1]
    candidate["holdout_verification"] = {"passed": False}
    candidate["ensemble_vote"]["strong_groups"] = 2
    assert ephemeris_refit_eligible(candidate) is True
    candidate["ensemble_vote"]["strong_groups"] = 1
    assert ephemeris_refit_eligible(candidate) is False


def test_passing_family_is_not_refit() -> None:
    candidate = _with_vote(
        _row("A", 1, 10.97, 100.0),
        positive_groups=9,
        positive_round_count=3,
    )
    candidate["discovered_blocks"] = [1, 2]
    candidate["holdout_verification"] = {"passed": True}
    assert ephemeris_refit_eligible(candidate) is False


def test_pre_refit_harmonic_plan_groups_but_never_removes_members() -> None:
    short = _with_vote(
        _row("SHORT", 1, 5.0, 2.0),
        positive_groups=9,
        positive_round_count=3,
    )
    long = _with_vote(
        _row("LONG", 2, 10.0, 2.0),
        positive_groups=8,
        positive_round_count=3,
    )
    unrelated = _with_vote(
        _row("OTHER", 3, 10.0, 4.5),
        positive_groups=7,
        positive_round_count=3,
    )
    for index, candidate in enumerate((short, long, unrelated), start=1):
        candidate["proposal_id"] = f"VOTE-{index}"
        candidate["discovered_blocks"] = [1, 2]
        candidate["holdout_verification"] = {"passed": False}
    relation = _pre_refit_integer_harmonic_relation(short, long)
    assert relation is not None
    assert relation["candidate_removed"] is False
    assert _pre_refit_integer_harmonic_relation(short, unrelated) is None
    items = list(enumerate((short, long, unrelated)))
    groups, relations = _plan_pre_refit_harmonic_groups(items)
    flattened = [index for group in groups for index, _candidate in group]
    assert sorted(flattened) == [0, 1, 2]
    assert len(flattened) == len(set(flattened))
    assert len(relations) == 1


def test_pre_refit_harmonic_plan_requires_measured_shared_low_points() -> None:
    rng = np.random.default_rng(17)
    times = []
    fluxes = []
    segments = []
    for segment, start in enumerate((0.0, 40.0), start=1):
        time = np.arange(start, start + 30.0, 0.02)
        flux = 1.0 + rng.normal(0.0, 0.0001, time.size)
        phase = (time - 2.0 + 2.5) % 5.0 - 2.5
        flux[np.abs(phase) <= 0.05] -= 0.01
        times.append(time)
        fluxes.append(flux)
        segments.append(np.full(time.size, segment, dtype=int))
    branch = SimpleNamespace(
        time=np.concatenate(times),
        flux=np.concatenate(fluxes),
        segment_id=np.concatenate(segments),
    )
    candidates = []
    for index, period in enumerate((5.0, 10.0), start=1):
        candidate = _with_vote(
            _row(f"ROW-{index}", index, period, 2.0),
            positive_groups=9,
            positive_round_count=3,
        )
        candidate["proposal_id"] = f"VOTE-{index}"
        candidate["discovered_blocks"] = [1, 2]
        candidate["holdout_verification"] = {"passed": False}
        candidates.append(candidate)

    groups, relations = _plan_pre_refit_harmonic_groups(
        list(enumerate(candidates)),
        branch=branch,
        require_event_proof=True,
    )

    assert len(groups) == 1
    assert [index for index, _candidate in groups[0]] == [0, 1]
    assert relations[0]["event_backed"] is True
    assert relations[0]["matched_supported_events"] >= 3
    assert len(relations[0]["matched_segments"]) == 2
    assert relations[0]["candidate_grid_membership_preserved"] is True


def test_near_period_duplicate_requires_same_measured_low_points() -> None:
    representative = {
        "proposal_id": "REP",
        "period_days": 5.0,
        "epoch_btjd": 2.0,
        "duration_hours": 3.0,
        "holdout_verification": {
            "event_repeatability": {
                "event_support": [
                    {
                        "segment": segment,
                        "center_btjd": center,
                        "supported": True,
                    }
                    for segment, center in (
                        (1, 2.0),
                        (1, 7.0),
                        (2, 32.0),
                        (2, 37.0),
                    )
                ]
            }
        },
    }
    duplicate = {
        "proposal_id": "DUP",
        "period_days": 5.0001,
        "epoch_btjd": 2.01,
        "duration_hours": 3.0,
        "holdout_verification": {
            "event_repeatability": {
                "event_support": [
                    {
                        "segment": segment,
                        "center_btjd": center + 0.01,
                        "supported": True,
                    }
                    for segment, center in (
                        (1, 2.0),
                        (1, 7.0),
                        (2, 32.0),
                        (2, 37.0),
                    )
                ]
            }
        },
    }
    decision = _near_period_event_train_duplicate(representative, duplicate)
    assert decision is not None
    assert decision["matched_supported_events"] == 4
    assert decision["matched_segments"] == [1, 2]
    assert decision["candidate_removed_from_audit"] is False
    arithmetic_only = {
        **duplicate,
        "proposal_id": "NO-EVENT-EVIDENCE",
        "holdout_verification": {"event_repeatability": {"event_support": []}},
    }
    assert _near_period_event_train_duplicate(representative, arithmetic_only) is None


def test_vote_thresholds_scale_with_more_rounds() -> None:
    thresholds = _scaled_vote_thresholds(
        {
            "groups_measured": 15,
            "rounds_measured": 5,
            "positive_groups_by_round": {str(index): 0 for index in range(1, 6)},
        }
    )
    assert thresholds == {
        "promotion_positive_groups": 7,
        "promotion_positive_rounds": 4,
        "refit_positive_groups": 9,
        "refit_strong_groups": 4,
        "refit_positive_rounds": 4,
    }


def test_large_groups_rescue_sparse_long_period_family() -> None:
    candidate = _with_vote(
        _row("LONG", 1, 120.0, 100.0),
        positive_groups=3,
        positive_round_count=2,
    )
    candidate["ensemble_vote"].update(
        {
            "groups_measured": 15,
            "rounds_measured": 5,
            "strong_groups": 2,
            "contradicting_groups": 8,
            "positive_groups_by_round": {str(value): 0 for value in range(1, 6)},
        }
    )
    candidate["long_period_vote"] = {
        "groups_measured": 10,
        "rounds_measured": 5,
        "positive_groups": 5,
        "strong_groups": 3,
        "contradicting_groups": 2,
        "untestable_groups": 3,
        "positive_rounds": [1, 2, 3, 4],
        "positive_round_count": 4,
        "positive_groups_by_round": {str(value): 1 for value in range(1, 6)},
        "weakest_positive_snr": 3.2,
        "combined_positive_snr": 9.0,
        "measurements": [],
    }
    promoted, audit, _diagnostics = build_vote_consensus_families([candidate])
    assert len(promoted) == 1
    assert "large_group_long_period_recurrence_rescue" in (audit[0]["promotion_reasons"])
