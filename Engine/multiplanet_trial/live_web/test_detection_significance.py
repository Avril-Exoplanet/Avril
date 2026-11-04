from types import SimpleNamespace

import numpy as np

from .detection_significance import (
    _event_gap_diagnostics,
    _harmonic_warnings,
    annotate_detection_significance,
    annotate_detection_significance_cuda_pipeline,
    annotate_detection_significance_parallel,
    candidate_detection_significance,
)


def test_gap_adjacent_dominant_event_is_flagged_when_remainder_is_weak() -> None:
    duration_days = 0.2
    cadence = 0.01
    full = np.arange(0.0, 10.0, cadence)
    # The dominant event occurs immediately after a long cadence gap and has
    # no usable left baseline.  Two complete events remain, but only one is a
    # modest >=3-sigma event, so the train must not be auto-promoted.
    compromised = full[(full < 4.0) | (full >= 5.0)]
    views = {
        1: {"time": compromised, "flux": np.ones(compromised.size), "cadence_days": cadence},
        2: {"time": full, "flux": np.ones(full.size), "cadence_days": cadence},
        3: {"time": full, "flux": np.ones(full.size), "cadence_days": cadence},
    }
    events = [
        {"segment": 1, "cycle": 0, "center_btjd": 5.02, "depth_ppm": 600.0, "duration_noise_ppm": 30.0, "ses_like": 20.0},
        {"segment": 2, "cycle": 1, "center_btjd": 5.0, "depth_ppm": 100.0, "duration_noise_ppm": 32.0, "ses_like": 3.125},
        {"segment": 3, "cycle": 2, "center_btjd": 5.0, "depth_ppm": -10.0, "duration_noise_ppm": 30.0, "ses_like": -0.333},
    ]

    diagnostics = _event_gap_diagnostics(
        views, events, duration_days=duration_days
    )

    assert diagnostics["dominant_event_segment"] == 1
    assert diagnostics["dominant_event_gap_compromised"] is True
    assert diagnostics["supported_uncompromised_events"] == 1
    assert diagnostics["gap_dominated_weak_remainder"] is True


def test_gap_adjacent_event_does_not_penalize_strong_complete_remainder() -> None:
    duration_days = 0.2
    cadence = 0.01
    full = np.arange(0.0, 10.0, cadence)
    compromised = full[(full < 4.0) | (full >= 5.0)]
    views = {
        1: {"time": compromised, "flux": np.ones(compromised.size), "cadence_days": cadence},
        2: {"time": full, "flux": np.ones(full.size), "cadence_days": cadence},
        3: {"time": full, "flux": np.ones(full.size), "cadence_days": cadence},
    }
    events = [
        {"segment": 1, "cycle": 0, "center_btjd": 5.02, "depth_ppm": 600.0, "duration_noise_ppm": 30.0, "ses_like": 20.0},
        {"segment": 2, "cycle": 1, "center_btjd": 5.0, "depth_ppm": 300.0, "duration_noise_ppm": 30.0, "ses_like": 10.0},
        {"segment": 3, "cycle": 2, "center_btjd": 5.0, "depth_ppm": 270.0, "duration_noise_ppm": 30.0, "ses_like": 9.0},
    ]

    diagnostics = _event_gap_diagnostics(
        views, events, duration_days=duration_days
    )

    assert diagnostics["dominant_event_gap_compromised"] is True
    assert diagnostics["supported_uncompromised_events"] == 2
    assert diagnostics["gap_dominated_weak_remainder"] is False


def _synthetic_branch():
    rng = np.random.default_rng(42)
    times = []
    fluxes = []
    segments = []
    for segment, start in enumerate((0.0, 30.0, 60.0), start=1):
        time = np.arange(start, start + 20.0, 0.02)
        flux = 1.0 + rng.normal(0.0, 0.0004, time.size)
        phase = (time - 1.0 + 1.0) % 2.0 - 1.0
        flux[np.abs(phase) <= 0.05] -= 0.004
        times.append(time)
        fluxes.append(flux)
        segments.append(np.full(time.size, segment, dtype=int))
    return SimpleNamespace(
        time=np.concatenate(times),
        flux=np.concatenate(fluxes),
        segment_id=np.concatenate(segments),
    )


def test_significance_is_non_destructive_and_marks_integer_harmonic() -> None:
    branch = _synthetic_branch()
    candidates = [
        {
            "candidate_id": "base",
            "rank": 1,
            "period_days": 2.0,
            "epoch_btjd": 1.0,
            "duration_hours": 2.4,
            "discovered_blocks": [1, 2, 3],
        },
        {
            "candidate_id": "harmonic",
            "rank": 2,
            "period_days": 4.0,
            "epoch_btjd": 1.0,
            "duration_hours": 2.4,
            "discovered_blocks": [1],
        },
        {
            "candidate_id": "noise",
            "rank": 3,
            "period_days": 3.1,
            "epoch_btjd": 0.4,
            "duration_hours": 2.4,
            "discovered_blocks": [1],
        },
    ]
    original = [
        (row["rank"], row["period_days"], row["epoch_btjd"], row["duration_hours"])
        for row in candidates
    ]

    output, report = annotate_detection_significance(
        candidates,
        branch,
        [[1], [2], [3]],
        phase_trials=64,
        search_trial_factor=1000,
    )

    assert [
        (row["rank"], row["period_days"], row["epoch_btjd"], row["duration_hours"])
        for row in output
    ] == original
    assert output[0]["detection_significance"]["combined"]["mes_like"] > 7.1
    assert output[0]["detection_significance"]["official_spoc_mes"] is False
    harmonic = output[1]["detection_significance"]
    assert harmonic["status"] == "integer_harmonic_not_independent"
    assert harmonic["harmonic_warnings"][0]["integer_order"] == 2
    assert output[2]["detection_significance"]["passes_7p1_reference"] is False
    assert report["locked_ephemerides_verified"] is True


def test_weak_fractional_period_alias_cannot_veto_repeating_long_period() -> None:
    longer = {
        "candidate_id": "long",
        "rank": 1,
        "period_days": 7.885,
        "epoch_btjd": 100.0,
        "duration_hours": 2.4,
        "holdout_verification": {
            "event_repeatability": {"support_fraction": 0.8}
        },
    }
    shorter = {
        "candidate_id": "short",
        "rank": 2,
        "period_days": 1.577,
        "epoch_btjd": 100.0,
        "duration_hours": 2.4,
        "holdout_verification": {
            "event_repeatability": {"support_fraction": 0.33}
        },
    }
    assert _harmonic_warnings(longer, [longer, shorter]) == []
    shorter["holdout_verification"]["event_repeatability"]["support_fraction"] = 0.9
    assert _harmonic_warnings(longer, [longer, shorter])[0]["integer_order"] == 5


def test_nonharmonic_candidate_mask_cannot_erase_locked_candidate_event() -> None:
    rng = np.random.default_rng(7)
    time = np.arange(0.0, 5.0, 0.002)
    flux = 1.0 + rng.normal(0.0, 0.0003, time.size)
    flux[np.abs(time - 2.0) <= 0.05] -= 0.004
    branch = SimpleNamespace(
        time=time,
        flux=flux,
        segment_id=np.ones(time.size, dtype=int),
    )
    candidates = [
        {
            "candidate_id": "locked",
            "rank": 1,
            "period_days": 10.0,
            "epoch_btjd": 2.0,
            "duration_hours": 2.4,
            "discovered_blocks": [1],
        },
        {
            "candidate_id": "nearby-nonharmonic",
            "rank": 2,
            "period_days": 10.03,
            "epoch_btjd": 2.0,
            "duration_hours": 2.4,
            "discovered_blocks": [1],
        },
    ]

    output, _report = annotate_detection_significance(
        candidates,
        branch,
        [[1], [], []],
        phase_trials=16,
        search_trial_factor=100,
    )

    significance = output[0]["detection_significance"]
    assert significance["event_count"] == 1
    assert significance["events"][0]["depth_ppm"] > 0


def test_dense_retained_ledger_does_not_erase_phase_shift_null() -> None:
    branch = _synthetic_branch()
    locked = {
        "candidate_id": "locked",
        "rank": 1,
        "period_days": 2.0,
        "epoch_btjd": 1.0,
        "duration_hours": 2.4,
        "discovered_blocks": [1, 2, 3],
    }
    candidates = [locked]
    for index in range(80):
        candidates.append(
            {
                "candidate_id": f"broad-ledger-{index}",
                "rank": index + 2,
                "period_days": 0.73 + 0.017 * index,
                "epoch_btjd": 0.1 + 0.019 * index,
                "duration_hours": 4.8,
                "discovered_blocks": [1],
            }
        )

    result = candidate_detection_significance(
        locked,
        candidates,
        branch,
        [[1], [2], [3]],
        phase_trials=64,
        search_trial_factor=1000,
    )

    assert result["combined"]["mes_like"] > 7.1
    assert result["phase_shift_null"]["valid_trials"] > 0


def test_parallel_significance_is_exact_and_order_preserving() -> None:
    branch = _synthetic_branch()
    candidates = [
        {
            "candidate_id": f"candidate-{index}",
            "rank": index + 1,
            "period_days": period,
            "epoch_btjd": 1.0,
            "duration_hours": 2.4,
            "discovered_blocks": [1, 2, 3],
        }
        for index, period in enumerate((2.0, 4.0, 3.1, 5.2))
    ]
    expected, _ = annotate_detection_significance(
        candidates,
        branch,
        [[1], [2], [3]],
        phase_trials=16,
        search_trial_factor=1000,
    )
    actual, report = annotate_detection_significance_parallel(
        candidates,
        branch,
        [[1], [2], [3]],
        phase_trials=16,
        search_trial_factor=1000,
        processes=2,
        threads_per_process=1,
    )

    assert actual == expected
    assert [row["candidate_id"] for row in actual] == [
        row["candidate_id"] for row in candidates
    ]
    assert report["processes"] == 2
    assert report["sector_views_cached_per_worker"] is True


def test_cuda_pipeline_preparation_is_exact_with_cpu_null(monkeypatch) -> None:
    # The orchestration is backend-agnostic: using the CPU exact null here
    # verifies double buffering without requiring CUDA in CI.
    monkeypatch.setenv("VLORA_CUDA_PHASE_NULL", "off")
    branch = _synthetic_branch()
    candidates = [
        {
            "candidate_id": f"candidate-{index}",
            "rank": index + 1,
            "period_days": period,
            "epoch_btjd": 1.0,
            "duration_hours": 2.4,
            "discovered_blocks": [1, 2, 3],
        }
        for index, period in enumerate((2.0, 4.0, 3.1))
    ]
    expected, _ = annotate_detection_significance(
        candidates,
        branch,
        [[1], [2], [3]],
        phase_trials=16,
        search_trial_factor=1000,
    )
    actual, report = annotate_detection_significance_cuda_pipeline(
        candidates,
        branch,
        [[1], [2], [3]],
        phase_trials=16,
        search_trial_factor=1000,
        preparation_workers=2,
        preparation_ahead=2,
    )
    assert actual == expected
    assert report["cpu_gpu_double_buffered"] is True
    assert report["sector_views_cached_once"] is True
