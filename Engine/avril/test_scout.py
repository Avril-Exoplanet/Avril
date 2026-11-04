from __future__ import annotations

import numpy as np

from avril.scout import (
    ScoutConfig,
    _catalog_relation,
    _period_relation,
    _quick_candidate_evidence,
    bin_prepared_curve,
    mask_locked_ephemerides,
    merge_channel_peaks,
    run_bls_scout,
)


def test_segment_binning_never_crosses_observing_gaps() -> None:
    time = np.asarray([0.0, 0.001, 0.002, 100.0, 100.001])
    flux = np.asarray([1.0, 0.9, 1.1, 2.0, 2.2])
    segment = np.asarray([1, 1, 1, 2, 2])
    binned_time, binned_flux, binned_segment = bin_prepared_curve(
        time, flux, segment, bin_minutes=10.0
    )
    assert binned_time.size == 2
    assert binned_segment.tolist() == [1, 2]
    assert np.allclose(binned_flux, [1.0, 2.1])


def test_union_merges_near_duplicates_but_keeps_harmonics() -> None:
    tls = {
        "peaks": [
            {"source": "tls", "source_rank": 1, "period_days": 10.0, "power": 9.0, "robust_power_z": 8.0, "power_percentile": 99.9},
            {"source": "tls", "source_rank": 2, "period_days": 20.0, "power": 8.0, "robust_power_z": 7.0, "power_percentile": 99.0},
        ]
    }
    bls = {
        "peaks": [
            {"source": "bls", "source_rank": 1, "period_days": 10.005, "power": 7.0, "robust_power_z": 6.0, "power_percentile": 98.0},
        ]
    }
    merged = merge_channel_peaks((tls, bls))
    assert len(merged) == 2
    assert merged[0]["channel_support_count"] == 2
    assert any(abs(row["period_days"] - 20.0) < 1e-12 for row in merged)


def test_fast_bls_recovers_synthetic_transit() -> None:
    rng = np.random.default_rng(7)
    time = np.linspace(0.0, 100.0, 10_000)
    period = 6.173
    phase = (time - 0.41 + 0.5 * period) % period - 0.5 * period
    flux = 1.0 + rng.normal(0.0, 0.0008, size=time.size)
    flux[np.abs(phase) <= 0.06] -= 0.004
    result = run_bls_scout(
        time,
        flux,
        ScoutConfig(
            period_max_days=12.0,
            bls_periods=8_000,
            max_peaks_per_channel=12,
        ),
    )
    nearest = min(
        result["peaks"], key=lambda row: abs(row["period_days"] - period)
    )
    assert abs(nearest["period_days"] - period) / period < 0.002


def test_adaptive_config_rejects_invalid_cadence_target() -> None:
    try:
        ScoutConfig(target_scout_cadences=999)
    except ValueError as exc:
        assert "target_scout_cadences" in str(exc)
    else:  # pragma: no cover - explicit failure message
        raise AssertionError("invalid Scout cadence target was accepted")


def test_period_arithmetic_alone_does_not_prove_catalog_alias() -> None:
    assert _period_relation(10.0, 20.0)["type"] == "integer_harmonic"
    candidate = {
        "period_days": 10.0,
        "epoch_btjd": 1.0,
        "duration_hours": 2.0,
        "holdout_verification": {"event_repeatability": {"event_support": []}},
    }
    catalog = {"period": 20.0, "epochBtjd": 1.0}
    assert _catalog_relation(candidate, catalog) is None


def test_event_backed_catalog_alias_is_routed_as_known() -> None:
    candidate = {
        "period_days": 10.0,
        "epoch_btjd": 1.0,
        "duration_hours": 2.0,
        "holdout_verification": {
            "event_repeatability": {
                "event_support": [
                    {"segment": 1, "center_btjd": 1.0, "supported": True},
                    {"segment": 2, "center_btjd": 21.0, "supported": True},
                    {"segment": 3, "center_btjd": 41.0, "supported": True},
                ]
            }
        },
    }
    catalog = {"period": 20.0, "epochBtjd": 1.0}
    relation = _catalog_relation(candidate, catalog)
    assert relation is not None
    assert relation["type"] == "integer_harmonic"
    assert relation["matched_supported_events"] == 3


def test_event_train_subset_can_prove_non_integer_known_alias() -> None:
    candidate = {
        "period_days": 13.7,
        "epoch_btjd": 1.0,
        "duration_hours": 2.0,
        "holdout_verification": {
            "event_repeatability": {
                "event_support": [
                    {"segment": 1, "center_btjd": 1.0, "supported": True},
                    {"segment": 2, "center_btjd": 21.0, "supported": True},
                    {"segment": 3, "center_btjd": 41.0, "supported": True},
                ]
            }
        },
    }
    relation = _catalog_relation(
        candidate, {"period": 20.0, "epochBtjd": 1.0}
    )
    assert relation is not None
    assert relation["type"] == "known_event_train_subset"
    assert relation["supported_event_containment"] == 1.0


def test_quick_evidence_uses_joint_direction_not_every_event_three_sigma() -> None:
    candidate = {
        "sources": ["bls", "tls"],
        "source_ranks": {"bls": 7, "tls": 5},
        "channel_support_count": 2,
        "holdout_verification": {
            "event_repeatability": {
                "event_support": [
                    {
                        "segment": index % 3,
                        "depth_ppm": 350.0,
                        "snr": 2.8,
                        "supported": False,
                    }
                    for index in range(9)
                ]
                + [
                    {
                        "segment": 4,
                        "depth_ppm": -30.0,
                        "snr": -0.3,
                        "supported": False,
                    }
                ]
            }
        },
    }
    evidence = _quick_candidate_evidence(candidate)
    assert evidence["repeated_direction_passed"] is True
    assert evidence["repeated_supported_passed"] is False
    assert evidence["quick_event_evidence_passed"] is True


def test_quick_evidence_rejects_mixed_direction_events() -> None:
    rows = []
    for index in range(10):
        depth = 400.0 if index % 2 == 0 else -400.0
        rows.append(
            {
                "segment": index % 3,
                "depth_ppm": depth,
                "snr": 4.0 if depth > 0 else -4.0,
                "supported": depth > 0,
            }
        )
    evidence = _quick_candidate_evidence(
        {
            "source_ranks": {"bls": 1, "tls": 1},
            "channel_support_count": 2,
            "holdout_verification": {
                "event_repeatability": {"event_support": rows}
            },
        }
    )
    assert evidence["quick_event_evidence_passed"] is False


def test_known_signal_mask_changes_only_disposable_scout_arrays() -> None:
    time = np.linspace(0.0, 20.0, 20_001)
    flux = np.ones_like(time)
    segments = np.ones(time.size, dtype=int)
    original_flux = flux.copy()
    remaining_time, remaining_flux, remaining_segments, audit = (
        mask_locked_ephemerides(
            time,
            flux,
            segments,
            [
                {
                    "proposal_id": "known",
                    "period_days": 5.0,
                    "epoch_btjd": 1.0,
                    "duration_hours": 2.0,
                }
            ],
        )
    )
    assert remaining_time.size < time.size
    assert remaining_flux.size == remaining_time.size
    assert remaining_segments.size == remaining_time.size
    assert audit["masked_cadences"] == time.size - remaining_time.size
    assert np.array_equal(flux, original_flux)
