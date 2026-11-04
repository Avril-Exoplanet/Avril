from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from . import meaningful_candidate_gate as gate


def _candidate(identifier: str, period: float, coherence: str) -> dict:
    return {
        "candidate_id": identifier,
        "proposal_id": identifier,
        "period_days": period,
        "epoch_btjd": 2.0,
        "duration_hours": 2.4,
        "event_coherence": {"level": coherence},
        "holdout_verification": {},
    }


def _repeatability(period: float) -> dict:
    rows = [
        {
            "segment": 1,
            "cycle": 0,
            "center_btjd": 2.0,
            "depth_ppm": 800.0,
            "snr": 5.0,
            "supported": True,
            "sealed": False,
        },
        {
            "segment": 1,
            "cycle": 1,
            "center_btjd": 2.0 + period,
            "depth_ppm": 750.0,
            "snr": 4.5,
            "supported": True,
            "sealed": False,
        },
        {
            "segment": 2,
            "cycle": 4,
            "center_btjd": 2.0 + 4 * period,
            "depth_ppm": 700.0,
            "snr": 4.0,
            "supported": True,
            "sealed": False,
        },
    ]
    return {
        "testable_events": len(rows),
        "supported_events": len(rows),
        "support_fraction": 1.0,
        "event_support": rows,
    }


def test_funnel_keeps_audit_but_vets_only_unexplained_candidate(monkeypatch) -> None:
    time_one = np.linspace(0.0, 10.0, 2_001)
    time_two = np.linspace(20.0, 32.0, 2_401)
    branch = SimpleNamespace(
        time=np.concatenate([time_one, time_two]),
        flux=np.ones(time_one.size + time_two.size),
        segment_id=np.concatenate(
            [np.ones(time_one.size, dtype=int), np.full(time_two.size, 2, dtype=int)]
        ),
    )
    candidates = [
        _candidate("known", 5.0, "repeated_coherent_event_train"),
        _candidate("new", 6.0, "repeated_coherent_event_train"),
        _candidate("noise", 8.0, "mixed_direction_event_train"),
    ]

    def fake_metrics(rows, **_kwargs):
        return [_repeatability(float(row["period_days"])) for row in rows]

    monkeypatch.setattr(gate, "event_repeatability_metrics_batch", fake_metrics)
    annotated, meaningful, report = gate.apply_meaningful_candidate_gate(
        candidates,
        branch,
        catalog=[{"name": "TOI-test", "period": 5.0, "epochBtjd": 2.0}],
        threads=2,
    )

    assert len(annotated) == 3
    assert [row["candidate_id"] for row in meaningful] == ["new"]
    assert annotated[0]["meaningful_candidate_gate"]["disposition"] == (
        "known_catalog_period"
    )
    assert annotated[2]["meaningful_candidate_gate"]["disposition"] == (
        "event_incoherent"
    )
    assert report["meaningful_candidates"] == 1
    assert report["short_circuited_candidates"] == 2


def test_catalog_arithmetic_never_proves_alias_without_event_overlap() -> None:
    candidate = _candidate("alias", 10.0, "repeated_coherent_event_train")
    candidate["holdout_verification"] = {
        "event_repeatability": {"event_support": []}
    }
    assert gate._period_relation(10.0, 20.0)["type"] == "integer_harmonic"
    assert gate._catalog_relation(
        candidate, {"period": 20.0, "epochBtjd": 2.0}
    ) is None


def test_funnel_reuses_detailed_events_and_short_circuits_gap_signal(
    monkeypatch,
) -> None:
    time = np.linspace(0.0, 40.0, 4_001)
    branch = SimpleNamespace(
        time=time,
        flux=np.ones(time.size),
        segment_id=np.ones(time.size, dtype=int),
    )
    candidate = _candidate(
        "gap",
        6.0,
        "repeated_coherent_event_train",
    )
    candidate["holdout_verification"] = {
        "event_repeatability": _repeatability(6.0)
    }

    def fail_recomputation(*_args, **_kwargs):
        raise AssertionError("stored event measurements must be reused")

    monkeypatch.setattr(
        gate,
        "event_repeatability_metrics_batch",
        fail_recomputation,
    )
    monkeypatch.setattr(
        gate,
        "_gap_diagnostics",
        lambda *_args, **_kwargs: {
            "gap_dominated_weak_remainder": True,
            "gap_adjacent_supported_events": 2,
        },
    )
    annotated, meaningful, report = gate.apply_meaningful_candidate_gate(
        [candidate],
        branch,
    )

    assert meaningful == []
    assert annotated[0]["meaningful_candidate_gate"]["disposition"] == (
        "gap_adjacent_event_dominated"
    )
    assert report["event_metrics_reused"] == 1
    assert report["event_metrics_recomputed"] == 0
