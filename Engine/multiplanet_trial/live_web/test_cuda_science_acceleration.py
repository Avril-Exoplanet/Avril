from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from multiplanet_trial.live_web import cuda_science_acceleration as cuda_science
from multiplanet_trial.live_web import ensemble_intersection as ensemble


def test_cuda_refit_finalizer_accepts_explicit_context(monkeypatch):
    branch = object()
    blocks = [[1], [2], [3]]
    calls = []

    monkeypatch.setattr(ensemble, "_REFIT_BRANCH", None)
    monkeypatch.setattr(ensemble, "_REFIT_BLOCKS", None)
    monkeypatch.setattr(
        ensemble.core,
        "_attach_event_repeatability",
        lambda candidate, verification, active: calls.append(
            ("repeatability", active)
        ),
    )
    monkeypatch.setattr(
        ensemble.core,
        "_apply_fold_event_direction_test",
        lambda candidate, active: calls.append(("direction", active)),
    )
    monkeypatch.setattr(
        ensemble,
        "annotate_event_coherence",
        lambda candidate, active: calls.append(("coherence", active)),
    )
    source = {
        "period_days": 10.0,
        "epoch_btjd": 100.0,
        "duration_hours": 2.0,
        "holdout_verification": {"passed": False},
    }
    candidate = {
        **deepcopy(source),
        "period_days": 10.01,
        "holdout_verification": {"passed": True},
    }
    _index, finished = ensemble._finish_refit_candidate(
        0,
        source,
        candidate,
        deepcopy(source["holdout_verification"]),
        branch=branch,
        blocks=blocks,
        execution="cuda-test",
    )

    assert calls == [
        ("repeatability", branch),
        ("direction", blocks),
        ("coherence", blocks),
    ]
    assert finished["ephemeris_refit"]["execution"] == "cuda-test"


def test_cuda_targeted_board_precedes_linear_and_dense_refit(monkeypatch):
    rows = []
    for index, period in enumerate((2.0, 7.0, 12.0, 30.0, 120.0, 220.0)):
        rows.append(
            {
                "proposal_id": f"CUDA-{index}",
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
        cuda_science.optimization_v4,
        "try_linear_event_anchor_rescue_batch",
        fake_linear,
    )
    monkeypatch.setattr(ensemble, "_finish_refit_candidate", fake_finish)

    output, diagnostics = cuda_science.refit_failed_vote_families_cuda(
        rows,
        [[1], [2], [3]],
        SimpleNamespace(),
        mode="targeted",
    )

    assert len(attempted_ids) == 2
    assert diagnostics["eligible_families"] == 6
    assert diagnostics["attempted_families"] == 2
    assert diagnostics["targeted_rescue_decision"][
        "selected_before_any_refit_compute"
    ] is True
    assert (
        sum(
            "targeted_rescue_deferred" in row.get("ephemeris_refit", {})
            for row in output
        )
        == 4
    )


def test_cuda_solver_scans_full_grid_then_cpu_corrects_selected(monkeypatch):
    monkeypatch.setenv("VLORA_CUDA_EXACT_CORRECTION", "cpu")
    prepared = object()
    exact_calls = []
    monkeypatch.setattr(
        cuda_science,
        "prepare_exact_bls",
        lambda _time, _flux: prepared,
    )

    def fake_exact(
        time_values,
        flux_values,
        *,
        periods,
        durations,
        threads,
        prepared,
    ):
        exact_calls.append((periods.copy(), durations.copy(), threads, prepared))
        return {
            "power": np.arange(periods.size, dtype=float) + 100.0,
            "epoch_btjd": periods + 2000.0,
            "backend": "test-exact",
        }

    monkeypatch.setattr(cuda_science, "exact_bls_solutions", fake_exact)

    class FakeEngine:
        maximum_shared_bytes = 64 * 1024

        def __init__(self):
            self.solve_periods = []

        def bind(self, **_kwargs):
            return object()

        def solve_bound(self, _binding, *, periods, durations, **_kwargs):
            self.solve_periods.append(periods.copy())
            return SimpleNamespace(
                power=np.full(periods.size, 7.0),
                epoch_btjd=np.full(periods.size, -1.0),
                duration_days=np.full(periods.size, durations[0]),
            )

    engine = FakeEngine()
    solver = cuda_science.CudaBoundBLSSolver(
        np.linspace(0.0, 10.0, 100),
        np.sin(np.linspace(0.0, 10.0, 100)) * 1e-4,
        engine,
    )
    complete = np.linspace(1.0, 2.0, 1001)
    selected = complete[[0, 100, 500, 900, 1000]]
    durations = np.asarray([0.1])

    gpu_power = solver.power(periods=complete, durations=durations)
    corrected = solver(periods=selected, durations=durations)

    assert np.all(gpu_power == 7.0)
    assert len(engine.solve_periods) == 1
    assert np.array_equal(engine.solve_periods[0], complete)
    assert len(exact_calls) == 1
    assert np.array_equal(exact_calls[0][0], selected)
    assert exact_calls[0][3] is prepared
    assert np.array_equal(corrected["power"], np.arange(5) + 100.0)
    assert np.array_equal(corrected["epoch_btjd"], selected + 2000.0)


def test_cuda_solver_reuses_complete_gpu_grid_by_default(monkeypatch):
    monkeypatch.delenv("VLORA_CUDA_EXACT_CORRECTION", raising=False)
    monkeypatch.setattr(
        cuda_science,
        "prepare_exact_bls",
        lambda _time, _flux: object(),
    )

    class FakeEngine:
        maximum_shared_bytes = 64 * 1024

        def bind(self, **_kwargs):
            return object()

        def solve_bound(self, _binding, *, periods, durations, **_kwargs):
            return SimpleNamespace(
                power=periods + 10.0,
                epoch_btjd=periods + 1000.0,
                duration_days=np.full(periods.size, durations[0]),
            )

    solver = cuda_science.CudaBoundBLSSolver(
        np.linspace(0.0, 10.0, 100),
        np.sin(np.linspace(0.0, 10.0, 100)) * 1e-4,
        FakeEngine(),
    )
    complete = np.linspace(1.0, 2.0, 1001)
    selected = complete[[0, 100, 500, 900, 1000]]
    durations = np.asarray([0.1])

    solver.power(periods=complete, durations=durations)
    solved = solver(periods=selected, durations=durations)

    assert np.array_equal(solved["power"], selected + 10.0)
    assert np.array_equal(solved["epoch_btjd"], selected + 1000.0)
    assert solver.diagnostics()["selected_period_solution_source"] == (
        "cuda-complete-grid-deterministic-int64-fp64"
    )
