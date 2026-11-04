"""Contract tests for the opt-in CUDA proposal path without requiring a GPU."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.timeseries import BoxLeastSquares

from multiplanet_trial.v07_eval.multibranch_search import (
    SearchBranch,
    search_branch_bls,
)
from multiplanet_trial.live_web import cuda_hybrid_search


def _branch() -> SearchBranch:
    time = np.linspace(0.0, 24.0, 1600)
    flux = np.ones_like(time)
    phase = (time - 0.3 + 1.15) % 2.3 - 1.15
    flux[np.abs(phase) < 0.07] -= 0.006
    return SearchBranch(
        key="test",
        label="test",
        processing="synthetic-unit-test-only",
        photometry_family="unit_test",
        method_family="bls",
        time=time,
        flux=flux,
        segment_id=np.ones(time.size, dtype=int),
    )


def _exact_provider(*, time, flux, scatter, periods, durations):
    model = BoxLeastSquares(
        np.asarray(time) * u.day,
        np.asarray(flux),
        dy=np.full(len(time), float(scatter)),
    )
    return np.asarray(
        model.power(
            np.asarray(periods) * u.day,
            np.asarray(durations) * u.day,
            method="fast",
        ).power,
        dtype=float,
    )


def _exact_solution_provider(*, time, flux, scatter, periods, durations):
    model = BoxLeastSquares(
        np.asarray(time) * u.day,
        np.asarray(flux),
        dy=np.full(len(time), float(scatter)),
    )
    solved = model.power(
        np.asarray(periods) * u.day,
        np.asarray(durations) * u.day,
        method="fast",
    )
    return {
        "power": np.asarray(solved.power, dtype=float),
        "epoch_btjd": np.asarray(
            solved.transit_time.to_value(u.day), dtype=float
        ),
        "duration_days": np.asarray(
            solved.duration.to_value(u.day), dtype=float
        ),
    }


def test_external_coarse_provider_preserves_exact_cpu_result():
    options = {
        "holdout_segment": None,
        "period_min": 0.5,
        "period_max": 5.0,
        "n_periods": 600,
        "max_candidates": 1,
        "sde_threshold": 2.0,
        "snr_threshold": 2.0,
        "min_events": 2,
        "max_seed_periods": 12,
        "preserve_all_local_peaks": True,
    }
    reference, _ = search_branch_bls(_branch(), **options)
    proposed, _ = search_branch_bls(
        _branch(), **options, coarse_power_provider=_exact_provider
    )
    assert proposed == reference


def test_external_fine_solution_provider_preserves_exact_cpu_result():
    options = {
        "holdout_segment": None,
        "period_min": 0.5,
        "period_max": 5.0,
        "n_periods": 600,
        "max_candidates": 1,
        "sde_threshold": 2.0,
        "snr_threshold": 2.0,
        "min_events": 2,
        "max_seed_periods": 12,
        "preserve_all_local_peaks": True,
    }
    reference, _ = search_branch_bls(_branch(), **options)
    proposed, _ = search_branch_bls(
        _branch(),
        **options,
        coarse_power_provider=_exact_provider,
        fine_solution_provider=_exact_solution_provider,
    )
    assert proposed == reference


def test_external_coarse_provider_must_cover_complete_grid():
    def broken(**kwargs):
        return np.zeros(len(kwargs["periods"]) - 1)

    with pytest.raises(ValueError, match="one value per period"):
        search_branch_bls(
            _branch(),
            holdout_segment=None,
            n_periods=100,
            max_candidates=1,
            coarse_power_provider=broken,
        )


def test_external_fine_solution_provider_must_cover_complete_grid():
    def broken(**kwargs):
        size = len(kwargs["periods"]) - 1
        return {
            "power": np.zeros(size),
            "epoch_btjd": np.zeros(size),
            "duration_days": np.ones(size),
        }

    with pytest.raises(ValueError, match="arrays must align"):
        search_branch_bls(
            _branch(),
            holdout_segment=None,
            n_periods=100,
            max_candidates=1,
            fine_solution_provider=broken,
        )


def test_exact_cuda_wrapper_keeps_reference_coarse_grid(monkeypatch):
    captured = {}

    class Adapter:
        power = staticmethod(_exact_provider)
        solution = staticmethod(_exact_solution_provider)

        @staticmethod
        def diagnostics():
            return {"backend": "fake-fp64-cuda"}

    def fake_search(*args, **kwargs):
        captured.update(kwargs)
        return [], []

    monkeypatch.setattr(
        cuda_hybrid_search,
        "get_cuda_astropy_search_adapter",
        lambda: Adapter(),
    )
    monkeypatch.setattr(
        cuda_hybrid_search.branch_search,
        "search_branch_bls",
        fake_search,
    )
    rows, diagnostics = cuda_hybrid_search.search_branch_cuda_astropy_exact(
        _branch(),
        holdout_segment=None,
        coarse_power_provider=lambda **_kwargs: np.asarray([1.0]),
    )
    assert rows == []
    assert "coarse_power_provider" not in captured
    assert captured["fine_power_provider"] is Adapter.power
    assert captured["fine_solution_provider"] is Adapter.solution
    assert captured["fine_peak_absolute_power_tolerance"] == 0.0
    assert diagnostics[-1]["coarse_seed_backend"] == (
        "reference-astropy-cpu-exact"
    )


def test_possible_peak_neighbourhood_covers_tied_maxima_and_neighbours():
    power = np.asarray([0.0, 2.0, 2.0 - 5e-10, 0.5, 3.0, 0.0])
    indices = cuda_hybrid_search.CudaAstropySearchAdapter._possible_peak_neighbourhood(
        power,
        absolute_error_bound=1e-9,
    )
    assert indices.tolist() == [0, 1, 2, 3, 4, 5]
