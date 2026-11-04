from __future__ import annotations

import numpy as np
import pytest
from astropy.timeseries import BoxLeastSquares

from .exact_bls_acceleration import (
    NUMBA_AVAILABLE,
    exact_bls_solutions,
    prepare_exact_bls,
)


pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is unavailable")


def _synthetic_light_curve() -> tuple[np.ndarray, np.ndarray]:
    random = np.random.default_rng(1407)
    time_values = np.sort(random.uniform(1300.0, 1350.0, 6000))
    flux_values = 1.0 + 0.0015 * random.normal(size=time_values.size)
    period = 3.14159
    epoch = 1301.234
    duration = 0.11
    phase = (time_values - epoch + 0.5 * period) % period - 0.5 * period
    flux_values[np.abs(phase) <= 0.5 * duration] -= 0.006
    return time_values, flux_values


@pytest.mark.parametrize("tile_size", [1, 2, 4, 8])
def test_exact_tiled_bls_matches_astropy_without_omission(tile_size: int) -> None:
    time_values, flux_values = _synthetic_light_curve()
    periods = np.concatenate(
        (
            np.linspace(2.9, 3.4, 71),
            np.linspace(5.9, 6.5, 37),
        )
    )
    duration = 0.11
    expected = BoxLeastSquares(time_values, flux_values).power(
        periods,
        duration,
        objective="snr",
        oversample=10,
    )
    actual = exact_bls_solutions(
        time_values,
        flux_values,
        periods=periods,
        durations=np.asarray([duration]),
        oversample=10,
        threads=2,
        tile_size=tile_size,
    )

    assert actual is not None
    np.testing.assert_array_equal(actual["periods_evaluated"], periods)
    np.testing.assert_array_equal(actual["power"], expected.power)
    np.testing.assert_array_equal(actual["epoch_btjd"], expected.transit_time)
    assert int(actual["period_trials_skipped"][0]) == 0


def test_exact_tiled_bls_rejects_multiple_durations() -> None:
    time_values, flux_values = _synthetic_light_curve()
    with pytest.raises(ValueError, match="requires one duration"):
        exact_bls_solutions(
            time_values,
            flux_values,
            periods=np.linspace(2.9, 3.4, 11),
            durations=np.asarray([0.08, 0.11]),
        )


def test_prepared_branch_reuse_is_bitwise_identical() -> None:
    time_values, flux_values = _synthetic_light_curve()
    periods = np.linspace(2.9, 3.4, 71)
    duration = np.asarray([0.11])
    expected = exact_bls_solutions(
        time_values,
        flux_values,
        periods=periods,
        durations=duration,
        threads=2,
        tile_size=8,
    )
    prepared = prepare_exact_bls(time_values, flux_values)
    actual = exact_bls_solutions(
        time_values,
        flux_values,
        periods=periods,
        durations=duration,
        threads=2,
        tile_size=8,
        prepared=prepared,
    )
    assert expected is not None and actual is not None
    np.testing.assert_array_equal(actual["power"], expected["power"])
    np.testing.assert_array_equal(actual["epoch_btjd"], expected["epoch_btjd"])


def test_cpp_phase_runs_selected_for_sorted_time(monkeypatch) -> None:
    time_values, flux_values = _synthetic_light_curve()
    periods = np.linspace(0.7, 18.0, 191)
    duration = np.asarray([0.08])
    expected = BoxLeastSquares(time_values, flux_values).power(
        periods, duration[0], objective="snr", oversample=10
    )
    monkeypatch.setenv("HOUXING_EXACT_BLS_BACKEND", "cpp")
    actual = exact_bls_solutions(
        time_values,
        flux_values,
        periods=periods,
        durations=duration,
        threads=4,
    )
    assert actual is not None
    assert actual["backend"] == "cpp"
    assert actual["strategy"] == "phase-runs"
    np.testing.assert_array_equal(actual["power"], expected.power)
    np.testing.assert_array_equal(actual["epoch_btjd"], expected.transit_time)


def test_cpp_neon_fallback_matches_unsorted_astropy(monkeypatch) -> None:
    time_values, flux_values = _synthetic_light_curve()
    permutation = np.random.default_rng(91).permutation(time_values.size)
    time_values = time_values[permutation]
    flux_values = flux_values[permutation]
    periods = np.linspace(2.7, 6.5, 129)
    duration = np.asarray([0.11])
    expected = BoxLeastSquares(time_values, flux_values).power(
        periods, duration[0], objective="snr", oversample=10
    )
    monkeypatch.setenv("HOUXING_EXACT_BLS_BACKEND", "cpp")
    actual = exact_bls_solutions(
        time_values,
        flux_values,
        periods=periods,
        durations=duration,
        threads=4,
        tile_size=8,
    )
    assert actual is not None
    assert actual["backend"] == "cpp"
    assert actual["strategy"] == "tiled-neon"
    np.testing.assert_array_equal(actual["power"], expected.power)
    np.testing.assert_array_equal(actual["epoch_btjd"], expected.transit_time)
