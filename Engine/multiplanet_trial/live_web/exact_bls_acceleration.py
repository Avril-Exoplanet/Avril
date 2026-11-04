"""Exact float64 BLS backend specialized for VLora ephemeris refits.

The reference implementation is Astropy's ``BoxLeastSquares.power`` with
``objective="snr"`` and one duration.  Astropy evaluates periods one at a
time, rescanning the complete light curve for every period.  The kernel below
transposes only that execution order: a small tile of periods shares one pass
over the immutable time, flux, and weight arrays.  Every requested period,
cadence, phase bin, and duration is still evaluated independently.

This module deliberately does not replace Astropy globally.  Astropy remains
the scientific oracle and callers opt in only after differential tests pass.
"""

from __future__ import annotations

import math
import os
import platform
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from . import cpp_exact_bls

try:
    from numba import get_num_threads, njit, prange, set_num_threads
except ImportError:  # pragma: no cover - only minimal installations
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True


THREAD_ENV = "HOUXING_EXACT_BLS_THREADS"
TILE_ENV = "HOUXING_EXACT_BLS_TILE_SIZE"
BACKEND_ENV = "HOUXING_EXACT_BLS_BACKEND"
# Eight periods per cadence pass was the fastest exact setting on the
# 2,251,952-cadence L98-59 regression light curve on Apple M4.  It remains an
# environment override rather than a scientific parameter.
DEFAULT_TILE_SIZE = 8
# On Apple Silicon, the cadence-major NEON kernel is materially faster for
# the two shortest TESS duration bins.  Longer boxes benefit from phase-run
# compression instead.  This is an execution choice only: both kernels visit
# every supplied cadence and period and are differential-tested bit-for-bit.
NEON_SHORT_DURATION_MAX_DAYS = 1.25 / 24.0


@dataclass(frozen=True)
class PreparedExactBLS:
    """Immutable light-curve terms reused by every period-grid request."""

    relative_time: np.ndarray
    centered_flux: np.ndarray
    inverse_variance: np.ndarray
    time_reference: float
    sum_y: float
    sum_ivar: float
    monotonic_time: bool


def _requested_threads() -> int:
    raw = os.environ.get(THREAD_ENV)
    if raw is None:
        return min(10, os.cpu_count() or 1)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{THREAD_ENV} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{THREAD_ENV} must be at least 1, got {value}")
    return value


def _requested_tile_size() -> int:
    raw = os.environ.get(TILE_ENV)
    if raw is None:
        return DEFAULT_TILE_SIZE
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{TILE_ENV} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{TILE_ENV} must be at least 1, got {value}")
    return value


def _requested_backend() -> str:
    value = os.environ.get(BACKEND_ENV, "auto").strip().lower()
    if value not in {"auto", "cpp", "numba"}:
        raise ValueError(
            f"{BACKEND_ENV} must be 'auto', 'cpp', or 'numba', got {value!r}"
        )
    return value


def _cpp_strategy(prepared: PreparedExactBLS, duration: float) -> str:
    """Select the exact native kernel without changing scientific coverage."""

    override = os.environ.get(cpp_exact_bls.CPP_STRATEGY_ENV)
    if override:
        selected = override.strip().lower()
        if selected not in {"phase-runs", "tiled-neon"}:
            raise ValueError(
                f"{cpp_exact_bls.CPP_STRATEGY_ENV} must be "
                "'phase-runs' or 'tiled-neon'"
            )
        return selected
    if not prepared.monotonic_time:
        return "tiled-neon"
    machine = platform.machine().lower()
    if (
        machine in {"arm64", "aarch64"}
        and duration <= NEON_SHORT_DURATION_MAX_DAYS
    ):
        return "tiled-neon"
    return "phase-runs"


if NUMBA_AVAILABLE:

    # Do not persist Numba's environment pickle.  This project is invoked both
    # as ``multiplanet_trial.live_web`` and by direct experiment entry points;
    # a cached module name from one entry point is not importable from another.
    @njit(cache=False)
    def _sequential_weighted_totals(
        centered_flux: np.ndarray, inverse_variance: np.ndarray
    ) -> tuple[float, float]:
        """Match Astropy's scalar accumulation order."""

        sum_y = 0.0
        sum_ivar = 0.0
        for index in range(centered_flux.size):
            weight = inverse_variance[index]
            sum_y += centered_flux[index] * weight
            sum_ivar += weight
        return sum_y, sum_ivar

    @njit(cache=False, inline="always")
    def _phase_bin_from_reciprocal(
        time_value: float,
        period: float,
        inverse_period: float,
        bin_duration: float,
    ) -> int:
        """Return the unchanged Astropy phase bin without a division."""

        cycles = math.floor(time_value * inverse_period)
        wrapped = time_value - period * cycles
        if wrapped >= period:
            wrapped -= period
        elif wrapped < 0.0:
            wrapped += period
        return int(wrapped / bin_duration) + 1

    @njit(cache=False, parallel=True)
    def _exact_snr_bls_tiled(
        relative_time: np.ndarray,
        centered_flux: np.ndarray,
        inverse_variance: np.ndarray,
        periods: np.ndarray,
        inverse_periods: np.ndarray,
        duration: float,
        oversample: int,
        tile_size: int,
        sum_y: float,
        sum_ivar: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the Astropy S/N objective for every supplied period.

        Accumulation for each individual period still follows cadence order,
        which keeps the floating-point reduction order aligned with Astropy.
        The only loop transformation is loading each cadence once per period
        tile instead of once per period.
        """

        period_count = periods.size
        cadence_count = relative_time.size
        tile_count = (period_count + tile_size - 1) // tile_size
        bin_duration = duration / float(oversample)
        powers = np.full(period_count, -np.inf, dtype=np.float64)
        best_phase_indices = np.full(period_count, -1, dtype=np.int64)

        for tile_index in prange(tile_count):
            start = tile_index * tile_size
            stop = min(period_count, start + tile_size)
            width = stop - start
            maximum_period = periods[start]
            for period_index in range(start + 1, stop):
                maximum_period = max(maximum_period, periods[period_index])
            maximum_bins = int(math.ceil(maximum_period / bin_duration)) + oversample
            mean_y = np.zeros((width, maximum_bins + 1), dtype=np.float64)
            mean_ivar = np.zeros((width, maximum_bins + 1), dtype=np.float64)

            # Loop interchange is the acceleration: immutable cadence values
            # are loaded once, then used by every independent period in the
            # tile.  The common width-eight path is explicitly unrolled so
            # LLVM need not dispatch the tiny inner loop 18 million times.
            # Accumulation for each period still follows cadence order.
            if tile_size == 8 and width == 8:
                p0, p1, p2, p3 = (
                    periods[start],
                    periods[start + 1],
                    periods[start + 2],
                    periods[start + 3],
                )
                p4, p5, p6, p7 = (
                    periods[start + 4],
                    periods[start + 5],
                    periods[start + 6],
                    periods[start + 7],
                )
                r0, r1, r2, r3 = (
                    inverse_periods[start],
                    inverse_periods[start + 1],
                    inverse_periods[start + 2],
                    inverse_periods[start + 3],
                )
                r4, r5, r6, r7 = (
                    inverse_periods[start + 4],
                    inverse_periods[start + 5],
                    inverse_periods[start + 6],
                    inverse_periods[start + 7],
                )
                for cadence_index in range(cadence_count):
                    time_value = relative_time[cadence_index]
                    # VLora's refit never supplies ``dy``; Astropy therefore
                    # uses an exact float64 weight of one for every cadence.
                    weight = 1.0
                    weighted_flux = centered_flux[cadence_index]
                    b0 = _phase_bin_from_reciprocal(time_value, p0, r0, bin_duration)
                    b1 = _phase_bin_from_reciprocal(time_value, p1, r1, bin_duration)
                    b2 = _phase_bin_from_reciprocal(time_value, p2, r2, bin_duration)
                    b3 = _phase_bin_from_reciprocal(time_value, p3, r3, bin_duration)
                    b4 = _phase_bin_from_reciprocal(time_value, p4, r4, bin_duration)
                    b5 = _phase_bin_from_reciprocal(time_value, p5, r5, bin_duration)
                    b6 = _phase_bin_from_reciprocal(time_value, p6, r6, bin_duration)
                    b7 = _phase_bin_from_reciprocal(time_value, p7, r7, bin_duration)
                    mean_y[0, b0] += weighted_flux
                    mean_y[1, b1] += weighted_flux
                    mean_y[2, b2] += weighted_flux
                    mean_y[3, b3] += weighted_flux
                    mean_y[4, b4] += weighted_flux
                    mean_y[5, b5] += weighted_flux
                    mean_y[6, b6] += weighted_flux
                    mean_y[7, b7] += weighted_flux
                    mean_ivar[0, b0] += weight
                    mean_ivar[1, b1] += weight
                    mean_ivar[2, b2] += weight
                    mean_ivar[3, b3] += weight
                    mean_ivar[4, b4] += weight
                    mean_ivar[5, b5] += weight
                    mean_ivar[6, b6] += weight
                    mean_ivar[7, b7] += weight
            else:
                for cadence_index in range(cadence_count):
                    time_value = relative_time[cadence_index]
                    weight = 1.0
                    weighted_flux = centered_flux[cadence_index]
                    for local_index in range(width):
                        phase_bin = _phase_bin_from_reciprocal(
                            time_value,
                            periods[start + local_index],
                            inverse_periods[start + local_index],
                            bin_duration,
                        )
                        mean_y[local_index, phase_bin] += weighted_flux
                        mean_ivar[local_index, phase_bin] += weight

            for local_index in range(width):
                output_index = start + local_index
                period = periods[output_index]
                n_bins = int(math.ceil(period / bin_duration)) + oversample

                wrapped_index = n_bins - oversample
                for phase_index in range(1, oversample + 1):
                    mean_y[local_index, wrapped_index] = mean_y[
                        local_index, phase_index
                    ]
                    mean_ivar[local_index, wrapped_index] = mean_ivar[
                        local_index, phase_index
                    ]
                    wrapped_index += 1

                for phase_index in range(1, n_bins + 1):
                    mean_y[local_index, phase_index] += mean_y[
                        local_index, phase_index - 1
                    ]
                    mean_ivar[local_index, phase_index] += mean_ivar[
                        local_index, phase_index - 1
                    ]

                duration_bins = int(math.floor(duration / bin_duration + 0.5))
                maximum_phase = n_bins - duration_bins
                best_power = -np.inf
                best_phase_index = -1
                for phase_index in range(maximum_phase + 1):
                    y_in = (
                        mean_y[local_index, phase_index + duration_bins]
                        - mean_y[local_index, phase_index]
                    )
                    ivar_in = (
                        mean_ivar[local_index, phase_index + duration_bins]
                        - mean_ivar[local_index, phase_index]
                    )
                    y_out = sum_y - y_in
                    ivar_out = sum_ivar - ivar_in
                    if ivar_in < np.finfo(np.float64).eps or ivar_out < np.finfo(
                        np.float64
                    ).eps:
                        continue
                    y_in /= ivar_in
                    y_out /= ivar_out
                    depth = y_out - y_in
                    depth_error = math.sqrt(1.0 / ivar_in + 1.0 / ivar_out)
                    objective = depth / depth_error
                    if y_out >= y_in and objective > best_power:
                        best_power = objective
                        fitted_duration = duration_bins * bin_duration
                        best_phase_index = phase_index
                powers[output_index] = best_power
                best_phase_indices[output_index] = best_phase_index

        return powers, best_phase_indices


def prepare_exact_bls(
    time_values: np.ndarray,
    flux_values: np.ndarray,
) -> PreparedExactBLS:
    """Prepare branch-only arrays once without changing Astropy's arithmetic."""

    if not NUMBA_AVAILABLE:
        raise RuntimeError("exact VLora refit backend requires Numba")
    time_array = np.ascontiguousarray(time_values, dtype=np.float64)
    flux_array = np.ascontiguousarray(flux_values, dtype=np.float64)
    if time_array.ndim != 1 or flux_array.ndim != 1:
        raise ValueError("time and flux must be one-dimensional")
    if time_array.size != flux_array.size or time_array.size == 0:
        raise ValueError("time and flux must be non-empty and aligned")
    if not np.all(np.isfinite(time_array)) or not np.all(np.isfinite(flux_array)):
        raise ValueError("time and flux must be finite")
    time_reference = float(np.min(time_array))
    relative_time = np.ascontiguousarray(time_array - time_reference)
    centered_flux = np.ascontiguousarray(flux_array - np.median(flux_array))
    inverse_variance = np.ones(centered_flux.size, dtype=np.float64)
    sum_y, sum_ivar = _sequential_weighted_totals(
        centered_flux, inverse_variance
    )
    return PreparedExactBLS(
        relative_time=relative_time,
        centered_flux=centered_flux,
        inverse_variance=inverse_variance,
        time_reference=time_reference,
        sum_y=float(sum_y),
        sum_ivar=float(sum_ivar),
        monotonic_time=bool(np.all(relative_time[1:] >= relative_time[:-1])),
    )


def exact_bls_solutions(
    time_values: np.ndarray,
    flux_values: np.ndarray,
    *,
    periods: np.ndarray,
    durations: np.ndarray,
    oversample: int = 10,
    threads: int | None = None,
    tile_size: int | None = None,
    prepared: PreparedExactBLS | None = None,
) -> Mapping[str, np.ndarray] | None:
    """Return exact Astropy-compatible S/N power and transit epochs.

    Only the one-duration path used by the ephemeris refit is supported.  A
    ``None`` return lets callers fall back to Astropy when Numba is absent.
    """

    if not NUMBA_AVAILABLE:
        return None
    period_array = np.ascontiguousarray(periods, dtype=np.float64)
    duration_array = np.atleast_1d(np.asarray(durations, dtype=np.float64))
    if period_array.ndim != 1 or period_array.size == 0:
        raise ValueError("periods must be a non-empty one-dimensional array")
    if duration_array.size != 1:
        raise ValueError("exact VLora refit backend requires one duration")
    duration = float(duration_array[0])
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and positive")
    if not np.all(np.isfinite(period_array)) or np.min(period_array) <= duration:
        raise ValueError("every period must be finite and longer than duration")
    oversample = int(oversample)
    if oversample < 1:
        raise ValueError("oversample must be at least one")

    requested_threads = _requested_threads() if threads is None else int(threads)
    requested_threads = max(1, min(requested_threads, os.cpu_count() or 1))
    set_num_threads(requested_threads)
    effective_tile_size = (
        _requested_tile_size() if tile_size is None else max(1, int(tile_size))
    )

    prepared_branch = (
        prepare_exact_bls(time_values, flux_values)
        if prepared is None
        else prepared
    )
    relative_time = prepared_branch.relative_time
    centered_flux = prepared_branch.centered_flux
    inverse_variance = prepared_branch.inverse_variance
    time_reference = float(prepared_branch.time_reference)
    inverse_periods = np.ascontiguousarray(1.0 / period_array)
    requested_backend = _requested_backend()
    native_result = None
    if requested_backend != "numba":
        try:
            native_result = cpp_exact_bls.exact_bls_phase_search(
                relative_time,
                centered_flux,
                period_array,
                inverse_periods,
                duration=duration,
                oversample=oversample,
                threads=requested_threads,
                tile_size=effective_tile_size,
                sum_y=float(prepared_branch.sum_y),
                sum_ivar=float(prepared_branch.sum_ivar),
                strategy=_cpp_strategy(prepared_branch, duration),
            )
        except RuntimeError:
            if requested_backend == "cpp":
                raise
    if native_result is None:
        power, best_phase_indices = _exact_snr_bls_tiled(
            relative_time,
            centered_flux,
            inverse_variance,
            period_array,
            inverse_periods,
            duration,
            oversample,
            effective_tile_size,
            float(prepared_branch.sum_y),
            float(prepared_branch.sum_ivar),
        )
        backend = "numba"
        simd = "llvm"
        actual_threads = get_num_threads()
    else:
        power = np.asarray(native_result["power"], dtype=np.float64)
        best_phase_indices = np.asarray(
            native_result["best_phase_indices"], dtype=np.int64
        )
        backend = str(native_result["backend"])
        simd = str(native_result["simd"])
        strategy = str(native_result["strategy"])
        actual_threads = int(native_result["threads"])
    if native_result is None:
        strategy = "tiled-numba"
    # Astropy's C kernel finishes the winning epoch with C ``fmod``.  Numba's
    # Python-remainder lowering can differ in the final float64 bit, so do the
    # same vectorized libm operation here after the parallel search.
    bin_duration = duration / float(oversample)
    duration_bins = int(math.floor(duration / bin_duration + 0.5))
    # Astropy's optimized C build reassociates
    # ``n*bin + 0.5*(duration_bins*bin)`` into one multiplication.  Keeping
    # that operation order reproduces its phase centre bit-for-bit.
    phase = np.fmod(
        (best_phase_indices.astype(np.float64) + 0.5 * duration_bins)
        * bin_duration,
        period_array,
    )
    phase[best_phase_indices < 0] = np.nan
    return {
        "power": power,
        "epoch_btjd": phase + time_reference,
        "periods_evaluated": np.asarray(period_array, dtype=np.float64),
        "period_trials_skipped": np.asarray([0], dtype=np.int64),
        "threads": np.asarray([actual_threads], dtype=np.int64),
        "tile_size": np.asarray([effective_tile_size], dtype=np.int64),
        "backend": backend,
        "simd": simd,
        "strategy": strategy,
    }
