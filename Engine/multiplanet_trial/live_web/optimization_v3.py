"""Exact parallel compute acceleration for the clean search pipeline.

This module is opt-in.  It does not change the reference period grid, duration
grid, phase resolution, candidate thresholds, or candidate-retention policy.

The Astropy BLS Cython kernel evaluates each requested period independently.
The wheel installed on this Mac does not include OpenMP, even though the
upstream C source supports it.  ``exact_parallel_bls_fast`` therefore divides
the *existing* period array into contiguous, cost-balanced slices, calls the
unchanged Astropy kernel in threads (the kernel releases the GIL), and joins
the seven result arrays in their original order.

The event-search optimization caches only the immutable segmentation of an
identical NumPy time array.  The reference segmentation function still makes
the result, and all callers receive the same indices in the same order.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from astropy.timeseries.periodograms.bls import methods as bls_methods

from multiplanet_trial.v03 import event_search
from multiplanet_trial.v07_eval import multibranch_search as branch_search

from . import fixed_ephemeris_acceleration, optimization_v1
from . import pipeline as reference


_ORIGINAL_BLS_FAST = bls_methods.bls_fast
_ORIGINAL_SEGMENTS = event_search._segments
_REFERENCE_SEARCH_BLOCK = reference._search_block
_ORIGINAL_DISTINCT_EPHEMERIS_PEAK_INDICES = (
    branch_search._distinct_ephemeris_peak_indices
)
_SEGMENT_CACHE = threading.local()

THREAD_ENV = "HOUXING_BLS_THREADS"
MIN_PERIODS_ENV = "HOUXING_BLS_MIN_PERIODS"
DEFAULT_THREADS = 1
DEFAULT_MIN_PERIODS = 2_048
SEARCH_FIXED_THREADS_ENV = "HOUXING_SEARCH_FIXED_EPHEMERIS_THREADS"


def _positive_int_environment(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def _cost_balanced_slices(
    periods: np.ndarray,
    *,
    data_points: int,
    durations: np.ndarray,
    oversample: int,
    workers: int,
) -> list[slice]:
    """Return contiguous slices containing every period exactly once.

    Astropy first visits every cadence and then scans phase bins for every
    duration.  The estimate below is used only for load balancing; it cannot
    alter a period or any BLS calculation.
    """

    periods = np.asarray(periods, dtype=np.float64)
    if periods.ndim != 1:
        raise ValueError("BLS periods must be one-dimensional")
    if periods.size == 0:
        return []
    workers = max(1, min(int(workers), int(periods.size)))
    if workers == 1:
        return [slice(0, int(periods.size))]

    durations = np.asarray(durations, dtype=np.float64)
    bin_duration = float(np.min(durations)) / int(oversample)
    phase_bins = np.ceil(periods / bin_duration) + int(oversample)
    estimated = float(data_points) + (float(durations.size) + 2.0) * phase_bins
    cumulative = np.cumsum(estimated, dtype=np.float64)
    total = float(cumulative[-1])

    boundaries = [0]
    for worker in range(1, workers):
        target = total * worker / workers
        boundary = int(np.searchsorted(cumulative, target, side="left")) + 1
        minimum = boundaries[-1] + 1
        maximum = int(periods.size) - (workers - worker)
        boundaries.append(min(max(boundary, minimum), maximum))
    boundaries.append(int(periods.size))
    return [
        slice(start, stop)
        for start, stop in zip(boundaries[:-1], boundaries[1:])
        if stop > start
    ]


def exact_parallel_bls_fast(
    t,
    y,
    ivar,
    period,
    duration,
    oversample,
    use_likelihood,
):
    """Evaluate the unchanged Astropy BLS kernel on exact period slices."""

    period = np.ascontiguousarray(period, dtype=np.float64)
    workers = min(
        _positive_int_environment(THREAD_ENV, DEFAULT_THREADS),
        max(1, int(period.size)),
    )
    minimum_periods = _positive_int_environment(
        MIN_PERIODS_ENV, DEFAULT_MIN_PERIODS
    )
    if workers == 1 or period.size < minimum_periods:
        return _ORIGINAL_BLS_FAST(
            t, y, ivar, period, duration, oversample, use_likelihood
        )

    slices = _cost_balanced_slices(
        period,
        data_points=len(t),
        durations=np.asarray(duration, dtype=np.float64),
        oversample=int(oversample),
        workers=workers,
    )

    def evaluate(period_slice: slice):
        # All non-period inputs are read-only inside Astropy's C kernel.  A
        # contiguous copy of the slice keeps the kernel's input contract.
        exact_periods = np.ascontiguousarray(period[period_slice], dtype=np.float64)
        return _ORIGINAL_BLS_FAST(
            t,
            y,
            ivar,
            exact_periods,
            duration,
            oversample,
            use_likelihood,
        )

    with ThreadPoolExecutor(
        max_workers=len(slices), thread_name_prefix="exact-bls"
    ) as pool:
        partial_results = list(pool.map(evaluate, slices))

    # Astropy returns seven arrays.  Concatenating each column in slice order
    # reconstructs the original result position-for-position.
    return tuple(
        np.concatenate([result[column] for result in partial_results])
        for column in range(7)
    )


def exact_cached_segments(time: np.ndarray, gap_days: float) -> list[np.ndarray]:
    """Cache reference gap segmentation for the identical immutable array."""

    cached_time = getattr(_SEGMENT_CACHE, "time", None)
    cached_gap = getattr(_SEGMENT_CACHE, "gap_days", None)
    if cached_time is time and cached_gap == gap_days:
        return _SEGMENT_CACHE.segments
    segments = _ORIGINAL_SEGMENTS(time, gap_days)
    _SEGMENT_CACHE.time = time
    _SEGMENT_CACHE.gap_days = gap_days
    _SEGMENT_CACHE.segments = segments
    return segments


def exact_indexed_distinct_ephemeris_peak_indices(
    indices,
    periods: np.ndarray,
    epochs: np.ndarray,
    durations: np.ndarray,
    *,
    observation_baseline_days: float,
) -> list[int]:
    """Apply the unchanged greedy event-train test through period buckets.

    The drift predicate proves a strict upper bound on the relative period
    separation of any possible duplicate.  The logarithmic index uses that
    bound only to reject impossible comparisons.  Every surviving pair is
    evaluated with the original phase, duration, and full-baseline drift
    formulas, while candidates remain in their original power order.
    """

    ordered = [int(value) for value in indices]
    if len(ordered) < 2:
        return ordered
    periods = np.asarray(periods, dtype=np.float64)
    epochs = np.asarray(epochs, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    baseline = max(0.0, float(observation_baseline_days))
    ordered_array = np.asarray(ordered, dtype=np.intp)
    candidate_periods = periods[ordered_array]
    candidate_epochs = epochs[ordered_array]
    candidate_durations = durations[ordered_array]

    if (
        baseline <= 0.0
        or not np.isfinite(baseline)
        or not np.all(np.isfinite(candidate_periods))
        or not np.all(candidate_periods > 0.0)
        or not np.all(np.isfinite(candidate_epochs))
        or not np.all(np.isfinite(candidate_durations))
        or not np.all(candidate_durations > 0.0)
    ):
        return _ORIGINAL_DISTINCT_EPHEMERIS_PEAK_INDICES(
            ordered,
            periods,
            epochs,
            durations,
            observation_baseline_days=observation_baseline_days,
        )

    maximum_tolerance = 0.5 * float(np.max(candidate_durations))
    relative_radius = maximum_tolerance / baseline
    if not (0.0 < relative_radius < 2.0):
        return _ORIGINAL_DISTINCT_EPHEMERIS_PEAK_INDICES(
            ordered,
            periods,
            epochs,
            durations,
            observation_baseline_days=observation_baseline_days,
        )
    bucket_width = float(
        np.log((2.0 + relative_radius) / (2.0 - relative_radius))
    )
    if not np.isfinite(bucket_width) or bucket_width <= 0.0:
        return _ORIGINAL_DISTINCT_EPHEMERIS_PEAK_INDICES(
            ordered,
            periods,
            epochs,
            durations,
            observation_baseline_days=observation_baseline_days,
        )

    buckets: dict[int, list[int]] = {}
    selected: list[int] = []
    for index in ordered:
        period = float(periods[index])
        epoch = float(epochs[index])
        duration = float(durations[index])
        bucket = int(np.floor(np.log(period) / bucket_width))
        possible: list[int] = []
        # One adjacent bucket is mathematically sufficient.  Two also covers
        # representable-float boundary cases without widening the final test.
        for nearby in range(bucket - 2, bucket + 3):
            possible.extend(buckets.get(nearby, ()))

        duplicate = False
        if possible and len(possible) <= 16:
            for old in possible:
                old_period = float(periods[old])
                reference_period = 0.5 * (period + old_period)
                phase_distance = abs(
                    (
                        epoch
                        - float(epochs[old])
                        + 0.5 * reference_period
                    )
                    % reference_period
                    - 0.5 * reference_period
                )
                event_tolerance = 0.5 * max(
                    duration, float(durations[old])
                )
                accumulated_drift = (
                    baseline * abs(period - old_period) / reference_period
                )
                if (
                    phase_distance <= event_tolerance
                    and accumulated_drift <= event_tolerance
                ):
                    duplicate = True
                    break
        elif possible:
            old = np.asarray(possible, dtype=np.intp)
            old_periods = periods[old]
            reference_periods = 0.5 * (period + old_periods)
            event_tolerances = 0.5 * np.maximum(duration, durations[old])
            phase_distances = np.abs(
                np.remainder(
                    epoch - epochs[old] + 0.5 * reference_periods,
                    reference_periods,
                )
                - 0.5 * reference_periods
            )
            accumulated_drift = (
                baseline * np.abs(period - old_periods) / reference_periods
            )
            duplicate = bool(
                np.any(
                    (phase_distances <= event_tolerances)
                    & (accumulated_drift <= event_tolerances)
                )
            )
        if duplicate:
            continue
        selected.append(index)
        buckets.setdefault(bucket, []).append(index)
    return selected


def install_worker_optimizations() -> None:
    """Install exact kernels inside the current process only."""

    bls_methods.bls_fast = exact_parallel_bls_fast
    event_search._segments = exact_cached_segments
    branch_search._distinct_ephemeris_peak_indices = (
        exact_indexed_distinct_ephemeris_peak_indices
    )


def search_block_exact(*args, **kwargs):
    """Spawn-safe wrapper that installs optimizations inside each fold."""

    install_worker_optimizations()
    previous = os.environ.get(fixed_ephemeris_acceleration.THREAD_ENV)
    search_threads = _positive_int_environment(
        SEARCH_FIXED_THREADS_ENV,
        max(1, min(4, (os.cpu_count() or 1) // 3)),
    )
    os.environ[fixed_ephemeris_acceleration.THREAD_ENV] = str(search_threads)
    try:
        return _REFERENCE_SEARCH_BLOCK(*args, **kwargs)
    finally:
        if previous is None:
            os.environ.pop(fixed_ephemeris_acceleration.THREAD_ENV, None)
        else:
            os.environ[fixed_ephemeris_acceleration.THREAD_ENV] = previous


def install() -> None:
    """Install v1 exact reductions plus v3 worker-local acceleration."""

    optimization_v1.install()
    install_worker_optimizations()
    reference._search_block = search_block_exact
