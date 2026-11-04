"""Exact compiled backend for locked-ephemeris measurements.

The reference implementation lives in :mod:`multiplanet_trial.v03.lightkurve_evidence`.
This module changes only *where* its loops execute: Numba compiles the same
phase windows, medians, MAD scatter, event count, and S/N equations to native
code and parallelizes independent candidates.  It never changes a candidate,
cadence, threshold, duration, period, or epoch.

Keeping this backend separate makes the optimization auditable and lets the
pipeline fall back to its NumPy implementation when Numba is unavailable.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np

try:
    from numba import get_num_threads, njit, prange, set_num_threads
except ImportError:  # pragma: no cover - exercised only in minimal installs
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True


THREAD_ENV = "HOUXING_FIXED_EPHEMERIS_THREADS"
FALLBACK_THREAD_ENV = "HOUXING_BLS_THREADS"
# The locked-ephemeris matrix is the CPU half of both the native and CUDA
# refit paths.  A fixed three-thread default left most modern desktop CPUs
# idle, especially after the GPU had made the period scan comparatively
# cheap.  Keep the explicit environment override, but otherwise use the
# available physical execution budget up to a conservative 16 threads.
DEFAULT_THREADS = min(16, os.cpu_count() or 1)


def _requested_threads() -> int:
    raw = os.environ.get(THREAD_ENV, os.environ.get(FALLBACK_THREAD_ENV))
    if raw is None:
        return DEFAULT_THREADS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{THREAD_ENV} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{THREAD_ENV} must be at least 1, got {value}")
    return value


if NUMBA_AVAILABLE:

    @njit(cache=False, inline="always")
    def _lower_bound(values: np.ndarray, target: float) -> int:
        left = 0
        right = values.size
        while left < right:
            middle = (left + right) // 2
            if values[middle] < target:
                left = middle + 1
            else:
                right = middle
        return left

    @njit(cache=False, inline="always")
    def _upper_bound(values: np.ndarray, target: float) -> int:
        left = 0
        right = values.size
        while left < right:
            middle = (left + right) // 2
            if values[middle] <= target:
                left = middle + 1
            else:
                right = middle
        return left

    @njit(cache=False, parallel=True)
    def _phase_support_kernel(
        seed_period: float,
        centers: np.ndarray,
        durations_days: np.ndarray,
        block_numbers: np.ndarray,
        snrs: np.ndarray,
        maximum_duration_days: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Find every circular-phase neighbour without a strongest member."""
        event_count = centers.size
        phases = centers % seed_period
        order = np.argsort(phases)
        ordered_phases = phases[order]
        block_masks = np.zeros(event_count, dtype=np.uint8)
        support_counts = np.zeros(event_count, dtype=np.int64)
        support_snr_sums = np.zeros(event_count, dtype=np.float64)
        support_counts_by_block = np.zeros((event_count, 3), dtype=np.int64)

        for seed_index in prange(event_count):
            phase = phases[seed_index]
            duration = durations_days[seed_index]
            starts = np.empty(2, dtype=np.int64)
            stops = np.empty(2, dtype=np.int64)
            interval_count = 0

            # When the coarse circular window covers the whole phase ring,
            # scan each event once.  Otherwise create one or two disjoint
            # intervals, eliminating the duplicate hits caused by checking
            # phase-period, phase, and phase+period separately.
            if 2.0 * maximum_duration_days >= seed_period:
                starts[0] = 0
                stops[0] = event_count
                interval_count = 1
            else:
                lower = phase - maximum_duration_days
                upper = phase + maximum_duration_days
                if lower < 0.0:
                    starts[0] = 0
                    stops[0] = np.searchsorted(
                        ordered_phases, upper, side="right"
                    )
                    starts[1] = np.searchsorted(
                        ordered_phases, seed_period + lower, side="left"
                    )
                    stops[1] = event_count
                    interval_count = 2
                elif upper >= seed_period:
                    starts[0] = np.searchsorted(
                        ordered_phases, lower, side="left"
                    )
                    stops[0] = event_count
                    starts[1] = 0
                    stops[1] = np.searchsorted(
                        ordered_phases, upper - seed_period, side="right"
                    )
                    interval_count = 2
                else:
                    starts[0] = np.searchsorted(
                        ordered_phases, lower, side="left"
                    )
                    stops[0] = np.searchsorted(
                        ordered_phases, upper, side="right"
                    )
                    interval_count = 1

            mask = np.uint8(0)
            count = 0
            snr_sum = 0.0
            for interval_index in range(interval_count):
                for ordered_index in range(
                    starts[interval_index], stops[interval_index]
                ):
                    other_index = order[ordered_index]
                    error = abs(
                        (
                            phases[other_index]
                            - phase
                            + 0.5 * seed_period
                        )
                        % seed_period
                        - 0.5 * seed_period
                    )
                    tolerance = max(
                        duration, durations_days[other_index]
                    )
                    if error > tolerance:
                        continue
                    block = block_numbers[other_index]
                    mask = np.uint8(mask | np.uint8(1 << (block - 1)))
                    count += 1
                    support_counts_by_block[seed_index, block - 1] += 1
                    snr_sum += snrs[other_index]

            block_masks[seed_index] = mask
            support_counts[seed_index] = count
            support_snr_sums[seed_index] = snr_sum

        return (
            block_masks,
            support_counts,
            support_snr_sums,
            support_counts_by_block,
        )

    @njit(cache=False, parallel=True)
    def _measure_kernel(
        time_values: np.ndarray,
        flux_values: np.ndarray,
        periods: np.ndarray,
        epochs: np.ndarray,
        durations_days: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Native equivalent of ``measure_fixed_ephemeris`` for many rows."""
        candidate_count = periods.size
        cadence_count = time_values.size
        depths = np.full(candidate_count, np.nan)
        snrs = np.full(candidate_count, np.nan)
        scatters = np.full(candidate_count, np.nan)
        inside_counts = np.zeros(candidate_count, dtype=np.int64)
        nearby_counts = np.zeros(candidate_count, dtype=np.int64)
        event_counts = np.zeros(candidate_count, dtype=np.int64)
        sufficient = np.zeros(candidate_count, dtype=np.uint8)

        for candidate_index in prange(candidate_count):
            period = periods[candidate_index]
            epoch = epochs[candidate_index]
            duration = durations_days[candidate_index]

            finite_inside = 0
            finite_nearby = 0
            last_event = np.int64(-9223372036854775807)
            distinct_events = 0

            # Per worker, not candidate_count x cadence_count.  This is the
            # crucial memory difference from a dense broadcasting matrix.
            inside_flux = np.empty(cadence_count, dtype=np.float64)
            nearby_flux = np.empty(cadence_count, dtype=np.float64)

            if period > 8.0 * duration:
                # The two requested windows occupy only a small fraction of a
                # long-period light curve.  Visit those sorted-time windows
                # directly instead of evaluating modulo for every cadence.
                # At this strict separation adjacent event windows cannot
                # overlap, so counts and median inputs are identical.
                first_cycle = int(
                    np.floor((time_values[0] - epoch) / period)
                ) - 1
                last_cycle = int(
                    np.ceil((time_values[cadence_count - 1] - epoch) / period)
                ) + 1
                for cycle in range(first_cycle, last_cycle + 1):
                    center = epoch + cycle * period
                    window_padding = 1e-10 * max(
                        1.0, abs(center), abs(epoch), abs(period)
                    )
                    left = _lower_bound(
                        time_values,
                        center - 4.0 * duration - window_padding,
                    )
                    right = _upper_bound(
                        time_values,
                        center + 4.0 * duration + window_padding,
                    )
                    event_has_inside = False
                    for cadence_index in range(left, right):
                        # Re-evaluate the reference modulo expression so even
                        # floating-point boundary membership is unchanged.
                        phase = (
                            time_values[cadence_index]
                            - epoch
                            + 0.5 * period
                        ) % period - 0.5 * period
                        absolute_phase = abs(phase)
                        if absolute_phase <= 0.5 * duration:
                            inside_counts[candidate_index] += 1
                            event_has_inside = True
                            if np.isfinite(flux_values[cadence_index]):
                                inside_flux[finite_inside] = flux_values[cadence_index]
                                finite_inside += 1
                        if (
                            absolute_phase >= 1.5 * duration
                            and absolute_phase <= 4.0 * duration
                        ):
                            nearby_counts[candidate_index] += 1
                            if np.isfinite(flux_values[cadence_index]):
                                nearby_flux[finite_nearby] = flux_values[cadence_index]
                                finite_nearby += 1
                    if event_has_inside:
                        distinct_events += 1
            else:
                # Overlapping event windows require the reference nearest-
                # phase rule.  Preserve the original full scan exactly for
                # this short-period corner case.
                for cadence_index in range(cadence_count):
                    phase = (
                        time_values[cadence_index]
                        - epoch
                        + 0.5 * period
                    ) % period - 0.5 * period
                    absolute_phase = abs(phase)
                    if absolute_phase <= 0.5 * duration:
                        inside_counts[candidate_index] += 1
                        if np.isfinite(flux_values[cadence_index]):
                            inside_flux[finite_inside] = flux_values[cadence_index]
                            finite_inside += 1
                        event_id = np.int64(
                            np.rint((time_values[cadence_index] - epoch) / period)
                        )
                        # Pipeline light curves are time-sorted, so event ids
                        # are monotonic and a change count equals np.unique.
                        if event_id != last_event:
                            distinct_events += 1
                            last_event = event_id
                    if (
                        absolute_phase >= 1.5 * duration
                        and absolute_phase <= 4.0 * duration
                    ):
                        nearby_counts[candidate_index] += 1
                        if np.isfinite(flux_values[cadence_index]):
                            nearby_flux[finite_nearby] = flux_values[cadence_index]
                            finite_nearby += 1

            if (
                inside_counts[candidate_index] < 3
                or nearby_counts[candidate_index] < 10
            ):
                continue

            sufficient[candidate_index] = 1
            baseline = np.median(nearby_flux[:finite_nearby])
            transit_level = np.median(inside_flux[:finite_inside])
            depth = baseline - transit_level
            for local_index in range(finite_nearby):
                nearby_flux[local_index] = abs(
                    nearby_flux[local_index] - baseline
                )
            scatter = 1.4826 * np.median(nearby_flux[:finite_nearby])

            # Exact _mad_sigma fallback.  It is rarely needed for real light
            # curves, but preserving it keeps this backend interchangeable.
            if not np.isfinite(scatter) or scatter <= 0:
                total = 0.0
                finite_count = 0
                for cadence_index in range(cadence_count):
                    phase = (
                        time_values[cadence_index]
                        - epoch
                        + 0.5 * period
                    ) % period - 0.5 * period
                    absolute_phase = abs(phase)
                    if (
                        absolute_phase >= 1.5 * duration
                        and absolute_phase <= 4.0 * duration
                        and np.isfinite(flux_values[cadence_index])
                    ):
                        total += flux_values[cadence_index]
                        finite_count += 1
                mean = total / finite_count
                squared_error = 0.0
                for cadence_index in range(cadence_count):
                    phase = (
                        time_values[cadence_index]
                        - epoch
                        + 0.5 * period
                    ) % period - 0.5 * period
                    absolute_phase = abs(phase)
                    if (
                        absolute_phase >= 1.5 * duration
                        and absolute_phase <= 4.0 * duration
                        and np.isfinite(flux_values[cadence_index])
                    ):
                        difference = flux_values[cadence_index] - mean
                        squared_error += difference * difference
                scatter = np.sqrt(squared_error / finite_count)

            if not np.isfinite(scatter) or scatter <= 0:
                scatter = 0.0
            snr = (
                depth
                / (scatter / np.sqrt(inside_counts[candidate_index]))
                if scatter > 0
                else 0.0
            )
            depths[candidate_index] = depth
            snrs[candidate_index] = snr
            scatters[candidate_index] = scatter
            event_counts[candidate_index] = distinct_events

        return (
            depths,
            snrs,
            scatters,
            inside_counts,
            nearby_counts,
            event_counts,
            sufficient,
        )


def measure_fixed_ephemerides_compiled(
    time_values: np.ndarray,
    flux_values: np.ndarray,
    candidates: list[Mapping],
    *,
    threads: int | None = None,
) -> list[dict] | None:
    """Return exact metrics, or ``None`` when the compiled backend is absent."""
    if not NUMBA_AVAILABLE:
        return None
    if not candidates:
        return []

    requested = _requested_threads() if threads is None else max(1, int(threads))
    if get_num_threads() != requested:
        set_num_threads(requested)

    time_values = np.asarray(time_values, dtype=np.float64)
    flux_values = np.asarray(flux_values, dtype=np.float64)
    if time_values.size > 1 and np.any(np.diff(time_values) < 0):
        # The event-count shortcut relies only on the pipeline's documented
        # sorted-time invariant.  Refuse acceleration rather than change a
        # result for an unexpected caller.
        return None
    periods = np.asarray(
        [float(row["period_days"]) for row in candidates], dtype=np.float64
    )
    epochs = np.asarray(
        [float(row["epoch_btjd"]) for row in candidates], dtype=np.float64
    )
    durations_days = np.asarray(
        [float(row["duration_hours"]) / 24.0 for row in candidates],
        dtype=np.float64,
    )
    (
        depths,
        snrs,
        scatters,
        inside_counts,
        nearby_counts,
        event_counts,
        sufficient,
    ) = _measure_kernel(
        time_values,
        flux_values,
        periods,
        epochs,
        durations_days,
    )

    output: list[dict] = []
    for index in range(len(candidates)):
        if not sufficient[index]:
            output.append(
                {
                    "depth_ppm": None,
                    "snr": None,
                    "events": 0,
                    "transit_points": int(inside_counts[index]),
                    "baseline_points": int(nearby_counts[index]),
                    "status": "insufficient_coverage",
                }
            )
            continue
        depth = float(depths[index])
        snr = float(snrs[index])
        status = (
            "not_supported"
            if depth <= 0 or snr < 3.0
            else "weak_support"
            if snr < 5.0
            else "supported"
        )
        output.append(
            {
                "depth_ppm": float(depth * 1e6),
                "snr": snr,
                "events": int(event_counts[index]),
                "transit_points": int(inside_counts[index]),
                "baseline_points": int(nearby_counts[index]),
                "local_scatter_ppm": float(scatters[index] * 1e6),
                "status": status,
            }
        )
    return output


def phase_support_summaries_compiled(
    seed_period: float,
    centers: np.ndarray,
    durations_days: np.ndarray,
    block_numbers: np.ndarray,
    snrs: np.ndarray,
    *,
    maximum_duration_days: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return complete support summaries for every local-event phase seed.

    No member is selected or removed.  The arrays preserve, for every seed,
    the exact supporting block mask, total member count, S/N sum used by the
    existing metadata formula, and count in each of the three folds.
    """
    if not NUMBA_AVAILABLE:
        return None
    requested = _requested_threads()
    if get_num_threads() != requested:
        set_num_threads(requested)
    return _phase_support_kernel(
        float(seed_period),
        np.asarray(centers, dtype=np.float64),
        np.asarray(durations_days, dtype=np.float64),
        np.asarray(block_numbers, dtype=np.int64),
        np.asarray(snrs, dtype=np.float64),
        float(maximum_duration_days),
    )
