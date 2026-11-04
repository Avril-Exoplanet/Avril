"""Exact compiled observed-MES gate before expensive phase-null trials.

The final significance rule first requires observed duration-noise MES-like
>= 7.1.  A candidate below that threshold cannot pass regardless of its null
distribution, so running 4096 shifted phases for it is mathematically
redundant.  This backend batch-computes the unchanged observed statistic and
per-block values.  Candidates that pass still receive the full reference null.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np

try:
    from numba import get_num_threads, njit, prange, set_num_threads
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True


THREAD_ENV = "HOUXING_RED_NOISE_GATE_THREADS"
FALLBACK_THREAD_ENV = "HOUXING_SIGNIFICANCE_THREADS"
DEFAULT_THREADS = min(16, os.cpu_count() or 1)
MAX_CONTROL_WINDOWS = 800
MIN_CONTROL_WINDOWS = 20


def _requested_threads() -> int:
    raw = os.environ.get(THREAD_ENV, os.environ.get(FALLBACK_THREAD_ENV))
    value = DEFAULT_THREADS if raw is None else int(raw)
    if value < 1:
        raise ValueError(f"{THREAD_ENV} must be at least 1")
    return value


if NUMBA_AVAILABLE:

    @njit(cache=False, inline="always")
    def _lower_bound(values, start, stop, target):
        left = start
        right = stop
        while left < right:
            middle = (left + right) // 2
            if values[middle] < target:
                left = middle + 1
            else:
                right = middle
        return left

    @njit(cache=False, inline="always")
    def _upper_bound(values, start, stop, target):
        left = start
        right = stop
        while left < right:
            middle = (left + right) // 2
            if values[middle] <= target:
                left = middle + 1
            else:
                right = middle
        return left

    @njit(cache=False, inline="always")
    def _mad_sigma_inplace(values, count):
        if count <= 0:
            return 0.0
        center = np.median(values[:count])
        total = 0.0
        for index in range(count):
            total += values[index]
            values[index] = abs(values[index] - center)
        sigma = 1.4826 * np.median(values[:count])
        if not np.isfinite(sigma) or sigma <= 0.0:
            mean = total / count
            squared = 0.0
            # Values currently contain absolute deviations, so the caller
            # refills before requesting the rare standard-deviation fallback.
            return -mean - 1.0  # sentinel encoding; handled at call sites
        return sigma

    @njit(cache=False, inline="always")
    def _median_mad(values, count, refill_values, refill_count):
        """MAD with exact population-std fallback using preserved values."""
        if count <= 0:
            return 0.0
        center = np.median(values[:count])
        for index in range(count):
            values[index] = abs(values[index] - center)
        sigma = 1.4826 * np.median(values[:count])
        if np.isfinite(sigma) and sigma > 0.0:
            return sigma
        if refill_count <= 0:
            return 0.0
        total = 0.0
        for index in range(refill_count):
            total += refill_values[index]
        mean = total / refill_count
        squared = 0.0
        for index in range(refill_count):
            difference = refill_values[index] - mean
            squared += difference * difference
        sigma = np.sqrt(squared / refill_count)
        return sigma if np.isfinite(sigma) and sigma > 0.0 else 0.0

    @njit(cache=False, inline="always")
    def _fill_window(
        time_values,
        flux_values,
        usable,
        view_start,
        view_stop,
        center,
        duration,
        inside_flux,
        nearby_flux,
    ):
        left = _lower_bound(
            time_values, view_start, view_stop, center - 4.0 * duration
        )
        right = _upper_bound(
            time_values, view_start, view_stop, center + 4.0 * duration
        )
        if right - left < 13:
            return 0, 0
        inside_count = 0
        nearby_count = 0
        for cadence_index in range(left, right):
            local_index = cadence_index - view_start
            if not usable[local_index]:
                continue
            distance = abs(time_values[cadence_index] - center)
            if distance <= 0.5 * duration:
                inside_flux[inside_count] = flux_values[cadence_index]
                inside_count += 1
            if distance >= 1.5 * duration and distance <= 4.0 * duration:
                nearby_flux[nearby_count] = flux_values[cadence_index]
                nearby_count += 1
        return inside_count, nearby_count

    @njit(cache=False, parallel=True)
    def _observed_red_noise_kernel(
        time_values,
        flux_values,
        view_offsets,
        cadences,
        block_indices,
        periods,
        epochs,
        durations,
        maximum_view_points,
        block_count,
    ):
        candidate_count = periods.size
        mes = np.full(candidate_count, np.nan)
        event_counts = np.zeros(candidate_count, dtype=np.int64)
        positive_counts = np.zeros(candidate_count, dtype=np.int64)
        ses3_counts = np.zeros(candidate_count, dtype=np.int64)
        block_mes = np.full((candidate_count, block_count), np.nan)
        segment_sigmas = np.zeros(
            (candidate_count, view_offsets.size - 1), dtype=np.float64
        )
        segment_centers = np.zeros_like(segment_sigmas)
        segment_control_counts = np.zeros_like(segment_sigmas, dtype=np.int64)

        for candidate_index in prange(candidate_count):
            period = periods[candidate_index]
            epoch = epochs[candidate_index]
            duration = durations[candidate_index]
            usable = np.empty(maximum_view_points, dtype=np.uint8)
            inside_flux = np.empty(maximum_view_points, dtype=np.float64)
            nearby_flux = np.empty(maximum_view_points, dtype=np.float64)
            nearby_original = np.empty(maximum_view_points, dtype=np.float64)
            control_depths = np.empty(MAX_CONTROL_WINDOWS, dtype=np.float64)
            white_values = np.empty(MAX_CONTROL_WINDOWS, dtype=np.float64)
            global_weight = 0.0
            global_weighted_depth = 0.0
            block_weight = np.zeros(block_count, dtype=np.float64)
            block_weighted_depth = np.zeros(block_count, dtype=np.float64)

            for view_index in range(view_offsets.size - 1):
                view_start = int(view_offsets[view_index])
                view_stop = int(view_offsets[view_index + 1])
                view_size = view_stop - view_start
                if view_size < 13:
                    continue
                for local_index in range(view_size):
                    usable[local_index] = np.uint8(1)
                first_excluded = int(
                    np.floor((time_values[view_start] - epoch) / period)
                ) - 1
                last_excluded = int(
                    np.ceil((time_values[view_stop - 1] - epoch) / period)
                ) + 1
                exclusion_half_width = 1.25 * 0.5 * duration
                for cycle in range(first_excluded, last_excluded + 1):
                    center = epoch + cycle * period
                    left = _lower_bound(
                        time_values,
                        view_start,
                        view_stop,
                        center - exclusion_half_width,
                    )
                    right = _upper_bound(
                        time_values,
                        view_start,
                        view_stop,
                        center + exclusion_half_width,
                    )
                    for cadence_index in range(left, right):
                        usable[cadence_index - view_start] = np.uint8(0)

                cadence = cadences[view_index]
                start_center = time_values[view_start] + 4.0 * duration
                stop_center = time_values[view_stop - 1] - 4.0 * duration
                step = max(duration, 10.0 * cadence)
                raw_centers = (
                    int(np.ceil((stop_center - start_center) / step))
                    if stop_center > start_center
                    else 0
                )
                selected_centers = min(raw_centers, MAX_CONTROL_WINDOWS)
                control_count = 0
                white_count = 0
                expected_inside = max(3, int(np.rint(duration / max(cadence, 1e-9))))
                minimum_inside = max(3, int(np.ceil(0.45 * expected_inside)))
                minimum_nearby = max(10, int(np.ceil(1.5 * expected_inside)))

                for center_index in range(selected_centers):
                    if raw_centers > MAX_CONTROL_WINDOWS:
                        original_index = int(
                            center_index * (raw_centers - 1) / (MAX_CONTROL_WINDOWS - 1)
                        )
                    else:
                        original_index = center_index
                    center = start_center + original_index * step
                    inside_count, nearby_count = _fill_window(
                        time_values,
                        flux_values,
                        usable,
                        view_start,
                        view_stop,
                        center,
                        duration,
                        inside_flux,
                        nearby_flux,
                    )
                    if inside_count < minimum_inside or nearby_count < minimum_nearby:
                        continue
                    baseline = np.median(nearby_flux[:nearby_count])
                    in_level = np.median(inside_flux[:inside_count])
                    control_depths[control_count] = baseline - in_level
                    control_count += 1
                    for local_index in range(nearby_count):
                        nearby_original[local_index] = nearby_flux[local_index]
                    point_sigma = _median_mad(
                        nearby_flux,
                        nearby_count,
                        nearby_original,
                        nearby_count,
                    )
                    if point_sigma > 0.0:
                        white_values[white_count] = point_sigma / np.sqrt(inside_count)
                        white_count += 1

                control_center = (
                    np.median(control_depths[:control_count])
                    if control_count > 0
                    else 0.0
                )
                for local_index in range(control_count):
                    nearby_original[local_index] = control_depths[local_index]
                duration_sigma = _median_mad(
                    control_depths,
                    control_count,
                    nearby_original,
                    control_count,
                )
                white_sigma = (
                    np.median(white_values[:white_count])
                    if white_count > 0
                    else 0.0
                )
                if control_count < MIN_CONTROL_WINDOWS or duration_sigma <= 0.0:
                    usable_flux_count = 0
                    for local_index in range(view_size):
                        if usable[local_index]:
                            nearby_flux[usable_flux_count] = flux_values[view_start + local_index]
                            nearby_original[usable_flux_count] = flux_values[view_start + local_index]
                            usable_flux_count += 1
                    point_sigma = _median_mad(
                        nearby_flux,
                        usable_flux_count,
                        nearby_original,
                        usable_flux_count,
                    )
                    duration_sigma = (
                        point_sigma / np.sqrt(expected_inside)
                        if point_sigma > 0.0
                        else 0.0
                    )
                    if white_sigma <= 0.0:
                        white_sigma = duration_sigma

                segment_sigmas[candidate_index, view_index] = duration_sigma
                segment_centers[candidate_index, view_index] = control_center
                segment_control_counts[candidate_index, view_index] = control_count
                if duration_sigma <= 0.0:
                    continue

                # The observed event train intentionally uses every cadence;
                # the exclusion mask is only for the empirical noise controls.
                for local_index in range(view_size):
                    usable[local_index] = np.uint8(1)
                first_cycle = int(
                    np.floor((time_values[view_start] - epoch) / period)
                ) - 1
                last_cycle = int(
                    np.ceil((time_values[view_stop - 1] - epoch) / period)
                ) + 1
                for cycle in range(first_cycle, last_cycle + 1):
                    center = epoch + cycle * period
                    inside_count, nearby_count = _fill_window(
                        time_values,
                        flux_values,
                        usable,
                        view_start,
                        view_stop,
                        center,
                        duration,
                        inside_flux,
                        nearby_flux,
                    )
                    if inside_count < minimum_inside or nearby_count < minimum_nearby:
                        continue
                    depth = (
                        np.median(nearby_flux[:nearby_count])
                        - np.median(inside_flux[:inside_count])
                        - control_center
                    )
                    weight = 1.0 / (duration_sigma * duration_sigma)
                    global_weight += weight
                    global_weighted_depth += depth * weight
                    block = int(block_indices[view_index])
                    if block >= 0:
                        block_weight[block] += weight
                        block_weighted_depth[block] += depth * weight
                    event_counts[candidate_index] += 1
                    if depth > 0.0:
                        positive_counts[candidate_index] += 1
                    if depth / duration_sigma >= 3.0:
                        ses3_counts[candidate_index] += 1

            if global_weight > 0.0:
                mes[candidate_index] = global_weighted_depth / np.sqrt(global_weight)
            for block in range(block_count):
                if block_weight[block] > 0.0:
                    block_mes[candidate_index, block] = (
                        block_weighted_depth[block] / np.sqrt(block_weight[block])
                    )

        return (
            mes,
            event_counts,
            positive_counts,
            ses3_counts,
            block_mes,
            segment_sigmas,
            segment_centers,
            segment_control_counts,
        )


def observed_red_noise_gate_batch(
    candidates: Sequence[Mapping],
    views: Mapping[int, Mapping],
    blocks: Sequence[Sequence[int]],
    *,
    threads: int | None = None,
) -> dict | None:
    """Return exact observed duration-noise MES for all locked candidates."""

    if not NUMBA_AVAILABLE:
        return None
    if not candidates:
        return {
            "mes_like": np.empty(0),
            "event_counts": np.empty(0, dtype=np.int64),
            "block_mes_like": np.empty((0, len(blocks))),
        }
    requested = _requested_threads() if threads is None else max(1, int(threads))
    if get_num_threads() != requested:
        set_num_threads(requested)

    segment_to_block = {
        int(segment): block_index
        for block_index, segments in enumerate(blocks)
        for segment in segments
    }
    times = []
    fluxes = []
    offsets = [0]
    cadences = []
    block_indices = []
    maximum_view_points = 0
    for segment, view in views.items():
        time = np.ascontiguousarray(view["time"], dtype=np.float64)
        flux = np.ascontiguousarray(view["flux"], dtype=np.float64)
        times.append(time)
        fluxes.append(flux)
        offsets.append(offsets[-1] + int(time.size))
        cadences.append(float(view["cadence_days"]))
        block_indices.append(int(segment_to_block.get(int(segment), -1)))
        maximum_view_points = max(maximum_view_points, int(time.size))
    periods = np.asarray([float(row["period_days"]) for row in candidates])
    epochs = np.asarray([float(row["epoch_btjd"]) for row in candidates])
    durations = np.asarray(
        [float(row["duration_hours"]) / 24.0 for row in candidates]
    )
    result = _observed_red_noise_kernel(
        np.concatenate(times),
        np.concatenate(fluxes),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(cadences, dtype=np.float64),
        np.asarray(block_indices, dtype=np.int64),
        periods,
        epochs,
        durations,
        max(1, maximum_view_points),
        len(blocks),
    )
    keys = (
        "mes_like",
        "event_counts",
        "positive_event_counts",
        "ses_ge_3_event_counts",
        "block_mes_like",
        "segment_duration_sigma",
        "segment_control_center",
        "segment_control_windows",
    )
    return dict(zip(keys, result, strict=True))


__all__ = ["NUMBA_AVAILABLE", "observed_red_noise_gate_batch"]
