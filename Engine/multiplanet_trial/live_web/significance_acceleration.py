"""Exact native event-window measurements for T16 phase-null trials."""

from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np

try:
    from numba import get_num_threads, njit, prange, set_num_threads
except ImportError:  # pragma: no cover - minimal fallback installations
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True


THREAD_ENV = "HOUXING_SIGNIFICANCE_THREADS"
FALLBACK_THREAD_ENV = "HOUXING_BLS_THREADS"
DEFAULT_THREADS = 3


def _requested_threads() -> int:
    raw = os.environ.get(THREAD_ENV, os.environ.get(FALLBACK_THREAD_ENV))
    if raw is None:
        return DEFAULT_THREADS
    value = int(raw)
    if value < 1:
        raise ValueError(f"{THREAD_ENV} must be at least 1, got {value}")
    return value


if NUMBA_AVAILABLE:

    @njit(cache=True, nogil=True)
    def _control_window_kernel(
        time_values: np.ndarray,
        flux_values: np.ndarray,
        usable_values: np.ndarray,
        centers: np.ndarray,
        duration_days: float,
        cadence_days: float,
        window_capacity: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the unchanged control-window statistic without Python loops."""
        count = centers.size
        depths = np.full(count, np.nan, dtype=np.float64)
        white_uncertainties = np.full(count, np.nan, dtype=np.float64)
        inside_counts = np.zeros(count, dtype=np.int64)
        nearby_counts = np.zeros(count, dtype=np.int64)
        expected_inside = max(
            3, int(np.rint(duration_days / max(cadence_days, 1e-9)))
        )
        minimum_inside = max(3, int(np.ceil(0.45 * expected_inside)))
        minimum_nearby = max(10, int(np.ceil(1.5 * expected_inside)))
        inside_flux = np.empty(window_capacity, dtype=np.float64)
        nearby_flux = np.empty(window_capacity, dtype=np.float64)
        for center_index in range(count):
            center = centers[center_index]
            left = np.searchsorted(
                time_values, center - 4.0 * duration_days, side="left"
            )
            right = np.searchsorted(
                time_values, center + 4.0 * duration_days, side="right"
            )
            if right - left < 13:
                continue
            inside_count = 0
            nearby_count = 0
            for cadence_index in range(left, right):
                if not usable_values[cadence_index]:
                    continue
                distance = abs(time_values[cadence_index] - center)
                if distance <= 0.5 * duration_days:
                    inside_flux[inside_count] = flux_values[cadence_index]
                    inside_count += 1
                if (
                    distance >= 1.5 * duration_days
                    and distance <= 4.0 * duration_days
                ):
                    nearby_flux[nearby_count] = flux_values[cadence_index]
                    nearby_count += 1
            inside_counts[center_index] = inside_count
            nearby_counts[center_index] = nearby_count
            if inside_count < minimum_inside or nearby_count < minimum_nearby:
                continue
            baseline = np.median(nearby_flux[:nearby_count])
            in_level = np.median(inside_flux[:inside_count])
            nearby_median = baseline
            absolute_deviation = np.empty(nearby_count, dtype=np.float64)
            for index in range(nearby_count):
                absolute_deviation[index] = abs(
                    nearby_flux[index] - nearby_median
                )
            point_sigma = 1.4826 * np.median(absolute_deviation)
            if not np.isfinite(point_sigma) or point_sigma <= 0:
                mean = 0.0
                for index in range(nearby_count):
                    mean += nearby_flux[index]
                mean /= nearby_count
                variance = 0.0
                for index in range(nearby_count):
                    difference = nearby_flux[index] - mean
                    variance += difference * difference
                point_sigma = np.sqrt(variance / nearby_count)
            depths[center_index] = baseline - in_level
            white_uncertainties[center_index] = (
                point_sigma / np.sqrt(inside_count)
                if np.isfinite(point_sigma) and point_sigma > 0
                else 0.0
            )
        return depths, white_uncertainties, inside_counts, nearby_counts

    @njit(cache=True)
    def _searchsorted_flat(
        values: np.ndarray,
        start: int,
        stop: int,
        target: float,
        right_side: bool,
    ) -> int:
        low = start
        high = stop
        while low < high:
            middle = (low + high) // 2
            if values[middle] < target or (
                right_side and values[middle] == target
            ):
                low = middle + 1
            else:
                high = middle
        return low


    @njit(cache=True, parallel=True)
    def _phase_trial_event_kernel(
        time_values: np.ndarray,
        flux_values: np.ndarray,
        usable_values: np.ndarray,
        segment_starts: np.ndarray,
        segment_stops: np.ndarray,
        cadences: np.ndarray,
        duration_sigmas: np.ndarray,
        control_centers: np.ndarray,
        period: float,
        trial_epochs: np.ndarray,
        duration_days: float,
        max_events: int,
        window_capacity: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        trial_count = trial_epochs.size
        depth_ppm = np.full((trial_count, max_events), np.nan)
        sigma_ppm = np.full((trial_count, max_events), np.nan)
        event_counts = np.zeros(trial_count, dtype=np.int64)
        overflow = np.zeros(trial_count, dtype=np.uint8)

        for trial_index in prange(trial_count):
            epoch = trial_epochs[trial_index]
            slot = 0
            # Only one +/-4-duration event window is materialized at a time.
            # The former whole-light-curve allocation was exact but caused
            # enormous allocator and memory-bandwidth churn for every trial.
            inside_flux = np.empty(window_capacity, dtype=np.float64)
            nearby_flux = np.empty(window_capacity, dtype=np.float64)
            for segment_index in range(segment_starts.size):
                sigma = duration_sigmas[segment_index]
                if sigma <= 0:
                    continue
                start = segment_starts[segment_index]
                stop = segment_stops[segment_index]
                first_cycle = (
                    int(np.floor((time_values[start] - epoch) / period)) - 1
                )
                last_cycle = (
                    int(np.ceil((time_values[stop - 1] - epoch) / period)) + 1
                )
                cadence = cadences[segment_index]
                expected_inside = max(
                    3,
                    int(
                        np.rint(
                            duration_days / max(cadence, 1e-9)
                        )
                    ),
                )
                minimum_inside = max(
                    3, int(np.ceil(0.45 * expected_inside))
                )
                minimum_nearby = max(
                    10, int(np.ceil(1.5 * expected_inside))
                )
                for cycle in range(first_cycle, last_cycle + 1):
                    center = epoch + cycle * period
                    left = _searchsorted_flat(
                        time_values,
                        start,
                        stop,
                        center - 4.0 * duration_days,
                        False,
                    )
                    right = _searchsorted_flat(
                        time_values,
                        start,
                        stop,
                        center + 4.0 * duration_days,
                        True,
                    )
                    if right - left < 13:
                        continue
                    inside_count = 0
                    nearby_count = 0
                    for cadence_index in range(left, right):
                        if not usable_values[cadence_index]:
                            continue
                        distance = abs(time_values[cadence_index] - center)
                        if distance <= 0.5 * duration_days:
                            if inside_count < window_capacity:
                                inside_flux[inside_count] = flux_values[cadence_index]
                            else:
                                overflow[trial_index] = 1
                            inside_count += 1
                        if (
                            distance >= 1.5 * duration_days
                            and distance <= 4.0 * duration_days
                        ):
                            if nearby_count < window_capacity:
                                nearby_flux[nearby_count] = flux_values[cadence_index]
                            else:
                                overflow[trial_index] = 1
                            nearby_count += 1
                    if inside_count > window_capacity or nearby_count > window_capacity:
                        continue
                    if (
                        inside_count < minimum_inside
                        or nearby_count < minimum_nearby
                    ):
                        continue
                    if slot >= max_events:
                        overflow[trial_index] = 1
                        continue
                    baseline = np.median(nearby_flux[:nearby_count])
                    in_level = np.median(inside_flux[:inside_count])
                    depth = (
                        baseline
                        - in_level
                        - control_centers[segment_index]
                    )
                    depth_ppm[trial_index, slot] = depth * 1e6
                    sigma_ppm[trial_index, slot] = sigma * 1e6
                    slot += 1
            event_counts[trial_index] = slot

        return depth_ppm, sigma_ppm, event_counts, overflow


def phase_trial_events_compiled(
    views: Mapping[int, Mapping],
    noise_models: Mapping[int, Mapping],
    usable: Mapping[int, np.ndarray],
    *,
    period: float,
    trial_epochs: np.ndarray,
    duration_days: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Measure every event of every null trial, preserving reference order."""
    cuda_mode = os.environ.get("VLORA_CUDA_PHASE_NULL", "auto").strip().lower()
    if cuda_mode not in {"off", "auto", "on"}:
        raise ValueError("VLORA_CUDA_PHASE_NULL must be 'off', 'auto', or 'on'")
    if cuda_mode != "off":
        try:
            from .cuda_phase_null import get_cuda_phase_null_measurer

            return get_cuda_phase_null_measurer().measure(
                views,
                noise_models,
                usable,
                period=period,
                trial_epochs=trial_epochs,
                duration_days=duration_days,
            )
        except Exception:
            if cuda_mode == "on":
                raise
    if not NUMBA_AVAILABLE:
        return None
    requested = _requested_threads()
    if get_num_threads() != requested:
        set_num_threads(requested)

    segments = list(views)
    times: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    usable_rows: list[np.ndarray] = []
    starts: list[int] = []
    stops: list[int] = []
    cadences: list[float] = []
    sigmas: list[float] = []
    centers: list[float] = []
    cursor = 0
    max_events = 0
    window_capacity = 1
    for segment in segments:
        view = views[segment]
        model = noise_models[int(segment)]
        time = np.asarray(view["time"], dtype=np.float64)
        flux = np.asarray(view["flux"], dtype=np.float64)
        local_usable = np.asarray(usable[int(segment)], dtype=bool)
        starts.append(cursor)
        cursor += time.size
        stops.append(cursor)
        times.append(time)
        fluxes.append(flux)
        usable_rows.append(local_usable)
        cadences.append(float(view["cadence_days"]))
        sigmas.append(float(model["duration_sigma"]))
        centers.append(float(model["control_center"]))
        if time.size:
            max_events += int(np.ceil((time[-1] - time[0]) / period)) + 5
            # Any arbitrary interval of width 8*duration has no more points
            # than the densest interval beginning at an observed cadence.
            # Two guard slots cover inclusive floating-point boundaries.
            rights = np.searchsorted(
                time,
                time + 8.0 * float(duration_days),
                side="right",
            )
            counts = rights - np.arange(time.size, dtype=np.int64)
            window_capacity = max(
                window_capacity,
                int(np.max(counts, initial=0)) + 2,
            )
    if not times:
        return (
            np.empty((len(trial_epochs), 0)),
            np.empty((len(trial_epochs), 0)),
            np.zeros(len(trial_epochs), dtype=np.int64),
        )

    result = _phase_trial_event_kernel(
        np.concatenate(times),
        np.concatenate(fluxes),
        np.concatenate(usable_rows),
        np.asarray(starts, dtype=np.int64),
        np.asarray(stops, dtype=np.int64),
        np.asarray(cadences, dtype=np.float64),
        np.asarray(sigmas, dtype=np.float64),
        np.asarray(centers, dtype=np.float64),
        float(period),
        np.asarray(trial_epochs, dtype=np.float64),
        float(duration_days),
        int(max_events),
        int(window_capacity),
    )
    depth_ppm, sigma_ppm, event_counts, overflow = result
    if np.any(overflow):
        return None
    return depth_ppm, sigma_ppm, event_counts


def control_window_measurements_compiled(
    time_values: np.ndarray,
    flux_values: np.ndarray,
    usable_values: np.ndarray,
    centers: np.ndarray,
    *,
    duration_days: float,
    cadence_days: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the exact valid control depths and white-noise uncertainties.

    The reference implementation creates several Python dictionaries per
    window.  This keeps the same searchsorted boundaries, masks, medians,
    MAD fallback and validity thresholds in one native loop.
    """
    if not NUMBA_AVAILABLE:
        return None
    time = np.ascontiguousarray(time_values, dtype=np.float64)
    flux = np.ascontiguousarray(flux_values, dtype=np.float64)
    usable = np.ascontiguousarray(usable_values, dtype=np.bool_)
    center_values = np.ascontiguousarray(centers, dtype=np.float64)
    if center_values.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    rights = np.searchsorted(
        time,
        time + 8.0 * float(duration_days),
        side="right",
    )
    window_capacity = max(
        1,
        int(np.max(rights - np.arange(time.size, dtype=np.int64), initial=0))
        + 2,
    )
    depths, white, _inside, _nearby = _control_window_kernel(
        time,
        flux,
        usable,
        center_values,
        float(duration_days),
        float(cadence_days),
        int(window_capacity),
    )
    valid = np.isfinite(depths)
    return depths[valid], white[valid]


__all__ = [
    "NUMBA_AVAILABLE",
    "control_window_measurements_compiled",
    "phase_trial_events_compiled",
]
