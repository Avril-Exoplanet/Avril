"""Exact batch backend for per-event repeatability measurements.

This module is deliberately execution-only.  It preserves the event windows,
minimum point counts, median/MAD equations, 3-sigma support rule, and serialized
diagnostics from :func:`sealed_validation.event_repeatability_metrics`.  The
only change is that independent candidates run in one Numba-parallel batch
instead of in a Python thread pool.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np

try:
    from numba import get_num_threads, njit, prange, set_num_threads
except ImportError:  # pragma: no cover - optional acceleration dependency
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True


THREAD_ENV = "HOUXING_EVENT_REPEATABILITY_THREADS"
FALLBACK_THREAD_ENV = "HOUXING_FIXED_EPHEMERIS_THREADS"
DEFAULT_THREADS = min(16, os.cpu_count() or 1)


def _requested_threads() -> int:
    raw = os.environ.get(THREAD_ENV, os.environ.get(FALLBACK_THREAD_ENV))
    if raw is None:
        return DEFAULT_THREADS
    value = int(raw)
    if value < 1:
        raise ValueError(f"{THREAD_ENV} must be at least 1, got {value}")
    return value


if NUMBA_AVAILABLE:

    @njit(cache=False, inline="always")
    def _lower_bound(values: np.ndarray, start: int, stop: int, target: float) -> int:
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
    def _upper_bound(values: np.ndarray, start: int, stop: int, target: float) -> int:
        left = start
        right = stop
        while left < right:
            middle = (left + right) // 2
            if values[middle] <= target:
                left = middle + 1
            else:
                right = middle
        return left

    @njit(cache=False, parallel=True)
    def _event_repeatability_kernel(
        time_values: np.ndarray,
        flux_values: np.ndarray,
        view_offsets: np.ndarray,
        view_segments: np.ndarray,
        periods: np.ndarray,
        epochs: np.ndarray,
        durations_days: np.ndarray,
        maximum_events: int,
        maximum_view_points: int,
        minimum_event_snr: float,
    ):
        candidate_count = periods.size
        row_counts = np.zeros(candidate_count, dtype=np.int64)
        segments = np.zeros((candidate_count, maximum_events), dtype=np.int32)
        cycles = np.zeros((candidate_count, maximum_events), dtype=np.int64)
        centers = np.full((candidate_count, maximum_events), np.nan)
        depths_ppm = np.full((candidate_count, maximum_events), np.nan)
        snrs = np.zeros((candidate_count, maximum_events), dtype=np.float64)
        supported = np.zeros((candidate_count, maximum_events), dtype=np.uint8)

        for candidate_index in prange(candidate_count):
            period = periods[candidate_index]
            epoch = epochs[candidate_index]
            duration = durations_days[candidate_index]
            inside_flux = np.empty(maximum_view_points, dtype=np.float64)
            nearby_flux = np.empty(maximum_view_points, dtype=np.float64)
            output_index = 0

            for view_index in range(view_segments.size):
                view_start = int(view_offsets[view_index])
                view_stop = int(view_offsets[view_index + 1])
                if view_stop - view_start < 13:
                    continue
                first_cycle = int(
                    np.floor((time_values[view_start] - epoch) / period)
                ) - 1
                last_cycle = int(
                    np.ceil((time_values[view_stop - 1] - epoch) / period)
                ) + 1

                for cycle in range(first_cycle, last_cycle + 1):
                    center = epoch + cycle * period
                    left = _lower_bound(
                        time_values,
                        view_start,
                        view_stop,
                        center - 4.0 * duration,
                    )
                    right = _upper_bound(
                        time_values,
                        view_start,
                        view_stop,
                        center + 4.0 * duration,
                    )
                    if right - left < 13:
                        continue

                    inside_count = 0
                    nearby_count = 0
                    finite_inside = 0
                    finite_nearby = 0
                    for cadence_index in range(left, right):
                        distance = abs(time_values[cadence_index] - center)
                        flux = flux_values[cadence_index]
                        if distance <= 0.5 * duration:
                            inside_count += 1
                            if np.isfinite(flux):
                                inside_flux[finite_inside] = flux
                                finite_inside += 1
                        if distance >= 1.5 * duration and distance <= 4.0 * duration:
                            nearby_count += 1
                            if np.isfinite(flux):
                                nearby_flux[finite_nearby] = flux
                                finite_nearby += 1

                    if inside_count < 3 or nearby_count < 10:
                        continue

                    depth_ppm = np.nan
                    scatter_ppm = 0.0
                    if finite_inside > 0 and finite_nearby > 0:
                        nearby_median = np.median(nearby_flux[:finite_nearby])
                        inside_median = np.median(inside_flux[:finite_inside])
                        depth_ppm = (nearby_median - inside_median) * 1.0e6
                        for local_index in range(finite_nearby):
                            nearby_flux[local_index] = abs(
                                nearby_flux[local_index] - nearby_median
                            )
                        scatter_ppm = (
                            1.4826 * np.median(nearby_flux[:finite_nearby]) * 1.0e6
                        )
                        if not np.isfinite(scatter_ppm) or scatter_ppm <= 0.0:
                            total = 0.0
                            # Refill because the scratch array now holds deviations.
                            finite_nearby_refill = 0
                            for cadence_index in range(left, right):
                                distance = abs(time_values[cadence_index] - center)
                                flux = flux_values[cadence_index]
                                if (
                                    distance >= 1.5 * duration
                                    and distance <= 4.0 * duration
                                    and np.isfinite(flux)
                                ):
                                    nearby_flux[finite_nearby_refill] = flux
                                    total += flux
                                    finite_nearby_refill += 1
                            mean = total / finite_nearby_refill
                            squared_error = 0.0
                            for local_index in range(finite_nearby_refill):
                                difference = nearby_flux[local_index] - mean
                                squared_error += difference * difference
                            scatter_ppm = (
                                np.sqrt(squared_error / finite_nearby_refill) * 1.0e6
                            )

                    snr = (
                        depth_ppm / (scatter_ppm / np.sqrt(inside_count))
                        if scatter_ppm > 0.0
                        else 0.0
                    )
                    if output_index >= maximum_events:
                        # The Python wrapper computes a strict upper bound.  A
                        # guard is retained so a corrupt view cannot overwrite.
                        continue
                    segments[candidate_index, output_index] = view_segments[view_index]
                    cycles[candidate_index, output_index] = cycle
                    centers[candidate_index, output_index] = center
                    depths_ppm[candidate_index, output_index] = depth_ppm
                    snrs[candidate_index, output_index] = snr
                    supported[candidate_index, output_index] = np.uint8(
                        np.isfinite(depth_ppm)
                        and depth_ppm > 0.0
                        and snr >= minimum_event_snr
                    )
                    output_index += 1

            row_counts[candidate_index] = output_index

        return row_counts, segments, cycles, centers, depths_ppm, snrs, supported


def _segment_set(value) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {int(item) for item in value}
    return {int(value)}


def _summarize(rows: list[dict]) -> dict:
    supported = [row for row in rows if row["supported"]]
    return {
        "testable_events": len(rows),
        "supported_events": len(supported),
        "support_fraction": len(supported) / len(rows) if rows else None,
        "median_supported_depth_ppm": (
            float(np.median([row["depth_ppm"] for row in supported]))
            if supported
            else None
        ),
        "minimum_supported_event_snr": (
            float(min(row["snr"] for row in supported)) if supported else None
        ),
    }


def _flatten_views(prepared_views: Sequence[Mapping]):
    times: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    offsets = [0]
    segments: list[int] = []
    maximum_view_points = 0
    for view in prepared_views:
        time = np.ascontiguousarray(view["time"], dtype=np.float64)
        flux = np.ascontiguousarray(view["flux"], dtype=np.float64)
        if time.size != flux.size:
            raise ValueError("event repeatability view has mismatched time/flux arrays")
        if time.size > 1 and np.any(np.diff(time) < 0.0):
            raise ValueError("event repeatability views must be time sorted")
        times.append(time)
        fluxes.append(flux)
        offsets.append(offsets[-1] + int(time.size))
        segments.append(int(view["segment"]))
        maximum_view_points = max(maximum_view_points, int(time.size))
    return (
        np.concatenate(times) if times else np.empty(0, dtype=np.float64),
        np.concatenate(fluxes) if fluxes else np.empty(0, dtype=np.float64),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(segments, dtype=np.int32),
        maximum_view_points,
    )


def event_repeatability_metrics_batch(
    candidates: Sequence[Mapping],
    *,
    prepared_views: Sequence[Mapping],
    holdout_segment=None,
    threads: int | None = None,
    minimum_event_snr: float = 3.0,
    event_repeatability_moduli: tuple[int, ...] = tuple(range(2, 11)),
) -> list[dict] | None:
    """Return the reference event diagnostics for every candidate in one batch."""

    if not NUMBA_AVAILABLE:
        return None
    if not candidates:
        return []
    requested = _requested_threads() if threads is None else max(1, int(threads))
    if get_num_threads() != requested:
        set_num_threads(requested)

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
    if np.any(~np.isfinite(periods)) or np.any(periods <= 0.0):
        raise ValueError("candidate periods must be finite and positive")

    time, flux, offsets, segments, maximum_view_points = _flatten_views(prepared_views)
    maximum_events = 0
    for period, epoch in zip(periods, epochs, strict=True):
        count = 0
        for view_index in range(segments.size):
            start = int(offsets[view_index])
            stop = int(offsets[view_index + 1])
            if stop - start < 13:
                continue
            first = int(np.floor((time[start] - epoch) / period)) - 1
            last = int(np.ceil((time[stop - 1] - epoch) / period)) + 1
            count += last - first + 1
        maximum_events = max(maximum_events, count)
    maximum_events = max(1, maximum_events)

    arrays = _event_repeatability_kernel(
        time,
        flux,
        offsets,
        segments,
        periods,
        epochs,
        durations_days,
        maximum_events,
        max(1, maximum_view_points),
        float(minimum_event_snr),
    )
    row_counts, out_segments, cycles, centers, depths, snrs, supported = arrays
    sealed_segments = _segment_set(holdout_segment)
    output: list[dict] = []
    for candidate_index in range(len(candidates)):
        rows: list[dict] = []
        for row_index in range(int(row_counts[candidate_index])):
            segment = int(out_segments[candidate_index, row_index])
            cycle = int(cycles[candidate_index, row_index])
            rows.append(
                {
                    "segment": segment,
                    "cycle": cycle,
                    "parity": int(cycle % 2),
                    "center_btjd": float(centers[candidate_index, row_index]),
                    "depth_ppm": float(depths[candidate_index, row_index]),
                    "snr": float(snrs[candidate_index, row_index]),
                    "supported": bool(supported[candidate_index, row_index]),
                    "sealed": segment in sealed_segments,
                }
            )

        summary = _summarize(rows)
        summary["search"] = _summarize([row for row in rows if not row["sealed"]])
        summary["sealed"] = _summarize([row for row in rows if row["sealed"]])
        summary["parity"] = {
            str(parity): _summarize(
                [row for row in rows if row["parity"] == parity]
            )
            for parity in (0, 1)
        }
        summary["modulo"] = {
            str(order): {
                str(remainder): _summarize(
                    [row for row in rows if row["cycle"] % order == remainder]
                )
                for remainder in range(order)
            }
            for order in event_repeatability_moduli
        }
        summary["event_support"] = [
            {
                "segment": int(row["segment"]),
                "cycle": int(row["cycle"]),
                "center_btjd": round(float(row["center_btjd"]), 8),
                "depth_ppm": float(row["depth_ppm"]),
                "snr": float(row["snr"]),
                "supported": bool(row["supported"]),
                "sealed": bool(row["sealed"]),
            }
            for row in rows
        ]
        summary["unsupported_event_centers_btjd"] = [
            round(float(row["center_btjd"]), 6)
            for row in rows
            if not row["supported"]
        ][:20]
        summary["minimum_event_snr"] = float(minimum_event_snr)
        output.append(summary)
    return output


def attach_event_repeatability_batch(
    candidates: Sequence[dict],
    *,
    prepared_views: Sequence[Mapping],
    holdout_segment=None,
    threads: int | None = None,
) -> list[dict] | None:
    """Attach exact batch metrics without changing candidate selection."""

    pending_indices = [
        index
        for index, candidate in enumerate(candidates)
        if not (candidate.get("holdout_verification") or {}).get(
            "event_repeatability"
        )
    ]
    pending = [candidates[index] for index in pending_indices]
    metrics = event_repeatability_metrics_batch(
        pending,
        prepared_views=prepared_views,
        holdout_segment=holdout_segment,
        threads=threads,
    )
    if metrics is None:
        return None
    for index, repeatability in zip(pending_indices, metrics, strict=True):
        verification = dict(candidates[index].get("holdout_verification") or {})
        verification["event_repeatability"] = repeatability
        candidates[index]["holdout_verification"] = verification
    return list(candidates)
