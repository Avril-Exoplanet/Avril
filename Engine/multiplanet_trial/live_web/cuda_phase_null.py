"""Exact CUDA event-window measurements for locked phase-null trials.

CPU keeps the reference cycle arithmetic and window boundaries.  CUDA only
gathers the exact same usable cadences and evaluates ragged medians in float64.
No trial, predicted event, or cadence is capped; VRAM limits only change the
number of event windows submitted per batch.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from collections.abc import Mapping

import numpy as np


_CUDA_SOURCE = r"""
extern "C" __global__ void gather_phase_event_windows(
    const double *time_values,
    const double *flux_values,
    const unsigned char *usable_values,
    const double *centers,
    const long long *left_values,
    const long long *right_values,
    const double duration,
    const int window_count,
    const int gather_mode,
    const int value_capacity,
    unsigned long long *inside_counts,
    unsigned long long *nearby_counts,
    double *inside_values,
    double *nearby_values
) {
    const int window = (int)blockIdx.x;
    if (window >= window_count) return;
    const double center = centers[window];
    for (long long cadence = left_values[window] + threadIdx.x;
         cadence < right_values[window];
         cadence += blockDim.x) {
        if (!usable_values[cadence]) continue;
        const double distance = fabs(time_values[cadence] - center);
        const double flux = flux_values[cadence];
        if (distance <= 0.5 * duration) {
            const unsigned long long position = atomicAdd(
                inside_counts + window, 1ULL
            );
            if (gather_mode != 0
                && position < (unsigned long long)value_capacity) {
                inside_values[(unsigned long long)window * value_capacity
                              + position] = flux;
            }
        }
        if (distance >= 1.5 * duration && distance <= 4.0 * duration) {
            const unsigned long long position = atomicAdd(
                nearby_counts + window, 1ULL
            );
            if (gather_mode != 0
                && position < (unsigned long long)value_capacity) {
                nearby_values[(unsigned long long)window * value_capacity
                              + position] = flux;
            }
        }
    }
}

__device__ __forceinline__ void swap_double(double *first, double *second) {
    const double value = *first;
    *first = *second;
    *second = value;
}

// In-place Numerical Recipes quickselect.  Unlike cupy.nanmedian this only
// selects the one (or two) middle order statistics and therefore does not
// fully sort every ragged event window.  All inputs are finite float64 fluxes.
__device__ double select_kth(double *values, long long count, long long kth) {
    long long low = 0;
    long long high = count - 1;
    for (;;) {
        if (high <= low) return values[kth];
        if (high == low + 1) {
            if (values[high] < values[low]) {
                swap_double(values + low, values + high);
            }
            return values[kth];
        }
        const long long middle = (low + high) >> 1;
        swap_double(values + middle, values + low + 1);
        if (values[low] > values[high]) {
            swap_double(values + low, values + high);
        }
        if (values[low + 1] > values[high]) {
            swap_double(values + low + 1, values + high);
        }
        if (values[low] > values[low + 1]) {
            swap_double(values + low, values + low + 1);
        }
        long long left = low + 1;
        long long right = high;
        const double pivot = values[low + 1];
        for (;;) {
            do { ++left; } while (values[left] < pivot);
            do { --right; } while (values[right] > pivot);
            if (right < left) break;
            swap_double(values + left, values + right);
        }
        values[low + 1] = values[right];
        values[right] = pivot;
        if (right >= kth) high = right - 1;
        if (right <= kth) low = left;
    }
}

__device__ double exact_median(double *values, unsigned long long count) {
    const long long size = (long long)count;
    const long long upper_index = size >> 1;
    const double upper = select_kth(values, size, upper_index);
    if ((size & 1LL) != 0) return upper;
    // quickselect leaves every lower-half value at or below the selected
    // upper median, so the largest of that half is the exact lower median.
    double lower = values[0];
    for (long long index = 1; index < upper_index; ++index) {
        lower = fmax(lower, values[index]);
    }
    return 0.5 * (lower + upper);
}

extern "C" __global__ void median_phase_event_windows(
    double *inside_values,
    double *nearby_values,
    const unsigned long long *inside_counts,
    const unsigned long long *nearby_counts,
    const int window_count,
    const int value_capacity,
    double *depth_values
) {
    for (int window = (int)(blockIdx.x * blockDim.x + threadIdx.x);
         window < window_count;
         window += (int)(blockDim.x * gridDim.x)) {
        const unsigned long long inside_count = inside_counts[window];
        const unsigned long long nearby_count = nearby_counts[window];
        if (inside_count == 0ULL || nearby_count == 0ULL) {
            depth_values[window] = nan("");
            continue;
        }
        double *inside = inside_values
            + (unsigned long long)window * value_capacity;
        double *nearby = nearby_values
            + (unsigned long long)window * value_capacity;
        depth_values[window] = exact_median(nearby, nearby_count)
            - exact_median(inside, inside_count);
    }
}
"""


class CudaPhaseNullMeasurer:
    """Bind sector light curves once and evaluate exact phase-null windows."""

    def __init__(self) -> None:
        try:
            import cupy as cp
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("CuPy is required for CUDA phase-null trials") from exc
        self.cp = cp
        module = cp.RawModule(
            code=_CUDA_SOURCE,
            options=(
                "-std=c++11",
                "--fmad=false",
                "--prec-div=true",
                "--prec-sqrt=true",
            ),
            name_expressions=(
                "gather_phase_event_windows",
                "median_phase_event_windows",
            ),
        )
        self.gather_kernel = module.get_function("gather_phase_event_windows")
        self.median_kernel = module.get_function("median_phase_event_windows")
        self.maximum_buffer_bytes = max(
            64 * 1024**2,
            int(
                os.environ.get(
                    "VLORA_CUDA_PHASE_NULL_BUFFER_BYTES",
                    str(768 * 1024**2),
                )
            ),
        )
        self.maximum_windows = max(
            1, int(os.environ.get("VLORA_CUDA_PHASE_NULL_WINDOWS", "16384"))
        )
        self._view_signature = None
        self._bindings: dict[int, dict] = {}
        self.calls = 0
        self.trials = 0
        self.windows = 0
        self.window_cadences = 0
        self.seconds = 0.0

    @staticmethod
    def _signature(views: Mapping[int, Mapping]) -> tuple:
        return tuple(
            (
                int(segment),
                id(views[segment]["time"]),
                id(views[segment]["flux"]),
                int(np.asarray(views[segment]["time"]).size),
            )
            for segment in views
        )

    def _bind_views(self, views: Mapping[int, Mapping]) -> None:
        signature = self._signature(views)
        if signature == self._view_signature:
            return
        cp = self.cp
        bindings: dict[int, dict] = {}
        times: list[np.ndarray] = []
        fluxes: list[np.ndarray] = []
        cursor = 0
        for segment, view in views.items():
            time_values = np.ascontiguousarray(view["time"], dtype=np.float64)
            flux_values = np.ascontiguousarray(view["flux"], dtype=np.float64)
            if time_values.size > 1 and np.any(np.diff(time_values) < 0):
                raise ValueError("CUDA phase-null time arrays must be sorted")
            bindings[int(segment)] = {
                "time": time_values,
                "flux": flux_values,
                "offset": cursor,
            }
            times.append(time_values)
            fluxes.append(flux_values)
            cursor += time_values.size
        self._time_gpu = cp.asarray(np.concatenate(times))
        self._flux_gpu = cp.asarray(np.concatenate(fluxes))
        self._bindings = bindings
        self._view_signature = signature

    @staticmethod
    def _plan_segment(
        time_values: np.ndarray,
        *,
        period: float,
        trial_epochs: np.ndarray,
        duration_days: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        epochs = np.asarray(trial_epochs, dtype=np.float64)
        first = np.floor((time_values[0] - epochs) / period).astype(np.int64) - 1
        last = np.ceil((time_values[-1] - epochs) / period).astype(np.int64) + 1
        maximum_cycles = int(np.max(last - first + 1, initial=0))
        if maximum_cycles <= 0:
            empty_i64 = np.empty(0, dtype=np.int64)
            return empty_i64, empty_i64, np.empty(0), empty_i64, empty_i64
        cycle_orders = np.arange(maximum_cycles, dtype=np.int64)
        cycles = first[:, None] + cycle_orders[None, :]
        valid_cycles = cycles <= last[:, None]
        trial_grid = np.broadcast_to(
            np.arange(epochs.size, dtype=np.int64)[:, None], cycles.shape
        )
        order_grid = np.broadcast_to(cycle_orders[None, :], cycles.shape)
        centers = epochs[:, None] + cycles.astype(np.float64) * period
        trial_values = trial_grid[valid_cycles]
        order_values = order_grid[valid_cycles]
        center_values = centers[valid_cycles]
        left = np.searchsorted(
            time_values,
            center_values - 4.0 * duration_days,
            side="left",
        ).astype(np.int64)
        right = np.searchsorted(
            time_values,
            center_values + 4.0 * duration_days,
            side="right",
        ).astype(np.int64)
        keep = right - left >= 13
        return (
            trial_values[keep],
            order_values[keep],
            center_values[keep],
            left[keep],
            right[keep],
        )

    def _measure_tasks(
        self,
        usable_gpu,
        trial_indices: np.ndarray,
        event_orders: np.ndarray,
        centers: np.ndarray,
        left_values: np.ndarray,
        right_values: np.ndarray,
        minimum_inside: np.ndarray,
        minimum_nearby: np.ndarray,
        duration_sigmas: np.ndarray,
        control_centers: np.ndarray,
        duration_days: float,
    ) -> list[tuple[int, int, float, float]]:
        cp = self.cp
        if trial_indices.size == 0:
            return []
        output: list[tuple[int, int, float, float]] = []
        start = 0
        while start < trial_indices.size:
            stop = min(trial_indices.size, start + self.maximum_windows)
            # The exact searchsorted bounds are already known on the CPU.  A
            # window can never gather more values than ``right-left``.  Use
            # that safe capacity directly instead of scanning every cadence
            # once merely to discover counts and then scanning it again to
            # gather values.
            while True:
                local_spans = right_values[start:stop] - left_values[start:stop]
                maximum_values = max(1, int(np.max(local_spans, initial=0)))
                count = stop - start
                bytes_per_window = maximum_values * 8 * 2
                allowed = max(1, self.maximum_buffer_bytes // bytes_per_window)
                if count <= allowed:
                    break
                stop = start + min(count - 1, int(allowed))
            local_centers = centers[start:stop]
            local_left = left_values[start:stop]
            local_right = right_values[start:stop]
            count = stop - start
            centers_gpu = cp.asarray(local_centers)
            left_gpu = cp.asarray(local_left)
            right_gpu = cp.asarray(local_right)
            inside_counts = cp.zeros(count, dtype=cp.uint64)
            nearby_counts = cp.zeros(count, dtype=cp.uint64)
            inside_values = cp.empty(
                (count, maximum_values), dtype=cp.float64
            )
            nearby_values = cp.empty(
                (count, maximum_values), dtype=cp.float64
            )
            common = (
                self._time_gpu,
                self._flux_gpu,
                usable_gpu,
                centers_gpu,
                left_gpu,
                right_gpu,
                np.float64(duration_days),
                np.int32(count),
            )
            self.gather_kernel(
                (count,),
                (256,),
                (
                    *common,
                    np.int32(1),
                    np.int32(maximum_values),
                    inside_counts,
                    nearby_counts,
                    inside_values,
                    nearby_values,
                ),
            )
            depths = cp.empty(count, dtype=cp.float64)
            median_threads = 128
            median_blocks = min(
                65535,
                max(1, (count + median_threads - 1) // median_threads),
            )
            self.median_kernel(
                (median_blocks,),
                (median_threads,),
                (
                    inside_values,
                    nearby_values,
                    inside_counts,
                    nearby_counts,
                    np.int32(count),
                    np.int32(maximum_values),
                    depths,
                ),
            )
            cp.cuda.Stream.null.synchronize()
            host_depths = cp.asnumpy(depths)
            host_inside = cp.asnumpy(inside_counts)
            host_nearby = cp.asnumpy(nearby_counts)
            for local_index, depth in enumerate(host_depths):
                task_index = start + local_index
                if (
                    host_inside[local_index] < minimum_inside[task_index]
                    or host_nearby[local_index] < minimum_nearby[task_index]
                ):
                    continue
                output.append(
                    (
                        int(trial_indices[task_index]),
                        int(event_orders[task_index]),
                        (
                            float(depth)
                            - float(control_centers[task_index])
                        )
                        * 1e6,
                        float(duration_sigmas[task_index]) * 1e6,
                    )
                )
            self.windows += count
            self.window_cadences += int(np.sum(local_right - local_left))
            start = stop
        return output

    def measure(
        self,
        views: Mapping[int, Mapping],
        noise_models: Mapping[int, Mapping],
        usable: Mapping[int, np.ndarray],
        *,
        period: float,
        trial_epochs: np.ndarray,
        duration_days: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        started = time.perf_counter()
        self._bind_views(views)
        rows: list[list[tuple[int, float, float]]] = [
            [] for _ in range(len(trial_epochs))
        ]
        usable_rows: list[np.ndarray] = []
        trial_rows: list[np.ndarray] = []
        event_order_rows: list[np.ndarray] = []
        center_rows: list[np.ndarray] = []
        left_rows: list[np.ndarray] = []
        right_rows: list[np.ndarray] = []
        minimum_inside_rows: list[np.ndarray] = []
        minimum_nearby_rows: list[np.ndarray] = []
        sigma_rows: list[np.ndarray] = []
        control_rows: list[np.ndarray] = []
        for segment_order, segment in enumerate(views):
            model = noise_models[int(segment)]
            binding = self._bindings[int(segment)]
            usable_rows.append(
                np.asarray(usable[int(segment)], dtype=np.uint8)
            )
            if float(model["duration_sigma"]) <= 0:
                continue
            trial_indices, cycle_orders, centers, left_values, right_values = (
                self._plan_segment(
                    binding["time"],
                    period=float(period),
                    trial_epochs=np.asarray(trial_epochs, dtype=np.float64),
                    duration_days=float(duration_days),
                )
            )
            if trial_indices.size == 0:
                continue
            left_values = left_values + int(binding["offset"])
            right_values = right_values + int(binding["offset"])
            cadence = float(views[segment]["cadence_days"])
            expected_inside = max(
                3, int(round(duration_days / max(cadence, 1e-9)))
            )
            minimum_inside = max(
                3, int(math.ceil(0.45 * expected_inside))
            )
            minimum_nearby = max(
                10, int(math.ceil(1.5 * expected_inside))
            )
            trial_rows.append(trial_indices)
            event_order_rows.append(
                segment_order * 1_000_000 + cycle_orders
            )
            center_rows.append(centers)
            left_rows.append(left_values)
            right_rows.append(right_values)
            minimum_inside_rows.append(
                np.full(trial_indices.size, minimum_inside, dtype=np.int64)
            )
            minimum_nearby_rows.append(
                np.full(trial_indices.size, minimum_nearby, dtype=np.int64)
            )
            sigma_rows.append(
                np.full(
                    trial_indices.size,
                    float(model["duration_sigma"]),
                    dtype=np.float64,
                )
            )
            control_rows.append(
                np.full(
                    trial_indices.size,
                    float(model["control_center"]),
                    dtype=np.float64,
                )
            )
        if trial_rows:
            measured = self._measure_tasks(
                self.cp.asarray(np.concatenate(usable_rows)),
                np.concatenate(trial_rows),
                np.concatenate(event_order_rows),
                np.concatenate(center_rows),
                np.concatenate(left_rows),
                np.concatenate(right_rows),
                np.concatenate(minimum_inside_rows),
                np.concatenate(minimum_nearby_rows),
                np.concatenate(sigma_rows),
                np.concatenate(control_rows),
                float(duration_days),
            )
        else:
            measured = []
        for trial, order, depth, sigma in measured:
            rows[trial].append((order, depth, sigma))
        event_counts = np.asarray([len(row) for row in rows], dtype=np.int64)
        maximum = int(np.max(event_counts, initial=0))
        depth_ppm = np.full((len(rows), maximum), np.nan, dtype=np.float64)
        sigma_ppm = np.full((len(rows), maximum), np.nan, dtype=np.float64)
        for trial, row in enumerate(rows):
            row.sort(key=lambda value: value[0])
            for event, (_order, depth, sigma) in enumerate(row):
                depth_ppm[trial, event] = depth
                sigma_ppm[trial, event] = sigma
        self.calls += 1
        self.trials += len(trial_epochs)
        self.seconds += time.perf_counter() - started
        return depth_ppm, sigma_ppm, event_counts

    def diagnostics(self) -> dict:
        return {
            "backend": "vlora-cupy-exact-phase-null-v1",
            "calls": int(self.calls),
            "trials": int(self.trials),
            "event_windows": int(self.windows),
            "window_cadences_examined": int(self.window_cadences),
            "seconds": round(float(self.seconds), 6),
            "float64": True,
            "trial_or_event_truncation": False,
        }


_ENGINE: CudaPhaseNullMeasurer | None = None


def get_cuda_phase_null_measurer() -> CudaPhaseNullMeasurer:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = CudaPhaseNullMeasurer()
    return _ENGINE


__all__ = ["CudaPhaseNullMeasurer", "get_cuda_phase_null_measurer"]
