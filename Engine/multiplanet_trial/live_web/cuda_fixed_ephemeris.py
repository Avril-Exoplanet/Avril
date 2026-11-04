"""CUDA batch measurements for VLora's locked-ephemeris validation.

The reference statistic is unchanged: median nearby flux minus median transit
flux, nearby MAD scatter, the same S/N thresholds, and the same distinct-event
count.  Long-period candidates use disjoint predicted-event windows, exactly
like the compiled CPU accelerator, but all window cadences are gathered in
parallel on CUDA and the ragged medians are evaluated by CuPy.  Short-period
overlapping-window cases retain the compiled CPU implementation.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .fixed_ephemeris_acceleration import measure_fixed_ephemerides_compiled


_CUDA_SOURCE = r"""
extern "C" __global__ void screen_fixed_ephemeris_windows(
    const double *time_values,
    const double *flux_values,
    const double *periods,
    const double *epochs,
    const double *durations,
    const int *window_candidates,
    const long long *window_left,
    const long long *window_right,
    const int window_count,
    unsigned long long *inside_counts,
    unsigned long long *nearby_counts,
    unsigned long long *event_counts,
    double *inside_sums,
    double *inside_sumsq,
    double *nearby_sums,
    double *nearby_sumsq
) {
    const int window_index = (int)blockIdx.x;
    if (window_index >= window_count) return;
    const int candidate = window_candidates[window_index];
    const double period = periods[candidate];
    const double epoch = epochs[candidate];
    const double duration = durations[candidate];
    __shared__ unsigned int event_has_inside;
    if (threadIdx.x == 0) event_has_inside = 0U;
    __syncthreads();

    for (long long cadence = window_left[window_index] + threadIdx.x;
         cadence < window_right[window_index]; cadence += blockDim.x) {
        const double shifted = time_values[cadence] - epoch + 0.5 * period;
        const double phase =
            shifted - floor(shifted / period) * period - 0.5 * period;
        const double absolute_phase = fabs(phase);
        const double flux = flux_values[cadence];
        if (!isfinite(flux)) continue;
        if (absolute_phase <= 0.5 * duration) {
            atomicAdd(inside_counts + candidate, 1ULL);
            atomicAdd(inside_sums + candidate, flux);
            atomicAdd(inside_sumsq + candidate, flux * flux);
            atomicExch(&event_has_inside, 1U);
        }
        if (absolute_phase >= 1.5 * duration
            && absolute_phase <= 4.0 * duration) {
            atomicAdd(nearby_counts + candidate, 1ULL);
            atomicAdd(nearby_sums + candidate, flux);
            atomicAdd(nearby_sumsq + candidate, flux * flux);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && event_has_inside) {
        atomicAdd(event_counts + candidate, 1ULL);
    }
}

extern "C" __global__ void gather_fixed_ephemeris_windows(
    const double *time_values,
    const double *flux_values,
    const double *periods,
    const double *epochs,
    const double *durations,
    const int *window_candidates,
    const long long *window_left,
    const long long *window_right,
    const int window_count,
    const int gather_mode,
    const int value_capacity,
    unsigned long long *inside_counts,
    unsigned long long *nearby_counts,
    unsigned long long *inside_finite_counts,
    unsigned long long *nearby_finite_counts,
    unsigned long long *event_counts,
    double *inside_values,
    double *nearby_values
) {
    const int window_index = (int)blockIdx.x;
    if (window_index >= window_count) return;
    const int candidate = window_candidates[window_index];
    const double period = periods[candidate];
    const double epoch = epochs[candidate];
    const double duration = durations[candidate];
    __shared__ unsigned int event_has_inside;
    if (threadIdx.x == 0) event_has_inside = 0U;
    __syncthreads();

    for (long long cadence = window_left[window_index] + threadIdx.x;
         cadence < window_right[window_index];
         cadence += blockDim.x) {
        const double shifted =
            time_values[cadence] - epoch + 0.5 * period;
        const double phase =
            shifted - floor(shifted / period) * period - 0.5 * period;
        const double absolute_phase = fabs(phase);
        const double flux = flux_values[cadence];
        if (absolute_phase <= 0.5 * duration) {
            if (gather_mode == 0) {
                atomicAdd(inside_counts + candidate, 1ULL);
                atomicExch(&event_has_inside, 1U);
                if (isfinite(flux)) {
                    atomicAdd(inside_finite_counts + candidate, 1ULL);
                }
            } else if (isfinite(flux)) {
                if (gather_mode == 2) {
                    atomicAdd(inside_counts + candidate, 1ULL);
                    atomicExch(&event_has_inside, 1U);
                }
                const unsigned long long position = atomicAdd(
                    inside_finite_counts + candidate, 1ULL
                );
                if (position < (unsigned long long)value_capacity) {
                    inside_values[
                        (unsigned long long)candidate * value_capacity + position
                    ] = flux;
                }
            }
        }
        if (absolute_phase >= 1.5 * duration
            && absolute_phase <= 4.0 * duration) {
            if (gather_mode == 0) {
                atomicAdd(nearby_counts + candidate, 1ULL);
                if (isfinite(flux)) {
                    atomicAdd(nearby_finite_counts + candidate, 1ULL);
                }
            } else if (isfinite(flux)) {
                if (gather_mode == 2) {
                    atomicAdd(nearby_counts + candidate, 1ULL);
                }
                const unsigned long long position = atomicAdd(
                    nearby_finite_counts + candidate, 1ULL
                );
                if (position < (unsigned long long)value_capacity) {
                    nearby_values[
                        (unsigned long long)candidate * value_capacity + position
                    ] = flux;
                }
            }
        }
    }
    __syncthreads();
    if ((gather_mode == 0 || gather_mode == 2)
        && threadIdx.x == 0 && event_has_inside) {
        atomicAdd(event_counts + candidate, 1ULL);
    }
}
"""


def _metric_rows(
    depths: np.ndarray,
    snrs: np.ndarray,
    scatters: np.ndarray,
    inside_counts: np.ndarray,
    nearby_counts: np.ndarray,
    event_counts: np.ndarray,
) -> list[dict]:
    output: list[dict] = []
    for index in range(inside_counts.size):
        if inside_counts[index] < 3 or nearby_counts[index] < 10:
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
        scatter = float(scatters[index])
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
                "local_scatter_ppm": float(scatter * 1e6),
                "status": status,
            }
        )
    return output


@dataclass
class _BlockBinding:
    time: np.ndarray
    flux: np.ndarray
    time_gpu: object
    flux_gpu: object


class CudaFixedEphemerisMeasurer:
    """Bind temporal blocks once and measure candidate batches on CUDA."""

    def __init__(self, prepared_blocks) -> None:
        try:
            import cupy as cp
        except ImportError as exc:  # pragma: no cover - optional backend
            raise RuntimeError("CuPy is required for CUDA fixed ephemerides") from exc
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
                "gather_fixed_ephemeris_windows",
                "screen_fixed_ephemeris_windows",
            ),
        )
        self.kernel = module.get_function("gather_fixed_ephemeris_windows")
        self.screen_kernel = module.get_function("screen_fixed_ephemeris_windows")
        self.blocks = []
        for local in prepared_blocks:
            time_values = np.ascontiguousarray(local.time, dtype=np.float64)
            flux_values = np.ascontiguousarray(local.flux, dtype=np.float64)
            if time_values.size > 1 and np.any(np.diff(time_values) < 0):
                raise ValueError("CUDA fixed-ephemeris time arrays must be sorted")
            self.blocks.append(
                _BlockBinding(
                    time=time_values,
                    flux=flux_values,
                    time_gpu=cp.asarray(time_values),
                    flux_gpu=cp.asarray(flux_values),
                )
            )
        self.maximum_candidates = max(
            1, int(os.environ.get("VLORA_CUDA_FIXED_EPHEMERIS_BATCH", "256"))
        )
        self.maximum_buffer_bytes = max(
            64 * 1024**2,
            int(
                os.environ.get(
                    "VLORA_CUDA_FIXED_EPHEMERIS_BUFFER_BYTES",
                    str(768 * 1024**2),
                )
            ),
        )
        self.one_pass_gather = os.environ.get(
            "VLORA_CUDA_FIXED_EPHEMERIS_ONE_PASS", "off"
        ).strip().lower() not in {"0", "false", "off", "no"}
        self.sort_by_window_size = os.environ.get(
            "VLORA_CUDA_FIXED_EPHEMERIS_SORT_WINDOWS", "on"
        ).strip().lower() not in {"0", "false", "off", "no"}
        self.calls = 0
        self.candidates = 0
        self.cuda_candidates = 0
        self.cpu_fallback_candidates = 0
        self.windows = 0
        self.window_points = 0
        self.seconds = 0.0
        self.kernel_window_passes = 0
        self.allocated_value_cells = 0
        self.screen_calls = 0
        self.screen_candidates = 0
        self.screen_seconds = 0.0
        self.screen_windows = 0
        self.screen_window_points = 0

    @staticmethod
    def _candidate_arrays(candidates: list[Mapping]):
        periods = np.asarray(
            [float(row["period_days"]) for row in candidates], dtype=np.float64
        )
        epochs = np.asarray(
            [float(row["epoch_btjd"]) for row in candidates], dtype=np.float64
        )
        durations = np.asarray(
            [float(row["duration_hours"]) / 24.0 for row in candidates],
            dtype=np.float64,
        )
        return periods, epochs, durations

    @staticmethod
    def _plan_candidate_windows(
        time_values: np.ndarray,
        period: float,
        epoch: float,
        duration: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        first_cycle = int(math.floor((time_values[0] - epoch) / period)) - 1
        last_cycle = int(math.ceil((time_values[-1] - epoch) / period)) + 1
        cycles = np.arange(first_cycle, last_cycle + 1, dtype=np.float64)
        centers = epoch + cycles * period
        paddings = 1e-10 * np.maximum.reduce(
            (
                np.ones(centers.shape, dtype=np.float64),
                np.abs(centers),
                np.full(centers.shape, abs(epoch), dtype=np.float64),
                np.full(centers.shape, abs(period), dtype=np.float64),
            )
        )
        left = np.searchsorted(
            time_values, centers - 4.0 * duration - paddings, side="left"
        ).astype(np.int64)
        right = np.searchsorted(
            time_values, centers + 4.0 * duration + paddings, side="right"
        ).astype(np.int64)
        keep = right > left
        return left[keep], right[keep]

    def _measure_cuda_long(
        self,
        binding: _BlockBinding,
        candidates: list[Mapping],
    ) -> list[dict]:
        cp = self.cp
        periods, epochs, durations = self._candidate_arrays(candidates)
        planned: list[tuple[np.ndarray, np.ndarray]] = []
        window_points = np.zeros(len(candidates), dtype=np.int64)
        for index, (period, epoch, duration) in enumerate(
            zip(periods, epochs, durations)
        ):
            left, right = self._plan_candidate_windows(
                binding.time, float(period), float(epoch), float(duration)
            )
            planned.append((left, right))
            window_points[index] = int(np.sum(right - left, dtype=np.int64))

        output: list[dict | None] = [None] * len(candidates)
        processing_order = (
            np.argsort(window_points, kind="stable")
            if self.sort_by_window_size
            else np.arange(len(candidates), dtype=np.int64)
        )
        start = 0
        while start < len(candidates):
            stop = start
            maximum_points = 1
            while stop < len(candidates) and stop - start < self.maximum_candidates:
                original_index = int(processing_order[stop])
                proposed_maximum = max(
                    maximum_points, int(window_points[original_index])
                )
                proposed_count = stop - start + 1
                # Two value matrices plus one equally sized temporary for MAD.
                required = proposed_count * proposed_maximum * 8 * 3
                if stop > start and required > self.maximum_buffer_bytes:
                    break
                maximum_points = proposed_maximum
                stop += 1

            batch_indices = np.asarray(
                processing_order[start:stop], dtype=np.int64
            )
            local_candidates = [candidates[int(index)] for index in batch_indices]
            local_periods = periods[batch_indices]
            local_epochs = epochs[batch_indices]
            local_durations = durations[batch_indices]
            candidate_windows: list[np.ndarray] = []
            left_windows: list[np.ndarray] = []
            right_windows: list[np.ndarray] = []
            for local_index, original_index in enumerate(batch_indices):
                left, right = planned[int(original_index)]
                candidate_windows.append(
                    np.full(left.size, local_index, dtype=np.int32)
                )
                left_windows.append(left)
                right_windows.append(right)
            window_candidates = np.concatenate(candidate_windows)
            window_left = np.concatenate(left_windows)
            window_right = np.concatenate(right_windows)
            local_count = len(local_candidates)
            capacity = max(1, maximum_points)

            periods_gpu = cp.asarray(local_periods)
            epochs_gpu = cp.asarray(local_epochs)
            durations_gpu = cp.asarray(local_durations)
            window_candidates_gpu = cp.asarray(window_candidates)
            window_left_gpu = cp.asarray(window_left)
            window_right_gpu = cp.asarray(window_right)
            counters = [cp.zeros(local_count, dtype=cp.uint64) for _ in range(5)]
            dummy_values = cp.empty(1, dtype=cp.float64)
            common_arguments = (
                binding.time_gpu,
                binding.flux_gpu,
                periods_gpu,
                epochs_gpu,
                durations_gpu,
                window_candidates_gpu,
                window_left_gpu,
                window_right_gpu,
                np.int32(window_candidates.size),
            )
            if self.one_pass_gather:
                # Every prepared branch is already finite.  The complete
                # predicted-window span is therefore a safe exact capacity.
                # Sorting candidates by this span keeps padding bounded and
                # removes the old full count pass over every cadence window.
                capacity = max(1, maximum_points)
            else:
                # Reference two-pass path retained for byte-for-byte A/B
                # testing and as a conservative deployment fallback.
                self.kernel(
                    (int(window_candidates.size),),
                    (256,),
                    (
                        *common_arguments,
                        np.int32(0),
                        np.int32(0),
                        *counters,
                        dummy_values,
                        dummy_values,
                    ),
                )
                self.kernel_window_passes += 1
                finite_inside = cp.asnumpy(counters[2])
                finite_nearby = cp.asnumpy(counters[3])
                capacity = max(
                    1,
                    int(np.max(finite_inside, initial=0)),
                    int(np.max(finite_nearby, initial=0)),
                )
                counters[2].fill(0)
                counters[3].fill(0)
            inside_values = cp.full(
                (local_count, capacity), cp.nan, dtype=cp.float64
            )
            nearby_values = cp.full(
                (local_count, capacity), cp.nan, dtype=cp.float64
            )
            self.kernel(
                (int(window_candidates.size),),
                (256,),
                (
                    *common_arguments,
                    np.int32(2 if self.one_pass_gather else 1),
                    np.int32(capacity),
                    *counters,
                    inside_values,
                    nearby_values,
                ),
            )
            self.kernel_window_passes += 1
            self.allocated_value_cells += int(2 * local_count * capacity)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                baseline = cp.nanmedian(nearby_values, axis=1)
                transit = cp.nanmedian(inside_values, axis=1)
                depths = baseline - transit
                deviations = cp.abs(nearby_values - baseline[:, None])
                scatters = 1.4826 * cp.nanmedian(deviations, axis=1)
                invalid = ~cp.isfinite(scatters) | (scatters <= 0)
                if bool(cp.any(invalid)):
                    fallback = cp.nanstd(nearby_values, axis=1)
                    scatters = cp.where(invalid, fallback, scatters)
                inside_counts_gpu = counters[0]
                snrs = cp.where(
                    scatters > 0,
                    depths / (scatters / cp.sqrt(inside_counts_gpu)),
                    0.0,
                )
            cp.cuda.Stream.null.synchronize()
            metric_rows = _metric_rows(
                cp.asnumpy(depths),
                cp.asnumpy(snrs),
                cp.asnumpy(scatters),
                cp.asnumpy(counters[0]),
                cp.asnumpy(counters[1]),
                cp.asnumpy(counters[4]),
            )
            for original_index, row in zip(batch_indices, metric_rows):
                output[int(original_index)] = row
            self.windows += int(window_candidates.size)
            self.window_points += int(np.sum(window_right - window_left))
            start = stop
        if not all(row is not None for row in output):
            raise RuntimeError("CUDA fixed-ephemeris output reorder was incomplete")
        return [row for row in output if row is not None]

    def _screen_cuda_long(
        self,
        binding: _BlockBinding,
        candidates: list[Mapping],
    ) -> list[dict]:
        """Return a cheap mean/variance screen without allocating ragged flux.

        This is a routing statistic only.  It never replaces the robust
        median/MAD measurement of a reported candidate.  The kernel visits
        the same disjoint predicted-event windows as the exact path but keeps
        seven scalar accumulators per trial instead of padded flux matrices.
        """

        cp = self.cp
        periods, epochs, durations = self._candidate_arrays(candidates)
        planned = [
            self._plan_candidate_windows(
                binding.time, float(period), float(epoch), float(duration)
            )
            for period, epoch, duration in zip(periods, epochs, durations)
        ]
        output: list[dict | None] = [None] * len(candidates)
        batch_limit = max(
            self.maximum_candidates,
            int(os.environ.get("VLORA_CUDA_FIXED_SCREEN_BATCH", "8192")),
        )
        for start in range(0, len(candidates), batch_limit):
            stop = min(start + batch_limit, len(candidates))
            local_count = stop - start
            candidate_windows: list[np.ndarray] = []
            left_windows: list[np.ndarray] = []
            right_windows: list[np.ndarray] = []
            for local_index, (left, right) in enumerate(planned[start:stop]):
                candidate_windows.append(
                    np.full(left.size, local_index, dtype=np.int32)
                )
                left_windows.append(left)
                right_windows.append(right)
            window_candidates = np.concatenate(candidate_windows)
            window_left = np.concatenate(left_windows)
            window_right = np.concatenate(right_windows)
            counts = [cp.zeros(local_count, dtype=cp.uint64) for _ in range(3)]
            moments = [cp.zeros(local_count, dtype=cp.float64) for _ in range(4)]
            self.screen_kernel(
                (int(window_candidates.size),),
                (256,),
                (
                    binding.time_gpu,
                    binding.flux_gpu,
                    cp.asarray(periods[start:stop]),
                    cp.asarray(epochs[start:stop]),
                    cp.asarray(durations[start:stop]),
                    cp.asarray(window_candidates),
                    cp.asarray(window_left),
                    cp.asarray(window_right),
                    np.int32(window_candidates.size),
                    *counts,
                    *moments,
                ),
            )
            inside, nearby, events = [cp.asnumpy(value) for value in counts]
            inside_sum, inside_sumsq, nearby_sum, nearby_sumsq = moments
            inside_gpu = counts[0]
            nearby_gpu = counts[1]
            safe_inside = cp.maximum(inside_gpu, 1)
            safe_nearby = cp.maximum(nearby_gpu, 1)
            transit = inside_sum / safe_inside
            baseline = nearby_sum / safe_nearby
            variance = cp.maximum(
                nearby_sumsq / safe_nearby - baseline * baseline, 0.0
            )
            scatter = cp.sqrt(variance)
            depth = baseline - transit
            snr = cp.where(
                scatter > 0,
                depth / (scatter / cp.sqrt(safe_inside)),
                0.0,
            )
            depth_cpu = cp.asnumpy(depth)
            scatter_cpu = cp.asnumpy(scatter)
            snr_cpu = cp.asnumpy(snr)
            for local_index in range(local_count):
                if inside[local_index] < 3 or nearby[local_index] < 10:
                    row = {
                        "depth_ppm": None,
                        "snr": None,
                        "events": 0,
                        "transit_points": int(inside[local_index]),
                        "baseline_points": int(nearby[local_index]),
                        "status": "insufficient_coverage",
                        "screen_only": True,
                    }
                else:
                    row = {
                        "depth_ppm": float(depth_cpu[local_index] * 1e6),
                        "snr": float(snr_cpu[local_index]),
                        "events": int(events[local_index]),
                        "transit_points": int(inside[local_index]),
                        "baseline_points": int(nearby[local_index]),
                        "local_scatter_ppm": float(
                            scatter_cpu[local_index] * 1e6
                        ),
                        "status": "screen_only",
                        "screen_only": True,
                    }
                output[start + local_index] = row
            self.screen_windows += int(window_candidates.size)
            self.screen_window_points += int(
                np.sum(window_right - window_left, dtype=np.int64)
            )
        if not all(row is not None for row in output):
            raise RuntimeError("CUDA screen output reorder was incomplete")
        return [row for row in output if row is not None]

    def screen_prepared_blocks(self, candidates, blocks, prepared_blocks):
        """Screen candidate ephemerides in all blocks; exact validation later."""

        if len(self.blocks) != len(prepared_blocks) or len(blocks) != len(self.blocks):
            raise ValueError("CUDA fixed-ephemeris block layout changed after binding")
        started = time.perf_counter()
        periods, _epochs, durations = self._candidate_arrays(candidates)
        cuda_indices = np.flatnonzero(periods > 8.0 * durations).tolist()
        cpu_indices = np.flatnonzero(periods <= 8.0 * durations).tolist()
        metrics_by_block = []
        for binding in self.blocks:
            rows: list[dict | None] = [None] * len(candidates)
            if cuda_indices:
                screened = self._screen_cuda_long(
                    binding, [candidates[index] for index in cuda_indices]
                )
                for index, row in zip(cuda_indices, screened):
                    rows[index] = row
            if cpu_indices:
                exact = measure_fixed_ephemerides_compiled(
                    binding.time,
                    binding.flux,
                    [candidates[index] for index in cpu_indices],
                )
                if exact is None:
                    raise RuntimeError("compiled short-period screen is unavailable")
                for index, row in zip(cpu_indices, exact):
                    rows[index] = {**row, "screen_only": False}
            if not all(row is not None for row in rows):
                raise RuntimeError("fixed-ephemeris screen block is incomplete")
            metrics_by_block.append([row for row in rows if row is not None])
        output: list[list[dict]] = [[] for _ in candidates]
        for block_index, (segments, metrics_rows) in enumerate(
            zip(blocks, metrics_by_block), start=1
        ):
            for candidate_index, metrics in enumerate(metrics_rows):
                output[candidate_index].append(
                    {
                        "block": block_index,
                        "segments": list(segments),
                        "independently_discovered": False,
                        "outcome": "screen_only",
                        **metrics,
                    }
                )
        elapsed = time.perf_counter() - started
        self.screen_calls += 1
        self.screen_candidates += len(candidates)
        self.screen_seconds += elapsed
        return output

    def _measure_block(
        self,
        binding: _BlockBinding,
        candidates: list[Mapping],
    ) -> list[dict]:
        if not candidates:
            return []
        periods, _epochs, durations = self._candidate_arrays(candidates)
        cuda_indices = [
            index
            for index, (period, duration) in enumerate(zip(periods, durations))
            if period > 8.0 * duration
        ]
        cuda_index_set = set(cuda_indices)
        cpu_indices = [
            index for index in range(len(candidates)) if index not in cuda_index_set
        ]
        output: list[dict | None] = [None] * len(candidates)
        if cuda_indices:
            cuda_rows = self._measure_cuda_long(
                binding, [candidates[index] for index in cuda_indices]
            )
            for index, row in zip(cuda_indices, cuda_rows):
                output[index] = row
        if cpu_indices:
            cpu_rows = measure_fixed_ephemerides_compiled(
                binding.time,
                binding.flux,
                [candidates[index] for index in cpu_indices],
            )
            if cpu_rows is None:
                raise RuntimeError("compiled fixed-ephemeris fallback is unavailable")
            for index, row in zip(cpu_indices, cpu_rows):
                output[index] = row
        self.cuda_candidates += len(cuda_indices)
        self.cpu_fallback_candidates += len(cpu_indices)
        assert all(row is not None for row in output)
        return [row for row in output if row is not None]

    def measure_prepared_blocks(
        self,
        candidates: list[Mapping],
        blocks,
        prepared_blocks,
    ) -> list[list[dict]]:
        """Return the same decorated fold rows as the CPU batch function."""
        from . import pipeline as reference

        if len(self.blocks) != len(prepared_blocks) or len(blocks) != len(self.blocks):
            raise ValueError("CUDA fixed-ephemeris block layout changed after binding")
        started = time.perf_counter()
        metrics_by_block = [
            self._measure_block(binding, candidates) for binding in self.blocks
        ]
        output: list[list[dict]] = [[] for _ in candidates]
        for block_index, (segments, metrics_rows) in enumerate(
            zip(blocks, metrics_by_block), start=1
        ):
            for candidate_index, metrics in enumerate(metrics_rows):
                candidate = candidates[candidate_index]
                discovered_blocks = {
                    int(value) for value in candidate.get("discovered_blocks", [])
                }
                if candidate.get("source_block") is not None:
                    discovered_blocks.add(int(candidate["source_block"]))
                decorated = dict(metrics)
                decorated["timing_offset_days"] = 0.0
                decorated["timing_tolerance_applied"] = False
                output[candidate_index].append(
                    {
                        "block": block_index,
                        "segments": list(segments),
                        "independently_discovered": block_index in discovered_blocks,
                        "outcome": reference._metric_outcome(
                            decorated,
                            independently_discovered=block_index in discovered_blocks,
                        ),
                        **decorated,
                    }
                )
        self.calls += 1
        self.candidates += len(candidates)
        self.seconds += time.perf_counter() - started
        return output

    def diagnostics(self) -> dict:
        return {
            "backend": "vlora-cupy-locked-ephemeris-v1",
            "calls": int(self.calls),
            "candidate_block_batches": int(self.candidates),
            "cuda_candidate_blocks": int(self.cuda_candidates),
            "cpu_overlapping_window_fallbacks": int(self.cpu_fallback_candidates),
            "predicted_event_windows": int(self.windows),
            "window_cadences_examined": int(self.window_points),
            "seconds": round(float(self.seconds), 6),
            "median_and_mad_on_cuda": True,
            "candidate_or_cadence_truncation": False,
            "one_pass_window_gather": bool(self.one_pass_gather),
            "sorted_by_window_span": bool(self.sort_by_window_size),
            "kernel_window_passes": int(self.kernel_window_passes),
            "allocated_value_cells": int(self.allocated_value_cells),
            "screen_calls": int(self.screen_calls),
            "screen_candidates": int(self.screen_candidates),
            "screen_seconds": round(float(self.screen_seconds), 6),
            "screen_predicted_event_windows": int(self.screen_windows),
            "screen_window_cadences_examined": int(self.screen_window_points),
        }


__all__ = ["CudaFixedEphemerisMeasurer"]
