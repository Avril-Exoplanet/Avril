"""Deterministic CUDA implementation of the Astropy BLS grid used by VLora.

This is an opt-in experimental backend.  It follows Astropy's public
``bls.c`` algorithm: phase-bin weighted sums at ``min(duration)/oversample``,
wrapped cumulative bins, the same discrete duration grid, and the same
likelihood/SNR objectives.  Unlike cuvarbase, it therefore preserves the
fine-periodogram peak structure used by the existing VLora candidate ledger.

Phase arithmetic and the objective use float64.  Flux histograms use a
power-of-two-scaled int64 representation so parallel atomic accumulation is
deterministic and cannot exchange nearly tied period ranks merely because a
different GPU scheduled warps in another order.  Final candidate acceptance
is still measured by VLora's reference fixed-ephemeris checks.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np

from .cuda_hybrid_search import _find_nvcc


def _quantize_weighted_flux(
    centered_flux: np.ndarray,
    inverse_variance: float,
) -> tuple[np.ndarray, float, int, int]:
    """Build a deterministic int64 representation for CUDA histograms.

    Floating-point atomic addition depends on warp scheduling.  A nearly tied
    pair of periods can therefore exchange rank on different GPUs even though
    every cadence was evaluated.  Signed int64 addition is associative modulo
    overflow, so choose a power-of-two scale from the *sum of absolute input
    weights* and retain a factor-of-two safety margin.  The resulting atomic
    histogram is deterministic, while the quantisation step is below the
    round-off accumulated by a chronological float64 sum for real TESS light
    curves.
    """

    weighted = np.ascontiguousarray(
        np.asarray(centered_flux, dtype=np.float64) * float(inverse_variance)
    )
    absolute_sum = float(
        np.sum(np.abs(weighted), dtype=np.longdouble)
    )
    if not np.isfinite(absolute_sum):
        raise ValueError("CUDA light curve contains non-finite weighted flux")
    if absolute_sum == 0.0:
        return np.zeros(weighted.shape, dtype=np.int64), 1.0, 0, 0

    # Keep the worst-case sum of absolute quantised samples below 2**61.
    # Rounding every sample can add at most N/2, which is negligible relative
    # to the remaining two signed-int64 guard bits.
    scale_limit = float((1 << 61) - 1) / absolute_sum
    exponent = int(math.floor(math.log2(scale_limit)))
    exponent = max(-1022, min(1022, exponent))
    scaled = np.ldexp(weighted, exponent)
    if not np.all(np.isfinite(scaled)):
        raise ValueError("CUDA fixed-point flux scaling overflowed")
    fixed = np.ascontiguousarray(np.rint(scaled), dtype=np.int64)
    absolute_fixed_sum = int(np.sum(np.abs(fixed), dtype=np.int64))
    if absolute_fixed_sum >= (1 << 62):
        raise OverflowError("CUDA fixed-point histogram lacks int64 headroom")
    fixed_sum = int(np.sum(fixed, dtype=np.int64))
    inverse_scale = float(math.ldexp(1.0, -exponent))
    return fixed, inverse_scale, fixed_sum, exponent


_CUDA_SOURCE = r"""
// NVRTC on a minimal Windows CUDA installation does not have the host MSVC
// standard-library include path.  CUDA provides the math intrinsics used
// below directly; spell the two IEEE-754 constants locally so this kernel
// does not require a multi-gigabyte Visual Studio installation.
#define DBL_MAX 1.7976931348623157e+308
#define DBL_EPSILON 2.2204460492503131e-16

#define VLORA_THREADS 256

// TESS cadences are time ordered, so neighbouring lanes very often land in
// the same phase bin.  Collapse those lanes before touching the histogram.
// This preserves every cadence contribution while replacing several
// contended FP64 atomics with one warp-local sum and one atomic write.
__device__ __forceinline__ void vlora_warp_histogram_add_fixed(
    const int index,
    const long long weighted_flux,
    long long *flux_histogram,
    unsigned int *count_histogram
) {
    const unsigned int active = __activemask();
    const int lane = (int)(threadIdx.x & 31);
    const int previous_index = __shfl_up_sync(active, index, 1);
    const bool has_previous = lane > 0 && (active & (1U << (lane - 1)));
    const bool starts_segment = !has_previous || previous_index != index;
    const unsigned int segment_heads = __ballot_sync(active, starts_segment);
    const unsigned int lanes_through_me = lane == 31
        ? 0xffffffffU
        : ((1U << (lane + 1)) - 1U);
    const int segment_start = 31 - __clz(segment_heads & lanes_through_me);

    long long flux_sum = weighted_flux;
    unsigned int count_sum = 1U;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const long long previous_flux =
            __shfl_up_sync(active, flux_sum, offset);
        const unsigned int previous_count =
            __shfl_up_sync(active, count_sum, offset);
        const int previous_segment =
            __shfl_up_sync(active, segment_start, offset);
        const bool source_active = lane >= offset
            && (active & (1U << (lane - offset)));
        if (source_active && previous_segment == segment_start) {
            flux_sum += previous_flux;
            count_sum += previous_count;
        }
    }

    const int next_index = __shfl_down_sync(active, index, 1);
    const bool has_next = lane < 31 && (active & (1U << (lane + 1)));
    if (!has_next || next_index != index) {
        atomicAdd(
            (unsigned long long *)(flux_histogram + index),
            (unsigned long long)flux_sum
        );
        atomicAdd(count_histogram + index, count_sum);
    }
}

__device__ __forceinline__ int vlora_lower_bound_time(
    const double *time_values,
    const int cadence_count,
    const double target
) {
    int first = 0;
    int count = cadence_count;
    while (count > 0) {
        const int step = count >> 1;
        const int middle = first + step;
        if (time_values[middle] < target) {
            first = middle + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    return first;
}

// Process complete orbital cycles independently.  Inside one cycle phase is
// monotonic, so all consecutive cadences in the same Astropy bin are summed
// locally and committed once.  This removes the per-cadence modulo/floor and
// most histogram atomics without skipping or approximating any cadence.
__device__ __forceinline__ void vlora_accumulate_cycle_runs(
    const double *time_values,
    const double *flux_values,
    const int cadence_count,
    const double inverse_variance,
    const double period,
    const double inverse_period,
    const double bin_duration,
    double *flux_histogram,
    unsigned int *count_histogram
) {
    const int last_cycle =
        (int)floor(time_values[cadence_count - 1] * inverse_period);
    for (int cycle = (int)threadIdx.x;
         cycle <= last_cycle;
         cycle += (int)blockDim.x) {
        const double cycle_start = period * (double)cycle;
        const double cycle_stop = cycle_start + period;
        const int start = vlora_lower_bound_time(
            time_values, cadence_count, cycle_start
        );
        const int stop = vlora_lower_bound_time(
            time_values, cadence_count, cycle_stop
        );
        int current_bin = -1;
        double run_flux = 0.0;
        unsigned int run_count = 0U;
        for (int cadence = start; cadence < stop; ++cadence) {
            double wrapped = time_values[cadence] - cycle_start;
            if (wrapped >= period) wrapped -= period;
            else if (wrapped < 0.0) wrapped += period;
            const int index = (int)(wrapped / bin_duration) + 1;
            if (index != current_bin && run_count != 0U) {
                atomicAdd(flux_histogram + current_bin, run_flux);
                atomicAdd(count_histogram + current_bin, run_count);
                run_flux = 0.0;
                run_count = 0U;
            }
            current_bin = index;
            run_flux += flux_values[cadence] * inverse_variance;
            ++run_count;
        }
        if (run_count != 0U) {
            atomicAdd(flux_histogram + current_bin, run_flux);
            atomicAdd(count_histogram + current_bin, run_count);
        }
    }
}

extern "C" __global__ void astropy_bls_shared(
    const double *time_values,
    const long long *fixed_flux_values,
    const int cadence_count,
    const double inverse_variance,
    const double fixed_flux_inverse_scale,
    const long long fixed_sum_y,
    const double sum_inverse_variance,
    const double *periods,
    const double *inverse_periods,
    const int period_count,
    const double *durations,
    const int duration_count,
    const int oversample,
    const int objective_flag,
    const double bin_duration,
    const double inverse_bin_duration,
    const int shared_bin_capacity,
    double *out_objective,
    double *out_duration,
    double *out_phase
) {
    const int period_index = (int)blockIdx.x;
    const int thread_index = (int)threadIdx.x;
    if (period_index >= period_count) return;

    extern __shared__ long long dynamic_values[];
    long long *fixed_sum_y_bins = dynamic_values;
    unsigned int *counts =
        (unsigned int *)(fixed_sum_y_bins + shared_bin_capacity);
    __shared__ double reduction_objective[VLORA_THREADS];
    __shared__ int reduction_phase[VLORA_THREADS];
    __shared__ int reduction_duration[VLORA_THREADS];

    const double period = periods[period_index];
    const double inverse_period = inverse_periods[period_index];
    const int n_bins = (int)ceil(period / bin_duration) + oversample;
    for (int index = thread_index; index <= n_bins; index += blockDim.x) {
        fixed_sum_y_bins[index] = 0LL;
        counts[index] = 0U;
    }
    __syncthreads();

    for (int cadence = thread_index;
         cadence < cadence_count;
         cadence += blockDim.x) {
        const double time_value = time_values[cadence];
        const double cycles = floor(time_value * inverse_period);
        double wrapped = time_value - period * cycles;
        if (wrapped >= period) wrapped -= period;
        else if (wrapped < 0.0) wrapped += period;
        const int index = (int)(wrapped * inverse_bin_duration) + 1;
        vlora_warp_histogram_add_fixed(
            index,
            fixed_flux_values[cadence],
            fixed_sum_y_bins,
            counts
        );
    }
    __syncthreads();

    if (thread_index == 0) {
        int wrapped_index = n_bins - oversample;
        for (int index = 1; index <= oversample; ++index, ++wrapped_index) {
            fixed_sum_y_bins[wrapped_index] = fixed_sum_y_bins[index];
            counts[wrapped_index] = counts[index];
        }
        for (int index = 1; index <= n_bins; ++index) {
            fixed_sum_y_bins[index] += fixed_sum_y_bins[index - 1];
            counts[index] += counts[index - 1];
        }
    }
    __syncthreads();

    double local_best = -DBL_MAX;
    int local_phase = -1;
    int local_duration = -1;
    for (int duration_index = 0; duration_index < duration_count; ++duration_index) {
        const int duration_bins = (int)round(durations[duration_index] / bin_duration);
        const int maximum_phase = n_bins - duration_bins;
        for (int phase_index = thread_index;
             phase_index <= maximum_phase;
             phase_index += blockDim.x) {
            const long long fixed_weighted_in =
                fixed_sum_y_bins[phase_index + duration_bins]
                - fixed_sum_y_bins[phase_index];
            const double weighted_in =
                (double)fixed_weighted_in * fixed_flux_inverse_scale;
            const double ivar_in = inverse_variance * (double)(
                counts[phase_index + duration_bins] - counts[phase_index]
            );
            const double weighted_out =
                (double)(fixed_sum_y - fixed_weighted_in)
                * fixed_flux_inverse_scale;
            const double ivar_out = sum_inverse_variance - ivar_in;
            if (ivar_in < DBL_EPSILON || ivar_out < DBL_EPSILON) continue;
            const double y_in = weighted_in / ivar_in;
            const double y_out = weighted_out / ivar_out;
            if (y_out < y_in) continue;
            const double depth = y_out - y_in;
            double objective;
            if (objective_flag) {
                objective = 0.5 * ivar_in * depth * depth;
            } else {
                const double depth_error = sqrt(1.0 / ivar_in + 1.0 / ivar_out);
                objective = depth / depth_error;
            }
            if (objective > local_best
                || (objective == local_best
                    && (local_phase < 0 || phase_index < local_phase))) {
                local_best = objective;
                local_phase = phase_index;
                local_duration = duration_bins;
            }
        }
    }

    reduction_objective[thread_index] = local_best;
    reduction_phase[thread_index] = local_phase;
    reduction_duration[thread_index] = local_duration;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride
            && (reduction_objective[thread_index + stride]
                    > reduction_objective[thread_index]
                || (reduction_objective[thread_index + stride]
                        == reduction_objective[thread_index]
                    && reduction_phase[thread_index + stride]
                        < reduction_phase[thread_index]))) {
            reduction_objective[thread_index] =
                reduction_objective[thread_index + stride];
            reduction_phase[thread_index] = reduction_phase[thread_index + stride];
            reduction_duration[thread_index] =
                reduction_duration[thread_index + stride];
        }
        __syncthreads();
    }

    if (thread_index == 0) {
        const int best_duration_bins = reduction_duration[0];
        const double best_duration = best_duration_bins * bin_duration;
        out_objective[period_index] = reduction_objective[0];
        out_duration[period_index] = best_duration;
        // Match the CPU oracle's parenthesization exactly; the algebraically
        // equivalent split multiply/add differs by one float64 ulp for a few
        // long-period epochs.
        double phase = (
            (double)reduction_phase[0]
            + 0.5 * (double)best_duration_bins
        ) * bin_duration;
        phase = fmod(phase, period);
        if (phase < 0.0) phase += period;
        out_phase[period_index] = phase;
    }
}

extern "C" __global__ void astropy_bls_global(
    const double *time_values,
    const long long *fixed_flux_values,
    const int cadence_count,
    const double inverse_variance,
    const double fixed_flux_inverse_scale,
    const long long fixed_sum_y,
    const double sum_inverse_variance,
    const double *periods,
    const double *inverse_periods,
    const int period_count,
    const double *durations,
    const int duration_count,
    const int oversample,
    const int objective_flag,
    const double bin_duration,
    const double inverse_bin_duration,
    const int global_bin_capacity,
    long long *global_fixed_sum_y,
    unsigned int *global_counts,
    double *out_objective,
    double *out_duration,
    double *out_phase
) {
    const int period_index = (int)blockIdx.x;
    const int thread_index = (int)threadIdx.x;
    if (period_index >= period_count) return;

    long long *fixed_sum_y_bins =
        global_fixed_sum_y + period_index * global_bin_capacity;
    unsigned int *counts =
        global_counts + period_index * global_bin_capacity;
    __shared__ double reduction_objective[VLORA_THREADS];
    __shared__ int reduction_phase[VLORA_THREADS];
    __shared__ int reduction_duration[VLORA_THREADS];

    const double period = periods[period_index];
    const double inverse_period = inverse_periods[period_index];
    const int n_bins = (int)ceil(period / bin_duration) + oversample;
    for (int index = thread_index; index <= n_bins; index += blockDim.x) {
        fixed_sum_y_bins[index] = 0LL;
        counts[index] = 0U;
    }
    __syncthreads();

    for (int cadence = thread_index;
         cadence < cadence_count;
         cadence += blockDim.x) {
        const double time_value = time_values[cadence];
        const double cycles = floor(time_value * inverse_period);
        double wrapped = time_value - period * cycles;
        if (wrapped >= period) wrapped -= period;
        else if (wrapped < 0.0) wrapped += period;
        const int index = (int)(wrapped * inverse_bin_duration) + 1;
        vlora_warp_histogram_add_fixed(
            index,
            fixed_flux_values[cadence],
            fixed_sum_y_bins,
            counts
        );
    }
    __syncthreads();

    if (thread_index == 0) {
        int wrapped_index = n_bins - oversample;
        for (int index = 1; index <= oversample; ++index, ++wrapped_index) {
            fixed_sum_y_bins[wrapped_index] = fixed_sum_y_bins[index];
            counts[wrapped_index] = counts[index];
        }
        for (int index = 1; index <= n_bins; ++index) {
            fixed_sum_y_bins[index] += fixed_sum_y_bins[index - 1];
            counts[index] += counts[index - 1];
        }
    }
    __syncthreads();

    double local_best = -DBL_MAX;
    int local_phase = -1;
    int local_duration = -1;
    for (int duration_index = 0; duration_index < duration_count; ++duration_index) {
        const int duration_bins = (int)round(durations[duration_index] / bin_duration);
        const int maximum_phase = n_bins - duration_bins;
        for (int phase_index = thread_index;
             phase_index <= maximum_phase;
             phase_index += blockDim.x) {
            const long long fixed_weighted_in =
                fixed_sum_y_bins[phase_index + duration_bins]
                - fixed_sum_y_bins[phase_index];
            const double weighted_in =
                (double)fixed_weighted_in * fixed_flux_inverse_scale;
            const double ivar_in = inverse_variance * (double)(
                counts[phase_index + duration_bins] - counts[phase_index]
            );
            const double weighted_out =
                (double)(fixed_sum_y - fixed_weighted_in)
                * fixed_flux_inverse_scale;
            const double ivar_out = sum_inverse_variance - ivar_in;
            if (ivar_in < DBL_EPSILON || ivar_out < DBL_EPSILON) continue;
            const double y_in = weighted_in / ivar_in;
            const double y_out = weighted_out / ivar_out;
            if (y_out < y_in) continue;
            const double depth = y_out - y_in;
            double objective;
            if (objective_flag) {
                objective = 0.5 * ivar_in * depth * depth;
            } else {
                const double depth_error = sqrt(1.0 / ivar_in + 1.0 / ivar_out);
                objective = depth / depth_error;
            }
            if (objective > local_best
                || (objective == local_best
                    && (local_phase < 0 || phase_index < local_phase))) {
                local_best = objective;
                local_phase = phase_index;
                local_duration = duration_bins;
            }
        }
    }

    reduction_objective[thread_index] = local_best;
    reduction_phase[thread_index] = local_phase;
    reduction_duration[thread_index] = local_duration;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride
            && (reduction_objective[thread_index + stride]
                    > reduction_objective[thread_index]
                || (reduction_objective[thread_index + stride]
                        == reduction_objective[thread_index]
                    && reduction_phase[thread_index + stride]
                        < reduction_phase[thread_index]))) {
            reduction_objective[thread_index] =
                reduction_objective[thread_index + stride];
            reduction_phase[thread_index] = reduction_phase[thread_index + stride];
            reduction_duration[thread_index] = reduction_duration[thread_index + stride];
        }
        __syncthreads();
    }

    if (thread_index == 0) {
        const int best_duration_bins = reduction_duration[0];
        const double best_duration = best_duration_bins * bin_duration;
        out_objective[period_index] = reduction_objective[0];
        out_duration[period_index] = best_duration;
        double phase = (
            (double)reduction_phase[0]
            + 0.5 * (double)best_duration_bins
        ) * bin_duration;
        phase = fmod(phase, period);
        if (phase < 0.0) phase += period;
        out_phase[period_index] = phase;
    }
}

// Exact correction pass used after the complete high-throughput scan.
//
// The fast kernels above parallelize cadences within one period and therefore
// accumulate FP64 histogram cells through atomics.  That is scientifically
// stable, but the non-deterministic addition order can move two almost tied
// neighbouring period cells.  The two kernels below keep the expensive phase
// calculation massively parallel, then let exactly one CUDA thread accumulate
// each period's flux values in chronological cadence order.  This reproduces
// the CPU/Astropy histogram operation order while retaining parallelism across
// thousands of independent periods.
extern "C" __global__ void astropy_bls_exact_phase_bins(
    const double *time_values,
    const int cadence_count,
    const double *periods,
    const double *inverse_periods,
    const int period_count,
    const int period_block_count,
    const double bin_duration,
    unsigned int *phase_bins
) {
    const int period_block = (int)blockIdx.x % period_block_count;
    const int cadence_block = (int)blockIdx.x / period_block_count;
    const int period_index =
        period_block * (int)blockDim.x + (int)threadIdx.x;
    const int cadence_index =
        cadence_block * (int)blockDim.y + (int)threadIdx.y;
    if (period_index >= period_count || cadence_index >= cadence_count) return;
    const double period = periods[period_index];
    const double inverse_period = inverse_periods[period_index];
    const double time_value = time_values[cadence_index];
    const double cycles = floor(time_value * inverse_period);
    double wrapped = time_value - period * cycles;
    if (wrapped >= period) wrapped -= period;
    else if (wrapped < 0.0) wrapped += period;
    // Cadence-major layout makes a warp of neighbouring periods read and
    // write one contiguous cache line during both phases of the exact pass.
    phase_bins[
        (unsigned long long)cadence_index * (unsigned long long)period_count
        + (unsigned long long)period_index
    ] = (unsigned int)(wrapped / bin_duration) + 1U;
}

extern "C" __global__ void astropy_bls_exact_from_bins(
    const double *flux_values,
    const int cadence_count,
    const double sum_y,
    const double sum_inverse_variance,
    const double *periods,
    const int period_count,
    const double *durations,
    const int duration_count,
    const int oversample,
    const int objective_flag,
    const double bin_duration,
    const int global_bin_capacity,
    const unsigned int *phase_bins,
    const double *depth_errors,
    double *global_mean_y,
    unsigned int *global_counts,
    double *out_objective,
    double *out_duration,
    double *out_phase
) {
    const int period_index =
        (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (period_index >= period_count) return;

    const double period = periods[period_index];
    const int n_bins = (int)ceil(period / bin_duration) + oversample;
    const unsigned long long histogram_offset =
        (unsigned long long)period_index
        * (unsigned long long)global_bin_capacity;
    double *mean_y = global_mean_y + histogram_offset;
    unsigned int *counts = global_counts + histogram_offset;

    for (int index = 0; index <= n_bins; ++index) {
        mean_y[index] = 0.0;
        counts[index] = 0U;
    }
    // Deliberately chronological and scalar for this individual period.  The
    // periods themselves execute concurrently across the whole GPU.
    for (int cadence_index = 0; cadence_index < cadence_count; ++cadence_index) {
        const unsigned int index = phase_bins[
            (unsigned long long)cadence_index * (unsigned long long)period_count
            + (unsigned long long)period_index
        ];
        mean_y[index] += flux_values[cadence_index];
        counts[index] += 1U;
    }

    int wrapped_index = n_bins - oversample;
    for (int index = 1; index <= oversample; ++index, ++wrapped_index) {
        mean_y[wrapped_index] = mean_y[index];
        counts[wrapped_index] = counts[index];
    }
    for (int index = 1; index <= n_bins; ++index) {
        mean_y[index] += mean_y[index - 1];
        counts[index] += counts[index - 1];
    }

    double best_objective = -DBL_MAX;
    int best_phase = -1;
    int best_duration_bins = -1;
    for (int duration_index = 0;
         duration_index < duration_count;
         ++duration_index) {
        const int duration_bins =
            (int)floor(durations[duration_index] / bin_duration + 0.5);
        const int maximum_phase = n_bins - duration_bins;
        for (int phase_index = 0; phase_index <= maximum_phase; ++phase_index) {
            const unsigned int count =
                counts[phase_index + duration_bins] - counts[phase_index];
            if (count == 0U || count >= (unsigned int)cadence_count) continue;
            const double weighted_in =
                mean_y[phase_index + duration_bins] - mean_y[phase_index];
            const double weighted_out = sum_y - weighted_in;
            const double ivar_in = (double)count;
            const double ivar_out = sum_inverse_variance - ivar_in;
            const double y_in = weighted_in / ivar_in;
            const double y_out = weighted_out / ivar_out;
            if (y_out < y_in) continue;
            const double depth = y_out - y_in;
            const double objective = objective_flag
                ? 0.5 * ivar_in * depth * depth
                : depth / depth_errors[count];
            // Astropy retains the first phase/duration on an exact tie.
            if (objective > best_objective) {
                best_objective = objective;
                best_phase = phase_index;
                best_duration_bins = duration_bins;
            }
        }
    }

    const double fitted_duration = (double)best_duration_bins * bin_duration;
    out_objective[period_index] = best_objective;
    out_duration[period_index] = fitted_duration;
    double phase =
        ((double)best_phase + 0.5 * (double)best_duration_bins) * bin_duration;
    phase = fmod(phase, period);
    if (phase < 0.0) phase += period;
    out_phase[period_index] = phase;
}
"""


@dataclass
class AstropyCudaBLSResult:
    power: np.ndarray
    epoch_btjd: np.ndarray
    duration_days: np.ndarray


@dataclass
class AstropyCudaBLSBinding:
    """One light curve resident on the GPU across many period-grid calls."""

    minimum_time: float
    cadence_count: int
    inverse_variance: float
    sum_y: float
    sum_inverse_variance: float
    time_gpu: object
    flux_gpu: object
    depth_errors_gpu: object | None = None
    fixed_flux_gpu: object | None = None
    fixed_flux_inverse_scale: float | None = None
    fixed_sum_y: int | None = None
    fixed_flux_scale_exponent: int | None = None


class CudaAstropyBLS:
    """Reusable deterministic complete-grid CUDA engine for one process."""

    threads = 256

    def __init__(self) -> None:
        _find_nvcc()
        import pycuda.autoprimaryctx
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule

        self.cuda = cuda
        self.device = pycuda.autoprimaryctx.device
        started = time.perf_counter()
        requested_threads = int(os.environ.get("VLORA_CUDA_THREADS", "256"))
        if requested_threads not in {32, 64, 128, 256}:
            raise ValueError(
                "VLORA_CUDA_THREADS must be one of 32, 64, 128, or 256"
            )
        self.threads = requested_threads
        module = SourceModule(
            _CUDA_SOURCE,
            options=["-O3"],
            no_extern_c=True,
        )
        self.function = module.get_function("astropy_bls_shared")
        self.global_function = module.get_function("astropy_bls_global")
        optin_shared = int(
            self.device.get_attribute(
                cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
            )
        )
        static_shared = int(
            self.function.get_attribute(
                cuda.function_attribute.SHARED_SIZE_BYTES
            )
        )
        dynamic_shared = optin_shared - static_shared
        self.function.set_attribute(
            cuda.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES,
            dynamic_shared,
        )
        self.maximum_shared_bytes = dynamic_shared
        self.compile_seconds = time.perf_counter() - started
        self.calls = 0
        self.periods_evaluated = 0
        self.kernel_seconds = 0.0
        self.shared_calls = 0
        self.global_calls = 0
        self.bind_calls = 0

    def bind(
        self,
        *,
        time_values: np.ndarray,
        flux_values: np.ndarray,
        scatter: float,
    ) -> AstropyCudaBLSBinding:
        """Upload and center one light curve once for repeated grid solves."""
        import pycuda.gpuarray as gpuarray

        minimum_time = float(np.nanmin(time_values))
        relative_time = np.ascontiguousarray(
            np.asarray(time_values, dtype=np.float64) - minimum_time
        )
        centered_flux = np.ascontiguousarray(
            np.asarray(flux_values, dtype=np.float64)
            - float(np.nanmedian(flux_values))
        )
        inverse_variance = 1.0 / max(float(scatter), 1e-12) ** 2
        (
            fixed_flux,
            fixed_flux_inverse_scale,
            fixed_sum_y,
            fixed_flux_scale_exponent,
        ) = _quantize_weighted_flux(centered_flux, inverse_variance)
        binding = AstropyCudaBLSBinding(
            minimum_time=minimum_time,
            cadence_count=int(relative_time.size),
            inverse_variance=inverse_variance,
            sum_y=float(
                np.sum(centered_flux * inverse_variance, dtype=np.float64)
            ),
            sum_inverse_variance=float(relative_time.size * inverse_variance),
            time_gpu=gpuarray.to_gpu(relative_time),
            flux_gpu=gpuarray.to_gpu(centered_flux),
            fixed_flux_gpu=gpuarray.to_gpu(fixed_flux),
            fixed_flux_inverse_scale=fixed_flux_inverse_scale,
            fixed_sum_y=fixed_sum_y,
            fixed_flux_scale_exponent=fixed_flux_scale_exponent,
        )
        self.bind_calls += 1
        return binding

    def solve(
        self,
        *,
        time_values: np.ndarray,
        flux_values: np.ndarray,
        scatter: float,
        periods: np.ndarray,
        durations: np.ndarray,
        objective: str = "likelihood",
        oversample: int = 10,
        period_batch_size: int = 2048,
    ) -> AstropyCudaBLSResult:
        binding = self.bind(
            time_values=time_values,
            flux_values=flux_values,
            scatter=scatter,
        )
        return self.solve_bound(
            binding,
            periods=periods,
            durations=durations,
            objective=objective,
            oversample=oversample,
            period_batch_size=period_batch_size,
        )

    def solve_bound(
        self,
        binding: AstropyCudaBLSBinding,
        *,
        periods: np.ndarray,
        durations: np.ndarray,
        objective: str = "likelihood",
        oversample: int = 10,
        period_batch_size: int = 2048,
    ) -> AstropyCudaBLSResult:
        """Evaluate a complete grid using an already resident light curve."""
        import pycuda.gpuarray as gpuarray

        periods = np.ascontiguousarray(periods, dtype=np.float64)
        durations = np.ascontiguousarray(durations, dtype=np.float64)
        if periods.ndim != 1 or periods.size == 0:
            raise ValueError("periods must be a non-empty one-dimensional array")
        if durations.ndim != 1 or durations.size == 0:
            raise ValueError("durations must be a non-empty one-dimensional array")
        if objective not in {"likelihood", "snr"}:
            raise ValueError("objective must be 'likelihood' or 'snr'")
        oversample = int(oversample)
        if oversample < 1:
            raise ValueError("oversample must be at least one")
        bin_duration = float(np.min(durations)) / oversample
        maximum_bins = int(math.ceil(float(np.max(periods)) / bin_duration)) + oversample
        shared_capacity = maximum_bins + 1
        shared_bytes = shared_capacity * (
            np.dtype(np.float64).itemsize + np.dtype(np.uint32).itemsize
        )
        use_shared = shared_bytes <= self.maximum_shared_bytes

        duration_gpu = gpuarray.to_gpu(durations)
        output_power = np.empty(periods.size, dtype=np.float64)
        output_duration = np.empty(periods.size, dtype=np.float64)
        output_epoch = np.empty(periods.size, dtype=np.float64)
        started = time.perf_counter()
        start = 0
        while start < periods.size:
            requested_stop = min(
                periods.size, start + max(1, int(period_batch_size))
            )
            if use_shared:
                stop = requested_stop
                local_capacity = shared_capacity
            else:
                # The exact same bins live in global memory when a long-period
                # grid exceeds the device's per-block shared-memory limit.
                # Keep the two work arrays below a conservative fraction of
                # currently free VRAM and adapt the number of periods per
                # launch; no period or duration cell is omitted.
                free_bytes, _total_bytes = self.cuda.mem_get_info()
                memory_budget = max(
                    64 * 1024**2,
                    min(2 * 1024**3, int(0.40 * free_bytes)),
                )
                tentative_maximum = float(np.max(periods[start:requested_stop]))
                local_capacity = (
                    int(math.ceil(tentative_maximum / bin_duration))
                    + oversample
                    + 1
                )
                bytes_per_period = local_capacity * (
                    np.dtype(np.float64).itemsize
                    + np.dtype(np.uint32).itemsize
                )
                affordable = max(1, memory_budget // bytes_per_period)
                stop = min(requested_stop, start + int(affordable))
                local_maximum = float(np.max(periods[start:stop]))
                local_capacity = (
                    int(math.ceil(local_maximum / bin_duration))
                    + oversample
                    + 1
                )
            local_periods = np.ascontiguousarray(periods[start:stop])
            period_gpu = gpuarray.to_gpu(local_periods)
            inverse_period_gpu = gpuarray.to_gpu(
                np.ascontiguousarray(1.0 / local_periods)
            )
            power_gpu = gpuarray.empty(local_periods.size, dtype=np.float64)
            duration_out_gpu = gpuarray.empty(local_periods.size, dtype=np.float64)
            phase_gpu = gpuarray.empty(local_periods.size, dtype=np.float64)
            common_arguments = (
                binding.time_gpu,
                binding.fixed_flux_gpu,
                np.int32(binding.cadence_count),
                np.float64(binding.inverse_variance),
                np.float64(binding.fixed_flux_inverse_scale),
                np.int64(binding.fixed_sum_y),
                np.float64(binding.sum_inverse_variance),
                period_gpu,
                inverse_period_gpu,
                np.int32(local_periods.size),
                duration_gpu,
                np.int32(durations.size),
                np.int32(oversample),
                np.int32(objective == "likelihood"),
                np.float64(bin_duration),
                np.float64(1.0 / bin_duration),
                np.int32(local_capacity),
            )
            if use_shared:
                self.function(
                    *common_arguments,
                    power_gpu,
                    duration_out_gpu,
                    phase_gpu,
                    block=(self.threads, 1, 1),
                    grid=(int(local_periods.size), 1, 1),
                    shared=shared_bytes,
                )
                self.shared_calls += 1
            else:
                mean_y_gpu = gpuarray.empty(
                    (local_periods.size, local_capacity), dtype=np.int64
                )
                count_gpu = gpuarray.empty(
                    (local_periods.size, local_capacity), dtype=np.uint32
                )
                self.global_function(
                    *common_arguments,
                    mean_y_gpu,
                    count_gpu,
                    power_gpu,
                    duration_out_gpu,
                    phase_gpu,
                    block=(self.threads, 1, 1),
                    grid=(int(local_periods.size), 1, 1),
                )
                self.global_calls += 1
            self.cuda.Context.synchronize()
            output_power[start:stop] = power_gpu.get()
            output_duration[start:stop] = duration_out_gpu.get()
            output_epoch[start:stop] = binding.minimum_time + phase_gpu.get()
            start = stop
        elapsed = time.perf_counter() - started
        self.calls += 1
        self.periods_evaluated += int(periods.size)
        self.kernel_seconds += elapsed
        return AstropyCudaBLSResult(
            power=output_power,
            epoch_btjd=output_epoch,
            duration_days=output_duration,
        )

    def diagnostics(self) -> dict:
        return {
            "backend": "vlora-deterministic-complete-grid-cuda-bls-v4",
            "device": self.device.name(),
            "compile_seconds": round(self.compile_seconds, 6),
            "calls": int(self.calls),
            "periods_evaluated": int(self.periods_evaluated),
            "kernel_seconds": round(self.kernel_seconds, 6),
            "lightcurve_bind_calls": int(self.bind_calls),
            "shared_memory_launches": int(self.shared_calls),
            "global_memory_launches": int(self.global_calls),
            "long_period_global_memory_fallback": True,
            "maximum_global_histogram_batch_bytes": 2 * 1024**3,
            "float64_phase_and_objective": True,
            "deterministic_int64_flux_histogram": True,
            "power_of_two_fixed_point_scale": True,
            "overflow_guard_bits": 2,
            "integer_weight_histogram": True,
            "warp_aggregated_histogram": True,
            "cycle_run_compression": False,
            "parallel_direct_long_period_windows": False,
            "astropy_grid_contract": True,
            "top_k_period_truncation": False,
            "all_periods_and_cadences_evaluated": True,
        }


_ENGINE: CudaAstropyBLS | None = None


def get_cuda_astropy_bls() -> CudaAstropyBLS:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = CudaAstropyBLS()
    return _ENGINE


__all__ = [
    "AstropyCudaBLSBinding",
    "AstropyCudaBLSResult",
    "CudaAstropyBLS",
    "get_cuda_astropy_bls",
]
