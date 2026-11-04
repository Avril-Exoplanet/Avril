// Exact VLora BLS refit kernel.
//
// The scientific operation order for one trial period is intentionally kept
// stable: cadences are accumulated in input order and phase bins are scanned
// from left to right.  Parallelism is only across independent period tiles.
// On AArch64, four two-lane NEON vectors calculate the phase bins for eight
// periods while each period retains its own private histogram row.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <system_error>
#include <thread>
#include <vector>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define VLORA_HAS_NEON 1
#else
#define VLORA_HAS_NEON 0
#endif

namespace {

inline std::int64_t phase_bin_scalar(
    const double time_value,
    const double period,
    const double inverse_period,
    const double bin_duration
) noexcept {
    const double cycles = std::floor(time_value * inverse_period);
    double wrapped = time_value - period * cycles;
    if (wrapped >= period) {
        wrapped -= period;
    } else if (wrapped < 0.0) {
        wrapped += period;
    }
    return static_cast<std::int64_t>(wrapped / bin_duration) + 1;
}

#if VLORA_HAS_NEON
inline void phase_bins_neon_8(
    const double time_value,
    const double* periods,
    const double* inverse_periods,
    const double bin_duration,
    std::int64_t* bins
) noexcept {
    const float64x2_t time_vector = vdupq_n_f64(time_value);
    const float64x2_t zero = vdupq_n_f64(0.0);
    const float64x2_t bin_vector = vdupq_n_f64(bin_duration);
    const int64x2_t one = vdupq_n_s64(1);

    for (int lane_group = 0; lane_group < 4; ++lane_group) {
        const int offset = 2 * lane_group;
        const float64x2_t period = vld1q_f64(periods + offset);
        const float64x2_t inverse = vld1q_f64(inverse_periods + offset);
        const float64x2_t cycles = vrndmq_f64(vmulq_f64(time_vector, inverse));
        double wrapped_lanes[2];
        // Keep multiplication and subtraction separate.  The build also uses
        // -ffp-contract=off so this remains identical to the scalar oracle.
        vst1q_f64(
            wrapped_lanes,
            vsubq_f64(time_vector, vmulq_f64(period, cycles))
        );
        float64x2_t wrapped = vld1q_f64(wrapped_lanes);
        const uint64x2_t above = vcgeq_f64(wrapped, period);
        wrapped = vsubq_f64(wrapped, vbslq_f64(above, period, zero));
        const uint64x2_t below = vcltq_f64(wrapped, zero);
        wrapped = vaddq_f64(wrapped, vbslq_f64(below, period, zero));
        const int64x2_t indices = vaddq_s64(
            vcvtq_s64_f64(vdivq_f64(wrapped, bin_vector)),
            one
        );
        vst1q_s64(bins + offset, indices);
    }
}
#endif

struct ThreadWorkspace {
    std::vector<double> flux_histogram;
    std::vector<std::uint32_t> count_histogram;

    bool resize_and_clear(const std::size_t cells) {
        try {
            flux_histogram.resize(cells);
            count_histogram.resize(cells);
        } catch (const std::bad_alloc&) {
            return false;
        }
        std::fill(flux_histogram.begin(), flux_histogram.end(), 0.0);
        std::fill(count_histogram.begin(), count_histogram.end(), 0U);
        return true;
    }

};

inline void scan_best_power_exact(
    const double* prefix_flux,
    const std::uint32_t* prefix_counts,
    const std::int64_t maximum_phase,
    const std::int64_t duration_bins,
    const std::int64_t total_count,
    const double sum_y,
    const double sum_ivar,
    const double* depth_errors,
    double& best_power,
    std::int64_t& best_phase
) noexcept {
    best_power = -std::numeric_limits<double>::infinity();
    best_phase = -1;
    for (std::int64_t phase_index = 0; phase_index <= maximum_phase; ++phase_index) {
        const std::uint32_t count =
            prefix_counts[phase_index + duration_bins]
            - prefix_counts[phase_index];
        if (count == 0 || count >= static_cast<std::uint32_t>(total_count)) {
            continue;
        }
        const std::size_t in_count = static_cast<std::size_t>(count);
        const double y_in_sum =
            prefix_flux[phase_index + duration_bins]
            - prefix_flux[phase_index];
        const double y_out_sum = sum_y - y_in_sum;
        const double ivar_in = static_cast<double>(count);
        const double ivar_out = sum_ivar - ivar_in;
        const double y_in = y_in_sum / ivar_in;
        const double y_out = y_out_sum / ivar_out;
        const double depth = y_out - y_in;
        const double objective = depth / depth_errors[in_count];
        if (y_out >= y_in && objective > best_power) {
            best_power = objective;
            best_phase = phase_index;
        }
    }
}

int solve_period_runs(
    const std::int64_t period_index,
    const double* relative_time,
    const double* centered_flux,
    const std::int64_t cadence_count,
    const double* periods,
    const double* inverse_periods,
    const double duration,
    const int oversample,
    const double sum_y,
    const double sum_ivar,
    const double* depth_errors,
    double* powers,
    std::int64_t* best_phase_indices,
    ThreadWorkspace& workspace
) {
    const double period = periods[period_index];
    const double inverse_period = inverse_periods[period_index];
    const double bin_duration = duration / static_cast<double>(oversample);
    const std::int64_t n_bins =
        static_cast<std::int64_t>(std::ceil(period / bin_duration)) + oversample;
    const std::size_t cells = static_cast<std::size_t>(n_bins + 1);
    if (!workspace.resize_and_clear(cells)) {
        return -2;
    }
    double* mean_y = workspace.flux_histogram.data();
    std::uint32_t* counts = workspace.count_histogram.data();

    // TESS fast-cadence curves normally keep the same BLS phase bin for many
    // adjacent samples.  Compute the expensive phase only at run boundaries,
    // but retain every original floating-point flux addition in chronological
    // order.  Boundary rounding is corrected with the scalar oracle before a
    // run is committed, so this is an execution compression, not rebinning.
    std::int64_t cadence_index = 0;
    while (cadence_index < cadence_count) {
        const double time_value = relative_time[cadence_index];
        double cycles = std::floor(time_value * inverse_period);
        double wrapped = time_value - period * cycles;
        if (wrapped >= period) {
            wrapped -= period;
            cycles += 1.0;
        } else if (wrapped < 0.0) {
            wrapped += period;
            cycles -= 1.0;
        }
        const std::int64_t bin =
            static_cast<std::int64_t>(wrapped / bin_duration) + 1;
        const double phase_upper = std::min(
            period,
            static_cast<double>(bin) * bin_duration
        );
        const double boundary = period * cycles + phase_upper;
        double accumulated = mean_y[bin];
        std::int64_t stop = cadence_index;
        const double boundary_tolerance = 64.0
            * std::numeric_limits<double>::epsilon()
            * std::max(1.0, std::abs(boundary));
        while (stop < cadence_count) {
            const double next_time = relative_time[stop];
            if (
                next_time >= boundary - boundary_tolerance
                && phase_bin_scalar(
                    next_time,
                    period,
                    inverse_period,
                    bin_duration
                ) != bin
            ) {
                break;
            }
            accumulated += centered_flux[stop];
            ++stop;
        }
        mean_y[bin] = accumulated;
        counts[bin] += static_cast<std::uint32_t>(stop - cadence_index);
        cadence_index = stop;
    }

    const std::int64_t duration_bins = static_cast<std::int64_t>(
        std::floor(duration / bin_duration + 0.5)
    );
    std::int64_t wrapped_index = n_bins - oversample;
    for (std::int64_t phase_index = 1; phase_index <= oversample; ++phase_index) {
        mean_y[wrapped_index] = mean_y[phase_index];
        counts[wrapped_index] = counts[phase_index];
        ++wrapped_index;
    }
    for (std::int64_t phase_index = 1; phase_index <= n_bins; ++phase_index) {
        mean_y[phase_index] += mean_y[phase_index - 1];
        counts[phase_index] += counts[phase_index - 1];
    }

    const std::int64_t maximum_phase = n_bins - duration_bins;
    double best_power;
    std::int64_t best_phase;
    scan_best_power_exact(
        mean_y,
        counts,
        maximum_phase,
        duration_bins,
        cadence_count,
        sum_y,
        sum_ivar,
        depth_errors,
        best_power,
        best_phase
    );
    powers[period_index] = best_power;
    best_phase_indices[period_index] = best_phase;
    return 0;
}

int solve_tile(
    const std::int64_t tile_index,
    const double* relative_time,
    const double* centered_flux,
    const std::int64_t cadence_count,
    const double* periods,
    const double* inverse_periods,
    const std::int64_t period_count,
    const double duration,
    const int oversample,
    const int tile_size,
    const double sum_y,
    const double sum_ivar,
    const double* depth_errors,
    double* powers,
    std::int64_t* best_phase_indices,
    ThreadWorkspace& workspace
) {
    const std::int64_t start = tile_index * tile_size;
    const std::int64_t stop = std::min<std::int64_t>(period_count, start + tile_size);
    const std::int64_t width = stop - start;
    const double bin_duration = duration / static_cast<double>(oversample);
    double maximum_period = periods[start];
    for (std::int64_t period_index = start + 1; period_index < stop; ++period_index) {
        maximum_period = std::max(maximum_period, periods[period_index]);
    }
    const std::int64_t maximum_bins =
        static_cast<std::int64_t>(std::ceil(maximum_period / bin_duration)) + oversample;
    const std::size_t stride = static_cast<std::size_t>(maximum_bins + 1);
    const std::size_t cells = static_cast<std::size_t>(width) * stride;
    if (!workspace.resize_and_clear(cells)) {
        return -2;
    }
    double* flux_histogram = workspace.flux_histogram.data();
    std::uint32_t* count_histogram = workspace.count_histogram.data();

#if VLORA_HAS_NEON
    if (tile_size == 8 && width == 8) {
        std::int64_t bins[8];
        for (std::int64_t cadence_index = 0; cadence_index < cadence_count; ++cadence_index) {
            phase_bins_neon_8(
                relative_time[cadence_index],
                periods + start,
                inverse_periods + start,
                bin_duration,
                bins
            );
            const double flux = centered_flux[cadence_index];
#define VLORA_SCATTER_BIN(I)                                                   \
            do {                                                               \
                const std::size_t cell = static_cast<std::size_t>(I) * stride \
                    + static_cast<std::size_t>(bins[I]);                       \
                flux_histogram[cell] += flux;                                  \
                ++count_histogram[cell];                                       \
            } while (false)
            VLORA_SCATTER_BIN(0);
            VLORA_SCATTER_BIN(1);
            VLORA_SCATTER_BIN(2);
            VLORA_SCATTER_BIN(3);
            VLORA_SCATTER_BIN(4);
            VLORA_SCATTER_BIN(5);
            VLORA_SCATTER_BIN(6);
            VLORA_SCATTER_BIN(7);
#undef VLORA_SCATTER_BIN
        }
    } else
#endif
    {
        for (std::int64_t cadence_index = 0; cadence_index < cadence_count; ++cadence_index) {
            const double time_value = relative_time[cadence_index];
            const double flux = centered_flux[cadence_index];
            for (std::int64_t local_index = 0; local_index < width; ++local_index) {
                const std::int64_t output_index = start + local_index;
                const std::int64_t bin = phase_bin_scalar(
                    time_value,
                    periods[output_index],
                    inverse_periods[output_index],
                    bin_duration
                );
                const std::size_t cell =
                    static_cast<std::size_t>(local_index) * stride
                    + static_cast<std::size_t>(bin);
                flux_histogram[cell] += flux;
                ++count_histogram[cell];
            }
        }
    }

    const std::int64_t duration_bins = static_cast<std::int64_t>(
        std::floor(duration / bin_duration + 0.5)
    );
    for (std::int64_t local_index = 0; local_index < width; ++local_index) {
        const std::int64_t output_index = start + local_index;
        const double period = periods[output_index];
        const std::int64_t n_bins =
            static_cast<std::int64_t>(std::ceil(period / bin_duration)) + oversample;
        double* mean_y = flux_histogram
            + static_cast<std::size_t>(local_index) * stride;
        std::uint32_t* counts = count_histogram
            + static_cast<std::size_t>(local_index) * stride;

        std::int64_t wrapped_index = n_bins - oversample;
        for (std::int64_t phase_index = 1; phase_index <= oversample; ++phase_index) {
            mean_y[wrapped_index] = mean_y[phase_index];
            counts[wrapped_index] = counts[phase_index];
            ++wrapped_index;
        }
        for (std::int64_t phase_index = 1; phase_index <= n_bins; ++phase_index) {
            mean_y[phase_index] += mean_y[phase_index - 1];
            counts[phase_index] += counts[phase_index - 1];
        }

        const std::int64_t maximum_phase = n_bins - duration_bins;
        double best_power;
        std::int64_t best_phase;
        scan_best_power_exact(
            mean_y,
            counts,
            maximum_phase,
            duration_bins,
            cadence_count,
            sum_y,
            sum_ivar,
            depth_errors,
            best_power,
            best_phase
        );
        powers[output_index] = best_power;
        best_phase_indices[output_index] = best_phase;
    }
    return 0;
}

}  // namespace

extern "C" int vlora_exact_bls_kernel_v2(
    const double* relative_time,
    const double* centered_flux,
    const std::int64_t cadence_count,
    const double* periods,
    const double* inverse_periods,
    const std::int64_t period_count,
    const double duration,
    const int oversample,
    const int requested_threads,
    const int tile_size,
    const int strategy,
    const double sum_y,
    const double sum_ivar,
    double* powers,
    std::int64_t* best_phase_indices,
    int* effective_threads,
    int* used_neon
) {
    if (
        relative_time == nullptr || centered_flux == nullptr || periods == nullptr
        || inverse_periods == nullptr || powers == nullptr
        || best_phase_indices == nullptr || effective_threads == nullptr
        || used_neon == nullptr || cadence_count <= 0 || period_count <= 0
        || duration <= 0.0 || oversample <= 0 || requested_threads <= 0
        || tile_size <= 0 || strategy < 0 || strategy > 1
    ) {
        return 1;
    }
    const std::int64_t work_count = strategy == 0
        ? (period_count + tile_size - 1) / tile_size
        : period_count;
    const int thread_count = std::max<int>(
        1,
        std::min<std::int64_t>(requested_threads, work_count)
    );
    *effective_threads = thread_count;
    *used_neon = strategy == 0 && VLORA_HAS_NEON && tile_size == 8 ? 1 : 0;
    std::atomic<int> error_code{0};
    std::vector<double> depth_errors;
    try {
        depth_errors.resize(static_cast<std::size_t>(cadence_count + 1));
    } catch (const std::bad_alloc&) {
        return -2;
    }
    for (std::int64_t count = 1; count < cadence_count; ++count) {
        const double ivar_in = static_cast<double>(count);
        const double ivar_out = sum_ivar - ivar_in;
        depth_errors[static_cast<std::size_t>(count)] = std::sqrt(
            1.0 / ivar_in + 1.0 / ivar_out
        );
    }

    auto worker = [&](const int thread_index) {
        ThreadWorkspace workspace;
        const std::int64_t first =
            work_count * static_cast<std::int64_t>(thread_index) / thread_count;
        const std::int64_t last =
            work_count * static_cast<std::int64_t>(thread_index + 1) / thread_count;
        for (std::int64_t work_index = first; work_index < last; ++work_index) {
            if (error_code.load(std::memory_order_relaxed) != 0) {
                return;
            }
            const int status = strategy == 0
                ? solve_tile(
                    work_index,
                    relative_time,
                    centered_flux,
                    cadence_count,
                    periods,
                    inverse_periods,
                    period_count,
                    duration,
                    oversample,
                    tile_size,
                    sum_y,
                    sum_ivar,
                    depth_errors.data(),
                    powers,
                    best_phase_indices,
                    workspace
                )
                : solve_period_runs(
                    work_index,
                    relative_time,
                    centered_flux,
                    cadence_count,
                    periods,
                    inverse_periods,
                    duration,
                    oversample,
                    sum_y,
                    sum_ivar,
                    depth_errors.data(),
                    powers,
                    best_phase_indices,
                    workspace
                );
            if (status != 0) {
                error_code.store(status, std::memory_order_relaxed);
                return;
            }
        }
    };

    if (thread_count == 1) {
        worker(0);
    } else {
        std::vector<std::thread> threads;
        try {
            threads.reserve(static_cast<std::size_t>(thread_count - 1));
            for (int thread_index = 1; thread_index < thread_count; ++thread_index) {
                threads.emplace_back(worker, thread_index);
            }
            worker(0);
            for (auto& thread : threads) {
                thread.join();
            }
        } catch (const std::system_error&) {
            for (auto& thread : threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
            return -3;
        }
    }
    return error_code.load(std::memory_order_relaxed);
}
