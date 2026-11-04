"""CuPy runtime for VLora's complete deterministic CUDA BLS kernel.

The older CUDA experiment uses PyCUDA, which is awkward to deploy on Windows
because it requires a separate MSVC build toolchain.  This backend launches
the same CUDA source through CuPy/NVRTC and keeps the public solver contract
unchanged.  It never truncates the period or duration grids.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from .cuda_astropy_bls import (
    AstropyCudaBLSBinding,
    AstropyCudaBLSResult,
    _CUDA_SOURCE,
    _quantize_weighted_flux,
)


class CuPyAstropyBLS:
    """Reusable native-Windows CUDA engine for one VLora process."""

    threads = 256

    def __init__(self) -> None:
        started = time.perf_counter()
        requested_threads = int(os.environ.get("VLORA_CUDA_THREADS", "256"))
        if requested_threads not in {32, 64, 128, 256}:
            raise ValueError(
                "VLORA_CUDA_THREADS must be one of 32, 64, 128, or 256"
            )
        self.threads = requested_threads
        try:
            import cupy as cp
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("CuPy CUDA runtime is not installed") from exc

        self.cp = cp
        try:
            device = cp.cuda.Device()
            device.use()
            properties = cp.cuda.runtime.getDeviceProperties(device.id)
            module = cp.RawModule(
                code=_CUDA_SOURCE,
                options=(
                    "-std=c++11",
                    "--fmad=false",
                    "--prec-div=true",
                    "--prec-sqrt=true",
                ),
                name_expressions=(
                    "astropy_bls_shared",
                    "astropy_bls_global",
                    "astropy_bls_exact_phase_bins",
                    "astropy_bls_exact_from_bins",
                ),
            )
            self.function = module.get_function("astropy_bls_shared")
            self.global_function = module.get_function("astropy_bls_global")
            self.exact_phase_function = module.get_function(
                "astropy_bls_exact_phase_bins"
            )
            self.exact_accumulate_function = module.get_function(
                "astropy_bls_exact_from_bins"
            )
            attributes = device.attributes
            optin_shared = int(
                attributes.get(
                    "MaxSharedMemoryPerBlockOptin",
                    attributes.get("MaxSharedMemoryPerBlock", 48 * 1024),
                )
            )
            static_shared = int(
                self.function.attributes.get("shared_size_bytes", 0)
            )
            self.maximum_shared_bytes = max(0, optin_shared - static_shared)
            if self.maximum_shared_bytes:
                try:
                    self.function.max_dynamic_shared_size_bytes = (
                        self.maximum_shared_bytes
                    )
                except Exception:
                    # Some Windows driver/runtime pairs expose only the
                    # ordinary per-block limit.  The global-memory kernel is
                    # an exact-coverage fallback, so deployment remains safe.
                    self.maximum_shared_bytes = max(
                        0,
                        int(attributes.get("MaxSharedMemoryPerBlock", 48 * 1024))
                        - static_shared,
                    )
            raw_name = properties.get("name", b"CUDA device")
            self.device_name = (
                raw_name.decode(errors="replace")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
        except Exception as exc:
            raise RuntimeError(f"CuPy CUDA initialization failed: {exc}") from exc

        self.compile_seconds = time.perf_counter() - started
        self.calls = 0
        self.periods_evaluated = 0
        self.kernel_seconds = 0.0
        self.shared_calls = 0
        self.global_calls = 0
        self.bind_calls = 0
        self.exact_calls = 0
        self.exact_periods_evaluated = 0
        self.exact_phase_seconds = 0.0
        self.exact_accumulate_seconds = 0.0
        self.exact_batches = 0

    def bind(
        self,
        *,
        time_values: np.ndarray,
        flux_values: np.ndarray,
        scatter: float | None = None,
    ) -> AstropyCudaBLSBinding:
        """Upload one light curve once using the CPU exact-backend weights.

        VLora constructs ``BoxLeastSquares(time, flux)`` without ``dy`` for
        ephemeris refits, so Astropy uses a unit inverse variance.  ``scatter``
        is accepted only for API compatibility with the older experiment.
        """

        cp = self.cp
        time_array = np.asarray(time_values, dtype=np.float64)
        flux_array = np.asarray(flux_values, dtype=np.float64)
        if time_array.ndim != 1 or flux_array.shape != time_array.shape:
            raise ValueError("CUDA time and flux arrays must be aligned vectors")
        if time_array.size == 0:
            raise ValueError("CUDA light curve must not be empty")
        minimum_time = float(np.nanmin(time_array))
        relative_time = np.ascontiguousarray(time_array - minimum_time)
        centered_flux = np.ascontiguousarray(
            flux_array - float(np.nanmedian(flux_array))
        )
        # ``np.sum`` may use a pairwise reduction.  The reference BLS oracle
        # accumulates chronologically, so cumsum supplies the same scalar
        # addition order once per bound light curve.
        sum_y = float(
            np.cumsum(centered_flux, dtype=np.float64)[-1]
        )
        depth_errors = np.zeros(relative_time.size + 1, dtype=np.float64)
        counts = np.arange(1, relative_time.size, dtype=np.float64)
        depth_errors[1:-1] = np.sqrt(
            1.0 / counts + 1.0 / (float(relative_time.size) - counts)
        )
        (
            fixed_flux,
            fixed_flux_inverse_scale,
            fixed_sum_y,
            fixed_flux_scale_exponent,
        ) = _quantize_weighted_flux(centered_flux, 1.0)
        binding = AstropyCudaBLSBinding(
            minimum_time=minimum_time,
            cadence_count=int(relative_time.size),
            inverse_variance=1.0,
            sum_y=sum_y,
            sum_inverse_variance=float(relative_time.size),
            time_gpu=cp.asarray(relative_time),
            flux_gpu=cp.asarray(centered_flux),
            depth_errors_gpu=cp.asarray(depth_errors),
            fixed_flux_gpu=cp.asarray(fixed_flux),
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
        """Evaluate every supplied grid point in VRAM-bounded batches."""

        cp = self.cp
        period_array = np.ascontiguousarray(periods, dtype=np.float64)
        duration_array = np.ascontiguousarray(durations, dtype=np.float64)
        if period_array.ndim != 1 or period_array.size == 0:
            raise ValueError("periods must be a non-empty one-dimensional array")
        if duration_array.ndim != 1 or duration_array.size == 0:
            raise ValueError("durations must be a non-empty one-dimensional array")
        if objective not in {"likelihood", "snr"}:
            raise ValueError("objective must be 'likelihood' or 'snr'")
        oversample = int(oversample)
        if oversample < 1:
            raise ValueError("oversample must be at least one")

        bin_duration = float(np.min(duration_array)) / oversample
        maximum_bins = (
            int(math.ceil(float(np.max(period_array)) / bin_duration))
            + oversample
        )
        shared_capacity = maximum_bins + 1
        shared_bytes = shared_capacity * (
            np.dtype(np.float64).itemsize + np.dtype(np.uint32).itemsize
        )
        use_shared = shared_bytes <= self.maximum_shared_bytes

        durations_gpu = cp.asarray(duration_array)
        output_power = np.empty(period_array.size, dtype=np.float64)
        output_duration = np.empty(period_array.size, dtype=np.float64)
        output_epoch = np.empty(period_array.size, dtype=np.float64)
        started = time.perf_counter()
        start = 0
        while start < period_array.size:
            requested_stop = min(
                period_array.size,
                start + max(1, int(period_batch_size)),
            )
            if use_shared:
                stop = requested_stop
                local_capacity = shared_capacity
            else:
                free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
                memory_budget = max(
                    32 * 1024**2,
                    min(2 * 1024**3, int(0.40 * free_bytes)),
                )
                tentative_maximum = float(
                    np.max(period_array[start:requested_stop])
                )
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
                local_maximum = float(np.max(period_array[start:stop]))
                local_capacity = (
                    int(math.ceil(local_maximum / bin_duration))
                    + oversample
                    + 1
                )

            local_periods = np.ascontiguousarray(period_array[start:stop])
            periods_gpu = cp.asarray(local_periods)
            inverse_periods_gpu = cp.asarray(
                np.ascontiguousarray(1.0 / local_periods)
            )
            power_gpu = cp.empty(local_periods.size, dtype=cp.float64)
            duration_out_gpu = cp.empty(local_periods.size, dtype=cp.float64)
            phase_gpu = cp.empty(local_periods.size, dtype=cp.float64)
            common_arguments = (
                binding.time_gpu,
                binding.fixed_flux_gpu,
                np.int32(binding.cadence_count),
                np.float64(binding.inverse_variance),
                np.float64(binding.fixed_flux_inverse_scale),
                np.int64(binding.fixed_sum_y),
                np.float64(binding.sum_inverse_variance),
                periods_gpu,
                inverse_periods_gpu,
                np.int32(local_periods.size),
                durations_gpu,
                np.int32(duration_array.size),
                np.int32(oversample),
                np.int32(objective == "likelihood"),
                np.float64(bin_duration),
                np.float64(1.0 / bin_duration),
                np.int32(local_capacity),
            )
            if use_shared:
                self.function(
                    (int(local_periods.size),),
                    (self.threads,),
                    (*common_arguments, power_gpu, duration_out_gpu, phase_gpu),
                    shared_mem=int(shared_bytes),
                )
                self.shared_calls += 1
            else:
                mean_y_gpu = cp.empty(
                    (local_periods.size, local_capacity), dtype=cp.int64
                )
                count_gpu = cp.empty(
                    (local_periods.size, local_capacity), dtype=cp.uint32
                )
                self.global_function(
                    (int(local_periods.size),),
                    (self.threads,),
                    (
                        *common_arguments,
                        mean_y_gpu,
                        count_gpu,
                        power_gpu,
                        duration_out_gpu,
                        phase_gpu,
                    ),
                )
                self.global_calls += 1
            cp.cuda.Stream.null.synchronize()
            output_power[start:stop] = cp.asnumpy(power_gpu)
            output_duration[start:stop] = cp.asnumpy(duration_out_gpu)
            output_epoch[start:stop] = (
                binding.minimum_time + cp.asnumpy(phase_gpu)
            )
            start = stop

        elapsed = time.perf_counter() - started
        self.calls += 1
        self.periods_evaluated += int(period_array.size)
        self.kernel_seconds += elapsed
        return AstropyCudaBLSResult(
            power=output_power,
            epoch_btjd=output_epoch,
            duration_days=output_duration,
        )

    def solve_bound_exact(
        self,
        binding: AstropyCudaBLSBinding,
        *,
        periods: np.ndarray,
        durations: np.ndarray,
        objective: str = "snr",
        oversample: int = 10,
        period_batch_size: int = 2048,
    ) -> AstropyCudaBLSResult:
        """Evaluate selected periods with chronological FP64 accumulation.

        The phase-bin matrix is calculated in parallel.  One CUDA thread then
        owns each period and replays cadences in original order, making the
        selected-period correction deterministic without returning to CPU.
        """

        cp = self.cp
        period_array = np.ascontiguousarray(periods, dtype=np.float64)
        duration_array = np.ascontiguousarray(durations, dtype=np.float64)
        if period_array.ndim != 1 or period_array.size == 0:
            raise ValueError("periods must be a non-empty one-dimensional array")
        if duration_array.ndim != 1 or duration_array.size == 0:
            raise ValueError("exact CUDA correction requires durations")
        if objective not in {"likelihood", "snr"}:
            raise ValueError("objective must be 'likelihood' or 'snr'")
        if binding.depth_errors_gpu is None:
            raise RuntimeError("exact CUDA binding lacks depth-error weights")
        oversample = int(oversample)
        bin_duration = float(np.min(duration_array)) / float(oversample)
        output_power = np.empty(period_array.size, dtype=np.float64)
        output_duration = np.empty(period_array.size, dtype=np.float64)
        output_epoch = np.empty(period_array.size, dtype=np.float64)

        start = 0
        phase_seconds = 0.0
        accumulate_seconds = 0.0
        batches = 0
        while start < period_array.size:
            requested_stop = min(
                period_array.size,
                start + max(1, int(period_batch_size)),
            )
            free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
            memory_budget = max(
                32 * 1024**2,
                min(2 * 1024**3, int(0.40 * free_bytes)),
            )
            tentative_maximum = float(
                np.max(period_array[start:requested_stop])
            )
            tentative_capacity = (
                int(math.ceil(tentative_maximum / bin_duration))
                + oversample
                + 1
            )
            bytes_per_period = (
                binding.cadence_count * np.dtype(np.uint32).itemsize
                + tentative_capacity
                * (
                    np.dtype(np.float64).itemsize
                    + np.dtype(np.uint32).itemsize
                )
            )
            affordable = max(1, memory_budget // max(1, bytes_per_period))
            stop = min(requested_stop, start + int(affordable))
            requested_periods = np.ascontiguousarray(period_array[start:stop])
            # Sorted neighbouring periods tend to land in neighbouring phase
            # bins, making the cadence-major exact matrices coalesce well.
            # Restore caller order before returning.
            sort_order = np.argsort(requested_periods, kind="stable")
            restore_order = np.argsort(sort_order, kind="stable")
            local_periods = np.ascontiguousarray(
                requested_periods[sort_order]
            )
            local_inverse_periods = np.ascontiguousarray(
                1.0 / local_periods
            )
            local_capacity = (
                int(math.ceil(float(np.max(local_periods)) / bin_duration))
                + oversample
                + 1
            )
            local_count = int(local_periods.size)
            periods_gpu = cp.asarray(local_periods)
            inverse_periods_gpu = cp.asarray(local_inverse_periods)
            durations_gpu = cp.asarray(duration_array)
            phase_bins_gpu = cp.empty(
                (binding.cadence_count, local_count), dtype=cp.uint32
            )
            mean_y_gpu = cp.empty(
                (local_count, local_capacity), dtype=cp.float64
            )
            count_gpu = cp.empty(
                (local_count, local_capacity), dtype=cp.uint32
            )
            power_gpu = cp.empty(local_count, dtype=cp.float64)
            duration_gpu = cp.empty(local_count, dtype=cp.float64)
            phase_gpu = cp.empty(local_count, dtype=cp.float64)

            phase_started = time.perf_counter()
            phase_period_threads = 32
            phase_cadence_threads = 8
            period_blocks = (
                local_count + phase_period_threads - 1
            ) // phase_period_threads
            cadence_blocks = (
                binding.cadence_count + phase_cadence_threads - 1
            ) // phase_cadence_threads
            self.exact_phase_function(
                (int(period_blocks * cadence_blocks),),
                (phase_period_threads, phase_cadence_threads),
                (
                    binding.time_gpu,
                    np.int32(binding.cadence_count),
                    periods_gpu,
                    inverse_periods_gpu,
                    np.int32(local_count),
                    np.int32(period_blocks),
                    np.float64(bin_duration),
                    phase_bins_gpu,
                ),
            )
            cp.cuda.Stream.null.synchronize()
            phase_seconds += time.perf_counter() - phase_started

            accumulate_started = time.perf_counter()
            exact_threads = 128
            exact_blocks = (local_count + exact_threads - 1) // exact_threads
            self.exact_accumulate_function(
                (int(exact_blocks),),
                (exact_threads,),
                (
                    binding.flux_gpu,
                    np.int32(binding.cadence_count),
                    np.float64(binding.sum_y),
                    np.float64(binding.sum_inverse_variance),
                    periods_gpu,
                    np.int32(local_count),
                    durations_gpu,
                    np.int32(duration_array.size),
                    np.int32(oversample),
                    np.int32(objective == "likelihood"),
                    np.float64(bin_duration),
                    np.int32(local_capacity),
                    phase_bins_gpu,
                    binding.depth_errors_gpu,
                    mean_y_gpu,
                    count_gpu,
                    power_gpu,
                    duration_gpu,
                    phase_gpu,
                ),
            )
            cp.cuda.Stream.null.synchronize()
            accumulate_seconds += time.perf_counter() - accumulate_started
            output_power[start:stop] = cp.asnumpy(power_gpu)[restore_order]
            output_duration[start:stop] = cp.asnumpy(duration_gpu)[restore_order]
            output_epoch[start:stop] = (
                binding.minimum_time + cp.asnumpy(phase_gpu)[restore_order]
            )
            batches += 1
            start = stop

        self.exact_calls += 1
        self.exact_periods_evaluated += int(period_array.size)
        self.exact_phase_seconds += phase_seconds
        self.exact_accumulate_seconds += accumulate_seconds
        self.exact_batches += batches
        return AstropyCudaBLSResult(
            power=output_power,
            epoch_btjd=output_epoch,
            duration_days=output_duration,
        )

    def diagnostics(self) -> dict:
        return {
            "backend": "vlora-cupy-deterministic-complete-grid-cuda-bls-v4",
            "device": self.device_name,
            "threads_per_block": int(self.threads),
            "compile_seconds": round(self.compile_seconds, 6),
            "calls": int(self.calls),
            "periods_evaluated": int(self.periods_evaluated),
            "kernel_seconds": round(self.kernel_seconds, 6),
            "exact_correction_calls": int(self.exact_calls),
            "exact_correction_periods": int(self.exact_periods_evaluated),
            "exact_phase_seconds": round(self.exact_phase_seconds, 6),
            "exact_chronological_accumulation_seconds": round(
                self.exact_accumulate_seconds, 6
            ),
            "exact_correction_batches": int(self.exact_batches),
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
            "unit_inverse_variance_matches_cpu_refit": True,
            "astropy_grid_contract": True,
            "top_k_period_truncation": False,
            "all_periods_and_cadences_evaluated": True,
        }


_ENGINE: CuPyAstropyBLS | None = None


def get_cupy_astropy_bls() -> CuPyAstropyBLS:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = CuPyAstropyBLS()
    return _ENGINE


__all__ = ["CuPyAstropyBLS", "get_cupy_astropy_bls"]
