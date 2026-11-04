"""Opt-in CUDA BLS backends for VLora discovery and ephemeris refinement.

The original hybrid entry point remains available and unchanged: CUDA creates
only the coarse spectrum and Astropy performs the fine search.  The new full
entry point also obtains the best phase and duration for every fine-grid
period on the GPU, while keeping the existing CPU fixed-ephemeris acceptance
metrics.  Both paths evaluate complete configured grids and never apply a
Top-K period truncation.
"""

from __future__ import annotations

import os
import shutil
import time as timing
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.timeseries import BoxLeastSquares

from multiplanet_trial.v07_eval import multibranch_search as branch_search

from . import optimization_v3
from . import pipeline as core


CUDA_DLOGQ_ENV = "VLORA_CUDA_DLOGQ"
CUDA_NOVERLAP_ENV = "VLORA_CUDA_NOVERLAP"
CUDA_MAX_NBLOCKS_ENV = "VLORA_CUDA_MAX_NBLOCKS"
CUDA_EXACT_PEAK_TOLERANCE_ENV = "VLORA_CUDA_EXACT_PEAK_TOLERANCE"
CUDA_EXACT_GATE_GUARD_SIGMA_ENV = "VLORA_CUDA_EXACT_GATE_GUARD_SIGMA"
_ENGINE = None
_ASTROPY_SEARCH_ADAPTER = None


def _find_nvcc() -> None:
    if shutil.which("nvcc") is not None:
        return
    for candidate in (
        Path("/usr/local/cuda/bin"),
        Path("/usr/local/cuda-12.8/bin"),
    ):
        if (candidate / "nvcc").is_file():
            os.environ["PATH"] = f"{candidate}:{os.environ.get('PATH', '')}"
            return
    raise RuntimeError("CUDA nvcc was not found on PATH or in /usr/local/cuda*")


class CudaCoarseBLS:
    """One compiled cuvarbase kernel reused for every temporal block."""

    def __init__(self) -> None:
        _find_nvcc()
        import pycuda.autoprimaryctx
        import pycuda.driver as cuda
        from cuvarbase.bls import _function_signatures, compile_bls

        self.cuda = cuda
        self.device = pycuda.autoprimaryctx.device
        self.dlogq = float(os.environ.get(CUDA_DLOGQ_ENV, "0.2"))
        self.noverlap = max(1, int(os.environ.get(CUDA_NOVERLAP_ENV, "3")))
        self.max_nblocks = max(
            1, int(os.environ.get(CUDA_MAX_NBLOCKS_ENV, "5000"))
        )
        started = timing.perf_counter()
        functions = compile_bls(
            function_names=["full_bls_no_sol"], prepare=False
        )
        optin_shared = int(
            self.device.get_attribute(
                cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
            )
        )
        raw_function = functions["full_bls_no_sol"]
        static_shared = int(
            raw_function.get_attribute(
                cuda.function_attribute.SHARED_SIZE_BYTES
            )
        )
        self.dynamic_shared = optin_shared - static_shared
        raw_function.set_attribute(
            cuda.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES,
            self.dynamic_shared,
        )
        functions["full_bls_no_sol"] = raw_function.prepare(
            _function_signatures["full_bls_no_sol"]
        )
        self.functions = functions
        self.compile_seconds = timing.perf_counter() - started
        self.solution_functions = None
        self.solution_compile_seconds = 0.0
        self.calls = 0
        self.kernel_seconds = 0.0
        self.solution_calls = 0
        self.solution_kernel_seconds = 0.0

    @staticmethod
    def _prepared_arrays(
        *,
        time: np.ndarray,
        flux: np.ndarray,
        scatter: float,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        minimum_time = float(np.nanmin(time))
        relative_time = np.asarray(
            np.asarray(time, dtype=np.float64) - minimum_time,
            dtype=np.float32,
        )
        # Center in float64 before conversion.  Normalized flux near one loses
        # shallow transits if the subtraction is delayed until float32.
        flux64 = np.asarray(flux, dtype=np.float64)
        flux64 = flux64 / float(np.nanmedian(flux64)) - 1.0
        flux32 = np.asarray(
            flux64 - float(np.nanmedian(flux64)), dtype=np.float32
        )
        uncertainty = np.full(
            relative_time.size, max(float(scatter), 1e-8), dtype=np.float32
        )
        return minimum_time, relative_time, flux32, uncertainty

    def _ensure_solution_functions(self) -> None:
        if self.solution_functions is not None:
            return
        from cuvarbase.bls import compile_bls

        started = timing.perf_counter()
        self.solution_functions = compile_bls()
        self.solution_compile_seconds = timing.perf_counter() - started

    def power(
        self,
        *,
        time: np.ndarray,
        flux: np.ndarray,
        scatter: float,
        periods: np.ndarray,
        durations: np.ndarray,
    ) -> np.ndarray:
        from cuvarbase.bls import eebls_gpu_fast

        periods = np.asarray(periods, dtype=np.float64)
        durations = np.asarray(durations, dtype=np.float64)
        _minimum_time, relative_time, flux32, uncertainty = (
            self._prepared_arrays(time=time, flux=flux, scatter=scatter)
        )
        frequencies = np.asarray(1.0 / periods, dtype=np.float32)
        qmin = np.asarray(float(np.min(durations)) / periods, dtype=np.float32)
        qmax = np.asarray(
            np.minimum(float(np.max(durations)) / periods, 0.49),
            dtype=np.float32,
        )
        started = timing.perf_counter()
        power = eebls_gpu_fast(
            relative_time,
            flux32,
            uncertainty,
            frequencies,
            qmin=qmin,
            qmax=qmax,
            ignore_negative_delta_sols=True,
            functions=self.functions,
            dlogq=self.dlogq,
            noverlap=self.noverlap,
            max_nblocks=self.max_nblocks,
            shmem_lim=self.dynamic_shared,
        )
        self.cuda.Context.synchronize()
        self.calls += 1
        self.kernel_seconds += timing.perf_counter() - started
        return np.asarray(power, dtype=np.float64)

    def solution(
        self,
        *,
        time: np.ndarray,
        flux: np.ndarray,
        scatter: float,
        periods: np.ndarray,
        durations: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return power, center epoch, and duration for every input period.

        ``cuvarbase`` reports ``phi`` at the start of the best box.  VLora's
        locked ephemeris convention is the box centre, hence ``phi + q/2``.
        This conversion is covered by a real L98-59 GPU/CPU regression.
        """

        from cuvarbase.bls import eebls_gpu

        self._ensure_solution_functions()
        periods = np.asarray(periods, dtype=np.float64)
        durations = np.asarray(durations, dtype=np.float64)
        if periods.ndim != 1 or periods.size == 0:
            raise ValueError("periods must be a non-empty one-dimensional array")
        if durations.ndim != 1 or durations.size == 0:
            raise ValueError("durations must be a non-empty one-dimensional array")
        minimum_time, relative_time, flux32, uncertainty = self._prepared_arrays(
            time=time,
            flux=flux,
            scatter=scatter,
        )
        frequencies = np.asarray(1.0 / periods, dtype=np.float32)
        qmin = np.asarray(float(np.min(durations)) / periods, dtype=np.float32)
        qmax = np.asarray(
            np.minimum(float(np.max(durations)) / periods, 0.49),
            dtype=np.float32,
        )
        started = timing.perf_counter()
        power, raw_solutions = eebls_gpu(
            relative_time,
            flux32,
            uncertainty,
            frequencies,
            qmin=qmin,
            qmax=qmax,
            ignore_negative_delta_sols=True,
            functions=self.solution_functions,
            dlogq=self.dlogq,
            noverlap=self.noverlap,
        )
        self.cuda.Context.synchronize()
        q_values = np.asarray(
            [float(row[0]) for row in raw_solutions], dtype=np.float64
        )
        phi_values = np.asarray(
            [float(row[1]) for row in raw_solutions], dtype=np.float64
        )
        epochs = minimum_time + np.mod(phi_values + 0.5 * q_values, 1.0) * periods
        self.solution_calls += 1
        self.solution_kernel_seconds += timing.perf_counter() - started
        return {
            "power": np.asarray(power, dtype=np.float64),
            "epoch_btjd": np.asarray(epochs, dtype=np.float64),
            "duration_days": np.asarray(q_values * periods, dtype=np.float64),
        }

    def diagnostics(self) -> dict:
        return {
            "backend": "cuvarbase-eebls-gpu-fast-proposal-only",
            "device": self.device.name(),
            "compile_seconds": round(self.compile_seconds, 6),
            "kernel_calls": int(self.calls),
            "kernel_seconds": round(self.kernel_seconds, 6),
            "solution_compile_seconds": round(
                self.solution_compile_seconds, 6
            ),
            "solution_calls": int(self.solution_calls),
            "solution_kernel_seconds": round(
                self.solution_kernel_seconds, 6
            ),
            "dlogq": self.dlogq,
            "noverlap": self.noverlap,
            "max_nblocks": self.max_nblocks,
            "float32_flux_centered_in_float64_first": True,
            "locked_ephemeris_from_cuda": False,
            "cpu_float64_refinement_required": True,
        }


def get_cuda_engine() -> CudaCoarseBLS:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = CudaCoarseBLS()
    return _ENGINE


class CudaAstropySearchAdapter:
    """Search-provider facade for the FP64 Astropy-equivalent CUDA kernel."""

    def __init__(self) -> None:
        from .cuda_astropy_bls import get_cuda_astropy_bls

        self.engine = get_cuda_astropy_bls()
        self._cache_key = None
        self._cache_period_bits: dict[int, int] = {}
        self._cache_result = None
        self.cache_hits = 0
        self.correction_calls = 0
        self.correction_periods = 0
        self.correction_seconds = 0.0

    @staticmethod
    def _possible_peak_neighbourhood(
        power: np.ndarray,
        *,
        absolute_error_bound: float,
    ) -> np.ndarray:
        """Return every cell that can affect a CPU local maximum.

        A CPU maximum that differs from the complete CUDA scan only within the
        measured FP64 error bound must be a tolerance-aware CUDA maximum.  Its
        two neighbours determine the strict predicate, so correct all three
        cells with Astropy before candidate extraction.
        """

        power = np.asarray(power, dtype=np.float64)
        if power.ndim != 1:
            raise ValueError("fine-grid power must be one-dimensional")
        if power.size == 0:
            return np.empty(0, dtype=np.intp)
        tolerance = max(0.0, 2.0 * float(absolute_error_bound))
        finite = np.isfinite(power)
        possible = np.zeros(power.size, dtype=np.bool_)
        if power.size == 1:
            possible[0] = finite[0]
        else:
            possible[0] = finite[0] and (
                not finite[1] or power[0] + tolerance >= power[1]
            )
            possible[-1] = finite[-1] and (
                not finite[-2] or power[-1] + tolerance >= power[-2]
            )
            possible[1:-1] = (
                finite[1:-1]
                & (~finite[:-2] | (power[1:-1] + tolerance >= power[:-2]))
                & (~finite[2:] | (power[1:-1] + tolerance >= power[2:]))
            )
        peaks = np.flatnonzero(possible)
        neighbourhood = np.unique(
            np.concatenate((peaks - 1, peaks, peaks + 1))
        )
        return neighbourhood[
            (neighbourhood >= 0) & (neighbourhood < power.size)
        ].astype(np.intp, copy=False)

    @staticmethod
    def _cpu_solution(*, time, flux, scatter, periods, durations) -> dict:
        model = BoxLeastSquares(
            np.asarray(time, dtype=np.float64) * u.day,
            np.asarray(flux, dtype=np.float64),
            dy=np.full(len(time), float(scatter), dtype=np.float64),
        )
        solved = model.power(
            np.asarray(periods, dtype=np.float64) * u.day,
            np.asarray(durations, dtype=np.float64) * u.day,
            method="fast",
        )
        return {
            "power": np.asarray(solved.power, dtype=np.float64),
            "epoch_btjd": np.asarray(
                solved.transit_time.to_value(u.day), dtype=np.float64
            ),
            "duration_days": np.asarray(
                solved.duration.to_value(u.day), dtype=np.float64
            ),
        }

    @staticmethod
    def _data_key(time: np.ndarray, flux: np.ndarray, durations: np.ndarray) -> tuple:
        time_array = np.asarray(time)
        flux_array = np.asarray(flux)
        return (
            int(time_array.__array_interface__["data"][0]),
            int(flux_array.__array_interface__["data"][0]),
            int(time_array.size),
            np.asarray(durations, dtype=np.float64).tobytes(),
        )

    def power(self, *, time, flux, scatter, periods, durations) -> np.ndarray:
        periods = np.asarray(periods, dtype=np.float64)
        durations = np.asarray(durations, dtype=np.float64)
        result = self.engine.solve(
            time_values=time,
            flux_values=flux,
            scatter=float(scatter),
            periods=periods,
            durations=durations,
            objective="likelihood",
        )
        self._cache_key = self._data_key(time, flux, durations)
        self._cache_period_bits = {
            int(bits): index
            for index, bits in enumerate(periods.view(np.uint64))
        }
        self._cache_result = result
        return np.asarray(result.power, dtype=np.float64)

    def correct_power(
        self,
        *,
        time,
        flux,
        scatter,
        periods,
        durations,
        power,
        baseline,
        power_scatter,
        sde_threshold,
    ) -> np.ndarray:
        """Correct only cells that can pass the unchanged scientific gate.

        The CUDA scan still evaluates every fine-grid period.  The downstream
        pipeline cannot retain a peak below ``baseline + threshold * MAD``.
        Re-evaluate a deliberately wider guard band, plus both neighbours, in
        Astropy FP64.  This avoids replaying the large population of low-power
        local extrema that is mathematically unable to become a candidate.
        """

        periods = np.asarray(periods, dtype=np.float64)
        durations = np.asarray(durations, dtype=np.float64)
        corrected_power = np.asarray(power, dtype=np.float64).copy()
        guard_sigma = max(
            0.0,
            float(os.environ.get(CUDA_EXACT_GATE_GUARD_SIGMA_ENV, "2.0")),
        )
        correction_floor = float(baseline) + max(
            0.0, float(sde_threshold) - guard_sigma
        ) * max(0.0, float(power_scatter))
        possible = np.flatnonzero(
            np.isfinite(corrected_power) & (corrected_power >= correction_floor)
        )
        if possible.size:
            correction_indices = np.unique(
                np.concatenate((possible - 1, possible, possible + 1))
            )
            correction_indices = correction_indices[
                (correction_indices >= 0)
                & (correction_indices < corrected_power.size)
            ].astype(np.intp, copy=False)
        else:
            correction_indices = np.empty(0, dtype=np.intp)
        correction_started = timing.perf_counter()
        if correction_indices.size:
            corrected = self._cpu_solution(
                time=time,
                flux=flux,
                scatter=scatter,
                periods=periods[correction_indices],
                durations=durations,
            )
            corrected_power[correction_indices] = corrected["power"]
        self.correction_calls += 1
        self.correction_periods += int(correction_indices.size)
        self.correction_seconds += timing.perf_counter() - correction_started
        return corrected_power

    def solution(self, *, time, flux, scatter, periods, durations) -> dict:
        periods = np.asarray(periods, dtype=np.float64)
        durations = np.asarray(durations, dtype=np.float64)
        # The candidate count is decided only after exact CPU correction of
        # possible maxima.  Recompute those selected periods with Astropy too,
        # so period, epoch, duration, and power have the same definition as the
        # optimized Mac reference rather than merely a close CUDA value.
        self.cache_hits += int(self._data_key(time, flux, durations) == self._cache_key)
        return self._cpu_solution(
            time=time,
            flux=flux,
            scatter=scatter,
            periods=periods,
            durations=durations,
        )

    def diagnostics(self) -> dict:
        return {
            **self.engine.diagnostics(),
            "solution_cache_hits": int(self.cache_hits),
            "possible_peak_correction_calls": int(self.correction_calls),
            "possible_peak_correction_periods": int(self.correction_periods),
            "possible_peak_correction_seconds": round(
                self.correction_seconds, 6
            ),
            "locked_ephemeris_from_cuda": False,
            "locked_ephemeris_from_cpu_astropy": True,
            "cpu_float64_fixed_ephemeris_acceptance_retained": True,
        }


class CuPyExactAstropySearchAdapter:
    """Exact chronological FP64 fine-grid solver with no CPU replay."""

    def __init__(self) -> None:
        from .cupy_astropy_bls import get_cupy_astropy_bls

        self.engine = get_cupy_astropy_bls()
        self._cache_key = None
        self._cache: dict[int, tuple[float, float, float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _data_key(time: np.ndarray, flux: np.ndarray, durations: np.ndarray) -> tuple:
        time_array = np.asarray(time)
        flux_array = np.asarray(flux)
        return (
            int(time_array.__array_interface__["data"][0]),
            int(flux_array.__array_interface__["data"][0]),
            int(time_array.size),
            np.asarray(durations, dtype=np.float64).tobytes(),
        )

    def _solve(self, *, time, flux, scatter, periods, durations) -> dict:
        periods = np.asarray(periods, dtype=np.float64)
        durations = np.asarray(durations, dtype=np.float64)
        binding = self.engine.bind(
            time_values=np.asarray(time, dtype=np.float64),
            flux_values=np.asarray(flux, dtype=np.float64),
            scatter=float(scatter),
        )
        solved = self.engine.solve_bound_exact(
            binding,
            periods=periods,
            durations=durations,
            objective="likelihood",
        )
        # The exact chronological kernel uses unit weights.  Astropy's search
        # model uses one constant ``dy=scatter`` for every cadence, which
        # multiplies likelihood power by 1/scatter**2 without changing the
        # fitted epoch or duration.
        inverse_variance = 1.0 / max(float(scatter), 1e-12) ** 2
        return {
            "power": np.asarray(solved.power, dtype=np.float64)
            * inverse_variance,
            "epoch_btjd": np.asarray(solved.epoch_btjd, dtype=np.float64),
            "duration_days": np.asarray(
                solved.duration_days, dtype=np.float64
            ),
        }


class CuPyCompleteGridSearchAdapter(CuPyExactAstropySearchAdapter):
    """High-throughput deterministic full-grid CUDA search adapter.

    Every configured period, duration, phase bin, cadence, epoch, and fitted
    duration is computed on CUDA.  The power-of-two fixed-point histogram
    makes accumulation deterministic without a CPU replay; FP64 is retained
    for phase arithmetic and the likelihood objective.
    """

    def _solve(self, *, time, flux, scatter, periods, durations) -> dict:
        periods = np.asarray(periods, dtype=np.float64)
        durations = np.asarray(durations, dtype=np.float64)
        binding = self.engine.bind(
            time_values=np.asarray(time, dtype=np.float64),
            flux_values=np.asarray(flux, dtype=np.float64),
            scatter=float(scatter),
        )
        solved = self.engine.solve_bound(
            binding,
            periods=periods,
            durations=durations,
            objective="likelihood",
        )
        inverse_variance = 1.0 / max(float(scatter), 1e-12) ** 2
        return {
            "power": np.asarray(solved.power, dtype=np.float64)
            * inverse_variance,
            "epoch_btjd": np.asarray(solved.epoch_btjd, dtype=np.float64),
            "duration_days": np.asarray(
                solved.duration_days, dtype=np.float64
            ),
        }

    def power(self, *, time, flux, scatter, periods, durations) -> np.ndarray:
        periods = np.asarray(periods, dtype=np.float64)
        solved = self._solve(
            time=time,
            flux=flux,
            scatter=scatter,
            periods=periods,
            durations=durations,
        )
        self._cache_key = self._data_key(time, flux, durations)
        self._cache = {
            int(bits): (
                float(solved["power"][index]),
                float(solved["epoch_btjd"][index]),
                float(solved["duration_days"][index]),
            )
            for index, bits in enumerate(periods.view(np.uint64))
        }
        return solved["power"]

    def solution(self, *, time, flux, scatter, periods, durations) -> dict:
        periods = np.asarray(periods, dtype=np.float64)
        same_data = self._data_key(time, flux, durations) == self._cache_key
        keys = [int(bits) for bits in periods.view(np.uint64)]
        if same_data and all(key in self._cache for key in keys):
            self.cache_hits += len(keys)
            rows = [self._cache[key] for key in keys]
            return {
                "power": np.asarray([row[0] for row in rows]),
                "epoch_btjd": np.asarray([row[1] for row in rows]),
                "duration_days": np.asarray([row[2] for row in rows]),
            }
        self.cache_misses += len(keys)
        return self._solve(
            time=time,
            flux=flux,
            scatter=scatter,
            periods=periods,
            durations=durations,
        )

    def diagnostics(self) -> dict:
        return {
            **self.engine.diagnostics(),
            "search_solution_cache_hits": int(self.cache_hits),
            "search_solution_cache_misses": int(self.cache_misses),
            "complete_period_duration_grid": True,
            "deterministic_fixed_point_histogram": True,
            "float64_phase_and_objective": True,
            "cpu_period_grid_replay": False,
            "locked_ephemeris_from_cuda": True,
            "cpu_float64_fixed_ephemeris_acceptance_retained": True,
            "cross_architecture_contract": (
                "final-scientific-state-equivalence-not-raw-peak-bit-identity"
            ),
        }


def get_cuda_astropy_search_adapter() -> CuPyCompleteGridSearchAdapter:
    global _ASTROPY_SEARCH_ADAPTER
    if _ASTROPY_SEARCH_ADAPTER is None:
        _ASTROPY_SEARCH_ADAPTER = CuPyCompleteGridSearchAdapter()
    return _ASTROPY_SEARCH_ADAPTER


def search_branch_cuda_hybrid(*args, **kwargs):
    """Run CUDA coarse proposal followed by the unchanged exact CPU search."""

    engine = get_cuda_engine()
    kwargs["coarse_power_provider"] = engine.power
    candidates, diagnostics = branch_search.search_branch_bls(*args, **kwargs)
    diagnostics.append(
        {
            "stage": "cuda_coarse_search",
            "status": "complete",
            **engine.diagnostics(),
        }
    )
    return candidates, diagnostics


def search_branch_cuda_full(*args, **kwargs):
    """Run complete CUDA broad/fine BLS with unchanged CPU acceptance tests."""

    engine = get_cuda_engine()
    kwargs["coarse_power_provider"] = engine.power
    kwargs["fine_solution_provider"] = engine.solution
    candidates, diagnostics = branch_search.search_branch_bls(*args, **kwargs)
    diagnostics.append(
        {
            "stage": "cuda_full_grid_search",
            "status": "complete",
            **engine.diagnostics(),
            "all_configured_fine_periods_solved_on_cuda": True,
            "top_k_period_truncation": False,
            "cpu_fixed_ephemeris_acceptance_retained": True,
        }
    )
    return candidates, diagnostics


def search_branch_cuda_complete_grid(*args, **kwargs):
    """Keep reference coarse seeds and solve every fine-grid cell on CUDA.

    The coarse spectrum defines which refinement windows exist, so replacing
    it with a merely similar GPU periodogram changes the scientific candidate
    set.  The production CPU pipeline therefore remains authoritative for the
    coarse grid.  CUDA accelerates only the complete fine grids and returns
    their locked period/epoch/duration solutions.
    """

    adapter = get_cuda_astropy_search_adapter()
    # Deliberately do not provide ``coarse_power_provider``.  The reference
    # Astropy spectrum must choose the same exhaustive 2%-separated windows as
    # the optimized Mac/CPU implementation.
    kwargs.pop("coarse_power_provider", None)
    kwargs["fine_power_provider"] = adapter.power
    kwargs["fine_solution_provider"] = adapter.solution
    # Use the reference strict local-maximum predicate.  Expanding a flat peak
    # by a numerical tolerance can turn adjacent fine-grid cells into thousands
    # of artificial hypotheses even when the FP64 powers differ only by
    # round-off.
    kwargs["fine_peak_absolute_power_tolerance"] = 0.0
    candidates, diagnostics = branch_search.search_branch_bls(*args, **kwargs)
    diagnostics.append(
        {
            "stage": "cuda_complete_fine_grid_search",
            "status": "complete",
            **adapter.diagnostics(),
            "coarse_seed_backend": "reference-astropy-cpu-exact",
            "coarse_seed_contract_retained": True,
            "cuda_scope": "complete-fine-grids-only",
        }
    )
    return candidates, diagnostics


def search_branch_cuda_all_complete_grid(*args, **kwargs):
    """Solve both broad and fine grids with the exact CUDA search kernel.

    This is an isolated Avril V2 experiment.  Unlike the older
    ``search_branch_cuda_complete_grid`` path, the 18,000-cell broad spectrum
    is not replayed by Astropy on the CPU.  The same deterministic FP64-phase,
    fixed-point-accumulation CUDA implementation evaluates the complete broad
    and fine grids.  The configured grids, duration bank, local-maximum rule,
    SDE/SNR thresholds, and uncapped peak retention are unchanged.

    The purpose is to remove the long CPU gap visible between CUDA fine-grid
    launches, not to truncate the search.  Acceptance requires a real ledger
    regression against the reference-coarse path before this can leave the
    experiment tree.
    """

    adapter = get_cuda_astropy_search_adapter()
    kwargs["coarse_power_provider"] = adapter.power
    kwargs["fine_power_provider"] = adapter.power
    kwargs["fine_solution_provider"] = adapter.solution
    kwargs["fine_peak_absolute_power_tolerance"] = 0.0
    candidates, diagnostics = branch_search.search_branch_bls(*args, **kwargs)
    diagnostics.append(
        {
            "stage": "cuda_complete_broad_and_fine_grid_search",
            "status": "complete",
            **adapter.diagnostics(),
            "coarse_seed_backend": "deterministic-complete-grid-cuda",
            "complete_broad_period_grid": True,
            "complete_fine_period_grids": True,
            "top_k_period_truncation": False,
            "scientific_thresholds_changed": False,
            "experimental_requires_ledger_regression": True,
        }
    )
    return candidates, diagnostics


def search_branch_cuda_astropy_exact(*args, **kwargs):
    """Compatibility alias for the pre-freeze experimental function name."""

    return search_branch_cuda_complete_grid(*args, **kwargs)


def search_block_cuda_hybrid(
    branch,
    block_index: int,
    segments: list[int],
    period_max: float,
):
    """Spawn-safe temporal-block worker for the CUDA-assisted pipeline."""

    optimization_v3.install_worker_optimizations()
    previous_search = core.search_branch_bls
    core.search_branch_bls = search_branch_cuda_hybrid
    try:
        rows, diagnostics = core._search_block(
            branch,
            block_index=int(block_index),
            segments=list(segments),
            period_max=float(period_max),
        )
    finally:
        core.search_branch_bls = previous_search
    diagnostics = dict(diagnostics)
    diagnostics["search_backend"] = get_cuda_engine().diagnostics()
    return rows, diagnostics


def search_block_cuda_full(
    branch,
    block_index: int,
    segments: list[int],
    period_max: float,
):
    """Spawn-safe worker for complete CUDA broad and fine-grid BLS."""

    optimization_v3.install_worker_optimizations()
    previous_search = core.search_branch_bls
    core.search_branch_bls = search_branch_cuda_full
    try:
        rows, diagnostics = core._search_block(
            branch,
            block_index=int(block_index),
            segments=list(segments),
            period_max=float(period_max),
        )
    finally:
        core.search_branch_bls = previous_search
    diagnostics = dict(diagnostics)
    diagnostics["search_backend"] = get_cuda_engine().diagnostics()
    return rows, diagnostics


def search_block_cuda_complete_grid(
    branch,
    block_index: int,
    segments: list[int],
    period_max: float,
):
    """Spawn-safe worker for reference-coarse, complete fine-grid CUDA."""

    optimization_v3.install_worker_optimizations()
    previous_search = core.search_branch_bls
    core.search_branch_bls = search_branch_cuda_complete_grid
    try:
        rows, diagnostics = core._search_block(
            branch,
            block_index=int(block_index),
            segments=list(segments),
            period_max=float(period_max),
        )
    finally:
        core.search_branch_bls = previous_search
    diagnostics = dict(diagnostics)
    diagnostics["search_backend"] = (
        get_cuda_astropy_search_adapter().diagnostics()
    )
    return rows, diagnostics


def search_block_cuda_all_complete_grid(
    branch,
    block_index: int,
    segments: list[int],
    period_max: float,
):
    """Spawn-safe complete broad+fine CUDA worker for the V2 experiment."""

    optimization_v3.install_worker_optimizations()
    previous_search = core.search_branch_bls
    core.search_branch_bls = search_branch_cuda_all_complete_grid
    try:
        rows, diagnostics = core._search_block(
            branch,
            block_index=int(block_index),
            segments=list(segments),
            period_max=float(period_max),
        )
    finally:
        core.search_branch_bls = previous_search
    diagnostics = dict(diagnostics)
    diagnostics["search_backend"] = (
        get_cuda_astropy_search_adapter().diagnostics()
    )
    return rows, diagnostics


def search_block_cuda_astropy_exact(*args, **kwargs):
    """Compatibility alias for the pre-freeze experimental function name."""

    return search_block_cuda_complete_grid(*args, **kwargs)


__all__ = [
    "CudaCoarseBLS",
    "CudaAstropySearchAdapter",
    "get_cuda_astropy_search_adapter",
    "get_cuda_engine",
    "search_branch_cuda_full",
    "search_branch_cuda_complete_grid",
    "search_branch_cuda_all_complete_grid",
    "search_branch_cuda_astropy_exact",
    "search_branch_cuda_hybrid",
    "search_block_cuda_full",
    "search_block_cuda_complete_grid",
    "search_block_cuda_all_complete_grid",
    "search_block_cuda_astropy_exact",
    "search_block_cuda_hybrid",
]
