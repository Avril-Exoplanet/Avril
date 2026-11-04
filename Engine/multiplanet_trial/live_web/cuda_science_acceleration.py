"""Opt-in CUDA acceleration for VLora's post-discovery numerical grids.

The frozen/reference implementations remain untouched.  This module moves the
large shared period grids used by global ephemeris rescue to CUDA, while the
existing float64 fixed-ephemeris measurements, event-direction checks, and
scientific acceptance rules remain on the CPU.  It is intentionally explicit
about that mixed execution model in every diagnostic payload.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from astropy.timeseries import BoxLeastSquares

from . import optimization_v4
from .exact_bls_acceleration import exact_bls_solutions, prepare_exact_bls


def get_cuda_astropy_bls():
    """Select the deployable CUDA runtime without changing science logic."""

    runtime = os.environ.get("VLORA_CUDA_RUNTIME", "auto").strip().lower()
    if runtime not in {"auto", "cupy", "pycuda"}:
        raise ValueError(
            "VLORA_CUDA_RUNTIME must be 'auto', 'cupy', or 'pycuda'"
        )
    cupy_error = None
    if runtime in {"auto", "cupy"}:
        try:
            from .cupy_astropy_bls import get_cupy_astropy_bls

            return get_cupy_astropy_bls()
        except (ImportError, RuntimeError) as exc:
            cupy_error = exc
            if runtime == "cupy":
                raise
    try:
        from .cuda_astropy_bls import get_cuda_astropy_bls as get_pycuda_bls

        return get_pycuda_bls()
    except (ImportError, RuntimeError) as exc:
        if cupy_error is not None:
            raise RuntimeError(
                f"no CUDA runtime is usable; CuPy: {cupy_error}; PyCUDA: {exc}"
            ) from exc
        raise


def _mad_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    sigma = float(1.4826 * np.median(np.abs(values - median)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(values))
    return sigma if np.isfinite(sigma) and sigma > 0 else 0.0


def _cuda_finalize_workers() -> int:
    """Return the bounded worker count for independent post-fit checks."""

    raw = os.environ.get("VLORA_CUDA_FINALIZE_WORKERS")
    if raw is None:
        return min(8, os.cpu_count() or 1)
    try:
        workers = int(raw)
    except ValueError as exc:
        raise ValueError(
            "VLORA_CUDA_FINALIZE_WORKERS must be an integer"
        ) from exc
    if workers < 1:
        raise ValueError("VLORA_CUDA_FINALIZE_WORKERS must be at least 1")
    return min(workers, os.cpu_count() or 1)


def _finish_cuda_batch(
    batch: list[tuple[int, dict]],
    candidates: list[dict],
    *,
    ensemble,
    execution: str,
    shared_grid_diagnostics: dict | None,
    branch,
    blocks: list[list[int]],
    workers: int,
) -> list[tuple[int, dict]]:
    """Run independent event checks concurrently without changing order."""

    def finish(item):
        (index, source), candidate = item
        before = deepcopy(source.get("holdout_verification") or {})
        return ensemble._finish_refit_candidate(
            index,
            source,
            candidate,
            before,
            execution=execution,
            shared_grid_diagnostics=shared_grid_diagnostics,
            branch=branch,
            blocks=blocks,
        )

    items = list(zip(batch, candidates, strict=False))
    if workers <= 1 or len(items) <= 1:
        return [finish(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        # Executor.map preserves input order, so diagnostics and output are
        # byte-for-byte comparable with the former serial loop.
        return list(pool.map(finish, items))


def _timed_finish_cuda_batch(*args, **kwargs):
    """Return exact finalization rows plus worker-side elapsed time."""

    started = time.perf_counter()
    rows = _finish_cuda_batch(*args, **kwargs)
    return rows, time.perf_counter() - started


@dataclass
class CudaBoundBLSSolver:
    """Bind one light curve to the reusable CUDA period-grid solver."""

    time_values: np.ndarray
    flux_values: np.ndarray
    engine: object

    def __post_init__(self) -> None:
        self.time_values = np.asarray(self.time_values, dtype=np.float64)
        self.flux_values = np.asarray(self.flux_values, dtype=np.float64)
        self.scatter = _mad_sigma(
            self.flux_values - float(np.nanmedian(self.flux_values))
        )
        if self.scatter <= 0:
            raise ValueError("bound CUDA light curve has zero robust scatter")
        self.binding = self.engine.bind(
            time_values=self.time_values,
            flux_values=self.flux_values,
            scatter=self.scatter,
        )
        # The normal CUDA path never needs the CPU BLS workspace.  Build it
        # lazily only when the explicit CPU audit/fallback mode is requested.
        self.cpu_exact_prepared = None
        self.calls = 0
        self.periods_evaluated = 0
        self.seconds = 0.0
        self.power_calls = 0
        self.power_periods_evaluated = 0
        self.power_seconds = 0.0
        self._cache_period_bits: dict[int, int] = {}
        self._cache_durations = b""
        self._cache_result = None
        self.cache_hits = 0
        self._prefetch_results: dict[
            bytes, list[tuple[dict[int, int], object]]
        ] = {}
        self.prefetch_calls = 0
        self.prefetch_grid_calls = 0
        self.prefetch_periods_evaluated = 0
        self.prefetch_seconds = 0.0
        self.cpu_exact_selected_calls = 0
        self.cpu_exact_selected_periods = 0
        self.cpu_exact_selected_seconds = 0.0
        self.cpu_exact_selected_backend = None
        self.selected_period_solution_source = None

    def _cpu_exact_selected_solutions(
        self,
        periods: np.ndarray,
        durations: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Recompute every promoted CUDA grid point with the CPU oracle.

        CUDA still scans the complete coarse/fine lattice.  Only the points
        promoted by the unchanged VLora selection rule enter this second
        pass, which removes GPU reduction-order drift from the reported
        period and epoch without truncating the GPU search.
        """

        started = time.perf_counter()
        if self.cpu_exact_prepared is None:
            self.cpu_exact_prepared = prepare_exact_bls(
                self.time_values, self.flux_values
            )
        result = exact_bls_solutions(
            self.time_values,
            self.flux_values,
            periods=periods,
            durations=durations,
            threads=min(16, os.cpu_count() or 1),
            prepared=self.cpu_exact_prepared,
        )
        if result is None:
            raise RuntimeError(
                "CPU exact BLS backend is required for CUDA result correction"
            )
        self.cpu_exact_selected_calls += 1
        self.cpu_exact_selected_periods += int(periods.size)
        self.cpu_exact_selected_seconds += time.perf_counter() - started
        self.cpu_exact_selected_backend = str(result.get("backend") or "unknown")
        self.selected_period_solution_source = "cpu-chronological-fp64"
        return {
            "power": np.asarray(result["power"], dtype=np.float64),
            "epoch_btjd": np.asarray(result["epoch_btjd"], dtype=np.float64),
            "duration_days": np.full(
                periods.size, float(durations[0]), dtype=np.float64
            ),
        }

    def _gpu_exact_selected_solutions(
        self,
        periods: np.ndarray,
        durations: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Run the chronological selected-period oracle entirely on CUDA."""

        solve_exact = getattr(self.engine, "solve_bound_exact", None)
        if solve_exact is None:
            raise RuntimeError("selected CUDA engine has no exact correction kernel")
        started = time.perf_counter()
        solved = solve_exact(
            self.binding,
            periods=periods,
            durations=durations,
            objective="snr",
        )
        self.cpu_exact_selected_calls += 1
        self.cpu_exact_selected_periods += int(periods.size)
        self.cpu_exact_selected_seconds += time.perf_counter() - started
        self.cpu_exact_selected_backend = "cuda-chronological-fp64"
        self.selected_period_solution_source = "cuda-chronological-fp64"
        return {
            "power": np.asarray(solved.power, dtype=np.float64),
            "epoch_btjd": np.asarray(solved.epoch_btjd, dtype=np.float64),
            "duration_days": np.asarray(solved.duration_days, dtype=np.float64),
        }

    def _from_last_power_grid(
        self,
        periods: np.ndarray,
        durations: np.ndarray,
    ) -> dict[str, np.ndarray] | None:
        """Return selected points from the immediately preceding full scan."""

        if (
            self._cache_result is None
            or self._cache_durations != durations.tobytes()
        ):
            return None
        indices = [
            self._cache_period_bits.get(int(bits))
            for bits in periods.view(np.uint64)
        ]
        if not all(index is not None for index in indices):
            return None
        selected = np.asarray(indices, dtype=int)
        self.cache_hits += 1
        self.selected_period_solution_source = (
            "cuda-complete-grid-deterministic-int64-fp64"
        )
        return {
            "power": np.asarray(
                self._cache_result.power[selected], dtype=np.float64
            ),
            "epoch_btjd": np.asarray(
                self._cache_result.epoch_btjd[selected], dtype=np.float64
            ),
            "duration_days": np.asarray(
                self._cache_result.duration_days[selected], dtype=np.float64
            ),
        }

    def _from_prefetch(
        self,
        periods: np.ndarray,
        durations: np.ndarray,
    ) -> dict[str, np.ndarray] | None:
        cached_groups = self._prefetch_results.get(durations.tobytes())
        if cached_groups is None:
            return None
        for period_bits, result in cached_groups:
            indices = [
                period_bits.get(int(bits))
                for bits in periods.view(np.uint64)
            ]
            if not all(index is not None for index in indices):
                continue
            selected = np.asarray(indices, dtype=int)
            self.cache_hits += 1
            return {
                "power": np.asarray(result.power[selected], dtype=np.float64),
                "epoch_btjd": np.asarray(
                    result.epoch_btjd[selected], dtype=np.float64
                ),
                "duration_days": np.asarray(
                    result.duration_days[selected], dtype=np.float64
                ),
            }
        return None

    def prefetch(
        self,
        requests: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Evaluate many disjoint grids in duration-grouped CUDA launches."""
        started = time.perf_counter()
        grouped: dict[
            tuple[bytes, str, int], tuple[np.ndarray, list[np.ndarray]]
        ] = {}
        for raw_periods, raw_durations in requests:
            periods = np.asarray(raw_periods, dtype=np.float64)
            durations = np.asarray(raw_durations, dtype=np.float64)
            duration_key = durations.tobytes()
            bin_duration = float(np.min(durations)) / 10.0
            maximum_bins = int(
                np.ceil(float(np.max(periods)) / bin_duration)
            ) + 11
            shared_bytes = 2 * maximum_bins * np.dtype(np.float64).itemsize
            if shared_bytes <= self.engine.maximum_shared_bytes:
                # Dynamic shared memory controls occupancy.  An 8 KiB tier
                # prevents a 20-day cluster from forcing short-period blocks
                # to reserve its much larger bin arrays.
                memory_kind = "shared-8k"
                memory_bucket = int(np.ceil(shared_bytes / (8 * 1024)))
            else:
                # Global-memory rows are instead grouped by a narrow period
                # ratio so the longest row cannot inflate every allocation.
                memory_kind = "global-log2-half"
                memory_bucket = int(
                    np.floor(2.0 * np.log2(float(np.max(periods))))
                )
            key = (duration_key, memory_kind, memory_bucket)
            if key not in grouped:
                grouped[key] = (durations, [])
            grouped[key][1].append(periods)
        prefetched: dict[bytes, list[tuple[dict[int, int], object]]] = {}
        points = 0
        for (duration_key, _kind, _bucket), (
            durations,
            period_groups,
        ) in grouped.items():
            periods = np.unique(np.concatenate(period_groups))
            solved = self.engine.solve_bound(
                self.binding,
                periods=periods,
                durations=durations,
                objective="snr",
            )
            period_bits = {
                int(bits): index
                for index, bits in enumerate(periods.view(np.uint64))
            }
            prefetched.setdefault(duration_key, []).append(
                (period_bits, solved)
            )
            points += int(periods.size)
        self._prefetch_results = prefetched
        self.prefetch_calls += 1
        self.prefetch_grid_calls += len(grouped)
        self.prefetch_periods_evaluated += points
        self.prefetch_seconds += time.perf_counter() - started

    def __call__(
        self,
        *,
        periods: np.ndarray,
        durations: np.ndarray,
    ) -> dict[str, np.ndarray]:
        started = time.perf_counter()
        periods = np.asarray(periods, dtype=np.float64)
        durations = np.asarray(durations, dtype=np.float64)
        correction = os.environ.get(
            "VLORA_CUDA_EXACT_CORRECTION", "gpu"
        ).strip().lower()
        if correction == "cpu":
            exact = self._cpu_exact_selected_solutions(periods, durations)
        elif correction == "gpu-chronological":
            exact = self._gpu_exact_selected_solutions(periods, durations)
        elif correction in {"gpu", "fast"}:
            exact = self._from_last_power_grid(periods, durations)
            if exact is None:
                raise RuntimeError(
                    "complete CUDA grid could not recover the selected points"
                )
        else:
            raise ValueError(
                "VLORA_CUDA_EXACT_CORRECTION must be 'gpu', "
                "'gpu-chronological', or 'cpu'"
            )
        self.calls += 1
        self.periods_evaluated += int(periods.size)
        self.seconds += time.perf_counter() - started
        return exact

    def power(
        self,
        *,
        periods: np.ndarray,
        durations: np.ndarray,
    ) -> np.ndarray:
        started = time.perf_counter()
        periods = np.asarray(periods, dtype=np.float64)
        durations = np.asarray(durations, dtype=np.float64)
        prefetched = self._from_prefetch(periods, durations)
        if prefetched is not None:
            self.power_calls += 1
            self.power_periods_evaluated += int(periods.size)
            self.power_seconds += time.perf_counter() - started
            return prefetched["power"]
        solved = self.engine.solve_bound(
            self.binding,
            periods=periods,
            durations=durations,
            objective="snr",
        )
        self._cache_period_bits = {
            int(bits): index
            for index, bits in enumerate(periods.view(np.uint64))
        }
        self._cache_durations = durations.tobytes()
        self._cache_result = solved
        self.power_calls += 1
        self.power_periods_evaluated += int(periods.size)
        self.power_seconds += time.perf_counter() - started
        return np.asarray(solved.power, dtype=np.float64)

    def diagnostics(self) -> dict:
        return {
            "backend": "vlora-deterministic-complete-grid-cuda-bls-v4",
            "calls": int(self.calls),
            "periods_evaluated": int(self.periods_evaluated),
            "seconds": round(float(self.seconds), 6),
            "grid_power_calls": int(self.power_calls),
            "grid_power_periods_evaluated": int(
                self.power_periods_evaluated
            ),
            "grid_power_seconds": round(float(self.power_seconds), 6),
            "solution_cache_hits": int(self.cache_hits),
            "prefetch_passes": int(self.prefetch_calls),
            "prefetch_duration_group_calls": int(self.prefetch_grid_calls),
            "prefetch_periods_evaluated": int(
                self.prefetch_periods_evaluated
            ),
            "prefetch_seconds": round(float(self.prefetch_seconds), 6),
            "cpu_exact_selected_calls": int(self.cpu_exact_selected_calls),
            "cpu_exact_selected_periods": int(
                self.cpu_exact_selected_periods
            ),
            "cpu_exact_selected_seconds": round(
                float(self.cpu_exact_selected_seconds), 6
            ),
            "cpu_exact_selected_backend": self.cpu_exact_selected_backend,
            "selected_period_solution_source": (
                self.selected_period_solution_source
            ),
            "complete_grids": True,
            "top_k_period_truncation": False,
            "epoch_convention": "astropy_mid_transit_phase",
            "cpu_float64_fixed_ephemeris_validation": True,
        }


def refit_failed_vote_families_cuda(
    validated: list[dict],
    blocks: list[list[int]],
    branch,
    *,
    workers: int = 1,
    mode: str = "batched-exact",
    progress=None,
) -> tuple[list[dict], dict]:
    """CUDA-grid equivalent of the evidence-gated global refit.

    No candidate or period cell is capped.  Shared mode evaluates the exact
    union of original grids and preserves each candidate's membership mask;
    only the BLS evaluator is replaced.  CPU validation and post-refit event
    rules are unchanged.
    """

    # Imported lazily to avoid a module cycle at interpreter startup.
    from . import ensemble_intersection as ensemble

    if mode not in {"targeted", "batched-exact", "exact", "shared"}:
        raise ValueError(
            "refit mode must be 'targeted', 'batched-exact', 'exact', or 'shared'"
        )
    requested_mode = mode
    eligible = [
        (index, row)
        for index, row in enumerate(validated)
        if ensemble.ephemeris_refit_eligible(row)
    ]
    output = list(validated)
    if not eligible:
        return output, {
            "eligible_families": 0,
            "attempted_families": 0,
            "rescued_families": 0,
            "workers": 0,
            "execution": "cuda_complete_grid_refit",
        }

    targeted_decision = None
    rescue_eligible = eligible
    if requested_mode == "targeted":
        rescue_eligible, targeted_decision = (
            ensemble._schedule_targeted_refits_before_compute(eligible, output)
        )
        mode = "batched-exact"

    # Preserve the optimized CPU pipeline in front of CUDA.  The weighted
    # two-parameter event clock is both exact with respect to the unchanged
    # acceptance checks and much cheaper than any dense BLS grid.  Earlier
    # CUDA experiments accidentally bypassed it and sent every eligible
    # family straight to the GPU.
    linear_started = time.perf_counter()
    linear_candidates = [deepcopy(row) for _index, row in rescue_eligible]
    linear_diagnostics = optimization_v4.try_linear_event_anchor_rescue_batch(
        linear_candidates,
        blocks,
        branch,
    )
    linear_accepted_indices: set[int] = set()
    dense_eligible: list[tuple[int, dict]] = []
    fixed_block_linear_rescues = 0
    for (index, source), candidate in zip(
        rescue_eligible, linear_candidates, strict=False
    ):
        if candidate.get("holdout_verification", {}).get("passed") is not True:
            dense_eligible.append((index, source))
            continue
        fixed_block_linear_rescues += 1
        completed_index, completed_candidate = ensemble._finish_refit_candidate(
            index,
            source,
            candidate,
            deepcopy(source.get("holdout_verification") or {}),
            execution="batched_weighted_linear_event_anchor_seed_before_cuda",
            branch=branch,
            blocks=blocks,
        )
        if (
            completed_candidate.get("holdout_verification", {}).get("passed")
            is True
        ):
            output[completed_index] = completed_candidate
            linear_accepted_indices.add(completed_index)
        else:
            dense_eligible.append((index, source))
    linear_seconds = time.perf_counter() - linear_started

    harmonic_groups, scheduling_relations = (
        ensemble._plan_pre_refit_harmonic_groups(
            dense_eligible,
            branch=branch,
            require_event_proof=True,
        )
    )
    scheduled = [item for group in harmonic_groups for item in group]
    scheduled_indices = [index for index, _row in scheduled]
    if (
        len(scheduled) != len(dense_eligible)
        or len(set(scheduled_indices)) != len(dense_eligible)
        or set(scheduled_indices) != {
            index for index, _row in dense_eligible
        }
    ):
        raise RuntimeError("CUDA harmonic scheduler did not preserve every family")

    if not scheduled:
        return output, {
            "eligible_families": len(eligible),
            "attempted_families": len(rescue_eligible),
            "dense_cuda_families": 0,
            "rescued_families": len(linear_accepted_indices),
            "linear_fixed_block_rescues": fixed_block_linear_rescues,
            "linear_accepted_rescues": len(linear_accepted_indices),
            "linear_diagnostics": linear_diagnostics,
            "requested_mode": requested_mode,
            "targeted_rescue_decision": targeted_decision,
            "harmonic_scheduling_relations": scheduling_relations,
            "workers": 0,
            "execution": "linear_event_anchor_rescue_completed_before_cuda",
            "elapsed_seconds": 0.0,
            "used_catalog_period": False,
        }

    engine = get_cuda_astropy_bls()
    solver = CudaBoundBLSSolver(branch.time, branch.flux, engine)
    prepared_blocks = optimization_v4._prepare_blocks(blocks, branch)
    fixed_ephemeris_mode = os.environ.get(
        "VLORA_CUDA_FIXED_EPHEMERIS", "off"
    ).strip().lower()
    if fixed_ephemeris_mode not in {"off", "auto", "on"}:
        raise ValueError(
            "VLORA_CUDA_FIXED_EPHEMERIS must be 'off', 'auto', or 'on'"
        )
    cuda_fixed_ephemeris = None
    if fixed_ephemeris_mode in {"auto", "on"}:
        try:
            from .cuda_fixed_ephemeris import CudaFixedEphemerisMeasurer

            cuda_fixed_ephemeris = CudaFixedEphemerisMeasurer(prepared_blocks)
        except Exception:
            if fixed_ephemeris_mode == "on":
                raise
    # The object is retained for the unmodified function interface.  CUDA is
    # selected by ``solution_provider`` and therefore does not call it.
    bls = BoxLeastSquares(branch.time, branch.flux)
    baseline_days = float(np.nanmax(branch.time) - np.nanmin(branch.time))
    if mode == "shared":
        maximum = max(
            1,
            int(
                os.environ.get(
                    "HOUXING_CUDA_SHARED_REFIT_MAX_CANDIDATES", "64"
                )
            ),
        )
        planned_clusters = ensemble._plan_shared_refit_batches(
            scheduled,
            branch,
            max_candidates=maximum,
        )
        # Preserve every overlap cluster, but pack several unrelated clusters
        # into one CUDA super-batch.  The generic shared evaluator rebuilds the
        # same exact-membership clusters, prefetches coarse grids by duration,
        # then does the same for the data-dependent fine grids.  Fixed-window
        # validation remains cluster-local, so this reduces GPU launches
        # without creating the enormous CPU validation matrix seen in the
        # independent-grid batch experiment.
        batches = []
        current: list[tuple[int, dict]] = []
        for cluster in planned_clusters:
            if current and len(current) + len(cluster) > maximum:
                batches.append(current)
                current = []
            current.extend(cluster)
        if current:
            batches.append(current)
    else:
        # Independent grids still benefit from a large duration-grouped
        # super-batch: every family's lattice and membership are preserved,
        # while one call can feed many same-duration arrays to the GPU and
        # compiled CPU oracle.  The previous hard-coded 64 produced hundreds
        # of tiny launches on candidate-explosion targets such as TOI-4307.
        maximum = max(
            1,
            int(os.environ.get("HOUXING_CUDA_REFIT_BATCH_SIZE", "256")),
        )
        batches = [
            scheduled[start : start + maximum]
            for start in range(0, len(scheduled), maximum)
        ]

    completed = len(linear_accepted_indices)
    rescued = len(linear_accepted_indices)
    shared_individual_points = 0
    shared_grid_points = 0
    exact_fallbacks = 0
    progress_interval = max(1, len(rescue_eligible) // 50)
    next_progress = progress_interval
    started = time.monotonic()
    balance_seconds = 0.0
    finalize_seconds = 0.0
    finalize_workers = _cuda_finalize_workers()
    measurement_scope = optimization_v4.fixed_ephemeris_measurement_provider(
        cuda_fixed_ephemeris.measure_prepared_blocks
        if cuda_fixed_ephemeris is not None
        else None
    )
    measurement_scope.__enter__()
    pending = None

    def consume_pending(pending_item) -> None:
        nonlocal completed, rescued, finalize_seconds, next_progress
        if pending_item is None:
            return
        pending_batch, future = pending_item
        finished_rows, elapsed = future.result()
        finalize_seconds += float(elapsed)
        for index, finished in finished_rows:
            output[index] = finished
            rescued += int(
                finished["ephemeris_refit"]["rescued_failed_fixed_ephemeris"]
            )
        completed += len(pending_batch)
        if progress and (
            completed == len(rescue_eligible) or completed >= next_progress
        ):
            progress(
                "refitting",
                f"CUDA global period/epoch refit {completed}/{len(rescue_eligible)}",
                {
                    "families_completed": completed,
                    "families_total": len(rescue_eligible),
                    "families_rescued": rescued,
                    "gpu_workers": 1,
                    "batch_candidates": maximum,
                    "execution": "cuda_complete_grid_cpu_exact_validation",
                    "gpu_cpu_double_buffered": True,
                },
            )
            while next_progress <= completed:
                next_progress += progress_interval

    try:
        # A bounded two-stage pipeline keeps at most one CPU finalization batch
        # pending.  CUDA evaluates batch N+1 while the unchanged event and
        # direction checks finalize batch N, then results are consumed in the
        # original deterministic order.
        with ThreadPoolExecutor(max_workers=1) as finalize_pipeline:
            for batch in batches:
                candidates = [deepcopy(source) for _index, source in batch]
                balance_started = time.perf_counter()
                if mode == "shared":
                    diagnostics = optimization_v4.balance_family_ephemerides_shared(
                        candidates,
                        blocks,
                        branch,
                        prepared_blocks=prepared_blocks,
                        bls=bls,
                        baseline_days=baseline_days,
                        max_candidates=len(candidates),
                        solution_provider=solver,
                        power_provider=solver.power,
                        power_prefetch=solver.prefetch,
                    )
                    shared_individual_points += int(
                        diagnostics["individual_coarse_points"]
                    )
                    shared_grid_points += int(
                        diagnostics["shared_coarse_points"]
                    )
                else:
                    diagnostics = {
                        "execution": "independent_complete_cuda_grids"
                    }
                    optimization_v4.balance_family_ephemerides_vectorized(
                        candidates,
                        blocks,
                        branch,
                        attach_event_repeatability=False,
                        batch_size=len(candidates),
                        prepared_blocks=prepared_blocks,
                        bls=bls,
                        baseline_days=baseline_days,
                        solution_provider=solver,
                        power_provider=solver.power,
                    )
                balance_seconds += time.perf_counter() - balance_started

                # Waiting here happens after current GPU work, so it overlaps
                # with the previous CPU batch instead of idling the device.
                consume_pending(pending)
                pending = (
                    batch,
                    finalize_pipeline.submit(
                        _timed_finish_cuda_batch,
                        batch,
                        candidates,
                        ensemble=ensemble,
                        execution="cuda_complete_grid_cpu_exact_validation",
                        shared_grid_diagnostics=diagnostics,
                        branch=branch,
                        blocks=blocks,
                        workers=finalize_workers,
                    ),
                )
            consume_pending(pending)
    finally:
        measurement_scope.__exit__(None, None, None)

    pipeline_wall_seconds = time.monotonic() - started

    return output, {
        "eligible_families": len(eligible),
        "attempted_families": len(rescue_eligible),
        "dense_cuda_families": len(scheduled),
        "requested_mode": requested_mode,
        "resolved_mode": mode,
        "targeted_rescue_decision": targeted_decision,
        "rescued_families": rescued,
        "linear_fixed_block_rescues": fixed_block_linear_rescues,
        "linear_accepted_rescues": len(linear_accepted_indices),
        "linear_diagnostics": linear_diagnostics,
        "harmonic_scheduling_relations": scheduling_relations,
        "workers": 1,
        "requested_cpu_workers_ignored": int(workers),
        "execution": "cuda_complete_grid_cpu_exact_validation",
        "process_batch_size": maximum,
        "shared_grid_individual_coarse_points": (
            shared_individual_points if mode == "shared" else None
        ),
        "shared_grid_evaluated_coarse_points": (
            shared_grid_points if mode == "shared" else None
        ),
        "shared_grid_coarse_point_reuse": (
            shared_individual_points / max(1, shared_grid_points)
            if mode == "shared"
            else None
        ),
        "shared_grid_exact_fallbacks": exact_fallbacks,
        "cuda_solver": solver.diagnostics(),
        "cuda_engine": engine.diagnostics(),
        "cuda_fixed_ephemeris": (
            cuda_fixed_ephemeris.diagnostics()
            if cuda_fixed_ephemeris is not None
            else {
                "backend": "cpu-compiled",
                "requested_mode": fixed_ephemeris_mode,
            }
        ),
        "stage_seconds": {
            "linear_event_anchor_prefit_and_checks": round(
                float(linear_seconds), 6
            ),
            "dense_grid_balance_and_fixed_block_validation": round(
                float(balance_seconds), 6
            ),
            "postfit_event_checks": round(float(finalize_seconds), 6),
        },
        "linear_event_anchor_prefit": {
            "input_eligible_families": len(eligible),
            "attempted_families": len(rescue_eligible),
            "fixed_block_rescues": fixed_block_linear_rescues,
            "accepted_after_event_checks": len(linear_accepted_indices),
            "dense_fallback_families_after_targeted_board": len(dense_eligible),
            "candidate_removed_without_validation": False,
        },
        "postfit_event_check_workers": int(finalize_workers),
        "gpu_cpu_double_buffered": True,
        "pipeline_overlap_seconds": round(
            max(
                0.0,
                balance_seconds + finalize_seconds - pipeline_wall_seconds,
            ),
            6,
        ),
        "elapsed_seconds": round(pipeline_wall_seconds, 6),
        "used_catalog_period": False,
    }


__all__ = [
    "CudaBoundBLSSolver",
    "refit_failed_vote_families_cuda",
]
