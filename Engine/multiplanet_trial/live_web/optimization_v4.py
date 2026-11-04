"""Exact vectorized acceleration layered on top of optimized V3.

V3 already parallelizes the unchanged Astropy BLS period grid.  This module
keeps that implementation and removes three remaining Python-level loops:

* locked ephemerides are measured for all candidates as one compiled batch;
* the global family-balancing grid is measured as one batch per fold; and
* every requested phase-null trial is measured by the compiled event-window
  kernel.

No period, duration, phase trial, cadence, candidate, threshold, or ranking
rule is changed.  Every compiled path has the original scalar implementation
as its fallback.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from collections import defaultdict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np
from astropy.timeseries import BoxLeastSquares

from . import detection_significance
from . import optimization_v3
from . import pipeline as reference
from .fixed_ephemeris_acceleration import measure_fixed_ephemerides_compiled
from .significance_acceleration import phase_trial_events_compiled


_ORIGINAL_PHASE_SHIFT_NULL = detection_significance._phase_shift_null
FIXED_BLOCK_WORKERS_ENV = "HOUXING_FIXED_EPHEMERIS_BLOCK_WORKERS"
_FIXED_EPHEMERIS_PROVIDER: ContextVar[Callable | None] = ContextVar(
    "vlora_fixed_ephemeris_provider", default=None
)


@contextmanager
def fixed_ephemeris_measurement_provider(provider: Callable | None):
    """Temporarily route locked-ephemeris batches to an alternate backend."""

    token = _FIXED_EPHEMERIS_PROVIDER.set(provider)
    try:
        yield
    finally:
        _FIXED_EPHEMERIS_PROVIDER.reset(token)


def _measure_fixed_ephemerides_exact(
    time_values: np.ndarray,
    flux_values: np.ndarray,
    candidates: list[Mapping],
    *,
    threads: int | None = None,
) -> list[dict]:
    """Measure every row exactly, using the scalar kernel as a fallback."""

    compiled = measure_fixed_ephemerides_compiled(
        np.asarray(time_values, dtype=np.float64),
        np.asarray(flux_values, dtype=np.float64),
        candidates,
        threads=threads,
    )
    if compiled is not None:
        return compiled
    return [
        reference.measure_fixed_ephemeris(
            time_values,
            flux_values,
            period_days=float(candidate["period_days"]),
            epoch_btjd=float(candidate["epoch_btjd"]),
            duration_hours=float(candidate["duration_hours"]),
        )
        for candidate in candidates
    ]


def _prepare_blocks(
    blocks: list[list[int]], branch: reference.SearchBranch
) -> list[reference.SearchBranch]:
    return [reference._subset_branch(branch, segments) for segments in blocks]


def _measure_prepared_blocks_batch(
    candidates: list[Mapping],
    blocks: list[list[int]],
    prepared_blocks: list[reference.SearchBranch],
) -> list[list[dict]]:
    """Return the unchanged per-fold dictionaries in candidate order."""

    provider = _FIXED_EPHEMERIS_PROVIDER.get()
    if provider is not None:
        return provider(candidates, blocks, prepared_blocks)

    output: list[list[dict]] = [[] for _ in candidates]
    block_workers = min(
        max(1, int(os.environ.get(FIXED_BLOCK_WORKERS_ENV, "1"))),
        len(prepared_blocks),
    )
    if block_workers > 1 and candidates:
        total_threads = max(
            1,
            int(
                os.environ.get(
                    "HOUXING_FIXED_EPHEMERIS_THREADS",
                    os.environ.get("HOUXING_BLS_THREADS", "1"),
                )
            ),
        )
        threads_per_block = max(1, total_threads // block_workers)
        with ThreadPoolExecutor(max_workers=block_workers) as pool:
            futures = [
                pool.submit(
                    _measure_fixed_ephemerides_exact,
                    local.time,
                    local.flux,
                    candidates,
                    threads=threads_per_block,
                )
                for local in prepared_blocks
            ]
            metrics_by_block = [future.result() for future in futures]
    else:
        metrics_by_block = [
            _measure_fixed_ephemerides_exact(
                local.time,
                local.flux,
                candidates,
            )
            for local in prepared_blocks
        ]
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
            metrics = dict(metrics)
            metrics["timing_offset_days"] = 0.0
            metrics["timing_tolerance_applied"] = False
            output[candidate_index].append(
                {
                    "block": block_index,
                    "segments": list(segments),
                    "independently_discovered": block_index in discovered_blocks,
                    "outcome": reference._metric_outcome(
                        metrics,
                        independently_discovered=block_index in discovered_blocks,
                    ),
                    **metrics,
                }
            )
    return output


def validate_raw_ledger_vectorized(
    ledger: list[dict],
    blocks: list[list[int]],
    branch: reference.SearchBranch,
) -> list[dict]:
    """Validate every raw row; vectorization changes only loop placement."""

    prepared = _prepare_blocks(blocks, branch)
    measurements_by_candidate = _measure_prepared_blocks_batch(
        ledger, blocks, prepared
    )
    return [
        {
            **row,
            "fixed_block_validation": reference._validation_decision(measurements),
        }
        for row, measurements in zip(ledger, measurements_by_candidate)
    ]


def validate_proposals_vectorized(
    proposals: list[dict],
    blocks: list[list[int]],
    branch: reference.SearchBranch,
) -> list[dict]:
    """Validate every proposal and preserve its original order and identity."""

    prepared = _prepare_blocks(blocks, branch)
    uncached_indices = [
        index
        for index, proposal in enumerate(proposals)
        if not isinstance(proposal.get("multiphase_prevalidation"), Mapping)
    ]
    uncached = [proposals[index] for index in uncached_indices]
    measured = _measure_prepared_blocks_batch(uncached, blocks, prepared)
    measured_by_index = dict(zip(uncached_indices, measured))

    output: list[dict] = []
    for index, proposal in enumerate(proposals):
        cached = proposal.get("multiphase_prevalidation")
        if isinstance(cached, Mapping):
            verification = dict(cached)
            verification["fixed_ephemeris_measurement_reused"] = True
        else:
            verification = reference._validation_decision(measured_by_index[index])
        verification["event_repeatability_role"] = (
            "deferred_until_family_selection"
            if verification.get("passed") is True
            else "skipped_after_failed_fixed_ephemeris"
        )
        output.append({**proposal, "holdout_verification": verification})
    return output


def balance_family_ephemeris_vectorized(
    candidate: dict,
    blocks: list[list[int]],
    branch: reference.SearchBranch,
    *,
    attach_event_repeatability: bool = True,
    prepared_blocks: list[reference.SearchBranch] | None = None,
    bls: BoxLeastSquares | None = None,
    baseline_days: float | None = None,
    solution_provider: Callable[..., Mapping] | None = None,
    power_provider: Callable[..., np.ndarray] | None = None,
) -> None:
    """Exact single-family entry point for the batched maximin balance."""

    balance_family_ephemerides_vectorized(
        [candidate],
        blocks,
        branch,
        attach_event_repeatability=attach_event_repeatability,
        batch_size=1,
        prepared_blocks=prepared_blocks,
        bls=bls,
        baseline_days=baseline_days,
        solution_provider=solution_provider,
        power_provider=power_provider,
    )


def _family_balance_spec(
    candidate: dict,
    branch: reference.SearchBranch,
    *,
    baseline_days: float | None = None,
) -> dict | None:
    """Materialize the unchanged period grid for one family."""

    bounds = candidate.get("balanced_family_period_bounds_days")
    if not bounds or len(bounds) != 2:
        return None
    lower, upper = map(float, bounds)
    if not (np.isfinite(lower) and np.isfinite(upper) and upper > lower > 0):
        return None

    duration_hours = float(
        candidate.get("balanced_family_duration_hours")
        or candidate["duration_hours"]
    )
    duration_days = duration_hours / 24.0
    if baseline_days is None:
        baseline_days = float(np.nanmax(branch.time) - np.nanmin(branch.time))
    else:
        baseline_days = float(baseline_days)
    center_period = 0.5 * (lower + upper)
    period_resolution = max(
        np.finfo(float).eps,
        duration_days * center_period / max(baseline_days, center_period) / 10.0,
    )
    grid_size = int(
        np.clip(math.ceil((upper - lower) / period_resolution) + 1, 101, 2001)
    )
    return {
        "candidate": candidate,
        "lower": lower,
        "upper": upper,
        "duration_hours": duration_hours,
        "duration_days": duration_days,
        "grid_size": grid_size,
        "periods": np.linspace(lower, upper, grid_size),
        "original_period_days": float(candidate["period_days"]),
        "original_epoch_btjd": float(candidate["epoch_btjd"]),
    }


def _linear_seed_score(measurements: list[Mapping]) -> tuple:
    """Return the unchanged maximin ordering for one locked ephemeris."""

    covered = [row for row in measurements if row["outcome"] != "untestable"]
    signed_snrs = [
        (
            float(row.get("snr") or 0.0)
            if float(row.get("depth_ppm") or 0.0) > 0
            else -abs(float(row.get("snr") or 0.0))
        )
        for row in covered
    ]
    return (
        len([value for value in signed_snrs if value > 0]),
        min(signed_snrs, default=-np.inf),
        sum(signed_snrs),
    )


def _member_event_anchor(
    member: Mapping,
    branch: reference.SearchBranch,
    *,
    segment_midpoints: Mapping[tuple[int, ...], float] | None = None,
) -> tuple[float, float] | None:
    """Turn one local-block BLS solution into an observed-time anchor.

    A local BLS epoch is periodic rather than tied to one particular transit.
    Shift it by an integer number of cycles to the middle of the exact sectors
    that produced the solution.  The resulting anchor is then suitable for a
    global linear ephemeris fit without reading a truth catalogue.
    """

    period = float(member.get("period_days") or 0.0)
    epoch = float(member.get("epoch_btjd") or np.nan)
    if not (np.isfinite(period) and period > 0 and np.isfinite(epoch)):
        return None
    segments = [int(value) for value in member.get("source_segments", [])]
    if not segments and member.get("source_block") is not None:
        segments = [int(member["source_block"])]
    segment_key = tuple(sorted(set(segments)))
    midpoint = (
        float(segment_midpoints[segment_key])
        if segment_midpoints is not None and segment_key in segment_midpoints
        else np.nan
    )
    if not np.isfinite(midpoint):
        selected = np.isin(np.asarray(branch.segment_id, dtype=int), segments)
        local_time = np.asarray(branch.time, dtype=float)[selected]
        local_time = local_time[np.isfinite(local_time)]
        if local_time.size == 0:
            return None
        midpoint = float(np.median(local_time))
    anchor = epoch + round((midpoint - epoch) / period) * period
    weight = max(
        1.0,
        float(member.get("source_strength") or 0.0),
        float(member.get("robust_sde") or 0.0),
    )
    return float(anchor), float(weight)


def linear_event_anchor_seed_trials(
    candidate: Mapping,
    branch: reference.SearchBranch,
    *,
    maximum_trials: int | None = None,
    segment_midpoints: Mapping[tuple[int, ...], float] | None = None,
) -> list[dict]:
    """Build catalog-blind period/epoch seeds from local event anchors.

    The expensive reference refit searches a dense period lattice.  Before
    doing that, local BLS solutions from independent temporal blocks provide
    a much smaller system ``event_time = epoch + cycle * period``.  Solving
    its weighted 2-column design matrix gives a direct global ephemeris.
    Cycle-count neighbours are retained so multi-year gaps cannot make the
    shortcut brittle.  Failure of every seed is explicitly handled by the
    caller's full-grid fallback.
    """

    period0 = float(candidate["period_days"])
    epoch0 = float(candidate["epoch_btjd"])
    duration_hours = float(
        candidate.get("balanced_family_duration_hours")
        or candidate["duration_hours"]
    )
    bounds = candidate.get("balanced_family_period_bounds_days") or [
        period0,
        period0,
    ]
    lower, upper = map(float, bounds)
    members = [
        dict(row) for row in candidate.get("member_ephemerides", [])
    ]
    identity = {
        "duration_hours": duration_hours,
        "discovered_blocks": candidate.get("discovered_blocks", []),
        "source_block": candidate.get("source_block"),
    }
    trials: list[dict] = []

    def append(period: float, epoch: float, method: str) -> None:
        if maximum_trials is not None and len(trials) >= maximum_trials:
            return
        if not (
            np.isfinite(period)
            and np.isfinite(epoch)
            and period > 0
            and lower - 1e-12 <= period <= upper + 1e-12
        ):
            return
        phase_tolerance = max(np.finfo(float).eps, duration_hours / 24.0 / 20.0)
        for row in trials:
            if abs(float(row["period_days"]) - period) <= max(
                np.finfo(float).eps, period * 1e-10
            ):
                phase_error = abs(
                    (
                        epoch
                        - float(row["epoch_btjd"])
                        + 0.5 * period
                    )
                    % period
                    - 0.5 * period
                )
                if phase_error <= phase_tolerance:
                    return
        trials.append(
            {
                **identity,
                "period_days": float(period),
                "epoch_btjd": float(epoch),
                "linear_seed_method": method,
            }
        )

    append(period0, epoch0, "family_representative")
    for member in sorted(
        members,
        key=lambda row: float(row.get("source_strength") or 0.0),
        reverse=True,
    ):
        append(
            float(member.get("period_days") or np.nan),
            float(member.get("epoch_btjd") or np.nan),
            "local_block_solution",
        )

    # A noisy block may contribute many nearby local peaks.  It must still
    # count as one independent timing observation rather than out-voting the
    # other temporal blocks.  Keep the strongest anchor per discovery block;
    # every member period remains present above as an explicit seed trial.
    anchored_by_block: dict[int, tuple[float, float, dict]] = {}
    for member in members:
        anchor = _member_event_anchor(
            member,
            branch,
            segment_midpoints=segment_midpoints,
        )
        if anchor is not None:
            block = int(member.get("source_block") or -1)
            previous = anchored_by_block.get(block)
            if previous is None or float(anchor[1]) > float(previous[1]):
                anchored_by_block[block] = (anchor[0], anchor[1], member)
    anchored = list(anchored_by_block.values())
    if len(anchored) < 2:
        return trials
    anchored.sort(key=lambda row: row[0])
    event_times = np.asarray([row[0] for row in anchored], dtype=np.float64)
    weights = np.asarray([row[1] for row in anchored], dtype=np.float64)
    reference_time = float(np.median(event_times))

    def fit_for_cycles(cycles: np.ndarray, method: str) -> None:
        if np.unique(cycles).size < 2:
            return
        design = np.column_stack((np.ones(cycles.size), cycles))
        root_weight = np.sqrt(weights / max(float(np.max(weights)), 1.0))
        weighted_design = design * root_weight[:, None]
        weighted_times = event_times * root_weight
        solution, _residuals, rank, _singular = np.linalg.lstsq(
            weighted_design, weighted_times, rcond=None
        )
        if int(rank) < 2:
            return
        append(float(solution[1]), float(solution[0]), method)

    def fit_for_period_guess(period_guess: float, method: str) -> None:
        if not np.isfinite(period_guess) or period_guess <= 0:
            return
        cycles = np.rint((event_times - reference_time) / period_guess)
        fit_for_cycles(cycles, method)

    fit_for_period_guess(period0, "weighted_linear_event_anchors")
    span = float(event_times[-1] - event_times[0])
    # Enumerate every integer cycle count allowed by the unchanged family
    # period bounds.  A fixed +/-2 window silently misses valid aliases on a
    # long baseline; this finite interval is the complete mathematical set.
    minimum_cycles = max(1, int(math.ceil(span / upper - 1e-12)))
    maximum_cycles = max(
        minimum_cycles,
        int(math.floor(span / lower + 1e-12)),
    )
    nominal_cycles = int(round(span / period0)) if period0 > 0 else 0
    for cycle_count in range(minimum_cycles, maximum_cycles + 1):
        pair_period = span / cycle_count
        if lower - 1e-12 <= pair_period <= upper + 1e-12:
            # Anchor the integer labels at the first event.  Centering an even
            # number of observations on their median creates +/-N/2 half-
            # integers for odd N; ``np.rint`` then silently maps two adjacent
            # cycle counts onto the same fit.  Fixing the endpoints to 0 and
            # N makes every mathematically permitted count a distinct trial.
            cycles = np.rint((event_times - event_times[0]) / pair_period)
            cycles[0] = 0.0
            cycles[-1] = float(cycle_count)
            fit_for_cycles(
                cycles,
                (
                    "weighted_linear_cycle_count_"
                    f"{cycle_count}_delta_{cycle_count - nominal_cycles:+d}"
                ),
            )
    return trials


def try_linear_event_anchor_rescue_batch(
    candidates: list[dict],
    blocks: list[list[int]],
    branch: reference.SearchBranch,
    *,
    prepared_blocks: list[reference.SearchBranch] | None = None,
    candidate_batch_size: int = 512,
) -> dict:
    """Try a small matrix-derived ephemeris set before dense BLS refitting.

    Every seed in a batch is measured by the existing compiled locked-
    ephemeris kernel on all temporal blocks.  Only candidates that already
    pass the unchanged validation rule are mutated.  All others remain byte-
    for-byte eligible for the original dense-grid calculation.
    """

    if not candidates:
        return {
            "candidate_count": 0,
            "seed_trials": 0,
            "matrix_solved_candidates": 0,
            "rescued_candidates": 0,
        }
    prepared = prepared_blocks or _prepare_blocks(blocks, branch)
    segment_keys = {
        tuple(
            sorted(
                {
                    int(value)
                    for value in member.get("source_segments", [])
                }
            )
        )
        for candidate in candidates
        for member in candidate.get("member_ephemerides", [])
        if member.get("source_segments")
    }
    segment_ids = np.asarray(branch.segment_id, dtype=int)
    time_values = np.asarray(branch.time, dtype=float)
    segment_midpoints: dict[tuple[int, ...], float] = {}
    for key in segment_keys:
        selected = np.isin(segment_ids, key)
        finite = time_values[selected]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            segment_midpoints[key] = float(np.median(finite))
    batch_size = max(1, int(candidate_batch_size))
    total_trials = 0
    matrix_solved = 0
    rescued = 0
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        flattened: list[dict] = []
        owners: list[int] = []
        for local_index, candidate in enumerate(batch):
            trials = linear_event_anchor_seed_trials(
                candidate,
                branch,
                segment_midpoints=segment_midpoints,
            )
            total_trials += len(trials)
            matrix_solved += int(
                any(
                    str(row.get("linear_seed_method", "")).startswith(
                        "weighted_linear"
                    )
                    for row in trials
                )
            )
            flattened.extend(trials)
            owners.extend([local_index] * len(trials))
        if not flattened:
            continue
        measured = _measure_prepared_blocks_batch(
            flattened, blocks, prepared
        )
        best_by_owner: dict[int, tuple[tuple, int, dict]] = {}
        for flat_index, (owner, measurements) in enumerate(zip(owners, measured)):
            verification = reference._validation_decision(measurements)
            if verification.get("passed") is not True:
                continue
            score = _linear_seed_score(measurements)
            previous = best_by_owner.get(owner)
            if previous is None or score > previous[0]:
                best_by_owner[owner] = (score, flat_index, verification)
        for owner, (score, flat_index, verification) in best_by_owner.items():
            candidate = batch[owner]
            selected = flattened[flat_index]
            candidate["period_days"] = float(selected["period_days"])
            candidate["epoch_btjd"] = float(selected["epoch_btjd"])
            candidate["duration_hours"] = float(selected["duration_hours"])
            verification["event_repeatability_role"] = (
                "deferred_until_linear_seed_selection"
            )
            candidate["holdout_verification"] = verification
            candidate["ephemeris_balance"] = {
                "method": "batched_weighted_linear_event_anchor_seed",
                "selected_seed_method": selected["linear_seed_method"],
                "selected_period_days": float(selected["period_days"]),
                "selected_epoch_btjd": float(selected["epoch_btjd"]),
                "weakest_fold_snr": float(score[1]),
                "seed_trials_for_batch": len(flattened),
                "used_catalog_period": False,
                "full_grid_fallback_on_failure": True,
            }
            rescued += 1
    return {
        "candidate_count": len(candidates),
        "seed_trials": total_trials,
        "matrix_solved_candidates": matrix_solved,
        "rescued_candidates": rescued,
        "execution": "weighted_2_column_linear_ephemeris_plus_compiled_batch_validation",
        "cached_source_segment_midpoints": len(segment_midpoints),
        "failure_policy": "unchanged_dense_grid_fallback",
    }


def _phase_anchor_seed_trials(
    candidate: Mapping,
    anchors: list[tuple[float, float, int]],
) -> list[dict]:
    """Return every bounded integer-cycle fit for independently solved phases."""

    period0 = float(candidate["period_days"])
    duration_hours = float(
        candidate.get("balanced_family_duration_hours")
        or candidate["duration_hours"]
    )
    lower, upper = map(
        float,
        candidate.get("balanced_family_period_bounds_days")
        or [period0, period0],
    )
    identity = {
        "duration_hours": duration_hours,
        "discovered_blocks": candidate.get("discovered_blocks", []),
        "source_block": candidate.get("source_block"),
    }
    trials: list[dict] = []

    def append(period: float, epoch: float, method: str) -> None:
        if not (
            np.isfinite(period)
            and np.isfinite(epoch)
            and period > 0
            and lower - 1e-12 <= period <= upper + 1e-12
        ):
            return
        phase_tolerance = max(np.finfo(float).eps, duration_hours / 24.0 / 20.0)
        for row in trials:
            if abs(float(row["period_days"]) - period) > max(
                np.finfo(float).eps, period * 1e-10
            ):
                continue
            phase_error = abs(
                (epoch - float(row["epoch_btjd"]) + 0.5 * period) % period
                - 0.5 * period
            )
            if phase_error <= phase_tolerance:
                return
        trials.append(
            {
                **identity,
                "period_days": float(period),
                "epoch_btjd": float(epoch),
                "linear_seed_method": method,
            }
        )

    # A correct period with a locally displaced epoch is common.  Every block
    # phase is therefore an explicit seed before any cross-block regression.
    for epoch, _weight, block in anchors:
        append(period0, epoch, f"fixed_period_best_phase_block_{block}")

    # Try every independent block pair.  If one block's strongest phase is a
    # red-noise dip, the other two can still define the true clock; unchanged
    # three-block validation decides whether that clock is acceptable.
    for left, right in combinations(sorted(anchors), 2):
        left_time, _left_weight, left_block = left
        right_time, _right_weight, right_block = right
        span = float(right_time - left_time)
        if span <= 0:
            continue
        minimum_cycles = max(1, int(math.ceil(span / upper - 1e-12)))
        maximum_cycles = max(
            minimum_cycles,
            int(math.floor(span / lower + 1e-12)),
        )
        for cycle_count in range(minimum_cycles, maximum_cycles + 1):
            append(
                span / cycle_count,
                left_time,
                (
                    f"block_phase_pair_{left_block}_{right_block}_"
                    f"cycles_{cycle_count}"
                ),
            )

    if len(anchors) >= 3:
        ordered = sorted(anchors)
        event_times = np.asarray([row[0] for row in ordered], dtype=np.float64)
        weights = np.asarray([row[1] for row in ordered], dtype=np.float64)
        span = float(event_times[-1] - event_times[0])
        minimum_cycles = max(1, int(math.ceil(span / upper - 1e-12)))
        maximum_cycles = max(
            minimum_cycles,
            int(math.floor(span / lower + 1e-12)),
        )
        for cycle_count in range(minimum_cycles, maximum_cycles + 1):
            guess = span / cycle_count
            cycles = np.rint((event_times - event_times[0]) / guess)
            cycles[0] = 0.0
            cycles[-1] = float(cycle_count)
            if np.unique(cycles).size < 2:
                continue
            design = np.column_stack((np.ones(cycles.size), cycles))
            root_weight = np.sqrt(weights / max(float(np.max(weights)), 1.0))
            solution, _residuals, rank, _singular = np.linalg.lstsq(
                design * root_weight[:, None],
                event_times * root_weight,
                rcond=None,
            )
            if int(rank) == 2:
                append(
                    float(solution[1]),
                    float(solution[0]),
                    f"all_block_phase_weighted_cycles_{cycle_count}",
                )
    return trials


def try_block_phase_linear_rescue_batch(
    candidates: list[dict],
    blocks: list[list[int]],
    branch: reference.SearchBranch,
    *,
    prepared_blocks: list[reference.SearchBranch] | None = None,
    candidate_batch_size: int = 256,
    threads: int | None = None,
) -> dict:
    """Solve one best BLS phase per block, then fit bounded global clocks.

    This is the single-discovery-block counterpart to the member-anchor
    prefit.  It evaluates each candidate's existing period once per complete
    temporal block, enumerates every integer cycle count permitted by the
    unchanged family bounds, and batch-validates the resulting clocks.  A
    failure never changes the candidate and always falls through to its full
    dense period grid.
    """

    if not candidates:
        return {
            "candidate_count": 0,
            "phase_bls_period_trials": 0,
            "seed_trials": 0,
            "rescued_candidates": 0,
        }
    from . import exact_bls_acceleration

    prepared = prepared_blocks or _prepare_blocks(blocks, branch)
    prepared_exact = [
        exact_bls_acceleration.prepare_exact_bls(local.time, local.flux)
        for local in prepared
    ]
    anchors_by_candidate: list[list[tuple[float, float, int]]] = [
        [] for _ in candidates
    ]
    grouped: dict[float, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        duration_days = float(
            candidate.get("balanced_family_duration_hours")
            or candidate["duration_hours"]
        ) / 24.0
        grouped[duration_days].append(index)

    phase_bls_period_trials = 0
    backend_counts: dict[str, int] = {}
    for duration_days, indices in grouped.items():
        periods = np.asarray(
            [float(candidates[index]["period_days"]) for index in indices],
            dtype=np.float64,
        )
        for block_index, (local, exact_cache) in enumerate(
            zip(prepared, prepared_exact), start=1
        ):
            solved = exact_bls_acceleration.exact_bls_solutions(
                local.time,
                local.flux,
                periods=periods,
                durations=np.asarray([duration_days], dtype=np.float64),
                oversample=10,
                threads=threads,
                prepared=exact_cache,
            )
            if solved is None:
                power = BoxLeastSquares(local.time, local.flux).power(
                    periods,
                    duration_days,
                    objective="snr",
                    oversample=10,
                )
                epochs = np.asarray(power.transit_time, dtype=np.float64)
                strengths = np.asarray(power.power, dtype=np.float64)
                backend = "astropy"
            else:
                epochs = np.asarray(solved["epoch_btjd"], dtype=np.float64)
                strengths = np.asarray(solved["power"], dtype=np.float64)
                backend = str(solved.get("backend") or "native")
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
            midpoint = float(np.median(local.time))
            for local_index, candidate_index in enumerate(indices):
                period = float(periods[local_index])
                epoch = float(epochs[local_index])
                anchor = epoch + round((midpoint - epoch) / period) * period
                anchors_by_candidate[candidate_index].append(
                    (
                        float(anchor),
                        max(1.0, float(strengths[local_index])),
                        block_index,
                    )
                )
            phase_bls_period_trials += periods.size

    batch_size = max(1, int(candidate_batch_size))
    total_seed_trials = 0
    rescued = 0
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        flattened: list[dict] = []
        owners: list[int] = []
        for local_index, candidate in enumerate(batch):
            trials = _phase_anchor_seed_trials(
                candidate,
                anchors_by_candidate[start + local_index],
            )
            total_seed_trials += len(trials)
            flattened.extend(trials)
            owners.extend([local_index] * len(trials))
        if not flattened:
            continue
        measured = _measure_prepared_blocks_batch(flattened, blocks, prepared)
        best_by_owner: dict[int, tuple[tuple, int, dict]] = {}
        for flat_index, (owner, measurements) in enumerate(zip(owners, measured)):
            verification = reference._validation_decision(measurements)
            if verification.get("passed") is not True:
                continue
            score = _linear_seed_score(measurements)
            previous = best_by_owner.get(owner)
            if previous is None or score > previous[0]:
                best_by_owner[owner] = (score, flat_index, verification)
        for owner, (score, flat_index, verification) in best_by_owner.items():
            candidate = batch[owner]
            selected = flattened[flat_index]
            candidate["period_days"] = float(selected["period_days"])
            candidate["epoch_btjd"] = float(selected["epoch_btjd"])
            candidate["duration_hours"] = float(selected["duration_hours"])
            verification["event_repeatability_role"] = (
                "deferred_until_block_phase_seed_selection"
            )
            candidate["holdout_verification"] = verification
            candidate["ephemeris_balance"] = {
                "method": "block_phase_bls_then_bounded_linear_clock",
                "selected_seed_method": selected["linear_seed_method"],
                "selected_period_days": float(selected["period_days"]),
                "selected_epoch_btjd": float(selected["epoch_btjd"]),
                "weakest_fold_snr": float(score[1]),
                "full_grid_fallback_on_failure": True,
                "used_catalog_period": False,
            }
            rescued += 1
    return {
        "candidate_count": len(candidates),
        "phase_bls_period_trials": int(phase_bls_period_trials),
        "seed_trials": int(total_seed_trials),
        "rescued_candidates": int(rescued),
        "duration_groups": len(grouped),
        "backend_counts": backend_counts,
        "failure_policy": "unchanged_dense_grid_fallback",
    }


def _evaluate_family_grid_batch(
    specs: list[dict],
    period_grids: list[np.ndarray],
    *,
    bls: BoxLeastSquares,
    blocks: list[list[int]],
    prepared_blocks: list[reference.SearchBranch],
    solution_provider: Callable[..., Mapping] | None = None,
    power_provider: Callable[..., np.ndarray] | None = None,
) -> list[dict]:
    """Evaluate many independent grids in one unchanged BLS request.

    Every requested period remains independent.  Concatenating grids that
    share a duration removes repeated Python/setup overhead and lets an
    alternate exact backend tile the immutable cadence pass; splitting the
    returned arrays at the recorded offsets reconstructs the exact per-family
    result arrays.  Astropy itself still scans the cadences once per period.
    """

    if not specs:
        return []
    duration_days = float(specs[0]["duration_days"])
    if any(float(spec["duration_days"]) != duration_days for spec in specs):
        raise ValueError("one refit batch must share an exact transit duration")
    sizes = np.asarray([grid.size for grid in period_grids], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
    combined_periods = np.concatenate(period_grids)
    combined_epochs: np.ndarray | None
    if power_provider is not None:
        combined_power = np.asarray(
            power_provider(
                periods=combined_periods,
                durations=np.asarray([duration_days], dtype=float),
            ),
            dtype=float,
        )
        combined_epochs = None
        if combined_power.shape != combined_periods.shape:
            raise ValueError("power_provider must return one value per period")
    elif solution_provider is None:
        power = bls.power(
            combined_periods,
            duration_days,
            objective="snr",
            oversample=10,
        )
        combined_power = np.asarray(power.power, dtype=float)
        combined_epochs = np.asarray(power.transit_time, dtype=float)
    else:
        solved = solution_provider(
            periods=combined_periods,
            durations=np.asarray([duration_days], dtype=float),
        )
        combined_power = np.asarray(solved["power"], dtype=float)
        combined_epochs = np.asarray(solved["epoch_btjd"], dtype=float)
        if not (
            combined_power.shape == combined_periods.shape
            and combined_epochs.shape == combined_periods.shape
        ):
            raise ValueError("solution_provider arrays must align with periods")

    selected_by_family: list[list[int]] = []
    for family_index, (spec, period_grid) in enumerate(zip(specs, period_grids)):
        start = int(offsets[family_index])
        stop = int(offsets[family_index + 1])
        power_values = combined_power[start:stop]
        if period_grid.size <= 401:
            selected_indices = set(range(period_grid.size))
        else:
            selected_indices = set(
                np.linspace(0, period_grid.size - 1, 401, dtype=int).tolist()
            )
        selected_indices.update(
            np.argsort(np.nan_to_num(power_values, nan=-np.inf))[-100:].tolist()
        )
        ordered_indices = sorted(selected_indices)
        selected_by_family.append(ordered_indices)

    selected_flat_indices = [
        int(offsets[family_index]) + int(index)
        for family_index, ordered_indices in enumerate(selected_by_family)
        for index in ordered_indices
    ]
    selected_epochs: np.ndarray | None = None
    if combined_epochs is None:
        if solution_provider is None:
            raise ValueError("power_provider requires a solution_provider")
        solved = solution_provider(
            periods=combined_periods[selected_flat_indices],
            durations=np.asarray([duration_days], dtype=float),
        )
        solved_power = np.asarray(solved["power"], dtype=float)
        selected_epochs = np.asarray(solved["epoch_btjd"], dtype=float)
        expected = (len(selected_flat_indices),)
        if not (
            solved_power.shape == expected
            and selected_epochs.shape == expected
        ):
            raise ValueError("solution_provider arrays must align with trials")
        combined_power[np.asarray(selected_flat_indices, dtype=int)] = solved_power

    flat_trials: list[dict] = []
    selected_epoch_offset = 0
    for family_index, (spec, period_grid, ordered_indices) in enumerate(
        zip(specs, period_grids, selected_by_family)
    ):
        start = int(offsets[family_index])
        candidate = spec["candidate"]
        for index in ordered_indices:
            combined_index = start + int(index)
            epoch = (
                float(combined_epochs[combined_index])
                if combined_epochs is not None
                else float(selected_epochs[selected_epoch_offset])
            )
            selected_epoch_offset += int(combined_epochs is None)
            # Fixed-ephemeris measurement needs only these identity fields.
            # Avoid copying the large nested vote audit into every trial.
            flat_trials.append(
                {
                    "period_days": float(period_grid[index]),
                    "epoch_btjd": epoch,
                    "duration_hours": float(spec["duration_hours"]),
                    "discovered_blocks": candidate.get("discovered_blocks", []),
                    "source_block": candidate.get("source_block"),
                }
            )

    measurements_by_trial = _measure_prepared_blocks_batch(
        flat_trials, blocks, prepared_blocks
    )
    results: list[dict] = []
    trial_offset = 0
    for family_index, (spec, period_grid, ordered_indices) in enumerate(
        zip(specs, period_grids, selected_by_family)
    ):
        start = int(offsets[family_index])
        stop = int(offsets[family_index + 1])
        power_values = combined_power[start:stop]
        family_trial_start = trial_offset
        family_measurements = measurements_by_trial[
            family_trial_start : family_trial_start + len(ordered_indices)
        ]
        trial_offset += len(ordered_indices)
        best_row: dict | None = None
        best_score: tuple | None = None
        for local_position, (index, measurements) in enumerate(
            zip(ordered_indices, family_measurements)
        ):
            trial_period = float(period_grid[index])
            trial_epoch = float(
                flat_trials[family_trial_start + local_position]["epoch_btjd"]
            )
            covered = [
                row for row in measurements if row["outcome"] != "untestable"
            ]
            signed_snrs = [
                (
                    float(row.get("snr") or 0.0)
                    if float(row.get("depth_ppm") or 0.0) > 0
                    else -abs(float(row.get("snr") or 0.0))
                )
                for row in covered
            ]
            score = (
                len([value for value in signed_snrs if value > 0]),
                min(signed_snrs, default=-np.inf),
                sum(signed_snrs),
                float(power_values[index]),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_row = {
                    "period_days": trial_period,
                    "epoch_btjd": trial_epoch,
                    "measurements": measurements,
                    "score": score,
                }
        assert best_row is not None and best_score is not None
        results.append(best_row)
    assert trial_offset == len(measurements_by_trial)
    return results


def _selected_power_indices(power_values: np.ndarray) -> list[int]:
    """Keep the unchanged dense coverage plus strongest-period policy."""

    if power_values.size <= 401:
        selected = set(range(power_values.size))
    else:
        selected = set(
            np.linspace(0, power_values.size - 1, 401, dtype=int).tolist()
        )
    selected.update(
        np.argsort(np.nan_to_num(power_values, nan=-np.inf))[-100:].tolist()
    )
    return sorted(selected)


def _materialize_shared_period_cluster(specs: list[dict]) -> dict:
    """Build the exact union of every family's original period lattice.

    The shared kernel may evaluate duplicate period points once, but a family
    is allowed to rank only the points that were present in its own original
    grid.  This membership rule makes sharing an execution optimization only:
    it cannot widen, densify, shift, or otherwise change a scientific search.
    """

    if not specs:
        raise ValueError("a shared period cluster must contain at least one family")
    duration_days = float(specs[0]["duration_days"])
    if any(float(spec["duration_days"]) != duration_days for spec in specs):
        raise ValueError("a shared period cluster must have one exact duration")
    period_grids = [np.asarray(spec["periods"], dtype=float) for spec in specs]
    periods = np.unique(np.concatenate(period_grids))
    memberships: list[np.ndarray] = []
    for period_grid in period_grids:
        membership = np.searchsorted(periods, period_grid).astype(np.int64)
        if not np.array_equal(periods[membership], period_grid):
            raise RuntimeError("shared period union lost an original grid point")
        memberships.append(membership)
    return {
        "duration_days": duration_days,
        "specs": specs,
        "periods": periods,
        "memberships": memberships,
        "individual_points": int(sum(spec["periods"].size for spec in specs)),
    }


def build_shared_period_clusters(
    specs: list[dict],
    *,
    max_candidates: int = 16,
    max_grid_points: int = 65_536,
) -> list[dict]:
    """Group overlapping windows for exact-union batched evaluation."""

    if not specs:
        return []
    max_candidates = max(1, int(max_candidates))
    max_grid_points = max(2, int(max_grid_points))
    by_duration: dict[float, list[dict]] = defaultdict(list)
    for spec in specs:
        by_duration[float(spec["duration_days"])].append(spec)

    output: list[dict] = []
    for duration_specs in by_duration.values():
        ordered = sorted(duration_specs, key=lambda row: (row["lower"], row["upper"]))
        current: list[dict] = []
        for spec in ordered:
            if not current:
                current = [spec]
                continue
            overlaps = float(spec["lower"]) <= max(
                float(row["upper"]) for row in current
            )
            proposal = current + [spec]
            shared = _materialize_shared_period_cluster(proposal)
            saves_work = shared["periods"].size <= shared["individual_points"]
            bounded = (
                len(proposal) <= max_candidates
                and shared["periods"].size <= max_grid_points
            )
            if overlaps and saves_work and bounded:
                current = proposal
            else:
                output.append(_materialize_shared_period_cluster(current))
                current = [spec]
        if current:
            output.append(_materialize_shared_period_cluster(current))
    return output


def _evaluate_shared_coarse_cluster(
    cluster: dict,
    *,
    bls: BoxLeastSquares,
    blocks: list[list[int]],
    prepared_blocks: list[reference.SearchBranch],
    select_all: bool = False,
    solution_provider: Callable[..., Mapping] | None = None,
    power_provider: Callable[..., np.ndarray] | None = None,
) -> list[dict]:
    """Evaluate overlapping family windows with one shared Astropy BLS call."""

    specs = cluster["specs"]
    periods = cluster["periods"]
    epochs: np.ndarray | None
    if power_provider is not None:
        power_values = np.asarray(
            power_provider(
                periods=periods,
                durations=np.asarray(
                    [float(cluster["duration_days"])], dtype=float
                ),
            ),
            dtype=float,
        )
        epochs = None
        if power_values.shape != periods.shape:
            raise ValueError("power_provider must return one value per period")
    elif solution_provider is None:
        power = bls.power(
            periods,
            float(cluster["duration_days"]),
            objective="snr",
            oversample=10,
        )
        power_values = np.asarray(power.power, dtype=float)
        epochs = np.asarray(power.transit_time, dtype=float)
    else:
        solved = solution_provider(
            periods=periods,
            durations=np.asarray(
                [float(cluster["duration_days"])], dtype=float
            ),
        )
        power_values = np.asarray(solved["power"], dtype=float)
        epochs = np.asarray(solved["epoch_btjd"], dtype=float)
        if not (
            power_values.shape == periods.shape
            and epochs.shape == periods.shape
        ):
            raise ValueError("solution_provider arrays must align with periods")
    selected_by_family: list[list[int]] = []
    for spec, membership in zip(specs, cluster["memberships"]):
        local_power = power_values[membership]
        if select_all:
            local_indices = list(range(membership.size))
        else:
            local_indices = _selected_power_indices(local_power)
        selected = [int(membership[index]) for index in local_indices]
        selected_by_family.append(selected)
    unique_indices = sorted(
        {index for selected in selected_by_family for index in selected}
    )
    if epochs is None:
        if solution_provider is None:
            raise ValueError("power_provider requires a solution_provider")
        solved = solution_provider(
            periods=periods[unique_indices],
            durations=np.asarray(
                [float(cluster["duration_days"])], dtype=float
            ),
        )
        solved_power = np.asarray(solved["power"], dtype=float)
        solved_epochs = np.asarray(solved["epoch_btjd"], dtype=float)
        expected = (len(unique_indices),)
        if not (
            solved_power.shape == expected
            and solved_epochs.shape == expected
        ):
            raise ValueError("solution_provider arrays must align with trials")
        epochs = np.full(periods.shape, np.nan, dtype=float)
        selected_array = np.asarray(unique_indices, dtype=int)
        epochs[selected_array] = solved_epochs
        power_values[selected_array] = solved_power
    compact_trials = [
        {
            "period_days": float(periods[index]),
            "epoch_btjd": float(epochs[index]),
            "duration_hours": float(specs[0]["duration_hours"]),
        }
        for index in unique_indices
    ]
    raw_by_index: dict[int, list[tuple[int, list[int], dict]]] = {
        index: [] for index in unique_indices
    }
    provider = _FIXED_EPHEMERIS_PROVIDER.get()
    if provider is not None:
        # The shared evaluator historically bypassed the alternate provider
        # and forced every selected trial back through the CPU.  Explosion
        # CUDA keeps the exact same trial list and block dictionaries, but
        # lets the configured locked-ephemeris backend measure the complete
        # compact batch.  Candidate-specific discovery flags and outcomes are
        # deliberately recomputed below, exactly as in the CPU path.
        measured_trials = provider(compact_trials, blocks, prepared_blocks)
        if len(measured_trials) != len(compact_trials):
            raise RuntimeError(
                "fixed-ephemeris provider changed the shared trial count"
            )
        for index, measurements in zip(unique_indices, measured_trials):
            if len(measurements) != len(blocks):
                raise RuntimeError(
                    "fixed-ephemeris provider changed the block count"
                )
            for fallback_block, row in enumerate(measurements, start=1):
                block_index = int(row.get("block") or fallback_block)
                segments = list(
                    row.get("segments") or blocks[block_index - 1]
                )
                metrics = {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "block",
                        "segments",
                        "independently_discovered",
                        "outcome",
                    }
                }
                raw_by_index[index].append(
                    (block_index, segments, metrics)
                )
    else:
        for block_index, (segments, local) in enumerate(
            zip(blocks, prepared_blocks), start=1
        ):
            raw_rows = _measure_fixed_ephemerides_exact(
                local.time, local.flux, compact_trials
            )
            for index, metrics in zip(unique_indices, raw_rows):
                raw_by_index[index].append(
                    (block_index, list(segments), metrics)
                )
    cluster["individual_validation_trials"] = int(
        sum(len(selected) for selected in selected_by_family)
    )
    cluster["shared_validation_trials"] = int(len(unique_indices))

    results: list[dict] = []
    for spec, selected in zip(specs, selected_by_family):
        candidate = spec["candidate"]
        discovered_blocks = {
            int(value) for value in candidate.get("discovered_blocks", [])
        }
        if candidate.get("source_block") is not None:
            discovered_blocks.add(int(candidate["source_block"]))
        best_row = None
        best_score = None
        for index in selected:
            measurements = []
            for block_index, segments, raw_metrics in raw_by_index[index]:
                metrics = dict(raw_metrics)
                independently_discovered = block_index in discovered_blocks
                metrics["timing_offset_days"] = 0.0
                metrics["timing_tolerance_applied"] = False
                measurements.append(
                    {
                        "block": block_index,
                        "segments": segments,
                        "independently_discovered": independently_discovered,
                        "outcome": reference._metric_outcome(
                            metrics,
                            independently_discovered=independently_discovered,
                        ),
                        **metrics,
                    }
                )
            covered = [
                row for row in measurements if row["outcome"] != "untestable"
            ]
            signed_snrs = [
                (
                    float(row.get("snr") or 0.0)
                    if float(row.get("depth_ppm") or 0.0) > 0
                    else -abs(float(row.get("snr") or 0.0))
                )
                for row in covered
            ]
            score = (
                len([value for value in signed_snrs if value > 0]),
                min(signed_snrs, default=-np.inf),
                sum(signed_snrs),
                float(power_values[index]),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_row = {
                    "period_days": float(periods[index]),
                    "epoch_btjd": float(epochs[index]),
                    "measurements": measurements,
                    "score": score,
                }
        if best_row is None:
            raise RuntimeError("shared period cluster produced no family trial")
        results.append(best_row)
    return results


def balance_family_ephemerides_shared(
    candidates: list[dict],
    blocks: list[list[int]],
    branch: reference.SearchBranch,
    *,
    prepared_blocks: list[reference.SearchBranch] | None = None,
    bls: BoxLeastSquares | None = None,
    baseline_days: float | None = None,
    max_candidates: int = 16,
    max_grid_points: int = 65_536,
    solution_provider: Callable[..., Mapping] | None = None,
    power_provider: Callable[..., np.ndarray] | None = None,
    power_prefetch: Callable[[list[tuple[np.ndarray, np.ndarray]]], None]
    | None = None,
) -> dict:
    """Refit families on exact unions of their original period grids.

    Coarse and fine grids both retain a per-family membership index.  The
    kernel may evaluate an exactly duplicated period once, but each family
    ranks only its own original points with the unchanged maximin score.
    """

    specs = [
        spec
        for candidate in candidates
        if (
            spec := _family_balance_spec(
                candidate, branch, baseline_days=baseline_days
            )
        )
        is not None
    ]
    if not specs:
        return {
            "candidate_count": 0,
            "cluster_count": 0,
            "individual_coarse_points": 0,
            "shared_coarse_points": 0,
            "coarse_point_reuse": 1.0,
        }
    prepared = prepared_blocks or _prepare_blocks(blocks, branch)
    bls = bls or BoxLeastSquares(branch.time, branch.flux)
    clusters = build_shared_period_clusters(
        specs,
        max_candidates=max_candidates,
        max_grid_points=max_grid_points,
    )
    individual_points = sum(cluster["individual_points"] for cluster in clusters)
    shared_points = sum(cluster["periods"].size for cluster in clusters)
    individual_validation_trials = 0
    shared_validation_trials = 0
    fine_individual_points = 0
    fine_shared_points = 0
    fine_individual_validation_trials = 0
    fine_shared_validation_trials = 0

    if power_prefetch is not None:
        power_prefetch(
            [
                (
                    np.asarray(cluster["periods"], dtype=float),
                    np.asarray([float(cluster["duration_days"])], dtype=float),
                )
                for cluster in clusters
            ]
        )

    cluster_states: list[tuple[dict, list[dict], list[dict]]] = []
    for cluster in clusters:
        coarse_rows = _evaluate_shared_coarse_cluster(
            cluster,
            bls=bls,
            blocks=blocks,
            prepared_blocks=prepared,
            solution_provider=solution_provider,
            power_provider=power_provider,
        )
        individual_validation_trials += int(
            cluster["individual_validation_trials"]
        )
        shared_validation_trials += int(cluster["shared_validation_trials"])
        fine_specs: list[dict] = []
        for fine_index, (spec, coarse) in enumerate(
            zip(cluster["specs"], coarse_rows)
        ):
            original_grid = np.asarray(spec["periods"], dtype=float)
            coarse_step = float(original_grid[1] - original_grid[0])
            fine_periods = np.linspace(
                max(float(spec["lower"]), float(coarse["period_days"]) - coarse_step),
                min(float(spec["upper"]), float(coarse["period_days"]) + coarse_step),
                101,
            )
            fine_specs.append(
                {
                    **spec,
                    "lower": float(fine_periods[0]),
                    "upper": float(fine_periods[-1]),
                    "periods": fine_periods,
                    "grid_size": 101,
                    "shared_order_index": fine_index,
                }
            )
        fine_clusters = build_shared_period_clusters(
            fine_specs,
            max_candidates=max_candidates,
            max_grid_points=max_grid_points,
        )
        cluster_states.append((cluster, fine_specs, fine_clusters))

    all_fine_clusters = [
        fine_cluster
        for _cluster, _fine_specs, fine_clusters in cluster_states
        for fine_cluster in fine_clusters
    ]
    if power_prefetch is not None:
        power_prefetch(
            [
                (
                    np.asarray(cluster["periods"], dtype=float),
                    np.asarray([float(cluster["duration_days"])], dtype=float),
                )
                for cluster in all_fine_clusters
            ]
        )

    for cluster, fine_specs, fine_clusters in cluster_states:
        fine_rows_by_index: dict[int, dict] = {}
        for fine_cluster in fine_clusters:
            fine_rows = _evaluate_shared_coarse_cluster(
                fine_cluster,
                bls=bls,
                blocks=blocks,
                prepared_blocks=prepared,
                select_all=True,
                solution_provider=solution_provider,
                power_provider=power_provider,
            )
            fine_individual_points += int(fine_cluster["individual_points"])
            fine_shared_points += int(fine_cluster["periods"].size)
            fine_individual_validation_trials += int(
                fine_cluster["individual_validation_trials"]
            )
            fine_shared_validation_trials += int(
                fine_cluster["shared_validation_trials"]
            )
            for fine_spec, fine_row in zip(fine_cluster["specs"], fine_rows):
                fine_rows_by_index[int(fine_spec["shared_order_index"])] = fine_row
        best_rows = [fine_rows_by_index[index] for index in range(len(fine_specs))]
        for spec, best in zip(cluster["specs"], best_rows):
            candidate = spec["candidate"]
            score = best["score"]
            candidate["period_days"] = float(best["period_days"])
            candidate["epoch_btjd"] = float(best["epoch_btjd"])
            candidate["duration_hours"] = float(spec["duration_hours"])
            candidate["holdout_verification"] = reference._validation_decision(
                best["measurements"]
            )
            candidate["ephemeris_balance"] = {
                "method": "shared_global_bls_then_maximise_weakest_fold_snr",
                "period_bounds_days": [float(spec["lower"]), float(spec["upper"])],
                "coarse_grid_size": int(spec["grid_size"]),
                "fine_grid_size": 101,
                "original_period_days": float(spec["original_period_days"]),
                "original_epoch_btjd": float(spec["original_epoch_btjd"]),
                "selected_period_days": float(candidate["period_days"]),
                "selected_epoch_btjd": float(candidate["epoch_btjd"]),
                "weakest_fold_snr": float(score[1]),
                "used_catalog_period": False,
                "no_coarser_shared_period_grid": True,
                "original_period_grid_membership_preserved": True,
            }
    return {
        "candidate_count": len(specs),
        "cluster_count": len(clusters),
        "individual_coarse_points": int(individual_points),
        "shared_coarse_points": int(shared_points),
        "coarse_point_reuse": float(individual_points / max(1, shared_points)),
        "individual_coarse_validation_trials": int(individual_validation_trials),
        "shared_coarse_validation_trials": int(shared_validation_trials),
        "coarse_validation_reuse": float(
            individual_validation_trials / max(1, shared_validation_trials)
        ),
        "individual_fine_points": int(fine_individual_points),
        "shared_fine_points": int(fine_shared_points),
        "fine_point_reuse": float(fine_individual_points / max(1, fine_shared_points)),
        "individual_fine_validation_trials": int(
            fine_individual_validation_trials
        ),
        "shared_fine_validation_trials": int(fine_shared_validation_trials),
        "fine_validation_reuse": float(
            fine_individual_validation_trials
            / max(1, fine_shared_validation_trials)
        ),
        "original_period_grid_membership_preserved": True,
    }


def balance_family_ephemerides_vectorized(
    candidates: list[dict],
    blocks: list[list[int]],
    branch: reference.SearchBranch,
    *,
    attach_event_repeatability: bool = True,
    batch_size: int | None = None,
    progress=None,
    prepared_blocks: list[reference.SearchBranch] | None = None,
    bls: BoxLeastSquares | None = None,
    baseline_days: float | None = None,
    solution_provider: Callable[..., Mapping] | None = None,
    power_provider: Callable[..., np.ndarray] | None = None,
) -> None:
    """Balance many families without changing any per-family calculation.

    Families are grouped only by their exact transit duration.  Each bounded
    batch concatenates the original coarse or fine period arrays, evaluates
    them with the unchanged Astropy kernel, then splits the results before the
    original lexicographic maximin score is applied.
    """

    if batch_size is None:
        batch_size = int(os.environ.get("HOUXING_REFIT_BATCH_SIZE", "64"))
    batch_size = max(1, int(batch_size))
    specs = [
        spec
        for candidate in candidates
        if (
            spec := _family_balance_spec(
                candidate, branch, baseline_days=baseline_days
            )
        )
        is not None
    ]
    if not specs:
        return

    grouped: dict[float, list[dict]] = defaultdict(list)
    for spec in specs:
        grouped[float(spec["duration_days"])].append(spec)
    prepared = prepared_blocks or _prepare_blocks(blocks, branch)
    bls = bls or BoxLeastSquares(branch.time, branch.flux)
    completed = 0
    for duration_specs in grouped.values():
        for batch_start in range(0, len(duration_specs), batch_size):
            batch = duration_specs[batch_start : batch_start + batch_size]
            coarse_grids = [spec["periods"] for spec in batch]
            coarse_rows = _evaluate_family_grid_batch(
                batch,
                coarse_grids,
                bls=bls,
                blocks=blocks,
                prepared_blocks=prepared,
                solution_provider=solution_provider,
                power_provider=power_provider,
            )
            fine_grids: list[np.ndarray] = []
            for spec, coarse in zip(batch, coarse_rows):
                periods = spec["periods"]
                coarse_step = float(periods[1] - periods[0])
                fine_grids.append(
                    np.linspace(
                        max(
                            float(spec["lower"]),
                            float(coarse["period_days"]) - coarse_step,
                        ),
                        min(
                            float(spec["upper"]),
                            float(coarse["period_days"]) + coarse_step,
                        ),
                        101,
                    )
                )
            best_rows = _evaluate_family_grid_batch(
                batch,
                fine_grids,
                bls=bls,
                blocks=blocks,
                prepared_blocks=prepared,
                solution_provider=solution_provider,
                power_provider=power_provider,
            )
            for spec, best in zip(batch, best_rows):
                candidate = spec["candidate"]
                score = best["score"]
                candidate["period_days"] = float(best["period_days"])
                candidate["epoch_btjd"] = float(best["epoch_btjd"])
                candidate["duration_hours"] = float(spec["duration_hours"])
                candidate["holdout_verification"] = reference._validation_decision(
                    best["measurements"]
                )
                if attach_event_repeatability:
                    reference._attach_event_repeatability(
                        candidate, candidate["holdout_verification"], branch
                    )
                ephemeris_balance = {
                    "method": "global_bls_then_maximise_weakest_fold_snr",
                    "period_bounds_days": [
                        float(spec["lower"]),
                        float(spec["upper"]),
                    ],
                    "coarse_grid_size": int(spec["grid_size"]),
                    "fine_grid_size": 101,
                    "original_period_days": float(spec["original_period_days"]),
                    "original_epoch_btjd": float(spec["original_epoch_btjd"]),
                    "selected_period_days": float(candidate["period_days"]),
                    "selected_epoch_btjd": float(candidate["epoch_btjd"]),
                    "weakest_fold_snr": float(score[1]),
                    "used_catalog_period": False,
                }
                if not attach_event_repeatability:
                    ephemeris_balance["event_repeatability_deferred"] = True
                candidate["ephemeris_balance"] = ephemeris_balance
            completed += len(batch)
            if progress is not None:
                progress(completed, len(specs))


def phase_shift_null_vectorized(
    candidate: Mapping,
    views: Mapping[int, Mapping],
    noise_models: Mapping[int, Mapping],
    observed_events: int,
    *,
    trials: int,
) -> dict:
    """Exact compiled form of the ordinary V3 phase-shift null test."""

    period = float(candidate["period_days"])
    epoch = float(candidate["epoch_btjd"])
    duration_days = float(candidate["duration_hours"]) / 24.0
    rng = np.random.default_rng(detection_significance._seed(candidate))
    offsets = (
        np.arange(trials, dtype=float) + rng.random(trials)
    ) * period / trials
    trial_epochs = epoch + offsets
    minimum_events = max(2, int(math.floor(0.8 * observed_events)))
    usable = {
        int(segment): np.asarray(model["usable"], dtype=bool)
        for segment, model in noise_models.items()
    }
    compiled = phase_trial_events_compiled(
        views,
        noise_models,
        usable,
        period=period,
        trial_epochs=trial_epochs,
        duration_days=duration_days,
    )
    if compiled is None:
        return _ORIGINAL_PHASE_SHIFT_NULL(
            candidate,
            views,
            noise_models,
            observed_events,
            trials=trials,
        )

    depth_ppm, sigma_ppm, event_counts = compiled
    values: list[float] = []
    for trial_index, event_count in enumerate(event_counts):
        if int(event_count) < minimum_events:
            continue
        events = [
            {
                "depth_ppm": float(depth_ppm[trial_index, event_index]),
                "duration_noise_ppm": float(
                    sigma_ppm[trial_index, event_index]
                ),
            }
            for event_index in range(int(event_count))
        ]
        combined = detection_significance._combine_events(events)
        if combined["mes_like"] is not None:
            values.append(float(combined["mes_like"]))
    return {
        "requested_trials": int(trials),
        "valid_trials": len(values),
        "minimum_events_per_trial": int(minimum_events),
        "values": np.asarray(values, dtype=float),
    }


def install() -> None:
    """Install V3 plus exact batch/vectorized replacements in this process."""

    optimization_v3.install()
    reference._validate_raw_ledger = validate_raw_ledger_vectorized
    reference._validate_proposals = validate_proposals_vectorized
    reference._balance_family_ephemeris = balance_family_ephemeris_vectorized
    detection_significance._phase_shift_null = phase_shift_null_vectorized


__all__ = [
    "balance_family_ephemeris_vectorized",
    "install",
    "phase_shift_null_vectorized",
    "validate_proposals_vectorized",
    "validate_raw_ledger_vectorized",
]
