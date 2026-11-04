"""Event-first transit search inspired by the published QuickFind description.

LcTools itself is not published as source code.  This module is an independent
implementation of the *ideas* described by Schmitt & Vanderburg (2021):

1. detect significant individual dips;
2. group dips with similar shapes into periodic, double, and single events;
3. retain observed-minus-calculated timing offsets (TTV diagnostics).

It is a v0.3 candidate generator, not a replacement for BLS and not a planet
validator.  BLS remains much better for very shallow signals that only become
visible after many transits are folded together.  This event-first pass is aimed
at medium/large, long-period, sparse, or mildly non-periodic signals.
"""

from __future__ import annotations

from array import array
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import multiprocessing
import os

import numpy as np
from scipy.signal import find_peaks


THREAD_ENV = "HOUXING_EVENT_THREADS"
PROCESS_ENV = "HOUXING_EVENT_PROCESSES"


@dataclass(frozen=True)
class EventSearchConfig:
    """Transparent settings for the experimental event-first pass."""

    durations_hours: tuple[float, ...] = (
        1.0,
        1.5,
        2.0,
        3.0,
        4.5,
        6.0,
        9.0,
        12.0,
        18.0,
        24.0,
    )
    min_event_snr: float = 6.0
    gap_days: float = 0.25
    max_events: int | None = 160
    min_period_days: float = 0.5
    max_period_days: float = 40.0
    min_periodic_instances: int = 3
    max_periodic_candidates: int | None = 20
    max_seed_bins: int | None = 2000
    max_cycle_divisor: int = 24
    shape_ratio_limit: float = 3.0
    ttv_tolerance_hours: float = 2.0
    materialize_double_candidates: bool = True
    high_activity_event_threshold: int | None = 400
    high_activity_min_periodic_instances: int = 5
    high_activity_recurrence_sigma: float = 4.0


@dataclass(frozen=True)
class DipEvent:
    index: int
    center_btjd: float
    duration_hours: float
    depth_ppm: float
    snr: float
    segment: int
    points: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PeriodicEventCandidate:
    period_days: float
    epoch_btjd: float
    instances: int
    expected_instances: int
    coverage_fraction: float
    duration_hours: float
    depth_ppm: float
    median_event_snr: float
    score: float
    mean_abs_ttv_minutes: float
    max_abs_ttv_minutes: float
    ttv_minutes: tuple[float, ...]
    matched_event_indices: tuple[int, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EventFirstResult:
    periodic: tuple[PeriodicEventCandidate, ...]
    doubles: tuple[dict, ...]
    singles: tuple[DipEvent, ...]
    all_events: tuple[DipEvent, ...]
    double_candidate_count: int = 0
    double_candidates_materialized: bool = True
    periodic_minimum_instances: int = 3
    sparse_trains_reconstructible_from_event_ledger: bool = False
    adaptive_recurrence_sigma: float | None = None
    method: str = "independent-event-first-v0.3-experimental"
    limitations: tuple[str, ...] = field(
        default=(
            "Not LcTools source code and not a byte-for-byte QuickFind clone.",
            "Single and double events are candidates only; their periods are not established.",
            "Very shallow signals should still be searched with BLS/TLS.",
        )
    )

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "limitations": list(self.limitations),
            "periodic": [item.to_dict() for item in self.periodic],
            "doubles": list(self.doubles),
            "singles": [item.to_dict() for item in self.singles],
            "all_events": [item.to_dict() for item in self.all_events],
            "periodic_minimum_instances": self.periodic_minimum_instances,
            "sparse_trains_reconstructible_from_event_ledger": (
                self.sparse_trains_reconstructible_from_event_ledger
            ),
            "adaptive_recurrence_sigma": self.adaptive_recurrence_sigma,
        }


def _mad_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    median = np.median(values)
    sigma = 1.4826 * np.median(np.abs(values - median))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.std(values)
    return float(sigma)


def _segments(time: np.ndarray, gap_days: float) -> list[np.ndarray]:
    if time.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(time) > gap_days) + 1
    return [part for part in np.split(np.arange(time.size), cuts) if part.size]


def _shape_compatible(
    duration_a: float,
    depth_a: float,
    duration_b: float,
    depth_b: float,
    ratio_limit: float,
) -> bool:
    if min(duration_a, duration_b, depth_a, depth_b) <= 0:
        return False
    duration_ratio = max(duration_a, duration_b) / min(duration_a, duration_b)
    depth_ratio = max(depth_a, depth_b) / min(depth_a, depth_b)
    return duration_ratio <= ratio_limit and depth_ratio <= ratio_limit


def _shape_compatible_mask(
    durations: np.ndarray,
    depths: np.ndarray,
    duration_reference: float,
    depth_reference: float,
    ratio_limit: float,
) -> np.ndarray:
    """Vectorized, element-for-element form of ``_shape_compatible``."""

    duration_minimum = np.minimum(durations, float(duration_reference))
    depth_minimum = np.minimum(depths, float(depth_reference))
    positive = (duration_minimum > 0.0) & (depth_minimum > 0.0)
    duration_ratio = np.divide(
        np.maximum(durations, float(duration_reference)),
        duration_minimum,
        out=np.full(durations.shape, np.inf, dtype=float),
        where=duration_minimum > 0.0,
    )
    depth_ratio = np.divide(
        np.maximum(depths, float(depth_reference)),
        depth_minimum,
        out=np.full(depths.shape, np.inf, dtype=float),
        where=depth_minimum > 0.0,
    )
    return positive & (duration_ratio <= ratio_limit) & (depth_ratio <= ratio_limit)


def detect_dip_events(
    time: np.ndarray,
    flux: np.ndarray,
    config: EventSearchConfig | None = None,
) -> list[DipEvent]:
    """Find individual downward events with a bank of box matched filters.

    The input is expected to be an already-normalized, approximately flattened
    light curve.  Gaps are processed independently so a convolution can never
    bridge two TESS sectors.
    """
    config = config or EventSearchConfig()
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    good = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[good], flux[good]
    order = np.argsort(time)
    time, flux = time[order], flux[order]
    raw: list[dict] = []

    for segment_number, indices in enumerate(_segments(time, config.gap_days)):
        if indices.size < 20:
            continue
        segment_time = time[indices]
        segment_flux = flux[indices]
        cadence = float(np.nanmedian(np.diff(segment_time)))
        if not np.isfinite(cadence) or cadence <= 0:
            continue
        baseline = float(np.nanmedian(segment_flux))
        residual = baseline - segment_flux
        point_sigma = _mad_sigma(residual)
        if not np.isfinite(point_sigma) or point_sigma <= 0:
            continue

        for duration_hours in config.durations_hours:
            duration_days = duration_hours / 24.0
            width = max(3, int(round(duration_days / cadence)))
            if width >= indices.size // 3:
                continue
            kernel = np.ones(width, dtype=float) / width
            smoothed = np.convolve(residual, kernel, mode="same")
            matched_snr = smoothed / (point_sigma / np.sqrt(width))
            edge = width // 2 + 1
            matched_snr[:edge] = -np.inf
            matched_snr[-edge:] = -np.inf
            peaks, properties = find_peaks(
                matched_snr,
                height=max(3.5, config.min_event_snr * 0.75),
                distance=max(1, width // 2),
            )
            for peak, preliminary_snr in zip(peaks, properties["peak_heights"]):
                center = float(segment_time[peak])
                distance = np.abs(segment_time - center)
                inside = distance <= 0.5 * duration_days
                nearby = (distance >= 1.25 * duration_days) & (
                    distance <= 3.0 * duration_days
                )
                if inside.sum() < 3 or nearby.sum() < 8:
                    continue
                local_sigma = _mad_sigma(segment_flux[nearby])
                if not np.isfinite(local_sigma) or local_sigma <= 0:
                    local_sigma = point_sigma
                local_baseline = float(np.nanmedian(segment_flux[nearby]))
                depth = local_baseline - float(np.nanmedian(segment_flux[inside]))
                snr = depth / (local_sigma / np.sqrt(inside.sum()))
                if depth <= 0 or snr < config.min_event_snr:
                    continue
                raw.append(
                    {
                        "center_btjd": center,
                        "duration_hours": float(duration_hours),
                        "depth_ppm": float(depth * 1e6),
                        "snr": float(max(snr, preliminary_snr)),
                        "segment": int(segment_number),
                        "points": int(inside.sum()),
                    }
                )

    # Multiple duration filters usually find the same dip.  Keep the best-fitting
    # representative and only then cap the list, so one event cannot monopolize it.
    selected: list[dict] = []
    for candidate in sorted(raw, key=lambda item: item["snr"], reverse=True):
        duplicate = False
        for kept in selected:
            separation_hours = abs(candidate["center_btjd"] - kept["center_btjd"]) * 24.0
            overlap_scale = 0.55 * max(
                candidate["duration_hours"], kept["duration_hours"]
            )
            if candidate["segment"] == kept["segment"] and separation_hours <= overlap_scale:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
        if config.max_events is not None and len(selected) >= config.max_events:
            break

    selected.sort(key=lambda item: item["center_btjd"])
    return [DipEvent(index=index, **item) for index, item in enumerate(selected)]


def _fit_period_solutions(
    events: list[DipEvent],
    trial_period: float,
    tolerance_days: float,
    shape_ratio_limit: float,
    min_instances: int = 3,
    adaptive_recurrence_sigma: float | None = None,
) -> list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray]]:
    """Fit every distinct phase/event grouping supported at ``trial_period``.

    A one-dimensional BLS periodogram and the old implementation both retained
    only the strongest phase at a period.  That is destructive when an unrelated
    deep dip and a real repeating signal share nearly the same trial period.  A
    solution is therefore identified by its matched event membership, not by
    period alone.
    """
    times = np.asarray([event.center_btjd for event in events], dtype=float)
    snrs = np.asarray([event.snr for event in events], dtype=float)
    durations = np.asarray([event.duration_hours for event in events], dtype=float)
    depths = np.asarray([event.depth_ppm for event in events], dtype=float)
    return _fit_period_solutions_prepared(
        times,
        snrs,
        durations,
        depths,
        trial_period,
        tolerance_days,
        shape_ratio_limit,
        min_instances,
        adaptive_recurrence_sigma,
    )


def _fit_period_solutions_prepared(
    times: np.ndarray,
    snrs: np.ndarray,
    durations: np.ndarray,
    depths: np.ndarray,
    trial_period: float,
    tolerance_days: float,
    shape_ratio_limit: float,
    min_instances: int,
    adaptive_recurrence_sigma: float | None = None,
) -> list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray]]:
    """Fit one seed using arrays prepared once for the entire block."""

    required_instances = int(min_instances)
    if adaptive_recurrence_sigma is not None:
        support_counts = np.zeros(times.size, dtype=int)
        for seed_index, epoch_seed in enumerate(times):
            seed_cycles = np.rint((times - epoch_seed) / trial_period).astype(int)
            seed_residual = times - (epoch_seed + seed_cycles * trial_period)
            seed_matched = np.abs(seed_residual) <= tolerance_days
            if not np.any(seed_matched):
                continue
            duration_reference = float(np.median(durations[seed_matched]))
            depth_reference = float(np.median(depths[seed_matched]))
            shape = _shape_compatible_mask(
                durations,
                depths,
                duration_reference,
                depth_reference,
                shape_ratio_limit,
            )
            matched = seed_matched & shape
            support_counts[seed_index] = int(matched.sum())
        required_instances = _adaptive_recurrence_threshold(
            support_counts,
            minimum_instances=min_instances,
            sigma=float(adaptive_recurrence_sigma),
        )

    solutions: dict[
        tuple[int, ...],
        tuple[float, tuple[float, float, np.ndarray, np.ndarray, np.ndarray]],
    ] = {}

    for epoch_seed in times:
        seed_cycles = np.rint((times - epoch_seed) / trial_period).astype(int)
        seed_residual = times - (epoch_seed + seed_cycles * trial_period)
        seed_matched = np.abs(seed_residual) <= tolerance_days
        if (
            seed_matched.sum() < required_instances
            or len(np.unique(seed_cycles[seed_matched])) < required_instances
        ):
            continue

        duration_reference = float(np.median(durations[seed_matched]))
        depth_reference = float(np.median(depths[seed_matched]))
        shape = _shape_compatible_mask(
            durations,
            depths,
            duration_reference,
            depth_reference,
            shape_ratio_limit,
        )
        matched = seed_matched & shape
        if (
            matched.sum() < required_instances
            or len(np.unique(seed_cycles[matched])) < required_instances
        ):
            continue

        x = seed_cycles[matched].astype(float)
        y = times[matched]
        weights = np.maximum(snrs[matched], 1.0)
        design = np.column_stack([np.ones_like(x), x])
        lhs = design * np.sqrt(weights)[:, None]
        rhs = y * np.sqrt(weights)
        epoch, period = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        if period <= 0:
            continue
        cycles = np.rint((times - epoch) / period).astype(int)
        residual = times - (epoch + cycles * period)
        matched = (np.abs(residual) <= tolerance_days) & shape
        if (
            matched.sum() < required_instances
            or len(np.unique(cycles[matched])) < required_instances
        ):
            continue

        membership = tuple(int(index) for index in np.flatnonzero(matched))
        residual_scale = float(np.mean(np.abs(residual[matched]))) / tolerance_days
        score = float(np.sum(snrs[matched]) / (1.0 + residual_scale))
        fitted = (float(period), float(epoch), cycles, residual, matched)
        previous = solutions.get(membership)
        if previous is None or score > previous[0]:
            solutions[membership] = (score, fitted)

    return [
        fitted
        for _score, fitted in sorted(
            solutions.values(), key=lambda item: item[0], reverse=True
        )
    ]


def _fit_period(
    events: list[DipEvent],
    trial_period: float,
    tolerance_days: float,
    shape_ratio_limit: float,
):
    """Compatibility wrapper returning the strongest phase solution."""
    solutions = _fit_period_solutions(
        events, trial_period, tolerance_days, shape_ratio_limit
    )
    return solutions[0] if solutions else None


def _event_signature(candidate: PeriodicEventCandidate) -> tuple[int, ...]:
    """Return the order-independent event identity used for exact deduplication.

    ``matched_event_indices`` is already unique, but its ordering is an
    implementation detail of the period fit.  A sorted tuple preserves the old
    set-equality semantics while making the identity hashable without rebuilding
    sets for every comparison.
    """

    return tuple(sorted(int(value) for value in candidate.matched_event_indices))


def _period_duplicate_mask(
    candidate_period: float, existing_periods: np.ndarray
) -> np.ndarray:
    """Vectorized form of the historical same/2..7 harmonic test."""

    periods = np.asarray(existing_periods, dtype=float)
    if periods.size == 0:
        return np.zeros(0, dtype=bool)
    ratios = float(candidate_period) / periods
    same = np.abs(ratios - 1.0) < 0.006
    harmonics = np.arange(2.0, 8.0, dtype=float)
    direct = (
        np.abs(ratios[:, None] - harmonics[None, :])
        / harmonics[None, :]
        < 0.003
    )
    inverse_harmonics = 1.0 / harmonics
    inverse = (
        np.abs(ratios[:, None] - inverse_harmonics[None, :])
        / inverse_harmonics[None, :]
        < 0.003
    )
    return same | np.any(direct | inverse, axis=1)


def _merge_exhaustive_candidate(
    results: list[PeriodicEventCandidate],
    indices_by_event_signature: dict[tuple[int, ...], list[int]],
    candidate: PeriodicEventCandidate,
) -> None:
    """Merge one uncapped candidate without a quadratic all-results scan.

    The historical implementation only compared periods after proving exact
    equality of the matched-event set.  Hashing that set therefore changes no
    scientific criterion.  If more than one compatible period is present, the
    highest-scoring one is selected, matching the old per-insertion global
    score sort.
    """

    signature = _event_signature(candidate)
    signature_indices = indices_by_event_signature.get(signature)
    if signature_indices:
        index_array = np.asarray(signature_indices, dtype=int)
        periods = np.fromiter(
            (results[index].period_days for index in signature_indices),
            dtype=float,
            count=len(signature_indices),
        )
        duplicate_mask = _period_duplicate_mask(candidate.period_days, periods)
        duplicate_indices = index_array[duplicate_mask]
        if duplicate_indices.size:
            duplicate_scores = np.fromiter(
                (results[int(index)].score for index in duplicate_indices),
                dtype=float,
                count=duplicate_indices.size,
            )
            duplicate_index = int(duplicate_indices[int(np.argmax(duplicate_scores))])
            if candidate.score > results[duplicate_index].score:
                results[duplicate_index] = candidate
            return

    result_index = len(results)
    results.append(candidate)
    indices_by_event_signature.setdefault(signature, []).append(result_index)


def _stable_score_order(
    candidates: list[PeriodicEventCandidate],
) -> list[PeriodicEventCandidate]:
    """Sort once in compiled NumPy code, preserving tie insertion order."""

    if len(candidates) < 2:
        return candidates
    scores = np.fromiter(
        (candidate.score for candidate in candidates),
        dtype=float,
        count=len(candidates),
    )
    order = np.argsort(-scores, kind="stable")
    return [candidates[int(index)] for index in order]


def _effective_min_periodic_instances(
    events: list[DipEvent], config: EventSearchConfig
) -> int:
    """Return the materialization threshold for an event-train candidate.

    In a high-activity light curve, enumerating every chance triplet/quartet is
    combinatorial rather than scientifically exhaustive.  The events
    themselves remain in the audit ledger, so every sparse train is still
    reconstructible.  BLS also remains uncapped and preserves every local
    period peak.  Requiring five independently detected events only in this
    high-activity regime prevents millions of random sparse combinations from
    masquerading as standalone periodic candidates.
    """

    minimum = max(3, int(config.min_periodic_instances))
    threshold = config.high_activity_event_threshold
    if (
        config.max_events is None
        and threshold is not None
        and len(events) >= int(threshold)
    ):
        minimum = max(minimum, int(config.high_activity_min_periodic_instances))
    return minimum


def _adaptive_recurrence_threshold(
    support_counts: np.ndarray,
    *,
    minimum_instances: int,
    sigma: float,
) -> int:
    """Robust empirical phase-null gate for high-activity light curves."""

    counts = np.asarray(support_counts, dtype=float)
    counts = counts[np.isfinite(counts)]
    if counts.size == 0:
        return int(minimum_instances)
    median = float(np.median(counts))
    robust_sigma = 1.4826 * float(np.median(np.abs(counts - median)))
    # Discrete support counts can have zero MAD even when several chance
    # occupancies exist.  One event is the conservative resolution floor.
    robust_sigma = max(1.0, robust_sigma)
    return max(
        int(minimum_instances),
        int(np.ceil(median + float(sigma) * robust_sigma)),
    )


def _event_search_threads() -> int:
    raw = os.environ.get(THREAD_ENV, "1")
    try:
        workers = int(raw)
    except ValueError as exc:
        raise ValueError(f"{THREAD_ENV} must be an integer, got {raw!r}") from exc
    if workers < 1:
        raise ValueError(f"{THREAD_ENV} must be at least 1")
    return workers


def _event_search_processes() -> int:
    raw = os.environ.get(PROCESS_ENV, "1")
    try:
        workers = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{PROCESS_ENV} must be an integer, got {raw!r}"
        ) from exc
    if workers < 1:
        raise ValueError(f"{PROCESS_ENV} must be at least 1")
    return workers


_EVENT_FIT_WORKER_STATE = None


def _initialize_event_fit_worker(
    times: np.ndarray,
    snrs: np.ndarray,
    durations: np.ndarray,
    depths: np.ndarray,
    tolerance_days: float,
    shape_ratio_limit: float,
    minimum_instances: int,
    adaptive_recurrence_sigma: float | None,
) -> None:
    global _EVENT_FIT_WORKER_STATE
    _EVENT_FIT_WORKER_STATE = (
        np.asarray(times, dtype=float),
        np.asarray(snrs, dtype=float),
        np.asarray(durations, dtype=float),
        np.asarray(depths, dtype=float),
        float(tolerance_days),
        float(shape_ratio_limit),
        int(minimum_instances),
        (
            None
            if adaptive_recurrence_sigma is None
            else float(adaptive_recurrence_sigma)
        ),
    )


def _fit_period_worker(trial_period: float):
    if _EVENT_FIT_WORKER_STATE is None:
        raise RuntimeError("event-fit worker was not initialized")
    times, snrs, durations, depths, tolerance, shape_limit, minimum, adaptive_sigma = (
        _EVENT_FIT_WORKER_STATE
    )
    return _fit_period_solutions_prepared(
        times,
        snrs,
        durations,
        depths,
        float(trial_period),
        tolerance,
        shape_limit,
        minimum,
        adaptive_sigma,
    )


def _build_seed_bins_vectorized(
    times: np.ndarray,
    durations: np.ndarray,
    depths: np.ndarray,
    config: EventSearchConfig,
    log_step: float,
) -> dict[int, array]:
    """Build every historical event-pair/divisor seed with NumPy broadcasting.

    The old implementation evaluated the same Cartesian product in three
    nested Python loops.  This version partitions only the left-event indices;
    each worker still evaluates every compatible right event and every integer
    divisor from 1 through ``max_cycle_divisor``.  Stable key sorting and
    ordered range merging preserve the historical value order inside each bin.
    """

    event_count = int(times.size)
    if event_count < 2:
        return {}
    workers = min(_event_search_threads(), event_count - 1)
    left_ranges = [
        indices
        for indices in np.array_split(np.arange(event_count - 1), workers)
        if indices.size
    ]
    divisors = np.arange(1, int(config.max_cycle_divisor) + 1, dtype=float)

    def build_range(left_indices: np.ndarray) -> dict[int, array]:
        local: dict[int, array] = {}
        for left_value in left_indices:
            left = int(left_value)
            right_slice = slice(left + 1, event_count)
            compatible = _shape_compatible_mask(
                durations[right_slice],
                depths[right_slice],
                float(durations[left]),
                float(depths[left]),
                config.shape_ratio_limit,
            )
            if not np.any(compatible):
                continue
            separations = times[right_slice][compatible] - times[left]
            periods = separations[:, None] / divisors[None, :]
            max_divisors = np.minimum(
                int(config.max_cycle_divisor),
                np.floor(separations / config.min_period_days).astype(int),
            )
            valid = (
                divisors[None, :] <= max_divisors[:, None]
            ) & (periods >= config.min_period_days) & (
                periods <= config.max_period_days
            )
            period_values = np.ascontiguousarray(periods[valid], dtype=np.float64)
            if period_values.size == 0:
                continue
            keys = np.rint(np.log(period_values) / log_step).astype(np.int64)
            order = np.argsort(keys, kind="stable")
            keys = keys[order]
            period_values = period_values[order]
            boundaries = np.flatnonzero(np.diff(keys)) + 1
            starts = np.concatenate(([0], boundaries))
            stops = np.concatenate((boundaries, [keys.size]))
            for start, stop in zip(starts, stops):
                key = int(keys[int(start)])
                bucket = local.setdefault(key, array("d"))
                bucket.frombytes(
                    np.ascontiguousarray(period_values[int(start) : int(stop)]).tobytes()
                )
        return local

    if len(left_ranges) == 1:
        partial_bins = [build_range(left_ranges[0])]
    else:
        with ThreadPoolExecutor(
            max_workers=len(left_ranges), thread_name_prefix="event-pairs"
        ) as pool:
            partial_bins = list(pool.map(build_range, left_ranges))

    merged: dict[int, array] = {}
    for partial in partial_bins:
        for key, values in partial.items():
            merged.setdefault(key, array("d")).frombytes(values.tobytes())
    return merged


def _fit_ranked_seed_bins(
    ranked_seeds: list[array],
    times: np.ndarray,
    snrs: np.ndarray,
    durations: np.ndarray,
    depths: np.ndarray,
    tolerance_days: float,
    shape_ratio_limit: float,
    minimum_instances: int,
    adaptive_recurrence_sigma: float | None,
):
    """Yield seed fits in original order with bounded exact parallelism."""

    def trial_period(values: array) -> float:
        period_values = np.frombuffer(values, dtype=np.float64)
        return float(np.median(period_values))

    def evaluate(values: array):
        period = trial_period(values)
        return period, _fit_period_solutions_prepared(
            times,
            snrs,
            durations,
            depths,
            period,
            tolerance_days,
            shape_ratio_limit,
            minimum_instances,
            adaptive_recurrence_sigma,
        )

    processes = min(_event_search_processes(), max(1, len(ranked_seeds)))
    if processes > 1:
        trial_periods = [trial_period(values) for values in ranked_seeds]
        period_iterator = iter(trial_periods)
        pending = deque()
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=processes,
            mp_context=context,
            initializer=_initialize_event_fit_worker,
            initargs=(
                times,
                snrs,
                durations,
                depths,
                tolerance_days,
                shape_ratio_limit,
                minimum_instances,
                adaptive_recurrence_sigma,
            ),
        ) as pool:
            for _ in range(min(len(trial_periods), processes * 2)):
                period = next(period_iterator)
                pending.append((period, pool.submit(_fit_period_worker, period)))
            while pending:
                period, future = pending.popleft()
                yield period, future.result()
                try:
                    next_period = next(period_iterator)
                except StopIteration:
                    continue
                pending.append(
                    (next_period, pool.submit(_fit_period_worker, next_period))
                )
        return

    workers = min(_event_search_threads(), max(1, len(ranked_seeds)))
    if workers == 1:
        for values in ranked_seeds:
            yield evaluate(values)
        return

    seed_iterator = iter(ranked_seeds)
    pending = deque()
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="event-seed"
    ) as pool:
        for _ in range(min(len(ranked_seeds), workers * 2)):
            pending.append(pool.submit(evaluate, next(seed_iterator)))
        while pending:
            # Submission-order collection preserves the historical stable
            # candidate order even when later seed bins finish first.
            yield pending.popleft().result()
            try:
                values = next(seed_iterator)
            except StopIteration:
                continue
            pending.append(pool.submit(evaluate, values))


def _periodic_candidates(
    events: list[DipEvent], time: np.ndarray, config: EventSearchConfig
) -> list[PeriodicEventCandidate]:
    if len(events) < config.min_periodic_instances:
        return []
    # The bounded interactive mode keeps the historical activity guard.  The
    # exhaustive scientific mode (max_events=None) must retain every event.
    working_limit = None if config.max_events is None else min(80, config.max_events)
    working = sorted(events, key=lambda item: item.snr, reverse=True)
    if working_limit is not None:
        working = working[:working_limit]
    working.sort(key=lambda item: item.center_btjd)
    times = np.asarray([event.center_btjd for event in working], dtype=float)
    snrs = np.asarray([event.snr for event in working], dtype=float)
    durations = np.asarray([event.duration_hours for event in working], dtype=float)
    depths = np.asarray([event.depth_ppm for event in working], dtype=float)
    event_indices = np.asarray([event.index for event in working], dtype=int)
    minimum_instances = _effective_min_periodic_instances(working, config)
    adaptive_recurrence_sigma = (
        float(config.high_activity_recurrence_sigma)
        if minimum_instances > max(3, int(config.min_periodic_instances))
        else None
    )

    log_step = 0.0015
    seed_bins = _build_seed_bins_vectorized(
        times, durations, depths, config, log_step
    )

    ranked_seeds = sorted(
        seed_bins.values(),
        key=lambda values: (
            len(values),
            -np.std(np.frombuffer(values, dtype=np.float64)),
        ),
        reverse=True,
    )
    if config.max_seed_bins is not None:
        ranked_seeds = ranked_seeds[: config.max_seed_bins]
    results: list[PeriodicEventCandidate] = []
    exhaustive = config.max_periodic_candidates is None
    indices_by_event_signature: dict[tuple[int, ...], list[int]] = {}
    median_duration_days = np.median(durations) / 24.0
    tolerance_days = max(
        0.75 * float(median_duration_days), config.ttv_tolerance_hours / 24.0
    )
    observation_time = np.asarray(time, dtype=float)
    segment_bounds = [
        (float(observation_time[indices][0]), float(observation_time[indices][-1]))
        for indices in _segments(observation_time, config.gap_days)
        if indices.size >= 3
    ]
    for _trial_period, fitted_solutions in _fit_ranked_seed_bins(
        ranked_seeds,
        times,
        snrs,
        durations,
        depths,
        tolerance_days,
        config.shape_ratio_limit,
        minimum_instances,
        adaptive_recurrence_sigma,
    ):
        for period, epoch, cycles, residual, matched in fitted_solutions:
            if not (config.min_period_days <= period <= config.max_period_days):
                continue
            chosen_indices = np.flatnonzero(matched)
            expected = 0
            for segment_start, segment_stop in segment_bounds:
                first_cycle = int(np.ceil((segment_start - epoch) / period))
                last_cycle = int(np.floor((segment_stop - epoch) / period))
                expected += max(0, last_cycle - first_cycle + 1)
            expected = max(expected, int(chosen_indices.size))
            coverage_fraction = int(chosen_indices.size) / expected
            # Event-first candidates are meant for individually visible dips.  A
            # short alias predicting many absent strong dips is therefore implausible.
            if coverage_fraction < 0.45:
                continue
            ttv_minutes = residual[matched] * 24.0 * 60.0
            rms = float(np.sqrt(np.mean(residual[matched] ** 2)))
            median_snr = float(np.median(snrs[chosen_indices]))
            score = float(
                int(chosen_indices.size)
                * median_snr
                * coverage_fraction**2
                / (1.0 + rms / tolerance_days)
            )
            candidate = PeriodicEventCandidate(
                period_days=period,
                epoch_btjd=epoch,
                instances=int(chosen_indices.size),
                expected_instances=expected,
                coverage_fraction=float(coverage_fraction),
                duration_hours=float(np.median(durations[chosen_indices])),
                depth_ppm=float(np.median(depths[chosen_indices])),
                median_event_snr=median_snr,
                score=score,
                mean_abs_ttv_minutes=float(np.mean(np.abs(ttv_minutes))),
                max_abs_ttv_minutes=float(np.max(np.abs(ttv_minutes))),
                ttv_minutes=tuple(float(value) for value in ttv_minutes),
                matched_event_indices=tuple(
                    int(value) for value in event_indices[chosen_indices]
                ),
            )
            if exhaustive:
                _merge_exhaustive_candidate(
                    results, indices_by_event_signature, candidate
                )
                continue

            duplicate_index: int | None = None
            candidate_events = set(candidate.matched_event_indices)
            for result_index, existing in enumerate(results):
                if candidate_events != set(existing.matched_event_indices):
                    continue
                ratio = candidate.period_days / existing.period_days
                same = abs(ratio - 1.0) < 0.006
                alias = any(
                    abs(ratio - harmonic) / harmonic < 0.003
                    or abs(ratio - 1.0 / harmonic) / (1.0 / harmonic) < 0.003
                    for harmonic in range(2, 8)
                )
                if same or alias:
                    duplicate_index = result_index
                    break
            if duplicate_index is None:
                results.append(candidate)
            elif candidate.score > results[duplicate_index].score:
                results[duplicate_index] = candidate
            results.sort(key=lambda item: item.score, reverse=True)
            if config.max_periodic_candidates is not None:
                results = results[: config.max_periodic_candidates]
    return _stable_score_order(results) if exhaustive else results


def _double_candidates(
    events: list[DipEvent], used: set[int], config: EventSearchConfig
) -> list[dict]:
    available = [event for event in events if event.index not in used]
    doubles: list[dict] = []
    for left, first in enumerate(available[:-1]):
        for second in available[left + 1 :]:
            period = second.center_btjd - first.center_btjd
            if period < config.min_period_days or period > config.max_period_days:
                continue
            if not _shape_compatible(
                first.duration_hours,
                first.depth_ppm,
                second.duration_hours,
                second.depth_ppm,
                config.shape_ratio_limit,
            ):
                continue
            doubles.append(
                {
                    "first_event_index": first.index,
                    "second_event_index": second.index,
                    "first_btjd": first.center_btjd,
                    "second_btjd": second.center_btjd,
                    "possible_period_days": period,
                    "combined_snr": float(np.hypot(first.snr, second.snr)),
                    "warning": "Two events constrain a possible interval, not a unique orbital period.",
                }
            )
    doubles.sort(key=lambda item: item["combined_snr"], reverse=True)
    if config.max_periodic_candidates is None:
        return doubles
    # A dip cannot simultaneously belong to many double-event candidates.  Keep
    # the strongest non-overlapping pairings and leave the rest as singles.
    selected: list[dict] = []
    occupied: set[int] = set()
    for item in doubles:
        pair = {int(item["first_event_index"]), int(item["second_event_index"])}
        if pair & occupied:
            continue
        selected.append(item)
        occupied.update(pair)
        if len(selected) >= 20:
            break
    return selected


def _count_double_candidates(
    events: list[DipEvent], used: set[int], config: EventSearchConfig
) -> int:
    """Count every valid double without allocating one Python dict per pair.

    Double-event intervals are non-unique diagnostics, not periodic candidates.
    Exhaustive science runs retain the full event ledger, from which every pair
    can be reconstructed, and store this exact count instead of materializing a
    quadratic list of dictionaries in each worker process.
    """

    available = [event for event in events if event.index not in used]
    if len(available) < 2:
        return 0
    times = np.fromiter(
        (event.center_btjd for event in available),
        dtype=float,
        count=len(available),
    )
    durations = np.fromiter(
        (event.duration_hours for event in available),
        dtype=float,
        count=len(available),
    )
    depths = np.fromiter(
        (event.depth_ppm for event in available),
        dtype=float,
        count=len(available),
    )
    count = 0
    for left in range(len(available) - 1):
        lower = int(
            np.searchsorted(
                times,
                times[left] + config.min_period_days,
                side="left",
            )
        )
        lower = max(left + 1, lower)
        upper = int(
            np.searchsorted(
                times,
                times[left] + config.max_period_days,
                side="right",
            )
        )
        if upper <= lower:
            continue
        candidate_durations = durations[lower:upper]
        candidate_depths = depths[lower:upper]
        duration_minimum = np.minimum(durations[left], candidate_durations)
        depth_minimum = np.minimum(depths[left], candidate_depths)
        positive = (duration_minimum > 0.0) & (depth_minimum > 0.0)
        duration_ratio = np.divide(
            np.maximum(durations[left], candidate_durations),
            duration_minimum,
            out=np.full(candidate_durations.shape, np.inf),
            where=duration_minimum > 0.0,
        )
        depth_ratio = np.divide(
            np.maximum(depths[left], candidate_depths),
            depth_minimum,
            out=np.full(candidate_depths.shape, np.inf),
            where=depth_minimum > 0.0,
        )
        count += int(
            np.count_nonzero(
                positive
                & (duration_ratio <= config.shape_ratio_limit)
                & (depth_ratio <= config.shape_ratio_limit)
            )
        )
    return count


def event_first_search(
    time: np.ndarray,
    flux: np.ndarray,
    config: EventSearchConfig | None = None,
) -> EventFirstResult:
    """Return periodic, double, and single-event candidates."""
    config = config or EventSearchConfig()
    observation_time = np.asarray(time, dtype=float)
    observation_time = np.sort(observation_time[np.isfinite(observation_time)])
    events = detect_dip_events(time, flux, config)
    periodic_minimum_instances = _effective_min_periodic_instances(events, config)
    adaptive_recurrence_sigma = (
        float(config.high_activity_recurrence_sigma)
        if periodic_minimum_instances > max(3, int(config.min_periodic_instances))
        else None
    )
    periodic = _periodic_candidates(events, observation_time, config)
    # Only strong, well-covered periodic groups consume events.  Weak aliases
    # remain useful evidence but must not erase every residual single event.
    event_owners = [
        candidate
        for candidate in periodic[:5]
        if candidate.coverage_fraction >= 0.70
    ]
    used = {index for candidate in event_owners for index in candidate.matched_event_indices}
    if config.materialize_double_candidates:
        doubles = _double_candidates(events, used, config)
        double_candidate_count = len(doubles)
        double_used = {
            int(item[key])
            for item in doubles
            for key in ("first_event_index", "second_event_index")
        }
    else:
        doubles = []
        double_candidate_count = _count_double_candidates(events, used, config)
        double_used = set()
    singles = tuple(
        event for event in events if event.index not in used and event.index not in double_used
    )
    return EventFirstResult(
        periodic=tuple(periodic),
        doubles=tuple(doubles),
        singles=singles,
        all_events=tuple(events),
        double_candidate_count=double_candidate_count,
        double_candidates_materialized=config.materialize_double_candidates,
        periodic_minimum_instances=periodic_minimum_instances,
        sparse_trains_reconstructible_from_event_ledger=(
            periodic_minimum_instances > max(3, int(config.min_periodic_instances))
        ),
        adaptive_recurrence_sigma=adaptive_recurrence_sigma,
    )


def measure_candidate_timings(
    events: list[DipEvent],
    *,
    period_days: float,
    epoch_btjd: float,
    duration_hours: float,
    tolerance_hours: float = 3.0,
) -> dict:
    """Attach observed event times and O-C/TTV values to a BLS candidate.

    This does not refit the BLS period.  It is safe to expose as a review feature:
    players can see which predicted transits arrived early or late.
    """
    if period_days <= 0 or duration_hours <= 0:
        raise ValueError("period_days and duration_hours must be positive")
    tolerance_days = max(tolerance_hours, 0.75 * duration_hours) / 24.0
    rows: list[dict] = []
    for event in events:
        cycle = int(round((event.center_btjd - epoch_btjd) / period_days))
        calculated = epoch_btjd + cycle * period_days
        offset = event.center_btjd - calculated
        if abs(offset) > tolerance_days:
            continue
        rows.append(
            {
                "event_index": event.index,
                "cycle": cycle,
                "calculated_btjd": calculated,
                "observed_btjd": event.center_btjd,
                "ttv_minutes": offset * 24.0 * 60.0,
                "event_snr": event.snr,
                "event_depth_ppm": event.depth_ppm,
            }
        )
    rows.sort(key=lambda item: item["cycle"])
    offsets = np.asarray([item["ttv_minutes"] for item in rows], dtype=float)
    return {
        "available": bool(rows),
        "matched_instances": len(rows),
        "mean_abs_ttv_minutes": float(np.mean(np.abs(offsets))) if rows else np.nan,
        "max_abs_ttv_minutes": float(np.max(np.abs(offsets))) if rows else np.nan,
        "events": rows,
    }
