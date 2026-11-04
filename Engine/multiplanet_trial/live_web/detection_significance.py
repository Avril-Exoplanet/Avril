"""Non-destructive, red-noise-aware significance for locked candidates.

This module deliberately runs *after* discovery has fixed period, epoch, and
duration.  It does not search for a better ephemeris and it never removes or
reorders a candidate.  The reported statistic is named ``MES-like`` because it
uses duration-scale empirical noise rather than the proprietary SPOC whitening
implementation; it must not be presented as an official SPOC MES.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
from scipy.stats import norm


LOCKED_FIELDS = ("candidate_id", "rank", "period_days", "epoch_btjd", "duration_hours")
REFERENCE_MES = 7.1
MIN_CONTROL_WINDOWS = 20


_SIGNIFICANCE_BRANCH = None
_SIGNIFICANCE_BLOCKS = None
_SIGNIFICANCE_CANDIDATES = None
_SIGNIFICANCE_VIEWS = None


def _initialize_significance_worker(
    branch,
    blocks: list[list[int]],
    candidates: list[dict],
    inner_threads: int,
) -> None:
    global _SIGNIFICANCE_BRANCH
    global _SIGNIFICANCE_BLOCKS
    global _SIGNIFICANCE_CANDIDATES
    global _SIGNIFICANCE_VIEWS

    os.environ["HOUXING_SIGNIFICANCE_THREADS"] = str(max(1, inner_threads))
    # Spawned workers must install the same exact compiled null kernel as the
    # parent process; importing this module alone intentionally does not patch
    # the frozen reference implementation.
    from . import optimization_v4

    optimization_v4.install()
    _SIGNIFICANCE_BRANCH = branch
    _SIGNIFICANCE_BLOCKS = blocks
    _SIGNIFICANCE_CANDIDATES = candidates
    _SIGNIFICANCE_VIEWS = _segment_views(branch)


def _significance_batch_worker(
    batch: list[tuple[int, dict]],
    phase_trials: int,
    search_trial_factor: int,
) -> list[tuple[int, dict]]:
    if (
        _SIGNIFICANCE_BRANCH is None
        or _SIGNIFICANCE_BLOCKS is None
        or _SIGNIFICANCE_CANDIDATES is None
        or _SIGNIFICANCE_VIEWS is None
    ):
        raise RuntimeError("significance worker was not initialized")
    return [
        (
            index,
            candidate_detection_significance(
                candidate,
                _SIGNIFICANCE_CANDIDATES,
                _SIGNIFICANCE_BRANCH,
                _SIGNIFICANCE_BLOCKS,
                phase_trials=phase_trials,
                search_trial_factor=search_trial_factor,
                prepared_views=_SIGNIFICANCE_VIEWS,
            ),
        )
        for index, candidate in batch
    ]


def _mad_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    sigma = float(1.4826 * np.median(np.abs(finite - median)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(finite))
    return sigma if np.isfinite(sigma) and sigma > 0 else 0.0


def _locked_snapshot(candidates: Iterable[Mapping]) -> list[tuple]:
    return [tuple(candidate.get(field) for field in LOCKED_FIELDS) for candidate in candidates]


def _phase_distance(time: np.ndarray, *, period: float, epoch: float) -> np.ndarray:
    return (time - epoch + 0.5 * period) % period - 0.5 * period


def _transit_exclusion_mask(
    time: np.ndarray,
    candidates: Iterable[Mapping],
    *,
    width_multiplier: float = 1.25,
) -> np.ndarray:
    excluded = np.zeros(np.asarray(time).size, dtype=bool)
    for candidate in candidates:
        period = float(candidate["period_days"])
        epoch = float(candidate["epoch_btjd"])
        duration = float(candidate["duration_hours"]) / 24.0
        distance = np.abs(_phase_distance(time, period=period, epoch=epoch))
        excluded |= distance <= width_multiplier * 0.5 * duration
    return excluded


def _window_measurement(
    time: np.ndarray,
    flux: np.ndarray,
    usable: np.ndarray,
    *,
    center: float,
    duration_days: float,
    cadence_days: float,
) -> dict | None:
    left = int(np.searchsorted(time, center - 4.0 * duration_days, side="left"))
    right = int(np.searchsorted(time, center + 4.0 * duration_days, side="right"))
    if right - left < 13:
        return None
    local_time = time[left:right]
    local_flux = flux[left:right]
    local_usable = usable[left:right]
    distance = np.abs(local_time - center)
    inside = local_usable & (distance <= 0.5 * duration_days)
    nearby = local_usable & (distance >= 1.5 * duration_days) & (
        distance <= 4.0 * duration_days
    )
    expected_inside = max(3, int(round(duration_days / max(cadence_days, 1e-9))))
    minimum_inside = max(3, int(math.ceil(0.45 * expected_inside)))
    minimum_nearby = max(10, int(math.ceil(1.5 * expected_inside)))
    if int(inside.sum()) < minimum_inside or int(nearby.sum()) < minimum_nearby:
        return None
    baseline = float(np.median(local_flux[nearby]))
    in_level = float(np.median(local_flux[inside]))
    point_sigma = _mad_sigma(local_flux[nearby])
    white_uncertainty = (
        point_sigma / math.sqrt(int(inside.sum())) if point_sigma > 0 else 0.0
    )
    return {
        "depth": baseline - in_level,
        "white_uncertainty": white_uncertainty,
        "in_points": int(inside.sum()),
        "baseline_points": int(nearby.sum()),
    }


def _segment_views(branch) -> dict[int, dict]:
    output: dict[int, dict] = {}
    for segment in sorted({int(value) for value in np.unique(branch.segment_id)}):
        selected = np.asarray(branch.segment_id) == segment
        time = np.asarray(branch.time[selected], dtype=float)
        flux = np.asarray(branch.flux[selected], dtype=float)
        finite = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[finite], flux[finite]
        order = np.argsort(time)
        time, flux = time[order], flux[order]
        differences = np.diff(time)
        differences = differences[(differences > 0) & np.isfinite(differences)]
        cadence = float(np.median(differences)) if differences.size else 2.0 / 1440.0
        output[segment] = {
            "time": time,
            "flux": flux,
            "cadence_days": cadence,
        }
    return output


def _duration_noise_models(
    views: Mapping[int, Mapping],
    candidates: list[Mapping],
    *,
    duration_days: float,
) -> dict[int, dict]:
    models: dict[int, dict] = {}
    for segment, view in views.items():
        time = np.asarray(view["time"], dtype=float)
        flux = np.asarray(view["flux"], dtype=float)
        cadence = float(view["cadence_days"])
        excluded = _transit_exclusion_mask(time, candidates)
        usable = ~excluded
        start = float(time[0] + 4.0 * duration_days)
        stop = float(time[-1] - 4.0 * duration_days)
        step = max(duration_days, 10.0 * cadence)
        centers = np.arange(start, stop, step) if stop > start else np.asarray([])
        if centers.size > 800:
            centers = centers[np.linspace(0, centers.size - 1, 800, dtype=int)]
        compiled_measurements = None
        try:
            from .significance_acceleration import (
                control_window_measurements_compiled,
            )

            compiled_measurements = control_window_measurements_compiled(
                time,
                flux,
                usable,
                centers,
                duration_days=duration_days,
                cadence_days=cadence,
            )
        except Exception:
            # The exact Python reference remains available on minimal systems
            # without Numba and as a correctness fallback for unsupported
            # native runtimes.
            compiled_measurements = None
        if compiled_measurements is None:
            measurements = [
                _window_measurement(
                    time,
                    flux,
                    usable,
                    center=float(center),
                    duration_days=duration_days,
                    cadence_days=cadence,
                )
                for center in centers
            ]
            measurements = [row for row in measurements if row is not None]
            control_depths = np.asarray(
                [row["depth"] for row in measurements], dtype=float
            )
            white_values = np.asarray(
                [row["white_uncertainty"] for row in measurements], dtype=float
            )
        else:
            control_depths, white_values = compiled_measurements
            control_depths = np.asarray(control_depths, dtype=float)
            white_values = np.asarray(white_values, dtype=float)
        control_center = float(np.median(control_depths)) if control_depths.size else 0.0
        duration_sigma = _mad_sigma(control_depths)
        white_values = white_values[white_values > 0]
        white_sigma = float(np.median(white_values)) if white_values.size else 0.0
        fallback = False
        if control_depths.size < MIN_CONTROL_WINDOWS or duration_sigma <= 0:
            fallback = True
            point_sigma = _mad_sigma(flux[usable])
            expected_inside = max(3, int(round(duration_days / max(cadence, 1e-9))))
            duration_sigma = point_sigma / math.sqrt(expected_inside) if point_sigma > 0 else 0.0
            if white_sigma <= 0:
                white_sigma = duration_sigma
        models[int(segment)] = {
            "usable": usable,
            "control_center": control_center,
            "duration_sigma": duration_sigma,
            "white_sigma": white_sigma,
            "red_noise_beta": (
                duration_sigma / white_sigma if duration_sigma > 0 and white_sigma > 0 else None
            ),
            "control_windows": int(control_depths.size),
            "fallback_white_noise": fallback,
            "control_depth_percentiles_ppm": (
                [float(value * 1e6) for value in np.percentile(control_depths, [1, 5, 50, 95, 99])]
                if control_depths.size
                else []
            ),
        }
    return models


def _predicted_centers(time: np.ndarray, *, period: float, epoch: float) -> list[tuple[int, float]]:
    first = int(math.floor((float(time[0]) - epoch) / period)) - 1
    last = int(math.ceil((float(time[-1]) - epoch) / period)) + 1
    return [(cycle, epoch + cycle * period) for cycle in range(first, last + 1)]


def _measure_event_train(
    views: Mapping[int, Mapping],
    noise_models: Mapping[int, Mapping],
    *,
    period: float,
    epoch: float,
    duration_days: float,
    usable_overrides: Mapping[int, np.ndarray] | None = None,
) -> list[dict]:
    events: list[dict] = []
    for segment, view in views.items():
        model = noise_models[int(segment)]
        sigma = float(model["duration_sigma"])
        if sigma <= 0:
            continue
        time = np.asarray(view["time"], dtype=float)
        flux = np.asarray(view["flux"], dtype=float)
        usable = (
            np.asarray(usable_overrides[int(segment)], dtype=bool)
            if usable_overrides is not None
            else np.ones(time.size, dtype=bool)
        )
        cadence = float(view["cadence_days"])
        for cycle, center in _predicted_centers(time, period=period, epoch=epoch):
            measured = _window_measurement(
                time,
                flux,
                usable,
                center=center,
                duration_days=duration_days,
                cadence_days=cadence,
            )
            if measured is None:
                continue
            depth = float(measured["depth"] - float(model["control_center"]))
            events.append(
                {
                    "segment": int(segment),
                    "cycle": int(cycle),
                    "center_btjd": float(center),
                    "depth_ppm": depth * 1e6,
                    "duration_noise_ppm": sigma * 1e6,
                    "ses_like": depth / sigma,
                    "in_points": int(measured["in_points"]),
                    "baseline_points": int(measured["baseline_points"]),
                }
            )
    return events


def _combine_events(events: Iterable[Mapping]) -> dict:
    rows = list(events)
    if not rows:
        return {
            "events": 0,
            "weighted_depth_ppm": None,
            "weighted_depth_uncertainty_ppm": None,
            "mes_like": None,
        }
    depths = np.asarray([float(row["depth_ppm"]) / 1e6 for row in rows])
    sigmas = np.asarray([float(row["duration_noise_ppm"]) / 1e6 for row in rows])
    good = np.isfinite(depths) & np.isfinite(sigmas) & (sigmas > 0)
    depths, sigmas = depths[good], sigmas[good]
    if not depths.size:
        return {
            "events": 0,
            "weighted_depth_ppm": None,
            "weighted_depth_uncertainty_ppm": None,
            "mes_like": None,
        }
    weights = 1.0 / np.square(sigmas)
    uncertainty = float(1.0 / math.sqrt(float(np.sum(weights))))
    weighted_depth = float(np.sum(depths * weights) / np.sum(weights))
    return {
        "events": int(depths.size),
        "weighted_depth_ppm": weighted_depth * 1e6,
        "weighted_depth_uncertainty_ppm": uncertainty * 1e6,
        "mes_like": weighted_depth / uncertainty,
    }


def _event_gap_diagnostics(
    views: Mapping[int, Mapping],
    events: list[Mapping],
    *,
    duration_days: float,
) -> dict:
    """Identify a high statistic dominated by one gap-adjacent event.

    This diagnostic never changes a locked period, epoch, duration, or the
    retained candidate ledger.  It prevents a single event with incomplete
    cadence coverage or one-sided baseline from promoting an otherwise weak
    event train to automatic high priority.
    """
    audited: list[dict] = []
    for event in events:
        segment = int(event["segment"])
        center = float(event["center_btjd"])
        view = views[segment]
        time = np.asarray(view["time"], dtype=float)
        cadence = max(float(view["cadence_days"]), 1e-9)
        differences = np.diff(time)
        gap_indices = np.flatnonzero(differences > 5.0 * cadence)
        expected_inside = max(3, int(round(duration_days / cadence)))
        inside_points = int(np.sum(np.abs(time - center) <= 0.5 * duration_days))
        coverage = min(1.0, inside_points / expected_inside)
        left_baseline = int(
            np.sum(
                (time >= center - 4.0 * duration_days)
                & (time <= center - 1.5 * duration_days)
            )
        )
        right_baseline = int(
            np.sum(
                (time >= center + 1.5 * duration_days)
                & (time <= center + 4.0 * duration_days)
            )
        )
        baseline_symmetry = min(left_baseline, right_baseline) / max(
            left_baseline, right_baseline, 1
        )
        nearest_gap_edge_hours = None
        containing_gap_hours = None
        if gap_indices.size:
            edges = np.concatenate([time[gap_indices], time[gap_indices + 1]])
            nearest_gap_edge_hours = float(np.min(np.abs(edges - center)) * 24.0)
            for index in gap_indices:
                if time[index] < center < time[index + 1]:
                    containing_gap_hours = float(
                        (time[index + 1] - time[index]) * 24.0
                    )
                    break
        gap_edge_within_half_duration = bool(
            nearest_gap_edge_hours is not None
            and nearest_gap_edge_hours <= 0.5 * duration_days * 24.0
        )
        critical = bool(
            containing_gap_hours is not None
            or gap_edge_within_half_duration
            or coverage < 0.8
            or baseline_symmetry < 0.2
        )
        audited.append(
            {
                "segment": segment,
                "cycle": int(event["cycle"]),
                "center_btjd": center,
                "ses_like": float(event["ses_like"]),
                "transit_coverage_fraction": float(coverage),
                "left_baseline_points": left_baseline,
                "right_baseline_points": right_baseline,
                "baseline_symmetry_fraction": float(baseline_symmetry),
                "nearest_long_gap_edge_hours": nearest_gap_edge_hours,
                "containing_long_gap_hours": containing_gap_hours,
                "critical_gap_compromise": critical,
            }
        )

    uncompromised = [
        event
        for event, audit in zip(events, audited)
        if not audit["critical_gap_compromise"]
    ]
    without_compromised = _combine_events(uncompromised)
    supported_uncompromised = sum(
        float(event["ses_like"]) >= 3.0 for event in uncompromised
    )
    dominant_index = (
        max(range(len(events)), key=lambda index: abs(float(events[index]["ses_like"])))
        if events
        else None
    )
    dominant_audit = audited[dominant_index] if dominant_index is not None else None
    remaining_mes = without_compromised.get("mes_like")
    gap_dominated_weak_remainder = bool(
        dominant_audit
        and dominant_audit["critical_gap_compromise"]
        and (remaining_mes is None or float(remaining_mes) < REFERENCE_MES)
        and supported_uncompromised < 2
    )
    return {
        "schema_version": "avril-event-gap-diagnostics-v1",
        "events": audited,
        "critical_event_count": sum(
            bool(row["critical_gap_compromise"]) for row in audited
        ),
        "dominant_event_segment": (
            dominant_audit.get("segment") if dominant_audit else None
        ),
        "dominant_event_ses_like": (
            dominant_audit.get("ses_like") if dominant_audit else None
        ),
        "dominant_event_gap_compromised": bool(
            dominant_audit and dominant_audit["critical_gap_compromise"]
        ),
        "uncompromised_event_count": len(uncompromised),
        "supported_uncompromised_events": int(supported_uncompromised),
        "combined_without_gap_compromised_events": without_compromised,
        "gap_dominated_weak_remainder": gap_dominated_weak_remainder,
        "policy": (
            "candidate retained; automatic high priority is blocked only when "
            "one compromised dominant event lifts an otherwise weak event train"
        ),
    }


def _sign_probability(events: list[Mapping]) -> float | None:
    if not events:
        return None
    positive = sum(float(row["depth_ppm"]) > 0 for row in events)
    count = len(events)
    return float(sum(math.comb(count, value) for value in range(positive, count + 1)) / 2**count)


def _seed(candidate: Mapping) -> int:
    material = "|".join(
        f"{float(candidate[field]):.12g}"
        for field in ("period_days", "epoch_btjd", "duration_hours")
    )
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _phase_shift_null(
    candidate: Mapping,
    views: Mapping[int, Mapping],
    noise_models: Mapping[int, Mapping],
    observed_events: int,
    *,
    trials: int,
) -> dict:
    period = float(candidate["period_days"])
    epoch = float(candidate["epoch_btjd"])
    duration_days = float(candidate["duration_hours"]) / 24.0
    rng = np.random.default_rng(_seed(candidate))
    offsets = (np.arange(trials, dtype=float) + rng.random(trials)) * period / trials
    minimum_events = max(2, int(math.floor(0.8 * observed_events)))
    usable = {segment: np.asarray(model["usable"], dtype=bool) for segment, model in noise_models.items()}
    values: list[float] = []
    for offset in offsets:
        # The null masks every retained candidate transit, including this
        # candidate's true phase, before evaluating a shifted event train.
        events = _measure_event_train(
            views,
            noise_models,
            period=period,
            epoch=epoch + float(offset),
            duration_days=duration_days,
            usable_overrides=usable,
        )
        combined = _combine_events(events)
        if int(combined["events"]) >= minimum_events and combined["mes_like"] is not None:
            values.append(float(combined["mes_like"]))
    null = np.asarray(values, dtype=float)
    return {
        "requested_trials": int(trials),
        "valid_trials": int(null.size),
        "minimum_events_per_trial": int(minimum_events),
        "values": null,
    }


def _harmonic_warnings(candidate: Mapping, candidates: list[Mapping]) -> list[dict]:
    period = float(candidate["period_days"])
    epoch = float(candidate["epoch_btjd"])
    duration = float(candidate["duration_hours"]) / 24.0
    warnings: list[dict] = []
    for other in candidates:
        if other is candidate or other.get("candidate_id") == candidate.get("candidate_id"):
            continue
        other_period = float(other["period_days"])
        if period <= other_period:
            continue
        order = int(round(period / other_period))
        ratio = period / other_period
        if order < 2 or order > 10 or abs(ratio - order) > 0.002:
            continue
        phase_error = abs(
            (epoch - float(other["epoch_btjd"]) + 0.5 * other_period) % other_period
            - 0.5 * other_period
        )
        tolerance = max(duration, float(other["duration_hours"]) / 24.0)
        if phase_error > tolerance:
            continue
        # A fractional-period alias predicts many extra events.  Do not let
        # that weak shorter alias invalidate a longer period whose complete
        # event train recurs.  A shorter solution is a viable harmonic parent
        # only when its own locked event sequence is comparably repeatable.
        candidate_repeatability = (
            candidate.get("holdout_verification", {}).get("event_repeatability", {})
        )
        shorter_repeatability = (
            other.get("holdout_verification", {}).get("event_repeatability", {})
        )
        candidate_support = candidate_repeatability.get("support_fraction")
        shorter_support = shorter_repeatability.get("support_fraction")
        if shorter_support is not None:
            shorter_support = float(shorter_support)
            if shorter_support < 0.6:
                continue
            if (
                candidate_support is not None
                and shorter_support + 0.1 < float(candidate_support)
            ):
                continue
        warnings.append(
            {
                "shorter_candidate_id": other.get("candidate_id"),
                "shorter_rank": other.get("rank"),
                "shorter_period_days": other_period,
                "integer_order": order,
                "period_ratio": ratio,
                "phase_error_days": phase_error,
                "longer_event_support_fraction": candidate_support,
                "shorter_event_support_fraction": shorter_support,
                "interpretation": "longer period is not independent until the integer harmonic is ruled out",
            }
        )
    return warnings


def _same_harmonic_event_family(first: Mapping, second: Mapping) -> bool:
    """Return true when masking one solution would erase the other's events."""
    if float(first["period_days"]) >= float(second["period_days"]):
        longer, shorter = first, second
    else:
        longer, shorter = second, first
    long_period = float(longer["period_days"])
    short_period = float(shorter["period_days"])
    order = int(round(long_period / short_period))
    if order < 1 or order > 10 or abs(long_period / short_period - order) > 0.002:
        return False
    phase_error = abs(
        (
            float(longer["epoch_btjd"])
            - float(shorter["epoch_btjd"])
            + 0.5 * short_period
        )
        % short_period
        - 0.5 * short_period
    )
    tolerance = max(
        float(longer["duration_hours"]), float(shorter["duration_hours"])
    ) / 24.0
    return phase_error <= tolerance


def _prepare_candidate_detection_significance(
    candidate: Mapping,
    views: Mapping[int, Mapping],
    blocks: list[list[int]],
) -> dict:
    period = float(candidate["period_days"])
    epoch = float(candidate["epoch_btjd"])
    duration_days = float(candidate["duration_hours"]) / 24.0
    # Build the duration-scale noise model by removing only this candidate's
    # locked transit windows.  The discovery ledger is deliberately broad and
    # can contain hundreds of weak or overlapping hypotheses.  Treating every
    # retained hypothesis as a real transit over-masks the light curve, can
    # leave zero valid phase-shift trials, and makes runtime quadratic in the
    # candidate count.  Other signals remain in the empirical controls, which
    # is conservative for the candidate being tested.
    noise_models = _duration_noise_models(
        views, [candidate], duration_days=duration_days
    )
    events = _measure_event_train(
        views,
        noise_models,
        period=period,
        epoch=epoch,
        duration_days=duration_days,
        usable_overrides=None,
    )
    combined = _combine_events(events)
    block_by_segment = {
        int(segment): block_index
        for block_index, segments in enumerate(blocks, start=1)
        for segment in segments
    }
    block_metrics = []
    for block_index in range(1, len(blocks) + 1):
        selected = [row for row in events if block_by_segment.get(int(row["segment"])) == block_index]
        block_metrics.append({"block": block_index, **_combine_events(selected)})
    return {
        "period": period,
        "epoch": epoch,
        "duration_days": duration_days,
        "views": views,
        "noise_models": noise_models,
        "events": events,
        "combined": combined,
        "block_metrics": block_metrics,
    }


def candidate_detection_significance(
    candidate: Mapping,
    candidates: list[Mapping],
    branch,
    blocks: list[list[int]],
    *,
    phase_trials: int = 4096,
    search_trial_factor: int = 540_000,
    prepared_views: Mapping[int, Mapping] | None = None,
    prepared_candidate: Mapping | None = None,
) -> dict:
    """Measure one locked candidate without changing its discovery solution."""
    views = (
        prepared_views
        if prepared_views is not None
        else _segment_views(branch)
    )
    prepared = (
        dict(prepared_candidate)
        if prepared_candidate is not None
        else _prepare_candidate_detection_significance(candidate, views, blocks)
    )
    period = float(prepared["period"])
    epoch = float(prepared["epoch"])
    duration_days = float(prepared["duration_days"])
    noise_models = prepared["noise_models"]
    events = prepared["events"]
    combined = prepared["combined"]
    block_metrics = prepared["block_metrics"]
    gap_diagnostics = _event_gap_diagnostics(
        views,
        events,
        duration_days=duration_days,
    )

    null_result = _phase_shift_null(
        candidate,
        views,
        noise_models,
        int(combined["events"]),
        trials=max(0, int(phase_trials)),
    )
    null_values = np.asarray(null_result.pop("values"), dtype=float)
    observed_mes = combined.get("mes_like")
    null_center = float(np.median(null_values)) if null_values.size else None
    null_sigma = _mad_sigma(null_values) if null_values.size else 0.0
    calibrated_mes = (
        (float(observed_mes) - float(null_center)) / null_sigma
        if observed_mes is not None and null_center is not None and null_sigma > 0
        else None
    )
    exceedances = (
        int(np.sum(null_values >= float(observed_mes)))
        if observed_mes is not None and null_values.size
        else None
    )
    empirical_fap = (
        (exceedances + 1) / (null_values.size + 1)
        if exceedances is not None
        else None
    )
    gaussian_tail = (
        float(norm.sf(calibrated_mes)) if calibrated_mes is not None else None
    )
    look_elsewhere = (
        min(1.0, gaussian_tail * max(1, int(search_trial_factor)))
        if gaussian_tail is not None
        else None
    )
    warnings = _harmonic_warnings(candidate, candidates)
    flags: list[str] = []
    observed_passes_reference = bool(
        observed_mes is not None and float(observed_mes) >= REFERENCE_MES
    )
    gap_dominated = bool(gap_diagnostics["gap_dominated_weak_remainder"])
    if calibrated_mes is None:
        flags.append("insufficient_null_calibration")
    if not observed_passes_reference:
        flags.append("below_7p1_mes_like_reference")
    if len(candidate.get("discovered_blocks", [])) < 2:
        flags.append("single_block_independent_discovery")
    if warnings:
        flags.append("integer_harmonic_of_retained_candidate")
    if gap_dominated:
        flags.append("gap_adjacent_single_event_dominated")
    if look_elsewhere is None or look_elsewhere > 0.01:
        flags.append("global_false_alarm_not_tightly_constrained")

    if gap_dominated:
        status = "gap_adjacent_single_event_dominated"
    elif warnings:
        status = "integer_harmonic_not_independent"
    elif calibrated_mes is None or observed_mes is None:
        status = "insufficient_statistics"
    elif not observed_passes_reference:
        status = "below_7p1_mes_like_reference"
    elif look_elsewhere is None or look_elsewhere > 0.01:
        status = "phase_shift_false_alarm_not_tightly_constrained"
    else:
        status = "meets_7p1_mes_like_and_phase_null_reference"

    passes_reference = bool(
        observed_passes_reference
        and calibrated_mes is not None
        and look_elsewhere is not None
        and look_elsewhere <= 0.01
        and not warnings
        and not gap_dominated
    )

    return {
        "schema_version": "locked-red-noise-significance-v2",
        "statistic_name": (
            "duration-noise MES-like plus independently calibrated phase-null FAP"
        ),
        "official_spoc_mes": False,
        "ephemeris_locked": True,
        "used_to_retune_or_delete": False,
        "period_days": period,
        "epoch_btjd": epoch,
        "duration_hours": float(candidate["duration_hours"]),
        "reference_mes": REFERENCE_MES,
        "duration_noise_models": [
            {
                "segment": int(segment),
                **{key: value for key, value in model.items() if key != "usable"},
            }
            for segment, model in noise_models.items()
        ],
        "events": events,
        "event_count": len(events),
        "positive_depth_events": sum(float(row["depth_ppm"]) > 0 for row in events),
        "ses_ge_3_events": sum(float(row["ses_like"]) >= 3.0 for row in events),
        "one_sided_event_sign_probability": _sign_probability(events),
        "combined": combined,
        "block_metrics": block_metrics,
        "event_gap_diagnostics": gap_diagnostics,
        "phase_shift_null": {
            **null_result,
            "null_median_mes_like": null_center,
            "null_mad_sigma": null_sigma if null_sigma > 0 else None,
            "null_percentiles": (
                [float(value) for value in np.percentile(null_values, [50, 90, 95, 99, 99.9])]
                if null_values.size
                else []
            ),
            "exceedances": exceedances,
            "local_empirical_fap": empirical_fap,
            "local_empirical_fap_resolution": (
                1.0 / (null_values.size + 1) if null_values.size else None
            ),
        },
        "null_calibrated_mes_like": calibrated_mes,
        "gaussian_tail_fap_estimate": gaussian_tail,
        "search_trial_factor": int(search_trial_factor),
        "search_trial_factor_note": (
            "declared BLS period-duration-block grid lower bound; event-first and correlated grid trials make this an estimate"
        ),
        "look_elsewhere_fap_estimate": look_elsewhere,
        "passes_7p1_reference": passes_reference,
        "harmonic_warnings": warnings,
        "status": status,
        "flags": flags,
    }


def annotate_detection_significance(
    candidates: list[dict],
    branch,
    blocks: list[list[int]],
    *,
    phase_trials: int = 4096,
    search_trial_factor: int = 540_000,
    progress=None,
) -> tuple[list[dict], dict]:
    """Attach significance reports while preserving candidate identity/order."""
    original = _locked_snapshot(candidates)
    output = [dict(candidate) for candidate in candidates]
    # Segment extraction, finite filtering, and chronological sorting depend
    # only on the target light curve.  Rebuilding the same views once per
    # candidate was pure duplicate work on candidate-explosion targets.
    prepared_views = _segment_views(branch)
    for index, candidate in enumerate(output, start=1):
        if progress:
            progress(
                "significance",
                92,
                f"Computing red-noise MES-like significance and false-alarm rate for candidate {index}/{len(output)}",
                {
                    "title": "Fixed-ephemeris detection significance",
                    "command": f"significance --locked-ephemeris --phase-trials {phase_trials}",
                    "metrics": {"candidate": index, "total": len(output)},
                },
            )
        candidate["detection_significance"] = candidate_detection_significance(
            candidate,
            output,
            branch,
            blocks,
            phase_trials=phase_trials,
            search_trial_factor=search_trial_factor,
            prepared_views=prepared_views,
        )
    if _locked_snapshot(output) != original:
        raise AssertionError("Detection significance changed candidate rank or ephemeris")
    return output, {
        "schema_version": "locked-red-noise-significance-stage-v1",
        "policy": "annotation only; no candidate deletion, reranking, or ephemeris tuning",
        "candidate_count": len(output),
        "phase_trials_each": int(phase_trials),
        "search_trial_factor": int(search_trial_factor),
        "passes_7p1_reference": sum(
            bool(row["detection_significance"]["passes_7p1_reference"])
            for row in output
        ),
        "locked_ephemerides_verified": True,
        "official_spoc_mes": False,
    }


def annotate_detection_significance_cuda_pipeline(
    candidates: list[dict],
    branch,
    blocks: list[list[int]],
    *,
    phase_trials: int = 4096,
    search_trial_factor: int = 540_000,
    preparation_workers: int = 4,
    preparation_ahead: int = 8,
    progress=None,
) -> tuple[list[dict], dict]:
    """Double-buffer CPU noise preparation against exact CUDA phase trials."""

    original = _locked_snapshot(candidates)
    output = [dict(candidate) for candidate in candidates]
    views = _segment_views(branch)
    # The production finalizer historically forced this argument to one even
    # when the CUDA kernel was starved by host-side median/control-window
    # preparation.  Allow the isolated fast path to restore the measured
    # multi-producer configuration without changing the scientific work.
    configured_workers = int(
        os.environ.get(
            "VLORA_SIGNIFICANCE_PREPARATION_WORKERS",
            str(preparation_workers),
        )
    )
    configured_ahead = int(
        os.environ.get(
            "VLORA_SIGNIFICANCE_PREPARATION_AHEAD",
            str(preparation_ahead),
        )
    )
    workers = min(max(1, configured_workers), max(1, len(output)))
    ahead = max(workers, configured_ahead)
    next_to_submit = 0
    futures: dict[int, object] = {}

    def submit_until_full(pool) -> None:
        nonlocal next_to_submit
        while next_to_submit < len(output) and len(futures) < ahead:
            index = next_to_submit
            futures[index] = pool.submit(
                _prepare_candidate_detection_significance,
                output[index],
                views,
                blocks,
            )
            next_to_submit += 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        submit_until_full(pool)
        for index, candidate in enumerate(output):
            prepared = futures.pop(index).result()
            # CPU workers continue building N+1..N+k while the calling thread
            # submits all 4096 phase trials for N to the single CUDA context.
            submit_until_full(pool)
            candidate["detection_significance"] = candidate_detection_significance(
                candidate,
                output,
                branch,
                blocks,
                phase_trials=phase_trials,
                search_trial_factor=search_trial_factor,
                prepared_views=views,
                prepared_candidate=prepared,
            )
            if progress:
                progress(
                    "significance",
                    92,
                    (
                        f"Computing red-noise MES-like significance and false-alarm rate "
                        f"for candidate {index + 1}/{len(output)}"
                    ),
                    {
                        "title": "Fixed-ephemeris detection significance",
                        "command": (
                            "significance --locked-ephemeris --cuda-exact "
                            f"--phase-trials {phase_trials} --double-buffer"
                        ),
                        "metrics": {
                            "candidate": index + 1,
                            "total": len(output),
                            "cpu_preparation_workers": workers,
                            "preparation_ahead": ahead,
                        },
                    },
                )

    if _locked_snapshot(output) != original:
        raise AssertionError("Detection significance changed candidate rank or ephemeris")
    return output, {
        "schema_version": "locked-red-noise-significance-stage-v1",
        "policy": "annotation only; no candidate deletion, reranking, or ephemeris tuning",
        "candidate_count": len(output),
        "phase_trials_each": int(phase_trials),
        "search_trial_factor": int(search_trial_factor),
        "passes_7p1_reference": sum(
            bool(row["detection_significance"]["passes_7p1_reference"])
            for row in output
        ),
        "locked_ephemerides_verified": True,
        "official_spoc_mes": False,
        "cpu_gpu_double_buffered": True,
        "cpu_preparation_workers": workers,
        "preparation_ahead": ahead,
        "sector_views_cached_once": True,
        "candidate_order_preserved": True,
    }


def annotate_detection_significance_parallel(
    candidates: list[dict],
    branch,
    blocks: list[list[int]],
    *,
    phase_trials: int = 4096,
    search_trial_factor: int = 540_000,
    processes: int = 4,
    threads_per_process: int = 3,
    progress=None,
) -> tuple[list[dict], dict]:
    """Exact multi-process significance with one cached view per worker.

    Candidates remain independent scientific trials.  Processes only divide
    those trials between CPU cores; each worker builds the immutable sector
    views once and reuses them for every candidate assigned to that worker.
    Results are written back by original index so scheduling cannot change
    candidate identity or order.
    """

    original = _locked_snapshot(candidates)
    output = [dict(candidate) for candidate in candidates]
    process_count = min(max(1, int(processes)), max(1, len(output)))
    inner_threads = max(1, int(threads_per_process))
    if process_count <= 1 or len(output) <= 1:
        os.environ["HOUXING_SIGNIFICANCE_THREADS"] = str(inner_threads)
        annotated, report = annotate_detection_significance(
            output,
            branch,
            blocks,
            phase_trials=phase_trials,
            search_trial_factor=search_trial_factor,
            progress=progress,
        )
        report["processes"] = 1
        report["threads_per_process"] = inner_threads
        report["sector_views_cached_per_worker"] = True
        return annotated, report

    # Several contiguous candidates per task amortize executor messaging while
    # leaving enough tasks for dynamic load balancing across uneven periods.
    target_batches = max(process_count, process_count * 4)
    batch_size = max(1, int(math.ceil(len(output) / target_batches)))
    batches = [
        list(enumerate(output))[start : start + batch_size]
        for start in range(0, len(output), batch_size)
    ]
    completed = 0
    results: dict[int, dict] = {}
    # Forkserver avoids forking the NumPy/Numba worker from a multi-threaded
    # parent.  Windows has no forkserver and correctly uses spawn.
    methods = multiprocessing.get_all_start_methods()
    context_name = (
        "forkserver" if "forkserver" in methods else "spawn"
    )
    context = multiprocessing.get_context(context_name)
    with ProcessPoolExecutor(
        max_workers=process_count,
        mp_context=context,
        initializer=_initialize_significance_worker,
        initargs=(branch, blocks, output, inner_threads),
    ) as pool:
        futures = {
            pool.submit(
                _significance_batch_worker,
                batch,
                int(phase_trials),
                int(search_trial_factor),
            ): batch
            for batch in batches
        }
        for future in as_completed(futures):
            batch_results = future.result()
            for index, significance in batch_results:
                results[int(index)] = significance
            completed += len(batch_results)
            if progress:
                progress(
                    "significance",
                    92,
                    (
                        f"Computing red-noise MES-like significance and false-alarm rate "
                        f"for candidate {completed}/{len(output)}"
                    ),
                    {
                        "title": "Fixed-ephemeris detection significance",
                        "command": (
                            "significance --locked-ephemeris "
                            f"--phase-trials {phase_trials} "
                            f"--processes {process_count} "
                            f"--threads-per-process {inner_threads}"
                        ),
                        "metrics": {
                            "candidate": completed,
                            "total": len(output),
                            "processes": process_count,
                            "threads_per_process": inner_threads,
                        },
                    },
                )

    if len(results) != len(output):
        raise RuntimeError(
            "parallel significance did not return every candidate: "
            f"{len(results)}/{len(output)}"
        )
    for index, candidate in enumerate(output):
        candidate["detection_significance"] = results[index]
    if _locked_snapshot(output) != original:
        raise AssertionError("Detection significance changed candidate rank or ephemeris")
    return output, {
        "schema_version": "locked-red-noise-significance-stage-v1",
        "policy": "annotation only; no candidate deletion, reranking, or ephemeris tuning",
        "candidate_count": len(output),
        "phase_trials_each": int(phase_trials),
        "search_trial_factor": int(search_trial_factor),
        "passes_7p1_reference": sum(
            bool(row["detection_significance"]["passes_7p1_reference"])
            for row in output
        ),
        "locked_ephemerides_verified": True,
        "official_spoc_mes": False,
        "processes": process_count,
        "threads_per_process": inner_threads,
        "sector_views_cached_per_worker": True,
        "candidate_order_restored_by_index": True,
    }


__all__ = [
    "REFERENCE_MES",
    "annotate_detection_significance",
    "annotate_detection_significance_parallel",
    "annotate_detection_significance_cuda_pipeline",
    "candidate_detection_significance",
]
