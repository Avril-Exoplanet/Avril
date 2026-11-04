"""Compiled survey-tier physical vetting for long TESS light curves.

The historical implementation repeatedly materialised full-cadence boolean
arrays for every transit event and for 121 possible secondary-eclipse phases.
That is scientifically unnecessary for the hard survey gates.  This module
keeps the same odd/even, phase-0.5 secondary, centroid, V-shape, and crowding
decisions while collecting per-event statistics in linear passes.

The all-phase secondary scan remains a useful review plot, but it is explicitly
deferred to the small final follow-up set because it never participated in the
historical hard veto.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from numba import njit

from multiplanet_trial.candidate_physical_features_t16_multiview import _json_safe
from multiplanet_trial.vetting import (
    VettingThresholds,
    crowding_test,
    trapezoid_shape_test,
)


@njit(cache=True, nogil=True)
def _mad_sigma(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    median = np.median(values)
    sigma = 1.4826 * np.median(np.abs(values - median))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = np.std(values)
    return sigma


@njit(cache=True, nogil=True)
def _primary_secondary_kernel(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
) -> tuple:
    """Collect primary and phase-0.5 secondary samples in two O(N) passes."""
    primary_inside_count = 0
    primary_nearby_count = 0
    secondary_inside_count = 0
    secondary_nearby_count = 0

    for i in range(time.size):
        value = flux[i]
        if not np.isfinite(value):
            continue
        phase = (time[i] - epoch + 0.5 * period) % period - 0.5 * period
        distance = abs(phase)
        if distance <= 0.5 * duration:
            primary_inside_count += 1
        elif distance >= 1.25 * duration and distance <= 3.0 * duration:
            primary_nearby_count += 1

        orbital_phase = ((time[i] - epoch) / period) % 1.0
        secondary_distance = abs((orbital_phase - 0.5 + 0.5) % 1.0 - 0.5)
        primary_distance_phase = abs((orbital_phase + 0.5) % 1.0 - 0.5)
        duration_phase = duration / period
        if secondary_distance <= 0.5 * duration_phase:
            secondary_inside_count += 1
        elif (
            secondary_distance >= 1.25 * duration_phase
            and secondary_distance <= 3.0 * duration_phase
            and primary_distance_phase > 1.5 * duration_phase
        ):
            secondary_nearby_count += 1

    primary_inside = np.empty(primary_inside_count, dtype=np.float64)
    primary_nearby = np.empty(primary_nearby_count, dtype=np.float64)
    secondary_inside = np.empty(secondary_inside_count, dtype=np.float64)
    secondary_nearby = np.empty(secondary_nearby_count, dtype=np.float64)
    pi = 0
    pn = 0
    si = 0
    sn = 0
    for i in range(time.size):
        value = flux[i]
        if not np.isfinite(value):
            continue
        phase = (time[i] - epoch + 0.5 * period) % period - 0.5 * period
        distance = abs(phase)
        if distance <= 0.5 * duration:
            primary_inside[pi] = value
            pi += 1
        elif distance >= 1.25 * duration and distance <= 3.0 * duration:
            primary_nearby[pn] = value
            pn += 1

        orbital_phase = ((time[i] - epoch) / period) % 1.0
        secondary_distance = abs((orbital_phase - 0.5 + 0.5) % 1.0 - 0.5)
        primary_distance_phase = abs((orbital_phase + 0.5) % 1.0 - 0.5)
        duration_phase = duration / period
        if secondary_distance <= 0.5 * duration_phase:
            secondary_inside[si] = value
            si += 1
        elif (
            secondary_distance >= 1.25 * duration_phase
            and secondary_distance <= 3.0 * duration_phase
            and primary_distance_phase > 1.5 * duration_phase
        ):
            secondary_nearby[sn] = value
            sn += 1

    primary_depth = 0.0
    if primary_inside_count >= 3 and primary_nearby_count >= 10:
        primary_depth = np.median(primary_nearby) - np.median(primary_inside)
        if primary_depth < 0.0:
            primary_depth = 0.0

    secondary_available = secondary_inside_count >= 3 and secondary_nearby_count >= 10
    secondary_depth = np.nan
    secondary_snr = np.nan
    if secondary_available:
        secondary_depth = np.median(secondary_nearby) - np.median(secondary_inside)
        scatter = _mad_sigma(secondary_nearby)
        if scatter > 0.0:
            secondary_snr = secondary_depth / (
                scatter / math.sqrt(secondary_inside_count)
            )
    return (
        primary_depth,
        primary_inside_count,
        primary_nearby_count,
        secondary_available,
        secondary_depth,
        secondary_snr,
        secondary_inside_count,
    )


@njit(cache=True, nogil=True)
def _event_diagnostics_kernel(
    time: np.ndarray,
    flux: np.ndarray,
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
) -> tuple:
    """Compute odd/even and centroid event summaries without O(events*N)."""
    if time.size == 0:
        return (0, 0, np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan, np.nan)

    # Time is sorted by the loader, so equal event numbers form contiguous runs.
    run_count = 0
    max_run = 0
    start = 0
    while start < time.size:
        event = int(np.rint((time[start] - epoch) / period))
        end = start + 1
        while end < time.size and int(np.rint((time[end] - epoch) / period)) == event:
            end += 1
        run_count += 1
        if end - start > max_run:
            max_run = end - start
        start = end

    event_numbers = np.empty(run_count, dtype=np.int64)
    depths = np.empty(run_count, dtype=np.float64)
    errors = np.empty(run_count, dtype=np.float64)
    odd_valid = 0

    dxs = np.empty(run_count, dtype=np.float64)
    dys = np.empty(run_count, dtype=np.float64)
    exs = np.empty(run_count, dtype=np.float64)
    eys = np.empty(run_count, dtype=np.float64)
    centroid_valid = 0

    flux_inside = np.empty(max_run, dtype=np.float64)
    flux_nearby = np.empty(max_run, dtype=np.float64)
    x_inside = np.empty(max_run, dtype=np.float64)
    y_inside = np.empty(max_run, dtype=np.float64)
    x_nearby = np.empty(max_run, dtype=np.float64)
    y_nearby = np.empty(max_run, dtype=np.float64)

    start = 0
    while start < time.size:
        event = int(np.rint((time[start] - epoch) / period))
        center = epoch + event * period
        end = start + 1
        while end < time.size and int(np.rint((time[end] - epoch) / period)) == event:
            end += 1

        fi = 0
        fn = 0
        ci = 0
        cn = 0
        for i in range(start, end):
            offset = abs(time[i] - center)
            value = flux[i]
            if np.isfinite(value):
                if offset <= 0.5 * duration:
                    flux_inside[fi] = value
                    fi += 1
                elif offset >= 1.25 * duration and offset <= 3.0 * duration:
                    flux_nearby[fn] = value
                    fn += 1
            x = centroid_x[i]
            y = centroid_y[i]
            if np.isfinite(x) and np.isfinite(y):
                if offset <= 0.5 * duration:
                    x_inside[ci] = x
                    y_inside[ci] = y
                    ci += 1
                elif offset >= 1.25 * duration and offset <= 3.0 * duration:
                    x_nearby[cn] = x
                    y_nearby[cn] = y
                    cn += 1

        if fi >= 2 and fn >= 5:
            local_scatter = _mad_sigma(flux_nearby[:fn])
            if np.isfinite(local_scatter) and local_scatter > 0.0:
                event_numbers[odd_valid] = event
                depths[odd_valid] = np.median(flux_nearby[:fn]) - np.median(flux_inside[:fi])
                errors[odd_valid] = 1.253 * local_scatter * math.sqrt(1.0 / fi + 1.0 / fn)
                odd_valid += 1

        if ci >= 2 and cn >= 5:
            sx = _mad_sigma(x_nearby[:cn])
            sy = _mad_sigma(y_nearby[:cn])
            ex = 1.253 * sx * math.sqrt(1.0 / ci + 1.0 / cn)
            ey = 1.253 * sy * math.sqrt(1.0 / ci + 1.0 / cn)
            if np.isfinite(ex) and np.isfinite(ey) and ex > 0.0 and ey > 0.0:
                dxs[centroid_valid] = np.median(x_inside[:ci]) - np.median(x_nearby[:cn])
                dys[centroid_valid] = np.median(y_inside[:ci]) - np.median(y_nearby[:cn])
                exs[centroid_valid] = ex
                eys[centroid_valid] = ey
                centroid_valid += 1
        start = end

    odd_count = 0
    even_count = 0
    odd_weight = 0.0
    even_weight = 0.0
    odd_weighted_depth = 0.0
    even_weighted_depth = 0.0
    for i in range(odd_valid):
        weight = 1.0 / max(errors[i] * errors[i], np.finfo(np.float64).tiny)
        if event_numbers[i] % 2 != 0:
            odd_count += 1
            odd_weight += weight
            odd_weighted_depth += weight * depths[i]
        else:
            even_count += 1
            even_weight += weight
            even_weighted_depth += weight * depths[i]

    odd_depth = np.nan
    even_depth = np.nan
    difference_sigma = np.nan
    relative_difference = np.nan
    if odd_count >= 2 and even_count >= 2:
        odd_depth = odd_weighted_depth / odd_weight
        even_depth = even_weighted_depth / even_weight
        odd_error = math.sqrt(1.0 / odd_weight)
        even_error = math.sqrt(1.0 / even_weight)
        if odd_count >= 3:
            local = np.empty(odd_count, dtype=np.float64)
            at = 0
            for i in range(odd_valid):
                if event_numbers[i] % 2 != 0:
                    local[at] = depths[i]
                    at += 1
            empirical = _mad_sigma(local) / math.sqrt(odd_count)
            if np.isfinite(empirical) and empirical > odd_error:
                odd_error = empirical
        if even_count >= 3:
            local = np.empty(even_count, dtype=np.float64)
            at = 0
            for i in range(odd_valid):
                if event_numbers[i] % 2 == 0:
                    local[at] = depths[i]
                    at += 1
            empirical = _mad_sigma(local) / math.sqrt(even_count)
            if np.isfinite(empirical) and empirical > even_error:
                even_error = empirical
        difference = abs(odd_depth - even_depth)
        difference_error = math.hypot(odd_error, even_error)
        if difference_error > 0.0:
            difference_sigma = difference / difference_error
        mean_depth = 0.5 * (abs(odd_depth) + abs(even_depth))
        if mean_depth > 0.0:
            relative_difference = difference / mean_depth
        else:
            relative_difference = np.inf

    centroid_dx = np.nan
    centroid_dy = np.nan
    centroid_significance = np.nan
    centroid_direction = np.nan
    if centroid_valid >= 2:
        wx_sum = 0.0
        wy_sum = 0.0
        dx_sum = 0.0
        dy_sum = 0.0
        for i in range(centroid_valid):
            wx = 1.0 / (exs[i] * exs[i])
            wy = 1.0 / (eys[i] * eys[i])
            wx_sum += wx
            wy_sum += wy
            dx_sum += wx * dxs[i]
            dy_sum += wy * dys[i]
        centroid_dx = dx_sum / wx_sum
        centroid_dy = dy_sum / wy_sum
        ex = math.sqrt(1.0 / wx_sum)
        ey = math.sqrt(1.0 / wy_sum)
        if centroid_valid >= 3:
            empirical_x = _mad_sigma(dxs[:centroid_valid]) / math.sqrt(centroid_valid)
            empirical_y = _mad_sigma(dys[:centroid_valid]) / math.sqrt(centroid_valid)
            if np.isfinite(empirical_x) and empirical_x > ex:
                ex = empirical_x
            if np.isfinite(empirical_y) and empirical_y > ey:
                ey = empirical_y
        centroid_significance = math.hypot(centroid_dx / ex, centroid_dy / ey)
        shift = math.hypot(centroid_dx, centroid_dy)
        if shift > 0.0:
            consistent = 0
            for i in range(centroid_valid):
                if (dxs[i] * centroid_dx + dys[i] * centroid_dy) / shift > 0.0:
                    consistent += 1
            centroid_direction = consistent / centroid_valid
        else:
            centroid_direction = 0.0

    return (
        odd_count,
        even_count,
        odd_depth,
        even_depth,
        difference_sigma,
        relative_difference,
        centroid_valid,
        centroid_dx,
        centroid_dy,
        centroid_significance,
        centroid_direction,
    )


def fast_survey_physical_vetting(
    data: Mapping[str, np.ndarray], candidate: Mapping
) -> dict:
    """Return survey hard-gate evidence with the locked ephemeris unchanged."""
    thresholds = VettingThresholds()
    period = float(candidate["period_days"])
    epoch = float(candidate["epoch_btjd"])
    duration = float(candidate["duration_hours"]) / 24.0
    time = np.ascontiguousarray(data["time"], dtype=np.float64)
    flux = np.ascontiguousarray(data["flux"], dtype=np.float64)
    centroid_x = np.ascontiguousarray(data["centroid_x"], dtype=np.float64)
    centroid_y = np.ascontiguousarray(data["centroid_y"], dtype=np.float64)

    primary = _primary_secondary_kernel(time, flux, period, epoch, duration)
    depth = float(primary[0])
    secondary_available = bool(primary[3])
    secondary_depth = float(primary[4])
    secondary_snr = float(primary[5])
    secondary_ratio = secondary_depth / depth if depth > 0 else np.nan
    secondary_veto = bool(
        secondary_available
        and secondary_snr >= thresholds.secondary_snr
        and np.isfinite(secondary_ratio)
        and secondary_ratio >= thresholds.secondary_depth_ratio
    )
    secondary = {
        "available": secondary_available,
        "phase": 0.5,
        "depth_ppm": secondary_depth * 1e6,
        "snr": secondary_snr,
        "depth_ratio": secondary_ratio,
        "points": int(primary[6]),
        "strongest_offphase_phase": None,
        "strongest_offphase_snr": None,
        "strongest_offphase_depth_ppm": None,
        "offphase_scan": "deferred_to_final_followup_non_veto_diagnostic",
        "veto": secondary_veto,
    }

    event = _event_diagnostics_kernel(
        time, flux, centroid_x, centroid_y, period, epoch, duration
    )
    odd_available = int(event[0]) >= 2 and int(event[1]) >= 2
    odd_difference = (
        abs(float(event[2]) - float(event[3])) if odd_available else np.nan
    )
    odd_relative = float(event[5])
    odd_sigma = float(event[4])
    odd_veto = bool(
        odd_available
        and odd_sigma >= thresholds.odd_even_sigma
        and odd_relative >= thresholds.odd_even_relative_depth
    )
    odd_even = {
        "available": odd_available,
        "odd_events": int(event[0]),
        "even_events": int(event[1]),
        "odd_depth_ppm": float(event[2]) * 1e6,
        "even_depth_ppm": float(event[3]) * 1e6,
        "difference_ppm": odd_difference * 1e6,
        "difference_sigma": odd_sigma,
        "relative_difference": odd_relative,
        "veto": odd_veto,
    }

    centroid_events = int(event[6])
    centroid_available = centroid_events >= 2
    dx = float(event[7])
    dy = float(event[8])
    shift = math.hypot(dx, dy) if centroid_available else np.nan
    centroid_sigma = float(event[9])
    centroid_direction = float(event[10])
    centroid_veto = bool(
        centroid_available
        and centroid_sigma >= thresholds.centroid_sigma
        and shift >= thresholds.centroid_shift_pixels
        and centroid_direction >= thresholds.centroid_direction_consistency
    )
    centroid = {
        "available": centroid_available,
        "events": centroid_events,
        "dx_pixels": dx,
        "dy_pixels": dy,
        "shift_pixels": shift,
        "shift_arcsec": shift * 21.0,
        "significance": centroid_sigma,
        "direction_consistency": centroid_direction,
        "veto": centroid_veto,
    }

    transit_snr = float(
        (candidate.get("features") or {}).get("median_search_snr") or 0.0
    )
    vshape = trapezoid_shape_test(
        time, flux, period, epoch, duration, transit_snr, thresholds
    )
    crowding = crowding_test(data.get("crowdsap"), depth, thresholds)
    vetoes = [
        name
        for name, result in (
            ("odd_even", odd_even),
            ("secondary_eclipse", secondary),
            ("v_shape", vshape),
            ("centroid_motion", centroid),
            ("neighbor_contamination", crowding),
        )
        if result["veto"]
    ]
    evidence = {
        "thresholds": thresholds.to_dict(),
        "odd_even": odd_even,
        "secondary_eclipse": secondary,
        "v_shape": vshape,
        "centroid_motion": centroid,
        "neighbor_contamination": crowding,
        "vetoes": vetoes,
        "passed": not vetoes,
        "tier": "compiled_survey_hard_gates",
    }
    features = {
        "primary_depth_ppm": depth * 1e6,
        "primary_inside_points": int(primary[1]),
        "primary_nearby_points": int(primary[2]),
        "odd_even_available": odd_available,
        "odd_even_difference_sigma": odd_sigma,
        "odd_even_relative_difference": odd_relative,
        "odd_even_veto": odd_veto,
        "secondary_available": secondary_available,
        "secondary_snr": secondary_snr,
        "secondary_depth_ratio": secondary_ratio,
        "secondary_veto": secondary_veto,
        "v_shape_available": bool(vshape.get("available")),
        "v_shape_ingress_fraction": vshape.get("ingress_fraction"),
        "v_shape_delta_chi2": vshape.get("delta_chi2_vs_u"),
        "v_shape_veto": bool(vshape.get("veto")),
        "centroid_available": centroid_available,
        "centroid_significance": centroid_sigma,
        "centroid_shift_pixels": shift,
        "centroid_direction_consistency": centroid_direction,
        "centroid_veto": centroid_veto,
        "crowding_available": bool(crowding.get("available")),
        "crowdsap": crowding.get("crowdsap"),
        "neighbor_can_mimic": bool(crowding.get("neighbor_can_mimic")),
        "crowding_veto": bool(crowding.get("veto")),
    }
    return {"evidence": _json_safe(evidence), "features": _json_safe(features)}


__all__ = ["fast_survey_physical_vetting"]
