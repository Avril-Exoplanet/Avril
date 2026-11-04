"""Physics-based false-positive checks for transit candidates.

The routines in this module are deliberately deterministic and return both a
boolean decision and the measurements behind it.  They are intended to run
*before* an iterative search accepts and masks a candidate.

The checks are conservative: a noisy or unavailable diagnostic is reported as
``available=False`` and does not silently become evidence against a candidate.
Whether an unavailable diagnostic is allowed in a particular product tier is a
policy decision made by the caller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class VettingThresholds:
    """All tunable v0.2 veto thresholds in one serialisable object.

    These values are starting values.  ``calibrate_vetting.py`` may change them
    on a development set, after which the resulting JSON file is frozen before
    the blind benchmark is opened.
    """

    odd_even_sigma: float = 3.0
    odd_even_relative_depth: float = 0.20
    secondary_snr: float = 5.0
    secondary_depth_ratio: float = 0.10
    vshape_ingress_fraction: float = 0.25
    vshape_delta_chi2: float = 9.0
    vshape_min_snr: float = 7.0
    centroid_sigma: float = 8.0
    centroid_shift_pixels: float = 0.0001
    centroid_direction_consistency: float = 0.60
    min_crowdsap: float = 0.65

    def to_dict(self) -> dict:
        return asdict(self)


def _finite(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


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


def phase_days(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    """Phase in days on ``[-period/2, period/2)``."""
    return (np.asarray(time, dtype=float) - epoch + 0.5 * period) % period - 0.5 * period


def _local_depth(
    phase: np.ndarray,
    flux: np.ndarray,
    duration: float,
    *,
    center: float = 0.0,
) -> tuple[float, float, int, int]:
    """Robust depth and S/N around one phase center.

    The comparison samples are close enough to share the same local baseline but
    are kept away from ingress and egress.
    """
    distance = np.abs(phase - center)
    inside = distance <= 0.5 * duration
    nearby = (distance >= 1.25 * duration) & (distance <= 3.0 * duration)
    good = _finite(flux)
    inside &= good
    nearby &= good
    if inside.sum() < 3 or nearby.sum() < 10:
        return np.nan, np.nan, int(inside.sum()), int(nearby.sum())
    depth = float(np.median(flux[nearby]) - np.median(flux[inside]))
    scatter = _mad_sigma(flux[nearby])
    snr = depth / (scatter / np.sqrt(inside.sum())) if scatter > 0 else np.nan
    return depth, float(snr), int(inside.sum()), int(nearby.sum())


def _per_event_depths(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return event numbers, local depths, and uncertainty estimates."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    event_number = np.rint((time - epoch) / period).astype(int)
    centers = epoch + event_number * period
    offset = time - centers
    inside_all = np.abs(offset) <= 0.5 * duration
    events: list[int] = []
    depths: list[float] = []
    errors: list[float] = []
    for event in np.unique(event_number[inside_all]):
        use_event = event_number == event
        inside = use_event & (np.abs(offset) <= 0.5 * duration)
        nearby = use_event & (np.abs(offset) >= 1.25 * duration) & (
            np.abs(offset) <= 3.0 * duration
        )
        inside &= np.isfinite(flux)
        nearby &= np.isfinite(flux)
        if inside.sum() < 2 or nearby.sum() < 5:
            continue
        local_scatter = _mad_sigma(flux[nearby])
        if not np.isfinite(local_scatter) or local_scatter <= 0:
            continue
        depth = float(np.median(flux[nearby]) - np.median(flux[inside]))
        # Median standard errors are approximately 1.253*sigma/sqrt(n).
        error = 1.253 * local_scatter * np.sqrt(
            1.0 / inside.sum() + 1.0 / nearby.sum()
        )
        events.append(int(event))
        depths.append(depth)
        errors.append(float(error))
    return (
        np.asarray(events, dtype=int),
        np.asarray(depths, dtype=float),
        np.asarray(errors, dtype=float),
    )


def odd_even_test(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    thresholds: VettingThresholds,
) -> dict:
    """Test whether alternating events have incompatible depths."""
    events, depths, errors = _per_event_depths(time, flux, period, epoch, duration)
    odd = events % 2 != 0
    even = ~odd
    result = {
        "available": False,
        "odd_events": int(odd.sum()),
        "even_events": int(even.sum()),
        "odd_depth_ppm": np.nan,
        "even_depth_ppm": np.nan,
        "difference_ppm": np.nan,
        "difference_sigma": np.nan,
        "relative_difference": np.nan,
        "veto": False,
    }
    if odd.sum() < 2 or even.sum() < 2:
        return result

    def weighted_mean(mask: np.ndarray) -> tuple[float, float]:
        variance = np.maximum(errors[mask] ** 2, np.finfo(float).tiny)
        weights = 1.0 / variance
        mean = float(np.sum(weights * depths[mask]) / np.sum(weights))
        formal = float(np.sqrt(1.0 / np.sum(weights)))
        if mask.sum() >= 3:
            empirical = _mad_sigma(depths[mask]) / np.sqrt(mask.sum())
            formal = max(formal, empirical if np.isfinite(empirical) else 0.0)
        return mean, formal

    odd_depth, odd_error = weighted_mean(odd)
    even_depth, even_error = weighted_mean(even)
    difference = abs(odd_depth - even_depth)
    difference_error = np.hypot(odd_error, even_error)
    significance = difference / difference_error if difference_error > 0 else np.nan
    mean_depth = 0.5 * (abs(odd_depth) + abs(even_depth))
    relative = difference / mean_depth if mean_depth > 0 else np.inf
    veto = bool(
        significance >= thresholds.odd_even_sigma
        and relative >= thresholds.odd_even_relative_depth
    )
    result.update(
        available=True,
        odd_depth_ppm=odd_depth * 1e6,
        even_depth_ppm=even_depth * 1e6,
        difference_ppm=difference * 1e6,
        difference_sigma=float(significance),
        relative_difference=float(relative),
        veto=veto,
    )
    return result


def secondary_eclipse_test(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    primary_depth: float,
    thresholds: VettingThresholds,
    *,
    phase_grid_size: int = 121,
) -> dict:
    """Measure a secondary eclipse at orbital phase 0.5.

    We also *report* the strongest off-primary dip, but it is not a hard veto in
    the iterative multi-planet search.  Before the other planets have been masked,
    an all-phase scan can mistake a different planet for an eccentric secondary.
    The phase-0.5 test remains safe for the common half-period EB failure mode;
    eccentric-secondary evidence is deferred to the post-search review stage.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    orbital_phase = ((time - epoch) / period) % 1.0
    duration_phase = duration / period
    best = (np.nan, -np.inf, np.nan, 0)
    half_phase = (0.5, np.nan, np.nan, 0)
    centers = np.linspace(0.08, 0.92, phase_grid_size)
    # Make sure the physically common circular-orbit location is tested exactly.
    centers = np.unique(np.r_[centers, 0.5])
    for center in centers:
        circular_distance = np.abs((orbital_phase - center + 0.5) % 1.0 - 0.5)
        primary_distance = np.abs((orbital_phase + 0.5) % 1.0 - 0.5)
        inside = circular_distance <= 0.5 * duration_phase
        nearby = (circular_distance >= 1.25 * duration_phase) & (
            circular_distance <= 3.0 * duration_phase
        )
        # Never let the primary transit set the comparison baseline.
        nearby &= primary_distance > 1.5 * duration_phase
        inside &= np.isfinite(flux)
        nearby &= np.isfinite(flux)
        if inside.sum() < 3 or nearby.sum() < 10:
            continue
        depth = float(np.median(flux[nearby]) - np.median(flux[inside]))
        scatter = _mad_sigma(flux[nearby])
        snr = depth / (scatter / np.sqrt(inside.sum())) if scatter > 0 else np.nan
        if np.isclose(center, 0.5):
            half_phase = (0.5, float(snr), depth, int(inside.sum()))
        if np.isfinite(snr) and snr > best[1]:
            best = (float(center), float(snr), depth, int(inside.sum()))

    if not np.isfinite(half_phase[1]):
        return {
            "available": False,
            "phase": 0.5,
            "depth_ppm": np.nan,
            "snr": np.nan,
            "depth_ratio": np.nan,
            "points": 0,
            "strongest_offphase_phase": best[0],
            "strongest_offphase_snr": best[1],
            "strongest_offphase_depth_ppm": best[2] * 1e6,
            "veto": False,
        }
    ratio = half_phase[2] / primary_depth if primary_depth > 0 else np.nan
    veto = bool(
        half_phase[1] >= thresholds.secondary_snr
        and np.isfinite(ratio)
        and ratio >= thresholds.secondary_depth_ratio
    )
    return {
        "available": True,
        "phase": 0.5,
        "depth_ppm": half_phase[2] * 1e6,
        "snr": half_phase[1],
        "depth_ratio": float(ratio),
        "points": half_phase[3],
        "strongest_offphase_phase": best[0],
        "strongest_offphase_snr": best[1],
        "strongest_offphase_depth_ppm": best[2] * 1e6,
        "veto": veto,
    }


def _binned_profile(
    phase: np.ndarray,
    flux: np.ndarray,
    duration: float,
    bins: int = 121,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = 2.5 * duration
    edges = np.linspace(-width, width, bins + 1)
    index = np.digitize(phase, edges) - 1
    x: list[float] = []
    y: list[float] = []
    error: list[float] = []
    global_scatter = _mad_sigma(flux[np.abs(phase) >= 1.25 * duration])
    for i in range(bins):
        values = flux[(index == i) & np.isfinite(flux)]
        if values.size < 2:
            continue
        scatter = _mad_sigma(values)
        if not np.isfinite(scatter) or scatter <= 0:
            scatter = global_scatter
        x.append(float(0.5 * (edges[i] + edges[i + 1])))
        y.append(float(np.median(values)))
        error.append(float(1.253 * scatter / np.sqrt(values.size)))
    if not error:
        return np.array([]), np.array([]), np.array([])
    error_array = np.asarray(error, dtype=float)
    positive = error_array[np.isfinite(error_array) & (error_array > 0)]
    floor = np.median(positive) * 0.25 if positive.size else 1e-6
    error_array = np.maximum(error_array, max(floor, 1e-8))
    return np.asarray(x), np.asarray(y), error_array


def trapezoid_shape_test(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    transit_snr: float,
    thresholds: VettingThresholds,
) -> dict:
    """Fit a symmetric trapezoid and reject only a *resolved* V shape.

    ``ingress_fraction`` is one ingress duration divided by total first-to-fourth
    contact duration.  A triangle has 0.5; a box approaches zero.  The profile
    likelihood comparison makes noisy shallow transits inconclusive rather than
    automatically V-shaped.
    """
    phase = phase_days(time, period, epoch)
    use = np.abs(phase) <= 2.5 * duration
    x, y, error = _binned_profile(phase[use], np.asarray(flux)[use], duration)
    if x.size < 20:
        return {
            "available": False,
            "ingress_fraction": np.nan,
            "duration_hours": np.nan,
            "depth_ppm": np.nan,
            "delta_chi2_vs_u": np.nan,
            "veto": False,
        }

    weights = 1.0 / error**2
    best = (np.inf, np.nan, np.nan, np.nan)
    best_u = np.inf
    total_durations = duration * np.linspace(0.70, 1.45, 25)
    ingress_fractions = np.linspace(0.05, 0.50, 19)
    for total_duration in total_durations:
        half = 0.5 * total_duration
        abs_x = np.abs(x)
        for ingress_fraction in ingress_fractions:
            ingress = ingress_fraction * total_duration
            flat_half = max(0.0, half - ingress)
            shape = np.zeros_like(x)
            shape[abs_x <= flat_half] = 1.0
            slope = (abs_x > flat_half) & (abs_x < half)
            if ingress > 0:
                shape[slope] = (half - abs_x[slope]) / ingress
            design = np.column_stack([np.ones_like(shape), -shape])
            normal = design.T @ (weights[:, None] * design)
            rhs = design.T @ (weights * y)
            try:
                baseline, depth = np.linalg.solve(normal, rhs)
            except np.linalg.LinAlgError:
                continue
            model = baseline - depth * shape
            chi2 = float(np.sum(weights * (y - model) ** 2))
            if depth <= 0:
                continue
            if chi2 < best[0]:
                best = (chi2, float(ingress_fraction), float(total_duration), float(depth))
            if ingress_fraction <= 0.30 and chi2 < best_u:
                best_u = chi2

    if not np.isfinite(best[0]):
        return {
            "available": False,
            "ingress_fraction": np.nan,
            "duration_hours": np.nan,
            "depth_ppm": np.nan,
            "delta_chi2_vs_u": np.nan,
            "veto": False,
        }
    delta = max(0.0, best_u - best[0]) if np.isfinite(best_u) else np.nan
    veto = bool(
        best[1] >= thresholds.vshape_ingress_fraction
        and delta >= thresholds.vshape_delta_chi2
        and transit_snr >= thresholds.vshape_min_snr
    )
    return {
        "available": True,
        "ingress_fraction": best[1],
        "duration_hours": best[2] * 24.0,
        "depth_ppm": best[3] * 1e6,
        "delta_chi2_vs_u": float(delta),
        "veto": veto,
    }


def centroid_motion_test(
    time: np.ndarray,
    centroid_x: np.ndarray | None,
    centroid_y: np.ndarray | None,
    period: float,
    epoch: float,
    duration: float,
    thresholds: VettingThresholds,
) -> dict:
    """Compare in-transit and adjacent out-of-transit image centroids.

    One vector is measured per transit.  Requiring both a physical displacement
    floor and repeated direction suppresses false alarms from tiny but formally
    significant pointing residuals.
    """
    null = {
        "available": False,
        "events": 0,
        "dx_pixels": np.nan,
        "dy_pixels": np.nan,
        "shift_pixels": np.nan,
        "shift_arcsec": np.nan,
        "significance": np.nan,
        "direction_consistency": np.nan,
        "veto": False,
    }
    if centroid_x is None or centroid_y is None:
        return null
    time = np.asarray(time, dtype=float)
    x = np.asarray(centroid_x, dtype=float)
    y = np.asarray(centroid_y, dtype=float)
    event_number = np.rint((time - epoch) / period).astype(int)
    offset = time - (epoch + event_number * period)
    event_vectors: list[tuple[float, float, float, float]] = []
    for event in np.unique(event_number[np.abs(offset) <= 0.5 * duration]):
        same = event_number == event
        inside = same & (np.abs(offset) <= 0.5 * duration) & _finite(x, y)
        nearby = same & (np.abs(offset) >= 1.25 * duration) & (
            np.abs(offset) <= 3.0 * duration
        ) & _finite(x, y)
        if inside.sum() < 2 or nearby.sum() < 5:
            continue
        dx = float(np.median(x[inside]) - np.median(x[nearby]))
        dy = float(np.median(y[inside]) - np.median(y[nearby]))
        sx = _mad_sigma(x[nearby])
        sy = _mad_sigma(y[nearby])
        ex = 1.253 * sx * np.sqrt(1.0 / inside.sum() + 1.0 / nearby.sum())
        ey = 1.253 * sy * np.sqrt(1.0 / inside.sum() + 1.0 / nearby.sum())
        if np.isfinite(ex) and np.isfinite(ey) and ex > 0 and ey > 0:
            event_vectors.append((dx, dy, float(ex), float(ey)))
    if len(event_vectors) < 2:
        return {**null, "events": len(event_vectors)}

    vectors = np.asarray(event_vectors, dtype=float)
    wx = 1.0 / vectors[:, 2] ** 2
    wy = 1.0 / vectors[:, 3] ** 2
    dx = float(np.sum(wx * vectors[:, 0]) / np.sum(wx))
    dy = float(np.sum(wy * vectors[:, 1]) / np.sum(wy))
    ex = float(np.sqrt(1.0 / np.sum(wx)))
    ey = float(np.sqrt(1.0 / np.sum(wy)))
    if len(vectors) >= 3:
        empirical_x = _mad_sigma(vectors[:, 0]) / np.sqrt(len(vectors))
        empirical_y = _mad_sigma(vectors[:, 1]) / np.sqrt(len(vectors))
        if np.isfinite(empirical_x):
            ex = max(ex, empirical_x)
        if np.isfinite(empirical_y):
            ey = max(ey, empirical_y)
    shift = float(np.hypot(dx, dy))
    significance = float(np.hypot(dx / ex, dy / ey))
    if shift > 0:
        projections = (vectors[:, 0] * dx + vectors[:, 1] * dy) / shift
        direction_consistency = float(np.mean(projections > 0))
    else:
        direction_consistency = 0.0
    veto = bool(
        significance >= thresholds.centroid_sigma
        and shift >= thresholds.centroid_shift_pixels
        and direction_consistency >= thresholds.centroid_direction_consistency
    )
    return {
        "available": True,
        "events": int(len(vectors)),
        "dx_pixels": dx,
        "dy_pixels": dy,
        "shift_pixels": shift,
        "shift_arcsec": shift * 21.0,
        "significance": significance,
        "direction_consistency": direction_consistency,
        "veto": veto,
    }


def crowding_test(
    crowdsap: np.ndarray | float | None,
    primary_depth: float,
    thresholds: VettingThresholds,
) -> dict:
    """Flag an unresolved neighbour capable of producing the observed dip.

    CROWDSAP is the estimated fraction of aperture flux from the target.  It does
    not identify the eclipsed source, so this check is intentionally an
    *unresolved contamination veto* for the high-confidence tier.  A later target
    pixel difference-image test can clear such a candidate.
    """
    if crowdsap is None:
        values = np.array([], dtype=float)
    else:
        values = np.asarray(crowdsap, dtype=float).reshape(-1)
        values = values[np.isfinite(values) & (values > 0) & (values <= 1.2)]
    if values.size == 0:
        return {
            "available": False,
            "crowdsap": np.nan,
            "contamination_fraction": np.nan,
            "required_neighbor_eclipse_fraction": np.nan,
            "neighbor_can_mimic": False,
            "veto": False,
        }
    crowd = float(np.median(values))
    contamination = max(0.0, 1.0 - min(crowd, 1.0))
    required = primary_depth / contamination if contamination > 0 else np.inf
    can_mimic = bool(primary_depth > 0 and required <= 1.0)
    veto = bool(crowd < thresholds.min_crowdsap and can_mimic)
    return {
        "available": True,
        "crowdsap": crowd,
        "contamination_fraction": contamination,
        "required_neighbor_eclipse_fraction": float(required),
        "neighbor_can_mimic": can_mimic,
        "veto": veto,
    }


def vet_candidate(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    primary_depth: float,
    transit_snr: float,
    *,
    centroid_x: np.ndarray | None = None,
    centroid_y: np.ndarray | None = None,
    crowdsap: np.ndarray | float | None = None,
    thresholds: VettingThresholds | None = None,
) -> dict:
    """Run all five checks and return one JSON-safe diagnostic dictionary."""
    thresholds = thresholds or VettingThresholds()
    odd_even = odd_even_test(time, flux, period, epoch, duration, thresholds)
    secondary = secondary_eclipse_test(
        time, flux, period, epoch, duration, primary_depth, thresholds
    )
    vshape = trapezoid_shape_test(
        time, flux, period, epoch, duration, transit_snr, thresholds
    )
    centroid = centroid_motion_test(
        time, centroid_x, centroid_y, period, epoch, duration, thresholds
    )
    crowding = crowding_test(crowdsap, primary_depth, thresholds)
    veto_names = [
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
    return {
        "thresholds": thresholds.to_dict(),
        "odd_even": odd_even,
        "secondary_eclipse": secondary,
        "v_shape": vshape,
        "centroid_motion": centroid,
        "neighbor_contamination": crowding,
        "vetoes": veto_names,
        "passed": not veto_names,
    }
