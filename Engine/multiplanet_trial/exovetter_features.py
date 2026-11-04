"""Lightweight reference vetting for locked VLora candidates.

This module uses only the small numerical diagnostics from STScI's
``exovetter`` package.  It deliberately excludes learned classifiers and must
not remove, reorder, or retune discovery candidates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def _finite_or_none(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _phase_distance(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    return (time - epoch + 0.5 * period) % period - 0.5 * period


def _candidate_depth(candidate: Mapping) -> float:
    features = candidate.get("features") or {}
    depth_ppm = _finite_or_none(features.get("primary_depth_ppm"))
    if depth_ppm is not None and depth_ppm > 0:
        return depth_ppm / 1e6
    verification = candidate.get("holdout_verification") or {}
    depth_ppm = _finite_or_none(verification.get("depth_ppm"))
    if depth_ppm is not None and depth_ppm > 0:
        return depth_ppm / 1e6
    return 0.0


def _compute_modshift_metrics_one_sort(
    modshift,
    time: np.ndarray,
    flux: np.ndarray,
    model: np.ndarray,
    period_days: float,
    epoch_days: float,
    duration_hrs: float,
) -> tuple[dict, np.ndarray]:
    """Compute ExoVetter model-shift metrics with one shared phase sort.

    ``exovetter.modshift.compute_modshift_metrics`` folds and sorts the same
    time array twice: once for the observed flux and once for the transit
    model.  Both calls use identical phase bins and ordering, so the second
    fold/sort is redundant.  This implementation preserves ExoVetter's exact
    bin definitions, sorted summation order, convolution, scatter estimator,
    and thresholds while sharing the candidate's phase calculation and sort.
    """
    from exovetter import utils
    from scipy import integrate as spint

    if not np.all(np.isfinite(time)):
        raise ValueError("time must contain all finite values")
    if not np.all(np.isfinite(flux)):
        raise ValueError("flux must contain all finite values")
    if len(time) != len(flux):
        raise ValueError("time and flux must be of same length")

    num_bins = int(10 * period_days * 24 / duration_hrs)
    bins = np.arange(num_bins + 1) / float(num_bins) * period_days
    phase = utils.compute_phases(time, period_days, epoch_days, offset=0)
    order = np.argsort(phase)
    sorted_phase = phase[order]
    counts = np.histogram(sorted_phase, bins=bins)[0]
    occupied = counts > 0
    bphase = bins[:-1][occupied]
    bflux = np.histogram(
        sorted_phase, bins=bins, weights=flux[order]
    )[0][occupied]
    bmodel = np.histogram(
        sorted_phase, bins=bins, weights=model[order]
    )[0][occupied]
    bmodel = bmodel / counts[occupied]

    integral = -spint.trapezoid(bmodel, bphase)
    bmodel /= integral
    convolution = _sparse_model_convolution(bphase, bflux, bmodel)
    if len(convolution) != len(bphase):
        raise ValueError("conv and bphase must be of same length")
    results = modshift.find_indices_of_key_locations(
        bphase, convolution, duration_hrs
    )
    sigma = modshift.estimate_scatter(
        phase,
        flux,
        results["pri"],
        results["sec"],
        2 * duration_hrs,
    )
    results.update(
        modshift.compute_event_significances(convolution, sigma, results)
    )
    results["false_alarm_threshold"] = modshift.compute_false_alarm_threshold(
        period_days, duration_hrs
    )
    results["Fred"] = np.std(convolution) / np.std(bflux)
    return results, convolution


def _sparse_model_convolution(
    phase: np.ndarray,
    flux: np.ndarray,
    model: np.ndarray,
) -> np.ndarray:
    """Return ExoVetter's circular valid convolution in O(N * K).

    The locked transit model occupies roughly ten phase bins, while the
    upstream implementation applies ``numpy.convolve`` to the full mostly-zero
    vector at O(N^2) cost.  ``K`` is the number of non-zero model bins.  This
    retains every phase bin and performs the same convolution; only explicit
    zero products are omitted.  Dense or very short models use the reference
    implementation because there is no useful sparse advantage.
    """
    if not np.all(np.isfinite(flux)):
        raise ValueError("flux must contain all finite values")
    if not np.all(np.isfinite(model)):
        raise ValueError("model must contain all finite values")
    if len(phase) != len(flux):
        raise ValueError("phase and flux must be of same length")
    if len(flux) != len(model):
        raise ValueError("model and flux must be of same length")

    size = len(flux)
    nonzero = np.flatnonzero(model)
    if size < 64 or nonzero.size * 8 >= size:
        doubled_flux = np.concatenate([flux, flux])
        return np.convolve(doubled_flux, -model, mode="valid")[:-1]

    doubled_flux = np.concatenate([flux, flux])
    convolution = np.zeros(size + 1, dtype=np.result_type(flux, model))
    # ``np.convolve`` reverses the model.  Iterate in that same direction;
    # omitting zero products changes only sub-ulp rounding, not any term.
    for model_index in nonzero[::-1]:
        start = size - 1 - int(model_index)
        convolution += (
            -model[model_index]
            * doubled_flux[start : start + size + 1]
        )
    return convolution[:-1]


def prepare_exovetter_data(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute candidate-invariant array preparation exactly once.

    The previous implementation repeated finite filtering, a full time sort,
    flux normalization, and deterministic SWEET sample construction for every
    candidate.  These operations depend only on the light curve, so sharing
    their immutable results changes no candidate calculation.
    """

    time = np.asarray(data["time"], dtype=float)
    flux = np.asarray(data["flux"], dtype=float)
    good = np.isfinite(time) & np.isfinite(flux)
    time = time[good]
    flux = flux[good]
    order = np.argsort(time)
    time = time[order]
    flux = flux[order]
    centered_flux = flux / float(np.nanmedian(flux)) - 1.0
    sweet_count = min(time.size, 100_000)
    sweet_index = np.linspace(0, time.size - 1, sweet_count, dtype=int)
    return {
        "time": time,
        "centered_flux": centered_flux,
        "sweet_index": sweet_index,
    }


def _warning_labels(
    *,
    primary_sigma: float | None,
    secondary_sigma: float | None,
    tertiary_sigma: float | None,
    positive_sigma: float | None,
    false_alarm_threshold: float | None,
    primary_aligned: bool | None,
    sweet_sigma: list[float | None],
    sweet_depth_ratios: list[float | None],
    phase_coverage: float | None,
) -> tuple[list[str], list[str]]:
    """Return review-only warnings; these are never deletion gates."""
    warnings: list[str] = []
    advisories: list[str] = []
    if primary_aligned is False:
        warnings.append("strongest_dip_not_at_locked_epoch")
    if false_alarm_threshold is not None and primary_sigma is not None:
        competing_dips = [
            value for value in (secondary_sigma, tertiary_sigma) if value is not None
        ]
        strongest_competitor = max(competing_dips, default=0.0)
        if primary_sigma - strongest_competitor < false_alarm_threshold:
            warnings.append("non_unique_primary_event")
        if (
            positive_sigma is not None
            and primary_sigma - positive_sigma < false_alarm_threshold
        ):
            warnings.append("positive_event_competes_with_primary")
        if secondary_sigma is not None and secondary_sigma >= false_alarm_threshold:
            advisories.append("additional_significant_dip")
        if tertiary_sigma is not None and tertiary_sigma >= false_alarm_threshold:
            advisories.append("multiple_significant_dips")
    labels = ("half_period", "period", "twice_period")
    for label, value, depth_ratio in zip(labels, sweet_sigma, sweet_depth_ratios):
        if value is not None and value >= 3.0:
            advisories.append(f"sinusoidal_variability_at_{label}")
        if (
            value is not None
            and value >= 5.0
            and depth_ratio is not None
            and depth_ratio >= 0.5
        ):
            warnings.append(f"material_sinusoidal_variability_at_{label}")
    if phase_coverage is not None and phase_coverage < 0.8:
        warnings.append("incomplete_transit_phase_coverage")
    return warnings, advisories


def exovetter_diagnostics(data: Mapping[str, np.ndarray], candidate: Mapping) -> dict:
    """Run model-shift, transit-coverage, and SWEET on a fixed ephemeris.

    The return value is evidence for review, not a calibrated planet
    probability.  Any unavailable diagnostic is recorded without affecting the
    candidate list.
    """
    try:
        from exovetter import modshift, sweet, transit_coverage
    except Exception as exc:  # pragma: no cover - exercised on optional installs
        return {
            "schema_version": "vlora-exovetter-reference-v1",
            "available": False,
            "status": "dependency_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "warnings": [],
            "policy": "review_only_never_delete_or_retune",
        }

    period = float(candidate["period_days"])
    epoch = float(candidate["epoch_btjd"])
    duration_hours = float(candidate["duration_hours"])
    duration_days = duration_hours / 24.0
    depth = _candidate_depth(candidate)
    prepared = data.get("_exovetter_prepared")
    if isinstance(prepared, Mapping):
        time = np.asarray(prepared["time"], dtype=float)
        centered_flux = np.asarray(prepared["centered_flux"], dtype=float)
        sweet_index = np.asarray(prepared["sweet_index"], dtype=int)
    else:
        prepared = prepare_exovetter_data(data)
        time = prepared["time"]
        centered_flux = prepared["centered_flux"]
        sweet_index = prepared["sweet_index"]
    if (
        time.size < 50
        or period <= 0
        or duration_days <= 0
        or depth <= 0
        or duration_days >= period
    ):
        return {
            "schema_version": "vlora-exovetter-reference-v1",
            "available": False,
            "status": "insufficient_inputs",
            "warnings": [],
            "policy": "review_only_never_delete_or_retune",
        }

    model = np.zeros(time.size, dtype=float)
    inside = np.abs(_phase_distance(time, period, epoch)) <= 0.5 * duration_days
    model[inside] = -depth

    failures: list[dict] = []
    model_shift: dict = {"available": False}
    try:
        metrics, _ = _compute_modshift_metrics_one_sort(
            modshift,
            time,
            centered_flux,
            model,
            period,
            epoch,
            duration_hours,
        )
        red_noise_factor = _finite_or_none(metrics.get("Fred"))
        correction = max(1.0, abs(red_noise_factor or 1.0))
        primary_sigma = _finite_or_none(metrics.get("sigma_pri"))
        secondary_sigma = _finite_or_none(metrics.get("sigma_sec"))
        tertiary_sigma = _finite_or_none(metrics.get("sigma_ter"))
        positive_sigma = _finite_or_none(metrics.get("sigma_pos"))
        primary_phase = _finite_or_none(metrics.get("phase_pri"))
        primary_phase_distance = (
            min(abs(primary_phase), abs(period - primary_phase))
            if primary_phase is not None
            else None
        )
        model_shift = {
            "available": True,
            "primary_dip_sigma": (
                max(0.0, -primary_sigma) / correction
                if primary_sigma is not None
                else None
            ),
            "secondary_dip_sigma": (
                max(0.0, -secondary_sigma) / correction
                if secondary_sigma is not None
                else None
            ),
            "tertiary_dip_sigma": (
                max(0.0, -tertiary_sigma) / correction
                if tertiary_sigma is not None
                else None
            ),
            "positive_event_sigma": (
                max(0.0, positive_sigma) / correction
                if positive_sigma is not None
                else None
            ),
            "false_alarm_threshold": _finite_or_none(
                metrics.get("false_alarm_threshold")
            ),
            "red_noise_factor": red_noise_factor,
            "primary_phase_days": primary_phase,
            "primary_phase_distance_days": primary_phase_distance,
            "primary_aligned_with_locked_epoch": (
                primary_phase_distance <= duration_days
                if primary_phase_distance is not None
                else None
            ),
            "significances_red_noise_corrected": True,
        }
    except Exception as exc:
        failures.append(
            {
                "diagnostic": "model_shift",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    coverage: dict = {"available": False}
    try:
        value, histogram, _ = transit_coverage.calc_coverage(
            time,
            period,
            epoch,
            duration_hours,
            ndur=2,
            nbins=10,
        )
        coverage = {
            "available": True,
            "fraction": _finite_or_none(value),
            "occupied_bins": int(np.count_nonzero(histogram)),
            "total_bins": int(histogram.size),
        }
    except Exception as exc:
        failures.append(
            {
                "diagnostic": "transit_phase_coverage",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    sinusoidal: dict = {"available": False}
    sweet_sigmas: list[float | None] = [None, None, None]
    sweet_depth_ratios: list[float | None] = [None, None, None]
    try:
        # SWEET performs three sinusoidal least-squares fits.  A deterministic
        # 100k-point time sample preserves phase coverage while preventing the
        # cost from scaling linearly through every cadence of multi-year CVZ
        # targets.  Model-shift and the transit tests still use every cadence.
        sweet_count = int(sweet_index.size)
        result = sweet.sweet(
            time[sweet_index],
            centered_flux[sweet_index].copy(),
            period,
            epoch,
            duration_days,
            plot=False,
        )
        rows = []
        for label, row in zip(("half_period", "period", "twice_period"), result):
            sigma = _finite_or_none(row[2])
            rows.append(
                {
                    "timescale": label,
                    "amplitude_fraction": _finite_or_none(row[0]),
                    "amplitude_uncertainty": _finite_or_none(row[1]),
                    "sigma": sigma,
                    "amplitude_to_transit_depth": (
                        abs(float(row[0])) / depth
                        if depth > 0 and math.isfinite(float(row[0]))
                        else None
                    ),
                }
            )
        sweet_sigmas = [row["sigma"] for row in rows]
        sweet_depth_ratios = [
            row["amplitude_to_transit_depth"] for row in rows
        ]
        sinusoidal = {
            "available": True,
            "fits": rows,
            "warning_sigma": 3.0,
            "cadences_used": int(sweet_count),
            "cadences_available": int(time.size),
            "deterministic_time_sampling": sweet_count < time.size,
        }
    except Exception as exc:
        failures.append(
            {
                "diagnostic": "sinusoidal_variability",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    warnings, advisories = _warning_labels(
        primary_sigma=model_shift.get("primary_dip_sigma"),
        secondary_sigma=model_shift.get("secondary_dip_sigma"),
        tertiary_sigma=model_shift.get("tertiary_dip_sigma"),
        positive_sigma=model_shift.get("positive_event_sigma"),
        false_alarm_threshold=model_shift.get("false_alarm_threshold"),
        primary_aligned=model_shift.get("primary_aligned_with_locked_epoch"),
        sweet_sigma=sweet_sigmas,
        sweet_depth_ratios=sweet_depth_ratios,
        phase_coverage=coverage.get("fraction"),
    )
    available_count = sum(
        bool(row.get("available")) for row in (model_shift, coverage, sinusoidal)
    )
    return {
        "schema_version": "vlora-exovetter-reference-v1",
        "available": available_count > 0,
        "status": "complete" if available_count == 3 else "partial",
        "model_shift": model_shift,
        "transit_phase_coverage": coverage,
        "sinusoidal_variability": sinusoidal,
        "warnings": warnings,
        "advisories": advisories,
        "failures": failures,
        "candidate_retained": True,
        "ephemeris_unchanged": True,
        "not_planet_probability": True,
        "policy": "review_only_never_delete_or_retune",
    }


__all__ = ["exovetter_diagnostics", "prepare_exovetter_data"]
