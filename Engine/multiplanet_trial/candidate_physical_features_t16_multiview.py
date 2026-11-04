"""Label-free physical-vetting features for a complete candidate pool.

Unlike the historical search loader, this reader deliberately keeps deep
negative excursions.  Removing them as generic outliers would erase the very
eclipsing-binary examples that the next ranking round needs to learn from.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits

from .vetting import VettingThresholds, vet_candidate


def _finite_or_none(value) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _json_safe(value):
    """Convert NumPy scalars and non-finite diagnostics to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _mean_bin_fast_cadence(
    arrays: dict[str, np.ndarray], target_seconds: float
) -> tuple[dict[str, np.ndarray], dict]:
    """Exposure-bin a single sector only when it is faster than the target.

    Integer exposure bins avoid allocating across the large gaps between
    sectors.  Finite centroids are averaged independently, while CROWDSAP is
    carried through as a sector-level median.  Native two-minute products are
    returned unchanged.
    """
    time = np.asarray(arrays["time"], dtype=np.float64)
    order = np.argsort(time)
    arrays = {key: np.asarray(value)[order] for key, value in arrays.items()}
    time = np.asarray(arrays["time"], dtype=np.float64)
    steps = np.diff(time)
    steps = steps[(steps > 0.0) & (steps < 0.02)]
    cadence_seconds = (
        float(np.median(steps) * 86400.0) if steps.size else float("nan")
    )
    should_bin = bool(
        np.isfinite(cadence_seconds)
        and cadence_seconds < 0.75 * float(target_seconds)
    )
    before = int(time.size)
    if not should_bin or not before:
        return arrays, {
            "input_cadences": before,
            "output_cadences": before,
            "median_input_cadence_seconds": cadence_seconds,
            "actual_output_cadence_seconds": cadence_seconds,
            "binned_to_seconds": None,
        }

    target_days = float(target_seconds) / 86400.0
    bin_id = np.floor((time - float(time[0])) / target_days).astype(np.int64)
    _unique, inverse = np.unique(bin_id, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)

    def finite_mean(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        finite_counts = np.bincount(inverse, weights=finite.astype(np.float64))
        totals = np.bincount(
            inverse, weights=np.where(finite, values, 0.0)
        )
        return np.divide(
            totals,
            finite_counts,
            out=np.full(counts.size, np.nan, dtype=np.float64),
            where=finite_counts > 0,
        )

    binned = {
        "time": np.bincount(inverse, weights=time) / counts,
        "flux": finite_mean(arrays["flux"]),
        "centroid_x": finite_mean(arrays["centroid_x"]),
        "centroid_y": finite_mean(arrays["centroid_y"]),
        "crowdsap": finite_mean(arrays["crowdsap"]),
    }
    return binned, {
        "input_cadences": before,
        "output_cadences": int(counts.size),
        "median_input_cadence_seconds": cadence_seconds,
        "actual_output_cadence_seconds": (
            float(np.median(np.diff(binned["time"])) * 86400.0)
            if binned["time"].size > 1
            else float("nan")
        ),
        "binned_to_seconds": float(target_seconds),
    }


def load_pdcsap_physics(
    paths: Iterable[str | Path],
    *,
    t16_aperture: int | None = None,
    t16_magnitude_column: str | None = None,
    target_cadence_seconds: float = 120.0,
    cadence_diagnostics: list[dict] | None = None,
) -> dict[str, np.ndarray]:
    """Load normalized SPOC or T16 data without clipping eclipses.

    T16 does not provide SPOC moment centroids or CROWDSAP. Those arrays remain
    NaN so the corresponding diagnostics report unavailable instead of
    treating source positions as measured transit centroids.
    """
    columns: dict[str, list[np.ndarray]] = {
        "time": [],
        "flux": [],
        "centroid_x": [],
        "centroid_y": [],
        "crowdsap": [],
    }
    for raw_path in paths:
        path = Path(raw_path)
        with fits.open(path, memmap=True) as hdul:
            table = hdul[1].data
            names = set(table.names)
            is_t16 = "TMID_BJD" in names and "PDCSAP_FLUX" not in names
            if is_t16:
                if t16_magnitude_column is not None:
                    magnitude_columns = [
                        value.strip().upper()
                        for value in str(t16_magnitude_column).split(",")
                        if value.strip()
                    ]
                    allowed = {
                        f"{correction}{aperture}"
                        for correction in ("IRM", "EPD", "TFA")
                        for aperture in (1, 2, 3)
                    }
                    if not magnitude_columns or any(
                        value not in allowed for value in magnitude_columns
                    ):
                        raise ValueError("Invalid T16 magnitude column specification")
                else:
                    if t16_aperture not in {1, 2, 3}:
                        raise ValueError(
                            "A T16 magnitude column or aperture 1, 2, or 3 is required"
                        )
                    magnitude_columns = [f"TFA{int(t16_aperture)}"]
                missing = [value for value in magnitude_columns if value not in names]
                if missing:
                    raise ValueError(f"Missing {', '.join(missing)} in {path.name}")
                time = np.asarray(table["TMID_BJD"], dtype=float)
                aperture_fluxes: list[np.ndarray] = []
                aperture_weights: list[float] = []
                for magnitude_column in magnitude_columns:
                    magnitude = np.asarray(table[magnitude_column], dtype=float)
                    finite_magnitude = np.isfinite(magnitude)
                    reference = float(np.nanmedian(magnitude[finite_magnitude]))
                    aperture_flux = np.power(
                        10.0, -0.4 * (magnitude - reference)
                    )
                    local = aperture_flux[np.isfinite(aperture_flux)]
                    center = float(np.nanmedian(local))
                    aperture_flux = aperture_flux / center
                    scatter = 1.4826 * float(
                        np.nanmedian(np.abs(local / center - 1.0))
                    )
                    aperture_fluxes.append(aperture_flux)
                    aperture_weights.append(
                        1.0 / scatter**2
                        if math.isfinite(scatter) and scatter > 0
                        else 0.0
                    )
                if len(aperture_fluxes) == 1:
                    flux = aperture_fluxes[0]
                else:
                    stacked = np.column_stack(aperture_fluxes)
                    weights = np.asarray(aperture_weights, dtype=float)
                    if not np.any(weights > 0):
                        weights = np.ones(len(aperture_fluxes), dtype=float)
                    finite_flux = np.isfinite(stacked)
                    row_weights = finite_flux * weights[np.newaxis, :]
                    denominators = np.sum(row_weights, axis=1)
                    flux = np.divide(
                        np.nansum(stacked * row_weights, axis=1),
                        denominators,
                        out=np.full(time.size, np.nan, dtype=float),
                        where=denominators > 0,
                    )
            else:
                time = np.asarray(table["TIME"], dtype=float)
                flux = np.asarray(table["PDCSAP_FLUX"], dtype=float)
            quality = (
                np.asarray(table["QUALITY"], dtype=np.int64)
                if "QUALITY" in names
                else np.zeros(time.size, dtype=np.int64)
            )
            centroid_x = (
                np.asarray(table["MOM_CENTR1"], dtype=float)
                if not is_t16 and "MOM_CENTR1" in names
                else np.full(time.size, np.nan)
            )
            centroid_y = (
                np.asarray(table["MOM_CENTR2"], dtype=float)
                if not is_t16 and "MOM_CENTR2" in names
                else np.full(time.size, np.nan)
            )
            if not is_t16 and "POS_CORR1" in names and "POS_CORR2" in names:
                corrected_x = centroid_x - np.asarray(table["POS_CORR1"], dtype=float)
                corrected_y = centroid_y - np.asarray(table["POS_CORR2"], dtype=float)
                centroid_x = np.where(np.isfinite(corrected_x), corrected_x, centroid_x)
                centroid_y = np.where(np.isfinite(corrected_y), corrected_y, centroid_y)
            crowd = (
                np.nan
                if is_t16
                else float(hdul[1].header.get("CROWDSAP", np.nan))
            )

        good = np.isfinite(time) & np.isfinite(flux) & (flux > 0) & (quality == 0)
        if not np.any(good):
            continue
        local_flux = flux[good]
        median_flux = float(np.nanmedian(local_flux))
        if not np.isfinite(median_flux) or median_flux <= 0:
            continue
        sector_arrays = {
            "time": time[good],
            "flux": local_flux / median_flux,
            "centroid_x": centroid_x[good],
            "centroid_y": centroid_y[good],
            "crowdsap": np.full(int(good.sum()), crowd, dtype=float),
        }
        sector_arrays, cadence_row = _mean_bin_fast_cadence(
            sector_arrays, float(target_cadence_seconds)
        )
        cadence_row["path"] = str(path)
        cadence_row["status"] = "ready"
        if cadence_diagnostics is not None:
            cadence_diagnostics.append(cadence_row)
        for key in columns:
            columns[key].append(sector_arrays[key])
    if not columns["time"]:
        raise ValueError("No quality-clean PDCSAP cadences available for physical vetting")
    assembled = {key: np.concatenate(value) for key, value in columns.items()}
    order = np.argsort(assembled["time"])
    return {key: value[order] for key, value in assembled.items()}


def _primary_depth(data: dict[str, np.ndarray], candidate: dict) -> tuple[float, int, int]:
    period = float(candidate["period_days"])
    epoch = float(candidate["epoch_btjd"])
    duration = float(candidate["duration_hours"]) / 24.0
    phase = (data["time"] - epoch + 0.5 * period) % period - 0.5 * period
    distance = np.abs(phase)
    inside = distance <= 0.5 * duration
    nearby = (distance >= 1.25 * duration) & (distance <= 3.0 * duration)
    inside &= np.isfinite(data["flux"])
    nearby &= np.isfinite(data["flux"])
    if inside.sum() < 3 or nearby.sum() < 10:
        return 0.0, int(inside.sum()), int(nearby.sum())
    depth = float(np.nanmedian(data["flux"][nearby]) - np.nanmedian(data["flux"][inside]))
    return max(0.0, depth), int(inside.sum()), int(nearby.sum())


def physical_vetting_features(data: dict[str, np.ndarray], candidate: dict) -> dict:
    """Return the full five-test evidence plus JSON-safe scalar model features."""
    depth, inside_points, nearby_points = _primary_depth(data, candidate)
    transit_snr = float(candidate.get("features", {}).get("median_search_snr") or 0.0)
    evidence = vet_candidate(
        data["time"],
        data["flux"],
        float(candidate["period_days"]),
        float(candidate["epoch_btjd"]),
        float(candidate["duration_hours"]) / 24.0,
        depth,
        transit_snr,
        centroid_x=data["centroid_x"],
        centroid_y=data["centroid_y"],
        crowdsap=data["crowdsap"],
        thresholds=VettingThresholds(),
    )
    odd_even = evidence["odd_even"]
    secondary = evidence["secondary_eclipse"]
    v_shape = evidence["v_shape"]
    centroid = evidence["centroid_motion"]
    crowding = evidence["neighbor_contamination"]
    available_count = sum(
        bool(row.get("available"))
        for row in (odd_even, secondary, v_shape, centroid, crowding)
    )
    scalars = {
        "physical_available_count": available_count,
        "physical_veto_count": len(evidence["vetoes"]),
        "primary_depth_ppm": depth * 1e6,
        "primary_inside_points": inside_points,
        "primary_nearby_points": nearby_points,
        "odd_even_available": bool(odd_even.get("available")),
        "odd_even_difference_sigma": _finite_or_none(odd_even.get("difference_sigma")),
        "odd_even_relative_difference": _finite_or_none(
            odd_even.get("relative_difference")
        ),
        "odd_even_veto": bool(odd_even.get("veto")),
        "secondary_available": bool(secondary.get("available")),
        "secondary_snr": _finite_or_none(secondary.get("snr")),
        "secondary_depth_ratio": _finite_or_none(secondary.get("depth_ratio")),
        "secondary_veto": bool(secondary.get("veto")),
        "v_shape_available": bool(v_shape.get("available")),
        "v_shape_ingress_fraction": _finite_or_none(v_shape.get("ingress_fraction")),
        "v_shape_delta_chi2": _finite_or_none(v_shape.get("delta_chi2_vs_u")),
        "v_shape_veto": bool(v_shape.get("veto")),
        "centroid_available": bool(centroid.get("available")),
        "centroid_significance": _finite_or_none(centroid.get("significance")),
        "centroid_shift_pixels": _finite_or_none(centroid.get("shift_pixels")),
        "centroid_direction_consistency": _finite_or_none(
            centroid.get("direction_consistency")
        ),
        "centroid_veto": bool(centroid.get("veto")),
        "crowding_available": bool(crowding.get("available")),
        "crowdsap": _finite_or_none(crowding.get("crowdsap")),
        "neighbor_can_mimic": bool(crowding.get("neighbor_can_mimic")),
        "crowding_veto": bool(crowding.get("veto")),
    }
    return {"evidence": _json_safe(evidence), "features": scalars}


def annotate_candidates(paths: Iterable[str | Path], candidates: list[dict]) -> list[dict]:
    """Attach label-free physical evidence to each compact candidate."""
    if not candidates:
        return candidates
    data = load_pdcsap_physics(paths)
    output = []
    for candidate in candidates:
        annotated = dict(candidate)
        physical = physical_vetting_features(data, annotated)
        annotated["features"] = {**annotated.get("features", {}), **physical["features"]}
        annotated["physical_vetting"] = physical["evidence"]
        output.append(annotated)
    return output
