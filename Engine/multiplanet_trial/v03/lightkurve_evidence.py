"""Lightkurve-backed soft evidence for the experimental v0.3 pipeline.

The frozen v0.2 search remains the primary PDCSAP detector.  This module never
changes or hard-rejects those detections.  It asks two narrower questions:

1. Does the locked ephemeris survive independent Lightkurve preprocessing of
   both PDCSAP and SAP photometry?
2. In a target-pixel file, is the dimming concentrated near the target source?

These are review signals, not planet confirmations.  In particular, a
target-consistent difference image does not rule out an unresolved companion.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import lightkurve as lk
import numpy as np


@dataclass(frozen=True)
class FluxBranch:
    """One independently prepared light-curve branch."""

    key: str
    label: str
    source_column: str
    method: str
    time: np.ndarray
    flux: np.ndarray
    sector_id: np.ndarray


def _mad_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    median = np.nanmedian(finite)
    sigma = 1.4826 * np.nanmedian(np.abs(finite - median))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.nanstd(finite)
    return float(sigma if np.isfinite(sigma) and sigma > 0 else 0.0)


def _odd_window(time: np.ndarray, window_days: float, length: int) -> int:
    finite_time = np.asarray(time, dtype=float)
    finite_time = finite_time[np.isfinite(finite_time)]
    if finite_time.size < 3:
        return 0
    cadence = float(np.nanmedian(np.diff(np.sort(finite_time))))
    if not np.isfinite(cadence) or cadence <= 0:
        return 0
    window = max(9, int(round(float(window_days) / cadence)))
    if window % 2 == 0:
        window += 1
    maximum = length if length % 2 == 1 else length - 1
    return int(min(window, maximum)) if maximum >= 9 else 0


def _prepare_sector(path: Path, flux_column: str, window_days: float):
    """Read one SPOC sector through Lightkurve and flatten it independently."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lightcurve = lk.read(path, flux_column=flux_column)
    if lightcurve is None:
        raise ValueError(f"Lightkurve could not read {path}")
    lightcurve = lightcurve.remove_nans().normalize()
    window = _odd_window(lightcurve.time.value, window_days, len(lightcurve))
    if window:
        lightcurve = lightcurve.flatten(
            window_length=window,
            polyorder=2,
            break_tolerance=5,
        )
    # Preserve negative transit-like excursions more generously than positive
    # spacecraft glitches and flares, matching the frozen search's intent.
    lightcurve = lightcurve.remove_outliers(sigma_upper=7.0, sigma_lower=12.0)
    time = np.asarray(lightcurve.time.value, dtype=float)
    flux = np.asarray(lightcurve.flux.value, dtype=float)
    sector = int(lightcurve.meta.get("SECTOR", -1))
    good = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
    order = np.argsort(time[good])
    return time[good][order], flux[good][order], sector


def load_lightkurve_branches(
    paths: Iterable[str | Path],
    *,
    flatten_window_days: float = 1.0,
) -> dict[str, FluxBranch]:
    """Load independent PDCSAP and SAP validation branches.

    The official Lightkurve quality mask is applied while reading.  Each sector
    is normalized and flattened separately so a long gap cannot bridge two
    observing seasons.
    """
    specifications = {
        "pdcsap_lightkurve": {
            "label": "PDCSAP + Lightkurve detrending",
            "source_column": "pdcsap_flux",
        },
        "sap_lightkurve": {
            "label": "SAP + Lightkurve detrending",
            "source_column": "sap_flux",
        },
    }
    output: dict[str, FluxBranch] = {}
    resolved_paths = [Path(path) for path in paths]
    if not resolved_paths:
        raise ValueError("At least one SPOC light-curve path is required")
    for key, specification in specifications.items():
        times: list[np.ndarray] = []
        fluxes: list[np.ndarray] = []
        sector_ids: list[np.ndarray] = []
        for path in resolved_paths:
            time, flux, sector = _prepare_sector(
                path,
                specification["source_column"],
                flatten_window_days,
            )
            times.append(time)
            fluxes.append(flux)
            sector_ids.append(np.full(time.size, sector, dtype=int))
        time = np.concatenate(times)
        flux = np.concatenate(fluxes)
        sector_id = np.concatenate(sector_ids)
        order = np.argsort(time)
        output[key] = FluxBranch(
            key=key,
            label=specification["label"],
            source_column=specification["source_column"],
            method=f"lightkurve-{lk.__version__}: normalize + flatten({flatten_window_days:g} d)",
            time=time[order],
            flux=flux[order],
            sector_id=sector_id[order],
        )
    return output


def measure_fixed_ephemeris(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period_days: float,
    epoch_btjd: float,
    duration_hours: float,
) -> dict:
    """Measure a locked candidate without performing a fresh period search."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    duration_days = float(duration_hours) / 24.0
    period_days = float(period_days)
    phase = (time - float(epoch_btjd) + 0.5 * period_days) % period_days - 0.5 * period_days
    inside = np.abs(phase) <= 0.5 * duration_days
    nearby = (np.abs(phase) >= 1.5 * duration_days) & (np.abs(phase) <= 4.0 * duration_days)
    if inside.sum() < 3 or nearby.sum() < 10:
        return {
            "depth_ppm": None,
            "snr": None,
            "events": 0,
            "transit_points": int(inside.sum()),
            "baseline_points": int(nearby.sum()),
            "status": "insufficient_coverage",
        }
    baseline = float(np.nanmedian(flux[nearby]))
    depth = baseline - float(np.nanmedian(flux[inside]))
    scatter = _mad_sigma(flux[nearby])
    snr = depth / (scatter / np.sqrt(int(inside.sum()))) if scatter > 0 else 0.0
    event_ids = np.rint((time[inside] - float(epoch_btjd)) / period_days).astype(int)
    events = int(np.unique(event_ids).size)
    if depth <= 0 or snr < 3.0:
        status = "not_supported"
    elif snr < 5.0:
        status = "weak_support"
    else:
        status = "supported"
    return {
        "depth_ppm": float(depth * 1e6),
        "snr": float(snr),
        "events": events,
        "transit_points": int(inside.sum()),
        "baseline_points": int(nearby.sum()),
        "local_scatter_ppm": float(scatter * 1e6),
        "status": status,
    }


def validate_candidates_across_branches(
    branches: Mapping[str, FluxBranch],
    candidates: Iterable[Mapping],
    *,
    holdout_sector: int,
) -> list[dict]:
    """Attach soft PDCSAP/SAP consistency evidence to locked candidates."""
    results: list[dict] = []
    for candidate in candidates:
        detection_sources = list(candidate.get("detection_sources", ["bls"]))
        holdout_is_sealed = bool(candidate.get(
            "holdout_is_sealed",
            "bls" in detection_sources,
        ))
        branch_results: dict[str, dict] = {}
        for key, branch in branches.items():
            search = branch.sector_id != int(holdout_sector)
            holdout = ~search
            arguments = {
                "period_days": float(candidate["period_days"]),
                "epoch_btjd": float(candidate["epoch_btjd"]),
                "duration_hours": float(candidate["duration_hours"]),
            }
            search_metrics = measure_fixed_ephemeris(
                branch.time[search], branch.flux[search], **arguments
            )
            holdout_metrics = measure_fixed_ephemeris(
                branch.time[holdout], branch.flux[holdout], **arguments
            )
            branch_results[key] = {
                "label": branch.label,
                "source_column": branch.source_column.upper(),
                "method": branch.method,
                "search_sectors": search_metrics,
                "sealed_holdout": holdout_metrics,
            }

        visible_support = [
            key
            for key, evidence in branch_results.items()
            if evidence["search_sectors"]["status"] in {"supported", "weak_support"}
        ]
        holdout_support = [
            key
            for key, evidence in branch_results.items()
            if evidence["sealed_holdout"]["status"] in {"supported", "weak_support"}
        ]
        if len(visible_support) == len(branch_results):
            status = "cross_pipeline_supported"
        elif visible_support:
            status = "partial_pipeline_support"
        else:
            status = "pipeline_support_unresolved"

        pdcsap_depth = branch_results["pdcsap_lightkurve"]["search_sectors"]["depth_ppm"]
        sap_depth = branch_results["sap_lightkurve"]["search_sectors"]["depth_ppm"]
        depth_ratio = None
        if pdcsap_depth is not None and sap_depth is not None and pdcsap_depth > 0:
            depth_ratio = float(sap_depth / pdcsap_depth)
        results.append(
            {
                "rank": int(candidate.get("rank", len(results) + 1)),
                "period_days": float(candidate["period_days"]),
                "detection_sources": detection_sources,
                "status": status,
                "supporting_search_branches": visible_support,
                "supporting_holdout_branches": holdout_support,
                "sap_to_pdcsap_depth_ratio": depth_ratio,
                "holdout_is_sealed": holdout_is_sealed,
                "holdout_interpretation": (
                    "independent future-sector check"
                    if holdout_is_sealed
                    else "consistency check only; the saved event-first search used this sector"
                ),
                "hard_rejected": False,
                "branches": branch_results,
            }
        )
    return results


def _weighted_centroid(weights: np.ndarray) -> tuple[float, float] | None:
    weights = np.asarray(weights, dtype=float)
    positive = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    total = float(positive.sum())
    if total <= 0:
        return None
    rows, columns = np.indices(positive.shape)
    return (
        float((columns * positive).sum() / total),
        float((rows * positive).sum() / total),
    )


def _json_image(values: np.ndarray) -> list[list[float | None]]:
    array = np.asarray(values, dtype=float)
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in array
    ]


def localize_difference_cube(
    time: np.ndarray,
    flux_cube: np.ndarray,
    pipeline_mask: np.ndarray,
    *,
    period_days: float,
    epoch_btjd: float,
    duration_hours: float,
) -> dict:
    """Create event-local in/out difference evidence from a calibrated pixel cube."""
    time = np.asarray(time, dtype=float)
    cube = np.asarray(flux_cube, dtype=float)
    aperture = np.asarray(pipeline_mask, dtype=bool)
    if cube.ndim != 3 or cube.shape[0] != time.size or cube.shape[1:] != aperture.shape:
        raise ValueError("Pixel cube, time, and aperture-mask shapes are inconsistent")

    duration_days = float(duration_hours) / 24.0
    period_days = float(period_days)
    phase = (time - float(epoch_btjd) + 0.5 * period_days) % period_days - 0.5 * period_days
    inside = np.abs(phase) <= 0.5 * duration_days
    nearby = (np.abs(phase) >= 1.5 * duration_days) & (np.abs(phase) <= 4.0 * duration_days)
    event_ids = np.rint((time[inside] - float(epoch_btjd)) / period_days).astype(int)
    events = int(np.unique(event_ids).size) if event_ids.size else 0
    if inside.sum() < 3 or nearby.sum() < 10:
        return {
            "status": "insufficient_pixel_coverage",
            "hard_rejected": False,
            "events": events,
            "in_transit_cadences": int(inside.sum()),
            "nearby_out_cadences": int(nearby.sum()),
        }

    median_image = np.nanmedian(cube, axis=0)
    in_image = np.nanmedian(cube[inside], axis=0)
    out_image = np.nanmedian(cube[nearby], axis=0)
    difference = out_image - in_image

    if aperture.any():
        background_values = median_image[~aperture]
        background = float(np.nanmedian(background_values)) if background_values.size else 0.0
        source_weights = np.clip(median_image - background, 0, None) * aperture
    else:
        source_weights = np.clip(median_image - np.nanmedian(median_image), 0, None)
    source_centroid = _weighted_centroid(source_weights)

    difference_background = float(np.nanmedian(difference))
    dimming_weights = np.clip(difference - difference_background, 0, None)
    dimming_centroid = _weighted_centroid(dimming_weights)
    if source_centroid is None or dimming_centroid is None:
        status = "pixel_localization_unresolved"
        separation = None
    else:
        separation = float(np.hypot(
            source_centroid[0] - dimming_centroid[0],
            source_centroid[1] - dimming_centroid[1],
        ))
        total_dimming = float(dimming_weights.sum())
        aperture_fraction = (
            float(dimming_weights[aperture].sum() / total_dimming)
            if total_dimming > 0 and aperture.any()
            else None
        )
        if separation <= 1.5 and (aperture_fraction is None or aperture_fraction >= 0.5):
            status = "on_target_consistent"
        elif separation > 2.5 or (aperture_fraction is not None and aperture_fraction < 0.25):
            status = "possible_off_target"
        else:
            status = "pixel_localization_unresolved"

    total_dimming = float(dimming_weights.sum())
    aperture_fraction = (
        float(dimming_weights[aperture].sum() / total_dimming)
        if total_dimming > 0 and aperture.any()
        else None
    )
    brightest_pixel = np.unravel_index(
        np.nanargmax(np.where(aperture, median_image, np.nan))
        if aperture.any()
        else np.nanargmax(median_image),
        median_image.shape,
    )
    dimmest_source_pixel = np.unravel_index(np.nanargmax(difference), difference.shape)
    return {
        "status": status,
        "hard_rejected": False,
        "events": events,
        "in_transit_cadences": int(inside.sum()),
        "nearby_out_cadences": int(nearby.sum()),
        "source_centroid_col_row": list(source_centroid) if source_centroid else None,
        "dimming_centroid_col_row": list(dimming_centroid) if dimming_centroid else None,
        "centroid_separation_pixels": separation,
        "dimming_fraction_inside_pipeline_aperture": aperture_fraction,
        "brightest_source_pixel_row_col": [int(x) for x in brightest_pixel],
        "strongest_dimming_pixel_row_col": [int(x) for x in dimmest_source_pixel],
        "median_image": _json_image(median_image),
        "difference_image": _json_image(difference),
        "pipeline_mask": np.asarray(aperture, dtype=bool).tolist(),
        "interpretation": (
            "soft source-location evidence only; not a planet confirmation and not yet Gaia/TIC neighbour association"
        ),
    }


def localize_candidate_tpf(path: str | Path, candidate: Mapping) -> dict:
    """Read a real TESS target-pixel file through Lightkurve and localize one candidate."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tpf = lk.read(Path(path))
    if tpf is None or not hasattr(tpf, "pipeline_mask"):
        raise ValueError(f"Lightkurve could not read a target-pixel file from {path}")
    evidence = localize_difference_cube(
        np.asarray(tpf.time.value, dtype=float),
        np.asarray(tpf.flux.value, dtype=float),
        np.asarray(tpf.pipeline_mask, dtype=bool),
        period_days=float(candidate["period_days"]),
        epoch_btjd=float(candidate["epoch_btjd"]),
        duration_hours=float(candidate["duration_hours"]),
    )
    evidence.update(
        {
            "tpf_file": Path(path).name,
            "sector": int(tpf.meta.get("SECTOR", -1)),
            "period_days": float(candidate["period_days"]),
        }
    )
    return evidence


def localize_candidates_tpf(
    path: str | Path,
    candidates: Iterable[Mapping],
) -> list[dict]:
    """Read one TPF once and localize a sequence of periodic candidates."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tpf = lk.read(Path(path))
    if tpf is None or not hasattr(tpf, "pipeline_mask"):
        raise ValueError(f"Lightkurve could not read a target-pixel file from {path}")
    time = np.asarray(tpf.time.value, dtype=float)
    cube = np.asarray(tpf.flux.value, dtype=float)
    mask = np.asarray(tpf.pipeline_mask, dtype=bool)
    output: list[dict] = []
    for candidate in candidates:
        evidence = localize_difference_cube(
            time,
            cube,
            mask,
            period_days=float(candidate["period_days"]),
            epoch_btjd=float(candidate["epoch_btjd"]),
            duration_hours=float(candidate["duration_hours"]),
        )
        evidence.update(
            {
                "tpf_file": Path(path).name,
                "sector": int(tpf.meta.get("SECTOR", -1)),
                "period_days": float(candidate["period_days"]),
            }
        )
        output.append(evidence)
    return output
