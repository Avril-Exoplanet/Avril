"""Detailed TPF localization plus TIC/Gaia neighbour association."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path

import astropy.units as u
import lightkurve as lk
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs

from .lightkurve_evidence import localize_difference_cube, measure_fixed_ephemeris


def _plain(value):
    if np.ma.is_masked(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (AttributeError, ValueError):
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _localization_aperture(tpf) -> tuple[np.ndarray, str]:
    """Return a usable aperture for official TPFs or raw TESSCut stamps."""

    aperture = np.asarray(tpf.pipeline_mask, dtype=bool)
    if aperture.ndim == 2 and aperture.any():
        return aperture, "pipeline_aperture"
    shape = tuple(int(value) for value in np.asarray(tpf.flux.value).shape[-2:])
    rows, columns = np.indices(shape)
    try:
        center = SkyCoord(float(tpf.ra), float(tpf.dec), unit="deg", frame="icrs")
        center_col, center_row = tpf.wcs.world_to_pixel(center)
        center_col = float(center_col)
        center_row = float(center_row)
    except Exception:
        center_row = 0.5 * (shape[0] - 1)
        center_col = 0.5 * (shape[1] - 1)
    aperture = np.hypot(columns - center_col, rows - center_row) <= 2.0
    if not aperture.any():
        aperture[
            int(np.clip(round(center_row), 0, shape[0] - 1)),
            int(np.clip(round(center_col), 0, shape[1] - 1)),
        ] = True
    return aperture, "tesscut_circular_2p0_pixel_fallback"


def query_tic_gaia_neighbors(
    tpf_path: str | Path,
    *,
    radius_arcsec: float = 180.0,
) -> dict:
    """Query TIC around a TPF and project Gaia-linked sources onto its pixels."""
    path = Path(tpf_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tpf = lk.read(path)
    center = SkyCoord(float(tpf.ra), float(tpf.dec), unit="deg", frame="icrs")
    table = Catalogs.query_region(
        center,
        radius=float(radius_arcsec) * u.arcsec,
        catalog="TIC",
    )
    coordinates = SkyCoord(
        np.asarray(table["ra"], dtype=float),
        np.asarray(table["dec"], dtype=float),
        unit="deg",
        frame="icrs",
    )
    columns, rows = tpf.wcs.world_to_pixel(coordinates)
    aperture, aperture_source = _localization_aperture(tpf)
    raw_target_tic = tpf.meta.get("TICID")
    try:
        target_tic = int(raw_target_tic)
    except (TypeError, ValueError):
        target_tic = -1
    target_rows = [index for index, row in enumerate(table) if int(row["ID"]) == target_tic]
    target_identity_source = "tpf_ticid"
    if not target_rows and len(table):
        target_rows = [int(np.nanargmin(np.asarray(table["dstArcSec"], dtype=float)))]
        target_tic = int(table[target_rows[0]]["ID"])
        target_identity_source = "nearest_tic_to_tesscut_center"
    target_tmag = (
        float(table[target_rows[0]]["Tmag"])
        if target_rows and np.isfinite(float(table[target_rows[0]]["Tmag"]))
        else None
    )
    neighbors = []
    for index, catalog_row in enumerate(table):
        col = float(columns[index])
        row = float(rows[index])
        inside = 0 <= col < aperture.shape[1] and 0 <= row < aperture.shape[0]
        nearest_col = int(np.clip(round(col), 0, aperture.shape[1] - 1))
        nearest_row = int(np.clip(round(row), 0, aperture.shape[0] - 1))
        tmag = _plain(catalog_row["Tmag"])
        tmag = float(tmag) if tmag is not None and np.isfinite(float(tmag)) else None
        delta_tmag = (
            float(tmag - target_tmag)
            if tmag is not None and target_tmag is not None
            else None
        )
        relative_flux = (
            float(10 ** (-0.4 * delta_tmag)) if delta_tmag is not None else None
        )
        maximum_blended_depth = (
            float(relative_flux / (1.0 + relative_flux) * 1e6)
            if relative_flux is not None
            else None
        )
        neighbors.append(
            {
                "tic_id": int(catalog_row["ID"]),
                "gaia_dr2_source_id": _plain(catalog_row["GAIA"]),
                "ra_deg": float(catalog_row["ra"]),
                "dec_deg": float(catalog_row["dec"]),
                "separation_arcsec": float(catalog_row["dstArcSec"]),
                "tmag": tmag,
                "delta_tmag_from_target": delta_tmag,
                "estimated_flux_ratio_to_target": relative_flux,
                "maximum_depth_if_fully_eclipsed_ppm": maximum_blended_depth,
                "pixel_col_row": [col, row],
                "inside_tpf_stamp": bool(inside),
                "inside_pipeline_aperture_nearest_pixel": bool(
                    inside and aperture[nearest_row, nearest_col]
                ),
                "is_target_tic": int(catalog_row["ID"]) == target_tic,
            }
        )
    neighbors.sort(key=lambda item: item["separation_arcsec"])
    return {
        "catalog": "TIC (including Gaia DR2 cross-identifiers where available)",
        "query_radius_arcsec": float(radius_arcsec),
        "target_tic_id": target_tic,
        "target_identity_source": target_identity_source,
        "target_tmag": target_tmag,
        "aperture_source": aperture_source,
        "tpf_file": path.name,
        "sector": int(tpf.meta.get("SECTOR", -1)),
        "neighbors": neighbors,
    }


def associate_centroid_with_neighbors(
    centroid_col_row: list[float] | tuple[float, float] | None,
    neighbor_catalog: Mapping,
) -> dict | None:
    if centroid_col_row is None:
        return None
    col, row = map(float, centroid_col_row)
    usable = [
        item
        for item in neighbor_catalog.get("neighbors", [])
        if item.get("inside_tpf_stamp")
    ]
    if not usable:
        return None
    nearest = min(
        usable,
        key=lambda item: np.hypot(
            col - float(item["pixel_col_row"][0]),
            row - float(item["pixel_col_row"][1]),
        ),
    )
    distance = float(
        np.hypot(
            col - float(nearest["pixel_col_row"][0]),
            row - float(nearest["pixel_col_row"][1]),
        )
    )
    return {
        "nearest_tic_id": nearest["tic_id"],
        "nearest_gaia_dr2_source_id": nearest["gaia_dr2_source_id"],
        "nearest_is_target_tic": nearest["is_target_tic"],
        "distance_pixels": distance,
        "nearest_tmag": nearest["tmag"],
        "nearest_delta_tmag_from_target": nearest["delta_tmag_from_target"],
        "nearest_estimated_flux_ratio_to_target": nearest[
            "estimated_flux_ratio_to_target"
        ],
        "nearest_maximum_depth_if_fully_eclipsed_ppm": nearest[
            "maximum_depth_if_fully_eclipsed_ppm"
        ],
    }


def detailed_tpf_localization(
    tpf_path: str | Path,
    candidate: Mapping,
    *,
    neighbor_catalog: Mapping | None = None,
) -> dict:
    """Localize the combined signal and each observed event separately."""
    path = Path(tpf_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tpf = lk.read(path)
    time = np.asarray(tpf.time.value, dtype=float)
    cube = np.asarray(tpf.flux.value, dtype=float)
    aperture, aperture_source = _localization_aperture(tpf)
    period = float(candidate["period_days"])
    epoch = float(candidate["epoch_btjd"])
    duration_hours = float(candidate["duration_hours"])
    combined = localize_difference_cube(
        time,
        cube,
        aperture,
        period_days=period,
        epoch_btjd=epoch,
        duration_hours=duration_hours,
    )
    combined["aperture_source"] = aperture_source
    aperture_lc = tpf.to_lightcurve(aperture_mask=aperture).remove_nans().normalize()
    aperture_measurement = measure_fixed_ephemeris(
        np.asarray(aperture_lc.time.value, dtype=float),
        np.asarray(aperture_lc.flux.value, dtype=float),
        period_days=period,
        epoch_btjd=epoch,
        duration_hours=duration_hours,
    )
    phase = (time - epoch + 0.5 * period) % period - 0.5 * period
    inside = np.abs(phase) <= duration_hours / 48.0
    cycle_ids = np.unique(np.rint((time[inside] - epoch) / period).astype(int))
    per_event = []
    duration_days = duration_hours / 24.0
    for cycle in cycle_ids:
        center = epoch + int(cycle) * period
        local = np.abs(time - center) <= 4.5 * duration_days
        if local.sum() < 13:
            continue
        result = localize_difference_cube(
            time[local],
            cube[local],
            aperture,
            period_days=period,
            epoch_btjd=epoch,
            duration_hours=duration_hours,
        )
        compact = {
            key: value
            for key, value in result.items()
            if key not in {"median_image", "difference_image", "pipeline_mask"}
        }
        compact.update({"cycle": int(cycle), "predicted_center_btjd": float(center)})
        if neighbor_catalog is not None:
            compact["nearest_catalog_source"] = associate_centroid_with_neighbors(
                compact.get("dimming_centroid_col_row"),
                neighbor_catalog,
            )
        per_event.append(compact)
    centroids = np.asarray(
        [
            item["dimming_centroid_col_row"]
            for item in per_event
            if item.get("dimming_centroid_col_row") is not None
        ],
        dtype=float,
    )
    if centroids.ndim == 2 and len(centroids) >= 2:
        col_scatter = _robust_scatter(centroids[:, 0])
        row_scatter = _robust_scatter(centroids[:, 1])
        radial_scatter = float(np.hypot(col_scatter, row_scatter))
        empirical_error = float(radial_scatter / np.sqrt(len(centroids)))
    else:
        col_scatter = row_scatter = radial_scatter = empirical_error = None
    if neighbor_catalog is not None:
        combined["nearest_catalog_source"] = associate_centroid_with_neighbors(
            combined.get("dimming_centroid_col_row"),
            neighbor_catalog,
        )
    plausible_contaminants = []
    dimming_centroid = combined.get("dimming_centroid_col_row")
    observed_depth = aperture_measurement.get("depth_ppm")
    localization_radius = max(
        1.5,
        3.0 * empirical_error if empirical_error is not None else 1.5,
    )
    if neighbor_catalog is not None and dimming_centroid is not None:
        dim_col, dim_row = map(float, dimming_centroid)
        for source in neighbor_catalog.get("neighbors", []):
            if source.get("is_target_tic") or not source.get("inside_tpf_stamp"):
                continue
            source_col, source_row = map(float, source["pixel_col_row"])
            distance = float(np.hypot(dim_col - source_col, dim_row - source_row))
            maximum_depth = source.get("maximum_depth_if_fully_eclipsed_ppm")
            bright_enough = (
                observed_depth is None
                or observed_depth <= 0
                or maximum_depth is None
                or maximum_depth >= 0.8 * observed_depth
            )
            if distance <= localization_radius and bright_enough:
                plausible_contaminants.append(
                    {
                        "tic_id": source["tic_id"],
                        "gaia_dr2_source_id": source["gaia_dr2_source_id"],
                        "distance_from_dimming_centroid_pixels": distance,
                        "tmag": source["tmag"],
                        "delta_tmag_from_target": source["delta_tmag_from_target"],
                        "maximum_depth_if_fully_eclipsed_ppm": maximum_depth,
                    }
                )
    plausible_contaminants.sort(
        key=lambda item: item["distance_from_dimming_centroid_pixels"]
    )
    combined.update(
        {
            "tpf_file": path.name,
            "sector": int(tpf.meta.get("SECTOR", -1)),
            "period_days": period,
            "per_event": per_event,
            "events_with_centroids": int(len(centroids)) if centroids.ndim == 2 else 0,
            "event_centroid_col_scatter_pixels": col_scatter,
            "event_centroid_row_scatter_pixels": row_scatter,
            "event_centroid_radial_scatter_pixels": radial_scatter,
            "empirical_mean_centroid_error_pixels": empirical_error,
            "pipeline_aperture_fixed_ephemeris_measurement": aperture_measurement,
            "catalog_contaminant_search_radius_pixels": localization_radius,
            "plausible_catalog_contaminants": plausible_contaminants,
            "centroid_error_note": (
                "robust event-to-event scatter / sqrt(N); an empirical repeatability metric, "
                "not a formal PRF-fit astrometric uncertainty"
            ),
        }
    )
    return combined


def _robust_scatter(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    scatter = 1.4826 * np.nanmedian(np.abs(values - median))
    if not np.isfinite(scatter) or scatter <= 0:
        scatter = np.nanstd(values)
    return float(scatter) if np.isfinite(scatter) else 0.0
