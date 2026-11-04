"""Create transit-preserving, SPOC-shaped light curves from TESSCut FFIs.

This is deliberately a label-free photometry stage.  It never accepts a
period, epoch, duration, or transit mask.  A single fixed circular aperture is
selected for all sectors using point-to-point robust scatter, after subtracting
the per-cadence median of distant cutout pixels as the local background.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import lightkurve as lk
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits


DEFAULT_RADII = (1.5, 2.0, 2.5, 3.0)


@dataclass
class ExtractedSector:
    path: Path
    sector: int
    camera: int
    ccd: int
    target_x: float
    target_y: float
    time: np.ndarray
    timecorr: np.ndarray
    cadenceno: np.ndarray
    quality: np.ndarray
    pos_corr1: np.ndarray
    pos_corr2: np.ndarray
    aperture_flux: dict[float, np.ndarray]
    aperture_error: dict[float, np.ndarray]
    centroid_x: np.ndarray
    centroid_y: np.ndarray
    primary_header: fits.Header
    table_header: fits.Header


def _robust_point_scatter(time: np.ndarray, flux: np.ndarray, quality: np.ndarray) -> float:
    good = np.isfinite(time) & np.isfinite(flux) & (flux > 0) & (quality == 0)
    if int(good.sum()) < 20:
        return math.inf
    clean_time = time[good]
    clean_flux = flux[good]
    order = np.argsort(clean_time)
    clean_time = clean_time[order]
    clean_flux = clean_flux[order]
    median_flux = float(np.nanmedian(clean_flux))
    if not math.isfinite(median_flux) or median_flux <= 0:
        return math.inf
    cadence = float(np.nanmedian(np.diff(clean_time)))
    adjacent = np.diff(clean_time) <= 3.0 * cadence
    differences = np.diff(clean_flux / median_flux)[adjacent]
    if differences.size < 10:
        return math.inf
    center = float(np.nanmedian(differences))
    mad = float(np.nanmedian(np.abs(differences - center)))
    return 1.4826 * mad / math.sqrt(2.0)


def _safe_vector(table, name: str, length: int, dtype=float) -> np.ndarray:
    if name in table.names:
        return np.asarray(table[name], dtype=dtype)
    return np.full(length, np.nan, dtype=dtype)


def extract_sector(
    path: Path,
    *,
    coordinate: SkyCoord,
    radii: tuple[float, ...] = DEFAULT_RADII,
) -> ExtractedSector:
    target_pixel_file = lk.read(path)
    target_x, target_y = target_pixel_file.wcs.world_to_pixel(coordinate)
    with fits.open(path, memmap=True) as hdul:
        primary_header = hdul[0].header.copy()
        table_header = hdul[1].header.copy()
        table = hdul[1].data
        time = np.asarray(table["TIME"], dtype=float)
        flux_cube = np.asarray(table["FLUX"], dtype=float)
        error_cube = np.asarray(table["FLUX_ERR"], dtype=float)
        length, rows, columns = flux_cube.shape
        quality = _safe_vector(table, "QUALITY", length, np.int64)
        timecorr = _safe_vector(table, "TIMECORR", length, float)
        cadenceno = _safe_vector(table, "CADENCENO", length, np.int64)
        pos_corr1 = _safe_vector(table, "POS_CORR1", length, float)
        pos_corr2 = _safe_vector(table, "POS_CORR2", length, float)

    yy, xx = np.mgrid[:rows, :columns]
    distance = np.hypot(xx - float(target_x), yy - float(target_y))
    background_mask = distance >= 4.5
    if int(background_mask.sum()) < 30:
        raise ValueError(f"Insufficient background pixels in {path.name}")
    background_pixels = flux_cube[:, background_mask]
    background = np.nanmedian(background_pixels, axis=1)
    background_mad = 1.4826 * np.nanmedian(
        np.abs(background_pixels - background[:, None]), axis=1
    )
    corrected = flux_cube - background[:, None, None]

    aperture_flux: dict[float, np.ndarray] = {}
    aperture_error: dict[float, np.ndarray] = {}
    for radius in radii:
        mask = distance <= radius
        count = int(mask.sum())
        aperture_flux[radius] = np.nansum(corrected[:, mask], axis=1)
        aperture_error[radius] = np.sqrt(
            np.nansum(np.square(error_cube[:, mask]), axis=1)
            + np.square(background_mad * count / math.sqrt(background_mask.sum()))
        )

    centroid_mask = distance <= max(radii)
    centroid_weights = np.clip(corrected[:, centroid_mask], 0.0, None)
    centroid_total = np.nansum(centroid_weights, axis=1)
    centroid_x = np.divide(
        np.nansum(centroid_weights * xx[centroid_mask], axis=1),
        centroid_total,
        out=np.full(length, np.nan),
        where=centroid_total > 0,
    )
    centroid_y = np.divide(
        np.nansum(centroid_weights * yy[centroid_mask], axis=1),
        centroid_total,
        out=np.full(length, np.nan),
        where=centroid_total > 0,
    )

    return ExtractedSector(
        path=path,
        sector=int(primary_header.get("SECTOR", -1)),
        camera=int(primary_header.get("CAMERA", -1)),
        ccd=int(primary_header.get("CCD", -1)),
        target_x=float(target_x),
        target_y=float(target_y),
        time=time,
        timecorr=timecorr,
        cadenceno=cadenceno,
        quality=quality,
        pos_corr1=pos_corr1,
        pos_corr2=pos_corr2,
        aperture_flux=aperture_flux,
        aperture_error=aperture_error,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        primary_header=primary_header,
        table_header=table_header,
    )


def select_global_radius(
    sectors: list[ExtractedSector], radii: tuple[float, ...]
) -> tuple[float, dict[float, dict]]:
    diagnostics: dict[float, dict] = {}
    for radius in radii:
        per_sector = {
            sector.sector: _robust_point_scatter(
                sector.time, sector.aperture_flux[radius], sector.quality
            )
            for sector in sectors
        }
        finite = [value for value in per_sector.values() if math.isfinite(value)]
        diagnostics[radius] = {
            "median_point_to_point_scatter": (
                float(np.nanmedian(finite)) if finite else None
            ),
            "per_sector": {
                str(key): (float(value) if math.isfinite(value) else None)
                for key, value in per_sector.items()
            },
        }
    available = [
        radius
        for radius in radii
        if diagnostics[radius]["median_point_to_point_scatter"] is not None
    ]
    if not available:
        raise RuntimeError("No aperture radius produced usable photometry")
    selected = min(
        available,
        key=lambda radius: diagnostics[radius]["median_point_to_point_scatter"],
    )
    return selected, diagnostics


def write_spoc_shaped_lightcurve(
    sector: ExtractedSector,
    *,
    output_dir: Path,
    target: str,
    tic_id: int,
    ra: float,
    dec: float,
    selected_radius: float,
    radii: tuple[float, ...],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"selfmade-ffi-tic{tic_id}-s{sector.sector:04d}-lc.fits"
    chosen_flux = np.asarray(sector.aperture_flux[selected_radius], dtype=np.float32)
    chosen_error = np.asarray(sector.aperture_error[selected_radius], dtype=np.float32)
    columns = [
        fits.Column(name="TIME", format="D", unit="BJD - 2457000, days", array=sector.time),
        fits.Column(name="TIMECORR", format="E", unit="d", array=sector.timecorr.astype(np.float32)),
        fits.Column(name="CADENCENO", format="J", array=sector.cadenceno.astype(np.int32)),
        fits.Column(name="SAP_FLUX", format="E", unit="e-/s", array=chosen_flux),
        fits.Column(name="SAP_FLUX_ERR", format="E", unit="e-/s", array=chosen_error),
        fits.Column(name="PDCSAP_FLUX", format="E", unit="e-/s", array=chosen_flux),
        fits.Column(name="PDCSAP_FLUX_ERR", format="E", unit="e-/s", array=chosen_error),
        fits.Column(name="QUALITY", format="J", array=sector.quality.astype(np.int32)),
        fits.Column(name="MOM_CENTR1", format="E", unit="pixel", array=sector.centroid_x.astype(np.float32)),
        fits.Column(name="MOM_CENTR2", format="E", unit="pixel", array=sector.centroid_y.astype(np.float32)),
        fits.Column(name="POS_CORR1", format="E", unit="pixel", array=sector.pos_corr1.astype(np.float32)),
        fits.Column(name="POS_CORR2", format="E", unit="pixel", array=sector.pos_corr2.astype(np.float32)),
    ]
    for radius in radii:
        label = str(radius).replace(".", "P")
        columns.append(
            fits.Column(
                name=f"APER_{label}",
                format="E",
                unit="e-/s",
                array=np.asarray(sector.aperture_flux[radius], dtype=np.float32),
            )
        )
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "TESS"
    primary.header["INSTRUME"] = "TESS Photometer"
    primary.header["MISSION"] = "TESS"
    primary.header["OBJECT"] = target
    primary.header["TICID"] = int(tic_id)
    primary.header["SECTOR"] = int(sector.sector)
    primary.header["CAMERA"] = int(sector.camera)
    primary.header["CCD"] = int(sector.ccd)
    primary.header["RA_OBJ"] = float(ra)
    primary.header["DEC_OBJ"] = float(dec)
    # Lightkurve identifies official-shaped TESS light-curve files from the
    # mission plus a CREATOR containing ``LightCurveExporterPipelineModule``.
    # SELFMADE and ORIGIN below keep the provenance explicit.
    primary.header["CREATOR"] = "SelfmadeLightCurveExporterPipelineModule"
    primary.header["ORIGIN"] = "MAST TESSCut calibrated FFI"
    primary.header["SELFMADE"] = True
    primary.header["APER_RAD"] = float(selected_radius)
    primary.header["PHOTMETH"] = "fixed circular aperture + local median background"

    table_hdu = fits.BinTableHDU.from_columns(columns, name="LIGHTCURVE")
    table_hdu.header["INHERIT"] = True
    table_hdu.header["TELESCOP"] = "TESS"
    table_hdu.header["MISSION"] = "TESS"
    table_hdu.header["OBJECT"] = target
    table_hdu.header["TICID"] = int(tic_id)
    table_hdu.header["SECTOR"] = int(sector.sector)
    table_hdu.header["BJDREFI"] = 2457000
    table_hdu.header["BJDREFF"] = 0.0
    table_hdu.header["TIMESYS"] = "TDB"
    table_hdu.header["TIMEUNIT"] = "d"
    table_hdu.header["APER_RAD"] = float(selected_radius)
    fits.HDUList([primary, table_hdu]).writeto(output, overwrite=True, checksum=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--tic-id", type=int, required=True)
    parser.add_argument("--ra", type=float, required=True)
    parser.add_argument("--dec", type=float, required=True)
    args = parser.parse_args()

    inputs = sorted(
        path
        for path in args.input_dir.glob("*.fits")
        if not path.name.startswith("._")
    )
    if not inputs:
        raise SystemExit("No TESSCut FITS files found")
    coordinate = SkyCoord(args.ra * u.deg, args.dec * u.deg)
    sectors = [extract_sector(path, coordinate=coordinate) for path in inputs]
    sectors.sort(key=lambda row: row.sector)
    selected_radius, aperture_diagnostics = select_global_radius(
        sectors, DEFAULT_RADII
    )
    outputs = [
        write_spoc_shaped_lightcurve(
            sector,
            output_dir=args.output_dir,
            target=args.target,
            tic_id=args.tic_id,
            ra=args.ra,
            dec=args.dec,
            selected_radius=selected_radius,
            radii=DEFAULT_RADII,
        )
        for sector in sectors
    ]
    manifest = {
        "schema_version": "selfmade-ffi-spoc-shaped-v1",
        "target": args.target,
        "tic_id": args.tic_id,
        "coordinate_deg": {"ra": args.ra, "dec": args.dec},
        "truth_free_photometry": True,
        "period_epoch_duration_used": False,
        "source": "MAST TESSCut calibrated FFI pixels",
        "background": "per-cadence median of pixels at least 4.5 pixels from target",
        "aperture_selection": "one global radius minimizing median per-sector robust point-to-point scatter",
        "selected_radius_pixels": selected_radius,
        "aperture_diagnostics": {
            str(radius): values for radius, values in aperture_diagnostics.items()
        },
        "sectors": [sector.sector for sector in sectors],
        "inputs": [str(path.resolve()) for path in inputs],
        "outputs": [str(path.resolve()) for path in outputs],
    }
    manifest_path = args.output_dir / "photometry_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
