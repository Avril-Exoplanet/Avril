"""Local desktop adapter; the Avril science pipeline remains unchanged."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import signal
import threading
from collections import Counter
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from astropy.io import fits

from vlora.web_server import JobStore, VLoraRequestHandler

MAX_FILE = 128 * 1024 * 1024
TARGET_SELECTION_ROOT = Path(__file__).with_name("target_selection")
TARGET_QUERY_LIMIT = 200
MIN_PIXEL_COVERAGE_RATIO = 0.90
FOLD_BIN_COUNT = 120
FOLD_HALF_WINDOW_DURATIONS = 5.0

_TARGET_STRING_FIELDS = frozenset({
    "disposition",
    "model_applicability",
    "gap_interpretation",
    "count_distribution_note",
    "detectability_mode",
    "discovery_lane",
    "search_recipe",
})
_TARGET_BOOLEAN_FIELDS = frozenset({
    "gap_distribution_truncated",
    "multi_epoch_coverage",
    "primary_to_recent_revisit",
    "strict_model_eligible",
})
_TARGET_INTEGER_FIELDS = frozenset({
    "ID",
    "known_toi_planetlike_count",
    "known_confirmed_planet_count",
    "known_recorded_planet_count",
    "feature_missing_count",
    "feature_outside_training_range_count",
    "known_catalog_count",
    "sectors",
    "sector_count_recomputed",
    "primary_sector_count",
    "first_extended_sector_count",
    "recent_sector_count",
    "observation_era_count",
    "sector_span",
    "queue_rank",
})
_TARGET_SORT_FIELDS = {
    "rank": "queue_rank",
    "queue_rank": "queue_rank",
    "tic": "ID",
    "score": "lane_score",
    "lane_score": "lane_score",
    "tmag": "Tmag",
    "gap": "gap_preselection_score",
    "gap_preselection_score": "gap_preselection_score",
    "expected_additional": "frozen_catalog_expected_additional_lower_bound",
    "more_than_known": "frozen_catalog_p_more_than_known",
    "detectability": "tess_detectable_percentile",
    "sectors": "sector_count_recomputed",
    "baseline": "sector_span",
    "known": "known_recorded_planet_count",
}
_TARGET_QUERY_KEYS = frozenset({"q", "lane", "sort", "order", "limit", "offset"})
_JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
_CANDIDATE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _target_value(field, value):
    value = value.strip()
    if not value:
        return None
    if field == "sector_list":
        return [int(part) for part in value.split("|") if part]
    if field in _TARGET_STRING_FIELDS:
        return value
    if field in _TARGET_BOOLEAN_FIELDS:
        normalized = value.casefold()
        if normalized not in {"true", "false"}:
            raise ValueError(f"invalid boolean in target catalog: {field}")
        return normalized == "true"
    if field in _TARGET_INTEGER_FIELDS:
        return int(value)
    number = float(value)
    return number if math.isfinite(number) else None


def _single_query_value(parameters, key, default):
    values = parameters.get(key)
    if values is None:
        return default
    if len(values) != 1:
        raise ValueError(f"query parameter must appear once: {key}")
    return values[0].strip()


def _bounded_query_integer(parameters, key, default, minimum, maximum):
    text = _single_query_value(parameters, key, str(default))
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"{key} must be an integer")
    value = int(text)
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def parse_target_query(raw_query, lanes):
    try:
        parameters = parse_qs(
            raw_query,
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=len(_TARGET_QUERY_KEYS),
        )
    except ValueError as exc:
        raise ValueError("invalid query string") from exc
    unknown = set(parameters).difference(_TARGET_QUERY_KEYS)
    if unknown:
        raise ValueError(f"unsupported query parameter: {sorted(unknown)[0]}")

    q = _single_query_value(parameters, "q", "")
    if len(q) > 32:
        raise ValueError("q is too long")
    if q:
        match = re.fullmatch(r"(?i:tic)\s*([0-9]+)|([0-9]+)", q)
        if not match:
            raise ValueError("q must be a TIC number, optionally prefixed with TIC")
        q = match.group(1) or match.group(2)

    lane = _single_query_value(parameters, "lane", "all")
    if lane != "all" and lane not in lanes:
        raise ValueError("unknown discovery lane")
    sort = _single_query_value(parameters, "sort", "rank")
    if sort not in _TARGET_SORT_FIELDS:
        raise ValueError("unsupported target sort")
    order = _single_query_value(parameters, "order", "asc").casefold()
    if order not in {"asc", "desc"}:
        raise ValueError("order must be asc or desc")
    limit = _bounded_query_integer(parameters, "limit", 50, 1, TARGET_QUERY_LIMIT)
    offset = _bounded_query_integer(parameters, "offset", 0, 0, 100000)
    return {
        "q": q,
        "lane": lane,
        "sort": sort,
        "order": order,
        "limit": limit,
        "offset": offset,
    }


class TargetCatalog:
    """Immutable, precomputed target-ranking catalog bundled with the desktop app."""

    queue_name = "tess_discovery_queue_v02.csv"
    metadata_name = "metadata.json"

    def __init__(self, root=TARGET_SELECTION_ROOT):
        self.root = Path(root)
        with (self.root / self.metadata_name).open(encoding="utf-8") as handle:
            self._metadata = json.load(handle)
        assets = self._metadata.get("assets")
        if not isinstance(assets, dict):
            raise ValueError("target metadata has no asset manifest")
        for name, expected in assets.items():
            path = self.root / name
            if not path.is_file() or _file_sha256(path) != expected:
                raise ValueError(f"target-selection asset failed integrity check: {name}")

        with (self.root / self.queue_name).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "ID", "queue_rank", "discovery_lane", "lane_score", "Tmag",
                "Teff", "rad", "mass", "known_recorded_planet_count",
                "frozen_candidate_expected_count_lower_bound",
                "frozen_catalog_p_more_than_known", "gap_preselection_score",
                "tess_detectable_percentile", "sector_count_recomputed",
                "sector_list", "search_recipe",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("target catalog schema is incomplete")
            self.rows = tuple(
                {field: _target_value(field, value) for field, value in raw.items()}
                for raw in reader
            )

        expected_rows = int(self._metadata.get("catalog", {}).get("rows", -1))
        if len(self.rows) != expected_rows:
            raise ValueError("target catalog row count does not match metadata")
        self.by_tic = {row["ID"]: row for row in self.rows}
        if len(self.by_tic) != len(self.rows):
            raise ValueError("target catalog contains duplicate TIC IDs")
        if {row["queue_rank"] for row in self.rows} != set(range(1, len(self.rows) + 1)):
            raise ValueError("target catalog queue ranks are not a complete sequence")
        self.lane_counts = dict(sorted(Counter(row["discovery_lane"] for row in self.rows).items()))

    @staticmethod
    def _compact(row):
        return {
            "tic_id": row["ID"],
            "queue_rank": row["queue_rank"],
            "discovery_lane": row["discovery_lane"],
            "lane_score": row["lane_score"],
            "tmag": row["Tmag"],
            "teff_k": row["Teff"],
            "radius_solar": row["rad"],
            "mass_solar": row["mass"],
            "known_planets": row["known_recorded_planet_count"],
            "catalog_expected_count_lower_bound": row["frozen_candidate_expected_count_lower_bound"],
            "catalog_p_more_than_known": row["frozen_catalog_p_more_than_known"],
            "gap_percentile": row["count_gap_percentile"],
            "gap_score": row["gap_preselection_score"],
            "detectability_percentile": row["tess_detectable_percentile"],
            "sector_count": row["sector_count_recomputed"],
            "sectors": row["sector_list"],
            "recent_sector_count": row["recent_sector_count"],
            "sector_span": row["sector_span"],
            "multi_epoch_coverage": row["multi_epoch_coverage"],
            "model_applicability": row["model_applicability"],
            "search_recipe": row["search_recipe"],
        }

    def query(self, raw_query):
        options = parse_target_query(raw_query, frozenset(self.lane_counts))
        rows = self.rows
        if options["q"]:
            rows = tuple(row for row in rows if options["q"] in str(row["ID"]))
        if options["lane"] != "all":
            rows = tuple(row for row in rows if row["discovery_lane"] == options["lane"])

        field = _TARGET_SORT_FIELDS[options["sort"]]
        present = [row for row in rows if row.get(field) is not None]
        missing = [row for row in rows if row.get(field) is None]
        # Stable secondary order stays the frozen queue rank in both directions.
        present.sort(key=lambda row: row["queue_rank"])
        present.sort(key=lambda row: row[field], reverse=options["order"] == "desc")
        ordered = present + missing
        start = options["offset"]
        page = ordered[start:start + options["limit"]]
        return {
            "items": [self._compact(row) for row in page],
            "total": len(ordered),
            "offset": start,
            "limit": options["limit"],
            "query": options,
            "interpretation": "ranking_prior_not_planet_probability",
        }

    def detail(self, tic):
        if not re.fullmatch(r"[0-9]{1,12}", tic):
            raise ValueError("invalid TIC ID")
        row = self.by_tic.get(int(tic))
        if row is None:
            raise KeyError(tic)
        target = dict(row)
        target["tic_id"] = target["ID"]
        return {
            "target": target,
            "interpretation": self._metadata["interpretation"],
            "provenance": {
                "catalog_id": self._metadata["catalog_id"],
                "training_source": self._metadata["training_provenance"]["source"],
                "survey_release": self._metadata["training_provenance"]["survey_release"],
                "queue_sha256": self._metadata["assets"][self.queue_name],
            },
        }

    def metadata(self):
        payload = dict(self._metadata)
        payload["runtime_catalog"] = {
            "rows": len(self.rows),
            "lane_counts": self.lane_counts,
            "maximum_page_size": TARGET_QUERY_LIMIT,
            "sort_keys": sorted(_TARGET_SORT_FIELDS),
            "compact_fields": list(self._compact(self.rows[0])),
        }
        return payload


def _points(time, values, mask, limit=6000):
    indices = np.flatnonzero(mask & np.isfinite(time) & np.isfinite(values))
    if len(indices) > limit:
        # Keep both extrema in each time-ordered bin, including shallow dips.
        groups = np.array_split(indices, limit // 2)
        indices = np.sort(np.array([i for g in groups for i in (g[np.argmin(values[g])], g[np.argmax(values[g])])]))
    return [[float(time[i]), float(values[i])] for i in indices]


def _matrix_payload(values):
    array = np.asarray(values, dtype=float)
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in array
    ]


def _time_diagnostics(time, mask):
    values = np.sort(np.asarray(time, dtype=float)[mask & np.isfinite(time)])
    if len(values) < 2:
        return {"cadence_median_days": None, "gaps": [], "time_min": None, "time_max": None}
    deltas = np.diff(values)
    positive = deltas[np.isfinite(deltas) & (deltas > 0)]
    cadence = float(np.median(positive)) if positive.size else None
    threshold = max(0.25, (cadence or 0.0) * 8.0)
    gap_indices = np.flatnonzero(deltas > threshold)
    gaps = [
        {
            "start": float(values[index]),
            "end": float(values[index + 1]),
            "duration_days": float(deltas[index]),
            "duration_hours": float(deltas[index] * 24.0),
        }
        for index in gap_indices
    ]
    return {
        "cadence_median_days": cadence,
        "gaps": gaps,
        "time_min": float(values[0]),
        "time_max": float(values[-1]),
    }


def _aperture_mask(hdus, shape):
    aperture = next(
        (
            hdu
            for hdu in hdus
            if str(getattr(hdu, "name", "")).upper() == "APERTURE"
            and getattr(hdu, "data", None) is not None
            and np.asarray(hdu.data).shape == shape
        ),
        None,
    )
    if aperture is None:
        return None
    raw = np.asarray(aperture.data)
    if np.issubdtype(raw.dtype, np.integer) and np.any((raw & 2) != 0):
        selected = (raw & 2) != 0
    else:
        selected = raw != 0
    return selected.astype(bool).tolist()


def inspect_fits(content):
    with fits.open(io.BytesIO(content), memmap=False) as hdus:
        table = next((h for h in hdus if getattr(h, "columns", None) is not None and
                      {"TIME", "FLUX"}.issubset({n.upper() for n in h.columns.names})), None)
        if table is None:
            table = next((h for h in hdus if getattr(h, "columns", None) is not None and
                          "TIME" in {n.upper() for n in h.columns.names} and
                          {"PDCSAP_FLUX", "SAP_FLUX"} & {n.upper() for n in h.columns.names}), None)
        if table is None:
            raise ValueError("No compatible TESS light-curve or target-pixel table was found; TIME and flux columns are required")
        names = {n.upper(): n for n in table.columns.names}
        time = np.asarray(table.data[names["TIME"]], dtype=float)
        quality = np.asarray(table.data[names["QUALITY"]]) if "QUALITY" in names else np.zeros(len(time))
        good = (quality == 0) & np.isfinite(time)
        header = hdus[0].header
        ref = float(table.header.get("BJDREFI", 0)) + float(table.header.get("BJDREFF", 0))
        meta = {"target": str(header.get("TICID", header.get("OBJECT", "Unknown"))),
                "sector": str(header.get("SECTOR", "Unknown")), "total": len(time),
                "quality_good": int(np.count_nonzero(good)), "has_quality": "QUALITY" in names,
                "time_label": "BTJD (days; BJD − 2457000)" if abs(ref - 2457000) < .01 else "TIME (days; FITS reference)",
                "time_reference": ref}
        meta.update(_time_diagnostics(time, good))
        if "FLUX" in names and np.asarray(table.data[names["FLUX"]]).ndim == 3:
            cube = np.asarray(table.data[names["FLUX"]], dtype=float)
            if cube.shape[1] * cube.shape[2] > 4096:
                raise ValueError("The pixel array exceeds the viewer's supported dimensions")
            finite_pixel_count = np.count_nonzero(np.isfinite(cube), axis=(1, 2))
            nonempty_good = good & (finite_pixel_count > 0)
            if not np.any(nonempty_good):
                raise ValueError("No quality-clean frame contains valid pixel data")
            # Use the normal good-frame coverage as the reference.  A frame with
            # materially fewer finite pixels would otherwise look like a real
            # downward excursion when its NaNs are silently omitted by nansum.
            reference_pixel_count = int(np.median(finite_pixel_count[nonempty_good]))
            minimum_pixel_count = max(
                1, int(math.ceil(reference_pixel_count * MIN_PIXEL_COVERAGE_RATIO))
            )
            coverage_ok = finite_pixel_count >= minimum_pixel_count
            available = np.flatnonzero(good & coverage_ok)
            if not len(available):
                raise ValueError("No quality-clean frame meets the minimum pixel-coverage requirement")
            i = int(available[len(available)//2])
            valid_cube = cube[available]
            with np.errstate(invalid="ignore"):
                median_frame = np.nanmedian(valid_cube, axis=0)
                variability = 1.4826 * np.nanmedian(
                    np.abs(valid_cube - median_frame[None, :, :]), axis=0
                )
                total_flux = np.full(len(cube), np.nan, dtype=float)
                total_flux[coverage_ok] = np.nansum(cube[coverage_ok], axis=(1, 2))
            series_mask = good & coverage_ok & np.isfinite(total_flux)
            cadence_number = None
            if "CADENCENO" in names:
                raw_cadence = table.data[names["CADENCENO"]][i]
                if np.isfinite(raw_cadence):
                    cadence_number = int(raw_cadence)
            frame = _matrix_payload(cube[i])
            meta.update(kind="pixel", column="FLUX", unit=str(table.columns[names["FLUX"]].unit or "unknown"),
                        cadence=cadence_number if cadence_number is not None else i,
                        cadence_source="CADENCENO" if cadence_number is not None else "frame_index",
                        frame_index=i, frame_time=float(time[i]), frame=frame,
                        images={
                            "frame": frame,
                            "median": _matrix_payload(median_frame),
                            "variability": _matrix_payload(variability),
                        },
                        total_flux_points=_points(time, total_flux, series_mask),
                        aperture_mask=_aperture_mask(hdus, cube.shape[1:]),
                        valid_frames=int(len(available)),
                        total_flux_reference_pixel_count=reference_pixel_count,
                        total_flux_minimum_pixel_count=minimum_pixel_count,
                        total_flux_minimum_coverage_ratio=MIN_PIXEL_COVERAGE_RATIO)
        else:
            col = "PDCSAP_FLUX" if "PDCSAP_FLUX" in names else "SAP_FLUX"
            values = np.asarray(table.data[names[col]], dtype=float)
            finite_values = values[good & np.isfinite(values)]
            flux_median = float(np.median(finite_values)) if finite_values.size else None
            robust_scatter = None
            if finite_values.size and flux_median not in (None, 0.0):
                robust_scatter = float(
                    1.4826 * np.median(np.abs(finite_values - flux_median))
                    / abs(flux_median) * 1e6
                )
            meta.update(kind="lightcurve", column=col, unit=str(table.columns[names[col]].unit or "unknown"),
                        points=_points(time, values, good), valid=int(np.count_nonzero(good & np.isfinite(values))),
                        flux_median=flux_median, robust_scatter_ppm=robust_scatter)
        return meta


def inspect_lightcurve_paths(paths, *, point_budget=24000):
    """Return a bounded, quality-clean native-plot payload for local products."""
    usable = [
        Path(path).resolve()
        for path in paths
        if Path(path).is_file() and not Path(path).name.startswith("._")
    ]
    if not usable:
        return {"status": "pending", "series": [], "sectors": [], "gaps": []}

    per_file_limit = max(500, min(6000, point_budget // len(usable)))
    series = []
    sectors = []
    gaps = []
    sample_count = 0
    all_times = []
    cadence_days = []
    scatter_ppm = []
    target = None
    product = None
    for path in usable:
        try:
            with fits.open(path, memmap=True) as hdus:
                table = next(
                    (
                        hdu
                        for hdu in hdus
                        if getattr(hdu, "columns", None) is not None
                        and "TIME" in {name.upper() for name in hdu.columns.names}
                        and {"PDCSAP_FLUX", "SAP_FLUX"}
                        & {name.upper() for name in hdu.columns.names}
                    ),
                    None,
                )
                if table is None:
                    continue
                names = {name.upper(): name for name in table.columns.names}
                column = "PDCSAP_FLUX" if "PDCSAP_FLUX" in names else "SAP_FLUX"
                time = np.asarray(table.data[names["TIME"]], dtype=float)
                flux = np.asarray(table.data[names[column]], dtype=float)
                quality = (
                    np.asarray(table.data[names["QUALITY"]])
                    if "QUALITY" in names
                    else np.zeros(len(time))
                )
                mask = (quality == 0) & np.isfinite(time) & np.isfinite(flux)
                if not np.any(mask):
                    continue
                median = float(np.median(flux[mask]))
                if not math.isfinite(median) or median == 0:
                    continue
                normalized = flux / median
                header = hdus[0].header
                sector_value = header.get("SECTOR", table.header.get("SECTOR"))
                try:
                    sector = int(sector_value)
                except (TypeError, ValueError):
                    sector = len(sectors) + 1
                diagnostics = _time_diagnostics(time, mask)
                sector_gaps = diagnostics["gaps"]
                for gap in sector_gaps:
                    gap.update(
                        label=f"Sector {sector} data gap",
                        quality_boundary=True,
                    )
                gaps.extend(sector_gaps)
                series.append(
                    {
                        "name": f"Sector {sector}",
                        "sector": sector,
                        "normalized": True,
                        "points": _points(time, normalized, mask, limit=per_file_limit),
                    }
                )
                sectors.append(sector)
                sample_count += int(np.count_nonzero(mask))
                all_times.extend(
                    value
                    for value in (diagnostics["time_min"], diagnostics["time_max"])
                    if value is not None
                )
                if diagnostics["cadence_median_days"] is not None:
                    cadence_days.append(diagnostics["cadence_median_days"])
                scatter_ppm.append(
                    float(1.4826 * np.median(np.abs(normalized[mask] - 1.0)) * 1e6)
                )
                target = target or str(header.get("TICID", header.get("OBJECT", "Unknown")))
                product = product or column
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            continue

    if not series:
        return {"status": "pending", "series": [], "sectors": [], "gaps": []}
    ordered = sorted(zip(sectors, series, strict=True), key=lambda pair: pair[0])
    sectors = [pair[0] for pair in ordered]
    series = [pair[1] for pair in ordered]
    return {
        "status": "ready",
        "target": target or "Unknown",
        "product": product or "TESS flux",
        "normalized": True,
        "series": series,
        "sectors": sectors,
        "gaps": gaps,
        "file_count": len(series),
        "valid_samples": sample_count,
        "cadence_median_days": float(np.median(cadence_days)) if cadence_days else None,
        "time_label": "Time (BTJD)",
        "baseline_days": max(all_times) - min(all_times) if len(all_times) >= 2 else None,
        "robust_scatter_ppm": float(np.median(scatter_ppm)) if scatter_ppm else None,
    }


def _job_lightcurve_paths(jobs, job_id):
    """Resolve only light curves belonging to the local Avril data root."""
    job = jobs.get(job_id)  # Validates that the requested job exists.
    job_root = (jobs.root / job_id).resolve()
    data_root = jobs.root.parent.resolve()
    run_root = job_root / "run"
    candidates = []

    ffi_receipt = run_root / "ffi_source_receipt.json"
    if ffi_receipt.is_file():
        try:
            candidates.extend(
                Path(value).resolve()
                for value in json.loads(ffi_receipt.read_text(encoding="utf-8")).get(
                    "lightcurve_paths", []
                )
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    run_receipt = run_root / "avril_run_receipt.json"
    if run_receipt.is_file():
        try:
            receipt = json.loads(run_receipt.read_text(encoding="utf-8"))
            manifest_path = Path(receipt.get("manifest", "")).resolve()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for target in manifest.get("targets", []):
                for product in target.get("products", []):
                    # Public Avril manifests use ``filename``/``product_id``;
                    # keep ``productFilename`` for older MAST-shaped receipts.
                    filename = Path(str(
                        product.get("filename")
                        or product.get("product_id")
                        or product.get("productFilename")
                        or ""
                    )).name
                    if filename and not filename.startswith("._"):
                        candidates.append(data_root / "cache" / "spoc" / filename)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    candidates.extend(run_root.rglob("*-lc.fits"))
    match = re.search(r"(?i)TIC\s*([0-9]+)", str(job.get("target") or ""))
    if match:
        tic = int(match.group(1))
        # A sector becomes scientifically viewable as soon as its SPOC FITS
        # lands in the cache, before the final run receipt is written.
        candidates.extend(
            (data_root / "cache" / "spoc").glob(f"*-{tic:016d}-*_lc.fits")
        )
        candidates.extend(
            (data_root / "cache" / "ffi" / f"tic-{tic}" / "lightcurves").glob(
                "*-lc.fits"
            )
        )

    resolved = []
    seen = set()
    for candidate in candidates:
        path = Path(candidate).resolve()
        if (
            path.name.startswith("._")
            or not path.is_file()
            or not (path.is_relative_to(data_root) or path.is_relative_to(job_root))
            or path in seen
        ):
            continue
        seen.add(path)
        resolved.append(path)
    return resolved


def _job_candidate_audit(jobs, job_id, candidate_id):
    """Return one complete audit row without exposing arbitrary file paths."""
    if _JOB_ID_PATTERN.fullmatch(job_id) is None:
        raise ValueError("invalid_job_id")
    if _CANDIDATE_ID_PATTERN.fullmatch(candidate_id) is None:
        raise ValueError("invalid_candidate_id")

    # Check existence separately so callers can distinguish a missing job from
    # a valid job whose audit ledger does not contain the requested candidate.
    jobs.get(job_id)
    audit_path = jobs.audit_path(job_id)
    with gzip.open(audit_path, "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError("audit receipt must contain a JSON list")
    for record in records:
        if not isinstance(record, dict):
            continue
        if (
            record.get("candidate_id") == candidate_id
            or record.get("proposal_id") == candidate_id
        ):
            return record
    raise LookupError(candidate_id)


def _job_science_timeline(jobs, job_id):
    """Return the real per-candidate event lanes sealed in the full audit.

    A candidate can contain the same predicted events in both the early gap
    diagnostic and the later red-noise significance receipt.  Merging those
    arrays would display one physical event twice.  Prefer the gap diagnostic
    when it is present because it is the only receipt that also records
    coverage and gap compromise; use ``detection_significance.events`` only
    when no gap-diagnostic events were sealed for that candidate.
    """
    if _JOB_ID_PATTERN.fullmatch(job_id) is None:
        raise ValueError("invalid_job_id")

    # Resolve through JobStore, then independently prove that the receipt is
    # inside the requested job.  A persisted absolute path is never accepted
    # merely because it happens to name an existing gzip file.
    job = jobs.get(job_id)
    jobs_root = jobs.root.resolve()
    job_root = (jobs_root / job_id).resolve()
    if not job_root.is_relative_to(jobs_root):
        raise ValueError("invalid_job_id")
    audit_path = jobs.audit_path(job_id).resolve()
    if not audit_path.is_relative_to(job_root) or not audit_path.is_file():
        raise TypeError("audit receipt path escapes the requested job")

    with gzip.open(audit_path, "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError("audit receipt must contain a JSON list")

    lanes = []
    sectors = set()
    centers = []
    total_events = 0
    candidate_ids = set()

    for record in records:
        if not isinstance(record, dict):
            raise TypeError("audit receipt contains a non-object candidate")

        candidate_id = record.get("candidate_id") or record.get("proposal_id")
        if (
            not isinstance(candidate_id, str)
            or _CANDIDATE_ID_PATTERN.fullmatch(candidate_id) is None
        ):
            raise TypeError("audit candidate has no valid identifier")
        if candidate_id in candidate_ids:
            raise TypeError("audit receipt contains duplicate candidate identifiers")
        candidate_ids.add(candidate_id)

        science_review = record.get("science_review")
        if science_review is not None and not isinstance(science_review, dict):
            raise TypeError("candidate science review must be a JSON object")
        science_review = science_review or {}

        meaningful_gate = record.get("meaningful_candidate_gate")
        if meaningful_gate is not None and not isinstance(meaningful_gate, dict):
            raise TypeError("candidate meaningful gate must be a JSON object")
        meaningful_gate = meaningful_gate or {}

        gap_diagnostic = meaningful_gate.get("event_gap_diagnostics")
        if gap_diagnostic is not None and not isinstance(gap_diagnostic, dict):
            raise TypeError("candidate gap diagnostic must be a JSON object")
        gap_events = []
        if gap_diagnostic is not None:
            gap_events = gap_diagnostic.get("events") or []
            if not isinstance(gap_events, list):
                raise TypeError("candidate gap events must be a JSON list")

        detection = record.get("detection_significance")
        if detection is not None and not isinstance(detection, dict):
            raise TypeError("candidate detection significance must be a JSON object")
        detection_events = []
        if detection is not None:
            detection_events = detection.get("events") or []
            if not isinstance(detection_events, list):
                raise TypeError("candidate significance events must be a JSON list")

        if gap_events:
            selected_events = gap_events
            event_source = (
                "meaningful_candidate_gate.event_gap_diagnostics.events"
            )
        elif detection_events:
            selected_events = detection_events
            event_source = "detection_significance.events"
        else:
            selected_events = []
            event_source = None

        # Preserve every selected event object and every measured value.  The
        # two nullable aliases make coverage/risk uniform for the native UI;
        # null means that the fallback significance receipt did not measure
        # that diagnostic, never that the event was proven clean.
        events = []
        for raw_event in selected_events:
            if not isinstance(raw_event, dict):
                raise TypeError("candidate event must be a JSON object")
            event = dict(raw_event)
            for key in ("segment", "center_btjd", "ses_like"):
                value = event.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise TypeError(f"candidate event has no valid {key}")
            segment = int(event["segment"])
            if segment <= 0 or float(segment) != float(event["segment"]):
                raise TypeError("candidate event has no valid segment")
            sectors.add(segment)
            centers.append(float(event["center_btjd"]))
            event["coverage"] = event.get("transit_coverage_fraction")
            event["gap_risk"] = event.get("critical_gap_compromise")
            events.append(event)

        # The sealed block measurements cover sectors even for candidates
        # whose event train failed before either per-event receipt was made.
        verification = record.get("holdout_verification")
        if verification is not None and not isinstance(verification, dict):
            raise TypeError("candidate holdout verification must be a JSON object")
        measurements = (
            verification.get("block_measurements") if verification else None
        )
        if measurements is not None:
            if not isinstance(measurements, list):
                raise TypeError("candidate block measurements must be a JSON list")
            for measurement in measurements:
                if not isinstance(measurement, dict):
                    raise TypeError("candidate block measurement must be an object")
                measured_segments = measurement.get("segments")
                if measured_segments is None:
                    continue
                if not isinstance(measured_segments, list):
                    raise TypeError("candidate block segments must be a JSON list")
                for raw_segment in measured_segments:
                    if (
                        isinstance(raw_segment, bool)
                        or not isinstance(raw_segment, (int, float))
                    ):
                        raise TypeError("candidate block has an invalid segment")
                    segment = int(raw_segment)
                    if segment <= 0 or float(segment) != float(raw_segment):
                        raise TypeError("candidate block has an invalid segment")
                    sectors.add(segment)

        priority = record.get("priority")
        if priority is None:
            priority = science_review.get("priority")
        disposition = record.get("disposition")
        if disposition is None:
            disposition = meaningful_gate.get("disposition")
        if priority is not None and not isinstance(priority, str):
            raise TypeError("candidate priority must be text")
        if disposition is not None and not isinstance(disposition, str):
            raise TypeError("candidate disposition must be text")

        lane = {
            "candidate_id": candidate_id,
            "rank": record.get("rank"),
            "period_days": record.get("period_days"),
            "epoch_btjd": record.get("epoch_btjd"),
            "duration_hours": record.get("duration_hours"),
            "priority": priority,
            "disposition": disposition,
            "event_source": event_source,
            "event_count": len(events),
            "events": events,
        }
        lanes.append(lane)
        total_events += len(events)

    return {
        "schema_version": "avril-native-science-timeline-v1",
        "job_id": job_id,
        "target": job.get("target"),
        "status": job.get("status"),
        "min_btjd": min(centers) if centers else None,
        "max_btjd": max(centers) if centers else None,
        "lane_count": len(lanes),
        "event_count": total_events,
        "sectors": sorted(sectors),
        "event_source_policy": (
            "gap diagnostics when available; detection significance otherwise; "
            "sources are never merged"
        ),
        "lanes": lanes,
    }


def _job_execution_events(jobs, job_id):
    """Read the complete persisted execution ledger for one local job.

    The desktop history view needs the exact event stream that was sealed in
    ``job.json`` rather than a reconstruction from the final result.  Keep the
    endpoint deliberately read-only: commands are returned as recorded text
    and are never interpreted or executed here.
    """
    if _JOB_ID_PATTERN.fullmatch(job_id) is None:
        raise ValueError("invalid_job_id")

    jobs_root = jobs.root.resolve()
    job_directory = (jobs_root / job_id).resolve()
    job_path = (job_directory / "job.json").resolve()
    if (
        not job_directory.is_relative_to(jobs_root)
        or job_path.parent != job_directory
        or not job_path.is_relative_to(jobs_root)
    ):
        raise ValueError("invalid_job_id")
    if not job_path.is_file():
        raise FileNotFoundError(job_path)

    with job_path.open(encoding="utf-8") as handle:
        persisted = json.load(handle)
    if not isinstance(persisted, dict):
        raise TypeError("job receipt must contain a JSON object")
    if persisted.get("id") != job_id:
        raise TypeError("job receipt identifier does not match its directory")
    raw_events = persisted.get("events")
    if not isinstance(raw_events, list):
        raise TypeError("job receipt events must be a JSON list")

    events = []
    for raw in raw_events:
        if not isinstance(raw, dict):
            raise TypeError("job receipt contains a non-object event")
        metrics = raw.get("metrics")
        if metrics is not None and not isinstance(metrics, dict):
            raise TypeError("job event metrics must be a JSON object")
        # ``at`` is the historical on-disk name.  The native contract calls
        # the same recorded timestamp ``time`` so clients do not need to know
        # storage internals; no timestamp is synthesized when it is absent.
        events.append(
            {
                "seq": raw.get("seq"),
                "time": raw.get("time", raw.get("at")),
                "stage": raw.get("stage"),
                "level": raw.get("level"),
                "title": raw.get("title"),
                "message": raw.get("message"),
                "command": raw.get("command"),
                "metrics": metrics,
            }
        )

    # JobStore derives elapsed durations and the live queue position from the
    # same persisted timestamps.  It also excludes control-token material.
    summary = jobs.get(job_id)
    return {
        "schema_version": "avril-native-job-events-v1",
        "job_id": job_id,
        "target": persisted.get("target"),
        "status": persisted.get("status"),
        "progress": {
            "stage": persisted.get("stage"),
            "percent": persisted.get("percent"),
            "message": persisted.get("message"),
            "created_at": persisted.get("created_at"),
            "updated_at": persisted.get("updated_at"),
            "started_at": persisted.get("started_at"),
            "completed_at": persisted.get("completed_at"),
        },
        "queue": {
            "position": summary.get("queue_position"),
            "seconds": summary.get("queue_seconds"),
        },
        "compute": {"seconds": summary.get("compute_seconds")},
        "total": {"seconds": summary.get("total_seconds")},
        # The original browser's ACTIONS counter is the number of recorded
        # execution events.  Derive it from this exact persisted ledger rather
        # than fabricating runnable controls from the command text.
        "actions": len(events),
        "events": events,
    }


def _candidate_sector_partition(candidate):
    """Recover the sealed discovery/holdout split from the audit receipt."""
    verification = candidate.get("holdout_verification")
    if not isinstance(verification, dict):
        raise TypeError("candidate has no holdout verification receipt")
    measurements = verification.get("block_measurements")
    if not isinstance(measurements, list):
        raise TypeError("candidate has no block measurements")

    discovered_blocks = set()
    for source in (
        candidate.get("discovered_blocks", []),
        verification.get("discovered_blocks", []),
    ):
        if source is None:
            continue
        if not isinstance(source, list):
            raise TypeError("discovered_blocks must be a list")
        for value in source:
            try:
                discovered_blocks.add(int(value))
            except (TypeError, ValueError) as exc:
                raise TypeError("invalid discovered block identifier") from exc

    discovery_sectors = set()
    holdout_sectors = set()
    block_partition = []
    for measurement in measurements:
        if not isinstance(measurement, dict):
            raise TypeError("invalid block measurement")
        try:
            block = int(measurement["block"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TypeError("block measurement has no valid block identifier") from exc
        segments = measurement.get("segments")
        if not isinstance(segments, list):
            raise TypeError("block measurement has no segment list")
        try:
            sectors = {int(value) for value in segments}
        except (TypeError, ValueError) as exc:
            raise TypeError("block measurement has an invalid sector") from exc
        if any(sector <= 0 for sector in sectors):
            raise TypeError("block measurement has an invalid sector")

        independently_discovered = measurement.get("independently_discovered") is True
        is_discovery = independently_discovered or block in discovered_blocks
        if is_discovery:
            discovery_sectors.update(sectors)
        else:
            holdout_sectors.update(sectors)
        block_partition.append(
            {
                "block": block,
                "sectors": sorted(sectors),
                "group": "discovery" if is_discovery else "holdout",
                "independently_discovered": independently_discovered,
            }
        )

    overlap = discovery_sectors.intersection(holdout_sectors)
    if overlap:
        raise TypeError("a sector appears in both discovery and holdout blocks")
    return (
        sorted(discovery_sectors),
        sorted(holdout_sectors),
        block_partition,
    )


def _fits_sector(hdus, table, path):
    value = hdus[0].header.get("SECTOR", table.header.get("SECTOR"))
    try:
        sector = int(value)
        if sector > 0:
            return sector
    except (TypeError, ValueError):
        pass
    match = re.search(r"-s([0-9]{4})-", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _fold_bins(phases, flux_ppm, half_width, count=FOLD_BIN_COUNT):
    """Aggregate a locked fold with a robust standard error per phase bin."""
    if not phases:
        return []
    phase = np.concatenate(phases)
    values = np.concatenate(flux_ppm)
    finite = np.isfinite(phase) & np.isfinite(values)
    phase = phase[finite]
    values = values[finite]
    if phase.size == 0:
        return []

    edges = np.linspace(-half_width, half_width, count + 1, dtype=float)
    indices = np.searchsorted(edges, phase, side="right") - 1
    # The right edge is part of the final bin.
    indices[np.isclose(phase, half_width, rtol=0.0, atol=np.finfo(float).eps * 8)] = count - 1
    valid = (indices >= 0) & (indices < count)
    indices = indices[valid]
    values = values[valid]
    bins = []
    for index in range(count):
        samples = values[indices == index]
        if samples.size == 0:
            continue
        median = float(np.median(samples))
        mad = float(np.median(np.abs(samples - median)))
        error = 1.4826 * mad / math.sqrt(int(samples.size))
        center = float((edges[index] + edges[index + 1]) / 2.0)
        bins.append(
            [
                center,
                median,
                median - error,
                median + error,
                int(samples.size),
            ]
        )
    return bins


def _job_candidate_folds(jobs, job_id, candidate_id):
    """Fold every quality-clean FITS sample at the sealed candidate ephemeris."""
    candidate = _job_candidate_audit(jobs, job_id, candidate_id)
    try:
        period_days = float(candidate["period_days"])
        epoch_btjd = float(candidate["epoch_btjd"])
        duration_hours = float(candidate["duration_hours"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TypeError("candidate has no valid sealed ephemeris") from exc
    if not (
        math.isfinite(period_days)
        and period_days > 0
        and math.isfinite(epoch_btjd)
        and math.isfinite(duration_hours)
        and duration_hours > 0
    ):
        raise TypeError("candidate has no valid sealed ephemeris")

    discovery_sectors, holdout_sectors, block_partition = (
        _candidate_sector_partition(candidate)
    )
    discovery_set = set(discovery_sectors)
    holdout_set = set(holdout_sectors)
    duration_days = duration_hours / 24.0
    raw_half_width = FOLD_HALF_WINDOW_DURATIONS * duration_days / period_days
    half_width = min(0.25, raw_half_width)
    if not math.isfinite(half_width) or half_width <= 0:
        raise TypeError("candidate has no valid phase window")

    group_data = {
        "discovery": {
            "declared_sectors": discovery_sectors,
            "sectors": set(),
            "quality_clean_samples": 0,
            "window_samples": 0,
            "file_count": 0,
            "phases": [],
            "flux_ppm": [],
            "products": set(),
        },
        "holdout": {
            "declared_sectors": holdout_sectors,
            "sectors": set(),
            "quality_clean_samples": 0,
            "window_samples": 0,
            "file_count": 0,
            "phases": [],
            "flux_ppm": [],
            "products": set(),
        },
    }
    skipped_files = 0
    unassigned_sectors = set()
    source_paths = _job_lightcurve_paths(jobs, job_id)
    for path in source_paths:
        try:
            with fits.open(path, memmap=True) as hdus:
                table = next(
                    (
                        hdu
                        for hdu in hdus
                        if getattr(hdu, "columns", None) is not None
                        and "TIME" in {name.upper() for name in hdu.columns.names}
                        and "QUALITY" in {name.upper() for name in hdu.columns.names}
                        and {"PDCSAP_FLUX", "SAP_FLUX"}
                        & {name.upper() for name in hdu.columns.names}
                    ),
                    None,
                )
                if table is None:
                    skipped_files += 1
                    continue
                names = {name.upper(): name for name in table.columns.names}
                product = "PDCSAP_FLUX" if "PDCSAP_FLUX" in names else "SAP_FLUX"
                time = np.asarray(table.data[names["TIME"]], dtype=float)
                flux = np.asarray(table.data[names[product]], dtype=float)
                quality = np.asarray(table.data[names["QUALITY"]])
                if time.ndim != 1 or flux.ndim != 1 or quality.ndim != 1:
                    skipped_files += 1
                    continue
                if not (len(time) == len(flux) == len(quality)):
                    skipped_files += 1
                    continue
                sector = _fits_sector(hdus, table, path)
                if sector in discovery_set:
                    group_name = "discovery"
                elif sector in holdout_set:
                    group_name = "holdout"
                else:
                    if sector is not None:
                        unassigned_sectors.add(sector)
                    skipped_files += 1
                    continue

                clean = (quality == 0) & np.isfinite(time) & np.isfinite(flux)
                clean_count = int(np.count_nonzero(clean))
                if clean_count == 0:
                    skipped_files += 1
                    continue
                median_flux = float(np.median(flux[clean]))
                if not math.isfinite(median_flux) or median_flux == 0:
                    skipped_files += 1
                    continue

                reference = float(table.header.get("BJDREFI", 0.0)) + float(
                    table.header.get("BJDREFF", 0.0)
                )
                # TESS products normally store TIME directly as BTJD with a
                # BJDREF of 2457000. Apply the declared reference explicitly so
                # an otherwise compatible FITS product cannot shift the fold.
                time_btjd = (
                    time + reference - 2457000.0
                    if math.isfinite(reference) and reference != 0.0
                    else time
                )
                clean_time = time_btjd[clean]
                normalized_ppm = (flux[clean] / median_flux - 1.0) * 1e6
                centered_phase = (
                    np.mod(clean_time - epoch_btjd + period_days / 2.0, period_days)
                    - period_days / 2.0
                ) / period_days
                in_window = np.abs(centered_phase) <= half_width

                group = group_data[group_name]
                group["quality_clean_samples"] += clean_count
                group["window_samples"] += int(np.count_nonzero(in_window))
                group["file_count"] += 1
                group["sectors"].add(sector)
                group["products"].add(product)
                if np.any(in_window):
                    # Only the plotting window is retained in memory, but every
                    # quality-clean sample participated in the locked fold.
                    group["phases"].append(centered_phase[in_window])
                    group["flux_ppm"].append(normalized_ppm[in_window])
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            skipped_files += 1

    folds = {}
    for name, group in group_data.items():
        bins = _fold_bins(group["phases"], group["flux_ppm"], half_width)
        folds[name] = {
            "declared_sectors": group["declared_sectors"],
            "sectors": sorted(group["sectors"]),
            "file_count": group["file_count"],
            "quality_clean_samples": group["quality_clean_samples"],
            "window_samples": group["window_samples"],
            "products": sorted(group["products"]),
            "requested_bin_count": FOLD_BIN_COUNT,
            "returned_bin_count": len(bins),
            "bins": bins,
        }

    return {
        "schema_version": "avril-native-locked-phase-fold-v1",
        "job_id": job_id,
        "candidate_id": candidate_id,
        "ephemeris_locked": True,
        "period_days": period_days,
        "epoch_btjd": epoch_btjd,
        "duration_hours": duration_hours,
        "duration_days": duration_days,
        "phase_window": {
            "minimum_cycles": -half_width,
            "maximum_cycles": half_width,
            "half_width_cycles": half_width,
            "half_width_days": half_width * period_days,
            "half_width_duration_multiples": min(
                FOLD_HALF_WINDOW_DURATIONS,
                0.25 * period_days / duration_days,
            ),
            "clipped_at_quarter_period": raw_half_width > 0.25,
        },
        "sector_partition": {
            "discovery": discovery_sectors,
            "holdout": holdout_sectors,
            "unassigned_available": sorted(unassigned_sectors),
            "blocks": block_partition,
        },
        "sectors": {
            "discovery": discovery_sectors,
            "holdout": holdout_sectors,
            "unassigned_available": sorted(unassigned_sectors),
        },
        "samples": {
            "source_files_resolved": len(source_paths),
            "source_files_used": sum(
                group["file_count"] for group in group_data.values()
            ),
            "source_files_skipped": skipped_files,
            "quality_clean_total": sum(
                group["quality_clean_samples"] for group in group_data.values()
            ),
            "window_total": sum(
                group["window_samples"] for group in group_data.values()
            ),
        },
        "method": {
            "time_basis": "BTJD = FITS TIME + BJDREF - 2457000 (raw TIME when BJDREF is absent)",
            "quality_filter": "QUALITY == 0 and finite TIME/flux",
            "flux_selection": "PDCSAP_FLUX preferred; SAP_FLUX fallback",
            "normalization": "independent quality-clean median per FITS product",
            "phase_definition": "((TIME - epoch + P/2) mod P - P/2) / P",
            "phase_coordinate": "equal-width bin center in cycles",
            "aggregation": "median normalized flux in ppm",
            "uncertainty": "median +/- 1.4826 * MAD / sqrt(n)",
            "decimation": "none",
            "ephemeris_refit": "disabled",
        },
        "folds": folds,
    }


class Handler(VLoraRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/desktop/target-model/metadata":
            if not self._authorized():
                return
            self._json(HTTPStatus.OK, self.server.target_catalog.metadata())
            return
        if path == "/api/desktop/targets":
            if not self._authorized():
                return
            try:
                self._json(HTTPStatus.OK, self.server.target_catalog.query(parsed.query))
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_target_query", "detail": str(exc)})
            return
        if path == "/api/desktop/jobs":
            if not self._authorized():
                return
            self._json(HTTPStatus.OK, self.server.jobs.list_records())
            return
        match = re.fullmatch(r"/api/desktop/jobs/([^/]+)/timeline", path)
        if match:
            if not self._authorized():
                return
            try:
                self._json(
                    HTTPStatus.OK,
                    _job_science_timeline(self.server.jobs, match.group(1)),
                )
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except KeyError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "job_not_found"})
            except FileNotFoundError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "audit_not_found"})
            except (OSError, TypeError, json.JSONDecodeError):
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "invalid_audit_receipt"},
                )
            return
        match = re.fullmatch(r"/api/desktop/jobs/([^/]+)/events", path)
        if match:
            if not self._authorized():
                return
            try:
                self._json(
                    HTTPStatus.OK,
                    _job_execution_events(self.server.jobs, match.group(1)),
                )
            except json.JSONDecodeError:
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "invalid_job_receipt"},
                )
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except (KeyError, FileNotFoundError):
                self._json(HTTPStatus.NOT_FOUND, {"error": "job_not_found"})
            except (OSError, TypeError):
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "invalid_job_receipt"},
                )
            return
        match = re.fullmatch(
            r"/api/desktop/jobs/([^/]+)/candidates/([^/]+)/folds", path
        )
        if match:
            if not self._authorized():
                return
            job_id, candidate_id = match.groups()
            try:
                payload = _job_candidate_folds(
                    self.server.jobs, job_id, candidate_id
                )
                self._json(HTTPStatus.OK, payload)
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except KeyError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "job_not_found"})
            except LookupError:
                self._json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "candidate_not_found"},
                )
            except FileNotFoundError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "audit_not_found"})
            except (OSError, TypeError, json.JSONDecodeError):
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "invalid_candidate_receipt"},
                )
            return
        match = re.fullmatch(
            r"/api/desktop/jobs/([^/]+)/candidates/([^/]+)", path
        )
        if match:
            if not self._authorized():
                return
            job_id, candidate_id = match.groups()
            try:
                payload = _job_candidate_audit(
                    self.server.jobs, job_id, candidate_id
                )
                self._json(HTTPStatus.OK, payload)
            except ValueError as exc:
                error = str(exc)
                self._json(HTTPStatus.BAD_REQUEST, {"error": error})
            except KeyError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "job_not_found"})
            except LookupError:
                self._json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "candidate_not_found"},
                )
            except FileNotFoundError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "audit_not_found"})
            except (OSError, TypeError, json.JSONDecodeError):
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "invalid_audit_receipt"},
                )
            return
        if path.startswith("/api/desktop/targets/"):
            if not self._authorized():
                return
            tic = path.removeprefix("/api/desktop/targets/")
            try:
                self._json(HTTPStatus.OK, self.server.target_catalog.detail(tic))
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_tic_id", "detail": str(exc)})
            except KeyError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "target_not_found"})
            return
        match = re.fullmatch(r"/api/desktop/jobs/([a-zA-Z0-9_-]+)/lightcurve", path)
        if match:
            if not self._authorized():
                return
            try:
                paths = _job_lightcurve_paths(self.server.jobs, match.group(1))
                payload = inspect_lightcurve_paths(paths)
                payload["job_status"] = self.server.jobs.get(match.group(1)).get("status")
                self._json(HTTPStatus.OK, payload)
            except KeyError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "job_not_found"})
            return
        super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/desktop/fits":
            return super().do_POST()
        if not self._authorized():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_FILE:
                raise ValueError("The file is empty or exceeds the 128 MiB viewer limit")
            self._json(HTTPStatus.OK, inspect_fits(self.rfile.read(length)))
        except (ValueError, OSError, KeyError, IndexError, TypeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--web-root", type=Path, required=True)
    parser.add_argument("--allow-origin", action="append", default=[])
    args = parser.parse_args()
    target_catalog = TargetCatalog()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.jobs = JobStore()
    server.target_catalog = target_catalog
    server.web_root = args.web_root.resolve()
    server.allowed_origins = frozenset(args.allow_origin)

    # QProcess normally sends SIGTERM during an orderly close. If the native
    # GUI is force-quit or crashes inside a graphics driver, however, macOS
    # reparents this process and no Qt destructor can run. Stop the loopback
    # service as soon as its owning Avril process disappears so repeated test
    # launches cannot accumulate orphan Python runtimes.
    if args.parent_pid > 0:
        def stop_when_parent_exits():
            while os.getppid() == args.parent_pid:
                if parent_watchdog_stop.wait(1.0):
                    return
            server.shutdown()

        parent_watchdog_stop = threading.Event()
        parent_watchdog = threading.Thread(
            target=stop_when_parent_exits,
            name="avril-parent-watchdog",
            daemon=True,
        )
        parent_watchdog.start()
    else:
        parent_watchdog_stop = None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if parent_watchdog_stop is not None:
            parent_watchdog_stop.set()
        server.jobs.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
