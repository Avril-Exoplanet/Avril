"""Lightkurve-backed archive discovery with explicit local storage.

This module deliberately separates *finding products* from *downloading them*.
Lightkurve knows how to query MAST consistently across TESS, Kepler, and K2;
the caller must still provide an explicit download directory so large products
never fall into an implicit system-disk cache.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import warnings
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import quote

import lightkurve as lk
import numpy as np
from astropy.io import fits


DEFAULT_PASSPORT_CACHE = Path.home() / ".cache" / "avril" / "lightkurve"


def configure_lightkurve_cache(cache_dir: str | Path) -> Path:
    """Set and create the one cache root used by Lightkurve downloads."""
    destination = Path(cache_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    lk.conf.cache_dir = str(destination)
    return destination


def _plain(value):
    if np.ma.is_masked(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _records(search_result, product_kind: str) -> list[dict]:
    fields = (
        "target_name",
        "mission",
        "author",
        "sequence_number",
        "exptime",
        "description",
        "productFilename",
        "dataURI",
        "size",
        "distance",
        "dataRights",
    )
    output = []
    for index, row in enumerate(search_result.table):
        record = {"search_index": index, "product_kind": product_kind}
        for field in fields:
            record[field] = _plain(row[field]) if field in row.colnames else None
        output.append(record)
    return output


def discover_products(
    target: str | int,
    *,
    missions: Iterable[str] = ("TESS", "Kepler", "K2"),
    author: str | tuple[str, ...] | None = None,
    exptime: str | float | None = None,
    include_lightcurves: bool = True,
    include_target_pixels: bool = True,
    limit: int | None = None,
) -> dict:
    """Return a mission-aware MAST inventory without downloading science files."""
    mission_tuple = tuple(missions)
    queries = {}
    products: list[dict] = []
    common = {
        "target": target,
        "mission": mission_tuple,
        "author": author,
        "exptime": exptime,
        "limit": limit,
    }
    if include_lightcurves:
        result = lk.search_lightcurve(**common)
        queries["lightcurve"] = result
        products.extend(_records(result, "lightcurve"))
    if include_target_pixels:
        result = lk.search_targetpixelfile(**common)
        queries["target_pixel"] = result
        products.extend(_records(result, "target_pixel"))
    return {
        "target": str(target),
        "missions_requested": list(mission_tuple),
        "products": products,
        "counts": {
            "lightcurves": sum(p["product_kind"] == "lightcurve" for p in products),
            "target_pixels": sum(p["product_kind"] == "target_pixel" for p in products),
            "total": len(products),
        },
        "_search_results": queries,
    }


def download_inventory(
    inventory: dict,
    *,
    download_dir: str | Path,
    product_kinds: Iterable[str] = ("lightcurve", "target_pixel"),
):
    """Download queried products via Lightkurve into an explicit directory."""
    destination = configure_lightkurve_cache(download_dir)
    wanted = set(product_kinds)
    downloaded = {}
    for kind, result in inventory.get("_search_results", {}).items():
        normalized = "target_pixel" if kind == "target_pixel" else "lightcurve"
        if normalized not in wanted or len(result) == 0:
            continue
        downloaded[normalized] = result.download_all(download_dir=str(destination))
    return downloaded


def _tess_public_mirror(product_filename: str) -> str | None:
    match = re.search(r"-s(\d{4})-(\d{16})-", product_filename)
    if match is None:
        return None
    sector, tic = match.groups()
    chunks = "/".join(tic[index : index + 4] for index in range(0, 16, 4))
    return (
        "https://stpubdata.s3.amazonaws.com/tess/public/tid/"
        f"s{sector}/{chunks}/{product_filename}"
    )


def _complete_fits(path: Path, expected_size: int | None) -> bool:
    try:
        if expected_size and path.stat().st_size < expected_size:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(path, memmap=True) as hdul:
                info = hdul.fileinfo(len(hdul) - 1)
                return path.stat().st_size >= int(info["datLoc"] + info["datSpan"])
    except Exception:
        return False


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_md5(url: str) -> str | None:
    """Return a simple S3 ETag MD5 when the server exposes one."""
    result = subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "20",
            "--max-time",
            "30",
            "--head",
            url,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    matches = re.findall(r'(?im)^etag:\s*"([0-9a-f]{32})"\s*$', result.stdout)
    return matches[-1].lower() if matches else None


def _checksum_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".md5")


def _record_checksum(path: Path, digest: str) -> None:
    _checksum_sidecar(path).write_text(f"{digest}  {path.name}\n", encoding="ascii")


def _recorded_checksum_matches(path: Path) -> bool:
    sidecar = _checksum_sidecar(path)
    if not sidecar.exists():
        return False
    expected = sidecar.read_text(encoding="ascii").split()[0].lower()
    return bool(re.fullmatch(r"[0-9a-f]{32}", expected)) and _file_md5(path) == expected


def file_integrity_record(path: str | Path) -> dict:
    """Return the exact local integrity evidence saved with a science file."""
    resolved = Path(path).resolve()
    actual = _file_md5(resolved)
    sidecar = _checksum_sidecar(resolved)
    recorded = None
    if sidecar.exists():
        candidate = sidecar.read_text(encoding="ascii").split()[0].lower()
        if re.fullmatch(r"[0-9a-f]{32}", candidate):
            recorded = candidate
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "fits_structurally_complete": _complete_fits(resolved, None),
        "md5": actual,
        "recorded_remote_md5": recorded,
        "remote_md5_match": actual == recorded if recorded is not None else None,
    }


def _download_one_resumable(product: Mapping, destination: Path) -> Path:
    filename = str(product["productFilename"])
    expected_size = int(product["size"]) if product.get("size") else None
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / filename
    data_uri = str(product["dataURI"])
    mast_url = (
        "https://mast.stsci.edu/api/v0.1/Download/file?uri="
        + quote(data_uri, safe=":/")
    )
    urls = []
    if str(product.get("mission", "")).startswith("TESS"):
        mirror = _tess_public_mirror(filename)
        if mirror:
            urls.append(mirror)
    urls.append(mast_url)
    if _complete_fits(target, expected_size):
        if _recorded_checksum_matches(target):
            return target
        remote_digest = next(
            (digest for digest in (_remote_md5(url) for url in urls) if digest),
            None,
        )
        if remote_digest is None:
            return target
        if _file_md5(target) == remote_digest:
            _record_checksum(target, remote_digest)
            return target
        quarantined = target.with_suffix(target.suffix + ".bad-md5")
        os.replace(target, quarantined)
    partial = target.with_suffix(target.suffix + ".part")
    last_returncode = None
    for url in urls:
        remote_digest = _remote_md5(url)
        for _attempt in range(10):
            result = subprocess.run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    # Large SPOC target-pixel files can exceed 500 MB.  A
                    # 90-second wall timeout makes every healthy transfer
                    # restart repeatedly on shared MAST/S3 bandwidth.  Keep
                    # byte-range resume, but allow a sustained transfer to
                    # finish before the next retry.  This changes transport
                    # scheduling only; FITS and checksum validation are
                    # unchanged.
                    "900",
                    "-C",
                    "-",
                    "-o",
                    str(partial),
                    url,
                ],
                check=False,
            )
            last_returncode = result.returncode
            if _complete_fits(partial, expected_size):
                if remote_digest is not None and _file_md5(partial) != remote_digest:
                    quarantined = partial.with_suffix(partial.suffix + ".bad-md5")
                    os.replace(partial, quarantined)
                    break
                os.replace(partial, target)
                if remote_digest is not None:
                    _record_checksum(target, remote_digest)
                return target
            # curl 22 means the server returned a permanent HTTP error (most
            # commonly a missing public mirror object).  Retrying the same URL
            # ten times cannot repair it; move immediately to the authoritative
            # MAST URL while retaining retries for timeouts/interrupted transfers.
            if result.returncode == 22:
                break
    raise IOError(f"Failed to download {filename} (curl={last_returncode})")


def download_inventory_resumable(
    inventory: dict,
    *,
    download_dir: str | Path,
    product_kinds: Iterable[str] = ("lightcurve", "target_pixel"),
) -> dict[str, list[Path]]:
    """Download the Lightkurve-discovered inventory with resume and mirrors."""
    destination = configure_lightkurve_cache(download_dir) / "archive-products"
    wanted = set(product_kinds)
    output: dict[str, list[Path]] = {kind: [] for kind in wanted}
    for product in inventory.get("products", []):
        kind = str(product["product_kind"])
        if kind not in wanted:
            continue
        mission = str(product.get("mission") or "unknown").split()[0].lower()
        path = _download_one_resumable(product, destination / mission / kind)
        output.setdefault(kind, []).append(path)
    return output


def serializable_inventory(inventory: dict) -> dict:
    """Remove live SearchResult objects before writing JSON."""
    return {key: value for key, value in inventory.items() if key != "_search_results"}
