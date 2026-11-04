"""Lossless compact storage for large clean-pipeline result dictionaries.

The scientific pipeline still constructs and returns the complete result.  At
the final write only, bulky audit-only collections are moved into one gzip JSON
sidecar.  ``result.json`` keeps every ranked candidate and contains explicit
references for every externalized field.  :func:`load_full_result` reconstructs
the original dictionary exactly.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

from . import pipeline as reference


SCHEMA_VERSION = "clean-result-lossless-external-audit-v1"
AUDIT_FILENAME = "result.audit.json.gz"
EXTERNAL_FIELDS = (
    "candidate_registry",
    "raw_candidate_ledger",
    "cross_block_match_edges",
    "cross_view_match_relations",
    "alias_decisions",
)
NONFINITE_FLOAT_TAG = "__clean_pipeline_nonfinite_float__"


def _encode_nonfinite(value):
    """Represent non-finite floats explicitly so strict JSON stays lossless."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            label = "NaN"
        elif value > 0:
            label = "Infinity"
        else:
            label = "-Infinity"
        return {NONFINITE_FLOAT_TAG: label}
    if isinstance(value, Mapping):
        return {key: _encode_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_nonfinite(item) for item in value]
    if isinstance(value, tuple):
        return [_encode_nonfinite(item) for item in value]
    return value


def _decode_nonfinite(value):
    if isinstance(value, dict):
        if set(value) == {NONFINITE_FLOAT_TAG}:
            return {
                "NaN": float("nan"),
                "Infinity": float("inf"),
                "-Infinity": float("-inf"),
            }[str(value[NONFINITE_FLOAT_TAG])]
        return {key: _decode_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_nonfinite(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_gzip_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                fileobj=raw,
                mode="wb",
                compresslevel=1,
                mtime=0,
            ) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                    for chunk in encoder.iterencode(payload):
                        text.write(chunk)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _collection_count(value) -> int | None:
    return len(value) if isinstance(value, (list, dict)) else None


def write_compact_result(
    path: Path,
    payload: Mapping,
    *,
    reference_writer=reference._write_json,
) -> dict:
    """Write a losslessly reconstructable compact result and return metadata."""

    external = {
        key: payload[key]
        for key in EXTERNAL_FIELDS
        if key in payload
    }
    audit_path = path.parent / AUDIT_FILENAME
    audit_payload = {
        "schema_version": SCHEMA_VERSION,
        "fields": external,
    }
    _atomic_gzip_json(audit_path, _encode_nonfinite(audit_payload))
    audit_sha256 = _sha256_file(audit_path)
    audit_bytes = audit_path.stat().st_size

    references = {
        key: {
            "externalized": True,
            "encoding": "gzip+json",
            "path": audit_path.name,
            "json_pointer": f"/fields/{key}",
            "count": _collection_count(value),
            "sha256": audit_sha256,
        }
        for key, value in external.items()
    }
    compact = dict(payload)
    compact.update(references)
    compact["storage_layout"] = {
        "schema_version": SCHEMA_VERSION,
        "lossless": True,
        "full_result_reconstructable": True,
        "ranked_candidates_externalized": False,
        "audit_path": audit_path.name,
        "audit_bytes": audit_bytes,
        "audit_sha256": audit_sha256,
        "external_fields": references,
    }
    reference_writer(path, _encode_nonfinite(compact))
    return compact["storage_layout"]


def load_full_result(path: str | Path) -> dict:
    """Load either a legacy inline result or a compact result losslessly."""

    result_path = Path(path)
    main = json.loads(result_path.read_text(encoding="utf-8"))
    layout = main.get("storage_layout")
    if not isinstance(layout, dict) or layout.get("schema_version") != SCHEMA_VERSION:
        return _decode_nonfinite(main)
    audit_path = result_path.parent / str(layout["audit_path"])
    if _sha256_file(audit_path) != layout["audit_sha256"]:
        raise ValueError(f"Audit sidecar hash mismatch: {audit_path}")
    with gzip.open(audit_path, "rt", encoding="utf-8") as handle:
        audit = _decode_nonfinite(json.load(handle))
    if audit.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported audit sidecar schema: {audit_path}")
    full = dict(main)
    full.pop("storage_layout", None)
    for key, value in audit["fields"].items():
        full[key] = value
    return _decode_nonfinite(full)


def install() -> None:
    """Install the compact final-result writer in an explicitly opted-in run."""

    current = reference._write_json
    if getattr(current, "_lossless_compact_result_storage", False):
        return

    def writer(path: Path, payload) -> None:
        if path.name == "result.json" and isinstance(payload, Mapping):
            write_compact_result(path, payload, reference_writer=current)
        else:
            current(path, payload)

    writer._lossless_compact_result_storage = True
    reference._write_json = writer


__all__ = [
    "AUDIT_FILENAME",
    "EXTERNAL_FIELDS",
    "SCHEMA_VERSION",
    "install",
    "load_full_result",
    "write_compact_result",
]
