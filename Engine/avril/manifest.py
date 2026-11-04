"""Build label-free, reproducible manifests from official TESS SPOC products."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from multiplanet_trial.v05_eval.integrity import atomic_json
from vlora.sources import discover_spoc


def _public_product(product: dict) -> dict:
    filename = str(product.get("productFilename") or product.get("filename") or "")
    if not filename:
        raise ValueError("archive product has no filename")
    sector = int(product.get("sequence_number") or product.get("segment"))
    data_uri = str(
        product.get("dataURI")
        or product.get("data_uri")
        or f"mast:TESS/product/{filename}"
    )
    exptime = product.get("exptime")
    return {
        "segment": sector,
        "segment_kind": "sector",
        "product_id": filename,
        "filename": filename,
        "data_uri": data_uri,
        "url": f"https://mast.stsci.edu/api/v0.1/Download/file/?uri={data_uri}",
        "author": str(product.get("author") or "SPOC"),
        "cadence": f"{float(exptime or 120.0):g}s",
        "size_bytes": product.get("size") or product.get("size_bytes"),
        "evaluation_role": "search",
    }


def build_target_manifest(target: str, output: Path) -> dict:
    """Discover one preferred official light curve per available sector."""

    selected, inventory = discover_spoc(target, deep=True)
    products = sorted((_public_product(row) for row in selected), key=lambda row: row["segment"])
    if len(products) < 3:
        raise RuntimeError(
            f"Avril requires at least three official light-curve sectors; found {len(products)}"
        )
    payload = {
        "schema_version": "evaluation-public-v1",
        "dataset_id": f"avril-v3-{''.join(c if c.isalnum() else '-' for c in target).strip('-').lower()}",
        "split": "development",
        "policy": (
            "label-free natural search; one preferred official SPOC or TESS-SPOC "
            "light curve per currently available sector"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "targets": [
            {
                "physical_id": target,
                "target_id": target,
                "mission": "TESS",
                "products": products,
            }
        ],
        "archive_inventory_counts": inventory.get("counts"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    return payload
