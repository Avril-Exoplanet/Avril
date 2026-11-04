"""Label-isolated train/development/blind dataset contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from .integrity import sha256_file


PUBLIC_FORBIDDEN_KEYS = {
    "label",
    "class_label",
    "disposition",
    "dispositions",
    "known_periods_days",
    "truth_periods_days",
    "truth_signals",
    "is_planet",
    "is_false_positive",
    "toi_disposition",
}
ALLOWED_SPLITS = {"train", "development", "blind"}
ALLOWED_CLASSES = {
    "planet_host",
    "eclipsing_binary",
    "variable_star",
    "quiet_star",
    "quiet_control",
    "instrumental_artifact",
    "catalog_false_positive",
}


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _walk_keys(value, location: str = "root") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            current = f"{location}.{key}"
            yield str(key), current
            yield from _walk_keys(child, current)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_keys(child, f"{location}[{index}]")


def assert_public_is_label_free(payload: Mapping) -> None:
    violations = [location for key, location in _walk_keys(payload) if key in PUBLIC_FORBIDDEN_KEYS]
    if violations:
        raise ValueError(f"Public manifest contains forbidden truth fields: {violations[:5]}")


def validate_public_manifest(payload: Mapping) -> dict:
    assert_public_is_label_free(payload)
    if payload.get("schema_version") != "evaluation-public-v1":
        raise ValueError("Unsupported public manifest schema")
    split = payload.get("split")
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Invalid split: {split}")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Public manifest must contain at least one target")
    physical_ids = []
    for target in targets:
        for key in ("physical_id", "target_id", "mission", "products"):
            if key not in target:
                raise ValueError(f"Public target is missing {key}")
        if not isinstance(target["products"], list) or not target["products"]:
            raise ValueError(f"{target['physical_id']} has no products")
        physical_ids.append(str(target["physical_id"]))
    if len(physical_ids) != len(set(physical_ids)):
        raise ValueError("Duplicate physical_id inside public manifest")
    return {
        "dataset_id": payload.get("dataset_id"),
        "split": split,
        "targets": len(targets),
        "physical_ids": set(physical_ids),
    }


def validate_truth_manifest(payload: Mapping, public_payload: Mapping, public_path: str | Path) -> dict:
    if payload.get("schema_version") != "evaluation-truth-v1":
        raise ValueError("Unsupported truth manifest schema")
    public_summary = validate_public_manifest(public_payload)
    if payload.get("dataset_id") != public_summary["dataset_id"]:
        raise ValueError("Truth/public dataset_id mismatch")
    if payload.get("split") != public_summary["split"]:
        raise ValueError("Truth/public split mismatch")
    expected_hash = payload.get("public_manifest_sha256")
    actual_hash = sha256_file(public_path)
    if expected_hash != actual_hash:
        raise ValueError("Truth manifest is bound to a different public manifest")
    targets = payload.get("targets")
    if not isinstance(targets, list):
        raise ValueError("Truth manifest targets must be a list")
    truth_ids = set()
    for target in targets:
        physical_id = str(target.get("physical_id"))
        truth_ids.add(physical_id)
        if target.get("class_label") not in ALLOWED_CLASSES:
            raise ValueError(f"Invalid class for {physical_id}: {target.get('class_label')}")
        if target.get("class_label") == "planet_host" and not target.get("truth_signals"):
            raise ValueError(f"Planet host {physical_id} has no truth_signals")
    if truth_ids != public_summary["physical_ids"]:
        raise ValueError("Truth and public target identities differ")
    return {**public_summary, "classes": sorted({row["class_label"] for row in targets})}


def validate_disjoint_public_manifests(manifests: Iterable[Mapping]) -> dict:
    seen: dict[str, str] = {}
    summaries = []
    for payload in manifests:
        summary = validate_public_manifest(payload)
        summaries.append(summary)
        for physical_id in summary["physical_ids"]:
            if physical_id in seen:
                raise ValueError(
                    f"Physical target leakage: {physical_id} occurs in {seen[physical_id]} and {summary['split']}"
                )
            seen[physical_id] = summary["split"]
    splits = [item["split"] for item in summaries]
    if len(splits) != len(set(splits)):
        raise ValueError("Each split may appear only once")
    return {
        "splits": {item["split"]: item["targets"] for item in summaries},
        "total_targets": len(seen),
        "cross_split_overlap": 0,
    }
