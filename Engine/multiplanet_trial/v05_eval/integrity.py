"""Small integrity primitives shared by the evaluation tools."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Mapping) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def verify_file_manifest(manifest_path: str | Path, project_root: str | Path) -> list[dict]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    root = Path(project_root)
    failures = []
    for record in manifest.get("files", []):
        path = root / record["path"]
        actual = sha256_file(path) if path.is_file() else None
        if actual != record["sha256"]:
            failures.append(
                {
                    "path": record["path"],
                    "expected_sha256": record["sha256"],
                    "actual_sha256": actual,
                }
            )
    return failures
