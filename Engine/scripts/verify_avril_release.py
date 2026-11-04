#!/usr/bin/env python3
"""Verify every file recorded in an Avril release ZIP manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def verify_archive(archive: Path) -> dict:
    archive = archive.expanduser().resolve()
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if not name.endswith("/")]
        if not names:
            raise ValueError("archive contains no files")
        paths = [PurePosixPath(name) for name in names]
        if any(path.is_absolute() or ".." in path.parts for path in paths):
            raise ValueError("archive contains an unsafe path")
        roots = {path.parts[0] for path in paths if path.parts}
        if len(roots) != 1:
            raise ValueError(f"archive must have one root directory, found {sorted(roots)}")
        root = roots.pop()
        manifest_name = f"{root}/RELEASE_MANIFEST.json"
        if manifest_name not in names:
            raise ValueError("archive has no RELEASE_MANIFEST.json")
        manifest = json.loads(bundle.read(manifest_name))
        expected = manifest.get("files")
        if not isinstance(expected, dict) or not expected:
            raise ValueError("release manifest has no file hashes")
        actual_names = {
            str(PurePosixPath(name).relative_to(root))
            for name in names
            if name != manifest_name
        }
        expected_names = set(expected)
        if actual_names != expected_names:
            raise ValueError(
                "archive contents differ from manifest: "
                f"missing={sorted(expected_names - actual_names)}, "
                f"extra={sorted(actual_names - expected_names)}"
            )
        mismatches = []
        for relative, expected_digest in sorted(expected.items()):
            digest = hashlib.sha256(bundle.read(f"{root}/{relative}")).hexdigest()
            if digest != expected_digest:
                mismatches.append(relative)
        if mismatches:
            raise ValueError(f"SHA-256 mismatch: {mismatches}")
    return {
        "archive": str(archive),
        "release_id": manifest.get("release_id"),
        "package_kind": manifest.get("package_kind"),
        "files_verified": len(expected),
        "status": "verified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_archive(args.archive), indent=2))


if __name__ == "__main__":
    main()
