#!/usr/bin/env python3
"""Build immutable Avril V3.0 algorithm and web release archives.

The development checkout keeps V1 untouched.  This builder resolves each file
from the V3 release overlay first, the isolated validated source snapshot
second, and the stable project tree last.  The packaged copy can also rebuild
itself using its own root as the sole source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

SCRIPT = Path(__file__).resolve()
DEVELOPMENT_ROOT = next(
    (parent for parent in SCRIPT.parents if (parent / "apps" / "vlora-observatory").is_dir()),
    SCRIPT.parents[1],
)
SNAPSHOT = DEVELOPMENT_ROOT / "experiments/refit-overload-audit-20261029/source_snapshot"
OVERLAY = DEVELOPMENT_ROOT / "experiments/refit-overload-audit-20261029/release_overlay"
PACKAGED = not SNAPSHOT.is_dir()

ROOT_FILES = (
    ".gitignore",
    "README.md",
    "README_zh.md",
    "AVRIL_RELEASE.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "Install Avril.command",
    "Install-Avril.bat",
    "SECURITY.md",
    "Start Avril.command",
    "Start-Avril.bat",
    "Start-Avril.ps1",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "setup_avril.sh",
    "setup_avril.ps1",
    "start_avril.sh",
    "run_avril.command",
    "run_avril_v3_science_batch.py",
    "run_avril_v3_threshold_round.py",
    "run_avril_v3_science_finalize.py",
    "run_vlora_cuda_vote_experiment.py",
    "run_vlora_vote_experiment.py",
    "run_vlora_vote_harmonic_finalize.py",
    "run_vlora_vote_rational_finalize.py",
    "scripts/compare_vlora_cross_platform.py",
    "scripts/build_avril_v3_release.py",
    "scripts/verify_avril_release.py",
)

DIRECTORIES = ("avril", "docs", "requirements")

VLORA_RUNTIME = (
    "vlora/__init__.py",
    "vlora/ffi.py",
    "vlora/io.py",
    "vlora/models.py",
    "vlora/sources.py",
    "vlora/web_server.py",
    "vlora/assets/download_tesscut_segmented.sh",
)

ENGINE_MODULES = (
    "multiplanet_trial/candidate_physical_features_t16_multiview.py",
    "multiplanet_trial/exovetter_features.py",
    "multiplanet_trial/ffi_self_photometry.py",
    "multiplanet_trial/vetting.py",
    "multiplanet_trial/clean_eval/run_blind_batch.py",
    "multiplanet_trial/v03/__init__.py",
    "multiplanet_trial/v03/archive.py",
    "multiplanet_trial/v03/event_search.py",
    "multiplanet_trial/v03/hybrid.py",
    "multiplanet_trial/v03/lightkurve_evidence.py",
    "multiplanet_trial/v03/multibranch_search.py",
    "multiplanet_trial/v03/pixel_vetting.py",
    "multiplanet_trial/v05_eval/__init__.py",
    "multiplanet_trial/v05_eval/dataset_protocol.py",
    "multiplanet_trial/v05_eval/integrity.py",
    "multiplanet_trial/v07_eval/__init__.py",
    "multiplanet_trial/v07_eval/multibranch_search.py",
    "multiplanet_trial/v08_eval/multibranch_search.py",
    "multiplanet_trial/live_web/__init__.py",
    "multiplanet_trial/live_web/candidate_tiers.py",
    "multiplanet_trial/live_web/compiled_event_repeatability.py",
    "multiplanet_trial/live_web/compiled_red_noise_gate.py",
    "multiplanet_trial/live_web/cpp_exact_bls.py",
    "multiplanet_trial/live_web/cuda_astropy_bls.py",
    "multiplanet_trial/live_web/cuda_fixed_ephemeris.py",
    "multiplanet_trial/live_web/cuda_hybrid_search.py",
    "multiplanet_trial/live_web/cuda_phase_null.py",
    "multiplanet_trial/live_web/cuda_science_acceleration.py",
    "multiplanet_trial/live_web/cupy_astropy_bls.py",
    "multiplanet_trial/live_web/detection_significance.py",
    "multiplanet_trial/live_web/ensemble_intersection.py",
    "multiplanet_trial/live_web/event_coherence.py",
    "multiplanet_trial/live_web/event_train_funnel.py",
    "multiplanet_trial/live_web/exact_bls_acceleration.py",
    "multiplanet_trial/live_web/fast_science_finalize.py",
    "multiplanet_trial/live_web/fast_survey_vetting.py",
    "multiplanet_trial/live_web/fixed_ephemeris_acceleration.py",
    "multiplanet_trial/live_web/harmonic_families.py",
    "multiplanet_trial/live_web/meaningful_candidate_gate.py",
    "multiplanet_trial/live_web/optimization_v1.py",
    "multiplanet_trial/live_web/optimization_v3.py",
    "multiplanet_trial/live_web/optimization_v4.py",
    "multiplanet_trial/live_web/pipeline.py",
    "multiplanet_trial/live_web/pixel_consensus.py",
    "multiplanet_trial/live_web/post_discovery_vetting.py",
    "multiplanet_trial/live_web/rational_aliases.py",
    "multiplanet_trial/live_web/result_storage_v1.py",
    "multiplanet_trial/live_web/science_review.py",
    "multiplanet_trial/live_web/sealed_validation.py",
    "multiplanet_trial/live_web/significance_acceleration.py",
    "multiplanet_trial/live_web/temporal_consensus.py",
    "multiplanet_trial/live_web/vlora_release.py",
    "multiplanet_trial/live_web/native/exact_bls_kernel.cpp",
)

ENGINE_TESTS = (
    "multiplanet_trial/live_web/test_ensemble_intersection.py",
    "multiplanet_trial/live_web/test_cuda_hybrid_search.py",
    "multiplanet_trial/live_web/test_cuda_science_acceleration.py",
    "multiplanet_trial/live_web/test_exact_bls_acceleration.py",
    "multiplanet_trial/live_web/test_detection_significance.py",
    "multiplanet_trial/live_web/test_post_discovery_vetting.py",
    "multiplanet_trial/live_web/test_rational_aliases.py",
    "multiplanet_trial/live_web/test_harmonic_families.py",
    "multiplanet_trial/live_web/test_event_coherence.py",
    "multiplanet_trial/live_web/test_science_review.py",
    "multiplanet_trial/live_web/test_pixel_consensus.py",
    "multiplanet_trial/live_web/test_fast_science_finalize.py",
    "multiplanet_trial/live_web/test_meaningful_candidate_gate.py",
)

WEB_ROOT_FILES = (
    ".gitignore",
    "README.md",
    "eslint.config.mjs",
    "index.html",
    "package-lock.json",
    "package.json",
    "tsconfig.json",
    "vite.cloudflare-pages.config.ts",
)

WEB_SOURCE_FILES = (
    "app/globals.css",
    "app/layout.tsx",
    "app/page.tsx",
    "components/observatory/glass-surface.tsx",
    "components/observatory/science-chart.tsx",
    "components/observatory/space-scene.tsx",
    "components/ui/badge.tsx",
    "components/ui/button.tsx",
    "components/ui/input.tsx",
    "components/ui/progress.tsx",
    "components/ui/tabs.tsx",
    "lib/targets.ts",
    "lib/utils.ts",
    "public/CREDITS.md",
    "public/earth-blue-marble.jpg",
    "public/favicon.svg",
    "spa/main.tsx",
    "vendor/shadcn-tailwind-4.13.0.LICENSE.md",
    "vendor/shadcn-tailwind-4.13.0.css",
)


def _source(relative: str) -> Path:
    if PACKAGED:
        candidates = (DEVELOPMENT_ROOT / relative,)
    else:
        candidates = (OVERLAY / relative, SNAPSHOT / relative, DEVELOPMENT_ROOT / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"release source is missing: {relative}")


def _directory_files(relative: str) -> list[tuple[str, Path]]:
    roots = (DEVELOPMENT_ROOT,) if PACKAGED else (DEVELOPMENT_ROOT, SNAPSHOT, OVERLAY)
    resolved: dict[str, Path] = {}
    for root in roots:
        directory = root / relative
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix in {".pyc", ".nbc", ".nbi"}:
                continue
            key = str(Path(relative) / path.relative_to(directory))
            resolved[key] = path
    return sorted(resolved.items())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _algorithm_files() -> list[tuple[str, Path]]:
    resolved = {
        name: _source(name)
        for name in (*ROOT_FILES, *VLORA_RUNTIME, *ENGINE_MODULES, *ENGINE_TESTS)
    }
    for directory in DIRECTORIES:
        resolved.update(dict(_directory_files(directory)))
    return sorted(resolved.items())


def _web_files() -> list[tuple[str, Path]]:
    roots = (
        DEVELOPMENT_ROOT / "apps/vlora-observatory",
        DEVELOPMENT_ROOT.parent
        / "tess_exoplanet_v03_lctools_handoff/apps/vlora-observatory",
        DEVELOPMENT_ROOT.parent / "Avril-V3.0-Web",
    )
    root = next((candidate for candidate in roots if candidate.is_dir()), roots[-1])
    paths = [root / name for name in (*WEB_ROOT_FILES, *WEB_SOURCE_FILES)]
    paths.extend(path for path in (root / "dist-pages").rglob("*") if path.is_file())
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"web release files are missing: {missing}")
    return sorted((str(path.relative_to(root)), path) for path in paths)


def _write_manifest(root: Path, package_kind: str, validation: dict | None) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    payload = {
        "schema_version": "avril-v3-package-manifest",
        "release_id": "avril-v3.0",
        "package_kind": package_kind,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "files": {str(path.relative_to(root)): _sha256(path) for path in files},
    }
    if validation is not None:
        payload["release_validation"] = validation
    (root / "RELEASE_MANIFEST.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _zip_tree(root: Path, archive: Path, prefix: str) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            info = zipfile.ZipInfo.from_file(
                path, arcname=str(Path(prefix) / path.relative_to(root))
            )
            if path.stat().st_mode & stat.S_IXUSR:
                info.external_attr = (0o100755 & 0xFFFF) << 16
            bundle.writestr(
                info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
            )


def _validate_zip(archive: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        names = [PurePosixPath(name) for name in bundle.namelist() if not name.endswith("/")]
        if any(path.is_absolute() or ".." in path.parts for path in names):
            raise ValueError(f"unsafe path in {archive}")
        if len({path.parts[0] for path in names}) != 1:
            raise ValueError(f"archive does not have one root: {archive}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    validation_path = args.validation.expanduser().resolve()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("passed") is not True:
        raise ValueError("V3 release validation did not pass")
    if output.exists():
        raise FileExistsError(f"release directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="avril-v3-release-") as temporary:
        staging = Path(temporary)
        algorithm = staging / "algorithm"
        web = staging / "web"
        for relative, source in _algorithm_files():
            _copy(source, algorithm / relative)
        _copy(validation_path, algorithm / "validation/avril_v3_release_validation.json")
        _write_manifest(algorithm, "algorithm", validation)
        for relative, source in _web_files():
            _copy(source, web / relative)
        _write_manifest(web, "web", None)

        output.mkdir()
        algorithm_archive = output / "Avril-V3.0-Algorithm.zip"
        web_archive = output / "Avril-V3.0-Web.zip"
        _zip_tree(algorithm, algorithm_archive, "Avril-V3.0-Algorithm")
        _zip_tree(web, web_archive, "Avril-V3.0-Web")
        _validate_zip(algorithm_archive)
        _validate_zip(web_archive)

    release = {
        "schema_version": "avril-v3-release-set",
        "release_id": "avril-v3.0",
        "algorithm_archive": algorithm_archive.name,
        "algorithm_archive_sha256": _sha256(algorithm_archive),
        "web_archive": web_archive.name,
        "web_archive_sha256": _sha256(web_archive),
        "validation_sha256": _sha256(validation_path),
        "algorithm_and_web_are_separate": True,
    }
    (output / "SHA256SUMS.json").write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(release, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
