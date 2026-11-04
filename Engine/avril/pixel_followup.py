"""Post-discovery pixel localization for locked Avril candidates.

Pixel evidence is intentionally absent from discovery.  This module runs only
after the red-noise and physical review has selected follow-up candidates, and
it verifies that rank and ephemerides remain byte-for-byte unchanged.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from pathlib import Path

from multiplanet_trial.live_web.pixel_consensus import (
    aggregate_pixel_localizations,
)
from multiplanet_trial.live_web.science_review import (
    rank_science_candidates,
    science_review_annotation,
)
from multiplanet_trial.v03.pixel_vetting import (
    detailed_tpf_localization,
    query_tic_gaia_neighbors,
)
from multiplanet_trial.v05_eval.integrity import atomic_json

Progress = Callable[[str, int, str, dict | None], None]
LOCKED_FIELDS = ("period_days", "epoch_btjd", "duration_hours")


def _candidate_key(candidate: Mapping) -> tuple:
    return (
        candidate.get("candidate_id") or candidate.get("proposal_id"),
        candidate.get("discovery_rank")
        or candidate.get("ensemble_rank")
        or candidate.get("rank"),
    )


def _locked_snapshot(candidates: Iterable[Mapping]) -> list[tuple]:
    return [
        (_candidate_key(candidate), *(candidate.get(field) for field in LOCKED_FIELDS))
        for candidate in candidates
    ]


def _aggregate_status(rows: list[dict]) -> str:
    """Compatibility wrapper around the shared cross-sector policy."""

    return str(aggregate_pixel_localizations(rows)["aggregate_status"])


def localize_locked_candidates(
    candidates: list[dict],
    pixel_paths: Iterable[str | Path],
    *,
    progress: Progress | None = None,
) -> tuple[list[dict], dict]:
    """Attach pixel evidence without deleting, reordering, or retuning rows."""

    original = _locked_snapshot(candidates)
    output = deepcopy(candidates)
    paths = [
        resolved
        for path in pixel_paths
        if not (resolved := Path(path).expanduser().resolve()).name.startswith("._")
        and resolved.is_file()
    ]
    neighbor_catalogs: dict[str, dict] = {}
    diagnostics: list[dict] = []
    for path in paths:
        try:
            neighbor_catalogs[path.name] = query_tic_gaia_neighbors(path)
        except Exception as exc:
            diagnostics.append(
                {
                    "file": path.name,
                    "stage": "tic_gaia_neighbor_query",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    for index, candidate in enumerate(output):
        localizations: list[dict] = []
        for path in paths:
            try:
                localizations.append(
                    detailed_tpf_localization(
                        path,
                        candidate,
                        neighbor_catalog=neighbor_catalogs.get(path.name),
                    )
                )
            except Exception as exc:
                diagnostics.append(
                    {
                        "candidate": _candidate_key(candidate),
                        "file": path.name,
                        "stage": "difference_image_localization",
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        consensus = aggregate_pixel_localizations(localizations)
        status = str(consensus["aggregate_status"])
        candidate["pixel_vetting"] = {
            "aggregate_status": status,
            "cross_sector_consensus": consensus,
            "sectors_requested": len(paths),
            "sectors_available": len(localizations),
            "per_sector": localizations,
            "policy": "diagnostic_only_never_delete_or_retune",
            "candidate_retained": True,
        }
        candidate["pixel_status"] = status
        if candidate.get("science_review") is not None:
            candidate["science_review"] = science_review_annotation(candidate)
        if progress:
            progress(
                "pixel_followup",
                96 + int(3 * (index + 1) / max(1, len(output))),
                f"Localized {index + 1}/{len(output)} locked follow-up candidates",
                {"pixel_files": len(paths)},
            )
    if _locked_snapshot(output) != original:
        raise AssertionError("pixel follow-up changed candidate order or locked ephemerides")
    return output, {
        "schema_version": "avril-v1-pixel-followup",
        "candidate_count": len(output),
        "target_pixel_files": [str(path) for path in paths],
        "candidate_order_preserved": True,
        "locked_ephemerides_verified": True,
        "diagnostics": diagnostics,
    }


def _read_gzip_json(path: Path) -> object:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _write_gzip_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def _merge_selected(all_candidates: list[dict], selected: list[dict]) -> list[dict]:
    by_key = {_candidate_key(candidate): candidate for candidate in selected}
    merged = [deepcopy(by_key.get(_candidate_key(candidate), candidate)) for candidate in all_candidates]
    if _locked_snapshot(merged) != _locked_snapshot(all_candidates):
        raise AssertionError("pixel follow-up merge changed locked candidates")
    return merged


def _priority_counts(candidates: Iterable[Mapping]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        priority = str(
            (candidate.get("science_review") or {}).get("priority") or "unreviewed"
        )
        counts[priority] = counts.get(priority, 0) + 1
    return counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locked_ephemerides_by_key(candidates: Iterable[Mapping]) -> dict[tuple, tuple]:
    return {
        _candidate_key(candidate): tuple(candidate.get(field) for field in LOCKED_FIELDS)
        for candidate in candidates
    }


def reaggregate_saved_pixel_followup(
    saved_result: Path,
    output: Path,
) -> dict:
    """Reapply only cross-sector consensus to saved per-sector evidence.

    Difference images, catalog queries, discovery ranks, and ephemerides are
    reused unchanged.  Results are written to a separate directory so the old
    one-sector-veto output remains available for a byte-level audit.
    """

    saved_result = Path(saved_result).expanduser().resolve()
    source_candidates = saved_result.parent / "science_candidates.json.gz"
    output = Path(output).expanduser().resolve()
    result = json.loads(saved_result.read_text(encoding="utf-8"))
    candidates = list(_read_gzip_json(source_candidates))
    before_locked = _locked_ephemerides_by_key(candidates)
    before_priorities = _priority_counts(candidates)
    reaggregated = 0

    for candidate in candidates:
        pixel = candidate.get("pixel_vetting")
        if not isinstance(pixel, dict) or "per_sector" not in pixel:
            continue
        per_sector = list(pixel.get("per_sector") or [])
        if not per_sector:
            continue
        consensus = aggregate_pixel_localizations(per_sector)
        pixel["aggregate_status"] = str(consensus["aggregate_status"])
        pixel["cross_sector_consensus"] = consensus
        candidate["pixel_status"] = str(consensus["aggregate_status"])
        if candidate.get("science_review") is not None:
            candidate["science_review"] = science_review_annotation(candidate)
        reaggregated += 1

    candidates = rank_science_candidates(candidates)
    if _locked_ephemerides_by_key(candidates) != before_locked:
        raise AssertionError("cross-sector reaggregation changed locked ephemerides")

    by_key = {_candidate_key(candidate): candidate for candidate in candidates}
    compact = []
    for source in result.get("ranked_candidates", []):
        row = deepcopy(source)
        updated = by_key.get(_candidate_key(row))
        if updated is not None:
            row["pixel_vetting"] = deepcopy(updated.get("pixel_vetting"))
            row["pixel_status"] = updated.get("pixel_status")
            row["science_review"] = deepcopy(updated.get("science_review"))
        compact.append(row)
    result["ranked_candidates"] = rank_science_candidates(compact)

    report = {
        "schema_version": "avril-v1-pixel-followup-consensus-v2",
        "status": "complete",
        "mode": "saved_per_sector_evidence_reaggregation_only",
        "source_result": str(saved_result),
        "source_result_sha256": _sha256(saved_result),
        "source_candidates": str(source_candidates),
        "source_candidates_sha256": _sha256(source_candidates),
        "candidate_count": len(candidates),
        "candidates_with_saved_pixel_evidence_reaggregated": reaggregated,
        "priority_counts_before": before_priorities,
        "priority_counts_after": _priority_counts(candidates),
        "candidate_identity_and_ephemerides_verified": True,
        "difference_images_recomputed": False,
        "pixel_files_redownloaded": False,
        "consensus_schema_version": "avril-pixel-cross-sector-consensus-v1",
    }
    science = result.get("science_annotation")
    if isinstance(science, dict):
        science["priority_counts_before_consensus_reaggregation"] = before_priorities
        science["priority_counts"] = _priority_counts(candidates)
        science["pixel_consensus_reaggregation"] = deepcopy(report)
    result["pixel_followup_consensus_reaggregation"] = deepcopy(report)

    output.mkdir(parents=True, exist_ok=True)
    _write_gzip_json(output / "science_candidates.json.gz", candidates)
    atomic_json(output / "result.json", result)
    atomic_json(output / "pixel_followup_report.json", report)
    return report


def _record_post_pixel_summary(result: dict, candidates: list[dict], report: dict) -> None:
    """Make the final pixel-follow-up result summarize its final priorities."""

    science = result.get("science_annotation")
    if isinstance(science, dict):
        before = dict(science.get("priority_counts") or {})
        science["priority_counts_before_pixel_followup"] = before
        science["priority_counts"] = _priority_counts(candidates)
        science["pixel_followup_science_rank_recomputed"] = True
    report["priority_counts_after_pixel_followup"] = _priority_counts(candidates)


def followup_spoc_result(
    final_result: Path,
    source_receipt: Path,
    cache_root: Path,
    *,
    target: str,
    download_workers: int = 80,
    progress: Progress | None = None,
) -> dict:
    """Run pixels only for candidates already marked high priority."""

    result = json.loads(final_result.read_text(encoding="utf-8"))
    full_path = final_result.parent / "science_candidates.json.gz"
    candidates = list(_read_gzip_json(full_path))
    selected = [
        candidate
        for candidate in candidates
        if (candidate.get("science_review") or {}).get("priority")
        == "high_priority_follow_up_candidate"
    ]
    output = final_result.parent.parent / "final-science-pixel-followup"
    output.mkdir(parents=True, exist_ok=True)
    if not selected:
        report = {
            "schema_version": "avril-v1-pixel-followup",
            "status": "not_needed",
            "reason": "no_high_priority_follow_up_candidates",
            "pixel_download_started": False,
        }
        atomic_json(output / "pixel_followup_report.json", report)
        return report

    receipt = json.loads(source_receipt.read_text(encoding="utf-8"))
    segments = sorted(int(row["sequence_number"]) for row in receipt["products"])
    from multiplanet_trial.live_web import pipeline as core

    _inventory, products, pixel_paths, diagnostics = core._prefetch_target_pixels(
        target,
        str(receipt.get("mission") or "TESS"),
        segments,
        deep=True,
        cache_root=cache_root,
        progress=progress,
        download_workers=download_workers,
    )
    localized, report = localize_locked_candidates(selected, pixel_paths, progress=progress)
    merged = rank_science_candidates(_merge_selected(candidates, localized))
    _write_gzip_json(output / "science_candidates.json.gz", merged)
    compact_by_key = {_candidate_key(row): row for row in localized}
    updated_compact = []
    for candidate in result.get("ranked_candidates", []):
        row = deepcopy(candidate)
        localized_row = compact_by_key.get(_candidate_key(row))
        if localized_row is not None:
            row["pixel_vetting"] = deepcopy(localized_row.get("pixel_vetting"))
            row["pixel_status"] = localized_row.get("pixel_status")
            row["science_review"] = deepcopy(localized_row.get("science_review"))
        updated_compact.append(row)
    result["ranked_candidates"] = rank_science_candidates(updated_compact)
    result["pixel_followup"] = {
        **report,
        "status": "complete",
        "selected_products": products,
        "download_diagnostics": diagnostics,
        "source_result_unchanged": str(final_result.resolve()),
    }
    _record_post_pixel_summary(result, merged, result["pixel_followup"])
    atomic_json(output / "result.json", result)
    atomic_json(output / "pixel_followup_report.json", result["pixel_followup"])
    return result["pixel_followup"]


def followup_ffi_result(
    final_result: Path,
    pixel_paths: Iterable[str | Path],
    *,
    progress: Progress | None = None,
) -> dict:
    """Reuse source FFI cutouts after optimized Avril science review."""

    result = json.loads(final_result.read_text(encoding="utf-8"))
    full_path = final_result.parent / "science_candidates.json.gz"
    candidates = list(_read_gzip_json(full_path))
    selected = [
        candidate
        for candidate in candidates
        if (candidate.get("science_review") or {}).get("priority")
        == "high_priority_follow_up_candidate"
    ]
    output = final_result.parent.parent / "final-science-pixel-followup"
    output.mkdir(parents=True, exist_ok=True)
    if not selected:
        report = {
            "schema_version": "avril-v1-pixel-followup",
            "status": "not_needed",
            "reason": "no_high_priority_follow_up_candidates",
            "pixel_download_started": False,
            "ffi_source_cutouts_reused": True,
        }
        atomic_json(output / "pixel_followup_report.json", report)
        return report

    localized, report = localize_locked_candidates(
        selected, pixel_paths, progress=progress
    )
    merged = rank_science_candidates(_merge_selected(candidates, localized))
    _write_gzip_json(output / "science_candidates.json.gz", merged)
    compact_by_key = {_candidate_key(row): row for row in localized}
    updated_compact = []
    for candidate in result.get("ranked_candidates", []):
        row = deepcopy(candidate)
        localized_row = compact_by_key.get(_candidate_key(row))
        if localized_row is not None:
            row["pixel_vetting"] = deepcopy(localized_row.get("pixel_vetting"))
            row["pixel_status"] = localized_row.get("pixel_status")
            row["science_review"] = deepcopy(localized_row.get("science_review"))
        updated_compact.append(row)
    result["ranked_candidates"] = rank_science_candidates(updated_compact)
    result["pixel_followup"] = {
        **report,
        "status": "complete",
        "ffi_source_cutouts_reused": True,
        "additional_pixel_download": False,
        "source_result_unchanged": str(final_result.resolve()),
    }
    _record_post_pixel_summary(result, merged, result["pixel_followup"])
    atomic_json(output / "result.json", result)
    atomic_json(output / "pixel_followup_report.json", result["pixel_followup"])
    return result["pixel_followup"]
