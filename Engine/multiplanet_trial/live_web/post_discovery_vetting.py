"""Non-destructive physical and pixel vetting for locked transit candidates.

This module is deliberately downstream of candidate discovery.  It may attach
measurements and review labels, but it must never remove a candidate, change its
rank, or retune its period, epoch, or duration.
"""

from __future__ import annotations

import os
import multiprocessing
import math
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from multiplanet_trial.candidate_physical_features_t16_multiview import (
    load_pdcsap_physics,
    physical_vetting_features,
)
from multiplanet_trial.exovetter_features import (
    exovetter_diagnostics,
    prepare_exovetter_data,
)
from multiplanet_trial.live_web.fast_survey_vetting import (
    fast_survey_physical_vetting,
)
from multiplanet_trial.live_web.pixel_consensus import (
    ALL_ON_TARGET_STATUS,
    ON_TARGET_CONSENSUS_STATUS,
    aggregate_pixel_localizations,
    is_off_target_consensus,
)
from multiplanet_trial.v03.pixel_vetting import (
    detailed_tpf_localization,
    query_tic_gaia_neighbors,
)


Progress = Callable[[str, int, str, dict | None], None]
_VETTING_PHYSICS_DATA: Mapping[str, object] | None = None
LOCKED_EPHEMERIS_FIELDS = ("period_days", "epoch_btjd", "duration_hours")
PHYSICAL_CHECKS = (
    "odd_even",
    "secondary_eclipse",
    "v_shape",
    "centroid_motion",
    "neighbor_contamination",
)
REFERENCE_CHECKS = (
    "model_shift",
    "transit_phase_coverage",
    "sinusoidal_variability",
)


def _locked_snapshot(candidates: Iterable[Mapping]) -> list[tuple]:
    """Return the discovery identity and immutable ephemeris of every row."""
    return [
        (
            row.get("candidate_id"),
            row.get("rank"),
            *(row.get(field) for field in LOCKED_EPHEMERIS_FIELDS),
        )
        for row in candidates
    ]


def _aggregate_pixel_status(localizations: list[dict]) -> str:
    """Compatibility wrapper around the shared cross-sector policy."""

    return str(aggregate_pixel_localizations(localizations)["aggregate_status"])


def _physical_failure(exc: Exception) -> dict:
    return {
        "status": "failed",
        "available": False,
        "error": f"{type(exc).__name__}: {exc}",
        "vetoes": [],
        "passed": None,
        "policy": "diagnostic_only_never_delete",
    }


def _annotate_physical_candidate(
    index: int,
    source: Mapping,
    physics_data: Mapping[str, object],
) -> tuple[int, dict, dict | None]:
    """Run the unchanged diagnostic set for one locked candidate."""

    candidate = dict(source)
    try:
        use_fast_survey_vetting = os.environ.get(
            "VLORA_SURVEY_PHYSICAL_BACKEND", "compiled"
        ).strip().lower() != "reference"
        physical = (
            fast_survey_physical_vetting(physics_data, candidate)
            if use_fast_survey_vetting
            else physical_vetting_features(physics_data, candidate)
        )
        candidate["features"] = {
            **candidate.get("features", {}),
            **physical["features"],
        }
        evidence = dict(physical["evidence"])
        evidence["policy"] = "diagnostic_only_never_delete"
        candidate["physical_vetting"] = evidence
        candidate["reference_vetting"] = (
            exovetter_diagnostics(physics_data, candidate)
            if evidence.get("passed") is True
            else {
                "schema_version": "vlora-exovetter-reference-v1",
                "available": False,
                "status": "skipped_after_physical_hard_veto",
                "warnings": [],
                "policy": "review_only_never_delete_or_retune",
            }
        )
        return index, candidate, None
    except Exception as exc:
        candidate["physical_vetting"] = _physical_failure(exc)
        candidate["reference_vetting"] = {
            "schema_version": "vlora-exovetter-reference-v1",
            "available": False,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "warnings": [],
            "policy": "review_only_never_delete_or_retune",
        }
        return index, candidate, {
            "candidate_id": candidate.get("candidate_id"),
            "rank": candidate.get("rank", index + 1),
            "stage": "physical_vetting",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _initialize_vetting_worker(physics_data=None) -> None:
    global _VETTING_PHYSICS_DATA
    if physics_data is not None:
        _VETTING_PHYSICS_DATA = physics_data


def _vetting_batch_worker(batch: list[tuple[int, dict]]) -> list[tuple]:
    if _VETTING_PHYSICS_DATA is None:
        raise RuntimeError("Physical-vetting worker has no cached light curve")
    return [
        _annotate_physical_candidate(index, source, _VETTING_PHYSICS_DATA)
        for index, source in batch
    ]


def _review_summary(candidate: Mapping) -> dict:
    """Translate raw diagnostics into labels without making a delete decision."""
    physical = candidate.get("physical_vetting") or {}
    physical_flags = [
        str(name) for name in physical.get("vetoes", []) if str(name) in PHYSICAL_CHECKS
    ]
    pixel = candidate.get("pixel_vetting") or {}
    pixel_status = str(pixel.get("aggregate_status") or "no_pixel_localization")
    risk_flags = [f"physical:{name}" for name in physical_flags]
    reference = candidate.get("reference_vetting") or {}
    reference_warnings = [str(value) for value in reference.get("warnings") or []]
    risk_flags.extend(f"reference:{value}" for value in reference_warnings)
    if is_off_target_consensus(pixel_status):
        risk_flags.append("pixel:possible_off_target")
        review_status = "possible_off_target_or_blend"
        label_zh = "Pixel evidence indicates a possible neighboring or background source"
    elif physical_flags or reference_warnings:
        review_status = "possible_astrophysical_false_positive"
        label_zh = "Physical or reference diagnostics indicate false-positive risk"
    elif pixel_status in {ALL_ON_TARGET_STATUS, ON_TARGET_CONSENSUS_STATUS}:
        review_status = "passed_available_vetting"
        label_zh = "No clear anomaly was identified by the available physical and pixel checks"
    elif pixel_status == "pixel_localization_mixed_or_unresolved":
        review_status = "pixel_evidence_unresolved"
        label_zh = "Pixel-localization evidence is inconsistent or currently inconclusive"
    else:
        review_status = "needs_pixel_localization"
        label_zh = "No target-pixel localization evidence is currently available"
    return {
        "policy": "annotation_only_never_delete_or_retune",
        "review_status": review_status,
        "label_zh": label_zh,
        "risk_flags": risk_flags,
        "physical_flags": physical_flags,
        "reference_warnings": reference_warnings,
        "pixel_status": pixel_status,
        "candidate_retained": True,
        "rank_unchanged": True,
        "ephemeris_unchanged": True,
    }


def annotate_locked_candidates(
    lightcurve_paths: Iterable[str | Path],
    candidates: list[dict],
    *,
    target_pixel_paths: Iterable[str | Path] = (),
    progress: Progress | None = None,
    t16_aperture: int | None = None,
    workers: int | None = None,
    processes: int = 1,
) -> tuple[list[dict], dict]:
    """Attach five physical checks and TPF localization to every candidate.

    Candidate count, order, rank, period, epoch, and duration are invariants.
    Individual unavailable or failed diagnostics are recorded and never cause a
    candidate to disappear.
    """
    original_count = len(candidates)
    original_snapshot = _locked_snapshot(candidates)
    output = [dict(candidate) for candidate in candidates]
    lightcurve_paths = [Path(path) for path in lightcurve_paths]
    target_pixel_paths = [Path(path) for path in target_pixel_paths]
    diagnostics: list[dict] = []

    if progress:
        progress(
            "vetting",
            93,
            f"Computing physical and lightweight reference evidence for {len(output)} locked candidates",
            {
                "title": "Annotate without removing candidates",
                "command": "vet --locked-ephemeris --annotation-only",
                "metrics": {
                    "candidates": len(output),
                    "checks_each": len(PHYSICAL_CHECKS) + len(REFERENCE_CHECKS),
                    "deep_learning_runtime": False,
                },
            },
        )

    physics_data = None
    survey_cadence_seconds = float(
        os.environ.get("VLORA_SURVEY_CADENCE_SECONDS", "120")
    )
    cadence_diagnostics: list[dict] = []
    try:
        if t16_aperture is None:
            physics_data = load_pdcsap_physics(
                lightcurve_paths,
                target_cadence_seconds=survey_cadence_seconds,
                cadence_diagnostics=cadence_diagnostics,
            )
        else:
            physics_data = load_pdcsap_physics(
                lightcurve_paths,
                t16_aperture=int(t16_aperture),
                target_cadence_seconds=survey_cadence_seconds,
                cadence_diagnostics=cadence_diagnostics,
            )
    except Exception as exc:  # An unavailable diagnostic cannot erase candidates.
        diagnostics.append(
            {
                "stage": "physical_data_load",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    diagnostics.append(
        {
            "stage": "cadence_normalization",
            "status": "ok",
            "target_cadence_seconds": survey_cadence_seconds,
            "segments": cadence_diagnostics,
        }
    )
    if physics_data is None:
        for candidate in output:
            candidate["physical_vetting"] = {
                "status": "unavailable",
                "available": False,
                "vetoes": [],
                "passed": None,
                "policy": "diagnostic_only_never_delete",
            }
            candidate["reference_vetting"] = {
                "schema_version": "vlora-exovetter-reference-v1",
                "available": False,
                "status": "physical_data_unavailable",
                "warnings": [],
                "policy": "review_only_never_delete_or_retune",
            }
    else:
        physics_data = dict(physics_data)
        if "time" in physics_data and "flux" in physics_data:
            physics_data["_exovetter_prepared"] = prepare_exovetter_data(
                physics_data
            )
        # These diagnostics repeatedly stream the full multi-sector light
        # curve.  On survey-sized arrays, more than four threads saturate
        # memory bandwidth and make the stage slower (12 workers took 74 s vs
        # 46 s for four workers on the 43-sector TOI-786 regression).  Keep the
        # bandwidth-aware default while still allowing an explicit override.
        requested_workers = workers if workers is not None else min(4, os.cpu_count() or 1)
        worker_count = min(max(1, int(requested_workers)), max(1, len(output)))
        process_count = min(max(1, int(processes)), max(1, len(output)))
        completed = 0
        if process_count > 1:
            indexed = list(enumerate(output))
            batch_size = max(
                1, int(math.ceil(len(indexed) / (process_count * 4)))
            )
            batches = [
                indexed[start : start + batch_size]
                for start in range(0, len(indexed), batch_size)
            ]
            methods = multiprocessing.get_all_start_methods()
            if "fork" in methods:
                # This function is entered after the significance pool has
                # closed, so no active Python worker threads are inherited.
                # Read-only arrays remain copy-on-write shared on Linux.
                context = multiprocessing.get_context("fork")
                global _VETTING_PHYSICS_DATA
                _VETTING_PHYSICS_DATA = physics_data
                initializer = None
                initargs = ()
            else:
                context = multiprocessing.get_context("spawn")
                initializer = _initialize_vetting_worker
                initargs = (physics_data,)
            executor_context = ProcessPoolExecutor(
                max_workers=process_count,
                mp_context=context,
                initializer=initializer,
                initargs=initargs,
            )
            futures_are_batches = True
        else:
            executor_context = ThreadPoolExecutor(max_workers=worker_count)
            batches = list(enumerate(output))
            futures_are_batches = False
        with executor_context as executor:
            if futures_are_batches:
                futures = [executor.submit(_vetting_batch_worker, batch) for batch in batches]
            else:
                futures = [
                    executor.submit(
                        _annotate_physical_candidate,
                        index,
                        candidate,
                        physics_data,
                    )
                    for index, candidate in batches
                ]
            for future in as_completed(futures):
                rows = future.result() if futures_are_batches else [future.result()]
                for index, candidate, diagnostic in rows:
                    output[index] = candidate
                    if diagnostic is not None:
                        diagnostics.append(diagnostic)
                    completed += 1
                if progress:
                    progress(
                        "vetting",
                        min(94, 93 + int(completed == len(output))),
                        f"Completed physical and lightweight reference diagnostics for {completed}/{len(output)} candidates",
                        {
                            "title": "Parallel post-significance candidate review",
                            "command": "vet --locked-ephemeris --annotation-only",
                            "metrics": {
                                "candidates_completed": completed,
                                "candidates_total": len(output),
                                "workers": (
                                    process_count if process_count > 1 else worker_count
                                ),
                                "parallelism": (
                                    "processes" if process_count > 1 else "threads"
                                ),
                                "deep_learning_runtime": False,
                            },
                        },
                    )

    neighbor_catalogs: dict[str, dict] = {}
    for path in target_pixel_paths:
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

    if progress:
        progress(
            "pixel_vetting",
            95,
            (
                f"Localizing the dimming source across {len(target_pixel_paths)} target-pixel segments"
                if target_pixel_paths
                else "No usable target-pixel files were found; candidates remain available and are flagged for localization"
            ),
            {
                "title": "Difference imaging and TIC/Gaia neighbor checks",
                "command": "pixel-vet --all-candidates --annotation-only",
                "metrics": {
                    "candidates": len(output),
                    "target_pixel_segments": len(target_pixel_paths),
                },
            },
        )

    for index, candidate in enumerate(output):
        localizations: list[dict] = []
        for path in target_pixel_paths:
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
                        "candidate_id": candidate.get("candidate_id"),
                        "rank": candidate.get("rank", index + 1),
                        "file": path.name,
                        "stage": "difference_image_localization",
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        consensus = aggregate_pixel_localizations(localizations)
        aggregate_status = str(consensus["aggregate_status"])
        candidate["pixel_vetting"] = {
            "aggregate_status": aggregate_status,
            "cross_sector_consensus": consensus,
            "sectors_requested": len(target_pixel_paths),
            "sectors_available": len(localizations),
            "per_sector": localizations,
            "policy": "diagnostic_only_never_delete",
            "candidate_retained": True,
        }
        candidate["pixel_status"] = aggregate_status
        candidate["vetting_summary"] = _review_summary(candidate)
        # Preserve the old field for consumers, but make its non-destructive
        # semantics explicit and unconditional.
        candidate["hard_rejected"] = False

    if len(output) != original_count:
        raise AssertionError("Post-discovery vetting changed the candidate count")
    if _locked_snapshot(output) != original_snapshot:
        raise AssertionError("Post-discovery vetting changed rank or locked ephemeris")

    physical_flagged = sum(
        bool(row.get("physical_vetting", {}).get("vetoes")) for row in output
    )
    reference_flagged = sum(
        bool(row.get("reference_vetting", {}).get("warnings")) for row in output
    )
    pixel_flagged = sum(
        is_off_target_consensus(row.get("pixel_status"))
        for row in output
    )
    report = {
        "schema_version": "post-discovery-vetting-v1",
        "policy": "all discovery candidates retained; evidence is annotation only",
        "candidate_count_before": original_count,
        "candidate_count_after": len(output),
        "candidate_order_preserved": True,
        "locked_ephemerides_verified": True,
        "physical_checks": list(PHYSICAL_CHECKS),
        "physical_flagged_candidates": physical_flagged,
        "reference_checks": list(REFERENCE_CHECKS),
        "reference_flagged_candidates": reference_flagged,
        "target_pixel_files": [str(path) for path in target_pixel_paths],
        "pixel_flagged_candidates": pixel_flagged,
        "diagnostics": diagnostics,
        "candidate_parallelism": (
            "processes" if int(processes) > 1 else "threads"
        ),
        "candidate_processes": min(
            max(1, int(processes)), max(1, len(output))
        ),
    }
    return output, report


__all__ = [
    "LOCKED_EPHEMERIS_FIELDS",
    "PHYSICAL_CHECKS",
    "REFERENCE_CHECKS",
    "annotate_locked_candidates",
    "_aggregate_pixel_status",
]
