"""v0.8 search extensions built on the frozen v0.7-alpha8 engine.

The v0.7 files are intentionally not edited.  This module re-exports that
engine and adds a truth-blind consensus-residual pass for multi-planet systems.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from multiplanet_trial.v07_eval.multibranch_search import *  # noqa: F401,F403


_load_spoc_independent_branches_frozen_v07 = load_spoc_independent_branches


def _robustly_bin_branch(
    branch: SearchBranch,
    max_points_per_segment: int,
) -> tuple[SearchBranch, list[dict]]:
    """Reduce very high cadence live data without changing the frozen engine.

    Binning happens after each photometry branch has been independently
    corrected and flattened.  Medians preserve transit depth better than
    simply throwing away five out of every six 20-second cadences.
    """
    times: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    segments: list[np.ndarray] = []
    diagnostics: list[dict] = []
    for segment in np.unique(branch.segment_id):
        mask = branch.segment_id == segment
        local_time = branch.time[mask]
        local_flux = branch.flux[mask]
        input_count = int(local_time.size)
        factor = max(1, int(np.ceil(input_count / max_points_per_segment)))
        if factor == 1:
            binned_time, binned_flux = local_time, local_flux
        else:
            usable = (input_count // factor) * factor
            binned_time = np.nanmedian(local_time[:usable].reshape(-1, factor), axis=1)
            binned_flux = np.nanmedian(local_flux[:usable].reshape(-1, factor), axis=1)
            if usable < input_count:
                binned_time = np.append(binned_time, np.nanmedian(local_time[usable:]))
                binned_flux = np.append(binned_flux, np.nanmedian(local_flux[usable:]))
        good = np.isfinite(binned_time) & np.isfinite(binned_flux)
        binned_time, binned_flux = binned_time[good], binned_flux[good]
        times.append(binned_time)
        fluxes.append(binned_flux)
        segments.append(np.full(binned_time.size, int(segment), dtype=int))
        diagnostics.append({
            "branch": branch.key,
            "segment": int(segment),
            "status": "live_cadence_binned" if factor > 1 else "live_cadence_kept",
            "input_cadences": input_count,
            "output_cadences": int(binned_time.size),
            "bin_factor": factor,
        })
    time = np.concatenate(times)
    flux = np.concatenate(fluxes)
    segment_id = np.concatenate(segments)
    order = np.argsort(time)
    return SearchBranch(
        key=branch.key,
        label=branch.label,
        processing=branch.processing + f"; robust live bin cap {max_points_per_segment}/segment",
        photometry_family=branch.photometry_family,
        method_family=branch.method_family,
        time=time[order],
        flux=flux[order],
        segment_id=segment_id[order],
        can_seed_candidates=branch.can_seed_candidates,
        metadata={**dict(branch.metadata), "live_max_points_per_segment": max_points_per_segment},
    ), diagnostics


def load_spoc_independent_branches(
    paths,
    *,
    flatten_window_days: float = 1.0,
    include_cbv: bool = False,
    cbv_dir=None,
    max_points_per_segment: int | None = None,
):
    """Opt-in live-cadence wrapper; defaults reproduce frozen v0.7 exactly."""
    branches, diagnostics = _load_spoc_independent_branches_frozen_v07(
        paths,
        flatten_window_days=flatten_window_days,
        include_cbv=include_cbv,
        cbv_dir=cbv_dir,
    )
    if max_points_per_segment is None:
        return branches, diagnostics
    if max_points_per_segment < 1000:
        raise ValueError("max_points_per_segment must be at least 1000")
    reduced = {}
    for key, branch in branches.items():
        reduced[key], binning_diagnostics = _robustly_bin_branch(
            branch,
            max_points_per_segment,
        )
        diagnostics.extend(binning_diagnostics)
    return reduced, diagnostics


def select_consensus_mask_ephemerides(
    clusters: Iterable[Mapping],
    *,
    max_masks: int | None = 3,
    minimum_supporting_branches: int = 2,
    minimum_snr: float = 7.0,
) -> list[dict]:
    """Select a few strong, truth-blind signals to remove before a second search.

    The first-pass candidates from several photometry branches are correlated,
    so support is not treated as independent statistical proof.  It is only a
    conservative gate: the residual pass is allowed to mask a signal when at
    least two processing branches found it and one member has useful S/N.
    """
    if max_masks is not None and max_masks < 0:
        raise ValueError("max_masks cannot be negative")
    selected: list[dict] = []
    ordered = sorted(
        (dict(cluster) for cluster in clusters),
        key=lambda row: float(row.get("uncalibrated_priority_score") or 0.0),
        reverse=True,
    )
    for cluster in ordered:
        branches = set(cluster.get("supporting_branches", []))
        if len(branches) < minimum_supporting_branches:
            continue
        members = list(cluster.get("members", []))
        strongest_snr = max(
            (
                float(member.get("search_metrics", {}).get("snr") or 0.0)
                for member in members
            ),
            default=0.0,
        )
        if strongest_snr < minimum_snr:
            continue
        period = float(cluster["period_days"])
        # Near-identical periods are one mask.  Integer-related periods stay
        # separate because an eclipsing binary at P/2 is not interchangeable
        # with a planet hypothesis at P.
        if any(
            abs(period - float(old["period_days"])) / float(old["period_days"])
            < 0.012
            for old in selected
        ):
            continue
        selected.append(
            {
                "period_days": period,
                "epoch_btjd": float(cluster["epoch_btjd"]),
                "duration_hours": float(cluster["duration_hours"]),
                "supporting_branches": sorted(branches),
                "strongest_search_snr": strongest_snr,
            }
        )
        if max_masks is not None and len(selected) >= max_masks:
            break
    return selected


def mask_consensus_ephemerides(
    branch: SearchBranch,
    ephemerides: Iterable[Mapping],
    *,
    width_in_durations: float = 1.25,
) -> tuple[SearchBranch, dict]:
    """Return a branch with strong first-pass transits removed.

    Only search cadences are ultimately used by BLS, but masking the complete
    branch keeps its arrays aligned and lets the ordinary holdout policy remain
    unchanged.  The branch key is deliberately preserved so fusion cannot
    mistake the residual view for an independent instrument or extra vote.
    """
    if width_in_durations <= 0:
        raise ValueError("width_in_durations must be positive")
    rows = [dict(row) for row in ephemerides]
    keep = np.ones(branch.time.size, dtype=bool)
    per_mask = []
    for row in rows:
        period = float(row["period_days"])
        epoch = float(row["epoch_btjd"])
        duration_days = float(row["duration_hours"]) / 24.0
        phase = (branch.time - epoch + 0.5 * period) % period - 0.5 * period
        remove = np.abs(phase) <= max(
            0.5 * width_in_durations * duration_days,
            0.02,
        )
        newly_removed = remove & keep
        keep &= ~remove
        per_mask.append(
            {
                "period_days": period,
                "epoch_btjd": epoch,
                "duration_hours": float(row["duration_hours"]),
                "newly_removed_cadences": int(newly_removed.sum()),
            }
        )
    residual = SearchBranch(
        key=branch.key,
        label=branch.label + "; consensus residual pass",
        processing=branch.processing + "; strong consensus transit windows removed",
        photometry_family=branch.photometry_family,
        method_family="bls_residual",
        time=branch.time[keep],
        flux=branch.flux[keep],
        segment_id=branch.segment_id[keep],
        can_seed_candidates=branch.can_seed_candidates,
        metadata={
            **dict(branch.metadata),
            "conditioned_on_consensus_candidates": True,
            "consensus_masks": per_mask,
        },
    )
    return residual, {
        "branch": branch.key,
        "status": "ready" if rows else "no_consensus_masks",
        "input_cadences": int(branch.time.size),
        "remaining_cadences": int(residual.time.size),
        "removed_cadences": int((~keep).sum()),
        "masks": per_mask,
    }


def _cluster_period_epoch_match(
    first: Mapping,
    second: Mapping,
    *,
    period_tolerance_fraction: float = 0.01,
) -> bool:
    first_period = float(first["period_days"])
    second_period = float(second["period_days"])
    period = 0.5 * (first_period + second_period)
    if abs(first_period - second_period) / min(first_period, second_period) > period_tolerance_fraction:
        return False
    phase = abs(
        (
            float(first["epoch_btjd"])
            - float(second["epoch_btjd"])
            + 0.5 * period
        )
        % period
        - 0.5 * period
    )
    duration_days = max(
        float(first["duration_hours"]),
        float(second["duration_hours"]),
    ) / 24.0
    return phase <= min(0.15 * period, max(2.0 * duration_days, 0.02 * period))


def merge_consensus_residual_candidates(
    initial_clusters: Iterable[Mapping],
    residual_candidates: Iterable[Mapping],
    *,
    minimum_new_branch_support: int = 2,
    max_new_clusters: int | None = 8,
) -> tuple[list[dict], dict]:
    """Append only corroborated *new* residual periods without moving old ones.

    A conditioned residual search is valuable for additional planets, but it
    must not rewrite periods or reorder evidence already found by the ordinary
    engine.  Matching residual signals are recorded as diagnostics.  A new
    periodic cluster enters the pool only when two original processing
    branches independently recover it after the same consensus masks.
    """
    original = [dict(row) for row in initial_clusters]
    initial_count = len(original)
    residual = fuse_branch_candidates(residual_candidates)
    if max_new_clusters is not None and max_new_clusters < 0:
        raise ValueError("max_new_clusters cannot be negative")
    appended = 0
    corroborated_existing = 0
    rejected_single_branch = 0
    rejected_by_output_cap = 0
    for candidate in residual:
        matching = next(
            (
                row
                for row in original
                if _cluster_period_epoch_match(row, candidate)
            ),
            None,
        )
        support = sorted(set(candidate.get("supporting_branches", [])))
        evidence = {
            "period_days": float(candidate["period_days"]),
            "epoch_btjd": float(candidate["epoch_btjd"]),
            "duration_hours": float(candidate["duration_hours"]),
            "supporting_branches": support,
            "uncalibrated_priority_score": float(
                candidate.get("uncalibrated_priority_score") or 0.0
            ),
        }
        if matching is not None:
            matching.setdefault("consensus_residual_support", []).append(evidence)
            corroborated_existing += 1
            continue
        if len(support) < minimum_new_branch_support:
            rejected_single_branch += 1
            continue
        if max_new_clusters is not None and appended >= max_new_clusters:
            rejected_by_output_cap += 1
            continue
        item = dict(candidate)
        item["candidate_source"] = "new_consensus_residual_period"
        item["conditioned_candidate"] = True
        original.append(item)
        appended += 1
    original.sort(
        key=lambda row: float(row.get("uncalibrated_priority_score") or 0.0),
        reverse=True,
    )
    for rank, row in enumerate(original, start=1):
        row["rank"] = rank
    return original, {
        "initial_clusters": initial_count,
        "residual_clusters": len(residual),
        "corroborated_existing_clusters": corroborated_existing,
        "new_multibranch_clusters_appended": appended,
        "new_single_branch_clusters_kept_diagnostic_only": rejected_single_branch,
        "new_multibranch_clusters_kept_diagnostic_only_by_cap": rejected_by_output_cap,
        "minimum_new_branch_support": minimum_new_branch_support,
        "max_new_clusters": max_new_clusters,
        "non_destructive_period_policy": True,
    }
