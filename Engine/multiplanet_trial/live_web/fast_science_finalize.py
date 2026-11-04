"""Exact, non-destructive short-circuiting for the Avril survey finalizer.

Every discovery candidate remains in the audit ledger.  Expensive phase-null
trials run only while a candidate can still satisfy every earlier, deterministic
science gate.  This changes no positive decision: a candidate that already
failed event coherence, a physical hard veto, or reference vetting cannot later
become a high-priority follow-up candidate merely by running a phase null.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .detection_significance import REFERENCE_MES, _harmonic_warnings
from .pixel_consensus import is_off_target_consensus

COHERENT_LEVELS = {
    "repeated_coherent_event_train",
    "sparse_coherent_event_train",
}
MIN_SUPPORTED_EVENTS_FOR_AUTOMATED_HIGH_PRIORITY = 4


def candidate_key(candidate: Mapping) -> str:
    value = candidate.get("candidate_id") or candidate.get("proposal_id")
    if value is None:
        raise ValueError("candidate is missing candidate_id/proposal_id")
    return str(value)


def eligible_for_physical_vetting(candidate: Mapping) -> bool:
    coherence = candidate.get("event_coherence") or {}
    return str(coherence.get("level") or "") in COHERENT_LEVELS


def eligible_for_automated_high_priority_event_evidence(
    candidate: Mapping,
) -> bool:
    coherence = candidate.get("event_coherence") or {}
    level = str(coherence.get("level") or "")
    if level == "repeated_coherent_event_train":
        return True
    if level != "sparse_coherent_event_train":
        return False
    repeatability = (
        candidate.get("holdout_verification", {})
        .get("event_repeatability", {})
    )
    return int(repeatability.get("supported_events") or 0) >= (
        MIN_SUPPORTED_EVENTS_FOR_AUTOMATED_HIGH_PRIORITY
    )


def eligible_for_phase_null(candidate: Mapping) -> bool:
    if not eligible_for_physical_vetting(candidate):
        return False
    if not eligible_for_automated_high_priority_event_evidence(candidate):
        return False
    physical = candidate.get("physical_vetting") or {}
    reference = candidate.get("reference_vetting") or {}
    pixel = candidate.get("pixel_vetting") or {}
    reference_status = str(reference.get("status") or "")
    return bool(
        physical.get("passed") is True
        and not physical.get("vetoes")
        and reference.get("available") is True
        and reference_status
        not in {
            "failed",
            "physical_data_unavailable",
            "skipped_after_event_coherence_gate",
            "skipped_after_physical_hard_veto",
        }
        and not reference.get("warnings")
        and not is_off_target_consensus(pixel.get("aggregate_status"))
    )


def skipped_vetting(candidate: Mapping, *, reason: str | None = None) -> dict:
    """Return audit placeholders for a deterministically short-circuited row."""

    output = dict(candidate)
    reason = reason or "skipped_after_event_coherence_gate"
    output["physical_vetting"] = {
        "status": reason,
        "available": False,
        "vetoes": [],
        "passed": None,
        "policy": "deterministic_short_circuit_candidate_retained",
    }
    output["reference_vetting"] = {
        "schema_version": "vlora-exovetter-reference-v1",
        "available": False,
        "status": reason,
        "warnings": [],
        "policy": "deterministic_short_circuit_candidate_retained",
    }
    output["pixel_vetting"] = {
        "aggregate_status": "no_pixel_localization",
        "sectors_requested": 0,
        "sectors_available": 0,
        "per_sector": [],
        "policy": "diagnostic_only_never_delete",
        "candidate_retained": True,
    }
    output["pixel_status"] = "no_pixel_localization"
    output["hard_rejected"] = False
    return output


def phase_null_skip_reason(candidate: Mapping) -> str:
    if not eligible_for_physical_vetting(candidate):
        return "skipped_after_event_coherence_gate"
    if not eligible_for_automated_high_priority_event_evidence(candidate):
        return "skipped_insufficient_events_for_automatic_high_priority"
    physical = candidate.get("physical_vetting") or {}
    if physical.get("passed") is not True or physical.get("vetoes"):
        return "skipped_after_physical_hard_veto"
    reference = candidate.get("reference_vetting") or {}
    if reference.get("available") is not True:
        return "skipped_after_reference_vetting_unavailable"
    if reference.get("warnings"):
        return "skipped_after_reference_vetting_warning"
    pixel = candidate.get("pixel_vetting") or {}
    if is_off_target_consensus(pixel.get("aggregate_status")):
        return "skipped_after_off_target_pixel_veto"
    return "not_eligible_for_phase_null"


def skipped_significance(
    candidate: Mapping,
    *,
    requested_trials: int,
    reason: str | None = None,
) -> dict:
    reason = reason or phase_null_skip_reason(candidate)
    return {
        "schema_version": "locked-red-noise-significance-v2",
        "official_spoc_mes": False,
        "ephemeris_locked": True,
        "used_to_retune_or_delete": False,
        "period_days": float(candidate["period_days"]),
        "epoch_btjd": float(candidate["epoch_btjd"]),
        "duration_hours": float(candidate["duration_hours"]),
        "reference_mes": REFERENCE_MES,
        "duration_noise_models": [],
        "events": [],
        "event_count": 0,
        "block_metrics": [],
        "phase_shift_null": {
            "requested_trials": int(requested_trials),
            "executed_trials": 0,
            "valid_trials": 0,
            "short_circuit_reason": reason,
        },
        "passes_7p1_reference": False,
        "harmonic_warnings": [],
        "status": reason,
        "flags": [reason],
        "candidate_retained": True,
        "short_circuit_preserves_positive_decisions": True,
    }


def direct_alias_winners(
    candidates: Iterable[Mapping], relations: Iterable[Mapping]
) -> dict[str, set[str]]:
    """Return direct *integer* alias edges whose endpoints remain viable.

    The old family-level loss count is insufficient: a real planet can lose a
    pairwise comparison to a noisy winner that later fails event coherence or
    physical/reference vetting.  Only a viable direct winner may defer a loser,
    and the finalizer still runs the loser if that winner fails exact phase-null
    significance.  Low-order rational relations remain useful audit metadata,
    but never short-circuit science validation: real multi-planet systems can
    occupy resonances, and sparse shared events do not prove that one rational
    period is merely an alias of the other.
    """

    viable = {
        candidate_key(row)
        for row in candidates
        if eligible_for_phase_null(row)
    }
    output: dict[str, set[str]] = {}
    for relation in relations:
        if relation.get("resolved") is not True:
            continue
        if str(relation.get("relation_type") or "") != "integer_harmonic":
            continue
        winner = str(relation.get("winner_proposal_id") or "")
        loser = str(relation.get("loser_proposal_id") or "")
        if winner in viable and loser in viable and winner != loser:
            output.setdefault(loser, set()).add(winner)
    return output


def merge_candidates(
    originals: list[dict], annotated: Iterable[Mapping]
) -> list[dict]:
    """Merge annotations by stable identity while preserving ledger order."""

    rows = list(annotated)
    by_key = {candidate_key(row): dict(row) for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("annotated candidate identities are not unique")
    output = []
    for original in originals:
        key = candidate_key(original)
        output.append(by_key.get(key, dict(original)))
    return output


def refresh_harmonic_decision(candidate: dict, all_candidates: list[dict]) -> None:
    """Re-evaluate harmonic status against the full retained candidate ledger."""

    significance = candidate.get("detection_significance") or {}
    warnings = _harmonic_warnings(candidate, all_candidates)
    significance["harmonic_warnings"] = warnings
    flags = [
        str(value)
        for value in significance.get("flags") or []
        if str(value) != "integer_harmonic_of_retained_candidate"
    ]
    observed = (significance.get("combined") or {}).get("mes_like")
    calibrated = significance.get("null_calibrated_mes_like")
    global_fap = significance.get("look_elsewhere_fap_estimate")
    observed_passes = bool(
        observed is not None and float(observed) >= REFERENCE_MES
    )
    if warnings:
        flags.append("integer_harmonic_of_retained_candidate")
        status = "integer_harmonic_not_independent"
    elif calibrated is None or observed is None:
        status = "insufficient_statistics"
    elif not observed_passes:
        status = "below_7p1_mes_like_reference"
    elif global_fap is None or float(global_fap) > 0.01:
        status = "phase_shift_false_alarm_not_tightly_constrained"
    else:
        status = "meets_7p1_mes_like_and_phase_null_reference"
    significance["flags"] = flags
    significance["status"] = status
    significance["passes_7p1_reference"] = bool(
        observed_passes
        and calibrated is not None
        and global_fap is not None
        and float(global_fap) <= 0.01
        and not warnings
    )
    candidate["detection_significance"] = significance


__all__ = [
    "COHERENT_LEVELS",
    "candidate_key",
    "direct_alias_winners",
    "eligible_for_automated_high_priority_event_evidence",
    "eligible_for_phase_null",
    "eligible_for_physical_vetting",
    "MIN_SUPPORTED_EVENTS_FOR_AUTOMATED_HIGH_PRIORITY",
    "merge_candidates",
    "phase_null_skip_reason",
    "refresh_harmonic_decision",
    "skipped_significance",
    "skipped_vetting",
]
