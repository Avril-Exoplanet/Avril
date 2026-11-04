"""Science-review labels and deterministic post-vetting candidate ranking."""

from __future__ import annotations

from collections.abc import Mapping

from .fast_science_finalize import (
    eligible_for_automated_high_priority_event_evidence,
)
from .pixel_consensus import (
    ALL_ON_TARGET_STATUS,
    ON_TARGET_CONSENSUS_STATUS,
    is_off_target_consensus,
)


SCIENCE_PRIORITY_ORDER = {
    "high_priority_follow_up_candidate": 0,
    "manual_review_required": 1,
    "low_priority_or_false_alarm_risk": 2,
}


def science_review_annotation(candidate: Mapping) -> dict:
    """Combine diagnostics into a follow-up priority, never a planet claim."""

    coherence = candidate.get("event_coherence") or {}
    significance = candidate.get("detection_significance") or {}
    physical = candidate.get("physical_vetting") or {}
    reference = candidate.get("reference_vetting") or {}
    pixel = candidate.get("pixel_vetting") or {}
    level = str(coherence.get("level") or "unavailable")
    physical_flags = [str(value) for value in physical.get("vetoes") or []]
    reference_warnings = [str(value) for value in reference.get("warnings") or []]
    physical_complete = physical.get("passed") is True
    reference_status = str(reference.get("status") or "")
    reference_complete = bool(
        reference.get("available") is True
        and reference_status not in {
            "failed",
            "physical_data_unavailable",
            "skipped_after_event_coherence_gate",
            "skipped_after_physical_hard_veto",
        }
    )
    harmonic_warnings = list(significance.get("harmonic_warnings") or [])
    gap_diagnostics = significance.get("event_gap_diagnostics") or {}
    gap_dominated = bool(
        gap_diagnostics.get("gap_dominated_weak_remainder")
        or significance.get("status") == "gap_adjacent_single_event_dominated"
    )
    direct_harmonic_alias = candidate.get("direct_harmonic_alias") or {}
    block_metrics = list(significance.get("block_metrics") or [])
    strong_blocks = sum(
        float(row.get("mes_like") or 0.0) >= 3.0 for row in block_metrics
    )
    block_consistent = len(block_metrics) < 2 or strong_blocks >= 2
    pixel_status = str(
        pixel.get("aggregate_status") or "no_pixel_localization"
    )
    pixel_off_target = is_off_target_consensus(pixel_status)
    pixel_on_target = pixel_status in {
        ALL_ON_TARGET_STATUS,
        ON_TARGET_CONSENSUS_STATUS,
    }
    pixel_attempted = int(
        pixel.get("sectors_requested") or pixel.get("sector_count") or 0
    ) > 0
    pixel_unresolved_after_followup = (
        pixel_attempted and not pixel_on_target and not pixel_off_target
    )
    risk_flags: list[str] = []
    if level in {"incoherent_event_train", "unavailable"}:
        risk_flags.append(f"event_coherence:{level}")
    if significance.get("passes_7p1_reference") is not True:
        risk_flags.append(
            f"red_noise:{significance.get('status') or 'not_computed'}"
        )
    elif not block_consistent:
        risk_flags.append("red_noise:single_time_block_dominated")
    if not physical_complete:
        risk_flags.append(
            f"physical:{physical.get('status') or 'not_completed'}"
        )
    if not reference_complete:
        risk_flags.append(
            f"reference:{reference_status or 'not_completed'}"
        )
    risk_flags.extend(f"physical:{value}" for value in physical_flags)
    risk_flags.extend(f"reference:{value}" for value in reference_warnings)
    if harmonic_warnings:
        risk_flags.append("harmonic:non_independent_event_family")
    if gap_dominated:
        risk_flags.append("event_integrity:gap_adjacent_single_event_dominated")
    if direct_harmonic_alias:
        risk_flags.append("harmonic:direct_alias_of_significant_winner")
    if not pixel_on_target:
        risk_flags.append(f"pixel:{pixel_status}")

    coherent = level in {
        "repeated_coherent_event_train",
        "sparse_coherent_event_train",
    }
    automated_event_evidence = (
        eligible_for_automated_high_priority_event_evidence(candidate)
    )
    if coherent and not automated_event_evidence:
        risk_flags.append(
            "event_coherence:insufficient_events_for_automatic_high_priority"
        )
    if (
        coherent
        and automated_event_evidence
        and significance.get("passes_7p1_reference") is True
        and block_consistent
        and physical_complete
        and reference_complete
        and not physical_flags
        and not reference_warnings
        and not harmonic_warnings
        and not gap_dominated
        and not direct_harmonic_alias
        and not pixel_off_target
        and not pixel_unresolved_after_followup
    ):
        priority = "high_priority_follow_up_candidate"
        explanation = (
            "Repeated event evidence and red-noise significance passed; still "
            "requires independent astrophysical and pixel-source confirmation."
        )
    elif (
        level == "incoherent_event_train"
        or physical_flags
        or reference_warnings
        or harmonic_warnings
        or gap_dominated
        or direct_harmonic_alias
        or pixel_off_target
        or significance.get("status") == "below_7p1_mes_like_reference"
        or (
            significance.get("passes_7p1_reference") is True
            and not block_consistent
        )
    ):
        priority = "low_priority_or_false_alarm_risk"
        explanation = (
            "At least one diagnostic raises material false-alarm or false-positive risk."
        )
    else:
        priority = "manual_review_required"
        explanation = "The available evidence is incomplete or mixed."

    return {
        "schema_version": "avril-science-review-v2",
        "priority": priority,
        "explanation": explanation,
        "risk_flags": risk_flags,
        "pixel_status": pixel_status,
        "candidate_retained": True,
        "discovery_rank_preserved": True,
        "used_for_science_ranking": True,
        "not_planet_probability": True,
        "confirmation_required": True,
    }


def rank_science_candidates(candidates: list[dict]) -> list[dict]:
    """Order locked candidates by review tier, then original discovery rank.

    Every candidate and ephemeris remains present.  The separate discovery
    rank makes the search output auditable after the final science list moves
    failed diagnostics below candidates that passed the automated review.
    """

    prepared: list[dict] = []
    for fallback_rank, candidate in enumerate(candidates, start=1):
        discovery_rank = candidate.get("discovery_rank")
        if discovery_rank is None:
            discovery_rank = candidate.get("ensemble_rank")
        if discovery_rank is None:
            discovery_rank = candidate.get("rank")
        if discovery_rank is None:
            discovery_rank = fallback_rank
        candidate["discovery_rank"] = int(discovery_rank)
        prepared.append(candidate)

    def order_key(candidate: Mapping) -> tuple[int, int, str]:
        review = candidate.get("science_review") or {}
        priority = str(review.get("priority") or "manual_review_required")
        return (
            SCIENCE_PRIORITY_ORDER.get(priority, 1),
            int(candidate["discovery_rank"]),
            str(candidate.get("candidate_id") or candidate.get("proposal_id") or ""),
        )

    ranked = sorted(prepared, key=order_key)
    for science_rank, candidate in enumerate(ranked, start=1):
        candidate["science_rank"] = science_rank
        candidate["rank"] = science_rank
        review = candidate.get("science_review")
        if isinstance(review, dict):
            review["discovery_rank"] = int(candidate["discovery_rank"])
            review["science_rank"] = science_rank
            review["ranking_tier"] = SCIENCE_PRIORITY_ORDER.get(
                str(review.get("priority") or "manual_review_required"), 1
            )
    return ranked


__all__ = [
    "SCIENCE_PRIORITY_ORDER",
    "rank_science_candidates",
    "science_review_annotation",
]
