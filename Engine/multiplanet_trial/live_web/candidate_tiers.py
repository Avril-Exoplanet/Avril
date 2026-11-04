"""Deterministic public tiers for locked, fully annotated candidates.

Discovery remains append-only.  This module does not erase or retune any
signal; it separates the complete audit pool from the smaller set that has
passed every available public-candidate gate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .pixel_consensus import (
    ALL_ON_TARGET_STATUS,
    ON_TARGET_CONSENSUS_STATUS,
    is_off_target_consensus,
)


def candidate_tier(candidate: Mapping) -> dict:
    significance = candidate.get("detection_significance") or {}
    physical = candidate.get("physical_vetting") or {}
    reference = candidate.get("reference_vetting") or {}
    pixel = candidate.get("pixel_vetting") or {}

    harmonic_warnings = list(significance.get("harmonic_warnings") or [])
    physical_vetoes = [str(value) for value in physical.get("vetoes") or []]
    pixel_status = str(
        pixel.get("aggregate_status")
        or candidate.get("pixel_status")
        or "no_pixel_localization"
    )
    significance_status = str(significance.get("status") or "not_computed")
    block_metrics = list(significance.get("block_metrics") or [])
    strong_blocks = sum(
        float(row.get("mes_like") or 0.0) >= 3.0 for row in block_metrics
    )
    reference_warnings = [str(value) for value in reference.get("warnings") or []]
    repeatability = (
        candidate.get("holdout_verification", {}).get("event_repeatability", {})
    )
    repeatability_fraction = repeatability.get("support_fraction")
    repeatability_events = int(repeatability.get("testable_events") or 0)

    rejection_reasons: list[str] = []
    review_reasons: list[str] = []
    passed_checks: list[str] = ["three_fold_fixed_ephemeris"]

    if repeatability_fraction is None or repeatability_events < 2:
        review_reasons.append("event_repeatability_insufficient")
    elif float(repeatability_fraction) < 0.5:
        rejection_reasons.append("event_repeatability_below_half")
    elif float(repeatability_fraction) < 0.6:
        review_reasons.append("event_repeatability_marginal")
    else:
        passed_checks.append("event_train_repeatability")

    if harmonic_warnings or significance_status == "integer_harmonic_not_independent":
        rejection_reasons.append("integer_harmonic_not_independent")
    elif significance.get("passes_7p1_reference") is True:
        passed_checks.append("red_noise_mes_like")
        if len(block_metrics) < 2:
            review_reasons.append("red_noise_time_blocks_insufficient")
        elif strong_blocks < 2:
            review_reasons.append("red_noise_single_time_block_dominated")
        else:
            passed_checks.append("multi_time_block_red_noise_support")
    elif significance_status == "insufficient_statistics":
        review_reasons.append("red_noise_statistics_insufficient")
    else:
        rejection_reasons.append("red_noise_significance_below_reference")

    if physical_vetoes:
        rejection_reasons.extend(f"physical:{name}" for name in physical_vetoes)
    elif physical.get("passed") is True:
        passed_checks.append("physical_false_positive_checks")
    elif physical.get("passed") is False:
        rejection_reasons.append("physical_false_positive_checks_failed")
    else:
        review_reasons.append("physical_false_positive_checks_unavailable")

    if reference_warnings:
        review_reasons.extend(
            f"reference:{warning}" for warning in reference_warnings
        )
    elif reference.get("available") is True:
        passed_checks.append("lightweight_reference_vetting")
    else:
        review_reasons.append("lightweight_reference_vetting_unavailable")

    if pixel_status in {ALL_ON_TARGET_STATUS, ON_TARGET_CONSENSUS_STATUS}:
        passed_checks.append("pixel_source_localization")
    elif is_off_target_consensus(pixel_status):
        rejection_reasons.append("pixel_source_possible_off_target")
    else:
        review_reasons.append(f"pixel:{pixel_status}")

    if rejection_reasons:
        tier = "rejected_signal"
        public_candidate = False
    elif review_reasons:
        tier = "review_required"
        public_candidate = False
    else:
        tier = "final_candidate"
        public_candidate = True

    return {
        "schema_version": "vlora-candidate-tier-v1",
        "tier": tier,
        "public_candidate": public_candidate,
        "passed_checks": passed_checks,
        "review_reasons": review_reasons,
        "rejection_reasons": rejection_reasons,
        "policy": (
            "final candidates must pass event-train repeatability, red-noise "
            "significance in multiple time blocks, harmonic independence, "
            "physical and lightweight reference false-positive checks, and "
            "pixel localization"
        ),
    }


def apply_candidate_tiers(candidates: Iterable[Mapping]) -> dict:
    audit: list[dict] = []
    for source in candidates:
        row = dict(source)
        row["candidate_decision"] = candidate_tier(row)
        audit.append(row)
    return {
        "audit": audit,
        "final": [
            row
            for row in audit
            if row["candidate_decision"]["tier"] == "final_candidate"
        ],
        "review": [
            row
            for row in audit
            if row["candidate_decision"]["tier"] == "review_required"
        ],
        "rejected": [
            row
            for row in audit
            if row["candidate_decision"]["tier"] == "rejected_signal"
        ],
    }


__all__ = ["apply_candidate_tiers", "candidate_tier"]
