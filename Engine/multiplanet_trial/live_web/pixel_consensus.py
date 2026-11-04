"""Robust cross-sector consensus for target-pixel localization evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping


ON_TARGET_STATUS = "on_target_consistent"
OFF_TARGET_STATUS = "possible_off_target"
ALL_ON_TARGET_STATUS = "all_available_sectors_on_target_consistent"
ON_TARGET_CONSENSUS_STATUS = "on_target_consensus_across_sectors"
OFF_TARGET_CONSENSUS_STATUS = "possible_off_target_consensus_across_sectors"
LEGACY_OFF_TARGET_STATUS = "possible_off_target_in_at_least_one_sector"
MIXED_STATUS = "pixel_localization_mixed_or_unresolved"
NO_LOCALIZATION_STATUS = "no_pixel_localization"

MIN_CONSENSUS_SECTORS = 2
MIN_CONSENSUS_FRACTION = 0.60


def is_off_target_consensus(status: object) -> bool:
    """Accept both the V2 consensus value and legacy saved-result value."""

    return str(status or "") in {
        OFF_TARGET_CONSENSUS_STATUS,
        LEGACY_OFF_TARGET_STATUS,
    }


def aggregate_pixel_localizations(localizations: Iterable[Mapping]) -> dict:
    """Aggregate directional sector evidence without a one-sector veto."""

    rows = list(localizations)
    statuses = [str(row.get("status") or "") for row in rows]
    counts = Counter(statuses)
    on_target = int(counts.get(ON_TARGET_STATUS, 0))
    off_target = int(counts.get(OFF_TARGET_STATUS, 0))
    informative = on_target + off_target
    unresolved = len(rows) - informative
    hard_rejected = sum(bool(row.get("hard_rejected")) for row in rows)
    off_target_fraction = float(off_target / informative) if informative else None
    on_target_fraction = float(on_target / informative) if informative else None

    if not rows:
        aggregate_status = NO_LOCALIZATION_STATUS
        decision_basis = "no_sector_localizations"
    elif hard_rejected:
        aggregate_status = OFF_TARGET_CONSENSUS_STATUS
        decision_basis = "at_least_one_sector_has_explicit_hard_rejection"
    elif (
        off_target >= MIN_CONSENSUS_SECTORS
        and off_target_fraction is not None
        and off_target_fraction >= MIN_CONSENSUS_FRACTION
    ):
        aggregate_status = OFF_TARGET_CONSENSUS_STATUS
        decision_basis = "repeated_off_target_majority_among_informative_sectors"
    elif on_target == len(rows):
        aggregate_status = ALL_ON_TARGET_STATUS
        decision_basis = "every_available_sector_is_on_target_consistent"
    elif (
        on_target >= MIN_CONSENSUS_SECTORS
        and on_target_fraction is not None
        and on_target_fraction >= MIN_CONSENSUS_FRACTION
    ):
        aggregate_status = ON_TARGET_CONSENSUS_STATUS
        decision_basis = "repeated_on_target_majority_among_informative_sectors"
    else:
        aggregate_status = MIXED_STATUS
        decision_basis = "insufficient_or_conflicting_cross_sector_evidence"

    return {
        "schema_version": "avril-pixel-cross-sector-consensus-v1",
        "aggregate_status": aggregate_status,
        "decision_basis": decision_basis,
        "sector_count": len(rows),
        "informative_sector_count": informative,
        "on_target_sector_count": on_target,
        "off_target_sector_count": off_target,
        "unresolved_or_insufficient_sector_count": unresolved,
        "hard_rejected_sector_count": hard_rejected,
        "on_target_fraction_among_informative_sectors": on_target_fraction,
        "off_target_fraction_among_informative_sectors": off_target_fraction,
        "status_counts": dict(sorted(counts.items())),
        "thresholds": {
            "minimum_consensus_sectors": MIN_CONSENSUS_SECTORS,
            "minimum_consensus_fraction_among_informative_sectors": (
                MIN_CONSENSUS_FRACTION
            ),
        },
        "policy": (
            "minority sector outliers never veto; repeated majority off-target "
            "evidence or an explicit hard rejection is required"
        ),
    }


__all__ = [
    "ALL_ON_TARGET_STATUS",
    "LEGACY_OFF_TARGET_STATUS",
    "MIXED_STATUS",
    "NO_LOCALIZATION_STATUS",
    "ON_TARGET_CONSENSUS_STATUS",
    "OFF_TARGET_CONSENSUS_STATUS",
    "aggregate_pixel_localizations",
    "is_off_target_consensus",
]
