from __future__ import annotations

import pytest

from .pixel_consensus import (
    ALL_ON_TARGET_STATUS,
    MIXED_STATUS,
    NO_LOCALIZATION_STATUS,
    ON_TARGET_CONSENSUS_STATUS,
    OFF_TARGET_CONSENSUS_STATUS,
    aggregate_pixel_localizations,
)


def _rows(status: str, count: int, *, hard_rejected: bool = False) -> list[dict]:
    return [
        {"status": status, "hard_rejected": hard_rejected}
        for _ in range(count)
    ]


def test_no_pixel_rows_remain_unavailable() -> None:
    result = aggregate_pixel_localizations([])
    assert result["aggregate_status"] == NO_LOCALIZATION_STATUS


def test_every_available_sector_on_target_passes() -> None:
    result = aggregate_pixel_localizations(_rows("on_target_consistent", 5))
    assert result["aggregate_status"] == ALL_ON_TARGET_STATUS
    assert result["on_target_sector_count"] == 5


def test_one_off_target_outlier_never_vetoes_many_on_target_sectors() -> None:
    result = aggregate_pixel_localizations(
        _rows("on_target_consistent", 12) + _rows("possible_off_target", 1)
    )
    assert result["aggregate_status"] == ON_TARGET_CONSENSUS_STATUS
    assert result["off_target_fraction_among_informative_sectors"] == pytest.approx(
        1 / 13
    )


def test_repeated_off_target_majority_is_material_evidence() -> None:
    result = aggregate_pixel_localizations(
        _rows("possible_off_target", 3)
        + _rows("on_target_consistent", 1)
        + _rows("pixel_localization_unresolved", 4)
    )
    assert result["aggregate_status"] == OFF_TARGET_CONSENSUS_STATUS
    assert result["decision_basis"] == (
        "repeated_off_target_majority_among_informative_sectors"
    )


def test_explicit_hard_rejection_remains_sufficient() -> None:
    result = aggregate_pixel_localizations(
        _rows("on_target_consistent", 20)
        + _rows("possible_off_target", 1, hard_rejected=True)
    )
    assert result["aggregate_status"] == OFF_TARGET_CONSENSUS_STATUS
    assert result["hard_rejected_sector_count"] == 1


def test_toi786_like_sector_mix_has_positive_on_target_consensus() -> None:
    result = aggregate_pixel_localizations(
        _rows("on_target_consistent", 31)
        + _rows("possible_off_target", 4)
        + _rows("pixel_localization_unresolved", 7)
        + _rows("insufficient_pixel_coverage", 1)
    )
    assert result["aggregate_status"] == ON_TARGET_CONSENSUS_STATUS
    assert result["off_target_sector_count"] == 4
    assert result["off_target_fraction_among_informative_sectors"] == pytest.approx(
        4 / 35
    )


def test_tied_informative_sectors_remain_unresolved() -> None:
    result = aggregate_pixel_localizations(
        _rows("on_target_consistent", 1)
        + _rows("possible_off_target", 1)
        + _rows("insufficient_pixel_coverage", 10)
    )
    assert result["aggregate_status"] == MIXED_STATUS
