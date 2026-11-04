"""Truth-blind harmonic recovery and sealed-period validation for live runs.

The discovery engine is intentionally left frozen.  This module restores two
rules used by the original project before a live candidate may be ranked:

1. deterministic ``P/n`` and ``nP`` hypotheses are checked when discovery has
   selected an alias; and
2. the ephemeris must reproduce a transit in the sealed segment.  The sealed
   segment may only resolve the small period uncertainty left by the discovery
   baseline; it is never searched over arbitrary periods.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from multiplanet_trial.v03.lightkurve_evidence import measure_fixed_ephemeris
from multiplanet_trial.v08_eval.multibranch_search import (
    SearchBranch,
    fuse_branch_candidates,
)


PRIMARY_BRANCH = "pdcsap_lightkurve_bls"
HARMONIC_FACTORS = (0.5, 2.0)
SEALED_PERIOD_GRID_SIZE = 401
MIN_SEALED_SNR = 6.0
MIN_WEAK_SEALED_SNR = 3.0
MIN_DEPTH_RATIO = 0.40
MAX_DEPTH_RATIO = 1.80
MIN_PRIMARY_RESIDUAL_SNR = 8.0
MIN_PRIMARY_RESIDUAL_SDE = 8.0
MIN_PRIMARY_RESIDUAL_EVENTS = 2
MAX_PRIMARY_RESIDUAL_HYPOTHESES = 32
MIN_EVENT_REPEATABILITY_SNR = 3.0
# Common low-order residues are materialized for backwards-compatible result
# files.  Every individual event is also serialized below, so the ranker can
# evaluate any integer order required by the actual candidate pool without a
# pre-search cap.
EVENT_REPEATABILITY_MODULI = tuple(range(2, 11))


def prepare_event_repeatability_views(branch: SearchBranch) -> tuple[dict, ...]:
    """Sort immutable sector views once for repeated event-train checks."""

    views: list[dict] = []
    segment_values = np.asarray(branch.segment_id)
    for segment in sorted({int(value) for value in np.unique(segment_values)}):
        mask = segment_values == segment
        time = np.asarray(branch.time[mask], dtype=float)
        flux = np.asarray(branch.flux[mask], dtype=float)
        order = np.argsort(time)
        views.append(
            {
                "segment": int(segment),
                "time": time[order],
                "flux": flux[order],
            }
        )
    return tuple(views)


def _segment_set(value) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {int(item) for item in value}
    return {int(value)}


def _metric_snr(metrics: Mapping | None) -> float:
    if not metrics or metrics.get("snr") is None:
        return float("-inf")
    value = float(metrics["snr"])
    return value if np.isfinite(value) else float("-inf")


def _effective_discovery_baseline_days(
    branch: SearchBranch, search_mask: np.ndarray, period_days: float
) -> float:
    """Return the longest continuously observed discovery block.

    The full time span can include multi-year gaps.  Treating those gaps as
    continuous coverage grossly overstates a raw BLS candidate's period
    precision and made the sealed local-refinement window too narrow.  Cycle
    counts across gaps remain valuable evidence, but the uncertainty window
    must be wide enough for the longest actually observed block to establish
    the ephemeris on its own.
    """
    ranges: list[tuple[float, float]] = []
    search_segments = np.asarray(branch.segment_id)[search_mask]
    search_time = np.asarray(branch.time, dtype=float)[search_mask]
    for segment in sorted({int(value) for value in np.unique(search_segments)}):
        selected = search_time[search_segments == segment]
        selected = selected[np.isfinite(selected)]
        if selected.size >= 2:
            ranges.append((float(np.min(selected)), float(np.max(selected))))
    if not ranges:
        return float(period_days)
    spans = [end - start for start, end in ranges]
    typical_span = float(np.median([span for span in spans if span > 0] or [0.0]))
    join_gap = max(2.0, 0.25 * typical_span)
    blocks: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if blocks and start - blocks[-1][1] <= join_gap:
            blocks[-1] = (blocks[-1][0], max(blocks[-1][1], end))
        else:
            blocks.append((start, end))
    longest = max((end - start for start, end in blocks), default=0.0)
    return max(float(period_days), float(longest))


def _mad_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    median = float(np.nanmedian(finite))
    sigma = float(1.4826 * np.nanmedian(np.abs(finite - median)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(finite))
    return sigma if np.isfinite(sigma) and sigma > 0 else 0.0


def event_repeatability_metrics(
    branch: SearchBranch,
    *,
    holdout_segment: int | list[int] | tuple[int, ...] | None,
    period_days: float,
    epoch_btjd: float,
    duration_hours: float,
    prepared_views: tuple[dict, ...] | None = None,
) -> dict:
    """Count covered predicted events that actually contain a local dip.

    Aggregate folded S/N alone cannot distinguish ``P`` from ``P/2``: the
    real events still add coherently at the shorter period while empty
    interleaved windows merely dilute the depth.  This diagnostic measures
    every covered prediction separately.  A correct ephemeris should not only
    have high folded S/N; most of its individually testable windows should
    dim.  It also lets a shorter period beat ``2P`` when both its parity sets
    contain real events.
    """
    sealed_segments = _segment_set(holdout_segment)
    period = float(period_days)
    epoch = float(epoch_btjd)
    duration_days = float(duration_hours) / 24.0
    rows: list[dict] = []
    views = (
        prepare_event_repeatability_views(branch)
        if prepared_views is None
        else prepared_views
    )
    for view in views:
        segment = int(view["segment"])
        time = np.asarray(view["time"], dtype=float)
        flux = np.asarray(view["flux"], dtype=float)
        if time.size < 13:
            continue
        first_cycle = int(np.floor((time[0] - epoch) / period)) - 1
        last_cycle = int(np.ceil((time[-1] - epoch) / period)) + 1
        for cycle in range(first_cycle, last_cycle + 1):
            center = epoch + cycle * period
            left = int(np.searchsorted(time, center - 4.0 * duration_days, side="left"))
            right = int(np.searchsorted(time, center + 4.0 * duration_days, side="right"))
            if right - left < 13:
                continue
            local_time = time[left:right]
            local_flux = flux[left:right]
            distance = np.abs(local_time - center)
            inside = distance <= 0.5 * duration_days
            nearby = (distance >= 1.5 * duration_days) & (
                distance <= 4.0 * duration_days
            )
            if int(inside.sum()) < 3 or int(nearby.sum()) < 10:
                continue
            depth_ppm = float(
                (np.nanmedian(local_flux[nearby]) - np.nanmedian(local_flux[inside]))
                * 1e6
            )
            scatter_ppm = _mad_sigma(local_flux[nearby]) * 1e6
            snr = (
                depth_ppm / (scatter_ppm / np.sqrt(int(inside.sum())))
                if scatter_ppm > 0
                else 0.0
            )
            supported = bool(depth_ppm > 0 and snr >= MIN_EVENT_REPEATABILITY_SNR)
            rows.append(
                {
                    "segment": segment,
                    "cycle": cycle,
                    "parity": int(cycle % 2),
                    "center_btjd": float(center),
                    "depth_ppm": depth_ppm,
                    "snr": float(snr),
                    "supported": supported,
                    "sealed": segment in sealed_segments,
                }
            )

    def summarize(selected: list[dict]) -> dict:
        supported = [row for row in selected if row["supported"]]
        return {
            "testable_events": len(selected),
            "supported_events": len(supported),
            "support_fraction": len(supported) / len(selected) if selected else None,
            "median_supported_depth_ppm": (
                float(np.median([row["depth_ppm"] for row in supported]))
                if supported
                else None
            ),
            "minimum_supported_event_snr": (
                float(min(row["snr"] for row in supported)) if supported else None
            ),
        }

    summary = summarize(rows)
    summary["search"] = summarize([row for row in rows if not row["sealed"]])
    summary["sealed"] = summarize([row for row in rows if row["sealed"]])
    summary["parity"] = {
        str(parity): summarize([row for row in rows if row["parity"] == parity])
        for parity in (0, 1)
    }
    summary["modulo"] = {
        str(order): {
            str(remainder): summarize(
                [row for row in rows if int(row["cycle"]) % order == remainder]
            )
            for remainder in range(order)
        }
        for order in EVENT_REPEATABILITY_MODULI
    }
    summary["event_support"] = [
        {
            "segment": int(row["segment"]),
            "cycle": int(row["cycle"]),
            "center_btjd": round(float(row["center_btjd"]), 8),
            "depth_ppm": float(row["depth_ppm"]),
            "snr": float(row["snr"]),
            "supported": bool(row["supported"]),
            "sealed": bool(row["sealed"]),
        }
        for row in rows
    ]
    summary["unsupported_event_centers_btjd"] = [
        round(float(row["center_btjd"]), 6)
        for row in rows
        if not row["supported"]
    ][:20]
    summary["minimum_event_snr"] = MIN_EVENT_REPEATABILITY_SNR
    return summary


def _verification_from_metrics(
    search: Mapping,
    sealed: Mapping,
    *,
    discovery_period_days: float,
    sealed_period_days: float,
    period_uncertainty_days: float,
    period_trials: int,
) -> dict:
    search_depth = search.get("depth_ppm")
    sealed_depth = sealed.get("depth_ppm")
    depth_ratio = None
    if (
        search_depth is not None
        and sealed_depth is not None
        and float(search_depth) > 0
        and np.isfinite(float(search_depth))
        and np.isfinite(float(sealed_depth))
    ):
        depth_ratio = float(sealed_depth) / float(search_depth)

    reasons: list[str] = []
    code = "sealed_passed"
    passed = True
    outcome = "passed"
    testable = True
    if search.get("status") != "supported" or _metric_snr(search) < 5.0:
        code = "primary_discovery_not_supported"
        reasons.append("发现期的 PDCSAP 主光变不支持这个周期")
        passed = False
    elif int(sealed.get("events") or 0) < 1 or sealed.get("status") == "insufficient_coverage":
        code = "sealed_insufficient_coverage"
        reasons.append("封存期没有有效数据覆盖预测的凌日窗口")
        passed = False
        outcome = "untestable"
        testable = False
    elif depth_ratio is None or not (MIN_DEPTH_RATIO <= depth_ratio <= MAX_DEPTH_RATIO):
        code = "sealed_depth_inconsistent"
        reasons.append("封存期的凌日深度与发现期不一致")
        passed = False
    elif (
        sealed.get("status") == "weak_support"
        and _metric_snr(sealed) >= MIN_WEAK_SEALED_SNR
    ):
        code = "sealed_weak_support"
        reasons.extend(
            [
                "封存期在预测窗口内有同方向弱变暗",
                "深度与发现期一致，但单折信噪比未达强通过门槛",
            ]
        )
        passed = False
        outcome = "inconclusive"
    elif sealed.get("status") != "supported" or _metric_snr(sealed) < MIN_SEALED_SNR:
        code = "sealed_not_reproduced"
        reasons.append("封存期在预测时间没有再次出现足够强的变暗")
        passed = False
    else:
        reasons.extend(
            [
                "封存期在预测窗口内重新出现变暗",
                "封存期与发现期的凌日深度一致",
            ]
        )
    if not passed and testable and outcome != "inconclusive":
        outcome = "failed"

    return {
        "passed": passed,
        "outcome": outcome if not passed else "passed",
        "testable": testable,
        "status": code,
        "status_zh": (
            "封存验证通过"
            if passed
            else (
                "封存期弱支持，待合并"
                if outcome == "inconclusive"
                else ("封存期无覆盖" if not testable else "未通过封存验证")
            )
        ),
        "reasons_zh": reasons,
        "primary_branch": PRIMARY_BRANCH,
        "search_metrics": dict(search),
        "sealed_metrics": dict(sealed),
        "discovery_period_days": float(discovery_period_days),
        "validated_period_days": float(sealed_period_days),
        "period_uncertainty_days": float(period_uncertainty_days),
        "period_trials": int(period_trials),
        "sealed_depth_to_discovery_depth_ratio": depth_ratio,
        "depth_ratio_allowed": [MIN_DEPTH_RATIO, MAX_DEPTH_RATIO],
        "minimum_sealed_snr": MIN_SEALED_SNR,
        "minimum_weak_sealed_snr": MIN_WEAK_SEALED_SNR,
        "truth_catalog_used": False,
    }


def validate_locked_ephemeris(
    branch: SearchBranch,
    *,
    holdout_segment: int | list[int] | tuple[int, ...] | None,
    period_days: float,
    epoch_btjd: float,
    duration_hours: float,
    search_metrics: Mapping | None = None,
    grid_size: int = SEALED_PERIOD_GRID_SIZE,
) -> dict:
    """Validate one discovery ephemeris in the sealed segment.

    The permitted period interval is the same uncertainty construction used by
    the original iterative search: about half a transit duration after
    propagating across the discovery baseline, bounded to 0.005--0.3 percent.
    """
    sealed_segments = _segment_set(holdout_segment)
    if not sealed_segments:
        return {
            "passed": False,
            "outcome": "untestable",
            "testable": False,
            "status": "no_sealed_segment",
            "status_zh": "没有封存数据",
            "reasons_zh": ["只有一个观测段，无法作独立封存验证"],
            "truth_catalog_used": False,
        }
    search_mask = ~np.isin(branch.segment_id, list(sealed_segments))
    sealed_mask = ~search_mask
    if search_metrics is None:
        search_metrics = measure_fixed_ephemeris(
            branch.time[search_mask],
            branch.flux[search_mask],
            period_days=period_days,
            epoch_btjd=epoch_btjd,
            duration_hours=duration_hours,
        )
    if search_mask.sum() < 3 or sealed_mask.sum() < 3:
        empty = {
            "depth_ppm": None,
            "snr": None,
            "events": 0,
            "transit_points": 0,
            "baseline_points": 0,
            "status": "insufficient_coverage",
        }
        return _verification_from_metrics(
            search_metrics,
            empty,
            discovery_period_days=period_days,
            sealed_period_days=period_days,
            period_uncertainty_days=0.0,
            period_trials=0,
        )

    baseline = _effective_discovery_baseline_days(
        branch, search_mask, float(period_days)
    )
    duration_days = float(duration_hours) / 24.0
    # Allow one transit duration of accumulated phase uncertainty.  Epoch and
    # period are both estimated from finite-duration events, so a half-duration
    # allowance for each is the conservative discovery-only uncertainty.  The
    # sealed data may resolve this predeclared local interval but may not search
    # arbitrary periods.
    half_width = duration_days * float(period_days) / baseline
    half_width = float(
        np.clip(half_width, float(period_days) * 5e-5, float(period_days) * 0.003)
    )
    trial_periods = np.linspace(
        float(period_days) - half_width,
        float(period_days) + half_width,
        int(grid_size),
    )
    # Resolve the predeclared local period uncertainty with one ephemeris that
    # must explain *both* discovery and sealed data.  Selecting a period from
    # the seal alone can move the prediction off the very discovery events
    # that created the candidate; that is two unrelated fits, not validation.
    # The maximin objective is deliberately conservative: improve the weaker
    # side first, then use their combined strength and proximity to the locked
    # discovery period only as tie-breakers.  After this single choice, every
    # sealed segment is tested at exactly the same frozen period and any
    # contradiction remains a veto.
    discovery_time = branch.time[search_mask]
    discovery_flux = branch.flux[search_mask]
    sealed_time = branch.time[sealed_mask]
    sealed_flux = branch.flux[sealed_mask]
    sealed_segment_masks = {
        int(segment): branch.segment_id == int(segment)
        for segment in sorted(sealed_segments)
    }
    # Freeze coverage geometry at the discovery ephemeris.  A trial period may
    # not improve its score by sliding a predicted event out of a segment that
    # was testable before the seal was opened.
    required_sealed_segments = []
    for segment, segment_mask in sealed_segment_masks.items():
        locked_metrics = measure_fixed_ephemeris(
            branch.time[segment_mask],
            branch.flux[segment_mask],
            period_days=float(period_days),
            epoch_btjd=epoch_btjd,
            duration_hours=duration_hours,
        )
        if (
            locked_metrics.get("status") != "insufficient_coverage"
            and int(locked_metrics.get("events") or 0) > 0
        ):
            required_sealed_segments.append(segment)
    best_joint_period = float(period_days)
    best_joint_search_metrics = dict(search_metrics)
    best_joint_sealed_metrics: dict = measure_fixed_ephemeris(
        sealed_time,
        sealed_flux,
        period_days=float(period_days),
        epoch_btjd=epoch_btjd,
        duration_hours=duration_hours,
    )
    best_joint_key = (float("-inf"), float("-inf"), float("-inf"))
    for trial_period in trial_periods:
        trial_search_metrics = measure_fixed_ephemeris(
            discovery_time,
            discovery_flux,
            period_days=float(trial_period),
            epoch_btjd=epoch_btjd,
            duration_hours=duration_hours,
        )
        search_snr = _metric_snr(trial_search_metrics)
        # The old objective compared discovery with all sealed cadences pooled
        # together.  A strong late segment could then hide an explicit failure
        # in an early segment and pull the selected period onto the wrong cycle
        # connection.  Every pre-covered sealed segment is now an equal member
        # of the maximin objective.
        segment_snrs = []
        for segment in required_sealed_segments:
            segment_mask = sealed_segment_masks[segment]
            trial_segment_metrics = measure_fixed_ephemeris(
                branch.time[segment_mask],
                branch.flux[segment_mask],
                period_days=float(trial_period),
                epoch_btjd=epoch_btjd,
                duration_hours=duration_hours,
            )
            if (
                trial_segment_metrics.get("status") == "insufficient_coverage"
                or int(trial_segment_metrics.get("events") or 0) < 1
            ):
                segment_snrs.append(float("-inf"))
            else:
                segment_snrs.append(_metric_snr(trial_segment_metrics))
        all_snrs = [search_snr, *segment_snrs]
        joint_key = (
            min(all_snrs),
            float(np.sqrt(sum(max(value, 0.0) ** 2 for value in all_snrs))),
            -abs(float(trial_period) - float(period_days)),
        )
        if joint_key > best_joint_key:
            best_joint_key = joint_key
            best_joint_period = float(trial_period)
            best_joint_search_metrics = dict(trial_search_metrics)
    best_joint_sealed_metrics = measure_fixed_ephemeris(
        sealed_time,
        sealed_flux,
        period_days=best_joint_period,
        epoch_btjd=epoch_btjd,
        duration_hours=duration_hours,
    )
    original_discovery_search_metrics = dict(search_metrics)
    search_metrics = best_joint_search_metrics
    per_segment: dict[str, dict] = {}
    strong: list[tuple[float, dict]] = []
    weak: list[tuple[float, dict]] = []
    ambiguous: list[tuple[float, dict]] = []
    contradictions: list[int] = []
    search_depth = float(search_metrics.get("depth_ppm") or 0.0)
    for segment in sorted(sealed_segments):
        segment_mask = branch.segment_id == segment
        best_period = best_joint_period
        best_metrics = measure_fixed_ephemeris(
            branch.time[segment_mask],
            branch.flux[segment_mask],
            period_days=best_joint_period,
            epoch_btjd=epoch_btjd,
            duration_hours=duration_hours,
        )
        best_metrics = {
            **best_metrics,
            "best_period_days": best_period,
            "period_selected_from_joint_sealed_group": True,
        }
        per_segment[str(segment)] = best_metrics
        testable = (
            best_metrics.get("status") != "insufficient_coverage"
            and int(best_metrics.get("events") or 0) > 0
        )
        depth = best_metrics.get("depth_ppm")
        depth_ratio = (
            float(depth) / search_depth
            if depth is not None and search_depth > 0 and np.isfinite(float(depth))
            else None
        )
        supported = (
            testable
            and best_metrics.get("status") == "supported"
            and _metric_snr(best_metrics) >= MIN_SEALED_SNR
            and depth_ratio is not None
            and MIN_DEPTH_RATIO <= depth_ratio <= MAX_DEPTH_RATIO
        )
        weak_supported = (
            testable
            and best_metrics.get("status") in {"supported", "weak_support"}
            and _metric_snr(best_metrics) >= MIN_WEAK_SEALED_SNR
            and depth_ratio is not None
            and MIN_DEPTH_RATIO <= depth_ratio <= MAX_DEPTH_RATIO
        )
        if supported:
            strong.append((best_period, best_metrics))
        elif weak_supported:
            weak.append((best_period, best_metrics))
        elif (
            testable
            and depth is not None
            and np.isfinite(float(depth))
            and float(depth) > 0
            and _metric_snr(best_metrics) >= MIN_WEAK_SEALED_SNR
        ):
            # A significant, correctly phased dimming with an anomalous depth
            # is not a contradiction.  It can reflect cadence dilution,
            # detrending, crowding, or real depth variability.  Preserve it as
            # inconclusive, but do not let two such depth-inconsistent folds
            # combine into a validation pass.
            ambiguous.append((best_period, best_metrics))
        elif testable:
            contradictions.append(segment)

    if strong:
        validated_period = float(np.median([row[0] for row in strong]))
        strong_metrics = [row[1] for row in strong]
        sealed_summary = {
            "depth_ppm": float(np.median([row["depth_ppm"] for row in strong_metrics])),
            "snr": float(min(_metric_snr(row) for row in strong_metrics)),
            "events": int(sum(int(row.get("events") or 0) for row in strong_metrics)),
            "transit_points": int(
                sum(int(row.get("transit_points") or 0) for row in strong_metrics)
            ),
            "baseline_points": int(
                sum(int(row.get("baseline_points") or 0) for row in strong_metrics)
            ),
            "local_scatter_ppm": float(
                np.median(
                    [
                        row["local_scatter_ppm"]
                        for row in strong_metrics
                        if row.get("local_scatter_ppm") is not None
                    ]
                )
            ),
            "status": "supported",
        }
    elif weak:
        validated_period = float(np.median([row[0] for row in weak]))
        weak_metrics = [row[1] for row in weak]
        sealed_summary = {
            "depth_ppm": float(np.median([row["depth_ppm"] for row in weak_metrics])),
            "snr": float(np.sqrt(sum(_metric_snr(row) ** 2 for row in weak_metrics))),
            "events": int(sum(int(row.get("events") or 0) for row in weak_metrics)),
            "transit_points": int(
                sum(int(row.get("transit_points") or 0) for row in weak_metrics)
            ),
            "baseline_points": int(
                sum(int(row.get("baseline_points") or 0) for row in weak_metrics)
            ),
            "local_scatter_ppm": float(
                np.median(
                    [
                        row["local_scatter_ppm"]
                        for row in weak_metrics
                        if row.get("local_scatter_ppm") is not None
                    ]
                )
            ),
            "status": "weak_support",
        }
    elif ambiguous:
        validated_period = float(np.median([row[0] for row in ambiguous]))
        ambiguous_metrics = [row[1] for row in ambiguous]
        sealed_summary = {
            "depth_ppm": float(
                np.median([row["depth_ppm"] for row in ambiguous_metrics])
            ),
            "snr": float(
                np.sqrt(sum(_metric_snr(row) ** 2 for row in ambiguous_metrics))
            ),
            "events": int(
                sum(int(row.get("events") or 0) for row in ambiguous_metrics)
            ),
            "transit_points": int(
                sum(int(row.get("transit_points") or 0) for row in ambiguous_metrics)
            ),
            "baseline_points": int(
                sum(int(row.get("baseline_points") or 0) for row in ambiguous_metrics)
            ),
            "local_scatter_ppm": float(
                np.median(
                    [
                        row["local_scatter_ppm"]
                        for row in ambiguous_metrics
                        if row.get("local_scatter_ppm") is not None
                    ]
                )
            ),
            "status": "supported",
        }
    else:
        validated_period = float(period_days)
        testable_metrics = [
            row
            for row in per_segment.values()
            if row.get("status") != "insufficient_coverage"
        ]
        sealed_summary = (
            max(testable_metrics, key=_metric_snr)
            if testable_metrics
            else {
                "depth_ppm": None,
                "snr": None,
                "events": 0,
                "transit_points": 0,
                "baseline_points": 0,
                "status": "insufficient_coverage",
            }
        )
    result = _verification_from_metrics(
        search_metrics,
        sealed_summary,
        discovery_period_days=period_days,
        sealed_period_days=validated_period,
        period_uncertainty_days=half_width,
        period_trials=len(trial_periods),
    )
    result["discovery_effective_baseline_days"] = float(baseline)
    result["period_window_basis"] = (
        "one_transit_duration_over_longest_continuous_discovery_block"
    )
    result["sealed_period_selection"] = (
        "single_discovery_and_each_precovered_sealed_segment_maximin_period"
    )
    result["period_selection_objective"] = (
        "maximize_minimum_snr_across_discovery_and_every_precovered_sealed_segment"
    )
    result["period_selection_required_sealed_segments"] = required_sealed_segments
    result["original_discovery_search_metrics"] = original_discovery_search_metrics
    result["joint_refined_discovery_metrics"] = dict(best_joint_search_metrics)
    result["joint_refined_sealed_group_metrics"] = dict(best_joint_sealed_metrics)
    result["joint_refined_period_days"] = float(best_joint_period)
    result["weak_support_eligible_for_combination"] = bool(
        result.get("outcome") == "inconclusive" and weak and not ambiguous
    )
    if contradictions:
        result["passed"] = False
        result["outcome"] = "failed"
        result["testable"] = True
        result["status"] = "sealed_segment_contradiction"
        result["status_zh"] = "封存段之间互相矛盾"
        result["reasons_zh"] = [
            "至少一个有有效覆盖的封存段没有按预测复现变暗"
        ]
        result["weak_support_eligible_for_combination"] = False
    elif ambiguous and not strong:
        result["passed"] = False
        result["outcome"] = "inconclusive"
        result["testable"] = True
        result["status"] = "sealed_depth_ambiguous"
        result["status_zh"] = "封存期同相变暗显著，但深度不一致"
        result["reasons_zh"] = [
            "封存期在预测窗口内显著变暗，不构成反证",
            "封存深度超出固定一致性范围，不能单独验证候选",
        ]
        result["weak_support_eligible_for_combination"] = False
    result["sealed_segments"] = sorted(sealed_segments)
    result["per_sealed_segment_metrics"] = per_segment
    result["sealed_segments_with_predicted_events"] = sum(
        int(metrics.get("events") or 0) > 0
        for metrics in per_segment.values()
    )
    result["sealed_segments_with_strong_support"] = len(strong)
    result["sealed_segments_with_weak_support"] = len(weak)
    result["sealed_segments_with_ambiguous_support"] = len(ambiguous)
    result["sealed_segments_with_contradictions"] = contradictions
    result["event_repeatability"] = event_repeatability_metrics(
        branch,
        holdout_segment=sorted(sealed_segments),
        period_days=validated_period,
        epoch_btjd=epoch_btjd,
        duration_hours=duration_hours,
    )
    if result["passed"]:
        result["reasons_zh"] = [
            f"{len(strong)} 个独立封存段在各自的预测窗口内复现变暗",
            (
                f"其余 {len(weak)} 个有覆盖封存段只有同方向弱支持，"
                f"{len(ambiguous)} 个同相变暗深度不确定，无反证"
            ),
        ]
    return result


def _all_branch_evidence(
    branches: Mapping[str, SearchBranch],
    *,
    holdout_segment: int | list[int] | tuple[int, ...] | None,
    period_days: float,
    epoch_btjd: float,
    duration_hours: float,
) -> dict:
    evidence = {}
    for key, branch in branches.items():
        sealed_segments = _segment_set(holdout_segment)
        search_mask = (
            np.ones(branch.time.size, dtype=bool)
            if not sealed_segments
            else ~np.isin(branch.segment_id, list(sealed_segments))
        )
        sealed_mask = ~search_mask if sealed_segments else np.zeros_like(search_mask)
        arguments = {
            "period_days": float(period_days),
            "epoch_btjd": float(epoch_btjd),
            "duration_hours": float(duration_hours),
        }
        evidence[key] = {
            "search": measure_fixed_ephemeris(
                branch.time[search_mask], branch.flux[search_mask], **arguments
            ),
            "holdout": measure_fixed_ephemeris(
                branch.time[sealed_mask], branch.flux[sealed_mask], **arguments
            ),
            "photometry_family": branch.photometry_family,
            "processing": branch.processing,
        }
    return evidence


def refine_fused_ephemerides_from_members(
    system: dict,
    branches: Mapping[str, SearchBranch],
) -> dict:
    """Undo destructive averaging inside a fused candidate cluster.

    The frozen fusion code correctly groups near-identical hypotheses, but it
    represents a group by a weighted average of period, epoch, and duration.
    With a long discovery baseline, even a sub-percent period shift can move a
    sealed prediction outside the transit.  Fusion should merge *identity*,
    not manufacture a new ephemeris.  Re-measure the fused value and every
    original member on discovery-only PDCSAP, then retain the member that best
    explains those untouched discovery points.  No sealed flux or catalog
    truth participates in this choice.
    """
    primary = branches.get(PRIMARY_BRANCH)
    clusters = [dict(row) for row in system.get("candidate_fusion", [])]
    if primary is None:
        return {
            "clusters_checked": len(clusters),
            "clusters_refined": 0,
            "status": "primary_branch_missing",
            "truth_catalog_used": False,
            "sealed_flux_used": False,
        }

    holdout_segments = _segment_set(system.get("holdout_sectors"))
    if not holdout_segments:
        holdout_segments = _segment_set(system.get("holdout_sector"))
    search_mask = (
        ~np.isin(primary.segment_id, list(holdout_segments))
        if holdout_segments
        else np.ones(primary.time.size, dtype=bool)
    )
    refined = 0
    proposals_checked = 0
    for cluster in clusters:
        proposals = [
            {
                "period_days": float(cluster["period_days"]),
                "epoch_btjd": float(cluster["epoch_btjd"]),
                "duration_hours": float(cluster["duration_hours"]),
                "source": "fused_average",
                "source_branch": None,
                "source_strength": 0.0,
            }
        ]
        for member in cluster.get("members", []):
            proposals.append(
                {
                    "period_days": float(member["period_days"]),
                    "epoch_btjd": float(member["epoch_btjd"]),
                    "duration_hours": float(member["duration_hours"]),
                    "source": "original_cluster_member",
                    "source_branch": member.get("branch"),
                    "source_strength": float(member.get("robust_sde") or 0.0),
                }
            )

        unique: list[dict] = []
        seen: set[tuple[float, float, float]] = set()
        for proposal in proposals:
            key = (
                round(proposal["period_days"], 10),
                round(proposal["epoch_btjd"], 8),
                round(proposal["duration_hours"], 6),
            )
            if key in seen:
                continue
            seen.add(key)
            metrics = measure_fixed_ephemeris(
                primary.time[search_mask],
                primary.flux[search_mask],
                period_days=proposal["period_days"],
                epoch_btjd=proposal["epoch_btjd"],
                duration_hours=proposal["duration_hours"],
            )
            unique.append({**proposal, "search_metrics": metrics})
        proposals_checked += len(unique)

        def proposal_key(row: Mapping) -> tuple:
            metrics = row["search_metrics"]
            return (
                metrics.get("status") == "supported",
                _metric_snr(metrics),
                int(metrics.get("events") or 0),
                float(row.get("source_strength") or 0.0),
            )

        winner = max(unique, key=proposal_key)
        original = {
            "period_days": float(cluster["period_days"]),
            "epoch_btjd": float(cluster["epoch_btjd"]),
            "duration_hours": float(cluster["duration_hours"]),
        }
        changed = any(
            not np.isclose(original[key], float(winner[key]), rtol=0.0, atol=1e-10)
            for key in original
        )
        if changed:
            refined += 1
            cluster["period_days"] = float(winner["period_days"])
            cluster["epoch_btjd"] = float(winner["epoch_btjd"])
            cluster["duration_hours"] = float(winner["duration_hours"])
        cluster["ephemeris_representative_selection"] = {
            "changed": changed,
            "original_fused_ephemeris": original,
            "selected_ephemeris": {
                "period_days": float(winner["period_days"]),
                "epoch_btjd": float(winner["epoch_btjd"]),
                "duration_hours": float(winner["duration_hours"]),
            },
            "selected_source": winner["source"],
            "selected_source_branch": winner["source_branch"],
            "selected_discovery_metrics": dict(winner["search_metrics"]),
            "proposal_count": len(unique),
            "selection_used_sealed_flux": False,
            "truth_catalog_used": False,
        }
        cluster["all_branch_fixed_ephemeris_evidence"] = _all_branch_evidence(
            branches,
            holdout_segment=sorted(holdout_segments),
            period_days=float(cluster["period_days"]),
            epoch_btjd=float(cluster["epoch_btjd"]),
            duration_hours=float(cluster["duration_hours"]),
        )

    system["candidate_fusion"] = clusters
    return {
        "clusters_checked": len(clusters),
        "clusters_refined": refined,
        "proposals_checked": proposals_checked,
        "selection_branch": PRIMARY_BRANCH,
        "truth_catalog_used": False,
        "sealed_flux_used": False,
    }


def _period_phase_duplicate(candidate: Mapping, existing: list[Mapping]) -> bool:
    period = float(candidate["period_days"])
    epoch = float(candidate["epoch_btjd"])
    duration_days = float(candidate["duration_hours"]) / 24.0
    for row in existing:
        old_period = float(row["period_days"])
        if abs(period - old_period) / max(period, old_period) > 0.0015:
            continue
        phase = abs((epoch - float(row["epoch_btjd"]) + 0.5 * period) % period - 0.5 * period)
        if phase <= max(duration_days, float(row["duration_hours"]) / 24.0):
            return True
    return False


def promote_all_accepted_search_hypotheses(
    system: dict,
    branches: Mapping[str, SearchBranch],
) -> dict:
    """Give every threshold-passing independent search peak a sealed exam.

    Older live profiles retained only a score-ranked prefix of accepted BLS
    peaks in ``branch_candidates`` even though the complete accepted set was
    left in ``search_diagnostics``.  That made candidate survival depend on an
    arbitrary compute-budget cap before the independent sealed data was ever
    opened.  Reconstruct every accepted diagnostic, de-duplicate only the same
    branch/period/phase identity, and rebuild fusion without a top-K limit.
    """
    holdout_segments = _segment_set(system.get("holdout_sectors"))
    if not holdout_segments:
        holdout_segments = _segment_set(system.get("holdout_sector"))
    existing = [dict(row) for row in system.get("branch_candidates", [])]
    promoted: list[dict] = []
    skipped_duplicates = 0
    skipped_missing_branch = 0
    for diagnostic in system.get("search_diagnostics", []):
        if diagnostic.get("accepted") is not True:
            continue
        branch_key = str(diagnostic.get("branch") or "")
        branch = branches.get(branch_key)
        if branch is None:
            skipped_missing_branch += 1
            continue
        proposal = {
            "period_days": float(diagnostic["period_days"]),
            "epoch_btjd": float(diagnostic["epoch_btjd"]),
            "duration_hours": float(diagnostic["duration_hours"]),
        }
        same_branch = [
            row
            for row in (*existing, *promoted)
            if row.get("branch") == branch_key
        ]
        if _period_phase_duplicate(proposal, same_branch):
            skipped_duplicates += 1
            continue
        sealed_mask = np.isin(branch.segment_id, sorted(holdout_segments))
        holdout_metrics = measure_fixed_ephemeris(
            branch.time[sealed_mask],
            branch.flux[sealed_mask],
            period_days=proposal["period_days"],
            epoch_btjd=proposal["epoch_btjd"],
            duration_hours=proposal["duration_hours"],
        )
        source = str(diagnostic.get("bls_search_pass_kind") or "")
        promoted.append(
            {
                "branch": branch_key,
                "branch_label": branch.label,
                "method_family": (
                    "bls_residual" if source.startswith("consensus_residual_") else branch.method_family
                ),
                "photometry_family": branch.photometry_family,
                "can_seed_candidates": branch.can_seed_candidates,
                "rank_in_branch": len(same_branch) + 1,
                **proposal,
                "robust_sde": diagnostic.get("robust_sde"),
                "search_metrics": dict(diagnostic.get("search_metrics") or {}),
                "holdout_metrics": holdout_metrics,
                "hard_rejected": False,
                "hypothesis_source": "accepted_search_diagnostic_for_sealed_exam",
                "bls_search_pass": diagnostic.get("bls_search_pass"),
                "bls_search_pass_kind": diagnostic.get("bls_search_pass_kind"),
            }
        )

    expanded = [*existing, *promoted]
    clusters = fuse_branch_candidates(expanded)
    for cluster in clusters:
        cluster["all_branch_fixed_ephemeris_evidence"] = _all_branch_evidence(
            branches,
            holdout_segment=sorted(holdout_segments),
            period_days=float(cluster["period_days"]),
            epoch_btjd=float(cluster["epoch_btjd"]),
            duration_hours=float(cluster["duration_hours"]),
        )
        if any(
            member.get("hypothesis_source")
            == "accepted_search_diagnostic_for_sealed_exam"
            for member in cluster.get("members", [])
        ):
            cluster["candidate_source"] = "complete_threshold_passing_search_registry"
    system["branch_candidates"] = expanded
    system["candidate_fusion"] = clusters
    return {
        "accepted_diagnostics": sum(
            row.get("accepted") is True
            for row in system.get("search_diagnostics", [])
        ),
        "preexisting_branch_candidates": len(existing),
        "promoted_for_sealed_test": len(promoted),
        "same_branch_identity_duplicates": skipped_duplicates,
        "missing_branch_diagnostics": skipped_missing_branch,
        "fused_hypotheses_for_sealed_test": len(clusters),
        "omitted_by_rank_or_compute_cap": 0,
        "truth_catalog_used": False,
    }


def promote_primary_residual_hypotheses(
    system: dict,
    branches: Mapping[str, SearchBranch],
    *,
    minimum_snr: float = MIN_PRIMARY_RESIDUAL_SNR,
    minimum_sde: float = MIN_PRIMARY_RESIDUAL_SDE,
    minimum_events: int = MIN_PRIMARY_RESIDUAL_EVENTS,
    max_hypotheses: int | None = MAX_PRIMARY_RESIDUAL_HYPOTHESES,
) -> dict:
    """Let strong primary-branch residual signals take the sealed-data exam.

    The frozen v0.8 residual merger intentionally requires two processing
    branches before a new period can enter the candidate pool.  That is a
    useful conservative default, but in a strict sealed-first live run it can
    discard a real additional planet *before* the independent seal gets a
    chance to test its prediction.  This adapter keeps only strong PDCSAP
    residual hypotheses as unranked exam candidates.  They gain no extra vote
    or score, and the live ranker will still discard every one that fails the
    pre-locked seal.
    """
    if max_hypotheses is not None and max_hypotheses < 0:
        raise ValueError("max_hypotheses cannot be negative")
    holdout_segments = _segment_set(system.get("holdout_sectors"))
    if not holdout_segments:
        holdout_segments = _segment_set(system.get("holdout_sector"))
    clusters = [dict(row) for row in system.get("candidate_fusion", [])]
    raw_candidates = []
    rejected_by_strength = 0
    for raw in system.get("branch_candidates", []):
        if raw.get("branch") != PRIMARY_BRANCH:
            continue
        if raw.get("hypothesis_source") != "consensus_residual_search":
            continue
        metrics = raw.get("search_metrics", {})
        strong = (
            not bool(raw.get("hard_rejected", False))
            and metrics.get("status") == "supported"
            and _metric_snr(metrics) >= float(minimum_snr)
            and float(raw.get("robust_sde") or 0.0) >= float(minimum_sde)
            and int(metrics.get("events") or 0) >= int(minimum_events)
        )
        if strong:
            raw_candidates.append(dict(raw))
        else:
            rejected_by_strength += 1

    raw_candidates.sort(
        key=lambda row: (
            _metric_snr(row.get("search_metrics")),
            float(row.get("robust_sde") or 0.0),
        ),
        reverse=True,
    )
    promoted: list[dict] = []
    duplicates = 0
    limited = 0
    for raw in raw_candidates:
        if _period_phase_duplicate(raw, clusters + promoted):
            duplicates += 1
            continue
        if max_hypotheses is not None and len(promoted) >= int(max_hypotheses):
            limited += 1
            continue
        fused = fuse_branch_candidates([raw])
        if not fused:
            continue
        cluster = fused[0]
        cluster["candidate_source"] = "single_primary_residual_for_sealed_test"
        cluster["conditioned_candidate"] = True
        cluster["consensus_residual_support"] = [
            {
                "period_days": float(raw["period_days"]),
                "epoch_btjd": float(raw["epoch_btjd"]),
                "duration_hours": float(raw["duration_hours"]),
                "supporting_branches": [PRIMARY_BRANCH],
                "uncalibrated_priority_score": float(
                    cluster.get("uncalibrated_priority_score") or 0.0
                ),
                "sealed_exam_only": True,
            }
        ]
        cluster["all_branch_fixed_ephemeris_evidence"] = _all_branch_evidence(
            branches,
            holdout_segment=sorted(holdout_segments),
            period_days=float(cluster["period_days"]),
            epoch_btjd=float(cluster["epoch_btjd"]),
            duration_hours=float(cluster["duration_hours"]),
        )
        promoted.append(cluster)

    clusters.extend(promoted)
    clusters.sort(
        key=lambda row: float(row.get("uncalibrated_priority_score") or 0.0),
        reverse=True,
    )
    for rank, cluster in enumerate(clusters, start=1):
        cluster["rank"] = rank
    system["candidate_fusion"] = clusters
    return {
        "eligible_primary_residual_hypotheses": len(raw_candidates),
        "promoted_for_sealed_test": len(promoted),
        "duplicates_already_in_pool": duplicates,
        "rejected_by_strength": rejected_by_strength,
        "omitted_by_safety_cap": limited,
        "policy": {
            "primary_branch": PRIMARY_BRANCH,
            "minimum_snr": float(minimum_snr),
            "minimum_robust_sde": float(minimum_sde),
            "minimum_events": int(minimum_events),
            "max_hypotheses": (
                int(max_hypotheses) if max_hypotheses is not None else None
            ),
            "may_rank_without_sealed_pass": False,
            "truth_catalog_used": False,
        },
    }


def _harmonic_cluster(
    source: Mapping,
    branches: Mapping[str, SearchBranch],
    *,
    holdout_segment: int | list[int] | tuple[int, ...],
    period_days: float,
    factor: float,
    verification: Mapping,
) -> dict:
    validated_period = float(verification["validated_period_days"])
    evidence = _all_branch_evidence(
        branches,
        holdout_segment=holdout_segment,
        period_days=validated_period,
        epoch_btjd=float(source["epoch_btjd"]),
        duration_hours=float(source["duration_hours"]),
    )
    members = []
    for key, row in evidence.items():
        if row["search"].get("status") not in {"supported", "weak_support"}:
            continue
        members.append(
            {
                "branch": key,
                "branch_label": key,
                "method_family": "harmonic_refinement",
                "photometry_family": row["photometry_family"],
                "can_seed_candidates": True,
                "rank_in_branch": 0,
                "period_days": validated_period,
                "epoch_btjd": float(source["epoch_btjd"]),
                "duration_hours": float(source["duration_hours"]),
                "robust_sde": float(row["search"].get("snr") or 0.0),
                "search_metrics": row["search"],
                "holdout_metrics": row["holdout"],
                "hard_rejected": False,
            }
        )
    supporting = sorted({row["branch"] for row in members})
    holdout_supporting = sorted(
        row["branch"]
        for row in members
        if row["holdout_metrics"].get("status") == "supported"
    )
    score = max(
        (float(row["search_metrics"].get("snr") or 0.0) for row in members),
        default=0.0,
    ) + 2.0 * np.log1p(len(supporting))
    return {
        "period_days": validated_period,
        "epoch_btjd": float(source["epoch_btjd"]),
        "duration_hours": float(source["duration_hours"]),
        "status": "harmonic_refinement_candidate",
        "supporting_branches": supporting,
        "method_families": ["harmonic_refinement"],
        "photometry_families": sorted(
            {row["photometry_family"] for row in evidence.values()}
        ),
        "holdout_supporting_branches": holdout_supporting,
        "uncalibrated_priority_score": float(score),
        "score_is_probability": False,
        "hard_rejected": False,
        "members": members,
        "all_branch_fixed_ephemeris_evidence": evidence,
        "candidate_source": "deterministic_harmonic_refinement",
        "conditioned_candidate": False,
        "harmonic_source": {
            "source_period_days": float(source["period_days"]),
            "factor": float(factor),
            "trial_period_days": float(period_days),
        },
        "strict_holdout_verification": dict(verification),
    }


def attach_sealed_validation_and_harmonics(
    system: dict,
    branches: Mapping[str, SearchBranch],
    *,
    period_min: float,
    period_max: float,
) -> dict:
    """Attach strict validation and append only sealed-passing harmonics."""
    holdout_segments = _segment_set(system.get("holdout_sectors"))
    if not holdout_segments:
        holdout_segments = _segment_set(system.get("holdout_sector"))
    clusters = [dict(row) for row in system.get("candidate_fusion", [])]
    primary = branches.get(PRIMARY_BRANCH)
    if primary is None:
        for cluster in clusters:
            cluster["strict_holdout_verification"] = {
                "passed": False,
                "outcome": "untestable",
                "testable": False,
                "status": "primary_branch_missing",
                "status_zh": "缺少 PDCSAP 主光变",
                "reasons_zh": ["无法执行严格封存验证"],
                "truth_catalog_used": False,
            }
        system["candidate_fusion"] = clusters
        return {"validated": 0, "rejected": len(clusters), "harmonics_added": 0}

    for cluster in clusters:
        primary_evidence = cluster.get("all_branch_fixed_ephemeris_evidence", {}).get(
            PRIMARY_BRANCH, {}
        )
        cluster["strict_holdout_verification"] = validate_locked_ephemeris(
            primary,
            holdout_segment=sorted(holdout_segments),
            period_days=float(cluster["period_days"]),
            epoch_btjd=float(cluster["epoch_btjd"]),
            duration_hours=float(cluster["duration_hours"]),
            search_metrics=primary_evidence.get("search"),
        )

    added: list[dict] = []
    if holdout_segments:
        original = list(clusters)
        for source in original:
            source_verification = source.get("strict_holdout_verification", {})
            if source_verification.get("passed"):
                continue
            source_primary = source.get(
                "all_branch_fixed_ephemeris_evidence", {}
            ).get(PRIMARY_BRANCH, {}).get("search", {})
            source_snr = max(0.0, _metric_snr(source_primary))
            for factor in HARMONIC_FACTORS:
                trial_period = float(source["period_days"]) * factor
                if not (float(period_min) <= trial_period <= float(period_max)):
                    continue
                proposal = {
                    "period_days": trial_period,
                    "epoch_btjd": float(source["epoch_btjd"]),
                    "duration_hours": float(source["duration_hours"]),
                }
                if _period_phase_duplicate(proposal, clusters + added):
                    continue
                search_mask = ~np.isin(primary.segment_id, list(holdout_segments))
                search_metrics = measure_fixed_ephemeris(
                    primary.time[search_mask],
                    primary.flux[search_mask],
                    period_days=trial_period,
                    epoch_btjd=proposal["epoch_btjd"],
                    duration_hours=proposal["duration_hours"],
                )
                if (
                    search_metrics.get("status") != "supported"
                    or int(search_metrics.get("events") or 0) < 2
                    or _metric_snr(search_metrics) < max(6.0, 0.95 * source_snr)
                ):
                    continue
                verification = validate_locked_ephemeris(
                    primary,
                    holdout_segment=sorted(holdout_segments),
                    period_days=trial_period,
                    epoch_btjd=proposal["epoch_btjd"],
                    duration_hours=proposal["duration_hours"],
                    search_metrics=search_metrics,
                )
                if not verification.get("passed"):
                    continue
                harmonic = _harmonic_cluster(
                    source,
                    branches,
                    holdout_segment=sorted(holdout_segments),
                    period_days=trial_period,
                    factor=factor,
                    verification=verification,
                )
                strong_search_branches = sum(
                    row.get("search", {}).get("status") == "supported"
                    for row in harmonic["all_branch_fixed_ephemeris_evidence"].values()
                )
                if strong_search_branches < 2:
                    continue
                added.append(harmonic)

    clusters.extend(added)
    clusters.sort(
        key=lambda row: float(row.get("uncalibrated_priority_score") or 0.0),
        reverse=True,
    )
    for rank, cluster in enumerate(clusters, start=1):
        cluster["rank"] = rank
    system["candidate_fusion"] = clusters
    passed = sum(
        bool(row.get("strict_holdout_verification", {}).get("passed")) for row in clusters
    )
    untestable = sum(
        row.get("strict_holdout_verification", {}).get("outcome") == "untestable"
        for row in clusters
    )
    inconclusive = sum(
        row.get("strict_holdout_verification", {}).get("outcome") == "inconclusive"
        for row in clusters
    )
    return {
        "input_clusters": len(clusters) - len(added),
        "output_clusters": len(clusters),
        "validated": passed,
        "rejected": len(clusters) - passed - untestable - inconclusive,
        "untestable": untestable,
        "inconclusive": inconclusive,
        "harmonics_added": len(added),
        "policy": {
            "primary_branch": PRIMARY_BRANCH,
            "sealed_period_grid_size": SEALED_PERIOD_GRID_SIZE,
            "harmonic_factors": list(HARMONIC_FACTORS),
            "depth_ratio_allowed": [MIN_DEPTH_RATIO, MAX_DEPTH_RATIO],
            "minimum_sealed_snr": MIN_SEALED_SNR,
            "minimum_weak_sealed_snr": MIN_WEAK_SEALED_SNR,
            "truth_catalog_used": False,
        },
    }
