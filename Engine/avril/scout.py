"""Fast, non-authoritative target triage for large Avril surveys.

Scout deliberately does *not* replace the production Avril search.  It uses
the exact same prepared/detrended light curve, makes a segment-aware linear
binning copy, and runs two complementary whole-baseline periodograms:

* a very fast BLS spectrum; and
* Transit Least Squares (TLS) when the optional dependency is installed.

The union of their independent local peaks is a routing hint: interesting
targets still run the complete, native-cadence Avril pipeline.  In particular,
Scout must never be used as the only route for sparse two-event or explicitly
requested long-period searches.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.timeseries import BoxLeastSquares
from scipy.signal import find_peaks

DEFAULT_DURATIONS_DAYS = (
    0.035,
    0.05,
    0.07,
    0.09,
    0.12,
    0.16,
    0.21,
    0.28,
    0.36,
)


@dataclass(frozen=True)
class ScoutConfig:
    """Configuration for the experimental fast-routing pass."""

    bin_minutes: float = 10.0
    max_bin_minutes: float = 30.0
    target_scout_cadences: int = 50_000
    period_min_days: float = 0.5
    period_max_days: float = 40.0
    bls_periods: int = 24_000
    max_peaks_per_channel: int = 64
    peak_separation_fraction: float = 0.002
    tls_threads: int = 2
    # Scout TLS is a fallback router, not the authoritative fit.  The native
    # Avril run recomputes the complete exact grid for every promoted target.
    tls_oversampling_factor: int = 1
    tls_duration_grid_step: float = 1.2
    local_refine_points: int = 81
    local_refine_fraction: float = 0.003
    event_threads: int = 2
    event_screen_limit: int = 64
    provisional_tls_sde: float = 9.0
    provisional_tls_snr: float = 5.0
    provisional_bls_robust_z: float = 12.0

    def __post_init__(self) -> None:
        if self.bin_minutes <= 0:
            raise ValueError("bin_minutes must be positive")
        if self.max_bin_minutes < self.bin_minutes:
            raise ValueError("max_bin_minutes must be at least bin_minutes")
        if self.target_scout_cadences < 1_000:
            raise ValueError("target_scout_cadences must be at least 1000")
        if self.period_min_days <= 0 or self.period_min_days >= self.period_max_days:
            raise ValueError("period_min_days must be positive and below period_max_days")
        if self.bls_periods < 1_000:
            raise ValueError("bls_periods must be at least 1000")
        if self.max_peaks_per_channel < 1:
            raise ValueError("max_peaks_per_channel must be positive")
        if self.tls_threads < 1:
            raise ValueError("tls_threads must be positive")
        if self.local_refine_points < 11 or self.local_refine_points % 2 == 0:
            raise ValueError("local_refine_points must be an odd integer of at least 11")
        if self.local_refine_fraction <= 0:
            raise ValueError("local_refine_fraction must be positive")
        if self.event_threads < 1:
            raise ValueError("event_threads must be positive")
        if self.event_screen_limit < 1:
            raise ValueError("event_screen_limit must be positive")


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 0.0
    median = float(np.median(finite))
    scale = 1.4826 * float(np.median(np.abs(finite - median)))
    return median, scale if math.isfinite(scale) and scale > 0 else 0.0


def bin_prepared_curve(
    time_values: Sequence[float],
    flux_values: Sequence[float],
    segment_ids: Sequence[int],
    *,
    bin_minutes: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-bin an already prepared curve without ever crossing a segment.

    Binning happens after Avril's normal one-day detrending.  It is therefore
    a disposable Scout view, not a change to the production photometry.
    """

    time_array = np.asarray(time_values, dtype=np.float64)
    flux_array = np.asarray(flux_values, dtype=np.float64)
    segment_array = np.asarray(segment_ids, dtype=np.int64)
    good = np.isfinite(time_array) & np.isfinite(flux_array)
    time_array = time_array[good]
    flux_array = flux_array[good]
    segment_array = segment_array[good]
    if time_array.size == 0:
        raise ValueError("Scout received no finite prepared cadences")

    width_days = float(bin_minutes) / 1440.0
    binned_time: list[np.ndarray] = []
    binned_flux: list[np.ndarray] = []
    binned_segment: list[np.ndarray] = []
    for segment in np.unique(segment_array):
        mask = segment_array == segment
        local_time = time_array[mask]
        local_flux = flux_array[mask]
        order = np.argsort(local_time)
        local_time = local_time[order]
        local_flux = local_flux[order]
        bins = np.floor((local_time - float(local_time[0])) / width_days).astype(
            np.int64
        )
        counts = np.bincount(bins).astype(np.float64)
        occupied = counts > 0
        means_time = np.bincount(bins, weights=local_time)[occupied] / counts[occupied]
        means_flux = np.bincount(bins, weights=local_flux)[occupied] / counts[occupied]
        binned_time.append(means_time)
        binned_flux.append(means_flux)
        binned_segment.append(np.full(means_time.size, int(segment), dtype=np.int64))

    output_time = np.concatenate(binned_time)
    output_flux = np.concatenate(binned_flux)
    output_segment = np.concatenate(binned_segment)
    order = np.argsort(output_time)
    return output_time[order], output_flux[order], output_segment[order]


def _rank_local_peaks(
    periods: np.ndarray,
    power: np.ndarray,
    *,
    source: str,
    limit: int,
    separation_fraction: float,
) -> list[dict]:
    periods = np.asarray(periods, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)
    valid = np.isfinite(periods) & np.isfinite(power) & (periods > 0)
    periods = periods[valid]
    power = power[valid]
    if power.size < 3:
        return []
    peak_indices, _ = find_peaks(power)
    if peak_indices.size == 0:
        peak_indices = np.asarray([int(np.nanargmax(power))], dtype=np.int64)
    ordered = peak_indices[np.argsort(power[peak_indices])[::-1]]
    median, scatter = _robust_location_scale(power)
    rows: list[dict] = []
    for raw_index in ordered:
        index = int(raw_index)
        period = float(periods[index])
        if any(
            abs(period - float(old["period_days"])) / float(old["period_days"])
            < separation_fraction
            for old in rows
        ):
            continue
        rows.append(
            {
                "source": source,
                "source_rank": len(rows) + 1,
                "period_days": period,
                "power": float(power[index]),
                "robust_power_z": (
                    float((power[index] - median) / scatter) if scatter > 0 else None
                ),
                "power_percentile": float(100.0 * np.mean(power <= power[index])),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def run_bls_scout(
    time_values: np.ndarray,
    flux_values: np.ndarray,
    config: ScoutConfig,
) -> dict:
    """Run one non-destructive whole-baseline BLS spectrum."""

    periods = np.geomspace(
        config.period_min_days,
        config.period_max_days,
        config.bls_periods,
        dtype=np.float64,
    )
    durations = np.asarray(
        [
            value
            for value in DEFAULT_DURATIONS_DAYS
            if value < config.period_min_days
        ],
        dtype=np.float64,
    )
    centered = np.asarray(flux_values, dtype=np.float64) - float(
        np.median(flux_values)
    )
    started = time.monotonic()
    solved = BoxLeastSquares(
        np.asarray(time_values, dtype=np.float64) * u.day,
        centered,
    ).power(periods * u.day, durations * u.day, method="fast")
    elapsed = time.monotonic() - started
    power = np.asarray(solved.power, dtype=np.float64)
    rows = _rank_local_peaks(
        periods,
        power,
        source="bls",
        limit=config.max_peaks_per_channel,
        separation_fraction=config.peak_separation_fraction,
    )
    return {
        "status": "complete",
        "seconds": round(elapsed, 6),
        "period_grid_size": int(periods.size),
        "maximum_robust_power_z": rows[0].get("robust_power_z") if rows else None,
        "peaks": rows,
    }


def run_tls_scout(
    time_values: np.ndarray,
    flux_values: np.ndarray,
    config: ScoutConfig,
) -> dict:
    """Run TLS 2 as an optional complementary Scout channel."""

    try:
        from transitleastsquares import transitleastsquares
    except ImportError as exc:  # pragma: no cover - environment-dependent path
        return {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "install": "pip install 'avril-transit-search[scout]'",
            "peaks": [],
        }

    started = time.monotonic()
    model = transitleastsquares(
        np.asarray(time_values, dtype=np.float64),
        np.asarray(flux_values, dtype=np.float64),
        verbose=False,
    )
    solved = model.power(
        period_min=config.period_min_days,
        period_max=config.period_max_days,
        use_threads=config.tls_threads,
        show_progress_bar=False,
        verbose=False,
        oversampling_factor=config.tls_oversampling_factor,
        duration_grid_step=config.tls_duration_grid_step,
    )
    elapsed = time.monotonic() - started
    periods = np.asarray(solved.periods, dtype=np.float64)
    power = np.asarray(solved.power, dtype=np.float64)
    rows = _rank_local_peaks(
        periods,
        power,
        source="tls",
        limit=config.max_peaks_per_channel,
        separation_fraction=config.peak_separation_fraction,
    )
    return {
        "status": "complete",
        "seconds": round(elapsed, 6),
        "period_grid_size": int(periods.size),
        "best_period_days": float(solved.period),
        "sde": float(solved.SDE),
        "snr": float(solved.snr),
        "peaks": rows,
    }


def merge_channel_peaks(
    channels: Iterable[Mapping],
    *,
    match_fraction: float = 0.002,
) -> list[dict]:
    """Merge only near-identical peaks; harmonics remain separate evidence."""

    ordered: list[dict] = []
    for channel in channels:
        for source_row in channel.get("peaks", []):
            row = dict(source_row)
            row["sources"] = [str(row.pop("source"))]
            row["source_ranks"] = {
                row["sources"][0]: int(row.pop("source_rank"))
            }
            matched = None
            for existing in ordered:
                reference = float(existing["period_days"])
                if abs(float(row["period_days"]) - reference) / reference < match_fraction:
                    matched = existing
                    break
            if matched is None:
                ordered.append(row)
                continue
            source = row["sources"][0]
            matched["sources"].append(source)
            matched["source_ranks"][source] = row["source_ranks"][source]
            if float(row.get("power_percentile") or 0.0) > float(
                matched.get("power_percentile") or 0.0
            ):
                # The period is a routing hint only; prefer the stronger
                # channel's location without pretending it is a final fit.
                matched["period_days"] = float(row["period_days"])
                matched["power"] = float(row["power"])
                matched["robust_power_z"] = row.get("robust_power_z")
                matched["power_percentile"] = row.get("power_percentile")

    for row in ordered:
        row["sources"] = sorted(set(row["sources"]))
        row["channel_support_count"] = len(row["sources"])
        row["routing_score"] = max(
            (
                100.0 - float(rank)
                for rank in row["source_ranks"].values()
            ),
            default=0.0,
        ) + 5.0 * (row["channel_support_count"] - 1)
    ordered.sort(
        key=lambda row: (
            int(row["channel_support_count"]),
            float(row["routing_score"]),
            float(row.get("power_percentile") or 0.0),
        ),
        reverse=True,
    )
    for index, row in enumerate(ordered, start=1):
        row["scout_rank"] = index
    return ordered


def refine_scout_ephemerides(
    time_values: np.ndarray,
    flux_values: np.ndarray,
    candidates: Sequence[Mapping],
    config: ScoutConfig,
) -> tuple[list[dict], float]:
    """Solve a small local BLS grid for every routing peak in one batch.

    TLS exposes the full period spectrum but only one global epoch/duration.
    This shared refinement supplies comparable ephemerides for catalogue and
    event-backed harmonic checks.  It remains a Scout estimate; native-cadence
    Avril independently refits every promoted target.
    """

    if not candidates:
        return [], 0.0
    grids = []
    slices: list[slice] = []
    cursor = 0
    for candidate in candidates:
        seed = float(candidate["period_days"])
        half_width = config.local_refine_fraction * seed
        grid = np.linspace(
            max(config.period_min_days, seed - half_width),
            min(config.period_max_days, seed + half_width),
            config.local_refine_points,
            dtype=np.float64,
        )
        grids.append(grid)
        slices.append(slice(cursor, cursor + grid.size))
        cursor += grid.size
    periods = np.concatenate(grids)
    durations = np.asarray(
        [
            value
            for value in DEFAULT_DURATIONS_DAYS
            if value < config.period_min_days
        ],
        dtype=np.float64,
    )
    centered = np.asarray(flux_values, dtype=np.float64) - float(
        np.median(flux_values)
    )
    started = time.monotonic()
    solved = BoxLeastSquares(
        np.asarray(time_values, dtype=np.float64) * u.day,
        centered,
    ).power(periods * u.day, durations * u.day, method="fast")
    elapsed = time.monotonic() - started
    power = np.asarray(solved.power, dtype=np.float64)
    epochs = np.asarray(solved.transit_time.to_value(u.day), dtype=np.float64)
    solved_durations = np.asarray(solved.duration.to_value(u.day), dtype=np.float64)
    refined: list[dict] = []
    for index, (candidate, local_slice) in enumerate(
        zip(candidates, slices, strict=True), start=1
    ):
        local_power = power[local_slice]
        offset = int(np.nanargmax(local_power))
        solved_index = int(local_slice.start or 0) + offset
        row = dict(candidate)
        row.update(
            {
                "proposal_id": f"SCOUT-{index:04d}",
                "initial_period_days": float(candidate["period_days"]),
                "period_days": float(periods[solved_index]),
                "epoch_btjd": float(epochs[solved_index]),
                "duration_hours": float(solved_durations[solved_index] * 24.0),
                "local_refine_power": float(power[solved_index]),
            }
        )
        refined.append(row)
    return refined, elapsed


def _period_relation(first_period: float, second_period: float) -> dict | None:
    """Return a direct near-equality or plausible fold relation."""

    shorter = min(float(first_period), float(second_period))
    longer = max(float(first_period), float(second_period))
    if shorter <= 0:
        return None
    same_error = abs(longer - shorter) / shorter
    if same_error <= 0.002:
        return {"type": "same_period", "relative_error": same_error}
    ratio = longer / shorter
    integer = int(round(ratio))
    if 2 <= integer <= 200:
        relative_error = abs(ratio - integer) / integer
        if relative_error <= 0.001:
            return {
                "type": "integer_harmonic",
                "numerator": integer,
                "denominator": 1,
                "relative_error": relative_error,
            }
    fraction = Fraction(ratio).limit_denominator(16)
    if 2 <= fraction.numerator <= 16:
        approximate = float(fraction)
        relative_error = abs(ratio - approximate) / approximate
        if relative_error <= 1e-4:
            return {
                "type": "low_order_rational_harmonic",
                "numerator": int(fraction.numerator),
                "denominator": int(fraction.denominator),
                "relative_error": relative_error,
            }
    return None


def _candidate_supported_events(candidate: Mapping) -> list[dict]:
    repeatability = (
        candidate.get("holdout_verification", {}).get("event_repeatability", {})
    )
    return [
        dict(row)
        for row in repeatability.get("event_support", [])
        if row.get("supported") is True
    ]


def _catalog_relation(candidate: Mapping, catalog_row: Mapping) -> dict | None:
    relation = _period_relation(
        float(candidate["period_days"]), float(catalog_row["period"])
    )
    if relation is not None and relation["type"] == "same_period":
        return {**relation, "event_evidence_required": False}
    epoch = catalog_row.get("epochBtjd")
    if epoch is None:
        return None
    events = _candidate_supported_events(candidate)
    if len(events) < 3:
        return None
    known_period = float(catalog_row["period"])
    known_epoch = float(epoch)
    tolerance = max(float(candidate["duration_hours"]) / 24.0, 0.04)
    matched_rows = []
    for event in events:
        center = float(event["center_btjd"])
        distance = abs(
            (center - known_epoch + 0.5 * known_period) % known_period
            - 0.5 * known_period
        )
        if distance <= tolerance:
            matched_rows.append(event)
    segments = sorted({int(row["segment"]) for row in matched_rows})
    containment = len(matched_rows) / len(events)
    # A candidate can be a cycle-count/side-band representation of a known
    # event train without sitting on a clean integer period ratio.  That is
    # still a known-signal alias, but only when nearly all of its supported
    # events actually coincide with the locked catalogue ephemeris.
    minimum_containment = 0.8 if relation is None else 0.5
    if (
        len(matched_rows) < 3
        or len(segments) < 2
        or containment < minimum_containment
    ):
        return None
    if relation is None:
        relation = {
            "type": "known_event_train_subset",
            "relative_error": None,
        }
    return {
        **relation,
        "event_evidence_required": True,
        "matched_supported_events": len(matched_rows),
        "matched_segments": segments,
        "supported_event_containment": containment,
    }


def _binomial_direction_probability(testable: int, positive: int) -> float:
    """One-sided probability of at least ``positive`` downward events."""

    if testable <= 0 or positive <= testable / 2:
        return 1.0
    return float(
        sum(math.comb(testable, count) for count in range(positive, testable + 1))
        / (2**testable)
    )


def _quick_candidate_evidence(candidate: Mapping) -> dict:
    """Summarise cheap evidence without pretending to perform full vetting.

    Individual events are *not* all required to exceed 3 sigma.  For repeated
    signals we also accept a significant excess of downward event depths and
    their joint quadrature S/N.  Sparse two-to-four-event cases get a separate
    follow-up route and are never described as validated planets.
    """

    repeatability = (
        candidate.get("holdout_verification", {}).get("event_repeatability", {})
    )
    events = [dict(row) for row in repeatability.get("event_support", [])]
    testable = len(events)
    positive_rows = [row for row in events if float(row.get("depth_ppm") or 0.0) > 0]
    supported_rows = [row for row in events if row.get("supported") is True]
    positive = len(positive_rows)
    supported = len(supported_rows)
    positive_fraction = positive / testable if testable else None
    supported_fraction = supported / testable if testable else None
    direction_probability = _binomial_direction_probability(testable, positive)
    joint_positive_snr = float(
        math.sqrt(
            sum(max(0.0, float(row.get("snr") or 0.0)) ** 2 for row in positive_rows)
        )
    )
    positive_segments = sorted({int(row["segment"]) for row in positive_rows})
    supported_segments = sorted({int(row["segment"]) for row in supported_rows})
    source_ranks = candidate.get("source_ranks") or {}
    minimum_source_rank = min(
        (int(value) for value in source_ranks.values()), default=10**9
    )
    channel_support = int(candidate.get("channel_support_count") or 0)
    channel_evidence = channel_support >= 2 or minimum_source_rank <= 16

    repeated_direction = bool(
        testable >= 5
        and positive_fraction is not None
        and positive_fraction >= 0.6
        and direction_probability <= 0.05
        and joint_positive_snr >= 7.0
        and len(positive_segments) >= 2
    )
    repeated_supported = bool(
        testable >= 3
        and supported >= 3
        and supported_fraction is not None
        and supported_fraction >= 0.6
        and positive_fraction is not None
        and positive_fraction >= 0.6
        and len(supported_segments) >= 2
    )
    sparse_followup = bool(
        2 <= testable <= 4
        and positive == testable
        and joint_positive_snr >= 7.0
        and len(positive_segments) >= 2
    )
    event_evidence_passed = bool(
        channel_evidence
        and (repeated_direction or repeated_supported or sparse_followup)
    )
    if event_evidence_passed and sparse_followup:
        evidence_class = "event_supported_sparse_followup"
    elif event_evidence_passed:
        evidence_class = "event_supported_repeated_candidate"
    elif testable:
        evidence_class = "insufficient_quick_evidence"
    else:
        evidence_class = "not_event_screened"
    return {
        "class": evidence_class,
        "quick_event_evidence_passed": event_evidence_passed,
        "channel_evidence": channel_evidence,
        "channel_support_count": channel_support,
        "minimum_source_rank": (
            minimum_source_rank if minimum_source_rank < 10**9 else None
        ),
        "testable_events": testable,
        "positive_events": positive,
        "supported_events": supported,
        "positive_event_fraction": positive_fraction,
        "supported_event_fraction": supported_fraction,
        "downward_direction_probability": direction_probability,
        "joint_positive_event_snr": joint_positive_snr,
        "positive_segments": positive_segments,
        "supported_segments": supported_segments,
        "repeated_direction_passed": repeated_direction,
        "repeated_supported_passed": repeated_supported,
        "sparse_followup_passed": sparse_followup,
        "interpretation": (
            "routing evidence only; full red-noise, gap-edge, binary, "
            "centroid, and contamination vetting remains mandatory"
        ),
    }


def _compact_repeatability(candidate: dict) -> dict:
    output = dict(candidate)
    output.pop("_harmonic_supported_events_cache", None)
    output.pop("_harmonic_positive_events_cache", None)
    output.pop("_validated_event_centers_by_segment", None)
    verification = dict(output.get("holdout_verification") or {})
    repeatability = dict(verification.get("event_repeatability") or {})
    if repeatability:
        repeatability.pop("event_support", None)
        repeatability.pop("unsupported_event_centers_btjd", None)
        repeatability.pop("modulo", None)
        verification["event_repeatability"] = repeatability
        output["holdout_verification"] = verification
    return output


def postfilter_scout_candidates(
    time_values: np.ndarray,
    flux_values: np.ndarray,
    segment_ids: np.ndarray,
    candidates: Sequence[Mapping],
    *,
    config: ScoutConfig,
    catalog: Sequence[Mapping] = (),
) -> dict:
    """Refine, family-compress, and catalogue-route Scout peaks.

    Every raw routing peak remains in ``search_candidates``.  Only direct
    event-backed harmonic decisions affect ``canonical_candidates``.
    Catalogue truth is attached after search and never changes a periodogram.
    """

    from multiplanet_trial.live_web.compiled_event_repeatability import (
        attach_event_repeatability_batch,
    )
    from multiplanet_trial.live_web.harmonic_families import (
        _commensurate_candidate_pairs,
        arbitrate_harmonic_families,
    )
    from multiplanet_trial.live_web.sealed_validation import (
        prepare_event_repeatability_views,
    )
    from multiplanet_trial.v07_eval.multibranch_search import SearchBranch

    started = time.monotonic()
    refined, refine_seconds = refine_scout_ephemerides(
        time_values, flux_values, candidates, config
    )
    pair_indices = _commensurate_candidate_pairs(refined)
    event_indices = {index for pair in pair_indices for index in pair}
    # Screen the highest-ranked union peaks even when they have no simple
    # arithmetic relation to another peak.  This turns a periodogram maximum
    # into an event-direction audit while keeping the work bounded.
    event_indices.update(range(min(len(refined), config.event_screen_limit)))
    for index, candidate in enumerate(refined):
        if any(
            _period_relation(
                float(candidate["period_days"]), float(answer["period"])
            )
            is not None
            for answer in catalog
        ):
            event_indices.add(index)

    event_seconds = 0.0
    if event_indices:
        branch = SearchBranch(
            key="scout-native-prepared",
            label="Scout native prepared view",
            processing="unchanged Avril prepared photometry",
            photometry_family="scout-routing-only",
            method_family="bls+tls",
            time=np.asarray(time_values, dtype=np.float64),
            flux=np.asarray(flux_values, dtype=np.float64),
            segment_id=np.asarray(segment_ids, dtype=np.int64),
        )
        prepared_views = prepare_event_repeatability_views(branch)
        subset = [refined[index] for index in sorted(event_indices)]
        event_started = time.monotonic()
        attached = attach_event_repeatability_batch(
            subset,
            prepared_views=prepared_views,
            threads=config.event_threads,
        )
        if attached is None:  # pragma: no cover - numba-free optional path
            raise RuntimeError("Compiled event repeatability is unavailable")
        event_seconds = time.monotonic() - event_started

    def quality(row: dict) -> tuple:
        return (
            int(row.get("channel_support_count") or 0),
            float(row.get("routing_score") or 0.0),
            float(row.get("power_percentile") or 0.0),
            -int(row.get("scout_rank") or 10**9),
        )

    canonical, aliases, relations, families = arbitrate_harmonic_families(
        refined, quality=quality
    )
    canonical.sort(key=quality, reverse=True)
    known: list[dict] = []
    unexplained: list[dict] = []
    event_supported_unexplained: list[dict] = []
    catalog_audit: list[dict] = []
    for candidate in canonical:
        matches = []
        for answer in catalog:
            relation = _catalog_relation(candidate, answer)
            if relation is not None:
                matches.append(
                    {
                        "catalog": dict(answer),
                        "relation": relation,
                    }
                )
        decorated = dict(candidate)
        decorated["quick_evidence"] = _quick_candidate_evidence(decorated)
        if matches:
            matches.sort(
                key=lambda row: (
                    row["relation"]["type"] != "same_period",
                    float(row["relation"].get("relative_error") or 0.0),
                )
            )
            decorated["catalog_relation"] = matches[0]
            decorated["routing_disposition"] = (
                "known_catalog_period"
                if matches[0]["relation"]["type"] == "same_period"
                else "event_proven_known_catalog_alias"
            )
            known.append(decorated)
            catalog_audit.append(
                {
                    "proposal_id": decorated["proposal_id"],
                    "period_days": decorated["period_days"],
                    **matches[0],
                }
            )
        else:
            decorated["catalog_relation"] = None
            if decorated["quick_evidence"]["quick_event_evidence_passed"]:
                decorated["routing_disposition"] = (
                    "unexplained_event_supported_scout_peak"
                )
                event_supported_unexplained.append(decorated)
            else:
                decorated["routing_disposition"] = (
                    "unexplained_below_quick_evidence_gate"
                )
            unexplained.append(decorated)

    return {
        "search_candidates": [_compact_repeatability(row) for row in refined],
        "canonical_candidates": [
            _compact_repeatability(row) for row in known + unexplained
        ],
        "known_or_alias_candidates": [
            _compact_repeatability(row) for row in known
        ],
        "unexplained_candidates": [
            _compact_repeatability(row) for row in unexplained
        ],
        "event_supported_unexplained_candidates": [
            _compact_repeatability(row) for row in event_supported_unexplained
        ],
        "harmonic_alias_audit": [
            _compact_repeatability(row) for row in aliases
        ],
        "harmonic_relations": relations,
        "harmonic_families": families,
        "catalog_relation_audit": catalog_audit,
        "catalog": [dict(row) for row in catalog],
        "timing_seconds": {
            "local_ephemeris_refine": round(refine_seconds, 6),
            "event_backed_family_evidence": round(event_seconds, 6),
            "total": round(time.monotonic() - started, 6),
        },
    }


def mask_locked_ephemerides(
    time_values: np.ndarray,
    flux_values: np.ndarray,
    segment_ids: np.ndarray,
    candidates: Sequence[Mapping],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Mask known transit windows in the disposable Scout view only."""

    time_array = np.asarray(time_values, dtype=np.float64)
    keep = np.ones(time_array.size, dtype=bool)
    rows = []
    for candidate in candidates:
        period = float(candidate["period_days"])
        epoch = float(candidate["epoch_btjd"])
        duration_days = float(candidate["duration_hours"]) / 24.0
        # Match the production iterative-search mask: one full fitted duration
        # on either side, with a conservative one-hour minimum.
        half_window = max(duration_days, 0.04)
        phase = (time_array - epoch + 0.5 * period) % period - 0.5 * period
        local = np.abs(phase) <= half_window
        newly_masked = int(np.count_nonzero(keep & local))
        keep &= ~local
        rows.append(
            {
                "proposal_id": candidate.get("proposal_id"),
                "period_days": period,
                "epoch_btjd": epoch,
                "duration_hours": float(candidate["duration_hours"]),
                "cadences_masked": newly_masked,
            }
        )
    return (
        time_array[keep],
        np.asarray(flux_values, dtype=np.float64)[keep],
        np.asarray(segment_ids, dtype=np.int64)[keep],
        {
            "input_cadences": int(time_array.size),
            "remaining_cadences": int(np.count_nonzero(keep)),
            "masked_cadences": int(np.count_nonzero(~keep)),
            "masked_fraction": float(np.mean(~keep)),
            "ephemerides": rows,
        },
    )


def scan_prepared_curve(
    time_values: Sequence[float],
    flux_values: Sequence[float],
    segment_ids: Sequence[int],
    *,
    config: ScoutConfig | None = None,
    catalog: Sequence[Mapping] = (),
) -> dict:
    """Run both Scout channels on an already prepared Avril light curve."""

    config = config or ScoutConfig()
    total_started = time.monotonic()
    requested_bin_minutes = float(config.bin_minutes)
    effective_bin_minutes = requested_bin_minutes
    binned_time, binned_flux, binned_segment = bin_prepared_curve(
        time_values,
        flux_values,
        segment_ids,
        bin_minutes=effective_bin_minutes,
    )
    if binned_time.size > config.target_scout_cadences:
        scaled_minutes = requested_bin_minutes * (
            binned_time.size / float(config.target_scout_cadences)
        )
        effective_bin_minutes = min(float(config.max_bin_minutes), scaled_minutes)
        if effective_bin_minutes > requested_bin_minutes * 1.01:
            binned_time, binned_flux, binned_segment = bin_prepared_curve(
                time_values,
                flux_values,
                segment_ids,
                bin_minutes=effective_bin_minutes,
            )
    bls = run_bls_scout(binned_time, binned_flux, config)

    def skipped_tls(reason: str) -> dict:
        return {"status": "skipped", "reason": reason, "peaks": []}

    # BLS is the inexpensive router.  TLS is complementary fallback evidence,
    # not mandatory work once BLS plus repeated events have already proved
    # that the target must run full Avril.
    tls = skipped_tls("BLS-first routing has not requested TLS")
    routing_peaks = merge_channel_peaks(
        (bls,), match_fraction=config.peak_separation_fraction
    )
    initial_postsearch = postfilter_scout_candidates(
        np.asarray(time_values, dtype=np.float64),
        np.asarray(flux_values, dtype=np.float64),
        np.asarray(segment_ids, dtype=np.int64),
        routing_peaks,
        config=config,
        catalog=catalog,
    )
    active_bls = bls
    active_tls = tls
    active_postsearch = initial_postsearch
    residual_scan = None
    direct_known = [
        row
        for row in initial_postsearch["known_or_alias_candidates"]
        if (row.get("catalog_relation") or {}).get("relation", {}).get("type")
        == "same_period"
    ]
    if direct_known:
        residual_time, residual_flux, residual_segment, mask_diagnostics = (
            mask_locked_ephemerides(
                binned_time, binned_flux, binned_segment, direct_known
            )
        )
        if residual_time.size >= 500:
            residual_started = time.monotonic()
            residual_bls = run_bls_scout(residual_time, residual_flux, config)
            residual_tls = skipped_tls(
                "Residual BLS evidence is evaluated before TLS fallback"
            )
            residual_peaks = merge_channel_peaks(
                (residual_bls,),
                match_fraction=config.peak_separation_fraction,
            )
            residual_postsearch = postfilter_scout_candidates(
                np.asarray(time_values, dtype=np.float64),
                np.asarray(flux_values, dtype=np.float64),
                np.asarray(segment_ids, dtype=np.int64),
                residual_peaks,
                config=config,
                catalog=catalog,
            )
            residual_scan = {
                "mask": mask_diagnostics,
                "channels": {"bls": residual_bls, "tls": residual_tls},
                "postsearch": residual_postsearch,
                "seconds": round(time.monotonic() - residual_started, 6),
            }
            active_bls = residual_bls
            active_tls = residual_tls
            active_postsearch = residual_postsearch
    active_bls_z = float(
        active_bls.get("maximum_robust_power_z") or -math.inf
    )
    active_bls_strong = bool(
        active_bls_z >= config.provisional_bls_robust_z
    )
    active_event_supported = active_postsearch[
        "event_supported_unexplained_candidates"
    ]
    tls_needed = not (active_bls_strong and active_event_supported)
    if tls_needed:
        if residual_scan is not None:
            tls_time = residual_time
            tls_flux = residual_flux
        else:
            tls_time = binned_time
            tls_flux = binned_flux
        active_tls = run_tls_scout(tls_time, tls_flux, config)
        combined_peaks = merge_channel_peaks(
            (active_bls, active_tls),
            match_fraction=config.peak_separation_fraction,
        )
        active_postsearch = postfilter_scout_candidates(
            np.asarray(time_values, dtype=np.float64),
            np.asarray(flux_values, dtype=np.float64),
            np.asarray(segment_ids, dtype=np.int64),
            combined_peaks,
            config=config,
            catalog=catalog,
        )
        if residual_scan is not None:
            residual_scan["channels"]["tls"] = active_tls
            residual_scan["postsearch"] = active_postsearch
            residual_scan["seconds"] = round(
                time.monotonic() - residual_started, 6
            )
        else:
            tls = active_tls
    else:
        active_tls = skipped_tls(
            "BLS and repeated-event evidence already require full Avril"
        )
        if residual_scan is not None:
            residual_scan["channels"]["tls"] = active_tls
    tls_sde = float(active_tls.get("sde") or -math.inf)
    tls_snr = float(active_tls.get("snr") or -math.inf)
    bls_z = float(active_bls.get("maximum_robust_power_z") or -math.inf)
    tls_strong = bool(
        tls_sde >= config.provisional_tls_sde
        and tls_snr >= config.provisional_tls_snr
    )
    bls_strong = bool(bls_z >= config.provisional_bls_robust_z)
    event_supported_unexplained = active_postsearch[
        "event_supported_unexplained_candidates"
    ]
    significant_unexplained = (
        event_supported_unexplained if (tls_strong or bls_strong) else []
    )
    provisional_escalate = bool(
        (tls_strong or bls_strong) and significant_unexplained
    )
    return {
        "schema_version": "avril-scout-v0-experimental",
        "authoritative": False,
        "config": asdict(config),
        "input": {
            "native_prepared_cadences": int(len(time_values)),
            "scout_cadences": int(binned_time.size),
            "requested_bin_minutes": requested_bin_minutes,
            "effective_bin_minutes": effective_bin_minutes,
            "segments": [int(value) for value in np.unique(binned_segment)],
            "baseline_days": float(np.ptp(binned_time)),
        },
        "channels": {"bls": bls, "tls": tls},
        "initial_search_candidates": initial_postsearch["search_candidates"],
        "search_candidates": active_postsearch["search_candidates"],
        "candidates": active_postsearch["canonical_candidates"],
        "known_or_alias_candidates": initial_postsearch[
            "known_or_alias_candidates"
        ],
        "unexplained_candidates": active_postsearch["unexplained_candidates"],
        "event_supported_unexplained_candidates": event_supported_unexplained,
        "significant_unexplained_candidates": significant_unexplained,
        "initial_postsearch": {
            key: value
            for key, value in initial_postsearch.items()
            if key
            not in {
                "search_candidates",
                "canonical_candidates",
                "known_or_alias_candidates",
                "unexplained_candidates",
                "event_supported_unexplained_candidates",
            }
        },
        "postsearch": {
            key: value
            for key, value in active_postsearch.items()
            if key
            not in {
                "search_candidates",
                "canonical_candidates",
                "known_or_alias_candidates",
                "unexplained_candidates",
                "event_supported_unexplained_candidates",
            }
        },
        "residual_scan": residual_scan,
        "routing": {
            "provisional_escalate_to_full_avril": provisional_escalate,
            "tls_threshold_passed": tls_strong,
            "bls_threshold_passed": bls_strong,
            "significant_unexplained_count": len(significant_unexplained),
            "event_supported_unexplained_count": len(
                event_supported_unexplained
            ),
            "production_gate_enabled": False,
            "reason": (
                (
                    "Residual search crossed an experimental channel threshold "
                    "and retained event-backed unexplained candidates after "
                    "known-signal masking"
                    if residual_scan is not None
                    else (
                        "At least one fast channel crossed its experimental "
                        "threshold and retained event-backed unexplained candidates"
                    )
                )
                if provisional_escalate
                else (
                    "Residual search did not retain an event-backed unexplained "
                    "candidate above an experimental channel threshold after "
                    "known-signal masking"
                    if residual_scan is not None
                    else (
                        "No event-backed unexplained candidate remained above "
                        "an experimental channel threshold"
                    )
                )
            ),
            "decision_spectrum": (
                "known-signal-masked residual"
                if residual_scan is not None
                else "initial whole-baseline"
            ),
            "mandatory_bypass": [
                "explicit long-period request",
                "sparse one/two-event trigger",
                "known ephemeris regression",
                "injection/recovery validation",
            ],
        },
        "timing_seconds": {"total": round(time.monotonic() - total_started, 6)},
    }


def scan_lightcurve_paths(
    lightcurve_paths: Sequence[str | Path],
    *,
    config: ScoutConfig | None = None,
    tic_id: int | None = None,
    query_catalog: bool = False,
) -> dict:
    """Prepare cached SPOC files exactly as Avril does, then run Scout."""

    from multiplanet_trial.live_web.pipeline import _load_pdcsap_branch

    paths = [Path(value).expanduser().resolve() for value in lightcurve_paths]
    if not paths:
        raise ValueError("No light-curve paths were supplied")
    load_started = time.monotonic()
    branch, diagnostics = _load_pdcsap_branch(paths)
    load_seconds = time.monotonic() - load_started
    catalog: Sequence[Mapping] = ()
    if query_catalog and tic_id is not None:
        from multiplanet_trial.live_web.pipeline import _query_toi_catalog

        catalog = _query_toi_catalog(int(tic_id))
    result = scan_prepared_curve(
        branch.time,
        branch.flux,
        branch.segment_id,
        config=config,
        catalog=catalog,
    )
    result["input"].update(
        {
            "lightcurve_files": [str(path) for path in paths],
            "load_diagnostics": diagnostics,
        }
    )
    result["timing_seconds"]["prepare"] = round(load_seconds, 6)
    result["timing_seconds"]["wall_with_prepare"] = round(
        load_seconds + float(result["timing_seconds"]["total"]), 6
    )
    return result
