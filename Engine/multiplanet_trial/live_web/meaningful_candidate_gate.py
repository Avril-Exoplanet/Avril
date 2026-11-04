"""Cheap, auditable discovery funnel before expensive Avril V3 vetting.

The discovery ledger is immutable: this module never deletes a candidate or
changes a locked ephemeris.  It only decides which candidates can still be an
*unexplained* transit signal worth red-noise, physical, and pixel follow-up.
Known catalogue events, event-proven aliases, incoherent event trains, and
signals dominated by compromised gap-edge events remain in the audit output
with an explicit short-circuit reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction

import numpy as np

from .compiled_event_repeatability import event_repeatability_metrics_batch
from .detection_significance import _event_gap_diagnostics, _segment_views
from .fast_science_finalize import candidate_key, eligible_for_physical_vetting
from .sealed_validation import prepare_event_repeatability_views


def _period_relation(first_period: float, second_period: float) -> dict | None:
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
        error = abs(ratio - integer) / integer
        if error <= 0.001:
            return {
                "type": "integer_harmonic",
                "numerator": integer,
                "denominator": 1,
                "relative_error": error,
            }
    fraction = Fraction(ratio).limit_denominator(16)
    approximate = float(fraction)
    error = abs(ratio - approximate) / approximate
    if 2 <= fraction.numerator <= 16 and error <= 1e-4:
        return {
            "type": "low_order_rational_harmonic",
            "numerator": int(fraction.numerator),
            "denominator": int(fraction.denominator),
            "relative_error": error,
        }
    return None


def _catalog_relation(candidate: Mapping, catalog_row: Mapping) -> dict | None:
    relation = _period_relation(
        float(candidate["period_days"]), float(catalog_row["period"])
    )
    if relation is not None and relation["type"] == "same_period":
        return {**relation, "event_evidence_required": False}
    epoch = catalog_row.get("epochBtjd")
    if epoch is None:
        return None
    repeatability = (
        candidate.get("holdout_verification", {}).get("event_repeatability", {})
    )
    supported = [
        dict(row)
        for row in repeatability.get("event_support", [])
        if row.get("supported") is True
    ]
    if len(supported) < 3:
        return None
    known_period = float(catalog_row["period"])
    known_epoch = float(epoch)
    tolerance = max(float(candidate["duration_hours"]) / 24.0, 0.04)
    matched = []
    for event in supported:
        center = float(event["center_btjd"])
        distance = abs(
            (center - known_epoch + 0.5 * known_period) % known_period
            - 0.5 * known_period
        )
        if distance <= tolerance:
            matched.append(event)
    containment = len(matched) / len(supported)
    segments = sorted({int(row["segment"]) for row in matched})
    minimum_containment = 0.8 if relation is None else 0.5
    if len(matched) < 3 or len(segments) < 2 or containment < minimum_containment:
        return None
    relation = relation or {
        "type": "known_event_train_subset",
        "relative_error": None,
    }
    return {
        **relation,
        "event_evidence_required": True,
        "matched_supported_events": len(matched),
        "matched_segments": segments,
        "supported_event_containment": containment,
    }


def _gap_diagnostics(candidate: Mapping, views: Mapping[int, Mapping]) -> dict:
    repeatability = (
        candidate.get("holdout_verification", {}).get("event_repeatability", {})
    )
    events = []
    for row in repeatability.get("event_support", []):
        if int(row["segment"]) not in views:
            continue
        depth = float(row.get("depth_ppm") or 0.0)
        snr = float(row.get("snr") or 0.0)
        noise = abs(depth / snr) if np.isfinite(snr) and abs(snr) > 1e-12 else np.inf
        events.append(
            {
                "segment": int(row["segment"]),
                "cycle": int(row["cycle"]),
                "center_btjd": float(row["center_btjd"]),
                "depth_ppm": depth,
                "duration_noise_ppm": noise,
                "ses_like": snr,
            }
        )
    return _event_gap_diagnostics(
        views,
        events,
        duration_days=float(candidate["duration_hours"]) / 24.0,
    )


def apply_meaningful_candidate_gate(
    candidates: Sequence[Mapping],
    branch,
    *,
    catalog: Sequence[Mapping] = (),
    threads: int = 1,
) -> tuple[list[dict], list[dict], dict]:
    """Annotate the full ledger and return candidates worth deep vetting."""

    annotated = [dict(row) for row in candidates]
    coherent_indices = [
        index
        for index, candidate in enumerate(annotated)
        if eligible_for_physical_vetting(candidate)
    ]
    reused_event_metrics = 0
    recompute_indices = []
    for index in coherent_indices:
        repeatability = (
            annotated[index]
            .get("holdout_verification", {})
            .get("event_repeatability", {})
        )
        if isinstance(repeatability.get("event_support"), list):
            reused_event_metrics += 1
        else:
            recompute_indices.append(index)
    if recompute_indices:
        prepared = prepare_event_repeatability_views(branch)
        subset = [annotated[index] for index in recompute_indices]
        metrics = event_repeatability_metrics_batch(
            subset,
            prepared_views=prepared,
            threads=max(1, int(threads)),
        )
        if metrics is None:  # pragma: no cover - optional numba-free install
            raise RuntimeError("compiled event repeatability is unavailable")
        for candidate, repeatability in zip(subset, metrics, strict=True):
            verification = dict(candidate.get("holdout_verification") or {})
            verification["event_repeatability"] = repeatability
            candidate["holdout_verification"] = verification

    views = _segment_views(branch)
    meaningful: list[dict] = []
    counts: dict[str, int] = {}
    for candidate in annotated:
        relation_matches = []
        if eligible_for_physical_vetting(candidate):
            for answer in catalog:
                relation = _catalog_relation(candidate, answer)
                if relation is not None:
                    relation_matches.append(
                        {"catalog": dict(answer), "relation": relation}
                    )
        relation_matches.sort(
            key=lambda row: (
                row["relation"]["type"] != "same_period",
                float(row["relation"].get("relative_error") or 0.0),
            )
        )
        gap = (
            _gap_diagnostics(candidate, views)
            if eligible_for_physical_vetting(candidate)
            else None
        )
        if not eligible_for_physical_vetting(candidate):
            disposition = "event_incoherent"
        elif relation_matches:
            disposition = (
                "known_catalog_period"
                if relation_matches[0]["relation"]["type"] == "same_period"
                else "event_proven_known_catalog_alias"
            )
        elif gap and gap.get("gap_dominated_weak_remainder") is True:
            disposition = "gap_adjacent_event_dominated"
        else:
            disposition = "meaningful_unexplained_candidate"
            meaningful.append(candidate)
        candidate["meaningful_candidate_gate"] = {
            "schema_version": "avril-v3-meaningful-candidate-gate-v1",
            "disposition": disposition,
            "run_expensive_vetting": disposition
            == "meaningful_unexplained_candidate",
            "catalog_relation": relation_matches[0] if relation_matches else None,
            "event_gap_diagnostics": gap,
            "candidate_retained_in_audit": True,
            "locked_ephemeris_unchanged": True,
        }
        counts[disposition] = counts.get(disposition, 0) + 1

    if len({candidate_key(row) for row in annotated}) != len(annotated):
        raise ValueError("meaningful-candidate gate changed candidate identity")
    return annotated, meaningful, {
        "schema_version": "avril-v3-meaningful-candidate-funnel-v1",
        "input_candidates": len(annotated),
        "event_screened_candidates": len(coherent_indices),
        "event_metrics_reused": reused_event_metrics,
        "event_metrics_recomputed": len(recompute_indices),
        "meaningful_candidates": len(meaningful),
        "short_circuited_candidates": len(annotated) - len(meaningful),
        "disposition_counts": counts,
        "catalog_rows": len(catalog),
        "catalog_available": bool(catalog),
        "policy": (
            "all candidates remain in the audit ledger; only unexplained, "
            "event-coherent, non-gap-dominated candidates run expensive vetting"
        ),
    }


__all__ = ["apply_meaningful_candidate_gate"]
