#!/usr/bin/env python3
"""Verify scientific equivalence of one Mac and CUDA VLora result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HIGH = "high_priority_follow_up_candidate"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nearest(candidates: list[dict], period: float) -> tuple[dict, float]:
    candidate = min(
        candidates,
        key=lambda row: abs(float(row["period_days"]) / period - 1.0),
    )
    relative_error = abs(float(candidate["period_days"]) / period - 1.0)
    return candidate, relative_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mac", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--known-period", type=float, action="append", default=[])
    parser.add_argument("--period-tolerance", type=float, default=1e-5)
    parser.add_argument("--minimum-mac-coverage", type=float, default=0.98)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    mac = _load(args.mac)
    cuda = _load(args.cuda)
    mac_candidates = list(mac["ranked_candidates"])
    cuda_candidates = list(cuda["ranked_candidates"])
    matches: list[tuple[dict, dict]] = []
    priority_disagreements: list[dict] = []
    for left in mac_candidates:
        right, relative_error = _nearest(
            cuda_candidates, float(left["period_days"])
        )
        if relative_error > args.period_tolerance:
            continue
        matches.append((left, right))
        left_priority = str(left["science_review"]["priority"])
        right_priority = str(right["science_review"]["priority"])
        if left_priority != right_priority:
            priority_disagreements.append(
                {
                    "mac_period_days": left["period_days"],
                    "cuda_period_days": right["period_days"],
                    "mac_priority": left_priority,
                    "cuda_priority": right_priority,
                }
            )

    mac_high = [
        row for row in mac_candidates
        if row["science_review"]["priority"] == HIGH
    ]
    cuda_high = [
        row for row in cuda_candidates
        if row["science_review"]["priority"] == HIGH
    ]
    high_matches: list[dict] = []
    unmatched_high: list[dict] = []
    for left in mac_high:
        right, relative_error = _nearest(
            cuda_high, float(left["period_days"])
        )
        if relative_error > args.period_tolerance:
            unmatched_high.append({"backend": "mac", "period_days": left["period_days"]})
            continue
        high_matches.append(
            {
                "mac_period_days": left["period_days"],
                "cuda_period_days": right["period_days"],
                "relative_period_error": relative_error,
                "epoch_difference_days": (
                    float(right["epoch_btjd"]) - float(left["epoch_btjd"])
                ),
                "duration_difference_hours": (
                    float(right["duration_hours"])
                    - float(left["duration_hours"])
                ),
                "mac_mes_like": left["detection_significance"]["combined"]["mes_like"],
                "cuda_mes_like": right["detection_significance"]["combined"]["mes_like"],
                "mac_local_fap": left["detection_significance"]["phase_shift_null"]["local_empirical_fap"],
                "cuda_local_fap": right["detection_significance"]["phase_shift_null"]["local_empirical_fap"],
            }
        )
    for right in cuda_high:
        left, relative_error = _nearest(mac_high, float(right["period_days"]))
        if relative_error > args.period_tolerance:
            unmatched_high.append({"backend": "cuda", "period_days": right["period_days"]})

    known_checks = []
    for period in args.known_period:
        left, left_error = _nearest(mac_candidates, period)
        right, right_error = _nearest(cuda_candidates, period)
        known_checks.append(
            {
                "expected_period_days": period,
                "mac_period_days": left["period_days"],
                "cuda_period_days": right["period_days"],
                "mac_relative_error": left_error,
                "cuda_relative_error": right_error,
                "mac_priority": left["science_review"]["priority"],
                "cuda_priority": right["science_review"]["priority"],
                "passed": (
                    left_error <= args.period_tolerance
                    and right_error <= args.period_tolerance
                    and left["science_review"]["priority"] == HIGH
                    and right["science_review"]["priority"] == HIGH
                ),
            }
        )

    mac_total = float(mac["timing_seconds"]["reusable_pipeline_compute_total"]) + float(
        mac["timing_seconds"]["science_annotation_wall"]
    )
    cuda_total = float(cuda["timing_seconds"]["reusable_pipeline_compute_total"]) + float(
        cuda["timing_seconds"]["science_annotation_wall"]
    )
    mac_coverage = len(matches) / max(1, len(mac_candidates))
    passed = (
        mac_coverage >= args.minimum_mac_coverage
        and not priority_disagreements
        and not unmatched_high
        and len(mac_high) == len(cuda_high)
        and all(row["passed"] for row in known_checks)
        and cuda_total < mac_total
    )
    report = {
        "schema_version": "vlora-cross-platform-regression-v1",
        "passed": passed,
        "inputs": {
            "mac": str(args.mac.resolve()),
            "mac_sha256": _sha256(args.mac),
            "cuda": str(args.cuda.resolve()),
            "cuda_sha256": _sha256(args.cuda),
        },
        "candidate_counts": {
            "mac": len(mac_candidates),
            "cuda": len(cuda_candidates),
            "matched": len(matches),
            "mac_coverage": mac_coverage,
        },
        "priority_counts": {
            "mac": mac["science_annotation"]["priority_counts"],
            "cuda": cuda["science_annotation"]["priority_counts"],
        },
        "priority_disagreements": priority_disagreements,
        "unmatched_high_priority": unmatched_high,
        "high_priority_matches": high_matches,
        "known_period_checks": known_checks,
        "timing_seconds": {
            "mac_full_pipeline": mac_total,
            "cuda_full_pipeline": cuda_total,
            "cuda_seconds_saved": mac_total - cuda_total,
            "cuda_speedup": mac_total / cuda_total,
        },
        "contract": (
            "final scientific state equivalence; cross-architecture raw "
            "boundary-peak bit identity is not required"
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
