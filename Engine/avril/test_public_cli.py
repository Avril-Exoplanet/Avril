from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from avril.cli import _parser, _run
from avril.ffi_runner import run_ffi
from avril.hardware import execution_profile, select_backend
from avril.pixel_followup import (
    followup_ffi_result,
    followup_spoc_result,
    localize_locked_candidates,
    reaggregate_saved_pixel_followup,
)
from vlora.models import PreparedInput


def test_cpu_profile_does_not_oversubscribe_search_threads() -> None:
    profile = execution_profile("cpu", 10)
    assert profile.backend == "cpu"
    assert profile.search_workers * profile.bls_threads <= 10
    assert profile.significance_processes <= profile.significance_threads


def test_cuda_profile_keeps_one_persistent_search_worker() -> None:
    profile = execution_profile("cuda", 12)
    assert profile.search_workers == 1
    assert profile.significance_processes == profile.logical_cpus
    assert profile.vetting_processes == profile.logical_cpus


def test_explicit_cpu_never_requires_cuda() -> None:
    assert select_backend("cpu") == "cpu"


def test_public_parser_defaults_to_auto_backend() -> None:
    args = _parser().parse_args(["run", "TOI-786"])
    assert args.backend == "auto"
    assert args.source == "auto"
    assert args.pixels == "none"
    assert args.ffi_engine == "core"
    assert args.phase_trials == 4096
    assert args.vote_rounds == 5
    assert args.precision_rescue_budget == 128


def test_ffi_runner_uses_optimized_core_by_default(
    monkeypatch, tmp_path: Path
) -> None:
    local = tmp_path / "local-lc"
    local.mkdir()
    paths = []
    products = []
    for sector in (1, 2, 3):
        path = local / f"sector-{sector:04d}-lc.fits"
        path.write_bytes(f"sector {sector}".encode())
        paths.append(path)
        products.append(
            {
                "productFilename": path.name,
                "sequence_number": sector,
                "author": "AVRIL-SELFMADE-FFI",
            }
        )
    captured = {}
    monkeypatch.setattr(
        "avril.ffi_runner.prepare_input",
        lambda config, progress: PreparedInput(
            source_kind="ffi-selfmade-local",
            target=config.target,
            products=products,
            lightcurve_paths=paths,
        ),
    )

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("avril.ffi_runner.subprocess.run", fake_run)
    args = SimpleNamespace(
        input="TIC 123",
        output_root=tmp_path / "out",
        cache_root=tmp_path / "cache",
        ffi_engine="core",
        backend="cpu",
        period_max=400.0,
        phase_trials=4096,
        vote_rounds=5,
        max_refit_families=2000,
        max_refit_period_trials=3_000_000,
        precision_rescue_budget=128,
        download_workers=80,
        pixels="none",
        tic_id=123,
        ra=1.0,
        dec=2.0,
        tesscut_size=15,
        tesscut_connections=16,
        tesscut_file_workers=8,
        photometry_workers=4,
        cpu_workers=8,
        local_lightcurves=local,
        local_pixels=None,
    )
    assert run_ffi(args, tmp_path) == 0
    command = captured["command"]
    assert command[command.index("--candidate-family-mode") + 1] == "core"
    receipt = json.loads((tmp_path / "out" / "avril_run_receipt.json").read_text())
    assert receipt["ffi_engine"] == "core"
    assert receipt["legacy_multiview_pipeline_used"] is False


def test_ffi_pro_is_rejected_at_five_sectors_before_science(
    monkeypatch, tmp_path: Path
) -> None:
    local = tmp_path / "local-lc"
    local.mkdir()
    for sector in range(1, 6):
        (local / f"sector-{sector:04d}-lc.fits").touch()
    args = _parser().parse_args(
        [
            "run",
            "TIC 123",
            "--source",
            "ffi",
            "--ffi-engine",
            "pro",
            "--backend",
            "cpu",
            "--local-lightcurves",
            str(local),
            "--data-root",
            str(tmp_path),
        ]
    )
    called = False

    def should_not_prepare(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("science preparation should not start")

    monkeypatch.setattr("avril.ffi_runner.prepare_input", should_not_prepare)
    with pytest.raises(ValueError, match="fewer than 5 sectors"):
        run_ffi(args, tmp_path)
    assert called is False


def test_spoc_followup_does_not_download_when_no_high_priority_candidate(
    tmp_path: Path,
) -> None:
    final = tmp_path / "target" / "final-science" / "result.json"
    final.parent.mkdir(parents=True)
    final.write_text(json.dumps({"ranked_candidates": []}), encoding="utf-8")
    candidates = [
        {
            "candidate_id": "A",
            "rank": 1,
            "period_days": 4.0,
            "epoch_btjd": 1.0,
            "duration_hours": 2.0,
            "science_review": {"priority": "manual_review_required"},
        }
    ]
    with gzip.open(final.parent / "science_candidates.json.gz", "wt") as handle:
        json.dump(candidates, handle)
    receipt = tmp_path / "target" / "source_receipt.json"
    receipt.write_text(json.dumps({"products": []}), encoding="utf-8")
    report = followup_spoc_result(
        final, receipt, tmp_path / "cache", target="TIC 1"
    )
    assert report["status"] == "not_needed"
    assert report["pixel_download_started"] is False


def test_pixel_localization_preserves_locked_ephemeris(monkeypatch, tmp_path: Path) -> None:
    candidate = {
        "candidate_id": "A",
        "rank": 1,
        "period_days": 4.0,
        "epoch_btjd": 1.0,
        "duration_hours": 2.0,
    }
    pixel = tmp_path / "pixel.fits"
    pixel.touch()
    monkeypatch.setattr(
        "avril.pixel_followup.query_tic_gaia_neighbors", lambda _path: {}
    )
    monkeypatch.setattr(
        "avril.pixel_followup.detailed_tpf_localization",
        lambda _path, _candidate, neighbor_catalog: {"status": "on_target_consistent"},
    )
    output, report = localize_locked_candidates([candidate], [pixel])
    assert output[0]["period_days"] == candidate["period_days"]
    assert output[0]["epoch_btjd"] == candidate["epoch_btjd"]
    assert output[0]["duration_hours"] == candidate["duration_hours"]
    assert output[0]["pixel_status"] == "all_available_sectors_on_target_consistent"
    assert output[0]["pixel_vetting"]["cross_sector_consensus"][
        "on_target_sector_count"
    ] == 1
    assert report["locked_ephemerides_verified"] is True


def test_pixel_followup_does_not_downgrade_for_one_sector_outlier(
    monkeypatch, tmp_path: Path
) -> None:
    candidate = {
        "candidate_id": "A",
        "rank": 1,
        "period_days": 4.0,
        "epoch_btjd": 1.0,
        "duration_hours": 2.0,
        "event_coherence": {
            "level": "repeated_coherent_event_train",
            "supported_events": 8,
        },
        "detection_significance": {
            "passes_7p1_reference": True,
            "status": "passed",
            "block_metrics": [{"mes_like": 8.0}, {"mes_like": 7.5}],
        },
        "physical_vetting": {"vetoes": [], "passed": True, "status": "passed"},
        "reference_vetting": {
            "warnings": [],
            "available": True,
            "status": "passed",
        },
        "science_review": {"priority": "high_priority_follow_up_candidate"},
    }
    paths = []
    for index in range(5):
        path = tmp_path / f"pixel-{index}.fits"
        path.touch()
        paths.append(path)
    monkeypatch.setattr(
        "avril.pixel_followup.query_tic_gaia_neighbors", lambda _path: {}
    )

    def localization(path, _candidate, *, neighbor_catalog):
        return {
            "status": (
                "possible_off_target" if Path(path).stem == "pixel-4" else "on_target_consistent"
            ),
            "hard_rejected": False,
        }

    monkeypatch.setattr(
        "avril.pixel_followup.detailed_tpf_localization", localization
    )
    output, _report = localize_locked_candidates([candidate], paths)

    assert output[0]["pixel_status"] == "on_target_consensus_across_sectors"
    assert output[0]["science_review"]["priority"] == (
        "high_priority_follow_up_candidate"
    )
    assert output[0]["pixel_vetting"]["cross_sector_consensus"][
        "off_target_sector_count"
    ] == 1


def test_saved_pixel_reaggregation_preserves_ephemerides_and_old_result(
    tmp_path: Path,
) -> None:
    source = tmp_path / "old" / "result.json"
    source.parent.mkdir(parents=True)
    candidate = {
        "candidate_id": "A",
        "rank": 1,
        "discovery_rank": 1,
        "period_days": 12.5,
        "epoch_btjd": 100.0,
        "duration_hours": 3.0,
        "event_coherence": {
            "level": "repeated_coherent_event_train",
            "supported_events": 8,
        },
        "detection_significance": {
            "passes_7p1_reference": True,
            "status": "passed",
            "block_metrics": [{"mes_like": 8.0}, {"mes_like": 7.5}],
        },
        "physical_vetting": {"vetoes": [], "passed": True, "status": "passed"},
        "reference_vetting": {
            "warnings": [],
            "available": True,
            "status": "passed",
        },
        "pixel_vetting": {
            "aggregate_status": "possible_off_target_in_at_least_one_sector",
            "per_sector": [
                *[{"status": "on_target_consistent"} for _ in range(8)],
                {"status": "possible_off_target"},
            ],
        },
        "science_review": {"priority": "low_priority_or_false_alarm_risk"},
    }
    source.write_text(
        json.dumps(
            {
                "ranked_candidates": [candidate],
                "science_annotation": {
                    "priority_counts": {"low_priority_or_false_alarm_risk": 1}
                },
            }
        ),
        encoding="utf-8",
    )
    with gzip.open(source.parent / "science_candidates.json.gz", "wt") as handle:
        json.dump([candidate], handle)
    old_bytes = source.read_bytes()

    report = reaggregate_saved_pixel_followup(source, tmp_path / "new")

    assert source.read_bytes() == old_bytes
    assert report["priority_counts_after"] == {
        "high_priority_follow_up_candidate": 1
    }
    assert report["candidates_with_saved_pixel_evidence_reaggregated"] == 1
    with gzip.open(tmp_path / "new" / "science_candidates.json.gz", "rt") as handle:
        updated = json.load(handle)[0]
    assert updated["period_days"] == 12.5
    assert updated["epoch_btjd"] == 100.0
    assert updated["duration_hours"] == 3.0
    assert updated["pixel_status"] == "on_target_consensus_across_sectors"


def test_ffi_pixel_followup_replaces_final_priority_summary(
    monkeypatch, tmp_path: Path
) -> None:
    final = tmp_path / "target" / "final-science" / "result.json"
    final.parent.mkdir(parents=True)
    candidate = {
        "candidate_id": "A",
        "rank": 1,
        "period_days": 4.0,
        "epoch_btjd": 1.0,
        "duration_hours": 2.0,
        "science_review": {"priority": "high_priority_follow_up_candidate"},
    }
    final.write_text(
        json.dumps(
            {
                "ranked_candidates": [candidate],
                "science_annotation": {
                    "priority_counts": {"high_priority_follow_up_candidate": 1}
                },
            }
        ),
        encoding="utf-8",
    )
    with gzip.open(final.parent / "science_candidates.json.gz", "wt") as handle:
        json.dump([candidate], handle)

    downgraded = {
        **candidate,
        "science_review": {"priority": "low_priority_or_false_alarm_risk"},
    }
    monkeypatch.setattr(
        "avril.pixel_followup.localize_locked_candidates",
        lambda *_args, **_kwargs: (
            [downgraded],
            {
                "schema_version": "avril-v1-pixel-followup",
                "candidate_order_preserved": True,
                "locked_ephemerides_verified": True,
            },
        ),
    )
    followup_ffi_result(final, [])
    output = json.loads(
        (tmp_path / "target" / "final-science-pixel-followup" / "result.json").read_text()
    )
    annotation = output["science_annotation"]
    assert annotation["priority_counts_before_pixel_followup"] == {
        "high_priority_follow_up_candidate": 1
    }
    assert annotation["priority_counts"] == {
        "low_priority_or_false_alarm_risk": 1
    }
    assert output["pixel_followup"]["priority_counts_after_pixel_followup"] == {
        "low_priority_or_false_alarm_risk": 1
    }


def test_ffi_pixel_followup_recomputes_science_rank_without_changing_discovery(
    monkeypatch, tmp_path: Path
) -> None:
    final = tmp_path / "target" / "final-science" / "result.json"
    final.parent.mkdir(parents=True)
    candidates = [
        {
            "candidate_id": "A",
            "proposal_id": "A",
            "rank": 1,
            "science_rank": 1,
            "discovery_rank": 1,
            "ensemble_rank": 1,
            "period_days": 4.0,
            "epoch_btjd": 1.0,
            "duration_hours": 2.0,
            "science_review": {"priority": "high_priority_follow_up_candidate"},
        },
        {
            "candidate_id": "B",
            "proposal_id": "B",
            "rank": 2,
            "science_rank": 2,
            "discovery_rank": 2,
            "ensemble_rank": 2,
            "period_days": 9.0,
            "epoch_btjd": 2.0,
            "duration_hours": 3.0,
            "science_review": {"priority": "high_priority_follow_up_candidate"},
        },
    ]
    final.write_text(
        json.dumps(
            {
                "ranked_candidates": candidates,
                "science_annotation": {
                    "priority_counts": {"high_priority_follow_up_candidate": 2}
                },
            }
        ),
        encoding="utf-8",
    )
    with gzip.open(final.parent / "science_candidates.json.gz", "wt") as handle:
        json.dump(candidates, handle)

    localized = json.loads(json.dumps(candidates))
    localized[0]["science_review"] = {
        "priority": "low_priority_or_false_alarm_risk"
    }
    monkeypatch.setattr(
        "avril.pixel_followup.localize_locked_candidates",
        lambda *_args, **_kwargs: (
            localized,
            {
                "schema_version": "avril-v1-pixel-followup",
                "candidate_order_preserved": True,
                "locked_ephemerides_verified": True,
            },
        ),
    )

    followup_ffi_result(final, [])

    output = json.loads(
        (tmp_path / "target" / "final-science-pixel-followup" / "result.json").read_text()
    )
    assert [row["candidate_id"] for row in output["ranked_candidates"]] == ["B", "A"]
    assert [row["science_rank"] for row in output["ranked_candidates"]] == [1, 2]
    assert [row["discovery_rank"] for row in output["ranked_candidates"]] == [2, 1]
    assert [row["period_days"] for row in output["ranked_candidates"]] == [9.0, 4.0]


def test_auto_source_falls_back_to_ffi_only_after_spoc_is_insufficient(
    monkeypatch, tmp_path: Path
) -> None:
    args = _parser().parse_args(
        ["run", "TIC 123", "--source", "auto", "--data-root", str(tmp_path)]
    )
    monkeypatch.setattr(
        "avril.cli._manifest_for_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("Avril requires at least three official light-curve sectors; found 0")
        ),
    )
    called = {}

    def fake_ffi(run_args, root):
        called["target"] = run_args.input
        called["root"] = root
        return 0

    monkeypatch.setattr("avril.ffi_runner.run_ffi", fake_ffi)
    assert _run(args) == 0
    assert called == {"target": "TIC 123", "root": tmp_path.resolve()}


def test_explicit_spoc_never_silently_falls_back_to_ffi(monkeypatch, tmp_path: Path) -> None:
    args = _parser().parse_args(
        ["run", "TIC 123", "--source", "spoc", "--data-root", str(tmp_path)]
    )
    monkeypatch.setattr(
        "avril.cli._manifest_for_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("insufficient SPOC")),
    )
    with pytest.raises(RuntimeError, match="insufficient SPOC"):
        _run(args)
