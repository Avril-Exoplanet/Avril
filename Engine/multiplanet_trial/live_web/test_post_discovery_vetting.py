from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from multiplanet_trial.v03.pixel_vetting import _localization_aperture

from . import post_discovery_vetting as stage
from . import pipeline
from .pipeline import (
    _download_products,
    _download_target_pixels,
    _select_target_pixels,
    _submit_background_pixel_prefetch,
)


def _candidate(candidate_id: str, rank: int, period: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "rank": rank,
        "period_days": period,
        "epoch_btjd": 100.0 + rank,
        "duration_hours": 2.4,
        "features": {"median_search_snr": 10.0},
        "physical_vetting": {},
        "pixel_status": "not_run",
        "hard_rejected": False,
    }


def test_raw_tesscut_without_pipeline_mask_gets_a_circular_aperture() -> None:
    class BrokenWcs:
        @staticmethod
        def world_to_pixel(_coordinate):
            raise ValueError("no celestial WCS in synthetic fixture")

    tpf = SimpleNamespace(
        pipeline_mask=np.zeros((15, 15), dtype=bool),
        flux=SimpleNamespace(value=np.zeros((2, 15, 15), dtype=float)),
        ra=1.0,
        dec=2.0,
        wcs=BrokenWcs(),
    )
    aperture, source = _localization_aperture(tpf)
    assert source == "tesscut_circular_2p0_pixel_fallback"
    assert aperture.shape == (15, 15)
    assert aperture.sum() > 1


def test_vetting_only_annotates_and_never_removes_or_retunes(monkeypatch) -> None:
    candidates = [_candidate("A", 1, 3.0), _candidate("B", 2, 7.0)]
    locked = [
        (row["candidate_id"], row["rank"], row["period_days"], row["epoch_btjd"])
        for row in candidates
    ]

    monkeypatch.setattr(
        stage, "load_pdcsap_physics", lambda _paths, **_kwargs: {"loaded": True}
    )

    def fake_physical(_data, candidate):
        flagged = candidate["candidate_id"] == "A"
        return {
            "features": {"physical_available_count": 5},
            "evidence": {
                "vetoes": ["odd_even"] if flagged else [],
                "passed": not flagged,
            },
        }

    monkeypatch.setattr(stage, "physical_vetting_features", fake_physical)
    monkeypatch.setattr(stage, "fast_survey_physical_vetting", fake_physical)
    monkeypatch.setattr(
        stage,
        "exovetter_diagnostics",
        lambda _data, _candidate: {"warnings": [], "available": True},
    )
    monkeypatch.setattr(
        stage,
        "query_tic_gaia_neighbors",
        lambda path: {"file": Path(path).name},
    )

    def fake_localization(path, candidate, *, neighbor_catalog):
        return {
            "tpf_file": Path(path).name,
            "status": (
                "possible_off_target"
                if candidate["candidate_id"] == "B"
                and Path(path).stem in {"second", "third"}
                else "on_target_consistent"
            ),
            "catalog_received": neighbor_catalog is not None,
        }

    monkeypatch.setattr(stage, "detailed_tpf_localization", fake_localization)

    output, report = stage.annotate_locked_candidates(
        [Path("lightcurve.fits")],
        candidates,
        target_pixel_paths=[
            Path("first.fits"),
            Path("second.fits"),
            Path("third.fits"),
        ],
    )

    assert len(output) == len(candidates) == 2
    assert [
        (row["candidate_id"], row["rank"], row["period_days"], row["epoch_btjd"])
        for row in output
    ] == locked
    assert output[0]["vetting_summary"]["review_status"] == (
        "possible_astrophysical_false_positive"
    )
    assert output[1]["vetting_summary"]["review_status"] == (
        "possible_off_target_or_blend"
    )
    assert output[1]["pixel_status"] == (
        "possible_off_target_consensus_across_sectors"
    )
    assert output[1]["pixel_vetting"]["cross_sector_consensus"][
        "off_target_sector_count"
    ] == 2
    assert all(row["hard_rejected"] is False for row in output)
    assert all(row["vetting_summary"]["candidate_retained"] is True for row in output)
    assert report["candidate_count_before"] == report["candidate_count_after"] == 2
    assert report["candidate_order_preserved"] is True
    assert report["locked_ephemerides_verified"] is True


def test_unavailable_pixel_evidence_keeps_candidate(monkeypatch) -> None:
    monkeypatch.setattr(
        stage, "load_pdcsap_physics", lambda _paths, **_kwargs: {"loaded": True}
    )
    monkeypatch.setattr(
        stage,
        "physical_vetting_features",
        lambda _data, _candidate: {
            "features": {"physical_available_count": 5},
            "evidence": {"vetoes": [], "passed": True},
        },
    )
    monkeypatch.setattr(
        stage,
        "fast_survey_physical_vetting",
        lambda _data, _candidate: {
            "features": {"physical_available_count": 5},
            "evidence": {"vetoes": [], "passed": True},
        },
    )
    monkeypatch.setattr(
        stage,
        "exovetter_diagnostics",
        lambda _data, _candidate: {"warnings": [], "available": True},
    )
    candidate = _candidate("A", 1, 5.0)

    output, report = stage.annotate_locked_candidates([], [candidate])

    assert len(output) == 1
    assert output[0]["pixel_status"] == "no_pixel_localization"
    assert output[0]["vetting_summary"]["review_status"] == "needs_pixel_localization"
    assert output[0]["hard_rejected"] is False
    assert report["candidate_count_after"] == 1


def test_target_pixel_selection_matches_every_searched_segment() -> None:
    products = [
        {
            "product_kind": "target_pixel",
            "sequence_number": 1,
            "author": "QLP",
            "exptime": 600,
            "productFilename": "slow.fits",
        },
        {
            "product_kind": "target_pixel",
            "sequence_number": 1,
            "author": "SPOC",
            "exptime": 120,
            "productFilename": "preferred.fits",
        },
        {
            "product_kind": "target_pixel",
            "sequence_number": 2,
            "author": "SPOC",
            "exptime": 120,
            "productFilename": "second.fits",
        },
        {
            "product_kind": "target_pixel",
            "sequence_number": 99,
            "author": "SPOC",
            "exptime": 120,
            "productFilename": "not-searched.fits",
        },
        {
            "product_kind": "lightcurve",
            "sequence_number": 1,
            "author": "SPOC",
            "exptime": 120,
            "productFilename": "not-a-tpf.fits",
        },
    ]

    selected = _select_target_pixels(products, [1, 2])

    assert [row["productFilename"] for row in selected] == [
        "preferred.fits",
        "second.fits",
    ]


def test_stage_one_lightcurves_download_concurrently_and_keep_order(
    monkeypatch, tmp_path
) -> None:
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_download(product, destination):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return Path(destination) / str(product["productFilename"])

    monkeypatch.setattr(pipeline, "_download_one_resumable", fake_download)
    products = [
        {
            "mission": f"TESS Sector {index}",
            "sequence_number": index,
            "productFilename": f"lc-{index}.fits",
        }
        for index in range(1, 5)
    ]

    paths = _download_products(products, tmp_path, None, max_workers=4)

    assert maximum_active >= 2
    assert [path.name for path in paths] == [f"lc-{index}.fits" for index in range(1, 5)]


def test_pixel_prefetch_is_submitted_without_blocking_period_search() -> None:
    started = threading.Event()
    release = threading.Event()

    def blocked_pixel_download():
        started.set()
        release.wait(timeout=2)
        return "pixels-ready"

    pool, future = _submit_background_pixel_prefetch(blocked_pixel_download)
    try:
        assert started.wait(timeout=1)
        assert not future.done()
        release.set()
        assert future.result(timeout=1) == "pixels-ready"
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_stage_two_pixels_download_concurrently_and_record_failures(
    monkeypatch, tmp_path
) -> None:
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_download(product, destination):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        if product["sequence_number"] == 2:
            raise IOError("synthetic failure")
        return Path(destination) / str(product["productFilename"])

    monkeypatch.setattr(pipeline, "_download_one_resumable", fake_download)
    products = [
        {
            "mission": f"TESS Sector {index}",
            "sequence_number": index,
            "productFilename": f"tpf-{index}.fits",
        }
        for index in range(1, 5)
    ]

    events = []

    def progress(stage_name, percent, message, detail):
        events.append((stage_name, percent, message, detail))

    paths, diagnostics = _download_target_pixels(
        products, tmp_path, progress, max_workers=4
    )

    assert maximum_active >= 2
    assert [path.name for path in paths] == ["tpf-1.fits", "tpf-3.fits", "tpf-4.fits"]
    assert diagnostics[0]["segment"] == 2
    assert diagnostics[0]["status"] == "failed"
    final_metrics = events[-1][3]["metrics"]
    assert final_metrics["files_verified"] == 3
    assert final_metrics["files_completed"] == 4
    assert final_metrics["files_failed"] == 1
    assert final_metrics["files_active"] == 0
