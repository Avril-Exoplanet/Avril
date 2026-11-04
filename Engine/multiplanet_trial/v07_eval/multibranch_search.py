"""Independent multi-photometry search and candidate fusion for v0.3.

Unlike :mod:`lightkurve_evidence`, this module lets each photometry branch
generate candidates on its own.  The branches are correlated views of the same
photons, so support counts are exposed as evidence rather than treated as
independent statistical votes.
"""

from __future__ import annotations

import re
import subprocess
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping

import astropy.units as u
import lightkurve as lk
import numpy as np
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares

from multiplanet_trial.live_web.fixed_ephemeris_acceleration import (
    measure_fixed_ephemerides_compiled,
)
from multiplanet_trial.v03.event_search import EventSearchConfig, event_first_search
from multiplanet_trial.v03.lightkurve_evidence import measure_fixed_ephemeris


@dataclass(frozen=True)
class SearchBranch:
    key: str
    label: str
    processing: str
    photometry_family: str
    method_family: str
    time: np.ndarray
    flux: np.ndarray
    segment_id: np.ndarray
    can_seed_candidates: bool = True
    metadata: Mapping = field(default_factory=dict)


def _mad_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    median = np.nanmedian(values)
    sigma = 1.4826 * np.nanmedian(np.abs(values - median))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.nanstd(values)
    return float(sigma if np.isfinite(sigma) and sigma > 0 else 0.0)


def _depth_difference(injected: Mapping, control: Mapping) -> float | None:
    injected_depth = injected.get("depth_ppm")
    control_depth = control.get("depth_ppm")
    if injected_depth is None or control_depth is None:
        return None
    return float(injected_depth - control_depth)


def _deterministic_correct(corrector, *, random_seed: int = 314159, **kwargs):
    """Run randomized Lightkurve internals reproducibly without leaking RNG state."""
    state = np.random.get_state()
    np.random.seed(random_seed)
    try:
        return corrector.correct(**kwargs)
    finally:
        np.random.set_state(state)


def _odd_window(time: np.ndarray, window_days: float, length: int) -> int:
    finite_time = np.asarray(time, dtype=float)
    finite_time = finite_time[np.isfinite(finite_time)]
    if finite_time.size < 9:
        return 0
    cadence = float(np.nanmedian(np.diff(np.sort(finite_time))))
    if not np.isfinite(cadence) or cadence <= 0:
        return 0
    window = max(9, int(round(window_days / cadence)))
    if window % 2 == 0:
        window += 1
    maximum = length if length % 2 else length - 1
    return int(min(window, maximum)) if maximum >= 9 else 0


def _segment_id(lightcurve) -> tuple[int, str]:
    for key, label in (("SECTOR", "sector"), ("QUARTER", "quarter"), ("CAMPAIGN", "campaign")):
        value = lightcurve.meta.get(key)
        if value is not None:
            return int(value), label
    return -1, "segment"


def _prepare_lightcurve(lightcurve, flatten_window_days: float) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lightcurve = lightcurve.remove_nans().normalize()
        window = _odd_window(lightcurve.time.value, flatten_window_days, len(lightcurve))
        if window:
            lightcurve = lightcurve.flatten(
                window_length=window,
                polyorder=2,
                break_tolerance=5,
            )
        lightcurve = lightcurve.remove_outliers(sigma_upper=7.0, sigma_lower=12.0)
    time = np.asarray(lightcurve.time.value, dtype=float)
    flux = np.asarray(lightcurve.flux.value, dtype=float)
    good = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
    order = np.argsort(time[good])
    return time[good][order], flux[good][order]


def _assemble_branch(
    key: str,
    label: str,
    processing: str,
    photometry_family: str,
    pieces: list[tuple[np.ndarray, np.ndarray, int]],
    *,
    can_seed_candidates: bool = True,
    metadata: Mapping | None = None,
) -> SearchBranch:
    if not pieces:
        raise ValueError(f"No usable pieces for branch {key}")
    time = np.concatenate([piece[0] for piece in pieces])
    flux = np.concatenate([piece[1] for piece in pieces])
    segment = np.concatenate(
        [np.full(piece[0].size, piece[2], dtype=int) for piece in pieces]
    )
    order = np.argsort(time)
    return SearchBranch(
        key=key,
        label=label,
        processing=processing,
        photometry_family=photometry_family,
        method_family="bls",
        time=time[order],
        flux=flux[order],
        segment_id=segment[order],
        can_seed_candidates=can_seed_candidates,
        metadata=dict(metadata or {}),
    )


def _fits_is_complete(path: Path) -> bool:
    try:
        with fits.open(path, memmap=True) as hdul:
            info = hdul.fileinfo(len(hdul) - 1)
            expected = int(info["datLoc"] + info["datSpan"])
        return path.stat().st_size >= expected
    except Exception:
        return False


def ensure_tess_cbv(path: str | Path, cbv_dir: str | Path) -> Path:
    """Cache the matching official TESS CBV FITS under an explicit directory."""
    path = Path(path)
    destination = Path(cbv_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with fits.open(path, memmap=True) as hdul:
        header = hdul[0].header
        sector = int(header["SECTOR"])
        camera = int(header["CAMERA"])
        ccd = int(header["CCD"])
    pattern = f"s{sector:04d}-{camera}-{ccd}-"
    existing = [item for item in destination.glob("*.fits") if pattern in item.name]
    for item in existing:
        if _fits_is_complete(item):
            return item
    script_url = (
        "https://archive.stsci.edu/missions/tess/download_scripts/sector/"
        f"tesscurl_sector_{sector}_cbv.sh"
    )
    script = urllib.request.urlopen(script_url, timeout=60).read().decode("utf-8")
    matching_line = next((line for line in script.splitlines() if pattern in line), None)
    if matching_line is None:
        raise FileNotFoundError(f"No CBV entry found for sector/camera/ccd {sector}/{camera}/{ccd}")
    match = re.search(r"https://[^\s'\"]+?\.fits", matching_line)
    if match is None:
        raise ValueError(f"Could not parse CBV URL for {pattern}")
    archive_url = match.group(0)
    mirror_url = archive_url.replace(
        "https://archive.stsci.edu/missions/tess/",
        "https://stpubdata.s3.amazonaws.com/tess/public/",
    )
    target = destination / archive_url.rsplit("/", 1)[-1]
    last_error = None
    # In practice the STScI archive is substantially faster and more reliable
    # for these small CBV products.  Keep the public S3 endpoint as a fallback,
    # but do not spend eight timeout cycles there before trying the archive.
    for url in (archive_url, mirror_url):
        for _attempt in range(8):
            result = subprocess.run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    "60",
                    "-C",
                    "-",
                    "-o",
                    str(target),
                    url,
                ],
                check=False,
            )
            if _fits_is_complete(target):
                return target
            last_error = result.returncode
    raise IOError(f"CBV download failed from mirror and archive (curl={last_error})")


def load_spoc_independent_branches(
    paths: Iterable[str | Path],
    *,
    flatten_window_days: float = 1.0,
    include_cbv: bool = False,
    cbv_dir: str | Path | None = None,
) -> tuple[dict[str, SearchBranch], list[dict]]:
    """Build independently searched PDCSAP, SAP, and optional SAP+CBV branches."""
    resolved = [Path(path) for path in paths]
    pieces: dict[str, list[tuple[np.ndarray, np.ndarray, int]]] = {
        "pdcsap_lightkurve_bls": [],
        "sap_lightkurve_bls": [],
        "sap_cbv_lightkurve_bls": [],
    }
    diagnostics: list[dict] = []
    cbv_metrics: list[dict] = []
    for path in resolved:
        for key, column in (
            ("pdcsap_lightkurve_bls", "pdcsap_flux"),
            ("sap_lightkurve_bls", "sap_flux"),
        ):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lightcurve = lk.read(path, flux_column=column)
                segment, segment_kind = _segment_id(lightcurve)
                time, flux = _prepare_lightcurve(lightcurve, flatten_window_days)
                pieces[key].append((time, flux, segment))
                diagnostics.append(
                    {
                        "branch": key,
                        "file": path.name,
                        "status": "ready",
                        "segment": segment,
                        "segment_kind": segment_kind,
                        "cadences": int(time.size),
                    }
                )
            except Exception as exc:
                diagnostics.append(
                    {
                        "branch": key,
                        "file": path.name,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        if include_cbv:
            key = "sap_cbv_lightkurve_bls"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sap = lk.read(path, flux_column="sap_flux").remove_nans()
                segment, segment_kind = _segment_id(sap)
                mission = str(sap.meta.get("MISSION", "")).upper()
                corrector_kwargs = {}
                if mission == "TESS" and cbv_dir is not None:
                    ensure_tess_cbv(path, cbv_dir)
                    corrector_kwargs["cbv_dir"] = str(Path(cbv_dir))
                corrector = sap.to_corrector("cbv", **corrector_kwargs)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corrected = _deterministic_correct(
                        corrector,
                        target_under_score=0.0,
                        max_iter=30,
                    )
                time, flux = _prepare_lightcurve(corrected, flatten_window_days)
                pieces[key].append((time, flux, segment))
                metric = {
                    "file": path.name,
                    "segment": segment,
                    "segment_kind": segment_kind,
                    "over_fitting_score": float(corrector.over_fitting_score),
                    "under_fitting_score": None,
                    "alpha": float(corrector.alpha),
                    "cadences": int(time.size),
                    "note": "under-fitting optimization disabled to avoid hidden neighbour downloads",
                }
                cbv_metrics.append(metric)
                diagnostics.append({"branch": key, "status": "ready", **metric})
            except Exception as exc:
                diagnostics.append(
                    {
                        "branch": key,
                        "file": path.name,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    branches = {
        "pdcsap_lightkurve_bls": _assemble_branch(
            "pdcsap_lightkurve_bls",
            "PDCSAP + Lightkurve flatten + independent BLS",
            f"PDCSAP; Lightkurve {lk.__version__}; flatten {flatten_window_days:g} d",
            "spoc_pipeline_aperture",
            pieces["pdcsap_lightkurve_bls"],
        ),
        "sap_lightkurve_bls": _assemble_branch(
            "sap_lightkurve_bls",
            "SAP + Lightkurve flatten + independent BLS",
            f"SAP; Lightkurve {lk.__version__}; flatten {flatten_window_days:g} d",
            "spoc_pipeline_aperture",
            pieces["sap_lightkurve_bls"],
        ),
    }
    if pieces["sap_cbv_lightkurve_bls"]:
        branches["sap_cbv_lightkurve_bls"] = _assemble_branch(
            "sap_cbv_lightkurve_bls",
            "SAP + official CBVCorrector + independent BLS",
            "SAP; official mission CBVs; optimized ridge correction; Lightkurve flatten",
            "spoc_pipeline_aperture",
            pieces["sap_cbv_lightkurve_bls"],
            metadata={"cbv_metrics": cbv_metrics},
        )
    return branches, diagnostics


# Public mission-neutral name.  The original name is kept for compatibility
# with the first TESS-only prototype.
load_mission_lightcurve_branches = load_spoc_independent_branches


def load_tpf_aperture_branches(
    paths: Iterable[str | Path],
    *,
    thresholds: Iterable[float] = (2.0, 3.0, 5.0),
    flatten_window_days: float = 1.0,
) -> tuple[dict[str, SearchBranch], list[dict]]:
    """Extract pipeline and threshold-aperture photometry from local TPFs."""
    specifications = [("pipeline", None)] + [
        (f"threshold_{threshold:g}", float(threshold)) for threshold in thresholds
    ]
    pieces = {f"tpf_{name}_bls": [] for name, _ in specifications}
    masks: dict[str, list[dict]] = {f"tpf_{name}_bls": [] for name, _ in specifications}
    diagnostics: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tpf = lk.read(path)
            segment, segment_kind = _segment_id(tpf)
        except Exception as exc:
            diagnostics.append(
                {
                    "branch": "tpf",
                    "file": path.name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        for name, threshold in specifications:
            key = f"tpf_{name}_bls"
            try:
                mask = (
                    np.asarray(tpf.pipeline_mask, dtype=bool)
                    if threshold is None
                    else np.asarray(
                        tpf.create_threshold_mask(
                            threshold=threshold,
                            reference_pixel="center",
                        ),
                        dtype=bool,
                    )
                )
                if not mask.any():
                    raise ValueError("aperture mask is empty")
                lightcurve = tpf.to_lightcurve(aperture_mask=mask)
                time, flux = _prepare_lightcurve(lightcurve, flatten_window_days)
                pieces[key].append((time, flux, segment))
                mask_info = {
                    "file": path.name,
                    "segment": segment,
                    "segment_kind": segment_kind,
                    "pixels": int(mask.sum()),
                    "mask": mask.tolist(),
                    "cadences": int(time.size),
                }
                masks[key].append(mask_info)
                diagnostics.append({"branch": key, "status": "ready", **mask_info})
            except Exception as exc:
                diagnostics.append(
                    {
                        "branch": key,
                        "file": path.name,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    branches = {}
    for name, threshold in specifications:
        key = f"tpf_{name}_bls"
        if not pieces[key]:
            continue
        aperture_label = "pipeline" if threshold is None else f"{threshold:g}-sigma threshold"
        branches[key] = _assemble_branch(
            key,
            f"TPF {aperture_label} aperture + independent BLS",
            f"TargetPixelFile aperture photometry ({aperture_label}); Lightkurve flatten",
            "tpf_custom_aperture",
            pieces[key],
            metadata={"apertures": masks[key]},
        )
    return branches, diagnostics


def load_protected_pld_branch(
    paths: Iterable[str | Path],
    ephemerides: Iterable[Mapping],
    *,
    flatten_window_days: float = 1.0,
) -> tuple[SearchBranch | None, list[dict]]:
    """Build a candidate-protected PLD follow-up branch.

    PLD is run only after an initial search.  Cadences around every initial
    candidate are excluded from the regression fit to reduce transit
    attenuation.  Because this branch is conditioned on existing candidates,
    it may support a cluster but cannot create one by itself.
    """
    protected = [dict(item) for item in ephemerides]
    if not protected:
        return None, [{"branch": "tpf_pld_protected_bls", "status": "no_ephemerides_to_protect"}]
    pieces: list[tuple[np.ndarray, np.ndarray, int]] = []
    diagnostics: list[dict] = []
    preservation: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tpf = lk.read(path)
            segment, segment_kind = _segment_id(tpf)
            aperture = np.asarray(tpf.pipeline_mask, dtype=bool)
            if not aperture.any():
                aperture = np.asarray(
                    tpf.create_threshold_mask(threshold=3, reference_pixel="center"),
                    dtype=bool,
                )
            time = np.asarray(tpf.time.value, dtype=float)
            fit_mask = np.isfinite(time)
            for candidate in protected:
                period = float(candidate["period_days"])
                epoch = float(candidate["epoch_btjd"])
                duration = float(candidate["duration_hours"]) / 24.0
                phase = (time - epoch + 0.5 * period) % period - 0.5 * period
                fit_mask &= np.abs(phase) > 0.75 * duration
            raw_lc = tpf.to_lightcurve(aperture_mask=aperture)
            corrector = tpf.to_corrector("pld", aperture_mask=aperture)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corrected = _deterministic_correct(
                    corrector,
                    pld_order=1,
                    pca_components=3,
                    restore_trend=False,
                    cadence_mask=fit_mask,
                    sigma=5,
                    niters=5,
                )
            raw_time, raw_flux = _prepare_lightcurve(raw_lc, flatten_window_days)
            corrected_time, corrected_flux = _prepare_lightcurve(
                corrected,
                flatten_window_days,
            )
            pieces.append((corrected_time, corrected_flux, segment))
            for candidate in protected:
                arguments = {
                    "period_days": float(candidate["period_days"]),
                    "epoch_btjd": float(candidate["epoch_btjd"]),
                    "duration_hours": float(candidate["duration_hours"]),
                }
                before = measure_fixed_ephemeris(raw_time, raw_flux, **arguments)
                after = measure_fixed_ephemeris(corrected_time, corrected_flux, **arguments)
                before_depth = before.get("depth_ppm")
                after_depth = after.get("depth_ppm")
                ratio = (
                    float(after_depth / before_depth)
                    if before_depth is not None
                    and after_depth is not None
                    and before_depth > 0
                    else None
                )
                preservation.append(
                    {
                        "file": path.name,
                        "segment": segment,
                        "period_days": arguments["period_days"],
                        "raw_aperture": before,
                        "protected_pld": after,
                        "measured_depth_ratio": ratio,
                        "is_injection_recovery": False,
                    }
                )
            diagnostics.append(
                {
                    "branch": "tpf_pld_protected_bls",
                    "file": path.name,
                    "status": "ready",
                    "segment": segment,
                    "segment_kind": segment_kind,
                    "cadences": int(corrected_time.size),
                    "fit_cadences": int(fit_mask.sum()),
                    "protected_cadences": int((~fit_mask).sum()),
                    "aperture_pixels": int(aperture.sum()),
                }
            )
        except Exception as exc:
            diagnostics.append(
                {
                    "branch": "tpf_pld_protected_bls",
                    "file": path.name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if not pieces:
        return None, diagnostics
    branch = _assemble_branch(
        "tpf_pld_protected_bls",
        "TPF transit-protected PLD + BLS follow-up",
        "PLD order 1; 3 PCA components; all initial candidate windows excluded from fit",
        "tpf_custom_aperture",
        pieces,
        can_seed_candidates=False,
        metadata={
            "conditioned_on_initial_candidates": True,
            "depth_preservation_measurements": preservation,
            "injection_recovery_completed": False,
        },
    )
    return branch, diagnostics


def _period_grid(period_min: float, period_max: float, n_periods: int) -> np.ndarray:
    frequency = np.linspace(1.0 / period_max, 1.0 / period_min, n_periods)
    return np.sort(1.0 / frequency)


def _same_period_family(
    period: float,
    previous: Iterable[float],
    *,
    suppress_harmonics: bool = True,
) -> bool:
    for old in previous:
        ratio = period / old
        if abs(ratio - 1.0) < 0.012:
            return True
        if not suppress_harmonics:
            continue
        for harmonic in (2, 3, 4, 5):
            if abs(ratio - harmonic) / harmonic < 0.003:
                return True
            if abs(ratio - 1.0 / harmonic) / (1.0 / harmonic) < 0.003:
                return True
    return False


def _seed_refinement_half_width(period: float) -> float:
    """Cover the full coarse-seed exclusion neighborhood.

    Coarse peaks within two percent are deliberately represented by one seed.
    The old 1.2% fine window could therefore miss a stronger narrow peak that
    had been suppressed only because it lay 1.2--2.0% from that representative
    seed.  A small margin above two percent closes that blind annulus.
    """
    return max(0.021 * float(period), 0.015)


def _fine_local_peak_indices(
    periods: np.ndarray,
    power: np.ndarray,
    *,
    max_peaks: int | None = 8,
    minimum_relative_separation: float = 0.0025,
    absolute_power_tolerance: float = 0.0,
) -> list[int]:
    """Choose several distinct local maxima from one fine-search window.

    Looking only at the absolute maximum loses a valid transit peak whenever a
    nearby systematic has slightly higher BLS power but fails the subsequent
    SNR/event checks.  Returning a short, separated list lets those checks fall
    through to the next local maximum without flooding the seed list with grid
    cells from the same peak.
    """
    periods = np.asarray(periods, dtype=float)
    power = np.asarray(power, dtype=float)
    if periods.ndim != 1 or power.ndim != 1 or periods.size != power.size:
        raise ValueError("Fine periods and powers must be aligned one-dimensional arrays")
    if periods.size == 0:
        return []
    tolerance = max(0.0, float(absolute_power_tolerance))
    finite = np.isfinite(power)
    local = np.zeros(power.size, dtype=bool)
    if power.size == 1:
        local[0] = finite[0]
    else:
        local[0] = finite[0] and (
            not finite[1] or power[0] + tolerance >= power[1]
        )
        local[-1] = finite[-1] and (
            not finite[-2] or power[-1] + tolerance >= power[-2]
        )
        local[1:-1] = (
            finite[1:-1]
            & (~finite[:-2] | (power[1:-1] + tolerance >= power[:-2]))
            & (~finite[2:] | (power[1:-1] + tolerance >= power[2:]))
        )
    indices = np.flatnonzero(local)
    if not indices.size and np.any(finite):
        indices = np.asarray([int(np.nanargmax(power))])
    ordered = indices[np.argsort(power[indices])[::-1]]
    # Complete mode requests zero period-only separation because phase, not
    # period distance, decides whether two peaks are the same event train.
    # The old loop still compared every new peak with every earlier peak even
    # though ``abs(...) < 0`` can never be true.  Returning the already sorted
    # local maxima is exactly equivalent and removes a pure O(N^2) no-op.
    if minimum_relative_separation <= 0:
        if max_peaks is not None:
            ordered = ordered[:max_peaks]
        return [int(index) for index in ordered]
    selected: list[int] = []
    for index in ordered:
        period = float(periods[index])
        if any(
            abs(period - float(periods[old])) / float(periods[old])
            < minimum_relative_separation
            for old in selected
        ):
            continue
        selected.append(int(index))
        if max_peaks is not None and len(selected) >= max_peaks:
            break
    return selected


def _distinct_ephemeris_peak_indices(
    indices: Iterable[int],
    periods: np.ndarray,
    epochs: np.ndarray,
    durations: np.ndarray,
    *,
    observation_baseline_days: float,
) -> list[int]:
    """Collapse grid duplicates only when they predict the same event train.

    Period distance alone is not candidate identity: two extremely close
    periods can have different phases and therefore predict disjoint transit
    windows.  Conversely, a flat BLS peak can produce hundreds of adjacent
    grid maxima with the same epoch, duration, and observed events.  Preserve
    the former and collapse only the latter.
    """
    periods = np.asarray(periods, dtype=float)
    epochs = np.asarray(epochs, dtype=float)
    durations = np.asarray(durations, dtype=float)
    selected: list[int] = []
    baseline = max(0.0, float(observation_baseline_days))
    for raw_index in indices:
        index = int(raw_index)
        period = float(periods[index])
        epoch = float(epochs[index])
        duration = float(durations[index])
        duplicate = False
        for old in selected:
            old_period = float(periods[old])
            reference_period = 0.5 * (period + old_period)
            phase_distance = abs(
                (
                    epoch
                    - float(epochs[old])
                    + 0.5 * reference_period
                )
                % reference_period
                - 0.5 * reference_period
            )
            event_tolerance = 0.5 * max(duration, float(durations[old]))
            accumulated_drift = (
                baseline * abs(period - old_period) / reference_period
            )
            if (
                phase_distance <= event_tolerance
                and accumulated_drift <= event_tolerance
            ):
                duplicate = True
                break
        if not duplicate:
            selected.append(index)
    return selected


def search_branch_bls(
    branch: SearchBranch,
    *,
    holdout_segment: int | None,
    period_min: float = 0.5,
    period_max: float = 40.0,
    n_periods: int = 18_000,
    max_candidates: int | None = 6,
    sde_threshold: float = 6.0,
    snr_threshold: float = 5.0,
    min_events: int = 2,
    durations_days: Iterable[float] | None = None,
    max_seed_periods: int | None = 30,
    suppress_harmonics: bool = True,
    retain_harmonic_hypotheses: bool = False,
    max_harmonic_hypotheses_per_round: int | None = 2,
    max_independent_hypotheses_before_masking: int | None = 0,
    preserve_all_local_peaks: bool = False,
    retain_all_ephemeris_peaks: bool = False,
    coarse_power_provider: Callable[..., np.ndarray] | None = None,
    fine_power_provider: Callable[..., np.ndarray] | None = None,
    fine_power_correction_provider: Callable[..., np.ndarray] | None = None,
    fine_solution_provider: Callable[..., Mapping] | None = None,
    fine_peak_absolute_power_tolerance: float = 0.0,
) -> tuple[list[dict], list[dict]]:
    """Run BLS independently on one branch using search segments only."""
    search_mask = (
        np.ones(branch.time.size, dtype=bool)
        if holdout_segment is None
        else branch.segment_id != int(holdout_segment)
    )
    holdout_mask = ~search_mask if holdout_segment is not None else np.zeros_like(search_mask)
    time = branch.time[search_mask]
    flux = branch.flux[search_mask]
    if time.size < 500:
        return [], [{"branch": branch.key, "status": "insufficient_search_cadences"}]
    active = np.ones(time.size, dtype=bool)
    durations = np.asarray(
        list(durations_days)
        if durations_days is not None
        else [0.035, 0.05, 0.07, 0.09, 0.12, 0.16, 0.21],
        dtype=float,
    )
    if durations.size == 0 or not np.all(np.isfinite(durations)):
        raise ValueError("BLS duration grid must contain finite values")
    if np.any(durations <= 0) or np.any(durations >= period_min):
        raise ValueError("Every BLS duration must be positive and shorter than period_min")
    if max_seed_periods is not None and max_seed_periods < 1:
        raise ValueError("max_seed_periods must be positive")
    if (
        max_harmonic_hypotheses_per_round is not None
        and max_harmonic_hypotheses_per_round < 0
    ):
        raise ValueError("max_harmonic_hypotheses_per_round cannot be negative")
    if (
        max_independent_hypotheses_before_masking is not None
        and max_independent_hypotheses_before_masking < 0
    ):
        raise ValueError("max_independent_hypotheses_before_masking cannot be negative")
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    periods = _period_grid(period_min, period_max, n_periods)
    candidates: list[dict] = []
    # Harmonic hypotheses are extra evidence, not iterative-mask primaries.
    # Keeping this list separate guarantees that enabling hypothesis retention
    # cannot change which legacy primary periods are selected and masked.
    primary_periods: list[float] = []
    diagnostics: list[dict] = []
    round_index = 0
    while max_candidates is None or round_index < max_candidates:
        use_time = time[active]
        use_flux = flux[active]
        if use_time.size < 500:
            break
        scatter = _mad_sigma(use_flux - np.nanmedian(use_flux))
        if scatter <= 0:
            break
        # BLS is invariant to an additive flux constant.  Center explicitly in
        # float64 so ARM, x86, and CUDA do not obtain different strict local
        # maxima from subtracting two large numbers close to normalized flux
        # one.  This is a numerical-definition fix, not a scientific filter.
        bls_flux = np.asarray(
            np.asarray(use_flux, dtype=np.float64)
            - float(np.nanmedian(use_flux)),
            dtype=np.float64,
        )
        bls = BoxLeastSquares(
            use_time * u.day,
            bls_flux,
            dy=np.full(use_time.size, scatter),
        )
        if coarse_power_provider is None:
            coarse = bls.power(periods * u.day, durations * u.day, method="fast")
            coarse_power = np.asarray(coarse.power, dtype=float)
            coarse_period_values = np.asarray(
                coarse.period.to_value(u.day), dtype=float
            )
        else:
            # An accelerator may propose the broad-search spectrum, but it
            # never supplies the locked ephemeris.  Every retained local peak
            # is still refined by the unchanged double-precision Astropy BLS
            # grid below and then measured by the normal CPU validation path.
            coarse_power = np.asarray(
                coarse_power_provider(
                    time=use_time,
                    flux=bls_flux,
                    scatter=float(scatter),
                    periods=periods,
                    durations=durations,
                ),
                dtype=float,
            )
            if coarse_power.shape != periods.shape:
                raise ValueError(
                    "coarse_power_provider must return one value per period"
                )
            coarse_period_values = periods
        baseline = float(np.nanmedian(coarse_power))
        power_scatter = _mad_sigma(coarse_power)
        seeds: list[float] = []
        for index in np.argsort(coarse_power)[::-1]:
            period = float(coarse_period_values[index])
            # Broad stellar/systematic peaks can occupy hundreds of adjacent
            # grid cells.  A 2% exclusion radius makes room for genuinely
            # different periods instead of spending every refinement on one
            # wide long-period hump.
            if any(abs(period - old) / old < 0.02 for old in seeds):
                continue
            seeds.append(period)
            if max_seed_periods is not None and len(seeds) >= max_seed_periods:
                break
        selected = None
        round_attempts = []
        if max_seed_periods is None:
            # Exhaustive mode still refines every separated coarse peak, but
            # evaluates the union as one BLS grid.  Calling BLS hundreds of
            # times with overlapping windows repeats the same time-series
            # setup and is orders of magnitude slower without adding a single
            # hypothesis.
            fine_windows = []
            for seed in seeds:
                half_width = _seed_refinement_half_width(seed)
                fine_windows.append(
                    np.linspace(
                        max(period_min, seed - half_width),
                        min(period_max, seed + half_width),
                        1400,
                    )
                )
            search_grids = (
                [np.unique(np.concatenate(fine_windows))]
                if fine_windows
                else []
            )
        else:
            search_grids = []
            for seed in seeds:
                half_width = _seed_refinement_half_width(seed)
                search_grids.append(
                    np.linspace(
                        max(period_min, seed - half_width),
                        min(period_max, seed + half_width),
                        1400,
                    )
                )
        for fine_periods in search_grids:
            if fine_solution_provider is None:
                fine = bls.power(
                    fine_periods * u.day, durations * u.day, method="fast"
                )
                fine_period_values = np.asarray(
                    fine.period.to_value(u.day), dtype=float
                )
                fine_power = np.asarray(fine.power, dtype=float)
                fine_epochs = np.asarray(
                    fine.transit_time.to_value(u.day), dtype=float
                )
                fine_durations = np.asarray(
                    fine.duration.to_value(u.day), dtype=float
                )
            else:
                fine_period_values = np.asarray(fine_periods, dtype=float)
                fine_power_function = (
                    fine_power_provider or coarse_power_provider
                )
                if fine_power_function is None:
                    fine_power = np.asarray(
                        bls.power(
                            fine_periods * u.day,
                            durations * u.day,
                            method="fast",
                        ).power,
                        dtype=float,
                    )
                else:
                    fine_power = np.asarray(
                        fine_power_function(
                            time=use_time,
                            flux=bls_flux,
                            scatter=float(scatter),
                            periods=fine_periods,
                            durations=durations,
                        ),
                        dtype=float,
                    )
                if fine_power.shape != fine_period_values.shape:
                    raise ValueError(
                        "fine power provider must return one value per period"
                    )
                if fine_power_correction_provider is not None:
                    fine_power = np.asarray(
                        fine_power_correction_provider(
                            time=use_time,
                            flux=bls_flux,
                            scatter=float(scatter),
                            periods=fine_period_values,
                            durations=durations,
                            power=fine_power,
                            baseline=baseline,
                            power_scatter=power_scatter,
                            sde_threshold=float(sde_threshold),
                        ),
                        dtype=float,
                    )
                    if fine_power.shape != fine_period_values.shape:
                        raise ValueError(
                            "fine power correction provider must return one "
                            "value per period"
                        )
                fine_epochs = np.full(fine_period_values.shape, np.nan)
                fine_durations = np.full(fine_period_values.shape, np.nan)
            peak_indices = _fine_local_peak_indices(
                fine_period_values,
                fine_power,
                max_peaks=(
                    None
                    if max_independent_hypotheses_before_masking is None
                    else 8
                ),
                # In exhaustive mode, nearby peaks with different epochs are
                # different hypotheses.  A fixed period-only exclusion radius
                # previously let a stronger noise peak at 13.895 d delete a
                # real 13.900 d peak before either reached sealed validation.
                minimum_relative_separation=(
                    0.0 if preserve_all_local_peaks else 0.0025
                ),
                absolute_power_tolerance=fine_peak_absolute_power_tolerance,
            )
            if fine_solution_provider is not None:
                # Power is evaluated at every configured fine-grid period.
                # Phase/duration are needed only at local maxima that can pass
                # the unchanged SDE gate.  This is not a Top-K truncation:
                # every qualifying local maximum receives a CUDA solution.
                peak_indices = [
                    int(index)
                    for index in peak_indices
                    if power_scatter > 0
                    and (
                        float(fine_power[int(index)]) - baseline
                    )
                    / power_scatter
                    >= sde_threshold
                ]
                if peak_indices:
                    solved = fine_solution_provider(
                        time=use_time,
                        flux=bls_flux,
                        scatter=float(scatter),
                        periods=fine_period_values[peak_indices],
                        durations=durations,
                    )
                    solved_power = np.asarray(solved["power"], dtype=float)
                    solved_epochs = np.asarray(
                        solved["epoch_btjd"], dtype=float
                    )
                    solved_durations = np.asarray(
                        solved["duration_days"], dtype=float
                    )
                    expected = (len(peak_indices),)
                    if not (
                        solved_power.shape == expected
                        and solved_epochs.shape == expected
                        and solved_durations.shape == expected
                    ):
                        raise ValueError(
                            "fine_solution_provider arrays must align with peaks"
                        )
                    fine_power[np.asarray(peak_indices, dtype=int)] = solved_power
                    fine_epochs[np.asarray(peak_indices, dtype=int)] = solved_epochs
                    fine_durations[
                        np.asarray(peak_indices, dtype=int)
                    ] = solved_durations
            if preserve_all_local_peaks and not retain_all_ephemeris_peaks:
                peak_indices = _distinct_ephemeris_peak_indices(
                    peak_indices,
                    fine_period_values,
                    fine_epochs,
                    fine_durations,
                    observation_baseline_days=(
                        float(np.nanmax(use_time) - np.nanmin(use_time))
                    ),
                )

            def peak_descriptor(index: int) -> dict | None:
                period = float(fine_period_values[index])
                epoch = float(fine_epochs[index])
                duration_hours = float(fine_durations[index] * 24.0)
                sde = (
                    (float(fine_power[index]) - baseline) / power_scatter
                    if power_scatter > 0
                    else 0.0
                )
                # SDE is a necessary acceptance condition and is already
                # available from BLS.  Avoid the much more expensive fixed-
                # ephemeris measurement for a peak that cannot possibly pass.
                if sde < sde_threshold:
                    return None
                return {
                    "period_days": period,
                    "epoch_btjd": epoch,
                    "duration_hours": duration_hours,
                    "robust_sde": float(sde),
                }

            def record_peak(descriptor: Mapping, metrics: Mapping) -> bool:
                nonlocal selected
                period = float(descriptor["period_days"])
                epoch = float(descriptor["epoch_btjd"])
                duration_hours = float(descriptor["duration_hours"])
                sde = float(descriptor["robust_sde"])
                duplicate = _same_period_family(
                    period,
                    primary_periods,
                    suppress_harmonics=suppress_harmonics,
                )
                accepted = (
                    not duplicate
                    and sde >= sde_threshold
                    and (metrics["snr"] or 0.0) >= snr_threshold
                    and metrics["events"] >= min_events
                    and duration_hours / 24.0 / period <= 0.15
                )
                attempt = {
                    "branch": branch.key,
                    "round": round_index + 1,
                    "period_days": period,
                    "epoch_btjd": epoch,
                    "duration_hours": duration_hours,
                    "robust_sde": float(sde),
                    "search_metrics": metrics,
                    "duplicate_or_harmonic": duplicate,
                    "accepted": accepted,
                }
                round_attempts.append(attempt)
                if accepted and selected is None:
                    selected = attempt
                return bool(accepted)

            if max_independent_hypotheses_before_masking is None:
                descriptors = [
                    descriptor
                    for index in peak_indices
                    if (descriptor := peak_descriptor(int(index))) is not None
                ]
                metric_rows = measure_fixed_ephemerides_compiled(
                    use_time,
                    use_flux,
                    descriptors,
                )
                if metric_rows is None:
                    metric_rows = [
                        measure_fixed_ephemeris(
                            use_time,
                            use_flux,
                            period_days=float(descriptor["period_days"]),
                            epoch_btjd=float(descriptor["epoch_btjd"]),
                            duration_hours=float(descriptor["duration_hours"]),
                        )
                        for descriptor in descriptors
                    ]
                for descriptor, metrics in zip(descriptors, metric_rows):
                    record_peak(descriptor, metrics)
            else:
                for index in peak_indices:
                    descriptor = peak_descriptor(int(index))
                    if descriptor is None:
                        continue
                    metrics = measure_fixed_ephemeris(
                        use_time,
                        use_flux,
                        period_days=float(descriptor["period_days"]),
                        epoch_btjd=float(descriptor["epoch_btjd"]),
                        duration_hours=float(descriptor["duration_hours"]),
                    )
                    accepted = record_peak(descriptor, metrics)
                    if accepted:
                        break
            if selected is not None and (
                max_independent_hypotheses_before_masking is not None
                and (
                    round_index > 0
                    or max_independent_hypotheses_before_masking == 0
                )
            ):
                break
        diagnostics.extend(round_attempts)
        if selected is None:
            break

        # Preserve several strong, non-harmonic alternatives from the original
        # unmasked light curve.  Iterative masking is intentionally driven only
        # by ``selected``; otherwise a high-power systematic can remove the
        # cadences of a real second planet before that planet gets a turn.
        independent_hypotheses = []
        if (
            max_independent_hypotheses_before_masking is None
            or (round_index == 0 and max_independent_hypotheses_before_masking)
        ):
            accepted_attempts = sorted(
                (
                    item
                    for item in round_attempts
                    if item.get("accepted") and item is not selected
                ),
                key=lambda item: (
                    float(item["search_metrics"].get("snr") or 0.0)
                    + 0.35 * float(item.get("robust_sde") or 0.0)
                ),
                reverse=True,
            )
            if preserve_all_local_peaks:
                independent_hypotheses = accepted_attempts
            else:
                distinct_periods = [float(selected["period_days"])]
                for attempt in accepted_attempts:
                    period = float(attempt["period_days"])
                    if _same_period_family(
                        period,
                        distinct_periods,
                        # Integer-related periods must remain separate hypotheses;
                        # odd/even and secondary-eclipse evidence decides between
                        # P, P/2 and 2P later.
                        suppress_harmonics=False,
                    ):
                        continue
                    independent_hypotheses.append(attempt)
                    distinct_periods.append(period)
                    if (
                        max_independent_hypotheses_before_masking is not None
                        and
                        len(independent_hypotheses)
                        >= max_independent_hypotheses_before_masking
                    ):
                        break

        # A box periodogram commonly gives its highest peak at P/2 or 2P.  The
        # legacy search immediately masked that peak and could therefore erase
        # the evidence needed to retain the physically correct period.  Newer
        # profiles explicitly measure a small set of integer-related periods
        # before masking.  They remain separate hypotheses; downstream
        # odd/even and secondary-eclipse tests decide which one is preferable.
        harmonic_hypotheses = []
        if retain_harmonic_hypotheses and max_harmonic_hypotheses_per_round != 0:
            related_periods = []
            for factor in (0.5, 2.0, 0.25, 4.0):
                value = float(selected["period_days"]) * factor
                if not (period_min <= value <= period_max):
                    continue
                if any(abs(value - old) / old < 0.012 for old in related_periods):
                    continue
                related_periods.append(value)
            for related in related_periods:
                half_width = max(0.012 * related, 0.015)
                fine_periods = np.linspace(
                    max(period_min, related - half_width),
                    min(period_max, related + half_width),
                    1400,
                )
                if fine_solution_provider is None:
                    fine = bls.power(
                        fine_periods * u.day,
                        durations * u.day,
                        method="fast",
                    )
                    related_power = np.asarray(fine.power, dtype=float)
                    related_epochs = np.asarray(
                        fine.transit_time.to_value(u.day), dtype=float
                    )
                    related_durations = np.asarray(
                        fine.duration.to_value(u.day), dtype=float
                    )
                else:
                    fine_power_function = (
                        fine_power_provider or coarse_power_provider
                    )
                    if fine_power_function is None:
                        related_power = np.asarray(
                            bls.power(
                                fine_periods * u.day,
                                durations * u.day,
                                method="fast",
                            ).power,
                            dtype=float,
                        )
                    else:
                        related_power = np.asarray(
                            fine_power_function(
                                time=use_time,
                                flux=bls_flux,
                                scatter=float(scatter),
                                periods=fine_periods,
                                durations=durations,
                            ),
                            dtype=float,
                        )
                    index = int(np.nanargmax(related_power))
                    solved = fine_solution_provider(
                        time=use_time,
                        flux=bls_flux,
                        scatter=float(scatter),
                        periods=np.asarray([fine_periods[index]], dtype=float),
                        durations=durations,
                    )
                    related_epochs = np.asarray(
                        np.full(fine_periods.shape, np.nan), dtype=float
                    )
                    related_durations = np.asarray(
                        np.full(fine_periods.shape, np.nan), dtype=float
                    )
                    related_epochs[index] = float(
                        np.asarray(solved["epoch_btjd"], dtype=float)[0]
                    )
                    related_durations[index] = float(
                        np.asarray(solved["duration_days"], dtype=float)[0]
                    )
                index = int(np.nanargmax(related_power))
                period = float(fine_periods[index])
                epoch = float(related_epochs[index])
                duration_hours = float(related_durations[index] * 24.0)
                sde = (
                    (float(related_power[index]) - baseline) / power_scatter
                    if power_scatter > 0
                    else 0.0
                )
                metrics = measure_fixed_ephemeris(
                    use_time,
                    use_flux,
                    period_days=period,
                    epoch_btjd=epoch,
                    duration_hours=duration_hours,
                )
                duplicate = _same_period_family(
                    period,
                    [item["period_days"] for item in candidates]
                    + [selected["period_days"]]
                    + [item["period_days"] for item in harmonic_hypotheses],
                    suppress_harmonics=False,
                )
                accepted = (
                    not duplicate
                    and sde >= sde_threshold
                    and (metrics["snr"] or 0.0) >= snr_threshold
                    and metrics["events"] >= min_events
                    and duration_hours / 24.0 / period <= 0.15
                )
                attempt = {
                    "branch": branch.key,
                    "round": round_index + 1,
                    "period_days": period,
                    "epoch_btjd": epoch,
                    "duration_hours": duration_hours,
                    "robust_sde": float(sde),
                    "search_metrics": metrics,
                    "duplicate_or_harmonic": duplicate,
                    "accepted": accepted,
                    "hypothesis_source": "integer_related_period_before_masking",
                    "related_to_period_days": float(selected["period_days"]),
                }
                diagnostics.append(attempt)
                if accepted:
                    harmonic_hypotheses.append(attempt)
            harmonic_hypotheses.sort(
                key=lambda item: (
                    float(item["search_metrics"].get("snr") or 0.0)
                    + 0.35 * float(item.get("robust_sde") or 0.0)
                ),
                reverse=True,
            )
            if max_harmonic_hypotheses_per_round is not None:
                harmonic_hypotheses = harmonic_hypotheses[
                    :max_harmonic_hypotheses_per_round
                ]

        def append_candidate(
            attempt: Mapping,
            hypothesis_of: float | None = None,
            hypothesis_source: str | None = None,
        ) -> None:
            holdout = measure_fixed_ephemeris(
                branch.time[holdout_mask],
                branch.flux[holdout_mask],
                period_days=attempt["period_days"],
                epoch_btjd=attempt["epoch_btjd"],
                duration_hours=attempt["duration_hours"],
            )
            candidate = {
                "branch": branch.key,
                "branch_label": branch.label,
                "method_family": branch.method_family,
                "photometry_family": branch.photometry_family,
                "can_seed_candidates": branch.can_seed_candidates,
                "rank_in_branch": len(candidates) + 1,
                "period_days": attempt["period_days"],
                "epoch_btjd": attempt["epoch_btjd"],
                "duration_hours": attempt["duration_hours"],
                "robust_sde": attempt["robust_sde"],
                "search_metrics": attempt["search_metrics"],
                "holdout_metrics": holdout,
                "hard_rejected": False,
            }
            if hypothesis_source is not None:
                candidate["hypothesis_source"] = hypothesis_source
            if hypothesis_of is not None:
                candidate["related_to_period_days"] = float(hypothesis_of)
            candidates.append(candidate)

        append_candidate(selected)
        primary_periods.append(float(selected["period_days"]))
        for hypothesis in independent_hypotheses:
            append_candidate(
                hypothesis,
                hypothesis_source="independent_period_before_masking",
            )
        for hypothesis in harmonic_hypotheses:
            append_candidate(
                hypothesis,
                float(selected["period_days"]),
                "integer_related_period_before_masking",
            )
        if max_candidates is None:
            # Exhaustive mode has already retained every threshold-passing
            # independent peak from the untouched branch.  Do not let one
            # single-branch hypothesis destructively erase the others.  The
            # higher-level consensus-residual pass masks all corroborated
            # signals together and then performs another complete scan.
            break
        phase = (
            time
            - selected["epoch_btjd"]
            + 0.5 * selected["period_days"]
        ) % selected["period_days"] - 0.5 * selected["period_days"]
        remove = active & (
            np.abs(phase) <= max(selected["duration_hours"] / 24.0, 0.04)
        )
        if remove.sum() < 3:
            break
        active[remove] = False
        round_index += 1
    return candidates, diagnostics


def event_first_candidates(
    branch: SearchBranch,
    *,
    holdout_segment: int | None,
    period_min: float = 0.5,
    period_max: float = 40.0,
    min_event_snr: float = 6.0,
    exhaustive_hypotheses: bool = False,
) -> tuple[list[dict], dict]:
    """Run event-first on search segments only and return periodic candidates."""
    search_mask = (
        np.ones(branch.time.size, dtype=bool)
        if holdout_segment is None
        else branch.segment_id != int(holdout_segment)
    )
    holdout_mask = ~search_mask if holdout_segment is not None else np.zeros_like(search_mask)
    config = EventSearchConfig(
        # Complete mode groups marginal individual dips because a coherent
        # train can be highly significant even when one member is below the
        # interactive single-event threshold.  Final fixed-ephemeris and
        # sealed tests still enforce the candidate-level significance gates.
        min_event_snr=min(float(min_event_snr), 3.0) if exhaustive_hypotheses else min_event_snr,
        min_period_days=period_min,
        max_period_days=period_max,
        max_events=None if exhaustive_hypotheses else 160,
        max_periodic_candidates=None if exhaustive_hypotheses else 20,
        max_seed_bins=None if exhaustive_hypotheses else 2000,
        materialize_double_candidates=not exhaustive_hypotheses,
    )
    result = event_first_search(branch.time[search_mask], branch.flux[search_mask], config)
    candidates = []
    for rank, item in enumerate(result.periodic, start=1):
        holdout = measure_fixed_ephemeris(
            branch.time[holdout_mask],
            branch.flux[holdout_mask],
            period_days=item.period_days,
            epoch_btjd=item.epoch_btjd,
            duration_hours=item.duration_hours,
        )
        candidates.append(
            {
                "branch": "event_first_pdcsap",
                "branch_label": "PDCSAP strict search-segment event-first",
                "method_family": "event_first",
                "photometry_family": branch.photometry_family,
                "can_seed_candidates": True,
                "rank_in_branch": rank,
                "period_days": float(item.period_days),
                "epoch_btjd": float(item.epoch_btjd),
                "duration_hours": float(item.duration_hours),
                "robust_sde": None,
                "search_metrics": {
                    "depth_ppm": float(item.depth_ppm),
                    "snr": float(item.median_event_snr),
                    "events": int(item.instances),
                    "status": "supported" if item.median_event_snr >= 5 else "weak_support",
                    "event_first_score": float(item.score),
                    "mean_abs_ttv_minutes": float(item.mean_abs_ttv_minutes),
                },
                "holdout_metrics": holdout,
                "hard_rejected": False,
            }
        )
    nonperiodic = {
        "doubles": [dict(item) for item in result.doubles],
        "double_candidate_count": int(result.double_candidate_count),
        "double_candidates_materialized": bool(
            result.double_candidates_materialized
        ),
        "periodic_minimum_instances": int(result.periodic_minimum_instances),
        "sparse_trains_reconstructible_from_event_ledger": bool(
            result.sparse_trains_reconstructible_from_event_ledger
        ),
        "adaptive_recurrence_sigma": result.adaptive_recurrence_sigma,
        "singles": [item.to_dict() for item in result.singles],
        "all_events": len(result.all_events),
        "event_ledger": [item.to_dict() for item in result.all_events],
        "policy": (
            "Every detected event is retained. Double-event intervals are "
            "non-unique diagnostics; exhaustive runs store their exact count "
            "and reconstruct them losslessly from the event ledger instead of "
            "materializing a quadratic dictionary list. High-activity light "
            "curves keep every event and every uncapped BLS peak, while sparse "
            "chance event trains below the reported recurrence threshold remain "
            "reconstructible from the same event ledger."
        ),
    }
    return candidates, nonperiodic


def _phase_distance(epoch_a: float, epoch_b: float, period: float) -> float:
    return float(abs((epoch_a - epoch_b + 0.5 * period) % period - 0.5 * period))


def _matches_cluster(candidate: Mapping, members: list[Mapping], tolerance: float) -> bool:
    periods = np.asarray([float(member["period_days"]) for member in members])
    period = float(np.median(periods))
    candidate_period = float(candidate["period_days"])
    if abs(candidate_period - period) / period > tolerance:
        return False
    reference = members[0]
    phase_delta = _phase_distance(
        float(candidate["epoch_btjd"]),
        float(reference["epoch_btjd"]),
        0.5 * (candidate_period + period),
    )
    duration_days = max(
        float(candidate["duration_hours"]),
        float(reference["duration_hours"]),
    ) / 24.0
    return phase_delta <= min(0.15 * period, max(2.0 * duration_days, 0.02 * period))


def _candidate_strength(candidate: Mapping) -> float:
    snr = candidate.get("search_metrics", {}).get("snr") or 0.0
    sde = candidate.get("robust_sde") or 0.0
    return float(max(snr, 0.0) + 0.35 * max(sde, 0.0))


def fuse_branch_candidates(
    candidates: Iterable[Mapping],
    *,
    period_tolerance_fraction: float = 0.01,
) -> list[dict]:
    """Cluster independently generated candidates by period and epoch."""
    ordered = sorted(
        (dict(item) for item in candidates),
        key=_candidate_strength,
        reverse=True,
    )
    seedable = [item for item in ordered if item.get("can_seed_candidates", True)]
    conditioned = [item for item in ordered if not item.get("can_seed_candidates", True)]
    clusters: list[list[dict]] = []
    for candidate in seedable:
        match = next(
            (
                members
                for members in clusters
                if _matches_cluster(candidate, members, period_tolerance_fraction)
            ),
            None,
        )
        if match is None:
            clusters.append([candidate])
        elif candidate["branch"] not in {item["branch"] for item in match}:
            match.append(candidate)
    unmatched_conditioned = []
    for candidate in conditioned:
        match = next(
            (
                members
                for members in clusters
                if _matches_cluster(candidate, members, period_tolerance_fraction)
            ),
            None,
        )
        if match is None:
            unmatched_conditioned.append(candidate)
        elif candidate["branch"] not in {item["branch"] for item in match}:
            match.append(candidate)

    fused = []
    for members in clusters:
        weights = np.asarray([max(_candidate_strength(item), 1.0) for item in members])
        periods = np.asarray([float(item["period_days"]) for item in members])
        durations = np.asarray([float(item["duration_hours"]) for item in members])
        strongest = max(members, key=_candidate_strength)
        period = float(np.average(periods, weights=weights))
        reference_epoch = float(strongest["epoch_btjd"])
        aligned_epochs = []
        for item in members:
            epoch = float(item["epoch_btjd"])
            cycle = round((reference_epoch - epoch) / period)
            aligned_epochs.append(epoch + cycle * period)
        epoch = float(np.average(aligned_epochs, weights=weights))
        branch_keys = sorted({str(item["branch"]) for item in members})
        methods = sorted({str(item["method_family"]) for item in members})
        photometry = sorted({str(item["photometry_family"]) for item in members})
        if len(photometry) >= 2:
            status = "multi_photometry_supported"
        elif len(branch_keys) >= 2:
            status = "multi_processing_supported"
        else:
            status = "single_branch_candidate"
        holdout_supported = [
            item["branch"]
            for item in members
            if item.get("holdout_metrics", {}).get("status") in {"supported", "weak_support"}
        ]
        score = (
            max(_candidate_strength(item) for item in members)
            + 2.0 * np.log1p(len(branch_keys))
            + 2.5 * max(0, len(photometry) - 1)
            + 1.5 * max(0, len(methods) - 1)
            + 0.75 * len(holdout_supported)
        )
        fused.append(
            {
                "period_days": period,
                "epoch_btjd": epoch,
                "duration_hours": float(np.average(durations, weights=weights)),
                "status": status,
                "supporting_branches": branch_keys,
                "method_families": methods,
                "photometry_families": photometry,
                "holdout_supporting_branches": sorted(holdout_supported),
                "uncalibrated_priority_score": float(score),
                "score_is_probability": False,
                "hard_rejected": False,
                "members": members,
            }
        )
    fused.sort(key=lambda item: item["uncalibrated_priority_score"], reverse=True)
    for rank, cluster in enumerate(fused, start=1):
        cluster["rank"] = rank
    if unmatched_conditioned:
        for item in unmatched_conditioned:
            item["fusion_note"] = "conditioned branch cannot create a candidate by itself"
    return fused


def measure_clusters_on_all_branches(
    clusters: Iterable[Mapping],
    branches: Mapping[str, SearchBranch],
    *,
    holdout_segment: int | None,
) -> list[dict]:
    """Measure fused ephemerides on every branch without changing membership."""
    output = []
    for raw_cluster in clusters:
        cluster = dict(raw_cluster)
        evidence = {}
        for key, branch in branches.items():
            search = (
                np.ones(branch.time.size, dtype=bool)
                if holdout_segment is None
                else branch.segment_id != int(holdout_segment)
            )
            holdout = ~search if holdout_segment is not None else np.zeros_like(search)
            arguments = {
                "period_days": float(cluster["period_days"]),
                "epoch_btjd": float(cluster["epoch_btjd"]),
                "duration_hours": float(cluster["duration_hours"]),
            }
            evidence[key] = {
                "search": measure_fixed_ephemeris(
                    branch.time[search], branch.flux[search], **arguments
                ),
                "holdout": measure_fixed_ephemeris(
                    branch.time[holdout], branch.flux[holdout], **arguments
                ),
                "photometry_family": branch.photometry_family,
                "processing": branch.processing,
            }
        cluster["all_branch_fixed_ephemeris_evidence"] = evidence
        output.append(cluster)
    return output


def run_postprocessing_injection_check(
    branch: SearchBranch,
    *,
    holdout_segment: int | None,
    period_days: float = 8.371,
    duration_hours: float = 2.5,
    depth_ppm: float = 1200.0,
    n_periods: int = 12_000,
) -> dict:
    """Inject a box into an already-produced branch and test BLS recovery.

    This diagnoses the search/fusion stage only.  It does *not* prove that an
    upstream correction such as CBV or PLD preserved an injection, because the
    synthetic signal is added after that correction; the scope is explicit in
    the returned record.
    """
    search = (
        np.ones(branch.time.size, dtype=bool)
        if holdout_segment is None
        else branch.segment_id != int(holdout_segment)
    )
    if search.sum() < 500:
        return {"status": "insufficient_search_cadences", "branch": branch.key}
    epoch = float(np.nanmin(branch.time[search]) + 0.37 * period_days)
    phase = (branch.time - epoch + 0.5 * period_days) % period_days - 0.5 * period_days
    inside = np.abs(phase) <= duration_hours / 48.0
    injected_flux = np.array(branch.flux, copy=True)
    injected_flux[inside] *= 1.0 - depth_ppm * 1e-6
    injected = SearchBranch(
        key=f"{branch.key}__software_injection",
        label=f"{branch.label} + software injection",
        processing=branch.processing,
        photometry_family=branch.photometry_family,
        method_family=branch.method_family,
        time=branch.time,
        flux=injected_flux,
        segment_id=branch.segment_id,
        can_seed_candidates=True,
        metadata=branch.metadata,
    )
    candidates, _ = search_branch_bls(
        injected,
        holdout_segment=holdout_segment,
        period_min=max(0.5, period_days * 0.45),
        period_max=period_days * 1.8,
        n_periods=n_periods,
        max_candidates=10,
    )
    matches = [
        item
        for item in candidates
        if abs(item["period_days"] - period_days) / period_days <= 0.01
        and _phase_distance(item["epoch_btjd"], epoch, period_days)
        <= max(duration_hours / 12.0, 0.02 * period_days)
    ]
    best = max(matches, key=_candidate_strength) if matches else None
    return {
        "branch": branch.key,
        "scope": "post-correction search-stage injection only",
        "is_scientific_performance_claim": False,
        "injected_period_days": float(period_days),
        "injected_epoch_btjd": epoch,
        "injected_duration_hours": float(duration_hours),
        "injected_depth_ppm": float(depth_ppm),
        "injected_cadences": int(inside.sum()),
        "recovered": best is not None,
        "matched_candidate": best,
        "candidates_examined": len(candidates),
        "status": "recovered" if best is not None else "not_recovered",
    }


def run_cbv_precorrection_injection_check(
    paths: Iterable[str | Path],
    *,
    cbv_dir: str | Path,
    period_days: float = 8.371,
    duration_hours: float = 2.5,
    depth_ppm: float = 1200.0,
    protect_injected_transits: bool = True,
    flatten_window_days: float = 1.0,
) -> dict:
    """Inject into SAP before CBV correction and measure differential recovery.

    The same real light curve is corrected twice, once untouched and once with
    the synthetic transit.  Subtracting the untouched control measurement
    keeps a pre-existing dip or stellar variation at the chosen ephemeris from
    being mistaken for recovered injected depth.
    """
    resolved = [Path(path) for path in paths]
    if not resolved:
        raise ValueError("At least one light-curve file is required")
    first = lk.read(resolved[0], flux_column="sap_flux").remove_nans()
    epoch = float(np.nanmin(first.time.value) + 0.37 * period_days)
    raw_control_pieces = []
    raw_injected_pieces = []
    corrected_control_pieces = []
    corrected_injected_pieces = []
    sector_metrics = []
    for path in resolved:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sap = lk.read(path, flux_column="sap_flux").remove_nans()
        mission = str(sap.meta.get("MISSION", "")).upper()
        kwargs = {}
        if mission == "TESS":
            ensure_tess_cbv(path, cbv_dir)
            kwargs["cbv_dir"] = str(Path(cbv_dir))
        injected = sap.copy()
        time = np.asarray(injected.time.value, dtype=float)
        phase = (time - epoch + 0.5 * period_days) % period_days - 0.5 * period_days
        inside = np.abs(phase) <= duration_hours / 48.0
        injected.flux[inside] *= 1.0 - depth_ppm * 1e-6
        cadence_mask = ~inside if protect_injected_transits else None
        control_corrector = sap.to_corrector("cbv", **kwargs)
        injected_corrector = injected.to_corrector("cbv", **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corrected_control = _deterministic_correct(
                control_corrector,
                cadence_mask=cadence_mask,
                target_under_score=0.0,
                max_iter=30,
            )
            corrected_injected = _deterministic_correct(
                injected_corrector,
                cadence_mask=cadence_mask,
                target_under_score=0.0,
                max_iter=30,
            )
        raw_control = _prepare_lightcurve(sap, flatten_window_days)
        raw_injected = _prepare_lightcurve(injected, flatten_window_days)
        corrected_control_values = _prepare_lightcurve(
            corrected_control, flatten_window_days
        )
        corrected_injected_values = _prepare_lightcurve(
            corrected_injected, flatten_window_days
        )
        segment, segment_kind = _segment_id(sap)
        raw_control_pieces.append((*raw_control, segment))
        raw_injected_pieces.append((*raw_injected, segment))
        corrected_control_pieces.append((*corrected_control_values, segment))
        corrected_injected_pieces.append((*corrected_injected_values, segment))
        sector_metrics.append(
            {
                "file": path.name,
                "segment": segment,
                "segment_kind": segment_kind,
                "injected_cadences": int(inside.sum()),
                "over_fitting_score": float(injected_corrector.over_fitting_score),
                "alpha": float(injected_corrector.alpha),
            }
        )
    arguments = {
        "period_days": period_days,
        "epoch_btjd": epoch,
        "duration_hours": duration_hours,
    }

    def measure_pieces(name: str, pieces) -> dict:
        branch = _assemble_branch(
            name,
            name,
            "paired software validation control",
            "spoc_pipeline_aperture",
            pieces,
        )
        return measure_fixed_ephemeris(branch.time, branch.flux, **arguments)

    measurements = {
        "raw_control": measure_pieces("raw_control", raw_control_pieces),
        "raw_injected": measure_pieces("raw_injected", raw_injected_pieces),
        "corrected_control": measure_pieces(
            "corrected_control", corrected_control_pieces
        ),
        "corrected_injected": measure_pieces(
            "corrected_injected", corrected_injected_pieces
        ),
    }
    raw_added = _depth_difference(
        measurements["raw_injected"], measurements["raw_control"]
    )
    corrected_added = _depth_difference(
        measurements["corrected_injected"], measurements["corrected_control"]
    )
    return {
        "correction": "CBV",
        "scope": "paired-control software injection into real SAP before CBV correction",
        "is_scientific_performance_claim": False,
        "protect_injected_transits": bool(protect_injected_transits),
        "injected_period_days": float(period_days),
        "injected_epoch_btjd": epoch,
        "injected_duration_hours": float(duration_hours),
        "injected_depth_ppm": float(depth_ppm),
        "measurements": measurements,
        "raw_added_depth_ppm": raw_added,
        "corrected_added_depth_ppm": corrected_added,
        "depth_recovery_ratio": (
            float(corrected_added / raw_added)
            if raw_added is not None and corrected_added is not None and raw_added > 0
            else None
        ),
        "nominal_depth_ratio_after_full_processing": (
            float(corrected_added / depth_ppm) if corrected_added is not None else None
        ),
        "segments": sector_metrics,
    }


def run_pld_precorrection_injection_check(
    paths: Iterable[str | Path],
    *,
    period_days: float = 8.371,
    duration_hours: float = 2.5,
    depth_ppm: float = 1200.0,
    protect_injected_transits: bool = True,
    flatten_window_days: float = 1.0,
) -> dict:
    """Inject into TPF aperture pixels before PLD and use a paired control.

    This is a software validation of the correction path, not an astrophysical
    completeness measurement.  The synthetic target dims all pixels in the
    selected target aperture while off-aperture pixels remain unchanged.
    """
    resolved = [Path(path) for path in paths]
    if not resolved:
        raise ValueError("At least one target-pixel file is required")
    first = lk.read(resolved[0])
    epoch = float(np.nanmin(first.time.value) + 0.37 * period_days)
    raw_control_pieces = []
    raw_injected_pieces = []
    corrected_control_pieces = []
    corrected_injected_pieces = []
    segment_metrics = []
    for path in resolved:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tpf = lk.read(path)
        aperture = np.asarray(tpf.pipeline_mask, dtype=bool)
        if not aperture.any():
            aperture = np.asarray(
                tpf.create_threshold_mask(threshold=3, reference_pixel="center"),
                dtype=bool,
            )
        # TargetPixelFile does not implement ``copy()`` in supported
        # Lightkurve releases; re-reading gives an independent in-memory HDU.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            injected = lk.read(path)
        time = np.asarray(tpf.time.value, dtype=float)
        phase = (time - epoch + 0.5 * period_days) % period_days - 0.5 * period_days
        inside = np.abs(phase) <= duration_hours / 48.0
        # ``inside`` indexes the quality-filtered public view, while the FITS
        # table contains every cadence.  Map back to its rows before mutating
        # the independent HDU in memory.
        source_rows = np.flatnonzero(injected.quality_mask)
        factor = 1.0 - depth_ppm * 1e-6
        for cadence_index in np.flatnonzero(inside):
            injected.hdu[1].data["FLUX"][source_rows[cadence_index]][aperture] *= factor
        cadence_mask = ~inside if protect_injected_transits else None
        control_corrector = tpf.to_corrector("pld", aperture_mask=aperture)
        injected_corrector = injected.to_corrector("pld", aperture_mask=aperture)
        correct_kwargs = {
            "pld_order": 1,
            "pca_components": 3,
            "restore_trend": False,
            "cadence_mask": cadence_mask,
            "sigma": 5,
            "niters": 5,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corrected_control = _deterministic_correct(
                control_corrector, **correct_kwargs
            )
            corrected_injected = _deterministic_correct(
                injected_corrector, **correct_kwargs
            )
        raw_control = _prepare_lightcurve(
            tpf.to_lightcurve(aperture_mask=aperture), flatten_window_days
        )
        raw_injected = _prepare_lightcurve(
            injected.to_lightcurve(aperture_mask=aperture), flatten_window_days
        )
        corrected_control_values = _prepare_lightcurve(
            corrected_control, flatten_window_days
        )
        corrected_injected_values = _prepare_lightcurve(
            corrected_injected, flatten_window_days
        )
        segment, segment_kind = _segment_id(tpf)
        raw_control_pieces.append((*raw_control, segment))
        raw_injected_pieces.append((*raw_injected, segment))
        corrected_control_pieces.append((*corrected_control_values, segment))
        corrected_injected_pieces.append((*corrected_injected_values, segment))
        segment_metrics.append(
            {
                "file": path.name,
                "segment": segment,
                "segment_kind": segment_kind,
                "injected_cadences": int(inside.sum()),
                "aperture_pixels": int(aperture.sum()),
            }
        )
    arguments = {
        "period_days": period_days,
        "epoch_btjd": epoch,
        "duration_hours": duration_hours,
    }

    def measure_pieces(name: str, pieces) -> dict:
        branch = _assemble_branch(
            name,
            name,
            "paired software validation control",
            "tpf_custom_aperture",
            pieces,
        )
        return measure_fixed_ephemeris(branch.time, branch.flux, **arguments)

    measurements = {
        "raw_control": measure_pieces("raw_control", raw_control_pieces),
        "raw_injected": measure_pieces("raw_injected", raw_injected_pieces),
        "corrected_control": measure_pieces(
            "corrected_control", corrected_control_pieces
        ),
        "corrected_injected": measure_pieces(
            "corrected_injected", corrected_injected_pieces
        ),
    }
    raw_added = _depth_difference(
        measurements["raw_injected"], measurements["raw_control"]
    )
    corrected_added = _depth_difference(
        measurements["corrected_injected"], measurements["corrected_control"]
    )
    return {
        "correction": "PLD",
        "scope": "paired-control software injection into real target pixels before PLD",
        "is_scientific_performance_claim": False,
        "protect_injected_transits": bool(protect_injected_transits),
        "injected_period_days": float(period_days),
        "injected_epoch_btjd": epoch,
        "injected_duration_hours": float(duration_hours),
        "injected_depth_ppm": float(depth_ppm),
        "measurements": measurements,
        "raw_added_depth_ppm": raw_added,
        "corrected_added_depth_ppm": corrected_added,
        "depth_recovery_ratio": (
            float(corrected_added / raw_added)
            if raw_added is not None and corrected_added is not None and raw_added > 0
            else None
        ),
        "nominal_depth_ratio_after_full_processing": (
            float(corrected_added / depth_ppm) if corrected_added is not None else None
        ),
        "segments": segment_metrics,
    }
