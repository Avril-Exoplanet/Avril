"""Local, loopback-only HTTP bridge between the Avril UI and engine."""

from __future__ import annotations

import hashlib
import hmac
import json
import mimetypes
import os
import re
import selectors
import signal
import subprocess
import sys
import threading
import traceback
import uuid
import webbrowser
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from avril import __version__

TARGET_REGISTRY: dict[str, str] = {
    "L 98-59": "TIC 307210830",
    "L-98-59": "TIC 307210830",
    "L98-59": "TIC 307210830",
    "HD 191939": "TIC 269701147",
    "HD-191939": "TIC 269701147",
    "TOI-1246": "TIC 230127302",
    "TOI 1246": "TIC 230127302",
    "TOI-1749": "TIC 233602827",
    "TOI 1749": "TIC 233602827",
    "TOI-789": "TIC 300710077",
    "TOI 789": "TIC 300710077",
    "TOI-700": "TIC 150428135",
    "TOI 700": "TIC 150428135",
    "TOI-4593": "TIC 399954349",
    "TOI 4593": "TIC 399954349",
    "TOI-4342": "TIC 354944123",
    "TOI 4342": "TIC 354944123",
    "TOI-5624": "TIC 53498154",
    "TOI 5624": "TIC 53498154",
    "TOI-125": "TIC 52368076",
    "TOI 125": "TIC 52368076",
    "TOI-216": "TIC 55652896",
    "TOI 216": "TIC 55652896",
    "TOI-2180": "TIC 298663873",
    "TOI 2180": "TIC 298663873",
}

DEFAULT_ALLOWED_ORIGINS = frozenset(
    {
        "http://127.0.0.1:4173",
        "http://localhost:4173",
        "https://vlora-orbit-lab.pages.dev",
        "https://shutongliang.github.io",
    }
)

DEFAULT_WEB_ROOT = (
    Path(__file__).resolve().parents[1] / "apps" / "vlora-observatory" / "dist-pages"
)

STAGE_COPY = {
    "queued": ("Request queued", "Waiting for the Avril science worker"),
    "starting": ("Science worker started", "Creating the run workspace and checking caches"),
    "source": ("Resolving the target", "Checking the official observation source for this target"),
    "discovering": ("Building the observation inventory", "Querying the archive for every eligible observation"),
    "downloading": ("High-concurrency light-curve transfer", "Downloading or validating all selected light curves before search"),
    "ffi_capacity": ("Checking FFI storage", "Verifying free space for full-frame cutouts"),
    "ffi_discovery": ("Discovering FFI sectors", "Querying TESSCut coverage for the target"),
    "ffi_downloading": ("Downloading TESSCut pixels", "Receiving and validating segmented FFI archives"),
    "ffi_extracting": ("Extracting TESSCut pixels", "Opening validated FFI archives"),
    "ffi_photometry": ("Building FFI photometry", "Measuring label-free aperture light curves"),
    "preparing": ("Locking independent time folds", "Partitioning observations before any signal search"),
    "pixel_prefetch": ("Two-lane pipeline active", "Searching light curves while target-pixel products download in the background"),
    "download": ("Loading TESS sectors", "Downloading or validating the requested light curves"),
    "searching": ("Searching the full time series", "Running the uncapped transit-signal search across independent folds"),
    "voting": ("Scoring search peaks", "Testing every local peak across deterministic sector resamplings"),
    "multi_seed_voting": ("Consensus voting", "Measuring signal stability across independent vote seeds"),
    "matching": ("Matching fold hypotheses", "Linking compatible signals found in independent time folds"),
    "validating": ("Validating locked ephemerides", "Retesting each fixed period across held-out observations"),
    "refitting": ("Refining period and epoch", "Running exact global refits for the evidence-backed rescue budget"),
    "revoting": ("Re-voting refined ephemerides", "Re-scoring changed ephemerides across every deterministic vote seed"),
    "discovery_complete": ("Discovery pass complete", "Search, consensus voting and exact refits are complete"),
    "event_repeatability": ("Measuring individual events", "Checking every predicted event in every stable family"),
    "deduplicating": ("Collapsing duplicate event trains", "Keeping one audit row for directly equivalent event trains"),
    "harmonics": ("Resolving integer aliases", "Compressing aliases proven by event residues"),
    "harmonic_families": ("Arbitrating harmonic families", "Selecting one primary signal per resolved harmonic family"),
    "physical_reference_gate": ("Building the science-review funnel", "Separating meaningful signals from gap-dominated and incoherent rows"),
    "exact_phase_null_gate": ("Running exact phase-null tests", "Calibrating eligible ephemerides against phase-shifted null trials"),
    "significance": ("Measuring detection significance", "Calibrating locked ephemerides against red noise and phase-shift nulls"),
    "ranking": ("Ranking the candidate ledger", "Scoring every retained signal without deleting candidates"),
    "vetting": ("Running physical checks", "Testing binary-star and false-positive diagnostics"),
    "pixel_download": ("Preparing pixel diagnostics", "Loading target-pixel evidence for candidate localization"),
    "pixel_vetting": ("Localizing candidate signals", "Checking whether each signal is centered on the target"),
    "catalog": ("Comparing with known catalogs", "Matching the locked ranking against known objects"),
    "rendering": ("Writing the scientific report", "Serializing plots, evidence and the complete candidate ledger"),
    "engine": ("Avril engine pass", "Running the selected scientific search engine"),
    "complete": ("Compute complete", "The complete candidate ledger and evidence report are ready"),
    "cancelling": ("Stopping compute", "Terminating the active VLora science process"),
    "cancelled": ("Run stopped", "The queued or active computation was stopped by the user"),
    "failed": ("Compute stopped", "The engine returned an error; the run record was preserved"),
}

STAGE_PERCENT = {
    "launch": 1,
    "target_start": 4,
    "loading": 4,
    "download": 8,
    "searching": 18,
    "voting": 32,
    "multi_seed_voting": 38,
    "matching": 44,
    "validating": 50,
    "refitting": 55,
    "revoting": 65,
    "discovery_complete": 70,
    "event_repeatability": 72,
    "deduplicating": 75,
    "harmonics": 78,
    "harmonic_families": 80,
    "physical_reference_gate": 82,
    "vetting": 86,
    "pixel_vetting": 95,
    "exact_phase_null_gate": 96,
    "significance": 97,
    "complete": 99,
    "target_complete": 99,
}

# The engine intentionally emits a bounded scientific progress record on
# stdout. Keep these fields structured in job events so native and browser
# clients can render the real funnel instead of scraping English messages.
PROGRESS_METRIC_FIELDS = frozenset(
    {
        "files_complete", "files_total", "failures", "files_per_second",
        "block", "raw_hypotheses", "vote_groups_per_seed", "vote_seed_count",
        "seed_index", "seed_count", "families_remeasured", "ranking_population",
        "promoted_families", "initial_stable_families", "refit_eligible_families",
        "estimated_exact_period_trials", "families_completed", "families_total",
        "families_rescued", "stable_families", "families", "direction_passed",
        "deduplicated_families", "deduplicated_candidates", "all_candidates",
        "eligible", "short_circuited_but_retained", "detailed_event_measurements_reused",
        "meaningful_candidate_funnel", "eligible_before_direct_alias_deferral",
        "first_pass", "deferred_direct_aliases", "phase_trials_each", "candidate",
        "total", "candidates", "target_pixel_segments", "workers", "backend",
        "base_blocks", "blocks", "timing_seconds", "audit_only_families",
    }
)


class JobCancelled(RuntimeError):
    """Internal control-flow marker for a user-cancelled web job."""


def _avril_data_root() -> Path:
    from avril.paths import default_data_root

    return default_data_root()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _stage_copy(stage: str) -> tuple[str, str]:
    normalized = str(stage).split(":")[-1]
    return STAGE_COPY.get(normalized, ("Avril engine activity", "The science engine reported a new execution step"))


def _stage_percent(stage: str, supplied: object) -> int:
    try:
        value = int(supplied or 0)
    except (TypeError, ValueError):
        value = 0
    normalized = str(stage).split(":")[-1]
    return value if value > 0 else STAGE_PERCENT.get(normalized, 1)


def _progress_stage(payload: dict) -> str:
    """Namespace a child engine's completion away from whole-job completion."""
    stage = str(payload.get("stage") or "engine")
    if stage == "complete":
        if payload.get("pipeline_stage"):
            return "discovery_complete"

        # Compatibility for jobs emitted by threshold runners frozen before
        # `pipeline_stage` was propagated. Keep this intentionally narrow: the
        # message and all three discovery metrics uniquely identify the
        # ensemble-intersection milestone, not final job completion.
        metrics = payload.get("metrics")
        discovery_keys = {
            "raw_hypotheses",
            "promoted_families",
            "stable_families",
        }
        if (
            payload.get("message")
            == "Multi-group vote-intersection experiment complete"
            and isinstance(metrics, dict)
            and discovery_keys.issubset(metrics)
        ):
            return "discovery_complete"
    return stage


def _progress_detail(payload: dict) -> dict | None:
    """Merge nested and top-level bounded progress metrics for UI clients."""
    nested = payload.get("detail")
    if not isinstance(nested, dict):
        nested = payload.get("details")
    detail = dict(nested) if isinstance(nested, dict) else {}

    metrics = detail.get("metrics")
    merged_metrics = dict(metrics) if isinstance(metrics, dict) else {}
    payload_metrics = payload.get("metrics")
    if isinstance(payload_metrics, dict):
        merged_metrics.update(payload_metrics)
    for key in PROGRESS_METRIC_FIELDS:
        if key in payload:
            merged_metrics[key] = payload[key]
    if merged_metrics:
        detail["metrics"] = merged_metrics

    for key in ("title", "command", "level"):
        if key in payload and key not in detail:
            detail[key] = payload[key]
    return detail or None


def _english_ui_text(value: object, fallback: str) -> str:
    text = str(value or "").strip()
    if not text or re.search(r"[\u3400-\u9fff]", text):
        return fallback
    return text


def _compact_public_candidate(source: dict, *, include_plots: bool) -> dict:
    """Remove evidence arrays that belong in the downloadable audit artifact."""
    candidate = deepcopy(source)
    if not include_plots:
        candidate.pop("fold", None)
    significance = candidate.get("detection_significance")
    if isinstance(significance, dict):
        for key in ("duration_noise_models", "events", "block_metrics", "phase_shift_null"):
            significance.pop(key, None)
    verification = candidate.get("holdout_verification")
    if isinstance(verification, dict):
        repeatability = verification.get("event_repeatability")
        if isinstance(repeatability, dict):
            for key in ("event_support", "unsupported_event_centers_btjd", "modulo", "parity"):
                repeatability.pop(key, None)
    pixel = candidate.get("pixel_vetting")
    if isinstance(pixel, dict):
        for sector in pixel.get("per_sector") or []:
            if not isinstance(sector, dict):
                continue
            for key in ("median_image", "difference_image", "pipeline_mask"):
                sector.pop(key, None)
            for event in sector.get("per_event") or []:
                if not isinstance(event, dict):
                    continue
                for key in ("median_image", "difference_image", "pipeline_mask"):
                    event.pop(key, None)
    return candidate


def write_public_result(full_path: Path, public_path: Path) -> None:
    """Write the bounded browser payload; the gzip audit keeps full evidence."""
    result = json.loads(full_path.read_text(encoding="utf-8"))
    final = (
        result.get("ranked_candidates")
        or result.get("final_candidates")
        or result.get("candidates")
        or []
    )
    audit = (
        result.get("ranked_candidates")
        or result.get("candidate_audit")
        or result.get("candidates")
        or []
    )
    result["candidates"] = [
        _compact_public_candidate(row, include_plots=True) for row in final
    ]
    result.pop("ranked_candidates", None)
    result.pop("final_candidates", None)
    result["candidate_audit"] = [
        _compact_public_candidate(row, include_plots=False) for row in audit
    ]
    result["review_candidate_count"] = len(result.get("review_candidates") or [])
    result["rejected_candidate_count"] = len(result.get("rejected_candidates") or [])
    for key in (
        "review_candidates",
        "rejected_candidates",
        "candidate_registry",
        "raw_candidate_ledger",
        "cross_block_match_edges",
        "search_diagnostics",
        "load_diagnostics",
        "archive_inventory",
        "target_pixel_inventory",
        "selected_products",
        "selected_target_pixel_products",
        "target_pixel_files",
    ):
        result.pop(key, None)
    temporary = public_path.with_suffix(public_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(public_path)


def resolve_target_number(value: str) -> str:
    """Resolve an official target name or accept a direct TIC identifier."""
    normalized = " ".join(str(value).strip().upper().split())
    if normalized in TARGET_REGISTRY:
        return TARGET_REGISTRY[normalized]
    compact = normalized.replace(" ", "")
    if compact.startswith("TIC") and compact[3:].isdigit():
        return f"TIC {int(compact[3:])}"
    if normalized.startswith("TOI"):
        raise KeyError(f"{normalized} is not registered in this VLora build")
    raise ValueError("target must be a registered TOI name or TIC identifier")


class JobStore:
    """Small persistent single-worker queue so one Mac is not oversubscribed."""

    def __init__(self, root: Path | None = None):
        self.root = (root or _avril_data_root() / "web-jobs").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, dict] = {}
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlora-web")
        self.futures: dict[str, Future] = {}
        self.cancel_events: dict[str, threading.Event] = {}
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        """Restore completed jobs and preserve interrupted runs as readable records."""
        for path in sorted(self.root.glob("*/job.json")):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                job_id = str(job["id"])
                if not isinstance(job.get("events"), list):
                    job["events"] = []
                if job.get("status") in {"queued", "running", "cancelling"}:
                    completed_at = _now()
                    job.update(
                        status="failed",
                        stage="failed",
                        message="The previous compute service stopped before this run completed",
                        error="backend_restarted_during_job",
                        updated_at=completed_at,
                        completed_at=completed_at,
                    )
                    self._append_event_locked(
                        job,
                        stage="failed",
                        level="error",
                        title="Compute service restarted",
                        message="Submit the target again to start a fresh run",
                    )
                    self._persist(job)
                self.jobs[job_id] = job
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue

    @staticmethod
    def _control_digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _persist(self, job: dict) -> None:
        directory = self.root / job["id"]
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "job.json.tmp"
        temporary.write_text(
            json.dumps(job, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(directory / "job.json")

    def _update(self, job_id: str, **changes) -> None:
        with self.lock:
            job = self.jobs[job_id]
            job.update(changes, updated_at=_now())
            self._persist(job)

    def _append_event_locked(
        self,
        job: dict,
        *,
        stage: str,
        level: str,
        title: str,
        message: str,
        command: str | None = None,
        metrics: dict | None = None,
    ) -> None:
        events = job.setdefault("events", [])
        sequence = int(events[-1].get("seq", 0)) + 1 if events else 1
        events.append(
            {
                "seq": sequence,
                "at": _now(),
                "stage": stage,
                "level": level,
                "title": title,
                "message": message,
                "command": command,
                "metrics": metrics or {},
            }
        )
        if len(events) > 360:
            del events[:-360]

    def _record_progress(
        self,
        job_id: str,
        *,
        status: str,
        stage: str,
        percent: int,
        message: str | None = None,
        detail: dict | None = None,
        level: str = "info",
    ) -> None:
        event_detail = dict(detail or {})
        command = event_detail.pop("command", None)
        custom_title = event_detail.pop("title", None)
        level = str(event_detail.pop("level", level))
        nested_metrics = event_detail.pop("metrics", None)
        metrics = dict(nested_metrics) if isinstance(nested_metrics, dict) else {}
        metrics.update(event_detail)
        default_title, default_message = _stage_copy(stage)
        title = _english_ui_text(custom_title, default_title)
        message = _english_ui_text(message, default_message)
        with self.lock:
            job = self.jobs[job_id]
            current_percent = int(job.get("percent", 0))
            incoming_percent = max(0, min(99, int(percent)))
            changes = {
                "status": status,
                "percent": max(current_percent, incoming_percent),
                "updated_at": _now(),
            }
            # Pixel prefetch runs beside the search.  Keep its events in the
            # terminal, but do not let a late low-percent callback move the
            # main progress label backwards from SEARCH to DOWNLOAD/PIXELS.
            if incoming_percent >= current_percent or status == "failed":
                changes.update(stage=stage, message=message)
            job.update(**changes)
            self._append_event_locked(
                job,
                stage=stage,
                level=level,
                title=title,
                message=message,
                command=str(command) if command else None,
                metrics=metrics,
            )
            self._persist(job)

    def create(self, request: dict) -> dict:
        display_number = str(request.get("target_number") or request.get("target") or "")
        target = resolve_target_number(display_number)
        backend = str(request.get("backend") or "auto")
        if backend not in {"auto", "cpu", "cuda"}:
            raise ValueError("backend must be auto, cpu, or cuda")
        source = str(request.get("source") or "auto")
        if source not in {"auto", "spoc", "ffi"}:
            raise ValueError("source must be auto, spoc, or ffi")
        pixels = str(request.get("pixels") or "none")
        if pixels not in {"none", "followup"}:
            raise ValueError("pixels must be none or followup")
        ffi_engine = str(request.get("ffi_engine") or "core")
        if ffi_engine not in {"core", "pro"}:
            raise ValueError("ffi_engine must be core or pro")
        legacy_engine = str(request.get("engine") or f"avril-v{__version__}")
        with self.lock:
            duplicate = next(
                (
                    row
                    for row in self.jobs.values()
                    if row.get("target") == target
                    and row.get("backend", "auto") == backend
                    and row.get("source", "auto") == source
                    and row.get("pixels", "none") == pixels
                    and row.get("ffi_engine", "core") == ffi_engine
                    and row.get("status") in {"queued", "running"}
                ),
                None,
            )
            if duplicate is not None:
                response = self.get(str(duplicate["id"]))
                response["deduplicated_request"] = True
                return response
        job_id = uuid.uuid4().hex[:16]
        control_token = uuid.uuid4().hex
        job = {
            "id": job_id,
            "display_number": display_number.strip().upper(),
            "target": target,
            "engine": legacy_engine,
            "backend": backend,
            "source": source,
            "pixels": pixels,
            "ffi_engine": ffi_engine,
            "status": "queued",
            "stage": "queued",
            "percent": 0,
            "message": "Waiting for the Avril science worker",
            "created_at": _now(),
            "updated_at": _now(),
            "result": None,
            "error": None,
            "events": [],
            "control_token_sha256": self._control_digest(control_token),
        }
        with self.lock:
            self._append_event_locked(
                job,
                stage="queued",
                level="info",
                title="Request queued",
                message=f"Preparing {target} with Avril {__version__} ({backend})",
                command=(
                    f"avril run {target!r} --source {source} --pixels {pixels} "
                    f"--ffi-engine {ffi_engine} --backend {backend}"
                ),
                metrics={
                    "mission": "TESS",
                    "target": target,
                    "backend": backend,
                    "source": source,
                    "pixels": pixels,
                    "ffi_engine": ffi_engine,
                },
            )
            self.jobs[job_id] = job
            self.cancel_events[job_id] = threading.Event()
            self._persist(job)
            self.futures[job_id] = self.pool.submit(self._run, job_id)
        response = self.get(job_id)
        response["control_token"] = control_token
        return response

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                process.terminate()
            except ProcessLookupError:
                pass

    def _finish_cancelled(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            if job.get("status") == "cancelled":
                return
            completed_at = _now()
            job.update(
                status="cancelled",
                stage="cancelled",
                message="The queued or active computation was stopped by the user",
                error=None,
                updated_at=completed_at,
                completed_at=completed_at,
            )
            title, message = _stage_copy("cancelled")
            self._append_event_locked(
                job,
                stage="cancelled",
                level="warning",
                title=title,
                message=message,
                command=f"job cancel --id {job_id}",
            )
            self._persist(job)

    def cancel(self, job_id: str, control_token: str) -> dict:
        process: subprocess.Popen[str] | None = None
        finish_immediately = False
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            job = self.jobs[job_id]
            expected = str(job.get("control_token_sha256") or "")
            supplied = self._control_digest(str(control_token or ""))
            if not expected or not hmac.compare_digest(expected, supplied):
                raise PermissionError("invalid_job_control_token")
            if job.get("status") in {"complete", "failed", "cancelled"}:
                response = self.get(job_id)
                response["already_terminal"] = True
                return response

            event = self.cancel_events.setdefault(job_id, threading.Event())
            event.set()
            future = self.futures.get(job_id)
            process = self.processes.get(job_id)
            finish_immediately = bool(future.cancel()) if future is not None else process is None
            if not finish_immediately:
                job.update(
                    status="cancelling",
                    stage="cancelling",
                    message="Stop requested; terminating the active science process",
                    updated_at=_now(),
                )
                title, message = _stage_copy("cancelling")
                self._append_event_locked(
                    job,
                    stage="cancelling",
                    level="warning",
                    title=title,
                    message=message,
                    command=f"job stop --id {job_id}",
                )
                self._persist(job)

        if process is not None:
            self._terminate_process(process)
        if finish_immediately:
            self._finish_cancelled(job_id)
            with self.lock:
                self.futures.pop(job_id, None)
                self.cancel_events.pop(job_id, None)
        return self.get(job_id)

    @staticmethod
    def _process_command(job: dict, run_directory: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "avril",
            "run",
            str(job["target"]),
            "--backend",
            str(job.get("backend") or "auto"),
            "--source",
            str(job.get("source") or "auto"),
            "--pixels",
            str(job.get("pixels") or "none"),
            "--ffi-engine",
            str(job.get("ffi_engine") or "core"),
            "--output-root",
            str(run_directory),
        ]

    def _run(self, job_id: str) -> None:
        with self.lock:
            cancel_event = self.cancel_events.setdefault(job_id, threading.Event())
            started_at = _now()
            self.jobs[job_id]["started_at"] = started_at
            self.jobs[job_id]["updated_at"] = started_at
            self._persist(self.jobs[job_id])
            job = dict(self.jobs[job_id])
        run_directory = self.root / job_id / "run"
        process: subprocess.Popen[str] | None = None
        selector: selectors.BaseSelector | None = None
        stderr_tail: list[str] = []
        try:
            if cancel_event.is_set():
                raise JobCancelled()
            self._record_progress(
                job_id,
                status="running",
                stage="starting",
                percent=1,
                detail={
                    "command": f"workspace init --job {job_id}",
                    "metrics": {"storage": "Passport", "cache": "enabled"},
                },
            )
            command = self._process_command(job, run_directory)
            process = subprocess.Popen(
                command,
                cwd=str(Path(__file__).resolve().parents[1]),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            with self.lock:
                self.processes[job_id] = process
            if cancel_event.is_set():
                self._terminate_process(process)

            selector = selectors.DefaultSelector()
            assert process.stdout is not None
            assert process.stderr is not None
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                if cancel_event.is_set():
                    self._terminate_process(process)
                for key, _ in selector.select(timeout=0.25):
                    line = key.fileobj.readline()
                    if not line:
                        selector.unregister(key.fileobj)
                        continue
                    if key.data == "stderr":
                        stderr_tail.append(line.rstrip())
                        del stderr_tail[:-40]
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "stage" not in payload:
                        continue
                    if cancel_event.is_set():
                        continue
                    progress_stage = _progress_stage(payload)
                    self._record_progress(
                        job_id,
                        status="running",
                        stage=progress_stage,
                        percent=_stage_percent(
                            progress_stage,
                            payload.get("percent"),
                        ),
                        message=str(payload.get("message") or ""),
                        detail=_progress_detail(payload),
                    )

            return_code = process.wait()
            if cancel_event.is_set():
                raise JobCancelled()
            if return_code != 0:
                detail = stderr_tail[-1] if stderr_tail else f"exit status {return_code}"
                raise RuntimeError(f"Avril science process failed: {detail}")

            threshold_summary = run_directory / "threshold_round_summary.json"
            if not threshold_summary.is_file():
                raise RuntimeError("Avril run did not write a threshold-round summary")
            round_summary = json.loads(
                threshold_summary.read_text(encoding="utf-8")
            )
            completed_targets = list(round_summary.get("complete") or [])
            if len(completed_targets) != 1:
                deferred = list(round_summary.get("deferred") or [])
                failures = list(round_summary.get("failures") or [])
                raise RuntimeError(
                    f"expected one completed target; deferred={len(deferred)}, failures={len(failures)}"
                )
            engine_result_path = Path(completed_targets[0]["result"])
            followup = engine_result_path.parent.parent / "final-science-pixel-followup" / "result.json"
            if followup.is_file():
                engine_result_path = followup
            run_receipt_path = run_directory / "avril_run_receipt.json"
            run_receipt = (
                json.loads(run_receipt_path.read_text(encoding="utf-8"))
                if run_receipt_path.is_file()
                else {}
            )
            source_kind = (
                "ffi-selfmade"
                if run_receipt.get("source") == "ffi"
                else "official-spoc-manifest"
            )
            audit_path = engine_result_path.with_name("science_candidates.json.gz")
            result = json.loads(engine_result_path.read_text(encoding="utf-8"))
            public_result_path = engine_result_path.with_name("result.public.json")
            write_public_result(engine_result_path, public_result_path)
            candidates = list(
                result.get("ranked_candidates")
                or result.get("final_candidates")
                or result.get("candidates")
                or []
            )
            manual_review_required = sum(
                1
                for candidate in candidates
                if (candidate.get("science_review") or {}).get("priority")
                == "manual_review_required"
            )
            low_priority_or_false_alarm_risk = sum(
                1
                for candidate in candidates
                if (candidate.get("science_review") or {}).get("priority")
                == "low_priority_or_false_alarm_risk"
            )
            passes_7p1_reference = sum(
                1
                for candidate in candidates
                if (candidate.get("detection_significance") or {}).get(
                    "passes_7p1_reference"
                )
                is True
            )
            summary = {
                "target": job["target"],
                "source_kind": source_kind,
                "candidate_count": len(candidates),
                "backend": job.get("backend") or "auto",
                "result_path": str(engine_result_path.resolve()),
                "public_result_path": str(public_result_path.resolve()),
                "audit_path": str(audit_path.resolve()),
                "result_url": f"/api/jobs/{job_id}/result",
                "audit_url": f"/api/jobs/{job_id}/audit",
                "runtime_seconds": (result.get("timing_seconds") or {}).get(
                    "full_pipeline_compute_total"
                ),
            }
            with self.lock:
                if cancel_event.is_set():
                    raise JobCancelled()
                job = self.jobs[job_id]
                completed_at = _now()
                job.update(
                    status="complete",
                    stage="complete",
                    percent=100,
                    message="The complete candidate ledger and evidence report are ready",
                    result=summary,
                    updated_at=completed_at,
                    completed_at=completed_at,
                )
                title, message = _stage_copy("complete")
                self._append_event_locked(
                    job,
                    stage="complete",
                    level="success",
                    title=title,
                    message=message,
                    command=f"report seal --job {job_id}",
                    metrics={
                        "candidates": summary.get("candidate_count"),
                        "backend": summary["backend"],
                        "manual_review_required": manual_review_required,
                        "low_priority_or_false_alarm_risk": low_priority_or_false_alarm_risk,
                        "passes_7p1_reference": passes_7p1_reference,
                    },
                )
                self._persist(job)
        except JobCancelled:
            self._finish_cancelled(job_id)
        except Exception as exc:  # pragma: no cover - depends on remote data/runtime
            if cancel_event.is_set():
                self._finish_cancelled(job_id)
            else:
                with self.lock:
                    job = self.jobs[job_id]
                    completed_at = _now()
                    job.update(
                        status="failed",
                        stage="failed",
                        message=f"{type(exc).__name__}: {exc}",
                        error=str(exc),
                        traceback=traceback.format_exc(),
                        updated_at=completed_at,
                        completed_at=completed_at,
                    )
                    self._append_event_locked(
                        job,
                        stage="failed",
                        level="error",
                        title="Compute stopped",
                        message=f"{type(exc).__name__}: {exc}",
                        command="pipeline halted; inspect the preserved job record",
                    )
                    self._persist(job)
        finally:
            if selector is not None:
                selector.close()
            if process is not None and process.poll() is None:
                self._terminate_process(process)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            with self.lock:
                self.processes.pop(job_id, None)
                self.futures.pop(job_id, None)
                self.cancel_events.pop(job_id, None)

    def get(self, job_id: str) -> dict:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            response = {
                key: value
                for key, value in self.jobs[job_id].items()
                if key not in {"traceback", "control_token_sha256"}
            }
            now = datetime.now(UTC)
            created_value = response.get("created_at") or response.get("updated_at")
            created = (
                datetime.fromisoformat(str(created_value)) if created_value else now
            )
            started_value = response.get("started_at")
            completed_value = response.get("completed_at")
            started = (
                datetime.fromisoformat(str(started_value)) if started_value else None
            )
            completed = (
                datetime.fromisoformat(str(completed_value)) if completed_value else None
            )
            response["queue_seconds"] = round(
                max(0.0, ((started or now) - created).total_seconds()), 3
            )
            response["compute_seconds"] = round(
                max(0.0, ((completed or now) - started).total_seconds()), 3
            ) if started else 0.0
            response["total_seconds"] = round(
                max(0.0, ((completed or now) - created).total_seconds()), 3
            )
            queued = sorted(
                (
                    row
                    for row in self.jobs.values()
                    if row.get("status") == "queued"
                ),
                key=lambda row: str(row.get("created_at") or ""),
            )
            response["queue_position"] = next(
                (
                    index
                    for index, row in enumerate(queued, start=1)
                    if row.get("id") == job_id
                ),
                0,
            )
            return response

    def list_records(self) -> dict:
        """Return bounded persistent run summaries for native history views."""
        with self.lock:
            ordered_ids = sorted(
                self.jobs,
                key=lambda identifier: str(
                    self.jobs[identifier].get("created_at")
                    or self.jobs[identifier].get("updated_at")
                    or ""
                ),
                reverse=True,
            )
        items = []
        for job_id in ordered_ids:
            job = self.get(job_id)
            items.append(
                {
                    key: job.get(key)
                    for key in (
                        "id", "target", "display_number", "status", "stage",
                        "percent", "message", "engine", "backend", "source",
                        "pixels", "ffi_engine", "created_at", "updated_at",
                        "started_at", "completed_at", "queue_seconds",
                        "compute_seconds", "total_seconds", "error", "result",
                    )
                }
            )
        return {"items": items, "total": len(items)}

    def result_path(self, job_id: str) -> Path:
        job = self.get(job_id)
        summary = job.get("result") or {}
        if summary.get("public_result_path"):
            return Path(summary["public_result_path"])
        raise FileNotFoundError("job has no Avril result receipt")

    def read_result(self, job_id: str) -> dict:
        job = self.get(job_id)
        if job.get("status") != "complete":
            raise RuntimeError("job_not_complete")
        return json.loads(self.result_path(job_id).read_text(encoding="utf-8"))

    def audit_path(self, job_id: str) -> Path:
        job = self.get(job_id)
        summary = job.get("result") or {}
        if summary.get("result_path"):
            candidate = Path(summary["result_path"]).with_name("science_candidates.json.gz")
            if candidate.is_file():
                return candidate
        if summary.get("audit_path"):
            candidate = Path(summary["audit_path"])
            if candidate.is_file():
                return candidate
        raise FileNotFoundError("job has no Avril audit receipt")

    def shutdown(self) -> None:
        """Stop queued work and terminate every child science process."""
        with self.lock:
            for event in self.cancel_events.values():
                event.set()
            futures = list(self.futures.values())
            processes = list(self.processes.values())
        for future in futures:
            future.cancel()
        for process in processes:
            self._terminate_process(process)
        self.pool.shutdown(wait=True, cancel_futures=True)


class VLoraRequestHandler(BaseHTTPRequestHandler):
    server: VLoraHTTPServer

    def log_message(self, format: str, *args) -> None:
        print(f"[avril-web] {self.address_string()} {format % args}")

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or origin in self.server.allowed_origins

    def _relay_allowed(self) -> bool:
        expected = os.environ.get("VLORA_RELAY_TOKEN", "")
        if not expected or not self.headers.get("CF-Ray"):
            return True
        supplied = self.headers.get("X-VLora-Relay-Key", "")
        return hmac.compare_digest(supplied, expected)

    def _authorized(self) -> bool:
        if not self._origin_allowed():
            self._json(HTTPStatus.FORBIDDEN, {"error": "origin_not_allowed"})
            return False
        if not self._relay_allowed():
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "relay_not_authorized"})
            return False
        return True

    def _cors_headers(self) -> None:
        origin = self.headers.get("Origin")
        if origin and origin in self.server.allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _file(self, path: Path, *, content_type: str, download_name: str) -> None:
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self._cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                self.wfile.write(chunk)

    def _static(self, path: Path) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                self.wfile.write(chunk)

    def _request_json(self, *, maximum: int = 64 * 1024) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > maximum:
            raise ValueError("invalid request size")
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        return request

    def do_OPTIONS(self) -> None:
        if not self._authorized():
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if not self._authorized():
            return
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json(
                HTTPStatus.OK,
                {"status": "ready", "engine": "avril", "version": __version__},
            )
            return
        if path.startswith("/api/jobs/") and path.endswith("/result"):
            job_id = path.split("/")[-2]
            try:
                job = self.server.jobs.get(job_id)
                if job.get("status") != "complete":
                    raise RuntimeError("job_not_complete")
                self._file(
                    self.server.jobs.result_path(job_id),
                    content_type="application/json; charset=utf-8",
                    download_name=f"avril-{job_id}-complete-result.json",
                )
            except KeyError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "job_not_found"})
            except RuntimeError:
                self._json(HTTPStatus.CONFLICT, {"error": "job_not_complete"})
            except OSError as exc:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        if path.startswith("/api/jobs/") and path.endswith("/audit"):
            job_id = path.split("/")[-2]
            try:
                job = self.server.jobs.get(job_id)
                if job.get("status") != "complete":
                    raise RuntimeError("job_not_complete")
                self._file(
                    self.server.jobs.audit_path(job_id),
                    content_type="application/gzip",
                    download_name=f"avril-{job_id}-audit.json.gz",
                )
            except KeyError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "job_not_found"})
            except RuntimeError:
                self._json(HTTPStatus.CONFLICT, {"error": "job_not_complete"})
            except (OSError, ValueError) as exc:
                self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            try:
                self._json(HTTPStatus.OK, self.server.jobs.get(job_id))
            except KeyError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "job_not_found"})
            return
        relative = "index.html" if path in {"", "/"} else path.lstrip("/")
        candidate = (self.server.web_root / relative).resolve()
        if candidate.is_relative_to(self.server.web_root) and candidate.is_file():
            self._static(candidate)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        if not self._authorized():
            return
        path = urlparse(self.path).path
        try:
            if path == "/api/jobs":
                self._json(
                    HTTPStatus.ACCEPTED,
                    self.server.jobs.create(self._request_json()),
                )
                return
            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = path.split("/")[-2]
                request = self._request_json(maximum=4 * 1024)
                self._json(
                    HTTPStatus.OK,
                    self.server.jobs.cancel(job_id, str(request.get("control_token") or "")),
                )
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except PermissionError as exc:
            self._json(HTTPStatus.FORBIDDEN, {"error": str(exc)})
        except KeyError as exc:
            if path == "/api/jobs":
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "job_not_found"})
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


class VLoraHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        *,
        jobs: JobStore,
        allowed_origins: set[str] | frozenset[str],
        web_root: Path = DEFAULT_WEB_ROOT,
    ):
        super().__init__(address, VLoraRequestHandler)
        self.jobs = jobs
        self.allowed_origins = frozenset(allowed_origins)
        self.web_root = web_root.resolve()


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    allowed_origins: set[str] | frozenset[str] = DEFAULT_ALLOWED_ORIGINS,
    open_browser: bool = False,
    web_root: Path | None = None,
) -> None:
    """Serve the local Avril bridge until interrupted."""
    server = VLoraHTTPServer(
        (host, port),
        jobs=JobStore(),
        allowed_origins=allowed_origins,
        web_root=web_root or DEFAULT_WEB_ROOT,
    )
    url = f"http://{host}:{port}/"
    print(f"Avril workbench ready at {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    previous_sigterm = None
    if threading.current_thread() is threading.main_thread():
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def stop_on_sigterm(_signum, _frame) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop_on_sigterm)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        server.jobs.shutdown()
        server.server_close()
