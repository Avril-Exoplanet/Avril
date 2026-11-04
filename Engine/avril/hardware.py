"""Hardware discovery and conservative, non-oversubscribed execution profiles."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExecutionProfile:
    """Arguments shared by the CPU and CUDA production schedulers."""

    backend: str
    logical_cpus: int
    search_workers: int
    bls_threads: int
    event_threads: int
    event_processes: int
    fixed_ephemeris_threads: int
    search_fixed_ephemeris_threads: int
    refit_workers: int
    event_workers: int
    significance_threads: int
    significance_processes: int
    vetting_threads: int
    vetting_processes: int

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)

    def scheduler_arguments(self) -> list[str]:
        pairs = {
            "stage1-engine": self.backend,
            "search-workers": self.search_workers,
            "bls-threads": self.bls_threads,
            "event-threads": self.event_threads,
            "event-processes": self.event_processes,
            "fixed-ephemeris-threads": self.fixed_ephemeris_threads,
            "search-fixed-ephemeris-threads": self.search_fixed_ephemeris_threads,
            "refit-workers": self.refit_workers,
            "event-workers": self.event_workers,
            "significance-threads": self.significance_threads,
            "significance-processes": self.significance_processes,
            "vetting-threads": self.vetting_threads,
            "vetting-processes": self.vetting_processes,
        }
        return [item for key, value in pairs.items() for item in (f"--{key}", str(value))]


def _command_works(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    detail = (result.stdout or result.stderr).strip().splitlines()
    return result.returncode == 0, detail[0] if detail else f"exit {result.returncode}"


def cuda_status() -> dict:
    """Check the exact CUDA prerequisites without silently selecting a GPU."""

    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    nvcc = shutil.which("nvcc")
    add("nvcc", nvcc is not None, nvcc or "not found")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        ok, detail = _command_works(
            [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        )
        add("nvidia-smi", ok, detail)
    else:
        add("nvidia-smi", False, "not found")
    for module_name in ("cupy", "pycuda"):
        available = importlib.util.find_spec(module_name) is not None
        add(f"python:{module_name}", available, "installed" if available else "not installed")
    return {
        "ready": all(row["ok"] for row in checks),
        "checks": checks,
    }


def select_backend(requested: str) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported backend: {requested}")
    if requested == "cpu":
        return "cpu"
    status = cuda_status()
    if requested == "cuda" and not status["ready"]:
        failures = ", ".join(row["name"] for row in status["checks"] if not row["ok"])
        raise RuntimeError(f"CUDA was requested but is not ready: {failures}")
    return "cuda" if status["ready"] else "cpu"


def execution_profile(backend: str, cpu_limit: int | None = None) -> ExecutionProfile:
    """Return defaults validated on Apple Silicon and a 12-vCPU CUDA host."""

    detected = max(1, os.cpu_count() or 1)
    cpus = max(1, min(detected, int(cpu_limit))) if cpu_limit else detected
    if backend == "cuda":
        workers = min(cpus, 12)
        return ExecutionProfile(
            backend="cuda",
            logical_cpus=cpus,
            search_workers=1,
            bls_threads=workers,
            event_threads=workers,
            event_processes=max(1, min(3, workers // 4)),
            fixed_ephemeris_threads=workers,
            search_fixed_ephemeris_threads=workers,
            refit_workers=workers,
            event_workers=workers,
            significance_threads=workers,
            significance_processes=workers,
            vetting_threads=workers,
            vetting_processes=workers,
        )
    # Avril searches exactly three complete temporal blocks.  On a 10-core
    # Apple Silicon host, one worker per block with three threads each avoids
    # leaving the third block queued behind a 2 x 5 layout.  This changes only
    # scheduling; every block, period cell, and scientific gate is identical.
    search_workers = max(1, min(3, cpus // 3))
    per_search = max(1, cpus // search_workers)
    return ExecutionProfile(
        backend="cpu",
        logical_cpus=cpus,
        search_workers=search_workers,
        bls_threads=per_search,
        event_threads=per_search,
        event_processes=max(1, min(3, cpus // 4)),
        fixed_ephemeris_threads=min(10, cpus),
        search_fixed_ephemeris_threads=max(1, min(4, cpus // 3)),
        refit_workers=min(10, cpus),
        event_workers=min(10, cpus),
        significance_threads=min(10, cpus),
        significance_processes=min(4, cpus),
        vetting_threads=min(10, cpus),
        vetting_processes=min(10, cpus),
    )


def system_report() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": max(1, os.cpu_count() or 1),
        "cuda": cuda_status(),
    }
