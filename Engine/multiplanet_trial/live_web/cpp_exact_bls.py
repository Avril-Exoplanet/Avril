"""Build and call VLora's exact standalone C++/NEON BLS kernel.

The extension is deliberately compiled from the shipped source on first use.
This keeps the open-source package portable across Apple Silicon and ordinary
Linux machines without making a compiler-specific binary part of the source
tree.  Failure to compile is non-fatal: the caller keeps the exact Numba path.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np


CPP_BACKEND_ENV = "HOUXING_EXACT_BLS_BACKEND"
CPP_CACHE_ENV = "VLORA_NATIVE_CACHE_DIR"
CPP_STRATEGY_ENV = "HOUXING_EXACT_BLS_CPP_STRATEGY"
_SOURCE = Path(__file__).with_name("native") / "exact_bls_kernel.cpp"
_LOAD_LOCK = threading.Lock()
_LIBRARY: ctypes.CDLL | None = None
_LOAD_ERROR: str | None = None


def _cache_directory() -> Path:
    override = os.environ.get(CPP_CACHE_ENV)
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "VLora" / "native"
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "vlora" / "native"


def _compiler() -> list[str] | None:
    requested = os.environ.get("CXX")
    if requested:
        command = shlex.split(requested)
        if command and shutil.which(command[0]):
            return command
        return None
    for candidate in ("clang++", "g++", "c++"):
        found = shutil.which(candidate)
        if found:
            return [found]
    return None


def _library_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if os.name == "nt":
        return ".dll"
    return ".so"


def _build_library() -> Path:
    # Desktop packaging adapter: this binary is built from the same frozen C++
    # source and flags for baseline Apple Silicon, so users need no compiler.
    bundled = os.environ.get("AVRIL_BUNDLED_EXACT_BLS")
    if bundled:
        packaged_library = Path(bundled).expanduser().resolve()
        if not packaged_library.is_file():
            raise RuntimeError(f"bundled native BLS library is missing: {packaged_library}")
        return packaged_library
    if not _SOURCE.is_file():
        raise RuntimeError(f"missing native BLS source: {_SOURCE}")
    compiler = _compiler()
    if compiler is None:
        raise RuntimeError("no C++17 compiler found")
    source_bytes = _SOURCE.read_bytes()
    identity = b"\0".join(
        (
            source_bytes,
            platform.platform().encode(),
            platform.machine().encode(),
            " ".join(compiler).encode(),
        )
    )
    digest = hashlib.sha256(identity).hexdigest()[:16]
    cache = _cache_directory()
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f"exact_bls_{digest}{_library_suffix()}"
    if output.is_file():
        return output

    temporary_directory = Path(tempfile.mkdtemp(prefix="vlora-native-", dir=cache))
    temporary_output = temporary_directory / output.name
    architecture_flags = (
        ["-mcpu=native"]
        if platform.machine().lower() in {"arm64", "aarch64"}
        else ["-march=native"]
    )
    shared_flags = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared"]
    command = [
        *compiler,
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-fPIC",
        "-pthread",
        "-ffp-contract=off",
        "-fno-fast-math",
        *architecture_flags,
        *shared_flags,
        str(_SOURCE),
        "-o",
        str(temporary_output),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0 or not temporary_output.is_file():
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"native BLS compilation failed ({completed.returncode}): {detail}"
            )
        try:
            os.replace(temporary_output, output)
        except OSError:
            # Another process may have completed the same content-addressed
            # build first.  The final file is interchangeable.
            if not output.is_file():
                raise
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)
    return output


def _load_library() -> ctypes.CDLL:
    global _LIBRARY, _LOAD_ERROR
    if _LIBRARY is not None:
        return _LIBRARY
    with _LOAD_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        if _LOAD_ERROR is not None:
            raise RuntimeError(_LOAD_ERROR)
        try:
            library = ctypes.CDLL(str(_build_library()))
            function = library.vlora_exact_bls_kernel_v2
            double_pointer = ctypes.POINTER(ctypes.c_double)
            int64_pointer = ctypes.POINTER(ctypes.c_int64)
            function.argtypes = [
                double_pointer,
                double_pointer,
                ctypes.c_int64,
                double_pointer,
                double_pointer,
                ctypes.c_int64,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
                double_pointer,
                int64_pointer,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            function.restype = ctypes.c_int
        except Exception as exc:  # pragma: no cover - machine/toolchain specific
            _LOAD_ERROR = str(exc)
            raise RuntimeError(_LOAD_ERROR) from exc
        _LIBRARY = library
        return library


def backend_available() -> bool:
    """Return whether the native library can be built and loaded."""

    try:
        _load_library()
    except RuntimeError:
        return False
    return True


def load_error() -> str | None:
    """Return the cached build/load error, if native setup failed."""

    return _LOAD_ERROR


def exact_bls_phase_search(
    relative_time: np.ndarray,
    centered_flux: np.ndarray,
    periods: np.ndarray,
    inverse_periods: np.ndarray,
    *,
    duration: float,
    oversample: int,
    threads: int,
    tile_size: int,
    sum_y: float,
    sum_ivar: float,
    strategy: str | None = None,
) -> dict[str, np.ndarray | int | str]:
    """Run every supplied period through the C++ kernel without omission."""

    time_array = np.ascontiguousarray(relative_time, dtype=np.float64)
    flux_array = np.ascontiguousarray(centered_flux, dtype=np.float64)
    period_array = np.ascontiguousarray(periods, dtype=np.float64)
    inverse_array = np.ascontiguousarray(inverse_periods, dtype=np.float64)
    if time_array.ndim != 1 or flux_array.shape != time_array.shape:
        raise ValueError("native time and flux arrays must be aligned vectors")
    if period_array.ndim != 1 or inverse_array.shape != period_array.shape:
        raise ValueError("native period and inverse-period arrays must be aligned vectors")
    power = np.full(period_array.size, -np.inf, dtype=np.float64)
    phase_indices = np.full(period_array.size, -1, dtype=np.int64)
    effective_threads = ctypes.c_int(0)
    used_neon = ctypes.c_int(0)
    selected_strategy = (
        os.environ.get(CPP_STRATEGY_ENV, "phase-runs")
        if strategy is None
        else strategy
    ).strip().lower()
    strategy_codes = {
        "tiled-neon": 0,
        "phase-runs": 1,
    }
    if selected_strategy not in strategy_codes:
        raise ValueError(
            f"{CPP_STRATEGY_ENV} must be 'tiled-neon' or 'phase-runs'"
        )
    strategy_code = strategy_codes[selected_strategy]
    function = _load_library().vlora_exact_bls_kernel_v2
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int64_pointer = ctypes.POINTER(ctypes.c_int64)
    status = function(
        time_array.ctypes.data_as(double_pointer),
        flux_array.ctypes.data_as(double_pointer),
        ctypes.c_int64(time_array.size),
        period_array.ctypes.data_as(double_pointer),
        inverse_array.ctypes.data_as(double_pointer),
        ctypes.c_int64(period_array.size),
        ctypes.c_double(float(duration)),
        ctypes.c_int(int(oversample)),
        ctypes.c_int(int(threads)),
        ctypes.c_int(int(tile_size)),
        ctypes.c_int(strategy_code),
        ctypes.c_double(float(sum_y)),
        ctypes.c_double(float(sum_ivar)),
        power.ctypes.data_as(double_pointer),
        phase_indices.ctypes.data_as(int64_pointer),
        ctypes.byref(effective_threads),
        ctypes.byref(used_neon),
    )
    if status != 0:
        raise RuntimeError(f"native exact BLS kernel failed with status {status}")
    return {
        "power": power,
        "best_phase_indices": phase_indices,
        "threads": int(effective_threads.value),
        "simd": "neon" if used_neon.value else "scalar",
        "backend": "cpp",
        "strategy": selected_strategy,
    }
