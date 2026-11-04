"""Configuration for Avril photometry acquisition and self-photometry."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from .paths import default_data_root


class SourceMode(StrEnum):
    AUTO = "auto"
    SPOC = "spoc"
    FFI = "ffi"


def large_data_root() -> Path:
    return default_data_root()


def safe_target_name(target: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", target.lower()).strip("-")
    return value or "target"


@dataclass(frozen=True)
class SourceConfig:
    target: str
    source: SourceMode = SourceMode.AUTO
    mission: str = "TESS"
    output_dir: Path | None = None
    cache_root: Path | None = None
    period_max_days: float = 400.0
    significance_phase_trials: int = 4096
    download_workers: int = 80
    fixed_ephemeris_threads: int = min(10, os.cpu_count() or 1)
    significance_threads: int = min(10, os.cpu_count() or 1)
    run_pixel_vetting: bool = False
    query_catalog_after_lock: bool = False
    deep_cadence_selection: bool = True
    local_lightcurve_dir: Path | None = None
    local_pixel_dir: Path | None = None
    tic_id: int | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    tesscut_size_pixels: int = 15
    tesscut_connections_per_file: int = 16
    tesscut_file_workers: int = 8
    photometry_workers: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", SourceMode(self.source))
        object.__setattr__(self, "mission", str(self.mission).upper())
        if not self.target.strip():
            raise ValueError("target cannot be empty")
        if self.mission != "TESS":
            raise ValueError("Avril FFI self-photometry is currently TESS-only")
        if self.period_max_days <= 0:
            raise ValueError("period_max_days must be positive")
        if self.significance_phase_trials < 0:
            raise ValueError("significance_phase_trials cannot be negative")
        for name in (
            "download_workers",
            "fixed_ephemeris_threads",
            "significance_threads",
            "tesscut_size_pixels",
            "tesscut_connections_per_file",
            "tesscut_file_workers",
            "photometry_workers",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        if (self.ra_deg is None) != (self.dec_deg is None):
            raise ValueError("ra_deg and dec_deg must be supplied together")

        root = large_data_root()
        output = self.output_dir or (root / "runs" / safe_target_name(self.target))
        cache = self.cache_root or (root / "cache")
        object.__setattr__(self, "output_dir", Path(output).expanduser().resolve())
        object.__setattr__(self, "cache_root", Path(cache).expanduser().resolve())
        for name in ("local_lightcurve_dir", "local_pixel_dir"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())

    def to_dict(self) -> dict:
        values = asdict(self)
        values["source"] = self.source.value
        for key, value in list(values.items()):
            if isinstance(value, Path):
                values[key] = str(value)
        return values
