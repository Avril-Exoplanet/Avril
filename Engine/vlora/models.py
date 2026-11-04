"""Small immutable records shared by VLora source and engine layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TargetIdentity:
    target: str
    tic_id: int
    ra_deg: float
    dec_deg: float


@dataclass
class PreparedInput:
    source_kind: str
    target: str
    products: list[dict]
    lightcurve_paths: list[Path] = field(default_factory=list)
    pixel_paths: list[Path] = field(default_factory=list)
    identity: TargetIdentity | None = None
    receipt: dict = field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        return bool(self.lightcurve_paths)
