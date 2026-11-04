"""Cross-platform storage locations for public Avril builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def default_data_root() -> Path:
    """Return an OS-native data directory without assuming a developer disk."""

    configured = os.environ.get("AVRIL_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Avril"
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "Avril"
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home).expanduser().resolve() / "avril"
    return Path.home() / ".local" / "share" / "avril"
