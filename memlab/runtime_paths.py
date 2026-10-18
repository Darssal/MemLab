"""Resolve the external DeerFlow checkout without copying its source tree."""
from __future__ import annotations

import os
from pathlib import Path


def deerflow_root() -> Path:
    configured = os.getenv("MEMLAB_DEERFLOW_ROOT", "").strip()
    return Path(configured or "deer-flow").expanduser().resolve()
