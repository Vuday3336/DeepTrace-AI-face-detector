"""Where scripts write reports and plots.

Default: ml/reports and docs/plots inside the repo. Set DEEPTRACE_ARTIFACTS_DIR to redirect both
(<dir>/reports, <dir>/plots) — used by smoke tests and CI so synthetic numbers never land in the repo.
"""

from __future__ import annotations

import os
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[3]
REPO_DIR = ML_DIR.parent


def _override() -> Path | None:
    value = os.environ.get("DEEPTRACE_ARTIFACTS_DIR")
    return Path(value) if value else None


def reports_dir() -> Path:
    root = _override()
    return root / "reports" if root else ML_DIR / "reports"


def plots_dir() -> Path:
    root = _override()
    return root / "plots" if root else REPO_DIR / "docs/plots"
