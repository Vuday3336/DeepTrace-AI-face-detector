"""JSON helpers that handle numpy/pandas scalars and paths (common in metrics dicts)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None  # JSON has no NaN/inf; null is explicit
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def write_json(data: Any, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")
    return out
