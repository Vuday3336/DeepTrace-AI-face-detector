"""Capture the software/hardware environment of a run (written to env.json)."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "numpy",
    "pillow",
    "pandas",
    "scipy",
    "scikit-learn",
    "imagehash",
    "matplotlib",
    "pyyaml",
    "torch",
    "torchvision",
    "timm",
    "albumentations",
    "facenet-pytorch",
    "open_clip_torch",
    "diffusers",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "onnx",
    "onnxruntime",
)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_commit(cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}
    info: dict[str, Any] = {
        "available": True,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def capture_environment(repo_dir: str | Path = ".") -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(Path(repo_dir)),
        "packages": _package_versions(),
        "torch": _torch_info(),
    }


def write_environment(path: str | Path, repo_dir: str | Path = ".") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(capture_environment(repo_dir), indent=2), encoding="utf-8")
    return out
