"""Typed, validated loading of the YAML configs in ml/configs/.

Configs are frozen dataclasses so a run can't mutate them halfway through, and every value is
range-checked on load: a typo in YAML fails immediately instead of silently producing bad data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

RESAMPLE_METHODS = ("nearest", "bilinear", "bicubic", "lanczos", "box")
SUBSAMPLING_MODES = ("4:4:4", "4:2:2", "4:2:0")


class ConfigError(ValueError):
    """Raised when a config file is missing keys or has out-of-range values."""


@dataclass(frozen=True)
class IOConfig:
    max_file_bytes: int
    max_pixels: int
    alpha_background: tuple[int, int, int]


@dataclass(frozen=True)
class FaceConfig:
    mtcnn_min_face_size: int
    mtcnn_thresholds: tuple[float, float, float]
    min_face_size: int
    prob_threshold: float
    detect_max_side: int
    max_faces: int


@dataclass(frozen=True)
class CropConfig:
    margin_ratio: float
    output_size: int
    resample: str


@dataclass(frozen=True)
class EncodeConfig:
    offline_quality_min: int
    offline_quality_max: int
    inference_quality: int
    subsampling: str


@dataclass(frozen=True)
class PreprocessConfig:
    version: str
    seed: int
    io: IOConfig
    face: FaceConfig
    crop: CropConfig
    encode: EncodeConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DedupeConfig:
    max_hamming_distance: int
    split_priority: tuple[str, ...]


@dataclass(frozen=True)
class DataConfig:
    seed: int
    dataset_140k_name: str
    real_source_140k: str
    fake_source_140k: str
    valid_source_split: str
    val_first: str
    val_second: str
    dedupe: DedupeConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"Config {path} must be a YAML mapping, got {type(data).__name__}")
    return data


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Public YAML mapping reader with the same error handling as the typed loaders."""
    return _read_yaml(Path(path))


def _require(section: dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise ConfigError(f"Missing key '{key}' in {where}")
    return section[key]


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def load_preprocess_config(path: str | Path) -> PreprocessConfig:
    """Load preprocess.yaml (ml/configs) or the exported preprocess.json (model bundle)."""
    p = Path(path)
    if p.suffix == ".json":
        if not p.is_file():
            raise ConfigError(f"Config file not found: {p}")
        return parse_preprocess_config(json.loads(p.read_text(encoding="utf-8")))
    return parse_preprocess_config(_read_yaml(p))


def parse_preprocess_config(raw: dict[str, Any]) -> PreprocessConfig:
    io_raw = _require(raw, "io", "preprocess")
    face_raw = _require(raw, "face", "preprocess")
    crop_raw = _require(raw, "crop", "preprocess")
    enc_raw = _require(raw, "encode", "preprocess")

    io_cfg = IOConfig(
        max_file_bytes=int(_require(io_raw, "max_file_bytes", "io")),
        max_pixels=int(_require(io_raw, "max_pixels", "io")),
        alpha_background=tuple(int(v) for v in _require(io_raw, "alpha_background", "io")),  # type: ignore[arg-type]
    )
    face_cfg = FaceConfig(
        mtcnn_min_face_size=int(_require(face_raw, "mtcnn_min_face_size", "face")),
        mtcnn_thresholds=tuple(float(v) for v in _require(face_raw, "mtcnn_thresholds", "face")),  # type: ignore[arg-type]
        min_face_size=int(_require(face_raw, "min_face_size", "face")),
        prob_threshold=float(_require(face_raw, "prob_threshold", "face")),
        detect_max_side=int(_require(face_raw, "detect_max_side", "face")),
        max_faces=int(_require(face_raw, "max_faces", "face")),
    )
    crop_cfg = CropConfig(
        margin_ratio=float(_require(crop_raw, "margin_ratio", "crop")),
        output_size=int(_require(crop_raw, "output_size", "crop")),
        resample=str(_require(crop_raw, "resample", "crop")),
    )
    enc_cfg = EncodeConfig(
        offline_quality_min=int(_require(enc_raw, "offline_quality_min", "encode")),
        offline_quality_max=int(_require(enc_raw, "offline_quality_max", "encode")),
        inference_quality=int(_require(enc_raw, "inference_quality", "encode")),
        subsampling=str(_require(enc_raw, "subsampling", "encode")),
    )

    _check(io_cfg.max_file_bytes > 0 and io_cfg.max_pixels > 0, "io limits must be positive")
    _check(
        len(io_cfg.alpha_background) == 3 and all(0 <= c <= 255 for c in io_cfg.alpha_background),
        "io.alpha_background must be 3 ints in [0, 255]",
    )
    _check(len(face_cfg.mtcnn_thresholds) == 3, "face.mtcnn_thresholds must have 3 values")
    _check(all(0 < t < 1 for t in face_cfg.mtcnn_thresholds), "mtcnn_thresholds must be in (0, 1)")
    _check(0 < face_cfg.prob_threshold < 1, "face.prob_threshold must be in (0, 1)")
    _check(face_cfg.min_face_size > 0 and face_cfg.mtcnn_min_face_size > 0, "face sizes must be > 0")
    _check(face_cfg.detect_max_side >= 256, "face.detect_max_side must be >= 256")
    _check(face_cfg.max_faces >= 1, "face.max_faces must be >= 1")
    _check(0 <= crop_cfg.margin_ratio <= 2, "crop.margin_ratio must be in [0, 2]")
    _check(32 <= crop_cfg.output_size <= 1024, "crop.output_size must be in [32, 1024]")
    _check(crop_cfg.resample in RESAMPLE_METHODS, f"crop.resample must be one of {RESAMPLE_METHODS}")
    _check(
        1 <= enc_cfg.offline_quality_min <= enc_cfg.offline_quality_max <= 100,
        "encode: need 1 <= offline_quality_min <= offline_quality_max <= 100",
    )
    _check(1 <= enc_cfg.inference_quality <= 100, "encode.inference_quality must be in [1, 100]")
    _check(enc_cfg.subsampling in SUBSAMPLING_MODES, f"encode.subsampling must be in {SUBSAMPLING_MODES}")

    return PreprocessConfig(
        version=str(_require(raw, "version", "preprocess")),
        seed=int(_require(raw, "seed", "preprocess")),
        io=io_cfg,
        face=face_cfg,
        crop=crop_cfg,
        encode=enc_cfg,
    )


def export_preprocess_json(cfg: PreprocessConfig, path: str | Path) -> Path:
    """Write the preprocessing contract consumed by the backend (single source of truth)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    return out


def load_data_config(path: str | Path) -> DataConfig:
    raw = _read_yaml(Path(path))
    ds = _require(raw, "dataset_140k", "data")
    sources = _require(ds, "sources", "dataset_140k")
    val = _require(raw, "validation_subsplits", "data")
    dd = _require(raw, "dedupe", "data")

    dedupe = DedupeConfig(
        max_hamming_distance=int(_require(dd, "max_hamming_distance", "dedupe")),
        split_priority=tuple(str(s) for s in _require(dd, "split_priority", "dedupe")),
    )
    _check(0 <= dedupe.max_hamming_distance <= 16, "dedupe.max_hamming_distance must be in [0, 16]")
    _check(
        len(set(dedupe.split_priority)) == len(dedupe.split_priority),
        "dedupe.split_priority has duplicates",
    )
    cfg = DataConfig(
        seed=int(_require(raw, "seed", "data")),
        dataset_140k_name=str(_require(ds, "name", "dataset_140k")),
        real_source_140k=str(_require(sources, "real", "dataset_140k.sources")),
        fake_source_140k=str(_require(sources, "fake", "dataset_140k.sources")),
        valid_source_split=str(_require(val, "source_split", "validation_subsplits")),
        val_first=str(_require(val, "first", "validation_subsplits")),
        val_second=str(_require(val, "second", "validation_subsplits")),
        dedupe=dedupe,
    )
    for split in ("train", cfg.val_first, cfg.val_second, "test"):
        _check(split in dedupe.split_priority, f"split '{split}' missing from dedupe.split_priority")
    return cfg
