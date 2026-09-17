from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from deeptrace_ml.config import ConfigError, export_preprocess_json, load_data_config, load_preprocess_config

ML_DIR = Path(__file__).resolve().parents[1]


def test_repo_configs_load(preprocess_cfg):
    assert preprocess_cfg.crop.output_size == 224
    data_cfg = load_data_config(ML_DIR / "configs/data.yaml")
    assert data_cfg.val_first == "val_select"
    assert data_cfg.dedupe.split_priority[0] == "train"


def test_invalid_quality_range_rejected(tmp_path):
    raw = yaml.safe_load((ML_DIR / "configs/preprocess.yaml").read_text(encoding="utf-8"))
    raw["encode"]["offline_quality_min"] = 96  # > max (95)
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="offline_quality"):
        load_preprocess_config(bad)


def test_missing_key_rejected(tmp_path):
    raw = yaml.safe_load((ML_DIR / "configs/preprocess.yaml").read_text(encoding="utf-8"))
    del raw["crop"]["margin_ratio"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="margin_ratio"):
        load_preprocess_config(bad)


def test_export_json_roundtrip(preprocess_cfg, tmp_path):
    out = export_preprocess_json(preprocess_cfg, tmp_path / "preprocess.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["encode"]["inference_quality"] == preprocess_cfg.encode.inference_quality
    assert data["version"] == preprocess_cfg.version
