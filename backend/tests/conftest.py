"""Test fixtures: a tiny but REAL ONNX bundle + a stub face detector.

The model mimics the serving contract of the exported CNNs: image -> (logit, 7x7 feature map), built
as Conv(stride 32) -> GlobalAveragePool -> Gemm. Its logit is driven by mean brightness, so tests can
steer the verdict by choosing bright or dark images.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from app.core.settings import Settings
from app.main import create_app
from deeptrace_ml.config import export_preprocess_json, load_preprocess_config
from deeptrace_ml.preprocessing.crop import Box
from deeptrace_ml.preprocessing.face import DetectedFace
from deeptrace_ml.serving.runtime import Analyzer, ModelBundle
from deeptrace_ml.utils.hashing import sha256_file

REPO = Path(__file__).resolve().parents[2]
PREPROCESS_YAML = REPO / "ml/configs/preprocess.yaml"
CHANNELS = 4


def _build_onnx(path: Path) -> tuple[list[float], float]:
    rng = np.random.default_rng(0)
    conv_w = rng.normal(0, 0.01, size=(CHANNELS, 3, 32, 32)).astype(np.float32) + 0.002  # brightness-sensitive
    conv_b = np.zeros(CHANNELS, dtype=np.float32)
    fc_w = np.full((1, CHANNELS), 0.5, dtype=np.float32)
    fc_b = np.zeros(1, dtype=np.float32)
    nodes = [
        helper.make_node("Conv", ["image", "conv_w", "conv_b"], ["features"], strides=[32, 32]),
        helper.make_node("GlobalAveragePool", ["features"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "fc_w", "fc_b"], ["logit_2d"], transB=1),
        helper.make_node("Reshape", ["logit_2d", "shape_minus1"], ["logit"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_deeptrace",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 224, 224])],
        [
            helper.make_tensor_value_info("logit", TensorProto.FLOAT, ["batch"]),
            helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", CHANNELS, 7, 7]),
        ],
        initializer=[
            numpy_helper.from_array(conv_w, "conv_w"),
            numpy_helper.from_array(conv_b, "conv_b"),
            numpy_helper.from_array(fc_w, "fc_w"),
            numpy_helper.from_array(fc_b, "fc_b"),
            numpy_helper.from_array(np.array([-1], dtype=np.int64), "shape_minus1"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return fc_w.reshape(-1).tolist(), float(fc_b[0])


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("bundle")
    weights, bias = _build_onnx(root / "model.onnx")
    cfg = load_preprocess_config(PREPROCESS_YAML)
    export_preprocess_json(cfg, root / "preprocess.json")
    card = {
        "card_version": 1,
        "model": {
            "name": "tiny_test",
            "timm_name": "none",
            "family": "cnn",
            "frozen_backbone": False,
            "version": "0.0.0-test",
            "input_size": 224,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "onnx": {"file": "model.onnx", "input": "image", "outputs": ["logit", "features"]},
        "decision": {
            "temperature": 1.0,
            "threshold": 0.5,
            "uncertainty_delta": 0.05,
            "target_fpr": 0.05,
            "positive_label": "AI_GENERATED",
        },
        "explanation": {"method": "gradcam", "head": {"weight": weights, "bias": bias, "norm": None}},
        "preprocess_version": cfg.version,
        "training_data": {"name": "synthetic test fixture"},
        "metrics": {"status": "not_evaluated"},
        "known_limitations": ["Test fixture only."],
        "disclaimer": "Test fixture.",
    }
    (root / "model_card.json").write_text(json.dumps(card), encoding="utf-8")
    files = ["model.onnx", "model_card.json", "preprocess.json"]
    (root / "bundle_manifest.json").write_text(
        json.dumps({"files": {f: sha256_file(root / f) for f in files}}), encoding="utf-8"
    )
    return root


class StubDetector:
    """Finds N centred 'faces' when the image is wide enough; images narrower than 100px have no face."""

    def __init__(self, faces_per_image: int = 1) -> None:
        self.faces_per_image = faces_per_image

    def detect(self, images: Sequence[Image.Image]) -> list[list[DetectedFace]]:
        out = []
        for im in images:
            if im.width < 100:
                out.append([])
                continue
            w = im.width / self.faces_per_image
            out.append(
                [
                    DetectedFace(Box(i * w + w * 0.2, im.height * 0.2, i * w + w * 0.8, im.height * 0.8), 0.99)
                    for i in range(self.faces_per_image)
                ]
            )
        return out


@pytest.fixture(scope="session")
def bundle(bundle_dir: Path) -> ModelBundle:
    return ModelBundle.load(bundle_dir)


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "model_dir": tmp_path / "unused",
        "upload_dir": tmp_path / "uploads",
        "rate_limit_requests": 1000,
        "cors_origins": ["http://testserver"],
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


@pytest.fixture
def client(tmp_path: Path, bundle: ModelBundle) -> Iterator[TestClient]:
    app = create_app(make_settings(tmp_path), analyzer=Analyzer(bundle, StubDetector()))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def history_client(tmp_path: Path, bundle: ModelBundle) -> Iterator[TestClient]:
    settings = make_settings(
        tmp_path, database_url=f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}", jwt_secret="x" * 40
    )
    app = create_app(settings, analyzer=Analyzer(bundle, StubDetector()))
    with TestClient(app) as c:
        yield c


def image_bytes(
    width: int = 320, height: int = 240, value: int = 128, fmt: str = "JPEG", **save_kwargs: object
) -> bytes:
    rng = np.random.default_rng(value)
    arr = np.clip(rng.normal(value, 20, size=(height, width, 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()
