"""Serving runtime: model bundle -> per-face verdicts and heatmaps. numpy + onnxruntime + Pillow only.

This is the exact code path the FastAPI backend runs, so offline benchmarks and parity tests measure
production behaviour rather than a re-implementation.

Pipeline per uploaded image:
  decoded RGB (metadata stripped) -> face detector -> for each face: square crop -> 224px ->
  JPEG at inference quality -> normalise -> ONNX (batched over faces) -> logit / T -> sigmoid ->
  REAL / AI_GENERATED / UNCERTAIN -> heatmap (closed-form Grad-CAM or attention rollout)
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from deeptrace_ml.config import PreprocessConfig, load_preprocess_config
from deeptrace_ml.explain.closed_form import attention_rollout, gap_gradcam, heatmap_rgba
from deeptrace_ml.preprocessing.face import DetectedFace, FaceDetector
from deeptrace_ml.preprocessing.pipeline import crop_faces, normalize_image, to_inference_image
from deeptrace_ml.serving.decision import verdict
from deeptrace_ml.utils.hashing import sha256_file

REQUIRED_FILES = ("model.onnx", "model_card.json", "preprocess.json")


class BundleError(RuntimeError):
    """The model bundle is missing, corrupted or inconsistent."""


@dataclass
class FaceResult:
    index: int
    box: tuple[float, float, float, float]
    detection_score: float
    prob_ai_generated: float
    label: str
    face_crop: Image.Image
    heatmap_rgba: np.ndarray | None


@dataclass
class AnalysisResult:
    width: int
    height: int
    faces: list[FaceResult]
    timings_ms: dict[str, float] = field(default_factory=dict)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class ModelBundle:
    def __init__(self, directory: Path, card: dict[str, Any], preprocess: PreprocessConfig, session: Any) -> None:
        self.directory = directory
        self.card = card
        self.preprocess = preprocess
        self.session = session
        model = card["model"]
        self.mean, self.std = model["mean"], model["std"]
        decision = card["decision"]
        self.temperature = float(decision["temperature"])
        self.threshold = float(decision["threshold"])
        self.uncertainty_delta = float(decision["uncertainty_delta"])
        self.explanation = card["explanation"]

    @property
    def name(self) -> str:
        return str(self.card["model"]["name"])

    @property
    def version(self) -> str:
        return str(self.card["model"]["version"])

    @classmethod
    def load(
        cls, directory: str | Path, verify_checksums: bool = True, intra_op_threads: int | None = None
    ) -> ModelBundle:
        import onnxruntime as ort

        root = Path(directory)
        missing = [f for f in REQUIRED_FILES if not (root / f).is_file()]
        if missing:
            raise BundleError(f"Model bundle at {root} is missing {missing}")
        if verify_checksums:
            manifest_path = root / "bundle_manifest.json"
            if not manifest_path.is_file():
                raise BundleError(f"{manifest_path} missing; cannot verify bundle integrity")
            expected = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
            for name, digest in expected.items():
                if sha256_file(root / name) != digest:
                    raise BundleError(f"Checksum mismatch for {name}: bundle is corrupted or was modified")
        card = json.loads((root / "model_card.json").read_text(encoding="utf-8"))
        preprocess = load_preprocess_config(root / "preprocess.json")
        if card.get("preprocess_version") != preprocess.version:
            raise BundleError(
                f"model_card expects preprocess {card.get('preprocess_version')} but bundle has {preprocess.version}"
            )
        options = ort.SessionOptions()
        if intra_op_threads:
            options.intra_op_num_threads = intra_op_threads
        session = ort.InferenceSession(str(root / "model.onnx"), options, providers=["CPUExecutionProvider"])
        return cls(root, card, preprocess, session)

    def infer(self, crops: Sequence[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        """Batched ONNX forward pass on 224px crops. Returns (logits [N], aux [N, ...])."""
        batch = np.concatenate([normalize_image(c, self.mean, self.std) for c in crops])
        logits, aux = self.session.run(None, {"image": batch})
        return np.asarray(logits, dtype=np.float64).reshape(-1), aux

    def heatmap(self, aux_single: np.ndarray, size: int) -> np.ndarray:
        method = self.explanation["method"]
        if method == "gradcam":
            cam = gap_gradcam(aux_single, self.explanation["head"], target="fake")
        elif method == "attention_rollout":
            cam = attention_rollout(aux_single, int(self.explanation["num_prefix_tokens"]))
        else:
            raise BundleError(f"Unknown explanation method '{method}'")
        return heatmap_rgba(cam, size)

    def decide(self, logits: np.ndarray) -> tuple[np.ndarray, list[str]]:
        probs = _sigmoid(logits / self.temperature)
        return probs, [verdict(float(p), self.threshold, self.uncertainty_delta) for p in probs]


class Analyzer:
    def __init__(self, bundle: ModelBundle, detector: FaceDetector) -> None:
        self.bundle = bundle
        self.detector = detector

    def detect(self, image: Image.Image) -> list[DetectedFace]:
        return self.detector.detect([image])[0]

    def analyze(self, image: Image.Image, explain: bool = True) -> AnalysisResult:
        cfg = self.bundle.preprocess
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        faces = self.detect(image)
        timings["detect"] = (time.perf_counter() - t0) * 1000
        result = AnalysisResult(width=image.width, height=image.height, faces=[], timings_ms=timings)
        if not faces:
            return result

        t0 = time.perf_counter()
        cropped = crop_faces(image, faces, cfg)
        model_inputs = [to_inference_image(c.image, cfg) for c in cropped]
        logits, aux = self.bundle.infer(model_inputs)
        probs, labels = self.bundle.decide(logits)
        timings["inference"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for i, (crop, prob, label) in enumerate(zip(cropped, probs, labels, strict=True)):
            box = crop.face.box
            result.faces.append(
                FaceResult(
                    index=i,
                    box=(box.x1, box.y1, box.x2, box.y2),
                    detection_score=crop.face.prob,
                    prob_ai_generated=float(prob),
                    label=label,
                    face_crop=model_inputs[i],
                    heatmap_rgba=self.bundle.heatmap(aux[i], cfg.crop.output_size) if explain else None,
                )
            )
        timings["explain"] = (time.perf_counter() - t0) * 1000 if explain else 0.0
        return result
