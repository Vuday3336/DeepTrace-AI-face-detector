"""Obtain and load the model bundle and face detector at startup."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from app.core.settings import Settings
from deeptrace_ml.preprocessing.face import DetectedFace, FaceDetector, MTCNNFaceDetector
from deeptrace_ml.serving.runtime import REQUIRED_FILES, BundleError, ModelBundle

log = logging.getLogger("deeptrace.loader")


def ensure_bundle_files(settings: Settings) -> Path:
    """Use the local bundle if complete; otherwise download the pinned revision from the Hub."""
    model_dir = settings.model_dir
    if all((model_dir / f).is_file() for f in REQUIRED_FILES):
        return model_dir
    if not settings.model_repo_id:
        raise BundleError(
            f"No model bundle in {model_dir} and DEEPTRACE_MODEL_REPO_ID is not set. "
            "Export one with ml/scripts/export_model.py or configure the Hub repo."
        )
    from huggingface_hub import snapshot_download

    log.info("Downloading model %s@%s to %s", settings.model_repo_id, settings.model_revision, model_dir)
    snapshot_download(repo_id=settings.model_repo_id, revision=settings.model_revision, local_dir=str(model_dir))
    return model_dir


def load_bundle(settings: Settings) -> ModelBundle:
    directory = ensure_bundle_files(settings)
    bundle = ModelBundle.load(
        directory, verify_checksums=settings.verify_model_checksums, intra_op_threads=settings.onnx_threads
    )
    log.info("Loaded model %s v%s (%s)", bundle.name, bundle.version, bundle.explanation["method"])
    return bundle


class LockedDetector:
    """MTCNN (PyTorch) is not guaranteed thread-safe; serialise calls from the threadpool."""

    def __init__(self, inner: FaceDetector) -> None:
        self._inner = inner
        self._lock = threading.Lock()

    def detect(self, images: Sequence[Image.Image]) -> list[list[DetectedFace]]:
        with self._lock:
            return self._inner.detect(images)


def load_detector(bundle: ModelBundle, settings: Settings) -> FaceDetector:
    return LockedDetector(MTCNNFaceDetector(bundle.preprocess.face, device=settings.detector_device))
