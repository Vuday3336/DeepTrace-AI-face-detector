"""Export a trained classifier to ONNX with the extra outputs needed for heatmaps.

  CNN: image -> (logit [N], features [N, C, 7, 7])        closed-form Grad-CAM in numpy
  ViT: image -> (logit [N], attentions [N, L, T, T])      attention rollout in numpy

Why the legacy TorchScript exporter (`dynamo=False`): the ViT attention maps are captured by forward
hooks during tracing, which the tracer records as graph outputs; it is also the most battle-tested
path for these timm architectures with ONNX Runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from deeptrace_ml.explain.torch_cam import AttentionRecorder
from deeptrace_ml.models.classifier import BinaryClassifier

OPSET = 17


class CnnExportWrapper(nn.Module):
    def __init__(self, model: BinaryClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_with_features(image)


class VitExportWrapper(nn.Module):
    def __init__(self, model: BinaryClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with AttentionRecorder(self.model) as recorder:
            logit = self.model(image)
        return logit, recorder.stacked()


def output_names(model: BinaryClassifier) -> list[str]:
    return ["logit", "features" if model.spec.family == "cnn" else "attentions"]


def export_onnx(model: BinaryClassifier, path: str | Path, input_size: int = 224) -> Path:
    model = model.eval().cpu()
    wrapper: nn.Module = CnnExportWrapper(model) if model.spec.family == "cnn" else VitExportWrapper(model)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, input_size, input_size)
    names = output_names(model)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out),
        input_names=["image"],
        output_names=names,
        dynamic_axes={"image": {0: "batch"}, names[0]: {0: "batch"}, names[1]: {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    return out


def verify_onnx(
    model: BinaryClassifier, path: str | Path, n: int = 2, input_size: int = 224, seed: int = 0
) -> dict[str, float]:
    """Max absolute difference between PyTorch and ONNX Runtime for every output."""
    import onnxruntime as ort

    torch.manual_seed(seed)
    x = torch.randn(n, 3, input_size, input_size)
    wrapper: nn.Module = CnnExportWrapper(model) if model.spec.family == "cnn" else VitExportWrapper(model)
    with torch.no_grad():
        expected = [t.numpy() for t in wrapper.eval()(x)]
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"image": x.numpy()})
    diffs = {
        name: float(np.max(np.abs(e - a))) for name, e, a in zip(output_names(model), expected, actual, strict=True)
    }
    return diffs
