"""Offline explanations with PyTorch (galleries, sanity checks, parity references).

- CNN:  Grad-CAM and Grad-CAM++ on the last feature map (pytorch-grad-cam).
- ViT:  Grad-CAM on the last block's token outputs, reshaped to a 14x14 grid (reshape_transform),
        plus attention rollout (class-agnostic) captured from the attention softmax.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from torch import nn

from deeptrace_ml.explain.closed_form import attention_rollout
from deeptrace_ml.models.classifier import BinaryClassifier


class _LogitAsColumn(nn.Module):
    """pytorch-grad-cam expects [N, classes]; our model returns [N]."""

    def __init__(self, model: BinaryClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).unsqueeze(1)


def vit_reshape_transform(num_prefix_tokens: int) -> Callable[[torch.Tensor], torch.Tensor]:
    def transform(tokens: torch.Tensor) -> torch.Tensor:
        patches = tokens[:, num_prefix_tokens:, :]
        side = int(round(patches.shape[1] ** 0.5))
        return patches.reshape(patches.shape[0], side, side, patches.shape[2]).permute(0, 3, 1, 2)

    return transform


def _target_layer_and_transform(model: BinaryClassifier) -> tuple[nn.Module, Callable | None]:
    backbone = model.backbone
    if model.spec.family == "vit":
        return backbone.blocks[-1].norm1, vit_reshape_transform(backbone.num_prefix_tokens)
    if hasattr(backbone, "conv_head"):  # EfficientNet: after conv_head + bn2/act
        return backbone.bn2, None
    if hasattr(backbone, "stages"):  # ConvNeXt
        return backbone.stages[-1], None
    raise ValueError(f"No Grad-CAM target layer known for {model.spec.name}")


def gradcam_maps(
    model: BinaryClassifier, batch: torch.Tensor, method: str = "gradcam", target: str = "fake"
) -> np.ndarray:
    """Normalised [N, 224, 224] maps from pytorch-grad-cam."""
    layer, reshape = _target_layer_and_transform(model)
    cam_cls = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus}[method]
    wrapped = _LogitAsColumn(model).eval()
    needs_grad = [p for p in model.parameters() if not p.requires_grad]
    for p in needs_grad:  # frozen probe backbone still needs activations' grads
        p.requires_grad_(True)
    try:
        with cam_cls(model=wrapped, target_layers=[layer], reshape_transform=reshape) as cam:
            targets = [BinaryClassifierOutputTarget(1 if target == "fake" else 0)] * len(batch)
            return cam(input_tensor=batch, targets=targets)
    finally:
        for p in needs_grad:
            p.requires_grad_(False)


class AttentionRecorder:
    """Collects head-averaged attention matrices from every ViT block during a forward pass.

    timm's fused attention (scaled_dot_product_attention) never materialises the matrix, so the
    recorder switches blocks to the explicit path and hooks the attention dropout's INPUT (the softmax).
    """

    def __init__(self, model: BinaryClassifier) -> None:
        if model.spec.family != "vit":
            raise ValueError("AttentionRecorder only applies to ViT models")
        self.blocks = model.backbone.blocks
        self.maps: list[torch.Tensor] = []
        self._handles: list = []
        self._fused: list[bool] = []

    def __enter__(self) -> AttentionRecorder:
        for block in self.blocks:
            self._fused.append(block.attn.fused_attn)
            block.attn.fused_attn = False
            self._handles.append(block.attn.attn_drop.register_forward_hook(self._hook))
        return self

    def _hook(self, _module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        self.maps.append(inputs[0].mean(dim=1))  # [N, T, T]

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        for block, fused in zip(self.blocks, self._fused, strict=True):
            block.attn.fused_attn = fused

    def stacked(self) -> torch.Tensor:
        """[N, L, T, T]"""
        return torch.stack(self.maps, dim=1)


@torch.no_grad()
def rollout_maps(model: BinaryClassifier, batch: torch.Tensor) -> np.ndarray:
    with AttentionRecorder(model) as rec:
        model(batch)
    attn = rec.stacked().float().cpu().numpy()
    prefix = model.backbone.num_prefix_tokens
    return np.stack([attention_rollout(a, prefix) for a in attn])


def randomize_weights(model: BinaryClassifier, seed: int = 0) -> BinaryClassifier:
    """Model-randomisation sanity check (Adebayo et al., 2018): re-initialise ALL weights.
    If heatmaps barely change, they reflect image structure rather than what the model learned."""
    import copy

    torch.manual_seed(seed)
    clone = copy.deepcopy(model)
    for module in clone.modules():
        if hasattr(module, "reset_parameters") and module is not clone:
            module.reset_parameters()
    return clone.eval()
