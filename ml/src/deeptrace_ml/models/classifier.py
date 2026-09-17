"""One wrapper for every backbone: image -> single logit (positive = AI-generated).

Structure: timm backbone WITHOUT its ImageNet head (num_classes=0) + our own nn.Linear(features, 1).
Using timm's `forward_features` / `forward_head(pre_logits=True)` split means every model exposes:
  * features: last feature map (CNN: N x C x 7 x 7) or token sequence (ViT: N x 197 x D) — for heatmaps
  * pooled:   the vector the linear head reads
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import timm
import torch
from torch import nn

from deeptrace_ml.models.registry import ModelSpec, get_spec


class BinaryClassifier(nn.Module):
    def __init__(self, spec: ModelSpec, pretrained: bool = True) -> None:
        super().__init__()
        self.spec = spec
        self.backbone = timm.create_model(spec.timm_name, pretrained=pretrained, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, 1)
        cfg = self.backbone.pretrained_cfg
        self.mean: tuple[float, float, float] = tuple(float(v) for v in cfg["mean"])  # type: ignore[assignment]
        self.std: tuple[float, float, float] = tuple(float(v) for v in cfg["std"])  # type: ignore[assignment]
        if spec.frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def pooled(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_head(features, pre_logits=True)

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(x)
        return self.head(self.pooled(feats)).squeeze(1), feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(x)[0]

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict[str, Any]]:
        """AdamW groups: no weight decay on biases/norm weights (1-D params) — decaying them hurts."""
        groups: list[dict[str, Any]] = []
        for module, lr in ((self.backbone, lr_backbone), (self.head, lr_head)):
            decay, no_decay = [], []
            for p in module.parameters():
                if p.requires_grad:
                    (no_decay if p.ndim <= 1 else decay).append(p)
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
        if not groups:
            raise ValueError("Model has no trainable parameters")
        return groups

    def train(self, mode: bool = True) -> BinaryClassifier:
        super().train(mode)
        if self.spec.frozen_backbone:
            self.backbone.eval()  # frozen backbone: keep dropout/norm layers in inference mode
        return self


def build_model(name: str, pretrained: bool = True) -> BinaryClassifier:
    return BinaryClassifier(get_spec(name), pretrained=pretrained)


def save_checkpoint(model: BinaryClassifier, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"spec": asdict(model.spec), "state_dict": model.state_dict(), "extra": extra or {}},
        out,
    )
    return out


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[BinaryClassifier, dict[str, Any]]:
    """Rebuild the architecture (no weight download) and load trained weights."""
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    if "spec" not in ckpt or "state_dict" not in ckpt:
        raise ValueError(f"{path} is not a DeepTrace checkpoint")
    model = BinaryClassifier(get_spec(ckpt["spec"]["name"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("extra", {})
