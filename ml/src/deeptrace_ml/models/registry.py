"""The models we compare, and everything the rest of the pipeline needs to know about each one.

Why these weights:
- efficientnet_b0.ra_in1k: small (5.3M params), fast on CPU — the baseline.
- convnext_tiny.fb_in22k_ft_in1k: ImageNet-22k pretraining transfers better than 1k-only weights.
- vit_small_patch16_224.augreg_in21k_ft_in1k: the standard "AugReg" ViT-S recipe, also 21k-pretrained.
- vit_base_patch16_clip_224.openai: OpenAI CLIP image tower, trained on 400M image-text pairs. Kept
  FROZEN (linear probe) because full fine-tuning tends to overwrite the general features that make
  CLIP transfer to unseen generators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Family = Literal["cnn", "vit"]


@dataclass(frozen=True)
class ModelSpec:
    name: str  # our short name (CLI, reports, file names)
    timm_name: str
    family: Family
    frozen_backbone: bool  # True = linear probe
    input_size: int = 224

    @property
    def explanation_method(self) -> str:
        # What the SERVING path can compute without autograd (docs/ARCHITECTURE.md §7.3)
        return "gradcam" if self.family == "cnn" else "attention_rollout"


MODEL_SPECS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec("effnet_b0", "efficientnet_b0.ra_in1k", "cnn", frozen_backbone=False),
        ModelSpec("convnext_tiny", "convnext_tiny.fb_in22k_ft_in1k", "cnn", frozen_backbone=False),
        ModelSpec("vit_small", "vit_small_patch16_224.augreg_in21k_ft_in1k", "vit", frozen_backbone=False),
        ModelSpec("clip_probe", "vit_base_patch16_clip_224.openai", "vit", frozen_backbone=True),
    )
}


def get_spec(name: str) -> ModelSpec:
    if name not in MODEL_SPECS:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_SPECS)}")
    return MODEL_SPECS[name]
