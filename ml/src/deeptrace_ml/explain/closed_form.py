"""Heatmaps from forward-pass outputs only (numpy) — used by the ONNX serving path.

1) Grad-CAM for "global-average-pool -> (optional LayerNorm) -> linear" heads (EfficientNet, ConvNeXt)
   ----------------------------------------------------------------------------------------------
   pooled_k = mean_{i,j} A_k[i,j]      so   d logit / d A_k[i,j] = g_k / (H*W),  g = d logit / d pooled
   The gradient is the SAME at every position, so Grad-CAM's channel weight alpha_k = mean(grad) = g_k/(HW)
   and the map is ReLU(sum_k g_k A_k) up to a positive constant (removed by normalisation anyway).
   - Linear head:            g = w
   - LayerNorm then linear:  y = gamma * xhat + beta, xhat = (x - mu) / s  =>
                             g = (1/s) * (v - mean(v) - xhat * mean(v * xhat)),  v = gamma * w
   Exactness is tested against torch autograd in tests/test_explain.py.

2) Attention rollout for ViTs (Abnar & Zuidema, 2020)
   ---------------------------------------------------
   Average attention over heads, add the identity for the residual path, renormalise rows, multiply
   through the layers; the CLS row says how much each patch flows into the classification token.
   It is CLASS-AGNOSTIC: it shows where the model looks, not what pushed it toward "fake".

Everything here depends on numpy only, so the backend can import it without torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def layernorm_input_gradient(x: np.ndarray, upstream: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    """d(upstream . LayerNorm(x)) / dx for a single vector x (shape [C])."""
    mu = x.mean()
    s = np.sqrt(((x - mu) ** 2).mean() + eps)
    xhat = (x - mu) / s
    v = upstream * gamma
    return (v - v.mean() - xhat * (v * xhat).mean()) / s


def pooled_gradient(features: np.ndarray, head: Mapping[str, Any]) -> np.ndarray:
    """g = d logit / d pooled for one image. features: [C, H, W]."""
    weight = np.asarray(head["weight"], dtype=np.float64).reshape(-1)
    if head.get("norm") is None:
        return weight
    norm = head["norm"]
    pooled = features.reshape(features.shape[0], -1).mean(axis=1).astype(np.float64)
    return layernorm_input_gradient(pooled, weight, np.asarray(norm["weight"], dtype=np.float64), float(norm["eps"]))


def gap_gradcam_pre_relu(features: np.ndarray, head: Mapping[str, Any], target: str = "fake") -> np.ndarray:
    """sum_k g_k * A_k BEFORE the ReLU, [H, W]. Equals H*W times autograd Grad-CAM's pre-ReLU map.

    target="fake" explains the AI-generated evidence; "real" flips the gradient sign.
    """
    if features.ndim != 3:
        raise ValueError(f"features must be [C, H, W], got {features.shape}")
    g = pooled_gradient(features, head)
    if target == "real":
        g = -g
    elif target != "fake":
        raise ValueError("target must be 'fake' or 'real'")
    return np.tensordot(g, features.astype(np.float64), axes=(0, 0))


def gap_gradcam(features: np.ndarray, head: Mapping[str, Any], target: str = "fake") -> np.ndarray:
    """Raw (un-normalised) Grad-CAM map [H, W]: ReLU keeps only evidence FOR the target class."""
    return np.maximum(gap_gradcam_pre_relu(features, head, target), 0.0)


def attention_rollout(attentions: np.ndarray, num_prefix_tokens: int = 1, residual_weight: float = 0.5) -> np.ndarray:
    """attentions: [L, T, T] head-averaged, one image. Returns a [g, g] map over patches."""
    if attentions.ndim != 3 or attentions.shape[1] != attentions.shape[2]:
        raise ValueError(f"attentions must be [L, T, T], got {attentions.shape}")
    tokens = attentions.shape[-1]
    rollout = np.eye(tokens)
    eye = np.eye(tokens)
    for layer in attentions.astype(np.float64):
        mixed = residual_weight * eye + (1.0 - residual_weight) * layer
        mixed /= mixed.sum(axis=-1, keepdims=True)
        rollout = mixed @ rollout
    patch_scores = rollout[0, num_prefix_tokens:]
    side = int(round(np.sqrt(len(patch_scores))))
    if side * side != len(patch_scores):
        raise ValueError(f"{len(patch_scores)} patch tokens do not form a square grid")
    return patch_scores.reshape(side, side)


def normalize_map(cam: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1]; a constant map becomes all zeros (nothing stands out)."""
    lo, hi = float(cam.min()), float(cam.max())
    if hi - lo < 1e-12:
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - lo) / (hi - lo)).astype(np.float32)


def turbo_colormap(values: np.ndarray) -> np.ndarray:
    """[0,1] -> uint8 RGB using the polynomial approximation of Google's Turbo colormap."""
    x = np.clip(values, 0.0, 1.0)[..., None]
    coeffs = np.array(
        [
            [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
            [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
            [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
        ]
    )
    powers = np.concatenate([x**i for i in range(6)], axis=-1)  # [..., 6]
    rgb = powers @ coeffs.T
    return (np.clip(rgb, 0.0, 1.0) * 255).round().astype(np.uint8)


def heatmap_rgba(cam: np.ndarray, size: int, min_alpha: float = 0.0) -> np.ndarray:
    """Normalised map -> RGBA uint8 [size, size, 4]; alpha follows intensity so cold areas stay see-through."""
    from PIL import Image

    norm = normalize_map(cam)
    up = (
        np.asarray(
            Image.fromarray((norm * 255).astype(np.uint8)).resize((size, size), Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
        / 255.0
    )
    rgb = turbo_colormap(up)
    alpha = (np.clip(min_alpha + (1 - min_alpha) * up, 0, 1) * 255).astype(np.uint8)
    return np.dstack([rgb, alpha])
