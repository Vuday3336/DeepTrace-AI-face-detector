from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from deeptrace_ml.explain.closed_form import (  # noqa: E402
    attention_rollout,
    gap_gradcam,
    gap_gradcam_pre_relu,
    heatmap_rgba,
    layernorm_input_gradient,
    normalize_map,
    turbo_colormap,
)
from deeptrace_ml.export.model_card import cam_head_params  # noqa: E402
from deeptrace_ml.models.classifier import build_model  # noqa: E402


def _autograd_gradcam(model, x: torch.Tensor, target: str) -> np.ndarray:
    """Reference Grad-CAM straight from autograd: ReLU(sum_k mean(dlogit/dA_k) * A_k)."""
    feats = model.features(x).detach().requires_grad_(True)
    logit = model.head(model.pooled(feats)).sum()
    (logit if target == "fake" else -logit).backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    return torch.relu((weights * feats).sum(dim=1))[0].detach().numpy()


@pytest.mark.parametrize("name", ["effnet_b0", "convnext_tiny"])
@pytest.mark.parametrize("target", ["fake", "real"])
def test_closed_form_gradcam_matches_autograd(name, target):
    torch.manual_seed(0)
    model = build_model(name, pretrained=False).eval()
    with torch.no_grad():  # non-trivial LayerNorm affine params so the test is meaningful
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x = torch.randn(1, 3, 224, 224)
    reference = _autograd_gradcam(model, x, target)
    with torch.no_grad():
        feats = model.features(x)[0].numpy()
    ours = gap_gradcam(feats, cam_head_params(model), target)
    # compare shapes after normalisation (the positive scale 1/HW doesn't matter)
    np.testing.assert_allclose(normalize_map(ours), normalize_map(reference), atol=2e-4)


def test_layernorm_gradient_matches_torch():
    torch.manual_seed(1)
    x = torch.randn(16, dtype=torch.float64, requires_grad=True)
    gamma, beta, w = (
        torch.randn(16, dtype=torch.float64),
        torch.randn(16, dtype=torch.float64),
        torch.randn(16, dtype=torch.float64),
    )
    y = torch.nn.functional.layer_norm(x, (16,), gamma, beta, eps=1e-6)
    (w * y).sum().backward()
    ours = layernorm_input_gradient(x.detach().numpy(), w.numpy(), gamma.numpy(), 1e-6)
    np.testing.assert_allclose(ours, x.grad.numpy(), atol=1e-9)


def test_rollout_identity_attention_is_uniform():
    tokens = 1 + 16
    attn = np.tile(np.eye(tokens), (4, 1, 1))
    rollout = attention_rollout(attn, num_prefix_tokens=1)
    assert rollout.shape == (4, 4)
    assert np.allclose(rollout, 0.0)  # CLS attends only to itself -> no patch contributes


def test_rollout_detects_attended_patch():
    tokens = 1 + 16
    attn = np.full((3, tokens, tokens), 1.0 / tokens)
    attn[:, 0, :] = 0.0
    attn[:, 0, 6] = 1.0  # CLS attends to token 6 = patch index 5 -> grid (1, 1)
    rollout = attention_rollout(attn, 1)
    assert np.unravel_index(np.argmax(rollout), rollout.shape) == (1, 1)


def test_vit_attention_recorder_and_rollout():
    from deeptrace_ml.explain.torch_cam import rollout_maps

    model = build_model("vit_small", pretrained=False).eval()
    maps = rollout_maps(model, torch.randn(2, 3, 224, 224))
    assert maps.shape == (2, 14, 14)
    assert all(block.attn.fused_attn for block in model.backbone.blocks)  # restored afterwards


def test_pytorch_grad_cam_runs_for_each_family():
    from deeptrace_ml.explain.torch_cam import gradcam_maps

    for name in ("effnet_b0", "vit_small"):
        model = build_model(name, pretrained=False).eval()
        maps = gradcam_maps(model, torch.randn(1, 3, 224, 224), method="gradcam")
        assert maps.shape == (1, 224, 224)


def test_colormap_and_rgba():
    rgb = turbo_colormap(np.array([0.15, 0.85]))
    assert rgb.shape == (2, 3) and rgb.dtype == np.uint8
    assert rgb[0, 2] > rgb[0, 0]  # low values are blue
    assert rgb[1, 0] > rgb[1, 2]  # high values are red
    rgba = heatmap_rgba(np.random.default_rng(0).random((7, 7)), 224)
    assert rgba.shape == (224, 224, 4)


def test_pre_relu_map_matches_autograd_scale():
    """Non-vacuous check: compares signed maps, so an all-negative (blank after ReLU) map still counts."""
    torch.manual_seed(3)
    model = build_model("convnext_tiny", pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    feats = model.features(x).detach().requires_grad_(True)
    model.head(model.pooled(feats)).sum().backward()
    reference = (49 * (feats.grad.mean(dim=(2, 3), keepdim=True) * feats).sum(dim=1))[0].detach().double().numpy()
    ours = gap_gradcam_pre_relu(feats.detach()[0].numpy(), cam_head_params(model))
    assert np.max(np.abs(ours - reference)) / np.max(np.abs(reference)) < 1e-4
