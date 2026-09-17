"""model_card.json: everything the backend and the Model Info page need, in one file.

Metrics are COPIED from evaluation report JSONs by the export script — never typed by hand.
"""

from __future__ import annotations

from typing import Any

from deeptrace_ml.models.classifier import BinaryClassifier

MODEL_CARD_VERSION = 1

TRAINING_DATA = {
    "name": "140k Real and Fake Faces (Kaggle)",
    "real_source": "FFHQ (Flickr-Faces-HQ)",
    "fake_source": "StyleGAN",
    "license_note": "FFHQ: CC BY-NC-SA 4.0 dataset compilation — non-commercial use only.",
    "preprocessing": "MTCNN face crop (30% margin), 224x224, JPEG re-encode q80-95, all metadata stripped",
}

KNOWN_LIMITATIONS = [
    "Trained only on StyleGAN fakes: expect weaker detection of diffusion-model faces (see cross-generator metrics).",
    "Heavy JPEG compression, downscaling, blur and screenshots reduce accuracy (see robustness curves).",
    "Probabilities are calibrated on StyleGAN-era validation data and are less reliable on other generators.",
    "Not designed for face swaps, partial edits/inpainting, or non-photographic images.",
    "Performance across age, skin tone and camera types has not been audited with demographic labels.",
    "Can be evaded by deliberate adversarial manipulation.",
    "A detection aid, not proof: never use a result as the sole basis for an accusation.",
]


def cam_head_params(model: BinaryClassifier) -> dict[str, Any]:
    """Head parameters for closed-form Grad-CAM (see explain/closed_form.py)."""
    if model.spec.family != "cnn":
        raise ValueError("Closed-form Grad-CAM only applies to CNN (GAP-head) models")
    head: dict[str, Any] = {
        "weight": model.head.weight.detach().cpu().double().reshape(-1).tolist(),
        "bias": float(model.head.bias.detach().cpu().item()),
        "norm": None,
    }
    backbone_head = getattr(model.backbone, "head", None)
    norm = getattr(backbone_head, "norm", None)
    if norm is not None and hasattr(norm, "weight"):
        head["norm"] = {
            "weight": norm.weight.detach().cpu().double().tolist(),
            "bias": norm.bias.detach().cpu().double().tolist(),
            "eps": float(norm.eps),
        }
    return head


def build_model_card(
    model: BinaryClassifier,
    version: str,
    calibration: dict[str, Any],
    metrics: dict[str, Any],
    preprocess_version: str,
    onnx_outputs: list[str],
) -> dict[str, Any]:
    card: dict[str, Any] = {
        "card_version": MODEL_CARD_VERSION,
        "model": {
            "name": model.spec.name,
            "timm_name": model.spec.timm_name,
            "family": model.spec.family,
            "frozen_backbone": model.spec.frozen_backbone,
            "version": version,
            "input_size": model.spec.input_size,
            "mean": list(model.mean),
            "std": list(model.std),
        },
        "onnx": {"file": "model.onnx", "input": "image", "outputs": onnx_outputs},
        "decision": {
            "temperature": calibration["temperature"],
            "threshold": calibration["threshold"],
            "uncertainty_delta": calibration["uncertainty_delta"],
            "target_fpr": calibration["target_fpr"],
            "positive_label": "AI_GENERATED",
        },
        "explanation": {"method": model.spec.explanation_method},
        "preprocess_version": preprocess_version,
        "training_data": TRAINING_DATA,
        "metrics": metrics,
        "known_limitations": KNOWN_LIMITATIONS,
        "disclaimer": "DeepTrace is a detection aid, not proof. Do not use it as the sole basis for any accusation.",
    }
    if model.spec.family == "cnn":
        card["explanation"]["head"] = cam_head_params(model)
    else:
        card["explanation"]["num_prefix_tokens"] = int(model.backbone.num_prefix_tokens)
    return card
