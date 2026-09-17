"""Package a trained model for serving: ONNX + model_card.json + preprocess.json + checksums.

The export FAILS (non-zero exit) if ONNX Runtime disagrees with PyTorch, or if the closed-form numpy
heatmap disagrees with autograd Grad-CAM — a silent mismatch would ship wrong heatmaps.

    python scripts/export_model.py --checkpoint runs/effnet_b0_seed42/best.pt \
        --calibration reports/calibration_effnet_b0.json --eval-report reports/eval_effnet_b0.json \
        --robustness-report reports/robustness_effnet_b0.json --version 1.0.0 --out-dir artifacts/effnet_b0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deeptrace_ml.config import export_preprocess_json, load_preprocess_config
from deeptrace_ml.explain.closed_form import attention_rollout, gap_gradcam_pre_relu, normalize_map
from deeptrace_ml.export.model_card import build_model_card
from deeptrace_ml.export.onnx_export import export_onnx, output_names, verify_onnx
from deeptrace_ml.models.classifier import BinaryClassifier, load_checkpoint
from deeptrace_ml.utils.hashing import sha256_file
from deeptrace_ml.utils.jsonio import write_json
from deeptrace_ml.utils.logging import get_logger

log = get_logger("export_model")
ML_DIR = Path(__file__).resolve().parents[1]
ONNX_TOLERANCE = 1e-3
CAM_TOLERANCE = 1e-3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--eval-report", type=Path, default=None)
    p.add_argument("--robustness-report", type=Path, default=None)
    p.add_argument("--preprocess-config", type=Path, default=ML_DIR / "configs/preprocess.yaml")
    p.add_argument("--version", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def summarize_metrics(eval_report: dict[str, Any] | None, robustness: dict[str, Any] | None) -> dict[str, Any]:
    """Copy (never recompute or retype) the headline numbers into the card."""
    if eval_report is None:
        return {"status": "not_evaluated"}
    sets = {}
    for name, m in eval_report["test_sets"].items():
        at_t = m["at_tuned_threshold"]
        sets[name] = {
            "description": m["description"],
            "n": m["n"],
            "roc_auc": m["roc_auc"],
            "roc_auc_ci95": m["ci95"]["roc_auc"],
            "balanced_accuracy": at_t["balanced_accuracy"],
            "precision": at_t["precision"],
            "recall": at_t["recall_tpr"],
            "f1": at_t["f1"],
            "fpr": at_t["fpr"],
            "ece": m["ece_15"],
        }
    out: dict[str, Any] = {"status": "evaluated", "test_sets": sets}
    if robustness is not None:
        out["robustness"] = robustness["results"]
    return out


def cam_parity(
    model: BinaryClassifier, card: dict[str, Any], session_outputs: list[np.ndarray], x: torch.Tensor
) -> float:
    """Relative max difference between the numpy map the backend will compute and an autograd reference.

    CNN maps are compared BEFORE the ReLU: an untrained or unlucky model can produce an all-zero map
    after ReLU, which would make the check pass without testing anything.
    """
    if model.spec.family == "cnn":
        feats = model.features(x).detach().requires_grad_(True)
        model.head(model.pooled(feats)).sum().backward()
        weights = feats.grad.mean(dim=(2, 3), keepdim=True)
        height, width = feats.shape[2], feats.shape[3]
        reference = (height * width * (weights * feats).sum(dim=1))[0].detach().double().numpy()
        ours = gap_gradcam_pre_relu(session_outputs[1][0], card["explanation"]["head"])
        return float(np.max(np.abs(ours - reference)) / (np.max(np.abs(reference)) + 1e-12))
    from deeptrace_ml.explain.torch_cam import rollout_maps

    reference = rollout_maps(model, x)[0]
    ours = attention_rollout(session_outputs[1][0], card["explanation"]["num_prefix_tokens"])
    return float(np.max(np.abs(normalize_map(ours) - normalize_map(reference))))


def main() -> None:
    import onnxruntime as ort

    args = parse_args()
    model, extra = load_checkpoint(args.checkpoint)
    pre_cfg = load_preprocess_config(args.preprocess_config)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    eval_report = json.loads(args.eval_report.read_text(encoding="utf-8")) if args.eval_report else None
    robustness = json.loads(args.robustness_report.read_text(encoding="utf-8")) if args.robustness_report else None
    if eval_report is None:
        log.warning("No --eval-report: the model card will say 'not_evaluated'.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_onnx(model, args.out_dir / "model.onnx", model.spec.input_size)
    diffs = verify_onnx(model, onnx_path, input_size=model.spec.input_size)
    log.info("ONNX vs PyTorch max abs diff: %s", diffs)
    if max(diffs.values()) > ONNX_TOLERANCE:
        raise SystemExit(f"ONNX output mismatch {diffs} exceeds {ONNX_TOLERANCE}")

    card = build_model_card(
        model,
        args.version,
        calibration,
        summarize_metrics(eval_report, robustness),
        pre_cfg.version,
        output_names(model),
    )
    card["training"] = {"checkpoint_epoch": extra.get("epoch"), "val_select_auc": extra.get("val_auc")}

    torch.manual_seed(0)
    x = torch.randn(1, 3, model.spec.input_size, model.spec.input_size)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    cam_diff = cam_parity(model, card, session.run(None, {"image": x.numpy()}), x)
    log.info("Heatmap parity (numpy from ONNX outputs vs PyTorch reference): %.2e", cam_diff)
    if cam_diff > CAM_TOLERANCE:
        raise SystemExit(f"Heatmap parity {cam_diff} exceeds {CAM_TOLERANCE}")

    write_json(card, args.out_dir / "model_card.json")
    export_preprocess_json(pre_cfg, args.out_dir / "preprocess.json")
    files = ["model.onnx", "model_card.json", "preprocess.json"]
    write_json(
        {
            "files": {f: sha256_file(args.out_dir / f) for f in files},
            "verification": {"onnx_max_abs_diff": diffs, "heatmap_parity_max_abs_diff": cam_diff},
        },
        args.out_dir / "bundle_manifest.json",
    )
    log.info("Bundle ready in %s", args.out_dir)


if __name__ == "__main__":
    main()
