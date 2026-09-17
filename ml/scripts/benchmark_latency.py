"""CPU latency of the serving path (ONNX Runtime + heatmap), optionally vs PyTorch CPU.

Measures a single-face request after detection: crop -> JPEG -> normalise -> ONNX -> heatmap PNG.
Detection is benchmarked separately (--with-detection) because it depends on the upload's size.

    python scripts/benchmark_latency.py --bundle-dir artifacts/effnet_b0 --threads 2 \
        --checkpoint runs/effnet_b0_seed42/best.pt
"""

from __future__ import annotations

import argparse
import io
import platform
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from deeptrace_ml.preprocessing.pipeline import normalize_image, to_inference_image
from deeptrace_ml.serving.runtime import ModelBundle
from deeptrace_ml.utils.jsonio import write_json
from deeptrace_ml.utils.logging import get_logger
from deeptrace_ml.utils.paths import reports_dir

log = get_logger("benchmark_latency")
ML_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None, help="Also time PyTorch CPU for comparison")
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--threads", type=int, default=2, help="Match the deployment CPU count (HF Spaces free: 2 vCPU)")
    p.add_argument("--with-detection", action="store_true", help="Also time MTCNN on a 1024px image")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def timeit(fn: Callable[[], object], runs: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    arr = np.asarray(samples)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "runs": runs,
    }


def main() -> None:
    args = parse_args()
    bundle = ModelBundle.load(args.bundle_dir, intra_op_threads=args.threads)
    rng = np.random.default_rng(0)
    face = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
    cfg = bundle.preprocess

    def model_only() -> None:
        bundle.infer([to_inference_image(face, cfg)])

    def model_and_heatmap() -> None:
        _, aux = bundle.infer([to_inference_image(face, cfg)])
        rgba = bundle.heatmap(aux[0], cfg.crop.output_size)
        Image.fromarray(rgba).save(io.BytesIO(), format="PNG")

    report: dict[str, object] = {
        "model": bundle.name,
        "threads": args.threads,
        "cpu": platform.processor() or platform.machine(),
        "onnx_model_only": timeit(model_only, args.runs, args.warmup),
        "onnx_model_and_heatmap": timeit(model_and_heatmap, args.runs, args.warmup),
    }

    if args.checkpoint:
        import torch

        from deeptrace_ml.models.classifier import load_checkpoint

        torch.set_num_threads(args.threads)
        model, _ = load_checkpoint(args.checkpoint)
        x = torch.from_numpy(normalize_image(face, model.mean, model.std))

        def torch_forward() -> None:
            with torch.no_grad():
                model(x)

        report["torch_cpu_model_only"] = timeit(torch_forward, args.runs, args.warmup)

    if args.with_detection:
        import torch

        from deeptrace_ml.preprocessing.face import MTCNNFaceDetector

        torch.set_num_threads(args.threads)
        detector = MTCNNFaceDetector(cfg.face, device="cpu")
        big = Image.fromarray(rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8))
        report["mtcnn_1024px"] = timeit(lambda: detector.detect([big]), max(10, args.runs // 5), 3)

    out = args.out or reports_dir() / f"latency_{bundle.name}.json"
    write_json(report, out)
    log.info("Latency: %s", report)


if __name__ == "__main__":
    main()
