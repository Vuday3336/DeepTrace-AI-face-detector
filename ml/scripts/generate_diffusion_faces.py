"""Generate AI faces with free, open diffusion models for the cross-generator TEST set.

Run on a Kaggle GPU notebook (T4 or P100, 16 GB). Resumable: rerun the same command after a timeout.

    pip install -r requirements/generation.txt
    python scripts/generate_diffusion_faces.py --model sdxl --count 700 --out /kaggle/working/generated
    python scripts/generate_diffusion_faces.py --model flux-schnell --count 700 --out /kaggle/working/generated

Output: <out>/<model>/<model>_<index>.png + <out>/<model>/metadata.jsonl (prompt, seed, settings).
Images are never used for training — only for measuring generalisation.

Notes
  * FLUX.1-schnell weights are ~34 GB to download; the transformer is loaded in 4-bit and the T5 text
    encoder in 8-bit so it fits a 16 GB GPU, with CPU offload. Expect it to be much slower per image
    than SDXL on a T4. If you see black images, rerun with --flux-dtype bfloat16.
  * Degenerate outputs (black / flat images from fp16 overflow) are detected and NOT saved; they are
    logged in metadata.jsonl with status "degenerate" so the failure rate is visible.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from deeptrace_ml.utils.env import write_environment
from deeptrace_ml.utils.logging import get_logger

log = get_logger("generate_diffusion_faces")
ML_DIR = Path(__file__).resolve().parents[1]
MIN_FREE_DISK_GB = {"sdxl": 15, "flux-schnell": 45}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["sdxl", "flux-schnell"], required=True)
    p.add_argument("--count", type=int, required=True, help="Target number of saved images")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prompts", type=Path, default=ML_DIR / "configs/diffusion_prompts.yaml")
    p.add_argument("--flux-dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument(
        "--max-attempts-factor",
        type=float,
        default=1.3,
        help="Stop after count * factor attempts even if some outputs were degenerate",
    )
    return p.parse_args()


def load_prompt_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key in ("template", "attributes", "models"):
        if key not in cfg:
            raise ValueError(f"{path} is missing '{key}'")
    return cfg


def build_prompt(cfg: dict[str, Any], seed: int) -> tuple[str, dict[str, str]]:
    rng = random.Random(seed)  # attributes derive from the image seed -> fully reproducible
    chosen = {name: rng.choice(options) for name, options in cfg["attributes"].items()}
    return cfg["template"].format(**chosen), chosen


def is_degenerate(img: Image.Image) -> bool:
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    return float(arr.std()) < 3.0 or not np.isfinite(arr).all()


def check_disk(path: Path, model: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / 1e9
    cache_free_gb = shutil.disk_usage(Path.home()).free / 1e9
    if min(free_gb, cache_free_gb) < MIN_FREE_DISK_GB[model]:
        log.warning(
            "Low disk: %.1f GB free at output, %.1f GB at HF cache; %s needs ~%d GB.",
            free_gb,
            cache_free_gb,
            model,
            MIN_FREE_DISK_GB[model],
        )


def load_pipeline(model: str, mcfg: dict[str, Any], flux_dtype: str) -> Any:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. On Kaggle: Settings -> Accelerator -> GPU.")

    if model == "sdxl":
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline

        vae = AutoencoderKL.from_pretrained(mcfg["vae_repo_id"], torch_dtype=torch.float16)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            mcfg["repo_id"], vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        return pipe

    try:
        import bitsandbytes  # noqa: F401  (4-bit loading backend for FLUX only)
    except Exception as exc:  # noqa: BLE001 — a broken install raises more than ImportError
        raise RuntimeError(
            "FLUX needs a working bitsandbytes: pip install -r requirements/generation-flux.txt. "
            f"Import failed with {type(exc).__name__}: {exc}"
        ) from exc

    from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
    from diffusers import FluxPipeline, FluxTransformer2DModel
    from transformers import BitsAndBytesConfig as TransformersBnbConfig
    from transformers import T5EncoderModel

    dtype = getattr(torch, flux_dtype)
    transformer = FluxTransformer2DModel.from_pretrained(
        mcfg["repo_id"],
        subfolder="transformer",
        torch_dtype=dtype,
        quantization_config=DiffusersBnbConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
        ),
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        mcfg["repo_id"],
        subfolder="text_encoder_2",
        torch_dtype=dtype,
        quantization_config=TransformersBnbConfig(load_in_8bit=True),
    )
    pipe = FluxPipeline.from_pretrained(
        mcfg["repo_id"], transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=dtype
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_one(pipe: Any, model: str, mcfg: dict[str, Any], prompt: str, negative: str, seed: int) -> Image.Image:
    import torch

    common = dict(
        prompt=prompt,
        width=mcfg["width"],
        height=mcfg["height"],
        num_inference_steps=mcfg["steps"],
        guidance_scale=mcfg["guidance_scale"],
    )
    if model == "sdxl":
        generator = torch.Generator("cuda").manual_seed(seed)
        return pipe(negative_prompt=negative, generator=generator, **common).images[0]
    generator = torch.Generator("cpu").manual_seed(seed)  # CPU generator is required with offload
    return pipe(generator=generator, max_sequence_length=mcfg["max_sequence_length"], **common).images[0]


def read_metadata(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    records[int(rec["index"])] = rec
    return records


def main() -> None:
    args = parse_args()
    cfg = load_prompt_config(args.prompts)
    mcfg = cfg["models"][args.model]
    out_dir = args.out / args.model
    check_disk(out_dir, args.model)
    meta_path = out_dir / "metadata.jsonl"
    records = read_metadata(meta_path)

    def saved_count() -> int:
        return sum(1 for r in records.values() if r["status"] == "ok" and (out_dir / r["file"]).exists())

    if saved_count() >= args.count:
        log.info("Already have %d images in %s — nothing to do.", saved_count(), out_dir)
        return

    write_environment(out_dir / "env.json", repo_dir=ML_DIR)
    log.info("Loading %s (first run downloads weights)...", mcfg["repo_id"])
    pipe = load_pipeline(args.model, mcfg, args.flux_dtype)
    negative = cfg.get("negative_prompt", "")
    max_attempts = int(args.count * args.max_attempts_factor)

    with meta_path.open("a", encoding="utf-8") as meta:
        index = 0
        while saved_count() < args.count and index < max_attempts:
            prev = records.get(index)
            if prev is not None and (prev["status"] != "ok" or (out_dir / prev["file"]).exists()):
                index += 1
                continue
            seed = int(mcfg["base_seed"]) + index
            prompt, attributes = build_prompt(cfg, seed)
            start = time.perf_counter()
            image = generate_one(pipe, args.model, mcfg, prompt, negative, seed)
            file_name = f"{args.model}_{index:05d}.png"
            status = "degenerate" if is_degenerate(image) else "ok"
            if status == "ok":
                image.save(out_dir / file_name, format="PNG")
            rec = {
                "index": index,
                "file": file_name,
                "status": status,
                "model": args.model,
                "repo_id": mcfg["repo_id"],
                "seed": seed,
                "prompt": prompt,
                "negative_prompt": negative if args.model == "sdxl" else None,
                "attributes": attributes,
                "width": mcfg["width"],
                "height": mcfg["height"],
                "steps": mcfg["steps"],
                "guidance_scale": mcfg["guidance_scale"],
                "seconds": round(time.perf_counter() - start, 2),
            }
            meta.write(json.dumps(rec) + "\n")
            meta.flush()
            records[index] = rec
            if status != "ok":
                log.warning("Index %d produced a degenerate image (not saved).", index)
            if index % 25 == 0:
                log.info("%s: %d/%d saved (last %.1fs/image)", args.model, saved_count(), args.count, rec["seconds"])
            index += 1

    degenerate = sum(1 for r in records.values() if r["status"] == "degenerate")
    log.info("Done: %d saved, %d degenerate, in %s", saved_count(), degenerate, out_dir)
    if saved_count() < args.count:
        log.warning(
            "Stopped at max attempts (%d). Degenerate outputs are common with fp16 FLUX; " "try --flux-dtype bfloat16.",
            max_attempts,
        )


if __name__ == "__main__":
    main()
