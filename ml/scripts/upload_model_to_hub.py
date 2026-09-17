"""Upload an exported bundle to a (free) Hugging Face Hub model repo. The backend downloads it at startup.

huggingface-cli login            # or set HF_TOKEN
python scripts/upload_model_to_hub.py --bundle-dir artifacts/effnet_b0 --repo-id <user>/deeptrace-model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deeptrace_ml.serving.runtime import ModelBundle
from deeptrace_ml.utils.logging import get_logger

log = get_logger("upload_model_to_hub")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--private", action="store_true")
    return p.parse_args()


def main() -> None:
    from huggingface_hub import HfApi

    args = parse_args()
    bundle = ModelBundle.load(args.bundle_dir)  # refuse to upload a bundle that fails its own checks
    card = json.loads((args.bundle_dir / "model_card.json").read_text(encoding="utf-8"))
    readme = args.bundle_dir / "README.md"
    readme.write_text(
        f"# DeepTrace model — {bundle.name} v{bundle.version}\n\n{card['disclaimer']}\n\n"
        "Trained on the 140k Real and Fake Faces dataset (FFHQ is non-commercial). "
        "See model_card.json for metrics and known limitations.\n",
        encoding="utf-8",
    )
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(args.bundle_dir),
        repo_type="model",
        commit_message=f"{bundle.name} v{bundle.version}",
    )
    log.info(
        "Uploaded to https://huggingface.co/%s (%s). Set MODEL_REPO_ID=%s and MODEL_REVISION=%s in the backend.",
        args.repo_id,
        commit.oid if hasattr(commit, "oid") else commit,
        args.repo_id,
        getattr(commit, "oid", "main"),
    )


if __name__ == "__main__":
    main()
