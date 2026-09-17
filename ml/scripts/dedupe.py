"""Find near-duplicate images across splits/datasets and write the list of images to drop.

    python scripts/dedupe.py \
        --manifest data/manifests/140k_raw.csv.gz --root 140k=/kaggle/input/.../real-vs-fake \
        --manifest data/manifests/crossgen_raw.csv.gz --root crossgen=data/raw/crossgen \
        --workers 4

Outputs:
    data/manifests/duplicates_drop.csv   images to exclude (consumed by prepare_dataset.py)
    reports/dedupe_pairs.csv             every near-duplicate pair found (incl. same-split)
    reports/dedupe_report.json           counts
    docs/plots/audit/near_duplicates.png the most borderline cross-split pairs, for eyeballing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from deeptrace_ml.audit.plots import plot_image_pairs
from deeptrace_ml.config import load_data_config
from deeptrace_ml.data.dedupe import compute_hashes, find_near_duplicates, resolve_drops
from deeptrace_ml.data.manifest import load_manifest, parse_roots, resolve_path
from deeptrace_ml.utils.jsonio import write_json
from deeptrace_ml.utils.logging import get_logger
from deeptrace_ml.utils.paths import plots_dir, reports_dir

log = get_logger("dedupe")
ML_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ML_DIR.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, action="append", required=True)
    p.add_argument("--root", action="append", required=True, help="DATASET=PATH, one per dataset")
    p.add_argument("--data-config", type=Path, default=ML_DIR / "configs/data.yaml")
    p.add_argument("--drops-out", type=Path, default=ML_DIR / "data/manifests/duplicates_drop.csv")
    p.add_argument("--pairs-out", type=Path, default=reports_dir() / "dedupe_pairs.csv")
    p.add_argument("--report", type=Path, default=reports_dir() / "dedupe_report.json")
    p.add_argument("--plot", type=Path, default=plots_dir() / "audit/near_duplicates.png")
    p.add_argument("--max-plot-pairs", type=int, default=12)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_data_config(args.data_config)
    roots = parse_roots(args.root)

    manifest = pd.concat([load_manifest(m) for m in args.manifest], ignore_index=True)
    unreadable = manifest["read_error"].notna()
    manifest = manifest[~unreadable].reset_index(drop=True)
    log.info("%d readable images (%d unreadable skipped)", len(manifest), int(unreadable.sum()))

    paths = [resolve_path(roots, d, r) for d, r in zip(manifest["dataset"], manifest["rel_path"], strict=False)]
    hashes, ok = compute_hashes(paths, workers=args.workers)
    if not ok.all():
        log.warning("%d images failed to hash and are excluded from dedupe", int((~ok).sum()))
    hashed = manifest[ok].reset_index(drop=True)
    hashes = hashes[ok]

    pairs = find_near_duplicates(hashes, cfg.dedupe.max_hamming_distance)
    annotated, drops = resolve_drops(hashed, pairs, cfg.dedupe.split_priority)

    args.drops_out.parent.mkdir(parents=True, exist_ok=True)
    args.pairs_out.parent.mkdir(parents=True, exist_ok=True)
    drops.to_csv(args.drops_out, index=False)
    annotated.to_csv(args.pairs_out, index=False)

    cross = annotated[annotated["cross_split"]] if len(annotated) else annotated
    report = {
        "max_hamming_distance": cfg.dedupe.max_hamming_distance,
        "images_hashed": int(len(hashed)),
        "hash_failures": int((~ok).sum()),
        "pairs_total": int(len(annotated)),
        "pairs_same_split": int(len(annotated) - len(cross)),
        "pairs_cross_split": int(len(cross)),
        "pairs_with_label_conflict": int(annotated["label_conflict"].sum()) if len(annotated) else 0,
        "dropped_images": int(len(drops)),
        "dropped_by_dataset_split": (
            drops.groupby(["dataset", "split"]).size().rename("n").reset_index().to_dict("records")
            if len(drops)
            else []
        ),
    }
    write_json(report, args.report)
    log.info("Dedupe report: %s", report)

    if len(cross):
        # Largest distances first: these are the pairs most likely to be FALSE positives,
        # so looking at them tells you whether the threshold is too loose.
        sample = cross.sort_values(["phash_dist", "dhash_dist"], ascending=False).head(args.max_plot_pairs)
        plot_pairs = [
            (
                resolve_path(roots, r["dataset_i"], r["rel_path_i"]),
                resolve_path(roots, r["dataset_j"], r["rel_path_j"]),
                f"{r['split_i']}/{r['label_i']} vs {r['split_j']}/{r['label_j']}  "
                f"p={r['phash_dist']} d={r['dhash_dist']}",
            )
            for _, r in sample.iterrows()
        ]
        plot_image_pairs(plot_pairs, args.plot, "Most borderline cross-split near-duplicates")
        log.info("Saved %s — check these by eye; if they're clearly different faces, lower the threshold.", args.plot)


if __name__ == "__main__":
    main()
