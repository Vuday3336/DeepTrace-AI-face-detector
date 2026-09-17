"""Shortcut-learning audit for one dataset, before and (optionally) after preprocessing.

Raw only (run BEFORE prepare_dataset.py):
    python scripts/run_audit.py --dataset 140k \
        --manifest data/manifests/140k_raw.csv.gz --root 140k=/kaggle/input/.../real-vs-fake

Raw + processed (run AFTER):
    python scripts/run_audit.py --dataset 140k \
        --manifest data/manifests/140k_raw.csv.gz --root 140k=/kaggle/input/.../real-vs-fake \
        --processed-manifest data/manifests/140k_processed.csv.gz --processed-root /kaggle/working/processed

Outputs: reports/audit_<dataset>.json and docs/plots/audit/<dataset>/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from deeptrace_ml.audit.features import (
    METADATA_FEATURES,
    PIXEL_FEATURES,
    metadata_feature_frame,
    pixel_feature_frame,
)
from deeptrace_ml.audit.plots import (
    plot_numeric_distributions,
    plot_radial_spectra,
    plot_rates,
    plot_spectra_2d,
)
from deeptrace_ml.audit.spectrum import mean_spectrum
from deeptrace_ml.audit.trivial_classifier import run_trivial_classifier
from deeptrace_ml.data.image_meta import ImageReadError, read_image_meta
from deeptrace_ml.data.manifest import load_manifest, parse_roots, resolve_path
from deeptrace_ml.utils.env import capture_environment
from deeptrace_ml.utils.jsonio import write_json
from deeptrace_ml.utils.logging import get_logger
from deeptrace_ml.utils.paths import plots_dir, reports_dir

log = get_logger("run_audit")
ML_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ML_DIR.parent

NUMERIC_META = ["width", "height", "file_size_kb", "bytes_per_pixel", "jpeg_quality_est", "aspect"]
FLAG_META = ["is_jpeg", "is_png", "is_webp", "has_exif", "has_icc", "has_xmp"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--root", action="append", required=True, help="DATASET=PATH")
    p.add_argument("--processed-manifest", type=Path, default=None)
    p.add_argument("--processed-root", type=Path, default=None)
    p.add_argument("--splits", nargs="*", default=None, help="Restrict to these splits (default: all)")
    p.add_argument("--sample-per-group", type=int, default=2000, help="Images per source/label for pixel stats")
    p.add_argument("--spectrum-per-group", type=int, default=1000)
    p.add_argument("--spectrum-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--plots-dir", type=Path, default=None)
    return p.parse_args()


def add_group(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(group=df["source"].astype(str) + " (" + df["label"].astype(str) + ")")


def sample_groups(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    parts = [g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby("group")]
    return pd.concat(parts).sort_index()


def group_quantiles(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    q = df.groupby("group")[columns].quantile([0.05, 0.5, 0.95])
    return {
        group: {col: {f"p{int(k * 100)}": q.loc[(group, k), col] for k in (0.05, 0.5, 0.95)} for col in columns}
        for group in df["group"].unique()
    }


def spectra_by_group(
    df: pd.DataFrame, path_col: str, n: int, size: int, seed: int, workers: int
) -> tuple[dict, dict, dict]:
    profiles, maps, counts = {}, {}, {}
    for group, g in sample_groups(df, n, seed).groupby("group"):
        result = mean_spectrum(list(g[path_col]), size=size, workers=workers)
        profiles[group], maps[group] = result.radial_profile, result.mean_log_power_2d
        counts[group] = {"used": result.n_used, "skipped_too_small_or_unreadable": result.n_skipped}
    return profiles, maps, counts


def audit_raw(args: argparse.Namespace, raw: pd.DataFrame, plots: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    readable = raw[raw["read_error"].isna()].copy()
    report["counts"] = readable.groupby("group").size().to_dict()
    report["unreadable"] = int(raw["read_error"].notna().sum())

    meta = pd.concat([readable[["group", "label"]], metadata_feature_frame(readable)], axis=1)
    plot_numeric_distributions(
        meta, NUMERIC_META, "group", plots / "raw_metadata_numeric.png", f"{args.dataset} RAW — file-level metadata"
    )
    plot_rates(
        meta, FLAG_META, "group", plots / "raw_metadata_flags.png", f"{args.dataset} RAW — format & metadata presence"
    )
    report["metadata_quantiles"] = group_quantiles(meta, ["width", "height", "file_size_kb", "jpeg_quality_est"])
    report["metadata_flag_rates"] = meta.groupby("group")[FLAG_META].mean().to_dict("index")

    sample = sample_groups(readable, args.sample_per_group, args.seed)
    sample_meta = meta.loc[sample.index]
    report["trivial_classifier_metadata"] = run_trivial_classifier(
        sample_meta[METADATA_FEATURES], sample["label"], "raw_metadata", args.seed
    ).to_dict()

    pixels = pixel_feature_frame(list(sample["abs_path"]), workers=args.workers)
    pixels.index = sample.index
    report["pixel_feature_failures"] = int(pixels.isna().any(axis=1).sum())
    plot_numeric_distributions(
        pd.concat([sample[["group"]], pixels], axis=1),
        PIXEL_FEATURES,
        "group",
        plots / "raw_pixel_stats.png",
        f"{args.dataset} RAW — pixel statistics",
    )
    report["trivial_classifier_pixels"] = run_trivial_classifier(
        pixels, sample["label"], "raw_pixels", args.seed
    ).to_dict()

    profiles, maps, counts = spectra_by_group(
        readable, "abs_path", args.spectrum_per_group, args.spectrum_size, args.seed, args.workers
    )
    plot_radial_spectra(profiles, plots / "raw_spectra_1d.png", f"{args.dataset} RAW — radial power spectrum")
    plot_spectra_2d(maps, plots / "raw_spectra_2d.png", f"{args.dataset} RAW — mean log power spectrum")
    report["spectrum_counts"] = counts
    return report


def audit_processed(args: argparse.Namespace, proc: pd.DataFrame, plots: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    status = proc.assign(
        is_ok=(proc["status"] == "ok").astype(int),
        is_no_face=(proc["status"] == "no_face").astype(int),
        is_read_error=(proc["status"] == "read_error").astype(int),
        is_duplicate_dropped=(proc["status"] == "duplicate_dropped").astype(int),
    )
    status_cols = ["is_ok", "is_no_face", "is_read_error", "is_duplicate_dropped"]
    plot_rates(status, status_cols, "group", plots / "processed_status.png", f"{args.dataset} — preprocessing outcome")
    report["status_rates"] = status.groupby("group")[status_cols].mean().to_dict("index")

    ok = proc[proc["status"] == "ok"].copy()
    if ok.empty:
        raise SystemExit("No successfully processed images to audit.")
    ok["crop_clamped"] = ok["crop_clamped"].astype(str).str.lower().eq("true").astype(int)
    report["crop_clamped_rate"] = ok.groupby("group")["crop_clamped"].mean().to_dict()
    # Clamping rate differing a lot between classes would itself be a (weak) shortcut: flag it.

    sample = sample_groups(ok, args.sample_per_group, args.seed)
    meta_rows = []
    for path in sample["abs_path"]:
        try:
            meta_rows.append(read_image_meta(path).to_dict())
        except ImageReadError as exc:
            raise SystemExit(f"Processed file unreadable (prep bug, not data issue): {exc}") from exc
    meta = metadata_feature_frame(pd.DataFrame(meta_rows, index=sample.index))
    report["trivial_classifier_metadata"] = run_trivial_classifier(
        meta, sample["label"], "processed_metadata", args.seed
    ).to_dict()
    # JPEG file size still reflects image CONTENT (smooth images compress smaller), so also report
    # the classifier without size-based features to separate encoding shortcuts from content.
    no_size = meta.drop(columns=["file_size_kb", "bytes_per_pixel"])
    report["trivial_classifier_metadata_no_filesize"] = run_trivial_classifier(
        no_size, sample["label"], "processed_metadata_no_filesize", args.seed
    ).to_dict()
    plot_numeric_distributions(
        pd.concat([sample[["group"]], meta], axis=1),
        NUMERIC_META,
        "group",
        plots / "processed_metadata_numeric.png",
        f"{args.dataset} PROCESSED — metadata",
    )

    pixels = pixel_feature_frame(list(sample["abs_path"]), workers=args.workers)
    pixels.index = sample.index
    plot_numeric_distributions(
        pd.concat([sample[["group"]], pixels], axis=1),
        PIXEL_FEATURES,
        "group",
        plots / "processed_pixel_stats.png",
        f"{args.dataset} PROCESSED — pixel statistics",
    )
    report["trivial_classifier_pixels"] = run_trivial_classifier(
        pixels, sample["label"], "processed_pixels", args.seed
    ).to_dict()

    profiles, maps, counts = spectra_by_group(
        ok, "abs_path", args.spectrum_per_group, args.spectrum_size, args.seed, args.workers
    )
    plot_radial_spectra(
        profiles, plots / "processed_spectra_1d.png", f"{args.dataset} PROCESSED — radial power spectrum"
    )
    plot_spectra_2d(maps, plots / "processed_spectra_2d.png", f"{args.dataset} PROCESSED — mean log power spectrum")
    report["spectrum_counts"] = counts
    return report


def main() -> None:
    args = parse_args()
    roots = parse_roots(args.root)
    plots = args.plots_dir or plots_dir() / f"audit/{args.dataset}"
    out_json = args.out_json or reports_dir() / f"audit_{args.dataset}.json"

    raw = load_manifest(args.manifest)
    raw = raw[raw["dataset"] == args.dataset]
    if args.splits:
        raw = raw[raw["split"].isin(args.splits)]
    if raw.empty:
        raise SystemExit(f"No rows for dataset '{args.dataset}' in {args.manifest}")
    raw = add_group(raw)
    raw["abs_path"] = [resolve_path(roots, d, r) for d, r in zip(raw["dataset"], raw["rel_path"], strict=False)]

    report: dict[str, Any] = {
        "dataset": args.dataset,
        "splits": args.splits or "all",
        "seed": args.seed,
        "sample_per_group": args.sample_per_group,
        "environment": capture_environment(ML_DIR),
    }
    log.info("Auditing RAW %s (%d rows)", args.dataset, len(raw))
    report["raw"] = audit_raw(args, raw, plots)

    if args.processed_manifest:
        if args.processed_root is None:
            raise SystemExit("--processed-root is required with --processed-manifest")
        proc = pd.read_csv(args.processed_manifest, dtype={"rel_path": str, "sha256": str, "split": str})
        proc = proc[proc["dataset"] == args.dataset]
        if args.splits:
            proc = proc[proc["split"].isin(args.splits)]
        proc = add_group(proc)
        proc["abs_path"] = [args.processed_root / str(p) if pd.notna(p) else None for p in proc["processed_rel_path"]]
        log.info("Auditing PROCESSED %s (%d rows)", args.dataset, len(proc))
        report["processed"] = audit_processed(args, proc, plots)

    write_json(report, out_json)
    for stage in ("raw", "processed"):
        if stage in report:
            for key in ("trivial_classifier_metadata", "trivial_classifier_pixels"):
                log.info("%s %s AUC: %s", stage, key, report[stage][key]["auc"])
    log.info("Saved %s and plots in %s", out_json, plots)


if __name__ == "__main__":
    main()
