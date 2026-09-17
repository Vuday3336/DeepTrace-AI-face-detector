"""Frequency-domain comparison of real vs fake face crops (supporting evidence for §6.2 / §8.3).

For each source: mean log-power spectrum of processed crops. Difference maps (source minus FFHQ)
make generator-specific periodic patterns visible, and the high-frequency share quantifies them.

    python scripts/fft_analysis.py --processed-manifest data/manifests/140k_processed.csv.gz \
        --processed-manifest data/manifests/crossgen_processed.csv.gz --data-root /kaggle/working/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from deeptrace_ml.audit.plots import plot_radial_spectra, plot_spectra_2d
from deeptrace_ml.audit.spectrum import mean_spectrum
from deeptrace_ml.utils.jsonio import write_json
from deeptrace_ml.utils.logging import get_logger
from deeptrace_ml.utils.paths import plots_dir, reports_dir

log = get_logger("fft_analysis")
ML_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ML_DIR.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-manifest", type=Path, action="append", required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--reference-source", default="ffhq")
    p.add_argument("--n-per-source", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.concat([pd.read_csv(m, dtype={"split": str}) for m in args.processed_manifest], ignore_index=True)
    frame = frame[(frame["status"] == "ok") & (frame["split"].isin(["test", "crossgen", "own"]))]
    if frame.empty:
        raise SystemExit("No processed test/crossgen/own rows found")

    spectra, profiles, summary = {}, {}, {}
    for (source, label), g in frame.groupby(["source", "label"]):
        sample = g.sample(min(args.n_per_source, len(g)), random_state=args.seed)
        result = mean_spectrum(
            [args.data_root / p for p in sample["processed_rel_path"]], size=224, workers=args.workers
        )
        key = f"{source} ({label})"
        spectra[key], profiles[key] = result.mean_log_power_2d, result.radial_profile
        tail = result.radial_profile[len(result.radial_profile) // 2 :]
        summary[key] = {"n": result.n_used, "mean_log_power_upper_half_frequencies": float(tail.mean())}

    ref_key = next((k for k in spectra if k.startswith(f"{args.reference_source} ")), None)
    plots = plots_dir() / "fft"
    plot_spectra_2d(spectra, plots / "mean_spectra.png", "Mean log-power spectrum of face crops")
    plot_radial_spectra(profiles, plots / "radial_profiles.png", "Radial power profile (processed crops)")
    if ref_key:
        diffs = {f"{k} - {ref_key}": s - spectra[ref_key] for k, s in spectra.items() if k != ref_key}
        if diffs:
            plot_spectra_2d(diffs, plots / "difference_vs_reference.png", f"Spectrum difference vs {ref_key}")
            for k in diffs:
                name = k.split(" - ")[0]
                summary[name]["mean_abs_diff_vs_reference"] = float(np.abs(diffs[k]).mean())
    else:
        log.warning("Reference source '%s' not found; skipping difference maps", args.reference_source)
    write_json({"reference": ref_key, "sources": summary}, reports_dir() / "fft_analysis.json")
    log.info("Saved FFT plots to %s", plots)


if __name__ == "__main__":
    main()
