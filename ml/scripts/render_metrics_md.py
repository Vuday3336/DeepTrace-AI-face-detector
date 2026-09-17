"""Render docs/METRICS.md from ml/reports/*.json. Missing reports render as TBD — numbers are never typed.

python scripts/render_metrics_md.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from deeptrace_ml.utils.paths import reports_dir

ML_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ML_DIR.parent
MODELS = ["effnet_b0", "convnext_tiny", "vit_small", "clip_probe"]
TBD = "TBD"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, default=reports_dir())
    p.add_argument("--out", type=Path, default=REPO_DIR / "docs/METRICS.md")
    return p.parse_args()


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return TBD
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_ci(ci: dict[str, float] | None) -> str:
    return f"[{ci['low']:.3f}, {ci['high']:.3f}]" if ci else TBD


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def audit_section(reports: Path) -> str:
    rows = []
    for dataset in ("140k", "crossgen", "own"):
        audit = load(reports / f"audit_{dataset}.json")
        for stage in ("raw", "processed"):
            for key in (
                "trivial_classifier_metadata",
                "trivial_classifier_metadata_no_filesize",
                "trivial_classifier_pixels",
            ):
                res = ((audit or {}).get(stage) or {}).get(key)
                if audit is None and dataset != "140k":
                    continue
                rows.append(
                    [
                        dataset,
                        stage,
                        key.replace("trivial_classifier_", ""),
                        fmt(res["auc"]["logistic_regression"]["mean"]) if res else TBD,
                        fmt(res["auc"]["gradient_boosting"]["mean"]) if res else TBD,
                        ", ".join(f["feature"] for f in res["top_features"][:3]) if res else TBD,
                    ]
                )
    return "## Shortcut audit (trivial classifiers, ROC-AUC; 0.5 = no shortcut)\n\n" + table(
        ["dataset", "stage", "features", "logistic reg.", "boosting", "top features"], rows
    )


def eval_section(reports: Path) -> str:
    out = ["## Test-set metrics (calibrated, tuned threshold)"]
    for model in MODELS:
        report = load(reports / f"eval_{model}.json")
        out.append(f"\n### {model}\n")
        if report is None:
            out.append(
                table(
                    ["test set", "n", "ROC-AUC", "95% CI", "bal. acc.", "precision", "recall", "F1", "FPR", "ECE"],
                    [["id_test", *[TBD] * 9], ["crossgen_full", *[TBD] * 9], ["own", *[TBD] * 9]],
                )
            )
            continue
        cal = report["calibration"]
        out.append(
            f"Temperature {fmt(cal['temperature'])}, threshold {fmt(cal['threshold'], 4)} "
            f"(target FPR {cal['target_fpr']}), uncertainty band ±{cal['uncertainty_delta']}.\n"
        )
        rows = []
        for name, m in report["test_sets"].items():
            t = m["at_tuned_threshold"]
            rows.append(
                [
                    name,
                    str(m["n"]),
                    fmt(m["roc_auc"]),
                    fmt_ci(m["ci95"]["roc_auc"]),
                    fmt(t["balanced_accuracy"]),
                    fmt(t["precision"]),
                    fmt(t["recall_tpr"]),
                    fmt(t["f1"]),
                    fmt(t["fpr"]),
                    fmt(m["ece_15"]),
                ]
            )
        out.append(
            table(["test set", "n", "ROC-AUC", "95% CI", "bal. acc.", "precision", "recall", "F1", "FPR", "ECE"], rows)
        )
    return "\n".join(out)


def robustness_section(reports: Path) -> str:
    out = ["## Robustness (balanced accuracy on id_test)"]
    headers, rows = ["model"], []
    for model in MODELS:
        report = load(reports / f"robustness_{model}.json")
        results = ((report or {}).get("results") or {}).get("id_test")
        if not results:
            rows.append([model, TBD])
            continue
        picks = [
            r
            for r in results
            if (r["family"], r["level"])
            in {("none", 0), ("jpeg_quality", 50), ("jpeg_quality", 30), ("downscale", 0.5), ("blur_sigma", 2.0)}
        ]
        if len(headers) == 1:
            headers += [f"{r['family']}={r['level']}" for r in picks]
        rows.append([model, *[fmt(r["balanced_accuracy"]) for r in picks]])
    rows = [r + [TBD] * (len(headers) - len(r)) for r in rows]
    if len(headers) == 1:
        headers += ["clean", "jpeg=50", "jpeg=30", "downscale=0.5", "blur=2.0"]
        rows = [[r[0], *[TBD] * 5] for r in rows]
    out.append(table(headers, rows))
    return "\n".join(out)


def selection_section(reports: Path) -> str:
    sel = load(reports / "model_selection.json")
    if sel is None:
        return "## Model selection\n\nTBD — run `scripts/compare_models.py` after evaluating all models."
    rows = [
        [r["model"], fmt(r["id_auc"]), fmt(r["crossgen_mean_auc"]), fmt(r["own_auc"]), fmt(r["p95_latency_ms"], 0)]
        for r in sel["models"]
    ]
    decision = sel["decision"]
    return (
        "## Model selection (pre-registered rule)\n\n"
        + table(["model", "ID AUC", "cross-gen mean AUC", "own-set AUC", "p95 ms"], rows)
        + f"\n\n**Selected:** {decision.get('winner') or TBD} — {decision.get('reason', '')}"
    )


def main() -> None:
    args = parse_args()
    body = "\n\n".join(
        [
            "# DeepTrace — Metrics\n\n> Generated by `ml/scripts/render_metrics_md.py` from `ml/reports/*.json`. "
            "Do not edit by hand. `TBD` = not run yet.",
            audit_section(args.reports),
            eval_section(args.reports),
            robustness_section(args.reports),
            selection_section(args.reports),
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
