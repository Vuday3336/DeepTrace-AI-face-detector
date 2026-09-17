"""Apply the pre-registered selection rule (docs/ARCHITECTURE.md §5.0) and build the comparison table.

Rule: eligible = ID ROC-AUC within `id_auc_margin` of the best AND heatmap-inclusive p95 latency under
budget; winner = highest mean cross-generator (fake-swap) ROC-AUC; ties (overlapping 95% CIs of the
winner's metric) go to the faster model.

    python scripts/compare_models.py --eval reports/eval_effnet_b0.json --eval reports/eval_clip_probe.json \
        --latency reports/latency_effnet_b0.json --latency reports/latency_clip_probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from deeptrace_ml.config import read_yaml
from deeptrace_ml.utils.jsonio import write_json
from deeptrace_ml.utils.logging import get_logger
from deeptrace_ml.utils.paths import reports_dir

log = get_logger("compare_models")
ML_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", type=Path, action="append", required=True)
    p.add_argument("--latency", type=Path, action="append", default=[])
    p.add_argument("--eval-config", type=Path, default=ML_DIR / "configs/eval.yaml")
    p.add_argument("--out", type=Path, default=reports_dir() / "model_selection.json")
    return p.parse_args()


def summarize(eval_report: dict[str, Any], latency: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    sets = eval_report["test_sets"]
    cross = [(n, m) for n, m in sets.items() if n.startswith(prefix) and m["roc_auc"] is not None]
    id_set = sets.get("id_test", {})
    return {
        "model": eval_report["model"],
        "id_auc": id_set.get("roc_auc"),
        "id_auc_ci95": (id_set.get("ci95") or {}).get("roc_auc"),
        "id_balanced_accuracy": (id_set.get("at_tuned_threshold") or {}).get("balanced_accuracy"),
        "crossgen_mean_auc": float(np.mean([m["roc_auc"] for _, m in cross])) if cross else None,
        "crossgen_auc_by_set": {n: m["roc_auc"] for n, m in cross},
        "crossgen_min_auc_ci_low": min(((m["ci95"] or {}).get("roc_auc") or {}).get("low", np.nan) for _, m in cross)
        if cross
        else None,
        "crossgen_full_auc": (sets.get("crossgen_full") or {}).get("roc_auc"),
        "own_auc": (sets.get("own") or {}).get("roc_auc"),
        "p95_latency_ms": (latency or {}).get("onnx_model_and_heatmap", {}).get("p95_ms"),
    }


def select(rows: list[dict[str, Any]], margin: float, budget_ms: float) -> dict[str, Any]:
    scored = [r for r in rows if r["id_auc"] is not None]
    if not scored:
        return {"winner": None, "reason": "No model has an ID test ROC-AUC yet."}
    best_id = max(r["id_auc"] for r in scored)
    eligible, excluded = [], {}
    for r in scored:
        if r["id_auc"] < best_id - margin:
            excluded[r["model"]] = f"ID AUC {r['id_auc']:.4f} < best {best_id:.4f} - {margin}"
        elif r["p95_latency_ms"] is None:
            excluded[r["model"]] = "no latency measurement"
        elif r["p95_latency_ms"] > budget_ms:
            excluded[r["model"]] = f"p95 latency {r['p95_latency_ms']:.0f} ms > budget {budget_ms:.0f} ms"
        else:
            eligible.append(r)
    if not eligible:
        return {"winner": None, "excluded": excluded, "reason": "No model satisfied the eligibility rule."}
    with_cross = [r for r in eligible if r["crossgen_mean_auc"] is not None]
    if not with_cross:
        winner = max(eligible, key=lambda r: r["id_auc"])
        return {
            "winner": winner["model"],
            "excluded": excluded,
            "reason": "No cross-generator results; fell back to best ID AUC (re-run once crossgen data exists).",
        }
    ranked = sorted(with_cross, key=lambda r: r["crossgen_mean_auc"], reverse=True)
    top = ranked[0]
    ties = [
        r
        for r in ranked[1:]
        if top["crossgen_min_auc_ci_low"] is not None and r["crossgen_mean_auc"] >= top["crossgen_min_auc_ci_low"]
    ]
    if ties:
        fastest = min([top, *ties], key=lambda r: r["p95_latency_ms"])
        return {
            "winner": fastest["model"],
            "excluded": excluded,
            "tied_with": [r["model"] for r in ties if r is not fastest],
            "reason": "Cross-generator AUC within the leader's CI; tie broken by lower latency.",
        }
    return {
        "winner": top["model"],
        "excluded": excluded,
        "reason": "Highest mean cross-generator ROC-AUC among eligible models.",
    }


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.eval_config)["selection"]
    latencies = {}
    for path in args.latency:
        data = json.loads(path.read_text(encoding="utf-8"))
        latencies[data["model"]] = data
    rows = []
    for path in args.eval:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append(summarize(report, latencies.get(report["model"]), cfg["crossgen_sets_prefix"]))
    decision = select(rows, float(cfg["id_auc_margin"]), float(cfg["latency_budget_ms_p95"]))
    write_json({"rule": cfg, "models": rows, "decision": decision}, args.out)
    log.info("Selection: %s", decision)


if __name__ == "__main__":
    main()
